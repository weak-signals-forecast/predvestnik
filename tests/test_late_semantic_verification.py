"""Адресная поздняя семантическая проверка: отбор целей, fail-closed, неизменность личности, бюджет.

Почему этот проход вообще появился. На владельческих живых прогонах ранняя фаза выдала 18
семантических меток, и НИ ОДНА не попала на кандидата, готового по доказательствам: короткий список
строится до сбора свидетельств и о готовности ничего не знает. После принятого ремонта привязки
семеро кандидатов упирались ТОЛЬКО в семантическую проверку.

Что проверяется здесь:

  A — отбор целей обобщённый и контрфактический (никаких зашитых технологий);
  B — поздний проход fail-closed по всем классам отказа;
  C — поздний ответ модели не может тронуть личность и свидетельства;
  D — бюджет: ранняя ≤10 с, поздняя ≤8 с, сумма ≤18 с, 90/82/8 не тронуты, один поздний вызов
      не может превысить остаток конверта;
  F — раннее и позднее разделение ответственности.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import http, provenance, query, relevance  # noqa: E402
from wsignals.decision import SEMANTICALLY_VERIFIED  # noqa: E402


# ---------------------------------------------------------------------------
# D. Бюджет
# ---------------------------------------------------------------------------

def test_D_split_is_ten_plus_eight_within_the_frozen_eighteen():
    assert relevance.EARLY_SEMANTIC_BUDGET == 10.0
    assert relevance.LATE_SEMANTIC_RESERVE == 8.0
    assert relevance.EARLY_SEMANTIC_BUDGET + relevance.LATE_SEMANTIC_RESERVE <= relevance.SEMANTIC_PHASE_FLOOR
    assert relevance.SEMANTIC_PHASE_FLOOR == 18.0, "общая семантическая доля не менялась"


def test_D_global_budget_untouched():
    plan = query.budget_plan()
    assert plan["hard_wall_budget_s"] == 90.0
    assert plan["network_deadline_s"] == 82.0
    assert plan["finalization_margin_s"] == 8.0


@pytest.mark.parametrize("left,expected_cap", [(82.0, 10.0), (40.0, 10.0), (12.0, 4.2)])
def test_D_early_budget_never_exceeds_ten(left, expected_cap):
    with http.fetch_context(budget_s=left):
        got = relevance._semantic_budget()
    assert got <= relevance.EARLY_SEMANTIC_BUDGET + 1e-6
    assert got == pytest.approx(expected_cap, abs=0.05)


def test_D_no_budget_means_offline_not_zero():
    """Отсутствие бюджета — это офлайн-режим (None), а не нулевая фаза."""
    assert relevance._semantic_budget() is None


def test_D_late_reserve_is_zero_without_a_configured_model(monkeypatch):
    """Без провайдера резервировать время под невозможный вызов нельзя."""
    monkeypatch.setattr(relevance, "configured_model", lambda: (None, "нет ключа"))
    assert relevance.late_reserve_seconds() == 0.0
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))
    assert relevance.late_reserve_seconds() == 8.0


def test_D_one_late_attempt_cannot_outlive_the_envelope():
    """Скалярный таймаут ограничивает соединение и чтение ПОРОЗНЬ: один вызов на конверте 8 с
    законно длится до 16 с (живой замер дал 8,73 и 11,50). Общий предел это закрывает."""
    from urllib3.util import Timeout as TS

    scalar = TS(connect=8.0, read=8.0)          # то, что requests строит из timeout=8.0
    bounded = relevance._bounded_timeout(8.0)
    assert scalar.total is None, "у скалярного таймаута общего предела нет"
    assert bounded.total == 8.0

    for t in (scalar, bounded):
        t.start_connect()
        t._start_connect -= 5.0                  # соединение уже съело 5 с
    assert scalar.read_timeout == 8.0, "чтение получает полный таймаут заново"
    assert bounded.read_timeout < 3.1, "чтение ужимается на потраченное соединением"
    assert bounded.read_timeout > 0.0


def test_D_bounded_timeout_is_never_negative():
    """Истёкший конверт даёт мгновенный отказ, а не исключение конфигурации urllib3."""
    for bad in (-5.0, 0.0):
        t = relevance._bounded_timeout(bad)
        assert t.total == relevance.LATE_MIN_TIMEOUT > 0


def test_D_late_pass_checks_the_deadline_before_every_attempt(monkeypatch):
    """Истёкший конверт означает отсутствие вызова, а не «последнюю попытку»."""
    calls = []
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))
    monkeypatch.setattr(relevance, "_late_call_batch",
                        lambda *a, **k: calls.append(1) or [])
    report = {}
    relevance.verify_late([{"phrase": "x", "original_phrase": "x"}], report=report, budget_s=0.5)
    assert calls == [], "при конверте меньше минимального вызов не начинается"
    assert report["deadline_exhausted"] is True
    assert report["attempted"] is False


# ---------------------------------------------------------------------------
# helpers for the selection and fail-closed tests
# ---------------------------------------------------------------------------

def _rows_fixture():
    """Минимальные строки с настоящими объектами решения — без сети."""
    from wsignals.evidence import OK, EvidenceState, Observation

    def state(**vals):
        s = EvidenceState()
        s.add(Observation(source="openalex", status=OK, values=vals))
        return s
    return state


# ---------------------------------------------------------------------------
# B. Fail-closed поздней проверки
# ---------------------------------------------------------------------------

def _target():
    return {"phrase": "federated analytics", "original_phrase": "federated analytics"}


FAILURES = [
    ("timeout", TimeoutError("late")),
    ("connect", ConnectionError("refused")),
    ("read", TimeoutError("read timed out")),
    ("malformed", ValueError("no json")),
]


@pytest.mark.parametrize("label,exc", FAILURES)
def test_B_transport_failures_leave_the_candidate_unverified(monkeypatch, label, exc):
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))

    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(relevance, "_late_call_batch", boom)
    cand = _target()
    report = {}
    n = relevance.verify_late([cand], report=report, budget_s=8.0)
    assert n == 0
    assert relevance.semantically_verified(cand) is False
    assert "assessment" not in cand, "неудачный вызов не пишет оценку"
    assert report["verified_count"] == 0


@pytest.mark.parametrize("rows,why", [
    ([], "пустой список строк"),
    ([{"i": 99, "canonical_name": "x", "domain_relevance": 1.0, "is_technology": True,
       "is_product_or_brand": False, "is_generic_phrase": False}], "чужой индекс"),
    ([{"canonical_name": "x", "domain_relevance": 1.0, "is_technology": True,
       "is_product_or_brand": False, "is_generic_phrase": False}], "нет индекса"),
    ([{"i": 0, "canonical_name": "x", "domain_relevance": "высокая", "is_technology": True,
       "is_product_or_brand": False, "is_generic_phrase": False}], "нечисловая релевантность"),
    (["не словарь"], "строка вместо объекта"),
])
def test_B_malformed_or_foreign_rows_never_verify(monkeypatch, rows, why):
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))
    monkeypatch.setattr(relevance, "_late_call_batch", lambda *a, **k: rows)
    cand = _target()
    assert relevance.verify_late([cand], report={}, budget_s=8.0) == 0, why
    assert relevance.semantically_verified(cand) is False


@pytest.mark.parametrize("rel,is_tech,verified", [
    (1.0, True, True),
    (0.40, True, True),                       # ровно на пороге
    (0.39, True, False),                      # ниже порога
    (1.0, False, False),                      # модель не считает технологией
])
def test_B_verification_follows_the_frozen_predicate(monkeypatch, rel, is_tech, verified):
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))
    monkeypatch.setattr(relevance, "_late_call_batch", lambda *a, **k: [
        {"i": 0, "canonical_name": "federated analytics", "domain_relevance": rel,
         "is_technology": is_tech, "is_product_or_brand": False, "is_generic_phrase": False}])
    cand = _target()
    relevance.verify_late([cand], report={}, budget_s=8.0)
    assert relevance.semantically_verified(cand) is verified
    assert query.verification_status(cand.get("assessment")) == (
        SEMANTICALLY_VERIFIED if verified else "DETERMINISTIC_ONLY")


def test_B_threshold_is_the_frozen_one():
    assert relevance.MIN_DOMAIN_RELEVANCE == 0.40


# ---------------------------------------------------------------------------
# C. Неизменность личности
# ---------------------------------------------------------------------------

HOSTILE_ROW = {
    "i": 0, "canonical_name": "СОВЕРШЕННО ДРУГАЯ ТЕХНОЛОГИЯ", "merge_key": "hijacked",
    "domain_relevance": 1.0, "is_technology": True,
    "is_product_or_brand": False, "is_generic_phrase": False,
}


def test_C_late_response_cannot_rewrite_identity(monkeypatch):
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))
    monkeypatch.setattr(relevance, "_late_call_batch", lambda *a, **k: [HOSTILE_ROW])
    cand = {"phrase": "federated analytics", "original_phrase": "federated analytics",
            "canonical_label": "federated analytics", "merge_key": "federated analytics",
            "merged_from": ["analytics"], "examples": [{"title": "doc"}], "repos": [{"name": "r"}],
            "channels": {"science": 3}, "noise_reason": None}
    before = copy.deepcopy({k: v for k, v in cand.items() if k != "assessment"})
    relevance.verify_late([cand], report={}, budget_s=8.0)
    after = {k: v for k, v in cand.items() if k != "assessment"}
    assert after == before, "поздний проход не имеет права трогать личность и свидетельства"
    assert cand["canonical_label"] == "federated analytics"
    assert cand["merge_key"] == "federated analytics"
    assert cand["merged_from"] == ["analytics"]
    # Подтверждение при этом состоялось: меняется ТОЛЬКО состояние проверки.
    assert relevance.semantically_verified(cand) is True


def test_C_only_assessment_key_is_written(monkeypatch):
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))
    monkeypatch.setattr(relevance, "_late_call_batch", lambda *a, **k: [HOSTILE_ROW])
    cand = _target()
    keys_before = set(cand)
    relevance.verify_late([cand], report={}, budget_s=8.0)
    assert set(cand) - keys_before == {"assessment"}


def test_C_negative_late_result_does_not_write_noise_reason(monkeypatch):
    """Поздний отказ оставляет кандидата неподтверждённым, но не переклассифицирует его в NOISE:
    склейка и классификация уже состоялись."""
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))
    monkeypatch.setattr(relevance, "_late_call_batch", lambda *a, **k: [
        {"i": 0, "canonical_name": "x", "domain_relevance": 0.1, "is_technology": False,
         "is_product_or_brand": False, "is_generic_phrase": True}])
    cand = _target()
    relevance.verify_late([cand], report={}, budget_s=8.0)
    assert relevance.semantically_verified(cand) is False
    assert "noise_reason" not in cand


# ---------------------------------------------------------------------------
# A/F. Отбор целей и разделение ответственности
# ---------------------------------------------------------------------------

def test_A_target_cap_is_four_and_deterministic(monkeypatch):
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))
    seen = {}

    def spy(items, *a, **k):
        seen["n"] = len(items)
        return []
    monkeypatch.setattr(relevance, "_late_call_batch", spy)
    targets = [{"phrase": f"t{i}", "original_phrase": f"t{i}"} for i in range(9)]
    relevance.verify_late(targets, report={}, budget_s=8.0)
    assert seen["n"] == relevance.LATE_MAX_TARGETS == 4
    assert all("assessment" not in c for c in targets[4:]), "непроверенные остаются fail-closed"


def test_A_no_owner_technology_is_hardcoded_in_production():
    """Семь измеренных фраз были свидетельством, а не конфигурацией."""
    import ast
    banned = ["ai-driven penetration", "threat intelligence enhancing", "autonomous cybersecurity",
              "cybersecurity monitoring", "kyc aml", "fintech fraud", "mcp server"]
    for mod in ("relevance.py", "query.py", "provenance.py"):
        src = (ROOT / "wsignals" / mod).read_text(encoding="utf-8")
        tree = ast.parse(src)
        # докстроки не считаются: там это описание измерения, а не поведение
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in banned:
                    raise AssertionError(f"{mod}: зашита предметная фраза {node.value!r}")
        # Вторая, независимая проверка: фраза не должна встречаться и в ИСПОЛНЯЕМОМ тексте —
        # ни в идентификаторе, ни в сравнении, ни в f-строке. Докстроки и комментарии исключены:
        # там это описание измерения, а не поведение.
        stripped = ast.unparse(ast.parse(src))
        for node in ast.walk(ast.parse(stripped)):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                node.value.value = ""          # докстрока: обнуляем, но узел сохраняем
        executable = ast.unparse(ast.parse(stripped)).lower()
        for b in banned:
            assert b not in executable, f"{mod}: предметная фраза {b!r} попала в исполняемый текст"


def test_F_late_helper_reuses_the_frozen_semantic_contract():
    """Ни второй модели, ни своего промпта, ни своей валидации."""
    import inspect
    src = inspect.getsource(relevance._late_call_batch)
    assert "_call_batch(" in src, "поздний вызов обязан идти через замороженный _call_batch"
    assert "CANON_PROMPT" not in src, "свой промпт недопустим"
    late = inspect.getsource(relevance.verify_late)
    assert "_validate_row" in late, "валидация строк должна быть замороженной"
    assert "semantically_verified" in late, "предикат подтверждения должен быть замороженным"
    for forbidden in ("canonical_label", "merge_key", "merged_from", "examples", "repos"):
        assert f'["{forbidden}"]' not in late, f"поздний проход пишет {forbidden}"


def test_F_late_pass_runs_after_collapse_and_cannot_feed_it():
    """Порядок в run(): склейка и сбор свидетельств раньше поздней проверки."""
    src = (ROOT / "wsignals" / "query.py").read_text(encoding="utf-8")
    i_collapse = src.index("relevance.collapse(")
    i_cascade = src.index("rows = cascade_collect(")
    i_late = src.index("relevance.verify_late(")
    assert i_collapse < i_cascade < i_late, "поздняя проверка обязана идти последней"


def test_telemetry_contract_is_complete(monkeypatch):
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))
    monkeypatch.setattr(relevance, "_late_call_batch", lambda *a, **k: [
        {"i": 0, "canonical_name": "x", "domain_relevance": 1.0, "is_technology": True,
         "is_product_or_brand": False, "is_generic_phrase": False}])
    report = {}
    relevance.verify_late([_target()], report=report, budget_s=8.0)
    for key in ("configured", "attempted", "target_count", "targets", "attempt_count",
                "batch_sizes", "valid_rows", "verified_count", "negative_or_uncertain_count",
                "failure_type", "budget_s", "wall_s", "deadline_exhausted"):
        assert key in report, f"в телеметрии нет {key}"
    assert report["verified_count"] == 1 and report["valid_rows"] == 1


def test_provenance_declares_the_split():
    st = provenance.semantic_normalizer_status()
    assert st["early_budget_s"] == 10.0
    assert st["late_reserve_configured_s"] == 8.0
    assert st["late_max_targets"] == 4
    assert st["pinned_model"] == "yandexgpt-5-lite"
    assert "AQVN" not in str(st) and "Api-Key" not in str(st)
