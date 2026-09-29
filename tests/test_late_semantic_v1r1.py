"""V1R1: четыре P1 адъюдикации мастера по адресной поздней семантической проверке.

R1 три новые ручки обязаны входить в поведенческий отпечаток конфигурации;
R2 поздний конверт урезается по РЕАЛЬНОМУ остатку общего дедлайна, даже когда budget_s передан
   явно, а результат, пришедший после дедлайна, отбрасывается;
R3 дообогащение материализуется ДО отбора целей, иначе кандидат, которому не хватало ровно одного
   независимого документа, целью не станет;
R4 поздний проход обновляет только поля подтверждения и не стирает личность внутри `assessment`.

Два P1 приняты мастером и здесь НЕ чинятся: ранняя фаза остаётся односнарядной при 10 с, а
фиксированный поздний резерв на коротких пользовательских бюджетах остаётся известным ограничением.
"""
from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import http, provenance, query, relevance  # noqa: E402
from wsignals.decision import ACCEPT, LOW_EVIDENCE, SEMANTICALLY_VERIFIED, decide  # noqa: E402
from wsignals.evidence import OK, EvidenceState, Observation  # noqa: E402


# ===========================================================================
# R1. Отпечаток конфигурации
# ===========================================================================

def test_R1_new_knobs_are_bound_into_behavioral_config():
    cfg = provenance.behavioral_config()["semantic_verification"]
    assert cfg["early_budget_s"] == relevance.EARLY_SEMANTIC_BUDGET
    assert cfg["late_reserve_s"] == relevance.LATE_SEMANTIC_RESERVE
    assert cfg["late_max_targets"] == relevance.LATE_MAX_TARGETS


@pytest.mark.parametrize("const,value", [
    ("EARLY_SEMANTIC_BUDGET", 7.0),
    ("LATE_SEMANTIC_RESERVE", 5.0),
    ("LATE_MAX_TARGETS", 9),
])
def test_R1_changing_one_knob_changes_the_hash(monkeypatch, const, value):
    base = provenance.config_hash()
    monkeypatch.setattr(relevance, const, value)
    assert provenance.config_hash() != base, f"смена {const} обязана менять отпечаток"


@pytest.mark.parametrize("env,value", [
    ("WSIGNALS_EARLY_SEMANTIC_BUDGET", "7.0"),
    ("WSIGNALS_LATE_SEMANTIC_RESERVE", "5.0"),
    ("WSIGNALS_LATE_MAX_TARGETS", "9"),
])
def test_R1_env_variable_alone_changes_the_hash(monkeypatch, env, value):
    """Не просто константа: переменная окружения обязана доезжать до отпечатка."""
    baseline = provenance.config_hash()
    monkeypatch.setenv(env, value)
    importlib.reload(relevance)
    try:
        changed = provenance.config_hash()
    finally:
        monkeypatch.delenv(env, raising=False)
        importlib.reload(relevance)
    assert changed != baseline, f"{env} не влияет на отпечаток конфигурации"
    assert provenance.config_hash() == baseline, "после отката отпечаток обязан вернуться"


def test_R1_config_hash_is_stable_and_leaks_nothing(monkeypatch):
    monkeypatch.setenv("YANDEX_API_KEY", "секретное-значение-ключа")
    assert provenance.config_hash() == provenance.config_hash()
    assert "секретное-значение-ключа" not in str(provenance.behavioral_config())


# ===========================================================================
# R2. Общий дедлайн и запоздавший результат
# ===========================================================================

VALID_ROW = {"i": 0, "canonical_name": "federated analytics", "domain_relevance": 1.0,
             "is_technology": True, "is_product_or_brand": False, "is_generic_phrase": False}


def _target():
    return {"phrase": "federated analytics", "original_phrase": "federated analytics"}


@pytest.mark.parametrize("global_left", [2.0, 4.0, 6.0])
def test_R2_global_time_left_clamps_an_explicit_budget(monkeypatch, global_left):
    """Запрошено 8 с, а глобально осталось меньше -> конверт урезается до остатка.

    При остатке 2 с вызова не происходит вовсе: он ниже SEMANTIC_MIN_CALL_BUDGET. Это тоже форма
    соблюдения предела, и подтвердить кандидата такой проход не может."""
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))
    seen = {}

    def spy(items, domain, uri, remaining):
        seen["remaining"] = remaining
        return [VALID_ROW]
    monkeypatch.setattr(relevance, "_late_call_batch", spy)
    cand = _target()
    report = {}
    with http.fetch_context(budget_s=global_left):
        relevance.verify_late([cand], report=report, budget_s=8.0)
    assert report["budget_s"] <= global_left, report
    if global_left < relevance.SEMANTIC_MIN_CALL_BUDGET:
        assert "remaining" not in seen, "ниже минимума вызов не начинается"
        assert relevance.semantically_verified(cand) is False
    else:
        assert seen["remaining"] <= global_left, "конверт попытки урезан общим дедлайном"


def test_R2_no_call_at_all_when_global_deadline_leaves_too_little(monkeypatch):
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))
    calls = []
    monkeypatch.setattr(relevance, "_late_call_batch", lambda *a, **k: calls.append(1) or [VALID_ROW])
    cand = _target()
    report = {}
    with http.fetch_context(budget_s=1.0):          # меньше SEMANTIC_MIN_CALL_BUDGET
        relevance.verify_late([cand], report=report, budget_s=8.0)
    assert calls == []
    assert report["deadline_exhausted"] is True
    assert relevance.semantically_verified(cand) is False


def test_R2_result_arriving_after_the_deadline_is_discarded(monkeypatch):
    """Валидная строка, но ответ пришёл после дедлайна: подтверждения не происходит."""
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))

    class Clock:
        def __init__(self):
            self.t = 0.0

        def monotonic(self):
            return self.t
    clock = Clock()
    import wsignals.http as http_mod
    monkeypatch.setattr(http_mod, "monotonic", clock.monotonic)

    def slow(items, domain, uri, remaining):
        clock.t += remaining + 5.0                   # ответ пришёл заведомо позже срока
        return [VALID_ROW]
    monkeypatch.setattr(relevance, "_late_call_batch", slow)
    cand = _target()
    report = {}
    relevance.verify_late([cand], report=report, budget_s=8.0)
    assert relevance.semantically_verified(cand) is False, "запоздавший ответ не подтверждает"
    assert "assessment" not in cand
    assert report["failure_type"] == "late_result_after_deadline"
    assert report["deadline_exhausted"] is True
    assert report["verified_count"] == 0


def test_R2_in_time_result_still_verifies(monkeypatch):
    """Контроль: сам по себе рубеж дедлайна не ломает нормальный путь."""
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))
    monkeypatch.setattr(relevance, "_late_call_batch", lambda *a, **k: [VALID_ROW])
    cand = _target()
    relevance.verify_late([cand], report={}, budget_s=8.0)
    assert relevance.semantically_verified(cand) is True


def test_R2_transport_hard_wall_claim_is_narrowed_in_the_docs():
    """Документация обязана называть реальные границы, а не обещать абсолютную стену."""
    import inspect
    doc = inspect.getdoc(relevance._bounded_timeout) or ""
    assert "НЕ абсолютная стена" in doc or "не абсолютная стена" in doc.lower()
    for unguaranteed in ("по капле", "getaddrinfo"):
        assert unguaranteed in doc, f"в докстроке не названо ограничение: {unguaranteed}"


# ===========================================================================
# R3. Дообогащение до отбора целей
# ===========================================================================

def _state_science():
    s = EvidenceState()
    # Значения подобраны под ФАКТИЧЕСКИЕ входы emergence()/maturity(): рост, свежесть, возраст.
    s.add(Observation(source="openalex", status=OK,
                      values={"oa_total": 40, "oa_recent": 26, "oa_prior": 2, "oa_fields": 5,
                              "oa_growth": 2.0, "oa_last_share": 0.45, "oa_age": 2,
                              "oa_preprint_share": 0.3, "oa_company_share": 0.1}))
    s.add(Observation(source="news", status=OK,
                      values={"news_count": 4, "news_hype": 0.05, "news_30d_share": 0.2}))
    return s


def _doc(title, family, url, kind="научная публикация", name="Journal"):
    from wsignals import claim_relevance as CR, trust
    from wsignals.schema import Document
    d = trust.assess(Document(title=title, url=url, source_name=name, source_type=kind,
                              published="2026-01-01", language="en")).with_provenance()
    return CR.annotate([d], {"phrase": "federated analytics",
                             "original_phrase": "federated analytics"})[0]


def test_R3_enrichment_is_materialized_before_target_selection():
    """Один научный документ: семантика ещё НЕ единственное препятствие.
    Добавляем независимый медийный документ: становится единственным — кандидат обязан стать целью.
    """
    cand = {"phrase": "federated analytics", "original_phrase": "federated analytics", "burst": 0.0}
    state = _state_science()
    science = [_doc("Federated analytics for privacy preserving telemetry", "science",
                    "https://doi.org/10.1000/fa")]
    # Отраслевое СМИ опознаётся по домену: неизвестное издание подтверждением не считается
    # (fail-closed R2-B), поэтому фикстура берёт реально распознаваемый источник.
    media = [_doc("Federated analytics rolled out across regional banks", "media",
                  "https://www.reuters.com/technology/federated-analytics",
                  kind="новости", name="Reuters")]

    before = decide(state, documents=science, candidate=cand, verification=SEMANTICALLY_VERIFIED)
    after = decide(state, documents=science + media, candidate=cand,
                   verification=SEMANTICALLY_VERIFIED)
    assert before.decision != ACCEPT, "до дообогащения доказательств не хватает"
    assert after.decision == ACCEPT, "после дообогащения семантика — единственное препятствие"
    # И без семантики он по-прежнему не принят: цель отбирается именно контрфактически.
    assert decide(state, documents=science + media, candidate=cand,
                  verification="DETERMINISTIC_ONLY").decision != ACCEPT


def test_R3_source_order_in_run_is_enrichment_then_selection_then_late():
    src = (ROOT / "wsignals" / "query.py").read_text(encoding="utf-8")
    i_fetch = src.index('r["extra_docs"] = (news.search')
    # именно ТА приклейка, что относится к дообогащению, а не первая в файле
    i_annotate = src.index('r["documents"] = list(r["documents"]) + claim_relevance.annotate(')
    i_select = src.index("verification=SEMANTICALLY_VERIFIED).accepted")
    i_late = src.index("relevance.verify_late(")
    assert i_fetch < i_annotate < i_select < i_late, (
        "порядок обязан быть: получить -> приклеить -> отобрать -> проверить")


def test_R3_enrichment_is_annotated_only_once():
    """Блок перенесён, а не продублирован: иначе документы приклеились бы дважды."""
    src = (ROOT / "wsignals" / "query.py").read_text(encoding="utf-8")
    assert src.count('r["documents"] = list(r["documents"]) + claim_relevance.annotate(') == 1


# ===========================================================================
# R4. Минимальная запись оценки
# ===========================================================================

HOSTILE = {"i": 0, "canonical_name": "homomorphic encryption", "merge_key": "hijacked-key",
           "domain_relevance": 1.0, "is_technology": True,
           "is_product_or_brand": False, "is_generic_phrase": False}


def _cand_with_early_assessment():
    return {
        "phrase": "federated analytics", "original_phrase": "federated analytics",
        "canonical_label": "federated analytics", "merge_key": "federated analytics",
        "merged_from": ["analytics"], "high_certainty": True,
        "assessment": {
            "original_phrase": "federated analytics", "canonical_name": "federated analytics",
            "merge_key": "federated analytics", "specificity": 0.83,
            "ambiguity_flags": ["brand_token"], "is_product_or_brand": False,
            "is_generic_phrase": False, "is_technology": False,
            "domain_relevance": 0.1, "reject_reason": "ранняя фаза отвергла", "source": "deterministic",
        },
    }


def test_R4_nested_identity_and_ranking_metadata_survive_a_hostile_late_row(monkeypatch):
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))
    monkeypatch.setattr(relevance, "_late_call_batch", lambda *a, **k: [HOSTILE])
    cand = _cand_with_early_assessment()
    top_before = copy.deepcopy({k: v for k, v in cand.items() if k != "assessment"})
    a_before = copy.deepcopy(cand["assessment"])

    relevance.verify_late([cand], report={}, budget_s=8.0)

    assert {k: v for k, v in cand.items() if k != "assessment"} == top_before, "верхний уровень цел"
    a = cand["assessment"]
    # Личность и метаданные ранжирования ВНУТРИ оценки сохранены.
    for keep in ("canonical_name", "merge_key", "original_phrase", "specificity",
                 "ambiguity_flags", "is_product_or_brand", "is_generic_phrase"):
        assert a[keep] == a_before[keep], f"поздний проход переписал {keep}"
    assert a["canonical_name"] != "homomorphic encryption"
    assert a["merge_key"] != "hijacked-key"
    # Обновлены ровно поля подтверждения.
    assert a["source"] == "llm" and a["is_technology"] is True
    assert a["domain_relevance"] == 1.0 and a["reject_reason"] is None
    assert relevance.semantically_verified(cand) is True


def test_R4_only_the_authorized_fields_differ(monkeypatch):
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))
    monkeypatch.setattr(relevance, "_late_call_batch", lambda *a, **k: [HOSTILE])
    cand = _cand_with_early_assessment()
    before = copy.deepcopy(cand["assessment"])
    relevance.verify_late([cand], report={}, budget_s=8.0)
    changed = {k for k in cand["assessment"] if cand["assessment"][k] != before.get(k)}
    assert changed <= set(relevance.LATE_ASSESSMENT_FIELDS), f"изменены лишние поля: {changed}"


def test_R4_negative_late_response_preserves_ambiguity_flags(monkeypatch):
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))
    monkeypatch.setattr(relevance, "_late_call_batch", lambda *a, **k: [
        {"i": 0, "canonical_name": "x", "domain_relevance": 0.05, "is_technology": False,
         "is_product_or_brand": False, "is_generic_phrase": True}])
    cand = _cand_with_early_assessment()
    relevance.verify_late([cand], report={}, budget_s=8.0)
    assert cand["assessment"]["ambiguity_flags"] == ["brand_token"], "флаги неоднозначности целы"
    assert cand["assessment"]["specificity"] == 0.83
    assert relevance.semantically_verified(cand) is False
    assert "noise_reason" not in cand, "отрицательный поздний ответ не переклассифицирует"


def test_R4_late_write_field_set_is_declared_and_minimal():
    assert relevance.LATE_ASSESSMENT_FIELDS == (
        "source", "is_technology", "domain_relevance", "reject_reason")


def test_R4_candidate_without_an_early_assessment_gets_identity_from_itself(monkeypatch):
    """Даже когда основы нет, личность берётся у кандидата, а не из ответа модели."""
    monkeypatch.setattr(relevance, "configured_model", lambda: ("gpt://x/yandexgpt-5-lite", None))
    monkeypatch.setattr(relevance, "_late_call_batch", lambda *a, **k: [HOSTILE])
    cand = _target()
    relevance.verify_late([cand], report={}, budget_s=8.0)
    assert cand["assessment"]["canonical_name"] == "federated analytics"
    assert cand["assessment"]["canonical_name"] != "homomorphic encryption"


# ===========================================================================
# Принятые мастером компромиссы: закреплены, не починены
# ===========================================================================

def test_accepted_early_phase_stays_single_shot():
    assert relevance.EARLY_SEMANTIC_BUDGET == 10.0
    assert relevance.SEMANTIC_RECOVERY_RESERVE == 8.0
    early = relevance.EARLY_SEMANTIC_BUDGET
    assert early - relevance._attempt_timeout(early) < relevance.SEMANTIC_MIN_CALL_BUDGET, (
        "ранняя фаза односнарядная — принято мастером, чинить здесь нельзя")


def test_accepted_validated_profile_is_the_default_budget_only():
    plan = query.budget_plan()
    assert (plan["hard_wall_budget_s"], plan["network_deadline_s"],
            plan["finalization_margin_s"]) == (90.0, 82.0, 8.0)
    assert relevance.LATE_SEMANTIC_RESERVE == 8.0, "адаптивного распределителя здесь не вводится"
