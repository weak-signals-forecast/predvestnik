"""Разделение намерения пользователя и предметной области (REPAIR A, REPAIR E).

Тесты проверяют СВОЙСТВА разбора, а не ожидаемые названия технологий. Кибербезопасность, финтех и ИИ
присутствуют только как регрессия по наблюдавшимся живым отказам; остальные области — контрольные
и в прошлых прогонах не встречались.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import decompose, intent, relevance  # noqa: E402
from wsignals.query import parse_query  # noqa: E402

REGRESSION = [
    ("слабые сигналы в кибербезопасности", ("кибербез",)),
    ("перспективные решения в финтехе", ("финтех",)),
    ("слабые сигналы в искусственном интеллекте", ("интеллект",)),
]
UNSEEN_CONTROLS = [
    ("перспективные технологии квантовых сенсоров", ("квантов", "сенсор")),
    ("новые способы охлаждения дата-центров", ("охлажден", "центр")),
    ("зарождающиеся тренды в робототехнике", ("робот",)),
    ("слабые сигналы в аддитивном производстве", ("аддитивн", "производств")),
    ("emerging technologies in photonics", ("photonic",)),
]
ALL_QUERIES = REGRESSION + UNSEEN_CONTROLS


# ---------- свойства разделения ----------

@pytest.mark.parametrize("query,domain_marks", ALL_QUERIES)
def test_domain_is_preserved(query, domain_marks):
    parsed = intent.split(query)
    low = parsed.domain.lower()
    assert any(mark in low for mark in domain_marks), (query, parsed.domain)


@pytest.mark.parametrize("query,_", ALL_QUERIES)
def test_domain_is_never_empty(query, _):
    parsed = intent.split(query)
    assert parsed.domain.strip(), query


@pytest.mark.parametrize("query,_", REGRESSION + UNSEEN_CONTROLS[:4])
def test_meta_intent_is_not_part_of_the_domain(query, _):
    """Слова о задаче поиска не должны оставаться предметом поиска."""
    low = intent.split(query).domain.lower()
    for meta in ("слаб", "сигнал", "перспективн", "зарождающ", "тренд", "emerging", "weak"):
        assert meta not in low, (query, low)


def test_intent_markers_are_detected_when_present():
    assert intent.split("слабые сигналы в кибербезопасности").horizon_scanning
    assert intent.split("emerging technologies in photonics").horizon_scanning


def test_plain_domain_query_has_no_intent_markers():
    parsed = intent.split("постквантовая криптография")
    assert not parsed.horizon_scanning and parsed.domain


def test_query_that_is_only_intent_falls_back_without_empty_domain():
    parsed = intent.split("слабые сигналы")
    assert parsed.domain_is_fallback and parsed.domain.strip()


# ---------- детерминированный путь соблюдает тот же контракт ----------

@pytest.mark.parametrize("query,_", ALL_QUERIES)
def test_deterministic_seeds_never_search_for_the_meta_intent(query, _, monkeypatch):
    monkeypatch.setattr("wsignals.llm.translate_query", lambda t: t)
    seeds = parse_query(query)
    joined = " ".join(seeds).lower()
    for meta in ("weak signal", "emerging trend", "слаб", "сигнал", "перспективн", "зарождающ"):
        assert meta not in joined, (query, seeds)


@pytest.mark.parametrize("query,_", ALL_QUERIES)
def test_deterministic_plan_is_never_empty(query, _, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr("wsignals.llm.translate_query", lambda t: t)
    plan = decompose.decompose(query, parse_query)
    assert plan.seeds(), query
    assert plan.intent and plan.intent["domain"]


def test_plan_records_the_intent_split(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr("wsignals.llm.translate_query", lambda t: t)
    plan = decompose.decompose("слабые сигналы в кибербезопасности", parse_query)
    assert plan.intent["horizon_scanning"] is True
    assert "кибербез" in plan.intent["domain"].lower()


# ---------- модельный путь соблюдает тот же контракт ----------

def _yandex(monkeypatch, text):
    import requests

    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")

    captured = {}

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"alternatives": [{"message": {"text": text}}]}}

    def post(url, **kw):
        captured["prompt"] = kw["json"]["messages"][0]["text"]
        return R()

    monkeypatch.setattr(requests, "post", post)
    return captured


def test_model_receives_the_domain_not_only_the_raw_query(monkeypatch):
    captured = _yandex(monkeypatch, '{"domain": "кибербезопасность", "english_queries": ["ai security"]}')
    decompose.decompose("слабые сигналы в кибербезопасности", parse_query)
    assert "Предметная область запроса: кибербезопасности" in captured["prompt"]
    assert "ОПИСАНИЕ ЗАДАЧИ" in captured["prompt"]


def test_meta_intent_is_stripped_from_model_seeds(monkeypatch):
    """Даже если модель вернула пересказ намерения, в поиск он не уходит."""
    _yandex(monkeypatch, '{"domain": "кибербезопасность", "english_queries": '
                         '["weak signals cyber security", "detecting weak signals", "ai security"]}')
    plan = decompose.decompose("слабые сигналы в кибербезопасности", parse_query)
    joined = " ".join(plan.seeds()).lower()
    assert "weak signal" not in joined, plan.seeds()
    assert plan.seeds(), "план не должен опустеть"


def test_model_returning_only_intent_restatements_falls_back(monkeypatch):
    _yandex(monkeypatch, '{"domain": "кибербезопасность", "english_queries": '
                         '["weak signals", "emerging trends"]}')
    plan = decompose.decompose("слабые сигналы в кибербезопасности", parse_query)
    assert plan.source == "deterministic"
    assert "пересказ намерения" in (plan.fallback_reason or "")
    assert plan.seeds()


# ---------- никаких подставленных технологий ----------

def test_no_expected_technology_aliases_are_hardcoded():
    import ast

    for module in ("intent.py", "decompose.py", "query.py"):
        tree = ast.parse((ROOT / "wsignals" / module).read_text(encoding="utf-8"))
        if ast.get_docstring(tree):
            tree.body = tree.body[1:]
        code = ast.unparse(tree)
        for leaked in ("rag poisoning", "agentic payments", "post-quantum cryptography",
                       "confidential computing", "stablecoin"):
            assert leaked not in code, (module, leaked)


# ---------- гигиена кандидатов (REPAIR E, исправлено P1-B) ----------
#
# Разбор ЗАПРОСА и идентичность КАНДИДАТА — разные задачи. Словарь намерения не является чёрным
# списком токенов для названий технологий: отбраковывается только фраза, КАЖДОЕ слово которой
# взято из служебной, рубричной или дискурсивной лексики.

@pytest.mark.parametrize("phrase", [
    "weak signals", "emerging technologies", "horizon scanning", "technology foresight",
    "future research directions", "related work section", "technology trend report",
    "market outlook", "adoption readiness", "technology hype", "state of the art",
])
def test_meta_concepts_are_not_technologies(phrase):
    a = relevance.assess(phrase, {"security"}, set(), channels={"литература": 3}, lit_docs=3)
    assert a.rejected and not a.is_technology, (phrase, a.reject_reason)


@pytest.mark.parametrize("phrase", [
    # слова задачи внутри НАСТОЯЩЕЙ технологии: отбраковывать нельзя (регрессия аудита P1-B)
    "weak signal detection", "weak signal propagation reporter", "early warning system",
    "early warning radar", "cybersecurity early warning",
    # документные головные слова с техническим определением
    "numerical weather forecast", "seismic hazard forecast", "crop yield forecast",
    "situational awareness", "cyber readiness", "structural health report",
    "automated vulnerability report", "predictive maintenance forecast",
    # обычные технические термины
    "weak supervision", "acoustic emission signal", "signal to noise ratio",
    "optical signal processing", "signal integrity", "early stopping", "trend filtering",
])
def test_generic_words_alone_never_reject_a_candidate(phrase):
    """Recall: ни «weak», ни «signal», ни «early warning», ни «forecast», ни «readiness»,
    ни «awareness», ни «report» сами по себе не являются основанием для отказа."""
    a = relevance.assess(phrase, {"security"}, set(), channels={"литература": 3, "репозитории": 2},
                         lit_docs=3)
    assert not a.rejected, (phrase, a.reject_reason)


@pytest.mark.parametrize("phrase", ["reports fintech", "market outlook", "adoption readiness",
                                    "technology hype"])
def test_report_and_market_vocabulary_is_not_a_technology(phrase):
    a = relevance.assess(phrase, {"fintech"}, set(), channels={"литература": 3}, lit_docs=3)
    assert a.rejected, phrase


def test_candidate_hygiene_does_not_import_a_global_token_blacklist():
    """Правило смотрит на фразу целиком: одиночное слово намерения кандидата не убивает."""
    for word in ("weak", "signal", "early", "forecast", "readiness", "awareness", "report"):
        phrase = f"{word} propagation analyzer"
        a = relevance.assess(phrase, {"security"}, set(), channels={"литература": 3}, lit_docs=3)
        assert not a.rejected, (phrase, a.reject_reason)


# ---------- буквальная предметная область со «слабым сигналом» (P1-B) ----------

LITERAL_DOMAINS = [
    ("технологии обнаружения слабых радиосигналов", ("слаб", "радиосигнал")),
    ("weak-signal detection circuits", ("weak-signal", "circuits")),
    ("methods for weak acoustic signal detection", ("weak", "acoustic", "signal")),
    ("усилители слабых сигналов в радиоастрономии", ("слаб", "сигнал", "усилител")),
    ("weak signal propagation reporter networks", ("weak", "signal", "propagation")),
]


@pytest.mark.parametrize("query,marks", LITERAL_DOMAINS)
def test_literal_weak_signal_domain_is_preserved(query, marks):
    """Там, где слабый сигнал и есть предмет, он обязан остаться в предметной области."""
    low = intent.split(query).domain.lower()
    for mark in marks:
        assert mark in low, (query, low)


@pytest.mark.parametrize("query,_", LITERAL_DOMAINS)
def test_literal_weak_signal_query_is_not_horizon_scanning(query, _):
    """Такой запрос — поиск конкретного предмета, а не горизонт-сканирование."""
    assert not intent.split(query).horizon_scanning, query


def test_instruction_frame_is_what_makes_a_marker_a_task():
    """«слабые сигналы В агротехе» — задача; «усилители слабых сигналов» — предмет."""
    task = intent.split("слабые сигналы в агротехе")
    subject = intent.split("усилители слабых сигналов в радиоастрономии")
    assert task.horizon_scanning and "слаб" not in task.domain.lower()
    assert not subject.horizon_scanning and "слаб" in subject.domain.lower()
