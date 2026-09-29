"""Состояние доказательства: ошибка источника, валидный ноль и их различимость (PHASE 1, PHASE 8).

Главный инвариант ремонта: отсутствие данных не может повышать уверенность в слабом сигнале.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import http  # noqa: E402
from wsignals.decision import ACCEPT, LOW_EVIDENCE, SEMANTICALLY_VERIFIED  # noqa: E402
from wsignals.decision import decide as _decide  # noqa: E402


def decide(state, **kw):
    kw.setdefault("verification", SEMANTICALLY_VERIFIED)
    return _decide(state, **kw)
from wsignals.evidence import ERROR, NO_RESULTS, OK, EvidenceState, Observation, observe  # noqa: E402


# ---------- наблюдение: успех, валидный ноль, отказ ----------

def test_successful_call_with_hits_is_ok():
    obs = observe("openalex", lambda q: {"oa_total": 42, "oa_growth": 1.0}, "x")
    assert obs.status == OK and obs.values["oa_total"] == 42


def test_successful_call_without_hits_is_no_results_not_error():
    obs = observe("openalex", lambda q: {"oa_total": 0, "oa_growth": 0.0}, "x")
    assert obs.status == NO_RESULTS and obs.succeeded and obs.error_type is None
    assert obs.values == {"oa_total": 0, "oa_growth": 0.0}     # валидный ноль остаётся данными


def test_adapter_failure_is_error_and_carries_no_values():
    def boom(q):
        raise http.FetchError("HTTP 429", http.RATE_LIMIT, 429)

    obs = observe("github", boom, "x")
    assert obs.status == ERROR and obs.error_type == "rate_limit"
    assert obs.values == {} and not obs.succeeded


@pytest.mark.parametrize("exc,expected", [
    (http.FetchError("t", http.TIMEOUT), "timeout"),
    (http.FetchError("s", http.SERVER_ERROR, 503), "server_error"),
    (http.DeadlineExceeded("d"), "deadline"),
    (ValueError("broken json"), "parse"),
])
def test_error_types_are_preserved(exc, expected):
    def boom(q):
        raise exc

    assert observe("news", boom, "x").error_type == expected


# ---------- различимость нуля и отказа ----------

def _state(**statuses) -> EvidenceState:
    state = EvidenceState()
    for name, (status, values) in statuses.items():
        if status == ERROR:
            state.add(Observation(name, ERROR, error_type="timeout", error="timeout"))
        else:
            state.add(Observation(name, status, values))
    return state


def test_empty_valid_search_and_provider_failure_are_distinguishable():
    zero = _state(openalex=(NO_RESULTS, {"oa_total": 0}))
    failed = _state(openalex=(ERROR, None))
    assert zero.to_dict()["no_result_adapters"] == ["openalex"]
    assert zero.to_dict()["failed_adapters"] == []
    assert failed.to_dict()["failed_adapters"] == ["openalex"]
    assert failed.to_dict()["no_result_adapters"] == []
    assert zero.ok("openalex") and not failed.ok("openalex")
    assert zero.completeness() == 1.0 and failed.completeness() == 0.0


def test_error_values_never_reach_the_feature_dictionary():
    state = _state(openalex=(OK, {"oa_total": 5}), github=(ERROR, None))
    assert state.values() == {"oa_total": 5}
    assert "gh_total" not in state.values()


# ---------- ключевой инвариант: отказ не повышает оценку ----------

# ---------- конкретные документы (REPAIR 1) ----------
# ACCEPT требует хотя бы одного подтверждающего документа уровня A или B. Эти документы собираются
# из уже полученного корпуса запроса, поэтому в тестах они задаются явно.

def _doc(title, url, source_type, source_name="s", language="en"):
    from wsignals.schema import Document
    from wsignals.trust import assess as assess_trust
    _d = assess_trust(Document(title=title, url=url, source_name=source_name, source_type=source_type,
                                 published="2026-03-01", language=language)).with_provenance()
    # P1-C. Признак претензионной релевантности читается строго: подтверждает только явное
    # True. Эта фикстура ПО ПОСТРОЕНИЮ представляет документ про рассматриваемого кандидата,
    # поэтому помечается явно. Неразмеченные документы проверяются отдельно тестом,
    # доказывающим, что они не открывают ACCEPT.
    _d.claim_relevant = True
    return _d


PAPER = _doc("Agent identity attestation", "https://doi.org/10.1/x", "научная публикация", "Nature")
REPO = _doc("acme/agent-identity", "https://github.com/acme/agent-identity", "репозиторий", "GitHub")
DOCS = [PAPER, REPO]


FULL = {
    "openalex": (OK, {"oa_total": 40, "oa_growth": 1.8, "oa_age": 2, "oa_last_share": 0.45,
                      "oa_preprint_share": 0.3, "oa_peak_ratio": 1.2}),
    "github": (OK, {"gh_total": 30, "gh_12m": 25, "gh_new_share": 0.8}),
    "news": (OK, {"news_1y": 8, "news_30d": 3, "news_30d_share": 0.37, "news_press_share": 0.1}),
    "hn": (OK, {"hn_total": 6, "hn_12m": 5, "hn_growth": 1.1}),
    "wiki": (NO_RESULTS, {"wiki_article": 0, "wiki_age": 0}),
}


def test_provider_error_does_not_increase_weak_signal_score():
    healthy = decide(_state(**FULL), burst=3.0, documents=DOCS)
    for broken_source in ("openalex", "github", "news", "hn", "wiki"):
        degraded = dict(FULL)
        degraded[broken_source] = (ERROR, None)
        d = decide(_state(**degraded), burst=3.0, documents=DOCS)
        assert d.emergence_score <= healthy.emergence_score, broken_source
        assert d.evidence_confidence <= healthy.evidence_confidence, broken_source
        assert d.rank_score() <= healthy.rank_score(), broken_source


def test_missing_wikipedia_does_not_count_as_absence_of_maturity():
    known_young = decide(_state(**FULL), burst=3.0, documents=DOCS)
    broken = dict(FULL)
    broken["wiki"] = (ERROR, None)
    unknown = decide(_state(**broken), burst=3.0, documents=DOCS)
    # Отказ Википедии нельзя читать как «статьи нет»: риск зрелости становится «неизвестно», а не ноль.
    assert unknown.maturity_risk > known_young.maturity_risk
    assert any("Википедия не ответила" in r for r in unknown.decision_reasons + [""]) or unknown.decision != ACCEPT


def test_all_providers_failing_cannot_produce_accepted_weak_signal():
    state = _state(**{k: (ERROR, None) for k in FULL})
    d = decide(state, burst=99.0, legacy_probability=0.99, documents=DOCS)
    assert d.decision == LOW_EVIDENCE and not d.accepted
    assert d.emergence_score == 0.0
    assert d.evidence_confidence == 0.0
    assert d.legacy_model_probability == 0.99 and d.legacy_model_reliable is False


def test_healthy_evidence_is_accepted_so_the_invariants_are_not_vacuous():
    assert decide(_state(**FULL), burst=3.0, documents=DOCS).decision == ACCEPT
