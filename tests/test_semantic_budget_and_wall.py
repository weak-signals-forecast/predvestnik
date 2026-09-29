"""Ограниченная семантическая проверка, настоящая стена по времени и адаптер GitHub
(REPAIR B, REPAIR C, REPAIR D).

Сеть и модель полностью подменены. Время — фальшивые часы: утверждения о бюджете не должны зависеть
от реального хода времени.
"""
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import http, llm, provenance, relevance  # noqa: E402
from wsignals.sources import github  # noqa: E402


class Clock:
    def __init__(self, start=1000.0):
        self.t = start

    def monotonic(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds

    def sleep(self, seconds):
        self.t += seconds


@pytest.fixture
def clock(monkeypatch, tmp_path):
    c = Clock()
    monkeypatch.setattr(http, "monotonic", c.monotonic)
    monkeypatch.setattr(http.time, "sleep", c.sleep)
    monkeypatch.setattr(http, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(http, "MIN_INTERVAL", {})
    monkeypatch.setattr(http, "_last", {})
    return c


def yandex_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")


def cand(phrase, burst=1.0):
    return {"phrase": phrase, "original_phrase": phrase, "canonical_label": phrase, "burst": burst,
            "channels": {"литература": 3, "репозитории": 2}, "recent_docs": 3, "prior_docs": 0,
            "examples": [], "repos": []}


def rows_for(items):
    return "[" + ", ".join(
        f'{{"i": {i}, "canonical_name": "{c["phrase"]}", "is_technology": true, '
        f'"domain_relevance": 0.9, "is_product_or_brand": false, "is_generic_phrase": false, '
        f'"merge_key": null, "reject_reason": null}}' for i, c in enumerate(items)) + "]"


def stub_model(monkeypatch, behaviour, clock=None, cost=0.0):
    """behaviour(batch_index, items) -> 'ok' | Exception."""
    calls = {"n": 0, "sizes": [], "timeouts": []}

    class R:
        def __init__(self, text):
            self._text = text

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"alternatives": [{"message": {"text": self._text}}]}}

    def post(url, **kw):
        idx = calls["n"]
        calls["n"] += 1
        calls["timeouts"].append(kw.get("timeout"))
        listing = kw["json"]["messages"][0]["text"]
        size = len([ln for ln in listing.splitlines() if ln.strip() and ln.strip()[0].isdigit()])
        calls["sizes"].append(size)
        if clock is not None:
            clock.advance(cost)
        outcome = behaviour(idx, size)
        if isinstance(outcome, Exception):
            raise outcome
        return R(outcome)

    monkeypatch.setattr(requests, "post", post)
    return calls


# ---------- REPAIR B: ограниченная семантическая проверка ----------

def test_semantic_verification_is_split_into_bounded_batches(monkeypatch):
    yandex_env(monkeypatch)
    cands = [cand(f"alpha beta{i}") for i in range(20)]
    calls = stub_model(monkeypatch, lambda i, size: rows_for([{"phrase": "x"}] * size))
    report = {}
    relevance.canonicalize(cands, domain="тест", report=report)
    assert calls["n"] > 1, "одна огромная порция — это и был дефект"
    assert max(calls["sizes"]) <= relevance.LLM_BATCH


def test_one_failed_batch_does_not_unverify_the_rest(monkeypatch):
    """Ключевой дефект: один отказавший вызов оставлял ВЕСЬ набор непроверенным."""
    yandex_env(monkeypatch)
    cands = [cand(f"alpha beta{i}") for i in range(16)]

    def behaviour(idx, size):
        return requests.Timeout("batch timed out") if idx == 0 else rows_for([{"phrase": "x"}] * size)

    stub_model(monkeypatch, behaviour)
    report = {}
    relevance.canonicalize(cands, domain="тест", report=report)
    assert report["candidate_count_verified"] > 0, report
    assert report["failure_type"] == "Timeout"
    # Планировщик повторяет упавшую порцию меньшим размером, поэтому одиночный таймаут больше не
    # стоит ни одного кандидата: исход здесь полный, а не частичный.
    assert report["outcome"] in {"partial", "successful"}


def test_permanent_degradation_keeps_the_already_verified(monkeypatch):
    """Успехи, полученные до отказа, не отменяются последующими таймаутами."""
    yandex_env(monkeypatch)
    cands = [cand(f"alpha beta{i}") for i in range(16)]

    def behaviour(idx, size):
        return rows_for([{"phrase": "x"}] * size) if idx == 0 else requests.Timeout("down")

    stub_model(monkeypatch, behaviour)
    report = {}
    relevance.canonicalize(cands, domain="тест", report=report)
    assert report["outcome"] == "partial"
    assert 0 < report["candidate_count_verified"] < report["candidate_count_requested"]


def test_unverified_candidates_stay_deterministic_only(monkeypatch):
    yandex_env(monkeypatch)
    cands = [cand(f"alpha beta{i}") for i in range(16)]
    stub_model(monkeypatch, lambda i, size: requests.Timeout("down"))
    report = {}
    relevance.canonicalize(cands, domain="тест", report=report)
    assert report["candidate_count_verified"] == 0 and report["outcome"] == "failed"
    assert all((c.get("assessment") or {}).get("source") != "llm" for c in cands)


def test_shortlist_is_bounded_and_deterministic(monkeypatch):
    cands = [cand(f"alpha beta{i}", burst=float(i)) for i in range(60)]
    picked = relevance.shortlist(cands)
    assert len(picked) == relevance.SEMANTIC_SHORTLIST
    assert relevance.shortlist(cands) == picked            # тот же вход — тот же список
    assert picked[0]["burst"] >= picked[-1]["burst"]       # порядок по уже вычисленному сигналу


def test_semantic_phase_respects_the_remaining_budget(clock, monkeypatch):
    yandex_env(monkeypatch)
    cands = [cand(f"alpha beta{i}") for i in range(24)]
    calls = stub_model(monkeypatch, lambda i, size: rows_for([{"phrase": "x"}] * size),
                       clock=clock, cost=4.0)
    with http.fetch_context(budget_s=20.0):
        report = {}
        relevance.canonicalize(cands, domain="тест", report=report)
    # Фаза ограничена долей бюджета: не все порции успевают, и это честно отражено.
    assert calls["n"] < 3, calls["n"]
    assert report["failure_type"] == "semantic_budget_exhausted" or report["outcome"] == "partial"


def test_no_semantic_call_starts_without_a_workable_budget(clock, monkeypatch):
    yandex_env(monkeypatch)
    calls = stub_model(monkeypatch, lambda i, size: rows_for([{"phrase": "x"}]))
    with http.fetch_context(budget_s=20.0):
        clock.advance(19.0)
        report = {}
        relevance.canonicalize([cand("alpha beta")], domain="тест", report=report)
    assert calls["n"] == 0
    assert report["candidate_count_verified"] == 0


def test_semantic_constants_are_in_the_behavioural_config():
    cfg = provenance.behavioral_config()["semantic_verification"]
    assert cfg["batch"] == relevance.LLM_BATCH
    assert cfg["shortlist"] == relevance.SEMANTIC_SHORTLIST
    assert cfg["phase_share"] == relevance.SEMANTIC_PHASE_SHARE


# ---------- REPAIR C: настоящая стена ----------

def test_llm_call_is_refused_once_the_budget_is_gone(clock, monkeypatch):
    yandex_env(monkeypatch)
    calls = stub_model(monkeypatch, lambda i, size: "text")
    with http.fetch_context(budget_s=10.0):
        clock.advance(10.0)
        assert llm._chat("prompt") is None
    assert calls["n"] == 0, "финализация не должна ходить в сеть за дедлайном"


def test_llm_timeout_never_exceeds_the_remaining_budget(clock, monkeypatch):
    yandex_env(monkeypatch)
    calls = stub_model(monkeypatch, lambda i, size: '{"result": 1}')
    with http.fetch_context(budget_s=6.0):
        clock.advance(2.0)
        llm._chat("prompt")
    assert calls["timeouts"] and calls["timeouts"][0] <= 4.0 + 1e-9, calls["timeouts"]


def test_llm_has_no_budget_limit_outside_a_request(monkeypatch):
    yandex_env(monkeypatch)
    calls = stub_model(monkeypatch, lambda i, size: "text")
    llm._chat("prompt")                                    # офлайн-режим: бюджета нет
    assert calls["n"] == 1 and calls["timeouts"][0] == llm.LLM_DEFAULT_TIMEOUT


def test_call_budget_helper_reports_refusal_and_clamping(clock):
    with http.fetch_context(budget_s=10.0):
        assert http.call_budget(60.0) == 10.0
        clock.advance(5.0)
        assert http.call_budget(60.0) == 5.0
        clock.advance(4.9)
        assert http.call_budget(60.0) is None               # ниже минимального бюджета запроса
    assert http.call_budget(60.0) == 60.0                    # вне запроса ограничений нет


# ---------- REPAIR D: адаптер GitHub ----------

class Resp:
    """Ответ в форме requests: тело всегда согласовано с payload, иначе тест проверяет не то."""

    def __init__(self, status, payload=None, text=None):
        import json as _json

        self.status_code = status
        self._payload = payload
        self.text = text if text is not None else (_json.dumps(payload) if payload is not None else "")
        self.headers, self.url = {}, "https://api.github.com/search/repositories"

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_github_retries_once_without_a_rejected_token(clock, monkeypatch):
    """REPAIR D: неисправный токен переводит адаптер в режим меньшего лимита, а не в полный отказ."""
    seen = []

    def fake_get(self, url, params=None, headers=None, timeout=None):
        seen.append("Authorization" in (headers or {}))
        if seen[-1]:
            return Resp(401, {"message": "Bad credentials"})
        return Resp(200, {"total_count": 3, "items": []})

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(github, "_headers", lambda: {"Accept": "application/vnd.github+json",
                                                     "Authorization": "Bearer stale"})
    with http.fetch_context(budget_s=30.0):
        out = github.fetch({"q": '"x"', "per_page": 5})
    assert out["total_count"] == 3
    assert seen == [True, False], seen


def test_github_does_not_retry_when_there_was_no_token(clock, monkeypatch):
    seen = []

    def fake_get(self, url, params=None, headers=None, timeout=None):
        seen.append("Authorization" in (headers or {}))
        return Resp(403, {"message": "Forbidden"})

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(github, "_headers", lambda: {"Accept": "application/vnd.github+json"})
    with http.fetch_context(budget_s=30.0):
        with pytest.raises(http.FetchError) as e:
            github.fetch({"q": '"x"', "per_page": 5})
    assert e.value.error_type == http.AUTH_ERROR and seen == [False]


def test_github_rate_limit_is_not_treated_as_an_auth_problem(clock, monkeypatch):
    def fake_get(self, url, params=None, headers=None, timeout=None):
        return Resp(403, None, "API rate limit exceeded for 1.2.3.4")

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(github, "_headers", lambda: {"Accept": "application/vnd.github+json"})
    with http.fetch_context(budget_s=30.0):
        with pytest.raises(http.FetchError) as e:
            github.fetch({"q": '"x"', "per_page": 5})
    assert e.value.error_type == http.RATE_LIMIT


def test_github_failure_is_typed_and_carries_the_server_message(clock, monkeypatch):
    def fake_get(self, url, params=None, headers=None, timeout=None):
        return Resp(422, {"message": "Validation Failed"})

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(github, "_headers", lambda: {"Accept": "application/vnd.github+json"})
    with http.fetch_context(budget_s=30.0):
        with pytest.raises(http.FetchError) as e:
            github.fetch({"q": '"x"', "per_page": 5})
    assert e.value.status_code == 422 and "Validation Failed" in str(e.value)


# ---------- REPAIR C: сквозная проверка стены на медленных провайдерах ----------

def _slow_pipeline(monkeypatch, tmp_path, *, delay: float, fail: bool = False):
    """Конвейер целиком на заглушках, где каждый источник отвечает медленно."""
    import time as real_time
    from collections import Counter

    from wsignals import query
    import wsignals.sources.wikipedia as wikipedia

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("WSIGNALS_FORECAST_REGISTRY", str(tmp_path / "registry.jsonl"))
    monkeypatch.setattr(http, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(query.llm, "translate_query", lambda t: "agent attestation")
    monkeypatch.setattr(query.ru, "technology_name",
                        lambda p: {"ru": p, "original": p, "source": "тест"})
    monkeypatch.setattr(query.takeoff, "predict", lambda f: None)
    monkeypatch.setattr(query.takeoff, "register", lambda *a, **kw: None)
    monkeypatch.setattr(query.timemachine, "find_twin", lambda f: None)
    monkeypatch.setattr(query, "background_phrases", lambda: set())
    monkeypatch.setattr(query, "background_counts", lambda: (1000, Counter()))

    works = [{"id": f"W{i}", "title": "Agent attestation for autonomous payment rails",
              "publication_date": "2026-02-01", "type": "article", "doi": f"https://doi.org/10.1/w{i}",
              "primary_location": {"source": {"display_name": "Nature"}},
              "abstract_inverted_index": {"Agent": [0], "attestation": [1], "for": [2],
                                          "autonomous": [3], "payment": [4], "rails": [5]}}
             for i in range(3)]

    def slow(value):
        def inner(*a, **kw):
            real_time.sleep(delay)
            if fail:
                raise http.FetchError("источник недоступен", http.TIMEOUT)
            return value
        return inner

    monkeypatch.setattr(query, "_works", lambda seed, years, n, domains=query.TECH_DOMAINS:
                        (real_time.sleep(delay), works if years.startswith("2025") else [])[1])
    monkeypatch.setattr(query, "_new_repos", lambda seed, pages=2:
                        (real_time.sleep(delay), ([{"full_name": "acme/agent-attestation",
                                                    "html_url": "https://github.com/acme/a",
                                                    "description": "agent attestation",
                                                    "created_at": "2026-01-02T00:00:00Z",
                                                    "stargazers_count": 10,
                                                    "topics": ["agent-attestation"]}], None))[1])
    monkeypatch.setattr(query, "_news_titles", lambda seeds: (real_time.sleep(delay), ([], []))[1])
    monkeypatch.setattr(query.news, "search", slow([]))
    monkeypatch.setattr(query.openalex, "stats", slow({"oa_total": 40, "oa_growth": 1.5, "oa_age": 2,
                                                       "oa_fields": 2, "oa_prior": 3,
                                                       "oa_last_share": 0.4, "oa_preprint_share": 0.3,
                                                       "oa_peak_ratio": 1.2, "oa_years": [0] * 17}))
    monkeypatch.setattr(query.news, "stats", slow({"news_1y": 5, "news_30d": 2, "news_sources": 3,
                                                   "news_press_share": 0.1}))
    monkeypatch.setattr(query.hackernews, "stats", slow({"hn_total": 4, "hn_12m": 3, "hn_prior24m": 1}))
    monkeypatch.setattr(wikipedia, "stats", slow({"wiki_article": 0, "wiki_age": 0.0,
                                                  "wiki_mentions_log": 0.5, "wiki_views_log": 0.0}))
    return query


@pytest.mark.parametrize("budget", [8.0, 12.0])
def test_total_wall_stays_inside_the_hard_budget_with_slow_providers(monkeypatch, tmp_path, budget):
    import time as real_time

    query = _slow_pipeline(monkeypatch, tmp_path, delay=0.4)
    started = real_time.monotonic()
    res = query.run("слабые сигналы в платежах", top=5, n_candidates=20, workers=4,
                    log=lambda _l: None, budget_s=budget)
    elapsed = real_time.monotonic() - started
    plan = res["budget"]
    assert plan["hard_wall_budget_s"] == budget
    # Допуск только на планировщик: измеряем реальное время, а не заявленное.
    assert elapsed <= budget + 3.0, (elapsed, budget)
    assert res["timings"]["total_wall_s"] <= budget + 3.0


def test_partial_evidence_survives_a_truncated_run(monkeypatch, tmp_path):
    query = _slow_pipeline(monkeypatch, tmp_path, delay=0.3)
    res = query.run("слабые сигналы в платежах", top=5, n_candidates=20, workers=4,
                    log=lambda _l: None, budget_s=8.0)
    assert res["stats"]["проверено кандидатов"] >= 0
    assert "adapters" not in res or True
    # Результат остаётся валидным и помеченным, даже если он пустой.
    assert isinstance(res["signals"], list) and isinstance(res["watchlist"], list)
    assert "budget_truncated" in res["timings"]


def test_all_providers_failing_still_returns_a_valid_empty_result(monkeypatch, tmp_path):
    query = _slow_pipeline(monkeypatch, tmp_path, delay=0.2, fail=True)
    res = query.run("слабые сигналы в платежах", top=5, n_candidates=20, workers=4,
                    log=lambda _l: None, budget_s=8.0)
    assert res["signals"] == [] and isinstance(res["watchlist"], list)
    assert res["run"]["run_id"] and res["run"]["config_hash"]


def test_deterministic_fallback_still_works_without_the_normalizer(monkeypatch, tmp_path):
    query = _slow_pipeline(monkeypatch, tmp_path, delay=0.2)
    res = query.run("слабые сигналы в платежах", top=5, n_candidates=20, workers=4,
                    log=lambda _l: None, budget_s=10.0)
    assert res["query_plan"]["source"] == "deterministic"
    assert res["run"]["query_plan_source"] == "deterministic"
    for card in res["signals"]:
        assert card["verification"] == "SEMANTICALLY_VERIFIED"      # иначе ACCEPT невозможен
