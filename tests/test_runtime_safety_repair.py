"""Финальный ремонт рантайма: GitHub на живом пути, хайп fail-closed, растущие порции, склейка.

Сеть и модель подменены целиком, часы фальшивые. Ни одного обращения к живому провайдеру.
"""
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import http, query, relevance, trust  # noqa: E402
from wsignals.decision import (ACCEPT, HYPE, LOW_EVIDENCE, SEMANTICALLY_VERIFIED,  # noqa: E402
                               decide, hype_evidence_available)
from wsignals.evidence import ERROR, NO_RESULTS, OK, EvidenceState, Observation  # noqa: E402
from wsignals.schema import Document  # noqa: E402
from wsignals.sources import github  # noqa: E402


class Clock:
    def __init__(self, start=1000.0):
        self.t = start

    def monotonic(self):
        return self.t

    def advance(self, s):
        self.t += s

    def sleep(self, s):
        self.t += s


@pytest.fixture
def clock(monkeypatch, tmp_path):
    c = Clock()
    monkeypatch.setattr(http, "monotonic", c.monotonic)
    monkeypatch.setattr(http.time, "sleep", c.sleep)
    monkeypatch.setattr(http, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(http, "MIN_INTERVAL", {})
    monkeypatch.setattr(http, "_last", {})
    return c


# ============================ P1-A: GitHub на живом пути ============================

class Resp:
    def __init__(self, code, body="{}", headers=None):
        self.status_code, self._b, self.text = code, body, body
        self.headers = headers or {}
        self.url = github.API

    def json(self):
        import json
        return json.loads(self._b)


OK_BODY = ('{"total_count":2,"items":[{"full_name":"a/b","description":"d",'
           '"html_url":"https://github.com/a/b","stargazers_count":9,"created_at":"2026-01-01T00:00:00Z"}]}')
BAD_CREDS = '{"message":"Bad credentials"}'
RATE = '{"message":"API rate limit exceeded for user."}'


def gh_stub(monkeypatch, responses):
    """Подменяет транспорт и записывает, был ли заголовок авторизации у каждой попытки."""
    seen = []
    it = iter(responses)

    def fake_get(self, url, params=None, headers=None, timeout=None):
        seen.append("Authorization" in (headers or {}))
        outcome = next(it)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    # Подменяется КЛАСС, а не экземпляр: monkeypatch на экземпляре оставил бы после себя
    # атрибут, перекрывающий классовый метод, и следующий тест ушёл бы в настоящую сеть.
    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(github, "_headers",
                        lambda: {"Accept": "application/vnd.github+json", "Authorization": "Bearer FAKE"})
    return seen


def test_live_corpus_path_uses_the_canonical_adapter(clock, monkeypatch):
    """Корневой дефект: корпусный путь ходил в GitHub напрямую и не знал про повтор без токена."""
    import inspect
    source = inspect.getsource(query._new_repos)
    assert "github.fetch(" in source
    assert "get_json(github.API" not in source, "второй реализации запроса быть не должно"


def test_invalid_token_falls_back_to_unauthenticated_and_corpus_stays_ok(clock, monkeypatch):
    """401 по токену + 200 без токена => корпус GitHub OK, ровно две попытки."""
    seen = gh_stub(monkeypatch, [Resp(401, BAD_CREDS), Resp(200, OK_BODY)])
    with http.fetch_context(budget_s=60):
        items, error = query._new_repos("post quantum", pages=1)
    assert error is None, error
    assert len(items) == 1
    assert seen == [True, False], "ровно один авторизованный запрос и ровно один повтор без токена"


def test_authenticated_success_makes_no_retry(clock, monkeypatch):
    seen = gh_stub(monkeypatch, [Resp(200, OK_BODY)])
    with http.fetch_context(budget_s=60):
        items, error = query._new_repos("post quantum", pages=1)
    assert error is None and seen == [True]


def test_authenticated_rate_limit_is_not_an_auth_fallback_loop(clock, monkeypatch):
    seen = gh_stub(monkeypatch, [Resp(403, RATE, {"X-RateLimit-Remaining": "0"})] * 8)
    with http.fetch_context(budget_s=60):
        _, error = query._new_repos("post quantum", pages=1)
    assert error == http.RATE_LIMIT
    assert all(seen), "лимит запросов не повод сбрасывать авторизацию"


def test_unauthenticated_rate_limit_is_reported_as_rate_limit(clock, monkeypatch):
    seen = gh_stub(monkeypatch, [Resp(401, BAD_CREDS)] + [Resp(403, RATE)] * 8)
    with http.fetch_context(budget_s=60):
        _, error = query._new_repos("post quantum", pages=1)
    assert error == http.RATE_LIMIT
    assert seen[0] is True and seen[1] is False


def test_network_failure_is_typed_as_network(clock, monkeypatch):
    gh_stub(monkeypatch, [requests.ConnectionError("no route")] * 8)
    with http.fetch_context(budget_s=60):
        _, error = query._new_repos("post quantum", pages=1)
    assert error == http.NETWORK


def test_timeout_is_typed_as_timeout(clock, monkeypatch):
    gh_stub(monkeypatch, [requests.Timeout("slow")] * 8)
    with http.fetch_context(budget_s=60):
        _, error = query._new_repos("post quantum", pages=1)
    assert error == http.TIMEOUT


def test_other_http_error_contract_is_preserved(clock, monkeypatch):
    gh_stub(monkeypatch, [Resp(404, '{"message":"Not Found"}')])
    with http.fetch_context(budget_s=60):
        _, error = query._new_repos("post quantum", pages=1)
    assert error == http.HTTP_ERROR


def test_no_credential_value_is_logged_or_returned(clock, monkeypatch, caplog):
    gh_stub(monkeypatch, [Resp(401, BAD_CREDS), Resp(200, OK_BODY)])
    with caplog.at_level("DEBUG"):
        with http.fetch_context(budget_s=60):
            query._new_repos("post quantum", pages=1)
    assert "FAKE" not in caplog.text


# ============================ P1-B: хайп fail-closed ============================

def _doc(title, url, source_type, name):
    d = trust.assess(Document(title=title, url=url, source_name=name, source_type=source_type,
                              published="2026-03-01", language="en")).with_provenance(
        retrieved_at="2026-09-21T00:00:00+00:00")
    # P1-C. Признак претензионной релевантности читается строго: подтверждает только явное True.
    # Эта фикстура по построению представляет документ про рассматриваемого кандидата.
    d.claim_relevant = True
    return d


TWO_FAMILIES = [
    _doc("Lattice calibration for cryogenic interconnects", "https://doi.org/10.1038/a",
         "научная публикация", "Nature"),
    _doc("someorg/lattice-cal: toolkit", "https://github.com/someorg/lattice-cal",
         "репозиторий", "GitHub"),
]
PRESS_HEAVY = {"news_1y": 40, "news_30d": 12, "news_30d_share": 0.5, "news_press_share": 0.9}


def _state(news_status, news_values):
    s = EvidenceState()
    s.add(Observation("openalex", OK, {"oa_total": 40, "oa_growth": 1.8, "oa_age": 2,
                                       "oa_last_share": 0.45, "oa_preprint_share": 0.3,
                                       "oa_peak_ratio": 1.2, "oa_fields": 2, "oa_prior": 3}))
    s.add(Observation("github", OK, {"gh_total": 30, "gh_12m": 25, "gh_new_share": 0.8}))
    s.add(Observation("news", news_status, news_values))
    s.add(Observation("hn", OK, {"hn_total": 6, "hn_12m": 5, "hn_growth": 1.1}))
    s.add(Observation("wiki", NO_RESULTS, {"wiki_article": 0, "wiki_age": 0}))
    return s


def _decide(news_status, news_values, documents=TWO_FAMILIES):
    return decide(_state(news_status, news_values), burst=0.8, documents=documents,
                  verification=SEMANTICALLY_VERIFIED)


def test_1_measured_press_concentration_is_hype():
    d = _decide(OK, PRESS_HEAVY)
    assert d.decision == HYPE and d.decision_reason_codes == ["hype_veto"]
    assert d.hype_measured is True


def test_2_provider_error_blocks_accept_without_claiming_hype():
    d = _decide(ERROR, {})
    assert d.decision == LOW_EVIDENCE
    assert d.decision_reason_codes == ["hype_evidence_unavailable"]
    assert d.decision != HYPE, "выдумывать хайп нельзя"
    assert d.hype_measured is False
    assert d.hype_risk == 0.0, "величина риска не выдумывается — она просто не измерена"


def test_3_no_results_stays_a_measured_observation():
    d = _decide(NO_RESULTS, {"news_1y": 0, "news_30d": 0, "news_30d_share": 0.0,
                             "news_press_share": 0.0})
    assert d.hype_measured is True and d.decision == ACCEPT


PROMO_DOMINATED = [
    _doc("Acme launches lattice tool", "https://www.prnewswire.com/a", "пресс-релиз", "PR Newswire"),
    _doc("Acme unveils lattice tool", "https://www.businesswire.com/b", "пресс-релиз", "Business Wire"),
    _doc("Acme lattice tool ships", "https://feed.example/c", "пресс-релиз", "Feed"),
    _doc("Lattice calibration for cryogenic interconnects", "https://doi.org/10.1038/a",
         "научная публикация", "Nature"),
]


def test_4_real_promotional_evidence_lets_normal_logic_continue():
    """Новости отказали, но сами документы несут промо-сигнал: расчёт продолжается по ним."""
    d = _decide(ERROR, {}, documents=PROMO_DOMINATED)
    assert d.hype_measured is True
    assert d.hype_risk > 0.0, "признак сработал по документам, а не выдуман"
    assert d.decision_reason_codes != ["hype_evidence_unavailable"]


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_science_and_code_document_count_never_measures_the_media_dimension(n):
    """Сколько бы научных и кодовых документов ни было, медийное измерение ими не закрывается."""
    docs = [_doc(f"Distinct study {i} on couplers", f"https://doi.org/10.11{i}/x",
                 "научная публикация", f"J{i}") for i in range(n)]
    assert _decide(ERROR, {}, documents=docs).hype_measured is False


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("news_values", [PRESS_HEAVY,
                                         {"news_1y": 6, "news_30d": 2, "news_30d_share": 0.3,
                                          "news_press_share": 0.05}])
def test_losing_the_hype_source_never_eases_admission(n, news_values):
    """Монотонность: замена успешного наблюдения на ERROR не делает допуск легче ни при каком N."""
    docs = ([_doc("Lattice calibration for cryogenic interconnects", "https://doi.org/10.1038/a",
                  "научная публикация", "Nature"),
             _doc("someorg/lattice-cal: toolkit", "https://github.com/someorg/lattice-cal",
                  "репозиторий", "GitHub")]
            + [_doc(f"Distinct study {i} on couplers", f"https://doi.org/10.11{i}/x",
                    "научная публикация", f"J{i}") for i in range(max(0, n - 2))])[:max(n, 0)] or []
    complete = _decide(OK, news_values, documents=docs)
    failed = _decide(ERROR, {}, documents=docs)
    assert not (failed.decision == ACCEPT and complete.decision != ACCEPT), (n, complete.decision,
                                                                            failed.decision)


@pytest.mark.parametrize("news_values", [PRESS_HEAVY,
                                         {"news_1y": 40, "news_30d": 12, "news_30d_share": 0.5,
                                          "news_press_share": 0.1}])
def test_5_provider_error_never_makes_a_candidate_easier_to_accept(news_values):
    """Отказ источника не может превратить непринятого кандидата в принятого."""
    measured = _decide(OK, news_values)
    failed = _decide(ERROR, {})
    assert not (failed.decision == ACCEPT and measured.decision != ACCEPT)


def test_hype_sufficiency_is_separable_from_hype_risk():
    assert hype_evidence_available(_state(OK, PRESS_HEAVY)) is True
    assert hype_evidence_available(_state(NO_RESULTS, {})) is True
    assert hype_evidence_available(_state(ERROR, {})) is False


def test_unmeasured_hype_is_serialized_honestly():
    d = _decide(ERROR, {})
    assert d.to_dict()["hype_measured"] is False
    assert d.to_dict()["hype_risk"] == 0.0


# ============================ P1-C: матрица задержек ============================

def _cand(phrase, burst=1.0):
    return {"phrase": phrase, "original_phrase": phrase, "canonical_label": phrase, "burst": burst,
            "channels": {"литература": 3, "репозитории": 2}, "recent_docs": 3, "prior_docs": 0,
            "examples": [], "repos": []}


def _rows(n):
    import json
    return json.dumps([{"i": i, "canonical_name": f"tech {i}", "is_technology": True,
                        "domain_relevance": 0.9, "is_product_or_brand": False,
                        "is_generic_phrase": False, "merge_key": None, "reject_reason": None}
                       for i in range(n)])


def latency_model(clock, per_item, overhead):
    """Провайдер с фиксированной частью и частью, растущей с размером порции."""
    calls = {"sizes": [], "timeouts": [], "outcomes": []}

    class R:
        def __init__(self, text):
            self._t = text

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"alternatives": [{"message": {"text": self._t}}]}}

    def post(url, **kw):
        listing = kw["json"]["messages"][0]["text"]
        size = len([l for l in listing.rsplit("Список:", 1)[-1].splitlines() if l.strip()])
        need = overhead + per_item * size
        allowed = kw.get("timeout")
        calls["sizes"].append(size)
        calls["timeouts"].append(round(allowed, 2) if allowed else allowed)
        if allowed is not None and need > allowed:
            clock.advance(allowed)
            calls["outcomes"].append("timeout")
            raise requests.Timeout(f"needed {need:.1f}s, allowed {allowed:.1f}s")
        clock.advance(need)
        calls["outcomes"].append("ok")
        return R(_rows(size))

    return post, calls


def run_matrix(monkeypatch, clock, per_item, overhead, budget_s, n=24, error=None):
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")
    cands = [_cand(f"alpha beta{i} gamma", burst=float(n - i)) for i in range(n)]
    if error is not None:
        def post(url, **kw):
            raise error
        calls = {"sizes": [], "timeouts": [], "outcomes": []}
    else:
        post, calls = latency_model(clock, per_item, overhead)
    monkeypatch.setattr(requests, "post", post)
    report = {}
    with http.fetch_context(budget_s=budget_s):
        relevance.canonicalize(cands, domain="тест", report=report)
    return report, calls, cands


def _summary(report, calls, cands):
    accepted = sum(1 for c in cands if c.get("canonical_source") and not c.get("noise_reason"))
    rejected = sum(1 for c in cands if c.get("canonical_source") and c.get("noise_reason"))
    return {
        "attempted": report["batches_attempted"], "succeeded": report["batches_succeeded"],
        "sizes": report["batch_sizes"], "assessed": report["candidate_count_verified"],
        "accepted": accepted, "rejected": rejected,
        "unverified": report["candidate_count_unverified"], "outcome": report["outcome"],
    }


def test_matrix_a_fast_success(clock, monkeypatch):
    report, calls, cands = run_matrix(monkeypatch, clock, per_item=0.05, overhead=0.2, budget_s=400)
    s = _summary(report, calls, cands)
    assert s["assessed"] == 24 and s["outcome"] == "successful"
    assert s["sizes"][0] == relevance.LLM_BATCH, s["sizes"]   # первая попытка полноразмерная
    assert max(s["sizes"]) <= relevance.LLM_BATCH
    assert s["unverified"] == 0


def test_matrix_b_medium_success(clock, monkeypatch):
    """Каждая начатая порция успешна; сколько их поместилось, задаёт РАННИЙ бюджет.

    Раньше сюда помещались все 24 кандидата: фаза была 18 с. После авторизованного разделения
    семантической доли (10 с ранняя + 8 с адресная поздняя) в раннюю фазу помещается меньше порций.
    Проверяемое свойство прежнее — частичный успех без потерь: ни одна успешная порция не
    отменяется, — а число оценённых следует из бюджета, а не задаётся руками."""
    report, calls, cands = run_matrix(monkeypatch, clock, per_item=0.4, overhead=2.0, budget_s=400)
    s = _summary(report, calls, cands)
    assert s["succeeded"] == s["attempted"], "начатая порция обязана завершиться успехом"
    assert 0 < s["assessed"] <= 24
    assert s["assessed"] == s["succeeded"] * relevance.LLM_BATCH


def test_matrix_c_first_timeout_then_smaller_batch_succeeds(clock, monkeypatch):
    """Решающее свойство: таймаут первой попытки больше не означает verified=0.

    Задержка подобрана к ТЕКУЩЕМУ стартовому размеру порции: полная порция не укладывается в
    предельный таймаут вызова, половинная укладывается. Сами числа задержки к предметной области
    отношения не имеют — проверяется свойство восстановления, а не конкретный размер."""
    ceiling, half = relevance.LLM_BATCH, max(relevance.SEMANTIC_MIN_BATCH, relevance.LLM_BATCH // 2)
    # Механика деления порции цела, но внутри 10 с ранней фазы она недостижима: после первой
    # попытки остатка не остаётся. Здесь проверяется САМА механика, поэтому фаза берётся той,
    # в которой она работает. Продакшен-бюджет ранней фазы проверяется отдельным тестом.
    monkeypatch.setattr(relevance, "EARLY_SEMANTIC_BUDGET", 60.0)
    report, calls, cands = run_matrix(monkeypatch, clock, per_item=6.0, overhead=1.0, budget_s=400)
    s = _summary(report, calls, cands)
    assert "timeout" in calls["outcomes"], "сценарий обязан содержать таймаут"
    assert s["assessed"] > 0, s
    assert ceiling in calls["sizes"] and half in calls["sizes"], calls["sizes"]
    assert calls["outcomes"][calls["sizes"].index(ceiling)] == "timeout"


def test_matrix_d_repeated_timeout_is_fail_closed(clock, monkeypatch):
    report, calls, cands = run_matrix(monkeypatch, clock, per_item=50.0, overhead=50.0, budget_s=60)
    s = _summary(report, calls, cands)
    assert s["assessed"] == 0 and s["accepted"] == 0
    assert s["outcome"] == "failed"
    assert s["unverified"] == 24
    assert s["attempted"] <= relevance.SEMANTIC_MAX_ATTEMPTS


def test_matrix_e_partial_success_then_timeout_keeps_earlier_work(clock, monkeypatch):
    calls_seen = {"n": 0}
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")
    cands = [_cand(f"alpha beta{i} gamma", burst=float(24 - i)) for i in range(24)]

    class R:
        def __init__(self, t):
            self._t = t

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"alternatives": [{"message": {"text": self._t}}]}}

    def post(url, **kw):
        listing = kw["json"]["messages"][0]["text"]
        size = len([l for l in listing.rsplit("Список:", 1)[-1].splitlines() if l.strip()])
        calls_seen["n"] += 1
        clock.advance(1.0)
        if calls_seen["n"] <= 2:
            return R(_rows(size))
        raise requests.Timeout("provider degraded")

    monkeypatch.setattr(requests, "post", post)
    report = {}
    with http.fetch_context(budget_s=400):
        relevance.canonicalize(cands, domain="тест", report=report)
    assert report["candidate_count_verified"] >= 6, report
    assert report["outcome"] == "partial"
    assert report["batches_succeeded"] == 2


def test_matrix_f_very_small_remaining_budget_starts_nothing(clock, monkeypatch):
    report, calls, cands = run_matrix(monkeypatch, clock, per_item=0.1, overhead=0.1, budget_s=6)
    s = _summary(report, calls, cands)
    # 6 s * SEMANTIC_PHASE_MAX_SHARE = 2.1 s < минимального бюджета вызова
    assert s["attempted"] == 0 and s["assessed"] == 0
    assert report["failure_type"] == "semantic_budget_exhausted"


def test_matrix_g_non_timeout_provider_error(clock, monkeypatch):
    report, calls, cands = run_matrix(monkeypatch, clock, per_item=0, overhead=0, budget_s=400,
                                      error=requests.ConnectionError("provider down"))
    s = _summary(report, calls, cands)
    assert s["assessed"] == 0 and s["outcome"] == "failed"
    assert report["failure_type"] == "ConnectionError"
    assert s["attempted"] <= relevance.SEMANTIC_MAX_ATTEMPTS, "бесконечных повторов быть не должно"


def test_no_single_attempt_consumes_the_whole_phase_budget(clock, monkeypatch):
    """Свойство, а не число: у первой попытки всегда остаётся запас на меньшую."""
    report, calls, cands = run_matrix(monkeypatch, clock, per_item=5.0, overhead=5.0, budget_s=400)
    phase_budget = min(400 * relevance.SEMANTIC_PHASE_MAX_SHARE,
                       max(400 * relevance.SEMANTIC_PHASE_SHARE, relevance.SEMANTIC_PHASE_FLOOR))
    assert calls["timeouts"][0] < phase_budget, (calls["timeouts"][0], phase_budget)
    assert phase_budget - calls["timeouts"][0] >= relevance.SEMANTIC_MIN_CALL_BUDGET


def test_semantic_strategy_is_deterministic_for_identical_provider_outcomes(clock, monkeypatch):
    runs = []
    for _ in range(3):
        c = Clock()
        monkeypatch.setattr(http, "monotonic", c.monotonic)
        report, calls, cands = run_matrix(monkeypatch, c, per_item=3.0, overhead=1.0, budget_s=400)
        runs.append((report["batch_sizes"], report["candidate_count_verified"]))
    assert len(set(map(str, runs))) == 1, runs


def test_new_semantic_constants_are_in_config_identity(monkeypatch):
    from wsignals import provenance
    cfg = provenance.behavioral_config()["semantic_verification"]
    assert cfg["min_batch"] == relevance.SEMANTIC_MIN_BATCH
    assert cfg["max_attempts"] == relevance.SEMANTIC_MAX_ATTEMPTS
    assert cfg["phase_floor_s"] == relevance.SEMANTIC_PHASE_FLOOR
    assert cfg["phase_max_share"] == relevance.SEMANTIC_PHASE_MAX_SHARE
    assert cfg["recovery_reserve_s"] == relevance.SEMANTIC_RECOVERY_RESERVE
    hype_cfg = provenance.behavioral_config()["hype_evidence"]
    assert hype_cfg["primary_source"] == "news" and hype_cfg["substitute"] == "document_hype_signal"
    base = provenance.config_hash()
    monkeypatch.setattr(relevance, "SEMANTIC_PHASE_FLOOR", 99.0)
    assert provenance.config_hash() != base


# ============================ P1-D: склейка не теряет проверку ============================

def _verified(phrase, burst, key, reject=None):
    return {"phrase": phrase, "burst": burst, "merge_key": key,
            "canonical_source": "gpt://f/yandexgpt-5-lite", "canonical_label": phrase,
            "assessment": {"source": "llm", "is_technology": True, "domain_relevance": 0.9,
                           "specificity": 1.0, "reject_reason": reject, "ambiguity_flags": [],
                           "canonical_name": phrase, "merge_key": key, "original_phrase": phrase,
                           "is_product_or_brand": False, "is_generic_phrase": False}}


def _plain(phrase, burst, key):
    return {"phrase": phrase, "burst": burst, "merge_key": key,
            "assessment": {"source": "deterministic", "is_technology": True,
                           "domain_relevance": 0.5, "specificity": 0.4, "reject_reason": None,
                           "ambiguity_flags": []}}


KEY = relevance.merge_key("federated learning")


def test_verified_member_survives_collapse_against_a_higher_burst_sibling():
    verified = _verified("federated learning", 5.0, KEY)
    plain = _plain("federated learnings", 12.0, KEY)
    kept, merged = relevance.collapse([verified, plain])
    assert len(kept) == 1
    assert query.verification_status(kept[0]["assessment"]) == SEMANTICALLY_VERIFIED
    assert kept[0]["canonical_source"]
    assert [c["phrase"] for c in merged] == ["federated learnings"]


def test_deterministic_only_family_is_unchanged():
    a, b = _plain("alpha beta", 9.0, KEY), _plain("alpha betas", 3.0, KEY)
    kept, merged = relevance.collapse([a, b])
    assert len(kept) == 1 and kept[0]["phrase"] == "alpha beta"
    assert query.verification_status(kept[0]["assessment"]) != SEMANTICALLY_VERIFIED


def test_model_rejection_is_never_promoted_into_a_verified_accept():
    rejected = _verified("federated learning", 5.0, KEY, reject="не технология")
    plain = _plain("federated learnings", 12.0, KEY)
    kept, _ = relevance.collapse([rejected, plain])
    assert len(kept) == 1
    assert query.verification_status(kept[0]["assessment"]) != SEMANTICALLY_VERIFIED
    assert not relevance.semantically_verified(rejected)


def test_collapse_outcome_is_invariant_under_order():
    import itertools
    members = [_verified("federated learning", 5.0, KEY),
               _plain("federated learnings", 12.0, KEY),
               _plain("learning federated", 8.0, KEY)]
    results = set()
    for order in itertools.permutations(range(3)):
        group = [dict(members[i]) for i in order]
        for g in group:
            g.pop("merged_from", None)
        kept, _ = relevance.collapse(group)
        results.add((kept[0]["phrase"], query.verification_status(kept[0]["assessment"])))
    assert results == {("federated learning", SEMANTICALLY_VERIFIED)}, results


def test_merged_from_and_canonical_label_stay_deterministic():
    verified = _verified("federated learning", 5.0, KEY)
    plain = _plain("federated learnings", 12.0, KEY)
    kept, _ = relevance.collapse([verified, plain])
    assert kept[0]["merged_from"] == ["federated learnings"]
    assert kept[0]["canonical_label"] == "federated learning"
    assert kept[0]["merge_key"] == KEY


# ============ P1-B: провайдер с фиксированной задержкой (микроремонт) ============
#
# Живое измерение владельца: порция из восьми фраз отвечает за ~8 с, и задержка почти не зависит от
# размера порции. Планировщик, срезающий предельный таймаут ниже этой величины, не успевает ничего.

OWNER_NETWORK_LEFT = 53.7          # столько остаётся к началу фазы на живом прогоне владельца


def owner_phase_budget():
    left = OWNER_NETWORK_LEFT
    return min(left * relevance.SEMANTIC_PHASE_MAX_SHARE,
               max(left * relevance.SEMANTIC_PHASE_SHARE, relevance.SEMANTIC_PHASE_FLOOR))


@pytest.mark.parametrize("overhead,per_item", [(5.0, 0.15), (6.0, 0.15), (7.0, 0.12)])
def test_fixed_overhead_provider_still_verifies(clock, monkeypatch, overhead, per_item):
    """Регрессия микроремонта: при фиксированной задержке 5–7 с проверка обязана состояться."""
    report, calls, cands = run_matrix(monkeypatch, clock, per_item=per_item, overhead=overhead,
                                      budget_s=OWNER_NETWORK_LEFT)
    s = _summary(report, calls, cands)
    assert s["assessed"] >= relevance.LLM_BATCH, (overhead, s)
    assert calls["sizes"][0] == relevance.LLM_BATCH
    assert calls["outcomes"][0] == "ok", "первая попытка обязана быть реалистичной для провайдера"


def test_first_attempt_is_provider_realistic_and_still_reserves_recovery(clock, monkeypatch):
    """Два требования сразу: первый вызов не короче измеренной живой задержки, резерв сохранён."""
    report, calls, cands = run_matrix(monkeypatch, clock, per_item=0.12, overhead=7.0,
                                      budget_s=OWNER_NETWORK_LEFT)
    budget = owner_phase_budget()
    first = calls["timeouts"][0]
    assert first >= 7.96, (first, "живой успешный вызов занимал 7,96 с")
    assert budget - first >= relevance.SEMANTIC_MIN_CALL_BUDGET, (budget, first)


def test_early_phase_is_single_shot_after_the_authorized_split(clock, monkeypatch):
    """ИЗМЕРЕННОЕ СЛЕДСТВИЕ разделения 10+8: у ранней фазы больше НЕТ попытки восстановления.

    При фазе 18 с `_attempt_timeout` отдавал первой попытке 10 с и оставлял 8 с на повтор
    половинной порцией. При 10 с остаток равен нулю, поэтому таймаут первой попытки означает ноль
    оценённых. Свойство восстановления не исчезло из кода — исчез бюджет, в котором оно работало.

    Тест фиксирует это ЯВНО, чтобы следствие нельзя было потерять молча."""
    assert relevance._attempt_timeout(18.0) == 10.0
    assert 18.0 - relevance._attempt_timeout(18.0) >= relevance.SEMANTIC_MIN_CALL_BUDGET, (
        "на 18 с повтор был возможен")
    early = relevance.EARLY_SEMANTIC_BUDGET
    assert early - relevance._attempt_timeout(early) < relevance.SEMANTIC_MIN_CALL_BUDGET, (
        "на 10 с повтор невозможен: первая попытка забирает всю фазу")
    report, calls, cands = run_matrix(monkeypatch, clock, per_item=3.0, overhead=1.0,
                                      budget_s=OWNER_NETWORK_LEFT)
    s = _summary(report, calls, cands)
    assert calls["outcomes"][0] == "timeout"
    assert s["assessed"] == 0, "восстановления в 10 с не происходит - это и есть цена разделения"


def test_semantic_phase_never_takes_the_whole_remaining_window(clock, monkeypatch):
    """Пол фазы ограничен долей остатка: на коротком бюджете фаза не съедает всё."""
    for left in (10.0, 20.0, 40.0, 53.7, 82.0):
        budget = min(left * relevance.SEMANTIC_PHASE_MAX_SHARE,
                     max(left * relevance.SEMANTIC_PHASE_SHARE, relevance.SEMANTIC_PHASE_FLOOR))
        assert budget <= left * relevance.SEMANTIC_PHASE_MAX_SHARE + 1e-9, (left, budget)
        assert budget < left, (left, budget)
