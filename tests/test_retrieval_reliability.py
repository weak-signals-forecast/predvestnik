"""Живой путь: таймауты, 429/5xx, частичный отказ, общий дедлайн, кэш как источник (PHASE 6, PHASE 8).

Сети в тестах нет: `requests.Session.get` подменяется, кэш пишется во временный каталог.
"""
import json
import sys
import time
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import evidence, http  # noqa: E402


class FakeResponse:
    def __init__(self, status_code=200, text="{}", headers=None, url="https://example.test/x"):
        self.status_code, self.text, self.headers, self.url = status_code, text, headers or {}, url


# Профиль живого пути с теми же числом попыток и семантикой, но без настоящих пауз:
# тесты проверяют правила повторов, а не умение ждать.
FAST = http.RetryPolicy(retries=http.LIVE.retries, timeout=http.LIVE.timeout, backoff_base=0.01, backoff_cap=0.02)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(http, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(http, "MIN_INTERVAL", {})
    monkeypatch.setattr(http, "_last", {})
    monkeypatch.setattr(http, "LIVE", FAST)
    yield


def respond(monkeypatch, *responses):
    """Подменяет сетевой вызов последовательностью ответов или исключений."""
    calls = {"n": 0}
    seq = list(responses)

    def fake_get(self, url, params=None, headers=None, timeout=None):
        calls["n"] += 1
        item = seq[min(calls["n"] - 1, len(seq) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(requests.Session, "get", fake_get)
    return calls


# ---------- классы ошибок ----------

def test_timeout_is_reported_as_timeout_not_as_zero(monkeypatch):
    calls = respond(monkeypatch, requests.Timeout("too slow"))
    with http.fetch_context(budget_s=5):
        with pytest.raises(http.FetchError) as e:
            http.get("https://example.test/a")
    assert e.value.error_type == http.TIMEOUT
    assert calls["n"] == FAST.retries + 1                # ретраи ограничены профилем живого пути


def test_429_is_retried_a_bounded_number_of_times(monkeypatch):
    calls = respond(monkeypatch, FakeResponse(429, "slow down", {"Retry-After": "0"}))
    with http.fetch_context(budget_s=5):
        with pytest.raises(http.FetchError) as e:
            http.get("https://example.test/b")
    assert e.value.error_type == http.RATE_LIMIT and e.value.status_code == 429
    assert calls["n"] == FAST.retries + 1


def test_5xx_is_server_error(monkeypatch):
    respond(monkeypatch, FakeResponse(503, "oops"))
    with http.fetch_context(budget_s=5):
        with pytest.raises(http.FetchError) as e:
            http.get("https://example.test/c")
    assert e.value.error_type == http.SERVER_ERROR and e.value.status_code == 503


def test_4xx_fails_immediately_without_retries(monkeypatch):
    calls = respond(monkeypatch, FakeResponse(404, "nope"))
    with http.fetch_context(budget_s=5):
        with pytest.raises(http.FetchError) as e:
            http.get("https://example.test/d")
    assert e.value.error_type == http.HTTP_ERROR and calls["n"] == 1


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_have_their_own_type_and_carry_the_server_message(monkeypatch, status):
    """REPAIR D: просроченный токен нельзя путать с кривым запросом — иначе отказ неотличим."""
    calls = respond(monkeypatch, FakeResponse(status, '{"message": "Bad credentials"}'))
    with http.fetch_context(budget_s=5):
        with pytest.raises(http.FetchError) as e:
            http.get("https://example.test/auth")
    assert e.value.error_type == http.AUTH_ERROR and e.value.status_code == status
    assert "Bad credentials" in str(e.value)
    assert calls["n"] == 1          # повторов на отказе авторизации нет


def test_rate_limited_403_is_still_a_rate_limit_not_an_auth_error(monkeypatch):
    respond(monkeypatch, FakeResponse(403, "API rate limit exceeded for 1.2.3.4"))
    with http.fetch_context(budget_s=5):
        with pytest.raises(http.FetchError) as e:
            http.get("https://example.test/rl")
    assert e.value.error_type == http.RATE_LIMIT


def test_malformed_json_is_a_parse_error_not_empty_data(monkeypatch):
    respond(monkeypatch, FakeResponse(200, "<html>not json</html>"))
    with http.fetch_context(budget_s=5):
        with pytest.raises(http.FetchError) as e:
            http.get_json("https://example.test/e")
    assert e.value.error_type == http.PARSE


def test_retry_succeeds_after_transient_failure(monkeypatch):
    respond(monkeypatch, FakeResponse(503, "oops"), FakeResponse(200, '{"ok": 1}'))
    with http.fetch_context(budget_s=5):
        assert http.get_json("https://example.test/f") == {"ok": 1}


# ---------- общий дедлайн ----------

def test_global_deadline_stops_further_attempts(monkeypatch):
    respond(monkeypatch, requests.Timeout("slow"))
    with http.fetch_context(budget_s=0.01):
        time.sleep(0.02)
        with pytest.raises(http.FetchError) as e:
            http.get("https://example.test/g")
    assert e.value.error_type == http.DEADLINE


def test_sleep_never_exceeds_remaining_budget(monkeypatch):
    respond(monkeypatch, FakeResponse(429, "slow down", {"Retry-After": "600"}))
    with http.fetch_context(budget_s=0.4):
        t0 = time.monotonic()
        with pytest.raises(http.FetchError):
            http.get("https://example.test/h")
        assert time.monotonic() - t0 < 2.0       # без ограничения это было бы 600 секунд


def test_batch_profile_keeps_the_long_retry_budget():
    """Офлайн-сбор по-прежнему терпелив, живой путь — нет. FAST сохраняет retries и timeout живого профиля."""
    assert http.BATCH.retries > FAST.retries and http.BATCH.timeout > FAST.timeout
    assert FAST.retries <= 3 and FAST.timeout <= 10


# ---------- кэш как реально полученное свидетельство ----------

def test_cache_hit_is_marked_with_its_own_retrieval_time(monkeypatch):
    respond(monkeypatch, FakeResponse(200, '{"v": 1}'))
    with http.fetch_context(budget_s=5):
        http.get("https://example.test/i")
        fresh = http.taken_fetches()
        http.get("https://example.test/i")
        cached = http.taken_fetches()
    assert fresh[0]["from_cache"] is False and cached[0]["from_cache"] is True
    assert cached[0]["retrieved_at"] == fresh[0]["retrieved_at"]      # дата получения, а не дата показа
    assert cached[0]["stale"] is False


def test_stale_cache_is_used_on_failure_and_marked_stale(monkeypatch):
    respond(monkeypatch, FakeResponse(200, '{"v": 1}'))
    with http.fetch_context(budget_s=5):
        http.get("https://example.test/j")
        http.taken_fetches()
    respond(monkeypatch, FakeResponse(503, "down"))
    with http.fetch_context(budget_s=5):
        body = http.get("https://example.test/j", max_age_days=0)     # свежий кэш не подходит, источник лежит
        meta = http.taken_fetches()
    assert json.loads(body) == {"v": 1}
    assert meta[-1]["stale"] is True and meta[-1]["from_cache"] is True


def test_no_cache_and_failure_means_error_not_invented_data(monkeypatch):
    respond(monkeypatch, FakeResponse(503, "down"))
    with http.fetch_context(budget_s=5):
        with pytest.raises(http.FetchError):
            http.get("https://example.test/k")


# ---------- частичный успех на уровне наблюдений ----------

def test_partial_adapter_failure_leaves_the_query_alive(monkeypatch):
    state = evidence.EvidenceState()
    state.add(evidence.observe("openalex", lambda q: {"oa_total": 12}, "x"))
    state.add(evidence.observe("github", lambda q: (_ for _ in ()).throw(http.FetchError("t", http.TIMEOUT)), "x"))
    state.add(evidence.observe("news", lambda q: {"news_1y": 4}, "x"))
    assert sorted(state.failed()) == ["github"]
    assert state.completeness() == pytest.approx(2 / 3)
    assert state.values() == {"oa_total": 12, "news_1y": 4}


def test_observation_carries_cache_provenance(monkeypatch):
    respond(monkeypatch, FakeResponse(200, '{"oa_total": 3}'))
    with http.fetch_context(budget_s=5):
        obs = evidence.observe("openalex", lambda q: http.get_json("https://example.test/l"), "x")
    assert obs.status == evidence.OK and obs.retrieved_at and obs.from_cache is False


# ---------- экономия запросов к ленте новостей (REPAIR 6) ----------

def test_news_stats_uses_one_request_when_the_feed_is_not_saturated(monkeypatch):
    """Свежие новости считаются по датам годовой ленты: лента троттлится по секунде на запрос."""
    from wsignals.sources import news

    calls = []

    def fake_items(query, when, lang, exact=True):
        calls.append(when)
        fresh = http.now_utc()
        old = fresh - __import__("datetime").timedelta(days=200)
        return [{"title": "a", "link": "x", "source": "Reuters", "date": fresh},
                {"title": "b", "link": "y", "source": "PR Newswire", "date": old}]

    monkeypatch.setattr(news, "_items", fake_items)
    out = news.stats("agent identity")
    assert calls == ["1y"]                       # второй запрос не делается
    assert out["news_1y"] == 2 and out["news_30d"] == 1
    assert out["news_sources"] == 2 and out["news_press_share"] == 0.5


def test_news_stats_falls_back_to_a_second_request_when_saturated(monkeypatch):
    from wsignals.sources import news

    calls = []

    def fake_items(query, when, lang, exact=True):
        calls.append(when)
        n = news.SATURATED if when == "1y" else 7
        return [{"title": "t", "link": "u", "source": "Reuters", "date": http.now_utc()} for _ in range(n)]

    monkeypatch.setattr(news, "_items", fake_items)
    out = news.stats("agent identity")
    assert calls == ["1y", "30d"]                # насыщенную ленту по датам считать нельзя
    assert out["news_1y"] == news.SATURATED and out["news_30d"] == 7
