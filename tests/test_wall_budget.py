"""Настоящая гарантия по общему времени: резерв на финализацию, минимальный бюджет запроса (REPAIR E).

Время подменяется фальшивыми часами: утверждения о бюджете не должны зависеть от реального хода
времени и плавать от запуска к запуску.
"""
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import http, query  # noqa: E402


class Clock:
    """Фальшивые часы. Время двигается только явно и через sleep."""

    def __init__(self, start: float = 1000.0):
        self.t = start
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
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


class Response:
    def __init__(self, status_code=200, text="{}", headers=None, url="https://example.test/x"):
        self.status_code, self.text, self.headers, self.url = status_code, text, headers or {}, url


def respond(monkeypatch, clock, *, takes: float = 0.0, response=None, raises=None):
    """Сетевой вызов, который «занимает» заданное время на фальшивых часах."""
    calls = {"n": 0, "timeouts": []}

    def fake_get(self, url, params=None, headers=None, timeout=None):
        calls["n"] += 1
        calls["timeouts"].append(timeout)
        clock.advance(takes)
        if raises is not None:
            raise raises
        return response or Response()

    monkeypatch.setattr(requests.Session, "get", fake_get)
    return calls


# ---------- три числа бюджета ----------

def test_budget_plan_exposes_three_numbers():
    plan = query.budget_plan()
    assert set(plan) == {"hard_wall_budget_s", "finalization_margin_s", "network_deadline_s"}
    assert plan["network_deadline_s"] == plan["hard_wall_budget_s"] - plan["finalization_margin_s"]
    assert plan["finalization_margin_s"] > 0


@pytest.mark.parametrize("wall", [20, 60, 90, 180])
def test_network_deadline_always_leaves_room_for_finalization(wall):
    plan = query.budget_plan(wall)
    assert plan["network_deadline_s"] < plan["hard_wall_budget_s"]
    assert plan["network_deadline_s"] >= 1.0


def test_health_reports_the_same_three_numbers():
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlalchemy")
    import api.main as app

    health = app.health()
    for key in ("hard_wall_budget_s", "finalization_margin_s", "network_deadline_s"):
        assert key in health, key


# ---------- новая сетевая работа не начинается у самого дедлайна ----------

def test_request_is_not_started_below_the_minimum_budget(clock, monkeypatch):
    calls = respond(monkeypatch, clock)
    with http.fetch_context(budget_s=http.MIN_REQUEST_BUDGET_S / 2):
        with pytest.raises(http.FetchError) as e:
            http.get("https://example.test/a")
    assert e.value.error_type == http.DEADLINE
    assert calls["n"] == 0, "запрос не должен был даже стартовать"


def test_too_late_to_start_matches_the_minimum(clock):
    with http.fetch_context(budget_s=10):
        assert not http.too_late_to_start()
        clock.advance(10 - http.MIN_REQUEST_BUDGET_S + 0.01)
        assert http.too_late_to_start()


def test_retries_stop_before_the_deadline_instead_of_overrunning_it(clock, monkeypatch):
    calls = respond(monkeypatch, clock, takes=2.0, response=Response(503, "down"))
    with http.fetch_context(budget_s=6.0):
        with pytest.raises(http.FetchError):
            http.get("https://example.test/b")
    assert clock.t <= 1000.0 + 6.0 + 1e-9, f"вышли за дедлайн: {clock.t - 1000.0:.2f} с"
    assert calls["n"] >= 1


# ---------- таймаут запроса не превышает остаток ----------

def test_request_timeout_never_exceeds_the_remaining_budget(clock, monkeypatch):
    calls = respond(monkeypatch, clock)
    with http.fetch_context(budget_s=4.0):
        http.get("https://example.test/c")
        clock.advance(1.5)
        http.get("https://example.test/d")
    assert all(t <= 4.0 for t in calls["timeouts"]), calls["timeouts"]
    assert calls["timeouts"][-1] <= 2.5 + 1e-9, calls["timeouts"]


def test_there_is_no_half_second_floor_under_the_timeout(clock, monkeypatch):
    """Прежний пол в 0.5 с позволял запросу пережить дедлайн. Теперь запрос просто не стартует."""
    calls = respond(monkeypatch, clock)
    with http.fetch_context(budget_s=10.0):
        clock.advance(10.0 - 0.4)
        with pytest.raises(http.FetchError) as e:
            http.get("https://example.test/e")
    assert e.value.error_type == http.DEADLINE
    assert calls["n"] == 0
    assert 0.5 not in calls["timeouts"]


# ---------- паузы не выходят за дедлайн ----------

def test_sleep_is_clamped_to_the_remaining_budget(clock, monkeypatch):
    respond(monkeypatch, clock, response=Response(429, "slow", {"Retry-After": "600"}))
    with http.fetch_context(budget_s=3.0):
        with pytest.raises(http.FetchError):
            http.get("https://example.test/f")
    assert clock.t - 1000.0 <= 3.0 + 1e-9
    assert all(s <= 3.0 for s in clock.slept), clock.slept


def test_phase_budget_only_tightens_never_extends(clock):
    with http.fetch_context(budget_s=10.0):
        with http.phase(30.0):
            assert http.time_left() == pytest.approx(10.0)
        with http.phase(2.0):
            assert http.time_left() == pytest.approx(2.0)
        assert http.time_left() == pytest.approx(10.0)


# ---------- финализация не ходит в сеть ----------

def test_after_the_network_deadline_no_request_can_start(clock, monkeypatch):
    """Финализация идёт уже за сетевым дедлайном: любая попытка сети обязана падать мгновенно."""
    calls = respond(monkeypatch, clock)
    with http.fetch_context(budget_s=5.0):
        clock.advance(5.0)
        for _ in range(3):
            with pytest.raises(http.FetchError):
                http.get("https://example.test/g")
    assert calls["n"] == 0
    assert clock.t - 1000.0 == pytest.approx(5.0)


def test_documentation_does_not_promise_more_than_the_code_gives():
    """Проверяем, что в документации сетевой дедлайн и стена описаны раздельно."""
    doc = (ROOT / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8")
    assert "network_deadline_s" in doc and "finalization_margin_s" in doc and "hard_wall_budget_s" in doc
