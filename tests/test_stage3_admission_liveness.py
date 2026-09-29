"""Живость допуска на третьем этапе в измеренном продакшен-конверте (37da4c3 -> liveness repair).

Что здесь проверяется и почему именно так.

Три владельческих живых прогона на 37da4c3 дали одно и то же: stage3_admission_critical = 8,
stage3_workset = 14, stage3_maturity_incomplete = 8, stage3_admission_complete = 0, ACCEPT = 0.
До третьего этапа доживало 5,83 / 7,22 / 6,93 с при сетевом дедлайне 82 с.

Предыдущий свидетель пропускал дефект, потому что не моделировал ГЛОБАЛЬНЫЙ сериализатор по хосту:
`http._throttle` держит один замок на хост, и k-й запрос не может стартовать раньше, чем через
(k−1)·MIN_INTERVAL. Здесь используются НАСТОЯЩИЕ `_throttle`, `too_late_to_start` и настоящий
`cascade_collect`; подменены только сами адаптеры. Сеть не трогается.

Ключевая арифметика: news.google.com — 1,0 с на запрос, api.openalex.org — 0,12 с. Восемь кандидатов
это 8,0 с только на выдачу новостных запросов против 5,8–7,2 с всего окна. Плюс таймаут запроса равен
min(policy, остаток), поэтому ОДИН зависший новостной запрос съедает окно целиком, и полный OpenAlex
после него не начинается ни у кого.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsignals import http, query  # noqa: E402
from wsignals.sources import github, hackernews, news, openalex, wikipedia  # noqa: E402

OWNER_ENVELOPES = {"ai": 5.83, "cyber": 7.22, "fintech": 6.93}
# Полоса, в которой по владельческим прогонам реально оказывался третий этап.
CAPTURED_LOW, CAPTURED_HIGH = 5.5, 8.0


def _candidates(n: int) -> list[dict]:
    return [{"phrase": f"candidate technology {i}", "score": 1.0 - i * 0.01, "burst": 0.0}
            for i in range(n)]


def _adapters(*, news_hangs: bool, oa_fails: bool, news_latency: float = 0.8,
              oa_latency: float = 0.30):
    """Адаптеры, проходящие через НАСТОЯЩИЙ троттлинг и НАСТОЯЩИЙ дедлайн."""

    def gate(host: str) -> float | None:
        if http.too_late_to_start():
            raise http.FetchError("исчерпан общий бюджет запроса", http.DEADLINE)
        http._throttle(host)
        left = http.time_left()
        if left is not None and left < http.MIN_REQUEST_BUDGET_S:
            raise http.FetchError("исчерпан общий бюджет запроса", http.DEADLINE)
        return left

    def oa(q, light=False):
        left = gate("api.openalex.org")
        time.sleep(0.05 if light else oa_latency)
        if not light and oa_fails:
            raise http.FetchError("провайдер отказал", http.SERVER_ERROR)
        values = {"oa_total": 40, "oa_recent": 12, "oa_prior": 6, "oa_years": {"2025": 12}}
        if not light:
            values.update({"oa_fields": 5, "oa_preprint_share": 0.2, "oa_company_share": 0.1})
        return values

    def news_stats(q):
        left = gate("news.google.com")
        if news_hangs:
            # Отказавший запрос сжигает min(policy, остаток) — ровно как в проде.
            time.sleep(max(0.0, min(8.0, left if left is not None else 8.0)))
            raise http.FetchError("таймаут", http.TIMEOUT)
        time.sleep(news_latency)
        return {"news_count": 7, "news_hype": 0.1}

    # Второй этап (Википедия) и необязательное обогащение под тест не попадают: их троттлинг
    # намеренно не моделируется, иначе тест меряет не то, что заявляет.
    return oa, news_stats, (lambda q: {"wiki_article": 0, "wiki_age": 0}), \
        (lambda q: {"hn_points": 5}), (lambda q: {"gh_repos": 2})


_UPSTREAM: dict[tuple[int, int], float] = {}


def _upstream_cost(n: int, workers: int) -> float:
    """Сколько реально стоят ступени 1–2 на этой машине. Меряется один раз на конфигурацию."""
    key = (n, workers)
    if key not in _UPSTREAM:
        http._last.clear()
        http._locks.clear()
        oa, nw, wiki, hn, gh = _adapters(news_hangs=False, oa_fails=False)
        saved = (openalex.stats, news.stats, wikipedia.stats, hackernews.stats, github.stats)
        openalex.stats, news.stats = oa, nw
        wikipedia.stats, hackernews.stats, github.stats = wiki, hn, gh
        timings: dict = {}
        try:
            with http.fetch_context(budget_s=120.0):
                query.cascade_collect(_candidates(n), workers=workers, log=lambda *a: None,
                                      keep_wiki=n, keep_full=n,
                                      budget={"stage1": None, "stage2": None},
                                      corpus_adapters={}, with_github_stats=False,
                                      timings=timings, admission_workset=0)
        finally:
            openalex.stats, news.stats, wikipedia.stats, hackernews.stats, github.stats = saved
        _UPSTREAM[key] = timings["stage1_s"] + timings["stage2_s"]
    return _UPSTREAM[key]


@pytest.fixture
def stage3():
    """Прогоняет настоящий cascade_collect так, чтобы до третьего этапа дошёл заданный остаток."""

    def run(available: float, *, n: int = 14, workers: int = 8, news_hangs: bool = False,
            oa_fails: bool = False, reverse_order: bool = False, monkeypatch=None):
        http._last.clear()
        http._locks.clear()
        oa, nw, wiki, hn, gh = _adapters(news_hangs=news_hangs, oa_fails=oa_fails)
        saved = (openalex.stats, news.stats, wikipedia.stats, hackernews.stats, github.stats)
        saved_order = query._by_host_cost
        openalex.stats, news.stats = oa, nw
        wikipedia.stats, hackernews.stats, github.stats = wiki, hn, gh
        if reverse_order:
            # Воспроизведение дефекта 37da4c3: дорогой хост первым.
            query._by_host_cost = lambda sources: {
                name: fn for name, (fn, _host) in
                sorted(sources.items(), key=lambda kv: -http.MIN_INTERVAL.get(kv[1][1], 0.3))}
        timings: dict = {}
        try:
            # Стоимость ступеней 1–2 от условий третьего этапа не зависит, поэтому меряется один
            # раз на размер набора и переиспользуется: иначе калибровка гоняет зависшие новости
            # по нескольку раз на каждый случай.
            budget = available + _upstream_cost(n, workers)
            for _ in range(4):
                timings = {}
                http._last.clear()
                http._locks.clear()
                with http.fetch_context(budget_s=budget):
                    query.cascade_collect(_candidates(n), workers=workers, log=lambda *a: None,
                                          keep_wiki=n, keep_full=n,
                                          budget={"stage1": None, "stage2": None},
                                          corpus_adapters={}, with_github_stats=False,
                                          timings=timings, admission_workset=30)
                upstream = timings["stage1_s"] + timings["stage2_s"]
                if abs((budget - upstream) - available) < 0.35:
                    break
                budget = available + upstream
        finally:
            openalex.stats, news.stats, wikipedia.stats, hackernews.stats, github.stats = saved
            query._by_host_cost = saved_order
        timings["_available"] = round(budget - (timings["stage1_s"] + timings["stage2_s"]), 2)
        return timings

    return run


# --------------------------------------------------------------------------------------
# Воспроизведение дефекта и его закрытие
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("label", sorted(OWNER_ENVELOPES))
def test_owner_defect_reproduces_with_expensive_host_first(stage3, label):
    """С прежним порядком (новости первыми) полный OpenAlex не доезжает НИ У ОДНОГО кандидата."""
    t = stage3(OWNER_ENVELOPES[label], news_hangs=True, reverse_order=True)
    assert t["stage3_workset"] == 14
    assert t["stage3_admission_critical"] == 8, "порция допуска в владельческих прогонах = 8"
    assert t["stage3_admission_complete"] == 0, "именно это и наблюдалось живьём"
    assert t["stage3_maturity_incomplete"] == 8


@pytest.mark.parametrize("label", sorted(OWNER_ENVELOPES))
def test_owner_envelope_now_completes_admission(stage3, label):
    """Тот же конверт, тот же отказ новостей — но зрелость теперь измеряется полностью."""
    t = stage3(OWNER_ENVELOPES[label], news_hangs=True)
    assert CAPTURED_LOW <= t["_available"] <= CAPTURED_HIGH
    assert t["stage3_admission_complete"] > 0
    assert t["stage3_admission_complete"] == t["stage3_admission_critical"]
    assert t["stage3_maturity_incomplete"] == 0


# --------------------------------------------------------------------------------------
# Обязательная матрица A–G
# --------------------------------------------------------------------------------------

def test_case_A_both_healthy_admission_completes(stage3):
    t = stage3(6.5, news_hangs=False, oa_fails=False)
    assert t["stage3_admission_complete"] > 0


def test_case_B_openalex_full_fails_cannot_complete_admission(stage3):
    """Отказ полного OpenAlex — это НЕ измеренная зрелость: допуск закрыт явным состоянием."""
    t = stage3(6.5, news_hangs=False, oa_fails=True)
    assert t["stage3_admission_complete"] == 0
    assert t["stage3_maturity_incomplete"] == t["stage3_admission_critical"] > 0


def test_case_C_news_error_does_not_manufacture_admission(stage3):
    """Отказ новостей не мешает измерить зрелость, но и не выдаёт себя за измеренный хайп."""
    t = stage3(6.5, news_hangs=True, oa_fails=False)
    assert t["stage3_admission_complete"] > 0, "зрелость измерима и без новостей"
    assert t["stage3_maturity_incomplete"] == 0


def test_case_D_both_critical_providers_fail(stage3):
    t = stage3(6.5, news_hangs=True, oa_fails=True)
    assert t["stage3_admission_complete"] == 0
    assert t["stage3_maturity_incomplete"] > 0


@pytest.mark.parametrize("n", (12, 14))
def test_case_F_worksets(stage3, n):
    t = stage3(6.5, n=n, news_hangs=True)
    assert t["stage3_workset"] == n
    assert t["stage3_admission_complete"] > 0
    assert t["stage3_maturity_incomplete"] == 0


def test_case_G_global_budget_unchanged():
    plan = query.budget_plan()
    assert plan["hard_wall_budget_s"] == 90.0
    assert plan["finalization_margin_s"] == 8.0
    assert plan["network_deadline_s"] == 82.0


# --------------------------------------------------------------------------------------
# Инварианты самого ремонта
# --------------------------------------------------------------------------------------

def test_critical_order_is_derived_from_declared_throttles_not_hardcoded():
    order = list(query._by_host_cost({"news": (news.stats, "news.google.com"),
                                      "openalex": (openalex.stats, "api.openalex.org")}))
    assert order == ["openalex", "news"], "дешёвый хост обязан идти первым"
    # Порядок должен СЛЕДОВАТЬ из интервалов, а не совпасть с ними случайно.
    assert http.MIN_INTERVAL["api.openalex.org"] < http.MIN_INTERVAL["news.google.com"]
    flipped = query._by_host_cost({"a": (None, "news.google.com"), "b": (None, "api.github.com")})
    assert list(flipped) == ["a", "b"], "news 1.0 c дешевле github 2.2 c — порядок обязан это отражать"


def test_no_source_is_dropped_from_the_critical_pass():
    got = query._by_host_cost({"news": (news.stats, "news.google.com"),
                               "openalex": (openalex.stats, "api.openalex.org")})
    assert set(got) == {"news", "openalex"}
    assert got["openalex"] is openalex.stats and got["news"] is news.stats


def test_stage3_gate_still_refuses_to_start_a_chunk_without_room(stage3):
    """Ниже STAGE3_SECONDS_PER_CANDIDATE порция не начинается вовсе — и это не ACCEPT."""
    t = stage3(2.0, news_hangs=True)
    assert t["stage3_admission_critical"] == 0
    assert t["stage3_admission_complete"] == 0


def test_budget_constants_not_touched_by_this_repair():
    assert query.STAGE3_SECONDS_PER_CANDIDATE == 3.0
    assert query.BUDGET_SHARE == {"corpus": 0.32, "stage1": 0.18, "stage2": 0.16}
    assert query.LIVE_KEEP_FULL == 14 and query.LIVE_KEEP_WIKI == 24
    assert http.MIN_REQUEST_BUDGET_S == 1.5
