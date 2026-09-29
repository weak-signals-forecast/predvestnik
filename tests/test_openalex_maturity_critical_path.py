"""Полный опрос OpenAlex критичен для допуска: лёгкие нули первой ступени не заменяют зрелость.

Измеренный дефект: у одного и того же кандидата полный OpenAlex давал зрелость 0,74 и вето MATURE,
а отложенный — 0,58 и ACCEPT. Разница целиком в полях, которые лёгкий запрос ступени 1 возвращает
нулями (широта областей, доля препринтов, доля компаний). Отложенное обогащение ослабляло вето,
то есть НЕСДЕЛАННАЯ работа делала допуск легче.

Сеть не используется, часы фальшивые, живых провайдеров нет.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import http, query, trust  # noqa: E402
from wsignals.decision import (ACCEPT, LOW_EVIDENCE, MATURE, SEMANTICALLY_VERIFIED,  # noqa: E402
                               decide, maturity)
from wsignals.evidence import NO_RESULTS, OK, EvidenceState, Observation  # noqa: E402
from wsignals.schema import Document  # noqa: E402
from wsignals.sources import github, hackernews, news, openalex, wikipedia  # noqa: E402

# Поля, которые возвращает ТОЛЬКО полный опрос; лёгкий отдаёт по ним нули.
FULL_ONLY = ("oa_fields", "oa_preprint_share", "oa_company_share")
MATURE_FULL = {"oa_total": 3000, "oa_recent": 900, "oa_prior": 400, "oa_growth": 0.8, "oa_age": 7,
               "oa_peak_ratio": 1.1, "oa_last_share": 0.30, "oa_preprint_share": 0.15,
               "oa_company_share": 0.2, "oa_fields": 6, "oa_peak_year": 2025}
LIGHT_ONLY = {**MATURE_FULL, "oa_fields": 0, "oa_preprint_share": 0.0, "oa_company_share": 0.0}


def doc(title, url, source_type="научная публикация", name="J"):
    d = trust.assess(Document(title=title, url=url, source_name=name, source_type=source_type,
                              published="2026-03-01", language="en")).with_provenance(
        retrieved_at="2026-09-21T00:00:00+00:00")
    d.claim_relevant = True
    return d


DOCS = [doc("Lattice calibration for cryogenic interconnects", "https://doi.org/10.1038/a", name="Nature"),
        doc("someorg/lattice-cal: toolkit", "https://github.com/someorg/lattice-cal",
            "репозиторий", "GitHub")]


def _state(oa_values):
    s = EvidenceState()
    s.add(Observation("openalex", OK, dict(oa_values)))
    s.add(Observation("news", OK, {"news_1y": 8, "news_30d": 3, "news_30d_share": 0.37,
                                   "news_press_share": 0.1}))
    s.add(Observation("hn", OK, {"hn_total": 6, "hn_12m": 5, "hn_growth": 1.1}))
    s.add(Observation("wiki", NO_RESULTS, {"wiki_article": 0, "wiki_age": 0}))
    return s


# ============ 1 и 7. измеренный свидетель и смысл лёгких нулей ============

def test_measured_witness_light_zeros_weaken_the_maturity_veto():
    """Свидетель фиксируется КАК ИЗМЕРЕНИЕ: разница существует и направлена в сторону допуска."""
    full = maturity(_state(MATURE_FULL))[0]
    light = maturity(_state(LIGHT_ONLY))[0]
    assert full > light, (full, light)
    assert decide(_state(MATURE_FULL), burst=0.8, documents=DOCS,
                  verification=SEMANTICALLY_VERIFIED).decision == MATURE
    assert decide(_state(LIGHT_ONLY), burst=0.8, documents=DOCS,
                  verification=SEMANTICALLY_VERIFIED).decision == ACCEPT


def test_light_request_really_does_return_zeros_for_full_only_fields(monkeypatch):
    """Лёгкий ответ неотличим от измеренного нуля — потому и нужен внешний признак полноты."""
    monkeypatch.setattr(openalex, "_groups", lambda q, field: {"2025": 5} if field == "publication_year" else {})
    light = openalex.stats("x", light=True)
    for field in FULL_ONLY:
        assert light[field] == 0 or light[field] == 0.0, field


# ============ конвейер ============

class Clock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def advance(self, d):
        self.t += d

    def sleep(self, d):
        self.t += d


@pytest.fixture
def pipeline(monkeypatch):
    clock, calls = Clock(), []

    monkeypatch.setattr(http, "monotonic", clock.monotonic)
    monkeypatch.setattr(http.time, "sleep", clock.sleep)

    def make(full_mode="ok", n=8):
        def oa(phrase, light=False, *a, **kw):
            clock.advance(0.4 if light else 1.6)
            calls.append(("openalex_light" if light else "openalex_full", phrase))
            if light:
                return dict(LIGHT_ONLY)
            if full_mode == "error":
                raise RuntimeError("openalex full failed")
            return dict(MATURE_FULL)

        def stub(name, values, cost):
            def fn(phrase, *a, **kw):
                clock.advance(cost)
                calls.append((name, phrase))
                return dict(values)
            return fn

        monkeypatch.setattr(openalex, "stats", oa)
        monkeypatch.setattr(wikipedia, "stats", stub("wiki", {"wiki_article": 0, "wiki_age": 0}, 0.3))
        monkeypatch.setattr(news, "stats", stub("news", {
            "news_1y": 8, "news_30d": 3, "news_30d_share": 0.37, "news_press_share": 0.1}, 0.5))
        monkeypatch.setattr(hackernews, "stats", stub("hn", {
            "hn_total": 6, "hn_12m": 5, "hn_growth": 1.1}, 0.4))
        monkeypatch.setattr(github, "stats", stub("github", {
            "gh_total": 30, "gh_12m": 25, "gh_new_share": 0.8}, 0.5))
        return n

    return clock, calls, make


def _cands(n, noise=0):
    out = [{"phrase": f"lattice calibration {i}", "original_phrase": f"lattice calibration {i}",
            "canonical_label": f"lattice calibration {i}", "burst": float(100 - i),
            "channels": {"литература": 3}, "recent_docs": 3, "prior_docs": 0,
            "examples": [], "repos": [], "assessment": {"specificity": 1.0}} for i in range(n)]
    for i in range(noise):
        c = dict(out[0])
        c.update({"phrase": f"noise {i}", "original_phrase": f"noise {i}",
                  "canonical_label": f"noise {i}", "noise_reason": "пересказ задачи поиска"})
        out.append(c)
    return out


def _run(pipeline, remaining, full_mode="ok", n=8, noise=0):
    clock, calls, make = pipeline
    make(full_mode=full_mode, n=n)
    timings = {}
    with http.fetch_context(budget_s=remaining):
        rows = query.cascade_collect(_cands(n, noise), workers=4, log=lambda *a: None,
                                     keep_wiki=24, keep_full=14,
                                     budget={"stage1": None, "stage2": None}, corpus_adapters={},
                                     with_github_stats=False, timings=timings,
                                     admission_workset=24)
    return rows, timings, calls


# ============ 2. полный OpenAlex сохраняет вето ============

def test_full_openalex_success_keeps_the_mature_veto(pipeline):
    rows, timings, _ = _run(pipeline, 40.0)
    done = [r for r in rows if r.get("openalex_full")]
    assert done, "полный опрос обязан состояться при достаточном бюджете"
    for row in done:
        assert row["state"].values()["oa_fields"] == MATURE_FULL["oa_fields"]
        assert decide(row["state"], burst=0.8, documents=DOCS,
                      verification=SEMANTICALLY_VERIFIED).decision == MATURE


# ============ 3 и 8. недоехавший полный опрос не даёт ACCEPT ============

def test_candidate_without_full_openalex_cannot_accept(pipeline):
    rows, timings, _ = _run(pipeline, 12.0)
    for row in rows:
        if row.get("stage") == 3 and not row.get("openalex_full"):
            assert row.get("decision") == LOW_EVIDENCE
            assert row.get("reason_code") in {"maturity_not_fully_measured", "budget_exhausted"}
            assert row.get("reasons"), "причина обязана быть явной"


def test_budget_incomplete_candidate_is_explicit_and_not_accepted(pipeline):
    rows, timings, _ = _run(pipeline, 12.0)
    incomplete = [r for r in rows if r.get("reason_code") in
                  {"maturity_not_fully_measured", "budget_exhausted"}]
    assert incomplete, "при тесном бюджете часть кандидатов обязана остаться неполной"
    assert all(r["decision"] == LOW_EVIDENCE for r in incomplete)


# ============ 4. отказ полного опроса не облегчает допуск ============

def test_full_openalex_error_cannot_make_acceptance_easier(pipeline):
    rows, timings, _ = _run(pipeline, 40.0, full_mode="error")
    stage3 = [r for r in rows if r.get("stage") == 3]
    assert stage3
    assert all(not r.get("openalex_full") for r in stage3)
    assert all(r.get("decision") == LOW_EVIDENCE for r in stage3)
    assert timings["stage3_maturity_incomplete"] == len(stage3)


# ============ 5. новости не голодают и идут раньше необязательного ============

def test_news_is_not_starved_and_precedes_optional_enrichment(pipeline):
    rows, timings, calls = _run(pipeline, 40.0)
    order = [name for name, _ in calls]
    assert "news" in order and "hn" in order
    assert order.index("news") < order.index("hn")
    measured = {p for name, p in calls if name == "news"}
    assert all(r["cand"]["phrase"] in measured for r in rows if r.get("stage") == 3)


def test_openalex_full_is_on_the_critical_path_before_optional(pipeline):
    rows, timings, calls = _run(pipeline, 40.0)
    order = [name for name, _ in calls]
    assert order.index("openalex_full") < order.index("hn")


# ============ 6. терминальные по-прежнему без запросов ============

def test_terminal_candidates_still_issue_no_requests(pipeline):
    rows, timings, calls = _run(pipeline, 40.0, n=6, noise=5)
    asked = {p for _, p in calls}
    terminal = {r["cand"]["phrase"] for r in rows if r["cand"].get("noise_reason")}
    assert timings["stage1_skipped_terminal"] == 5
    assert asked.isdisjoint(terminal)


# ============ отсутствие ложных переворотов ============

def test_no_false_accept_flip_remains_when_only_full_openalex_is_missing(pipeline):
    """Единственная разница — недоехавший полный опрос. ACCEPT не должен появляться."""
    rows, _, _ = _run(pipeline, 12.0)
    flips = 0
    for row in rows:
        if row.get("stage") != 3 or row.get("openalex_full"):
            continue
        # по лёгким значениям слой решения сам по себе дал бы ACCEPT
        would = decide(row["state"], burst=0.8, documents=DOCS,
                       verification=SEMANTICALLY_VERIFIED).decision
        if would == ACCEPT and row.get("decision") != LOW_EVIDENCE:
            flips += 1
    assert flips == 0


def test_no_deadline_or_worker_count_was_increased():
    plan = query.budget_plan()
    assert plan["hard_wall_budget_s"] == 90.0 and plan["network_deadline_s"] == 82.0
    assert query.STAGE3_SECONDS_PER_CANDIDATE == 3
