"""Сквозной проход открытого запроса без сети (PHASE 8).

Все адаптеры подменены. Проверяется то, что нельзя проверить на уровне отдельных функций:
запрос выживает при частичном отказе, отказ всех источников не порождает принятых сигналов,
в карточке нет выдуманных ссылок, происхождение доезжает до карточки.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import http, query  # noqa: E402

# Настоящий сборщик заголовков: фикстура offline подменяет его, а часть тестов проверяет
# именно его поведение и возвращает оригинал на место.
REAL_NEWS_TITLES = query._news_titles

# «agent attestation» несёт головное слово-процесс и подтверждается детерминированно (R2-A);
# «agent identity» лексически чистая, но детерминированных оснований для подтверждения не даёт
# и должна уходить в список наблюдения.
WORK_RECENT = [
    {"id": "W1", "title": "Agent attestation for autonomous payment rails",
     "publication_date": "2026-02-01", "type": "article", "doi": "https://doi.org/10.1/w1",
     "primary_location": {"source": {"display_name": "Nature Fintech"}},
     "abstract_inverted_index": {"Agent": [0], "attestation": [1], "reduces": [2], "fraud": [3], "in": [4],
                                 "autonomous": [5], "payment": [6], "rails": [7], "across": [8], "banks": [9],
                                 "agent": [10], "identity": [11]}},
    {"id": "W2", "title": "Agent attestation and agent identity in banking", "publication_date": "2026-03-01",
     "type": "preprint", "doi": "https://doi.org/10.1/w2",
     "primary_location": {"source": {"display_name": "arXiv"}},
     "abstract_inverted_index": {"Agent": [0], "attestation": [1], "verification": [2], "improves": [3],
                                 "trust": [4], "between": [5], "autonomous": [6], "banking": [7], "agents": [8],
                                 "agent": [9], "identity": [10]}},
    {"id": "W3", "title": "Agent attestation for agent identity", "publication_date": "2025-11-01",
     "type": "preprint", "doi": "https://doi.org/10.1/w3",
     "primary_location": {"source": {"display_name": "arXiv"}},
     "abstract_inverted_index": {"Agent": [0], "attestation": [1], "enables": [2], "safe": [3],
                                 "delegation": [4], "for": [5], "autonomous": [6], "agents": [7],
                                 "agent": [8], "identity": [9]}},
]
REPOS = [{"full_name": "acme/agent-attestation", "html_url": "https://github.com/acme/agent-attestation",
          "description": "agent attestation and agent identity for autonomous agents",
          "created_at": "2026-01-02T00:00:00Z",
          "stargazers_count": 120, "topics": ["agent-attestation", "agent-identity"]},
         {"full_name": "beta-labs/attestation-kit", "html_url": "https://gitlab.com/beta-labs/attestation-kit",
          "description": "agent attestation toolkit", "created_at": "2026-02-02T00:00:00Z",
          "stargazers_count": 40, "topics": ["agent-attestation"]}]

OA_FULL = {"oa_total": 40, "oa_recent": 20, "oa_prior": 3, "oa_growth": 1.8, "oa_age": 2, "oa_peak_ratio": 1.4,
           "oa_peak_year": 2026, "oa_last_share": 0.45, "oa_preprint_share": 0.35, "oa_company_share": 0.2,
           "oa_fields": 3, "oa_years": [0] * 14 + [2, 18, 20]}
GH = {"gh_total": 30, "gh_12m": 25, "gh_new_share": 0.83, "gh_max_stars_log": 4.8}
NEWS = {"news_1y": 8, "news_30d": 3, "news_sources": 5, "news_press_share": 0.1}
HN = {"hn_12m": 5, "hn_prior24m": 1, "hn_total": 6}
WIKI = {"wiki_mentions_log": 1.0, "wiki_article": 0, "wiki_age": 0.0, "wiki_views_log": 0.0}


def boom(*_a, **_kw):
    raise http.FetchError("источник недоступен", http.TIMEOUT)


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Подменяет все сетевые адаптеры. Возвращает словарь, через который тест ломает отдельные источники."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("WSIGNALS_FORECAST_REGISTRY", str(tmp_path / "registry.jsonl"))
    monkeypatch.setattr(http, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(query.llm, "translate_query", lambda t: "agent identity")
    monkeypatch.setattr(query.llm, "summarize", lambda *a, **kw: None)
    monkeypatch.setattr(query.llm, "model_name", lambda: None)
    monkeypatch.setattr(query.ru, "technology_name",
                        lambda p: {"ru": p, "original": p, "source": "тест"})
    monkeypatch.setattr(query.takeoff, "predict", lambda f: None)
    monkeypatch.setattr(query.takeoff, "register", lambda *a, **kw: None)
    monkeypatch.setattr(query.timemachine, "find_twin", lambda f: None)
    monkeypatch.setattr(query, "background_phrases", lambda: set())
    monkeypatch.setattr(query, "background_counts", lambda: (1000, __import__("collections").Counter()))
    monkeypatch.setattr(query, "_works", lambda seed, years, n, domains=query.TECH_DOMAINS:
                        WORK_RECENT if years.startswith("2025") else [])
    monkeypatch.setattr(query, "_new_repos", lambda seed, pages=2: (REPOS, None))
    monkeypatch.setattr(query, "_news_titles",
                        lambda seeds: (["Startup raises for agent identity attestation"], []))
    monkeypatch.setattr(query.news, "search", lambda *a, **kw: [])

    stubs = {"openalex": OA_FULL, "github": GH, "news": NEWS, "hn": HN, "wiki": WIKI}
    monkeypatch.setattr(query.openalex, "stats", lambda q, light=False: _give(stubs, "openalex"))
    monkeypatch.setattr(query.github, "stats", lambda q: _give(stubs, "github"))
    monkeypatch.setattr(query.news, "stats", lambda q: _give(stubs, "news"))
    monkeypatch.setattr(query.hackernews, "stats", lambda q: _give(stubs, "hn"))
    import wsignals.sources.wikipedia as wikipedia
    monkeypatch.setattr(wikipedia, "stats", lambda q: _give(stubs, "wiki"))
    return stubs


@pytest.fixture
def semantic(monkeypatch):
    """Моделирует доступную семантическую нормализацию: оценка кандидата помечается как пришедшая
    от модели. Без этой фикстуры конвейер работает в режиме отказа нормализатора (R2-B)."""
    from wsignals import relevance

    def canonicalize(cands, domain="", batch=8, report=None):
        if report is not None:
            report.update({"configured": True, "attempted": True, "outcome": "successful",
                           "failure_type": None, "candidate_count_requested": len(cands),
                           "candidate_count_verified": len(cands), "candidate_count_unverified": 0,
                           "model": "gpt://test-folder/yandexgpt-5-lite"})
        for c in cands:
            a = dict(c.get("assessment") or {})
            a.update({"source": "llm", "is_technology": True, "reject_reason": None,
                      "domain_relevance": 0.9})
            c["assessment"] = a
            c["canonical_source"] = "gpt://test-folder/yandexgpt-5-lite"
        return [c for c in cands if not c.get("noise_reason")], [c for c in cands if c.get("noise_reason")]

    monkeypatch.setattr(query.relevance, "canonicalize", canonicalize)
    monkeypatch.setattr(relevance, "canonicalize", canonicalize)
    return canonicalize


def _give(stubs, name):
    value = stubs[name]
    if value is None:
        boom()
    return dict(value)


def run(**kw):
    return query.run("слабые сигналы в платежах ИИ-агентов", top=5, n_candidates=20, workers=4,
                     log=lambda _l: None, budget_s=30, **kw)


# ---------- нормальный проход ----------

def test_pipeline_accepts_a_corroborated_candidate(offline, semantic):
    res = run()
    assert res["signals"], res["stats"]
    card = res["signals"][0]
    for key in ("emergence_score", "maturity_risk", "evidence_confidence", "hype_risk",
                "decision", "decision_reasons", "legacy_model_probability"):
        assert key in card, key
    assert card["decision"] == "ACCEPT"
    assert card["evidence"]["corroborating_families"]


def test_no_fabricated_citations(offline, semantic):
    """Каждая ссылка в карточке ведёт на документ, реально пришедший от подменённых адаптеров."""
    allowed = {w["doi"] for w in WORK_RECENT} | {r["html_url"] for r in REPOS}
    for card in run()["signals"]:
        for src in card["sources"]:
            assert src["url"] in allowed, src["url"]
            assert src["title"]
        case = card.get("case")
        if isinstance(case, dict):
            assert case["url"] in allowed


def test_card_carries_retrieval_provenance(offline, semantic):
    card = run()["signals"][0]
    assert card["retrieved_at"]
    for src in card["sources"]:
        assert src["retrieved_at"] == card["retrieved_at"]
        assert src["source_family"] and src["adapter_status"] in ("OK", "STALE_CACHE")
        assert "published_at" in src and "canonical_url" in src


def test_original_phrase_and_canonical_label_are_both_stored(offline, semantic):
    card = run()["signals"][0]
    assert card["original_phrase"] and card["canonical_label"]


# ---------- отказы адаптеров ----------

def test_one_adapter_failure_does_not_fail_the_query(offline):
    offline["news"] = None
    res = run()
    assert res["stats"]["отказы адаптеров"].get("news", 0) > 0
    assert res["signals"] or res["excluded"]          # запрос завершился, а не упал


def test_corpus_repository_failure_marks_the_code_family_as_error(offline, monkeypatch):
    """Семейство «код» берётся из корпуса запроса; его отказ это ERROR, а не ноль репозиториев."""
    monkeypatch.setattr(query, "_new_repos", boom)
    res = run()
    assert res["stats"]["адаптеры корпуса"]["github"] == "ERROR"
    assert res["stats"]["отказы адаптеров"].get("github", 0) > 0


def test_adapter_failure_lowers_evidence_confidence(offline, semantic):
    healthy = run()["signals"][0]
    offline["news"] = None
    degraded = [c for c in run()["signals"] if c["technology_original"] == healthy["technology_original"]]
    if degraded:
        assert degraded[0]["evidence_confidence"] <= healthy["evidence_confidence"]


def test_all_adapters_unavailable_yields_no_accepted_signals(offline):
    for name in offline:
        offline[name] = None
    res = run()
    assert res["signals"] == []
    assert all(e["decision"] in ("LOW_EVIDENCE", "NOISE") for e in res["excluded"]), res["excluded"][:3]


def test_openalex_corpus_failure_is_isolated(offline, monkeypatch):
    monkeypatch.setattr(query, "_works", boom)
    res = run()                                        # раньше это роняло весь запрос
    assert res["stats"]["адаптеры корпуса"]["openalex"] == "ERROR"


def test_valid_zero_results_are_not_an_adapter_failure(offline):
    offline["news"] = {"news_1y": 0, "news_30d": 0, "news_sources": 0, "news_press_share": 0.0}
    res = run()
    assert "news" not in res["stats"]["отказы адаптеров"]


# ---------- шлюзы решения на живом проходе ----------

def test_mature_candidate_is_vetoed_end_to_end(offline):
    offline["wiki"] = {"wiki_mentions_log": 8.0, "wiki_article": 1, "wiki_age": 12.0, "wiki_views_log": 12.0}
    res = run()
    assert res["signals"] == []
    assert any(e["decision"] == "MATURE" for e in res["excluded"])


def test_hype_candidate_is_vetoed_end_to_end(offline):
    offline["news"] = {**NEWS, "news_press_share": 0.9}
    res = run()
    assert res["signals"] == []
    assert any(e["decision"] == "HYPE" for e in res["excluded"])


def test_lead_only_evidence_is_rejected_end_to_end(offline):
    offline["openalex"] = {**OA_FULL, "oa_total": 0, "oa_recent": 0, "oa_prior": 0, "oa_growth": 0.0,
                           "oa_years": [0] * 17}
    offline["github"] = {"gh_total": 0, "gh_12m": 0, "gh_new_share": 0.0, "gh_max_stars_log": 0.0}
    offline["news"] = {"news_1y": 0, "news_30d": 0, "news_sources": 0, "news_press_share": 0.0}
    res = run()
    assert res["signals"] == []
    assert any(e["decision"] == "LOW_EVIDENCE" for e in res["excluded"])


def test_stats_report_rejection_counts_by_reason(offline):
    stats = run()["stats"]
    for key in ("принято (ACCEPT)", "отклонено LOW_EVIDENCE", "отклонено MATURE", "отклонено HYPE",
                "отклонено NOISE", "отказы адаптеров", "источников на принятого кандидата",
                "бюджет, с", "секунд"):
        assert key in stats, key


def test_query_plan_is_recorded(offline):
    res = run()
    assert res["query_plan"]["source"] == "deterministic"
    assert res["stats"]["план запроса"]["source"] == "deterministic"


# ---------- REPAIR 1: ACCEPT требует конкретного документа ----------

def test_no_accepted_card_has_an_empty_source_list(offline, semantic):
    res = run()
    assert res["signals"]
    for card in res["signals"]:
        assert card["sources"], card["technology_original"]
        assert card["supporting_document_count"] >= 1


def test_candidate_without_concrete_documents_is_not_accepted(offline, monkeypatch):
    """Фраза, найденная только в заголовках новостей: агрегаты есть, конкретных документов нет."""
    monkeypatch.setattr(query, "_works", lambda *a, **kw: [])
    monkeypatch.setattr(query, "_new_repos", lambda *a, **kw: ([], None))
    monkeypatch.setattr(query, "_news_titles",
                        lambda seeds: (["agent identity attestation raises seed round"] * 3, []))
    res = run()
    assert res["signals"] == []
    codes = {e.get("code") for e in res["excluded"]}
    assert codes & {"supporting_evidence_not_materialized", "insufficient_corroborating_families",
                    "no_independent_corroboration", "all_adapters_failed", "semantic_rejection"}


# ---------- REPAIR 4: семантика баллов ----------

def test_card_exposes_all_seven_public_score_fields(offline, semantic):
    card = run()["signals"][0]
    for key in ("emergence_score", "maturity_risk", "evidence_confidence", "hype_risk",
                "legacy_model_probability", "legacy_model_reliable", "rank_score"):
        assert key in card, key
    assert card["score"] == card["rank_score"]          # `score` в БД это ранжирующий балл


def test_ui_never_calls_a_heuristic_axis_model_confidence():
    ui = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    assert "Уверенность модели" not in ui
    assert "экспериментальный базовый предиктор" in ui
    assert "ранжирующий балл" in ui.lower()


# ---------- REPAIR 7: ТОП и список наблюдения ----------

def test_counts_are_reported_separately(offline):
    stats = run()["stats"]
    for key in ("accepted_count", "watchlist_count", "rejected_count"):
        assert key in stats and isinstance(stats[key], int)


def test_watchlist_is_never_labelled_a_confirmed_weak_signal(offline):
    res = run()
    for card in res.get("watchlist", []):
        assert card["tier"] == "требует дополнительных доказательств"
        assert card["decision"] != "ACCEPT"
    for card in res["signals"]:
        assert card["tier"] == "подтверждённый слабый сигнал"
        assert card["decision"] == "ACCEPT"


# ---------- REPAIR 6: тайминги по этапам ----------

def test_stage_timings_are_recorded(offline):
    t = run()["timings"]
    for key in ("candidate_generation_s", "semantic_filter_s", "stage1_s", "stage2_s", "stage3_s",
                "document_enrichment_s", "synthesis_s", "total_wall_s"):
        assert key in t, key
        assert isinstance(t[key], (int, float))


# ---------- REPAIR 3: склейка доходит до выдачи ----------

def test_phrase_variants_do_not_occupy_two_top_slots(offline, semantic):
    keys = [c.get("merge_key") for c in run()["signals"]]
    assert len(keys) == len(set(keys))


def test_dead_openalex_does_not_starve_the_other_corpus_sources(offline, monkeypatch):
    """Мёртвый OpenAlex не должен съедать бюджет этапа и лишать запрос репозиториев и новостей."""
    monkeypatch.setattr(query, "_works", boom)
    res = run()
    adapters = res["stats"]["адаптеры корпуса"]
    assert adapters["openalex"] == "ERROR"
    assert adapters["github"] == "OK", adapters          # репозитории всё равно собраны
    assert res["stats"]["новых репозиториев"] >= 1


# ---------- claim-relative: релевантность и таксономия новостного корпуса на сквозном проходе ----------

def test_corpus_news_failure_is_an_error_not_no_results(offline, monkeypatch):
    """Отказ новостной ленты на сборе корпуса больше не выдаётся за ответ «ничего не найдено»."""
    monkeypatch.setattr(query, "_news_titles", REAL_NEWS_TITLES)
    monkeypatch.setattr(query.news, "search", boom)
    res = run()
    assert res["stats"]["адаптеры корпуса"]["news"] == "ERROR"
    assert res["stats"]["отказы новостей на сборе корпуса"]
    assert res["stats"]["заголовков новостей"] == 0


def test_corpus_news_empty_feed_stays_no_results(offline, monkeypatch):
    """Разобранная пустая лента — это ответ источника, и статус остаётся NO_RESULTS."""
    monkeypatch.setattr(query, "_news_titles", lambda seeds: ([], []))
    res = run()
    assert res["stats"]["адаптеры корпуса"]["news"] == "NO_RESULTS"
    assert "отказы новостей на сборе корпуса" not in res["stats"]


def test_every_accepted_source_is_claim_relevant(offline, semantic):
    """На живом пути каждый засчитанный документ несёт разметку релевантности кандидату."""
    for card in run()["signals"]:
        quality = card["evidence_quality"]
        for claim in quality["claims"]:
            for ref in claim["documents"]:
                assert ref["claim_relevant"] is True, (card["technology_original"], ref["title"])
                assert ref["claim_match"], ref
        assert quality["supporting_document_count"] >= 1
