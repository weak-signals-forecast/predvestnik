"""Финальная сборка к демонстрации: ремонт только доказанных дефектов выдачи.

  1. Ширина третьего этапа была потолком подтверждённых на ВЕСЬ запрос (14 < 15).
  2. Строгий контракт строк модели: bool вместо числа, дубли индексов, противоречивые строки.
  3. Отказ провайдера и недопроверенный поиск отделены от настоящего «ничего не найдено».
  4. Записи без записанного решения ACCEPT не показываются как подтверждённые.
  5. Прогноз взлёта и двойник не выдаются по вектору признаков с нулями вместо измерений.
  6. Пометки «цитата» и «сгенерировано» соответствуют фактическому происхождению текста.

Все тесты герметичны: сети и живой модели нет.
"""
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import db, http, query, relevance  # noqa: E402

# Настоящие реализации до любых подмен фикстур: тест ТОП-15 по настоящему пути возвращает их на место.
REAL_CANONICALIZE = relevance.canonicalize
REAL_FILTER = relevance.filter_candidates

MODEL_URI = "gpt://test-folder/yandexgpt-5-lite"


# ---------- 2. строгий контракт строк модели ----------

def _row(i, **kw):
    base = {"i": i, "canonical_name": "tech name", "is_technology": True, "domain_relevance": 0.9,
            "is_product_or_brand": False, "is_generic_phrase": False, "merge_key": None, "reject_reason": None}
    base.update(kw)
    return base


def _cands(n=3):
    return [{"phrase": f"zorvian lattice {k}", "burst": 3.0 - k} for k in range(n)]


def test_bool_index_is_not_an_index():
    by_index = dict(enumerate(_cands()))
    assert relevance._validate_row(_row(True), by_index) is None
    assert relevance._validate_row(_row(False), by_index) is None


def test_bool_relevance_is_not_a_confidence():
    by_index = dict(enumerate(_cands()))
    assert relevance._validate_row(_row(0, domain_relevance=True), by_index) is None
    assert relevance._validate_row(_row(0, domain_relevance=False), by_index) is None
    assert relevance._validate_row(_row(0, domain_relevance=1), by_index) is not None


@pytest.mark.parametrize("i", [7, -1, 1.0, "0", None])
def test_foreign_or_malformed_index_is_rejected(i):
    assert relevance._validate_row(_row(i), dict(enumerate(_cands()))) is None


def _canonicalize(monkeypatch, rows, n=3):
    monkeypatch.setattr(relevance, "configured_model", lambda: (MODEL_URI, None))
    monkeypatch.setattr(relevance, "_call_batch", lambda chunk, domain, uri, timeout: rows)
    cands = _cands(n)
    report = {}
    keep, dropped = relevance.canonicalize(cands, report=report)
    return cands, keep, dropped, report


@pytest.mark.parametrize("rows", [
    [_row(0), _row(0)],                                           # дубль
    [_row(0, is_technology=False), _row(0)],                      # отказ, затем подтверждение
    [_row(0), _row(0, is_technology=False)],                      # подтверждение, затем отказ
])
def test_duplicate_or_contradictory_rows_leave_candidate_unverified(monkeypatch, rows):
    cands, keep, dropped, report = _canonicalize(monkeypatch, rows)
    c0 = cands[0]
    assert not relevance.semantically_verified(c0)
    assert "assessment" not in c0 and not c0.get("noise_reason")
    assert c0 in keep and c0 not in dropped
    assert report["candidate_count_verified"] == 0
    assert report["candidate_count_unverified"] == len(cands)


def test_duplicates_cannot_inflate_verified_count_to_success(monkeypatch):
    """Четыре строки об одном кандидате раньше давали verified=4 и outcome=successful."""
    _cands_, _k, _d, report = _canonicalize(monkeypatch, [_row(0)] * 4, n=4)
    assert report["candidate_count_verified"] == 0
    assert report["outcome"] == "failed"


def test_missing_row_stays_unverified_and_others_count_once(monkeypatch):
    cands, _keep, _dropped, report = _canonicalize(monkeypatch, [_row(0), _row(2)])
    assert relevance.semantically_verified(cands[0]) and relevance.semantically_verified(cands[2])
    assert not relevance.semantically_verified(cands[1])
    assert report["candidate_count_verified"] == 2
    assert report["candidate_count_verified"] + report["candidate_count_unverified"] == len(cands)


@pytest.mark.parametrize("junk", ["0", 0, None, [0], 1.5, True])
def test_non_dict_row_gains_no_authority(monkeypatch, junk):
    """Не-словарь в списке строк ничего не подтверждает и не мешает соседним валидным строкам."""
    cands, keep, dropped, report = _canonicalize(monkeypatch, [junk, _row(1)])
    assert not relevance.semantically_verified(cands[0]) and "assessment" not in cands[0]
    assert relevance.semantically_verified(cands[1])
    assert report["candidate_count_verified"] == 1 and not dropped


def test_only_non_dict_rows_verify_nothing(monkeypatch):
    cands, _keep, dropped, report = _canonicalize(monkeypatch, ["x", 1, None, [{"i": 0}]])
    assert not any(relevance.semantically_verified(c) for c in cands)
    assert report["candidate_count_verified"] == 0 and report["outcome"] == "failed" and not dropped


def test_late_pass_non_dict_row_gains_no_authority(monkeypatch):
    monkeypatch.setattr(relevance, "configured_model", lambda: (MODEL_URI, None))
    monkeypatch.setattr(relevance, "_late_call_batch", lambda *a, **k: ["garbage", None, 0])
    target = {"phrase": "zorvian lattice", "original_phrase": "zorvian lattice"}
    report = {}
    assert relevance.verify_late([target], report=report, budget_s=8.0) == 0
    assert report["valid_rows"] == 0 and not relevance.semantically_verified(target)


def test_late_pass_rejects_duplicate_rows(monkeypatch):
    monkeypatch.setattr(relevance, "configured_model", lambda: (MODEL_URI, None))
    monkeypatch.setattr(relevance, "_late_call_batch", lambda *a, **k: [_row(0), _row(0)])
    target = {"phrase": "zorvian lattice", "original_phrase": "zorvian lattice"}
    report = {}
    assert relevance.verify_late([target], report=report, budget_s=8.0) == 0
    assert report["valid_rows"] == 0 and report["verified_count"] == 0
    assert not relevance.semantically_verified(target)


def test_late_pass_rejects_bool_relevance(monkeypatch):
    monkeypatch.setattr(relevance, "configured_model", lambda: (MODEL_URI, None))
    monkeypatch.setattr(relevance, "_late_call_batch", lambda *a, **k: [_row(0, domain_relevance=True)])
    target = {"phrase": "zorvian lattice", "original_phrase": "zorvian lattice"}
    assert relevance.verify_late([target], budget_s=8.0) == 0
    assert not relevance.semantically_verified(target)


# ---------- 1. ширина третьего этапа и ТОП-15 ----------

def test_stage3_width_never_below_requested_top():
    assert query.LIVE_KEEP_FULL == 14               # объявленная настройка не менялась
    assert query.stage3_width(15) == 15
    assert query.stage3_width(5) == query.LIVE_KEEP_FULL
    assert query.stage3_width(30) == 30


def test_keep_full_rule_is_in_the_config_fingerprint():
    from wsignals import provenance
    assert provenance.behavioral_config()["retrieval"]["keep_full_rule"] == "max(keep_full, top)"


MODS = ["zorvian", "quellic", "brantic", "dovric", "felmic", "gorvic", "halvic", "jostric", "kelvric",
        "lornic", "mervic", "norvic", "pelvric", "quorvic", "relmic", "sorvic", "tolvric", "urvic"]
HEADS = ["lattice", "prism", "vortex", "quill", "ember", "fjord", "glyph", "harbor", "beacon", "jade",
         "kelp", "lumen", "mosaic", "nectar", "spindle", "pylon", "quartz", "rune"]
OA = {"oa_total": 40, "oa_recent": 20, "oa_prior": 3, "oa_growth": 1.8, "oa_age": 2, "oa_peak_ratio": 1.4,
      "oa_peak_year": 2026, "oa_last_share": 0.45, "oa_preprint_share": 0.35, "oa_company_share": 0.2,
      "oa_fields": 3, "oa_years": [0] * 14 + [2, 18, 20]}
NEWS = {"news_1y": 8, "news_30d": 3, "news_sources": 5, "news_press_share": 0.1}
HN = {"hn_12m": 5, "hn_prior24m": 1, "hn_total": 6}
GH = {"gh_total": 30, "gh_12m": 25, "gh_new_share": 0.83, "gh_max_stars_log": 4.8}
WIKI = {"wiki_mentions_log": 1.0, "wiki_article": 0, "wiki_age": 0.0, "wiki_views_log": 0.0}


def _synthetic(k: int) -> dict:
    phrase = f"{MODS[k]} {HEADS[k]}"
    slug = phrase.replace(" ", "-")
    works = [{"id": f"W{k}{j}", "title": f"{phrase.capitalize()} for autonomous payment rails {j}",
              "publication_date": "2026-02-01", "type": "preprint" if j else "article",
              "doi": f"https://doi.org/10.1/{slug}-{j}",
              "primary_location": {"source": {"display_name": "arXiv" if j else "Nature Fintech"}},
              "abstract_inverted_index": {w: [i] for i, w in enumerate(
                  f"{phrase} reduces fraud in autonomous payment rails across banks".split())}}
             for j in range(3)]
    repos = [{"full_name": f"org{k}/{slug}", "html_url": f"https://github.com/org{k}/{slug}",
              "description": f"{phrase} for autonomous agents", "created_at": "2026-01-02T00:00:00Z",
              "stargazers_count": 50, "topics": [slug]}]
    return {"phrase": phrase, "original_phrase": phrase, "canonical_label": phrase, "burst": 3.0,
            "recent_docs": 3, "prior_docs": 0, "examples": works, "repos": repos,
            "channels": {"литература": 3, "препринты": 2, "репозитории": 1}}


@pytest.fixture
def eligible(monkeypatch, tmp_path):
    """18 одинаково готовых кандидатов с разными именами: каждый проходит все проверки."""
    n = len(MODS)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("WSIGNALS_FORECAST_REGISTRY", str(tmp_path / "registry.jsonl"))
    monkeypatch.setattr(http, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(query.llm, "summarize", lambda *a, **kw: None)
    monkeypatch.setattr(query.llm, "model_name", lambda: None)
    monkeypatch.setattr(query.ru, "technology_name", lambda p: {"ru": p, "original": p, "source": "тест"})
    monkeypatch.setattr(query.takeoff, "register", lambda *a, **kw: None)
    corpus = {"работ за 2025–2026": 3 * n, "работ за 2019–2022": 0, "новых репозиториев": n,
              "заголовков новостей": 0, "адаптеры корпуса": {"openalex": "OK", "github": "OK", "news": "OK"}}
    monkeypatch.setattr(query, "extract_candidates", lambda seeds, k: ([_synthetic(i) for i in range(n)],
                                                                      dict(corpus)))
    monkeypatch.setattr(query.relevance, "filter_candidates", lambda cands, seeds, excl: (cands, []))

    def canonicalize(cands, domain="", batch=8, report=None):
        if report is not None:
            report.update({"configured": True, "attempted": True, "outcome": "successful",
                           "failure_type": None, "candidate_count_requested": len(cands),
                           "candidate_count_verified": len(cands), "candidate_count_unverified": 0,
                           "model": MODEL_URI})
        for c in cands:
            c["assessment"] = {"source": "llm", "is_technology": True, "reject_reason": None,
                               "domain_relevance": 0.9, "specificity": 1.0}
            c["canonical_source"] = MODEL_URI
        return cands, []

    monkeypatch.setattr(query.relevance, "canonicalize", canonicalize)
    monkeypatch.setattr(query.news, "search", lambda *a, **kw: [])
    monkeypatch.setattr(query.openalex, "stats", lambda q, light=False: dict(OA))
    monkeypatch.setattr(query.news, "stats", lambda q: dict(NEWS))
    monkeypatch.setattr(query.hackernews, "stats", lambda q: dict(HN))
    monkeypatch.setattr(query.github, "stats", lambda q: dict(GH))
    import wsignals.sources.wikipedia as wikipedia
    monkeypatch.setattr(wikipedia, "stats", lambda q: dict(WIKI))
    return n


def _run(top=15):
    return query.run("слабые сигналы в платежах ИИ-агентов", top=top, n_candidates=40, workers=8,
                     log=lambda _l: None, budget_s=60)


def test_fifteen_confirmed_when_fifteen_or_more_are_eligible(eligible):
    res = _run()
    assert res["stats"]["accepted_count"] >= 15, res["stats"]["причины отклонения"]
    assert len(res["signals"]) == 15
    assert all(c["decision"] == "ACCEPT" for c in res["signals"])
    assert res["timings"]["stage3_keep_full"] == 15
    assert len({c["technology_original"] for c in res["signals"]}) == 15


def test_cap15_real_semantic_path(eligible, monkeypatch):
    """ТОП-15 через НАСТОЯЩИЕ filter_candidates, canonicalize, _validate_rows, collapse, каскад и decide.

    Подменён только внешний вызов модели (`_call_batch`): модель отвечает строками по схеме, и
    подтверждение кандидаты получают тем же путём, что и в живом запросе. ACCEPT не впрыскивается."""
    monkeypatch.setattr(relevance, "canonicalize", REAL_CANONICALIZE)   # снимаем подмены фикстуры
    monkeypatch.setattr(relevance, "filter_candidates", REAL_FILTER)
    monkeypatch.setattr(relevance, "configured_model", lambda: (MODEL_URI, None))
    calls = []

    def model(chunk, domain, uri, timeout):
        calls.append(len(chunk))
        return [{"i": i, "canonical_name": c["phrase"], "is_technology": True, "domain_relevance": 0.9,
                 "is_product_or_brand": False, "is_generic_phrase": False, "merge_key": None,
                 "reject_reason": None} for i, c in enumerate(chunk)]
    monkeypatch.setattr(relevance, "_call_batch", model)
    res = _run(top=15)
    sem = res["run"]["semantic_normalizer"]
    assert calls and sem["outcome"] == "successful", sem
    assert sem["candidate_count_verified"] == eligible
    assert len(res["signals"]) == 15, res["stats"]["причины отклонения"]
    assert all(c["decision"] == "ACCEPT" and c["verification"] == "SEMANTICALLY_VERIFIED"
               and (c["assessment"] or {}).get("source") == "llm" for c in res["signals"])


def test_cap15_real_path_fails_closed_without_model_rows(eligible, monkeypatch):
    """Контроль той же фикстуры: модель вернула мусор — подтверждённых нет, ACCEPT не возникает."""
    monkeypatch.setattr(relevance, "canonicalize", REAL_CANONICALIZE)
    monkeypatch.setattr(relevance, "filter_candidates", REAL_FILTER)
    monkeypatch.setattr(relevance, "configured_model", lambda: (MODEL_URI, None))
    monkeypatch.setattr(relevance, "_call_batch", lambda chunk, domain, uri, timeout: ["x", None])
    res = _run(top=15)
    assert res["signals"] == []
    assert res["stats"]["search_completeness"]["state"] != query.COMPLETE


def test_old_width_was_a_whole_request_ceiling(eligible, monkeypatch):
    """Контрольная рука: прежняя ширина 14 даёт не больше 14 подтверждённых на весь запрос,
    хотя готовых кандидатов 18. Остальные ни в выдачу, ни в список наблюдения не попадают."""
    monkeypatch.setattr(query, "stage3_width", lambda top: 14)
    res = _run()
    assert res["stats"]["accepted_count"] == 14
    assert len(res["signals"]) == 14


def test_completeness_is_complete_when_everything_answered(eligible):
    comp = _run(top=eligible)["stats"]["search_completeness"]
    assert comp["state"] == query.COMPLETE, comp
    assert comp["semantic_verification_available"] is True


def test_width_cut_is_reported_as_not_evaluated_not_as_provider_failure(eligible):
    """18 готовых при ширине 15: трое не опрашивались. Это недопроверка, а не отказ источника."""
    res = _run()
    codes = res["stats"]["причины отклонения"]
    assert codes.get("not_evaluated_stage3_width") == eligible - 15, codes
    assert "hype_evidence_unavailable" not in codes
    comp = res["stats"]["search_completeness"]
    assert comp["state"] == query.INCOMPLETE_SEARCH and not comp["provider_errors"], comp


# ---------- 3. отказ провайдера против «ничего не найдено»: причинность ----------

SEM_OK = {"configured": True, "outcome": "successful", "batches_attempted": 2, "batches_succeeded": 2}
NO_SIGNALS_CLAIM = "ни один кандидат не прошёл"


def _comp(corpus=None, corpus_types=None, cand=None, codes=None, semantic=None, late=None, accepted=0):
    return query.search_completeness(corpus or {"openalex": "OK", "github": "OK", "news": "OK"},
                                     corpus_types or {}, cand or {}, Counter(codes or {}),
                                     SEM_OK if semantic is None else semantic, late or {}, accepted)


def test_completeness_genuine_complete_zero():
    comp = _comp()
    assert comp["state"] == query.COMPLETE
    assert NO_SIGNALS_CLAIM in comp["message"]


def test_completeness_real_corpus_provider_failure():
    comp = _comp(corpus={"openalex": "ERROR", "github": "OK", "news": "OK"},
                 corpus_types={"openalex": ["server_error"]})
    assert comp["state"] == query.PROVIDER_ERROR
    assert "openalex" in comp["message"] and "server_error" in comp["message"]


def test_completeness_real_candidate_provider_failure():
    comp = _comp(cand={"news": {"timeout": 6}}, codes={"hype_evidence_unavailable": 6})
    assert comp["state"] == query.PROVIDER_ERROR
    assert any("news" in p for p in comp["provider_errors"])


def test_completeness_deadline_only_is_incomplete_not_provider():
    """DeadlineExceeded — наш бюджет, а не источник: ни на корпусе, ни по кандидатам."""
    comp = _comp(corpus={"openalex": "PARTIAL", "github": "OK", "news": "ERROR"},
                 corpus_types={"openalex": ["deadline"], "news": ["deadline"]},
                 cand={"news": {"deadline": 5}, "openalex": {"deadline": 5}},
                 codes={"hype_evidence_unavailable": 5})
    assert comp["state"] == query.INCOMPLETE_SEARCH, comp
    assert comp["provider_errors"] == []
    assert "не ответил" not in comp["message"]
    assert "бюджет времени" in comp["message"]


def test_completeness_semantic_budget_exhausted_before_any_call():
    comp = _comp(semantic={"configured": True, "outcome": "failed", "failure_type": "semantic_budget_exhausted",
                           "batches_attempted": 0, "batches_succeeded": 0},
                 codes={"semantic_verification_pending": 4})
    assert comp["state"] == query.INCOMPLETE_SEARCH, comp
    assert "не ответил" not in comp["message"]
    assert "не начата" in comp["message"]


def test_completeness_semantic_call_really_failed_is_provider_error():
    comp = _comp(semantic={"configured": True, "outcome": "failed", "failure_type": "ReadTimeout",
                           "batches_attempted": 2, "batches_succeeded": 0},
                 late={"verified_count": 0})
    assert comp["state"] == query.PROVIDER_ERROR
    assert "ReadTimeout" in comp["message"]


def test_completeness_late_pass_internal_vs_provider():
    internal = _comp(late={"attempt_count": 0, "failure_type": "late_budget_exhausted", "verified_count": 0})
    assert internal["state"] == query.INCOMPLETE_SEARCH and "не ответил" not in internal["message"]
    after = _comp(late={"attempt_count": 1, "failure_type": "late_result_after_deadline", "verified_count": 0})
    assert after["state"] == query.INCOMPLETE_SEARCH and "не ответил" not in after["message"]
    failed = _comp(late={"attempt_count": 1, "failure_type": "ConnectionError", "verified_count": 0})
    assert failed["state"] == query.PROVIDER_ERROR


def test_completeness_width_cut_is_incomplete():
    comp = _comp(codes={"not_evaluated_stage3_width": 3})
    assert comp["state"] == query.INCOMPLETE_SEARCH and comp["provider_errors"] == []


def test_completeness_budget_truncation_is_incomplete():
    comp = _comp(codes={"budget_exhausted": 3}, accepted=2)
    assert comp["state"] == query.INCOMPLETE_SEARCH
    assert "Выдача может быть неполной" in comp["message"]


def test_completeness_mixed_provider_and_budget_keeps_provider_error():
    comp = _comp(cand={"news": {"timeout": 2, "deadline": 4}}, codes={"budget_exhausted": 3})
    assert comp["state"] == query.PROVIDER_ERROR
    assert comp["incomplete_reasons"], comp
    assert "Кроме того, поиск выполнен не полностью" in comp["message"]


def test_completeness_no_key_is_incomplete_with_explicit_message():
    comp = _comp(semantic={"configured": False, "failure_type": "LLM_PROVIDER не равен yandexgpt"},
                 codes={"semantic_verification_pending": 4})
    assert comp["state"] == query.INCOMPLETE_SEARCH
    assert comp["semantic_verification_available"] is False
    assert "модель не настроена" in comp["message"] and "не может быть подтверждён" in comp["message"]


def test_zero_candidates_with_configured_model_is_not_called_unconfigured():
    comp = _comp(semantic={"configured": False, "failure_type": None})
    assert comp["semantic_verification_available"] is True
    assert "не настроена" not in comp["message"]


@pytest.mark.parametrize("kw", [
    {"cand": {"news": {"timeout": 1}}},
    {"cand": {"news": {"deadline": 1}}},
    {"codes": {"not_evaluated_stage3_width": 1}},
    {"semantic": {"configured": False, "failure_type": "нет ключа"}},
    {"cand": {"news": {"timeout": 1, "deadline": 1}}, "codes": {"budget_exhausted": 1}},
])
def test_no_non_complete_state_claims_that_no_signals_exist(kw):
    comp = _comp(**kw)
    assert comp["state"] != query.COMPLETE
    assert NO_SIGNALS_CLAIM not in comp["message"]
    assert "не означает, что их нет" in comp["message"]


def test_run_level_deadline_is_incomplete(eligible, monkeypatch):
    """Сквозной путь: все опросы новостей по кандидатам упёрлись в наш дедлайн."""
    def deadline(q):
        raise http.DeadlineExceeded("исчерпан общий бюджет запроса")
    monkeypatch.setattr(query.news, "stats", deadline)
    comp = _run(top=eligible)["stats"]["search_completeness"]
    assert comp["state"] == query.INCOMPLETE_SEARCH, comp
    assert comp["provider_errors"] == [] and "не ответил" not in comp["message"]


def test_run_level_real_provider_failure(eligible, monkeypatch):
    def timeout(q):
        raise http.FetchError("news: read timeout", http.TIMEOUT)
    monkeypatch.setattr(query.news, "stats", timeout)
    comp = _run(top=eligible)["stats"]["search_completeness"]
    assert comp["state"] == query.PROVIDER_ERROR, comp
    assert any("news" in p for p in comp["provider_errors"])


def test_corpus_error_types_are_recorded(monkeypatch, tmp_path):
    """Класс отказа корпуса доходит до статистики: иначе дедлайн нельзя отличить от отказа."""
    monkeypatch.setattr(http, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(query, "background_phrases", lambda: set())
    monkeypatch.setattr(query, "background_counts", lambda: (1000, Counter()))

    def works(seed, years, n, domains=query.TECH_DOMAINS):
        raise http.DeadlineExceeded("исчерпан общий бюджет запроса")
    monkeypatch.setattr(query, "_works", works)
    monkeypatch.setattr(query, "_new_repos", lambda seed, pages=2: ([], "server_error"))
    monkeypatch.setattr(query, "_news_titles", lambda seeds: ([], []))
    _picked, stats = query.extract_candidates(["zorvian lattice"], 10)
    assert stats["типы отказов корпуса"] == {"openalex": ["deadline"], "github": ["server_error"]}


# ---------- 4. записи без записанного решения ----------

def test_confirmed_tier_matches_the_authoritative_terminal_state():
    assert db.CONFIRMED_TIER == query.terminal_state(query.ACCEPT)[0]
    assert db.WATCHLIST_TIER == query.terminal_state(query.LOW_EVIDENCE)[0]


def test_legacy_rows_are_never_confirmed(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_url", lambda: f"sqlite:///{tmp_path / 'legacy.db'}")
    monkeypatch.setattr(db, "_engine", None)
    from sqlalchemy.orm import Session

    run_id = db.new_run("legacy")
    with Session(db.engine()) as s:
        run = s.get(db.SearchRun, run_id)
        run.status = "done"
        cards = [("accepted", db.CONFIRMED_TIER, {"decision": "ACCEPT"}),
                 ("watch", db.WATCHLIST_TIER, {"decision": "LOW_EVIDENCE"}),
                 ("old null tier", None, {}),                             # до колонки tier
                 ("old tier no decision", db.CONFIRMED_TIER, {}),          # до слоя решения
                 ("contradictory", db.CONFIRMED_TIER, {"decision": "LOW_EVIDENCE"})]
        for i, (name, tier, extra) in enumerate(cards, 1):
            run.signals.append(db.Signal(rank=i, technology=name, score=0.1, tier=tier,
                                         card={"technology": name, **extra}))
        s.commit()
    got = db.get_run(run_id)
    assert [c["technology"] for c in got["signals"]] == ["accepted"]
    assert [c["technology"] for c in got["watchlist"]] == ["watch"]
    assert [c["technology"] for c in got["unverified_legacy"]] == [
        "old null tier", "old tier no decision", "contradictory"]


# ---------- 5–6. карточка: прогноз взлёта, двойник, пометки происхождения ----------

FAKE_TAKEOFF = {"probability": 0.9, "percentile": 0.95, "horizon": "2029–2031", "baseline_per_year": 5.0,
                "criterion": "x"}
FAKE_TWIN = {"name": "twin", "query": "q", "year": 2021, "fate": "стала мейнстримом", "label_now": "mature",
             "distance": 1.0}


def test_takeoff_and_twin_hidden_without_measured_vector(eligible, monkeypatch):
    """Живой путь по умолчанию: счётчики GitHub не запрашиваются — прогноза и двойника нет."""
    monkeypatch.setattr(query.takeoff, "predict", lambda f: dict(FAKE_TAKEOFF))
    monkeypatch.setattr(query.timemachine, "find_twin", lambda f: dict(FAKE_TWIN))
    monkeypatch.setattr(query, "LIVE_GITHUB_STATS", False)
    card = _run(top=3)["signals"][0]
    assert card["forward_looking_supported"] is False
    assert card["takeoff"] is None and card["historical_twin"] is None


def test_takeoff_shown_only_with_full_vector(eligible, monkeypatch):
    monkeypatch.setattr(query.takeoff, "predict", lambda f: dict(FAKE_TAKEOFF))
    monkeypatch.setattr(query.timemachine, "find_twin", lambda f: dict(FAKE_TWIN))
    monkeypatch.setattr(query, "LIVE_GITHUB_STATS", True)
    card = _run(top=3)["signals"][0]
    assert card["forward_looking_supported"] is True
    assert card["takeoff"] == FAKE_TAKEOFF and card["historical_twin"] == FAKE_TWIN


def test_abstract_sentence_is_labelled_as_quote(eligible):
    card = _run(top=3)["signals"][0]
    assert card["advantage_source"] == "abstract_quote" and card["advantage_is_original_en"] is True
    assert card["description_generated"] is False and card["case_generated"] is False


def test_fallback_advantage_is_not_labelled_as_quote(eligible, monkeypatch):
    import re
    monkeypatch.setattr(query, "ADVANTAGE", re.compile(r"(?!x)x"))
    card = _run(top=3)["signals"][0]
    assert card["advantage_source"] == "none"
    assert card["advantage_is_original_en"] is False


def test_generated_texts_are_labelled_generated(eligible, monkeypatch):
    monkeypatch.setattr(query.llm, "summarize", lambda *a, **kw: {
        "description": "Русский пересказ.", "advantage": "Пересказ преимущества.", "case": "Кейс из источников."})
    monkeypatch.setattr(query.llm, "model_name", lambda: "yandexgpt-5-lite")
    card = _run(top=3)["signals"][0]
    assert card["description_generated"] is True and card["case_generated"] is True
    assert card["advantage_source"] == "generated" and card["advantage_is_original_en"] is False
    assert card["summary_source_languages"] == ["en"]


def test_partial_generation_does_not_mislabel_template(eligible, monkeypatch):
    """Модель вернула только кейс: описание собрано по шаблону и сгенерированным не считается."""
    monkeypatch.setattr(query.llm, "summarize", lambda *a, **kw: {"case": "Кейс."})
    monkeypatch.setattr(query.llm, "model_name", lambda: "yandexgpt-5-lite")
    card = _run(top=3)["signals"][0]
    assert card["description_generated"] is False
    assert card["advantage_source"] == "abstract_quote"


# ---------- интерфейс: статические инварианты выдачи ----------

UI = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")


def test_ui_reads_the_counter_the_pipeline_actually_writes():
    """Раньше интерфейс читал ключ «с качеством доказательства ≥ 75 %», которого конвейер больше
    не пишет, и всегда показывал 0."""
    src = (ROOT / "wsignals" / "query.py").read_text(encoding="utf-8")
    assert '"с доступностью источников ≥ 75 %"' in src and '"с доступностью источников ≥ 75 %"' in UI
    assert "с качеством доказательства ≥ 75 %" not in UI


def test_ui_has_no_false_quote_or_unqualified_accuracy_claim():
    assert "Цитата из первоисточника" not in UI
    assert "точность: {" not in UI
    assert "не участвует" in UI


def test_ui_gates_forward_looking_blocks():
    assert UI.count('s.get("forward_looking_supported")') >= 2
    assert "взлёт к {tk['horizon']}" not in UI


def test_ui_renders_completeness_and_legacy_sections():
    assert 'stats.get("search_completeness")' in UI
    assert 'run.get("unverified_legacy")' in UI
