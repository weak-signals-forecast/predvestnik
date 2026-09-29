"""Решение по четырём осям: шлюз достаточности, вето зрелости и хайпа (PHASE 2, PHASE 3, PHASE 8)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals.decision import (ACCEPT, HYPE, LOW_EVIDENCE, MATURE, MIN_CORROBORATING_FAMILIES,  # noqa: E402
                               NOISE, SEMANTICALLY_VERIFIED, decide as _decide)


def decide(state, **kw):
    """R2-A: ACCEPT требует состояния семантической проверки. Тесты, проверяющие оси решения
    и вето, объявляют проверку выполненной явно; отдельные тесты проверяют сам этот шлюз."""
    kw.setdefault("verification", SEMANTICALLY_VERIFIED)
    return _decide(state, **kw)
from wsignals.evidence import ERROR, NO_RESULTS, OK, EvidenceState, Observation  # noqa: E402

YOUNG_SCIENCE = {"oa_total": 40, "oa_growth": 1.8, "oa_age": 2, "oa_last_share": 0.45,
                 "oa_preprint_share": 0.3, "oa_peak_ratio": 1.2}
FRESH_CODE = {"gh_total": 30, "gh_12m": 25, "gh_new_share": 0.8}
CLEAN_NEWS = {"news_1y": 8, "news_30d": 3, "news_30d_share": 0.37, "news_press_share": 0.1}
COMMUNITY = {"hn_total": 6, "hn_12m": 5, "hn_growth": 1.1}
NO_WIKI = {"wiki_article": 0, "wiki_age": 0}

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



def state(**kw) -> EvidenceState:
    s = EvidenceState()
    for name, (status, values) in kw.items():
        s.add(Observation(name, status, values or {}, error_type="timeout" if status == ERROR else None))
    return s


def healthy(**override) -> EvidenceState:
    base = {"openalex": (OK, YOUNG_SCIENCE), "github": (OK, FRESH_CODE), "news": (OK, CLEAN_NEWS),
            "hn": (OK, COMMUNITY), "wiki": (NO_RESULTS, NO_WIKI)}
    base.update(override)
    return state(**base)


# ---------- четыре оси разделены и присутствуют всегда ----------

def test_decision_returns_all_four_axes_separately():
    d = decide(healthy(), burst=3.0, legacy_probability=0.61, documents=DOCS).to_dict()
    for key in ("emergence_score", "maturity_risk", "evidence_confidence", "hype_risk",
                "decision", "decision_reasons", "legacy_model_probability"):
        assert key in d, key
    assert d["decision"] == ACCEPT
    # Вероятность модели этапа 1 отдана отдельным полем и не смешана с решением.
    assert d["legacy_model_probability"] == 0.61


def test_legacy_probability_alone_cannot_accept_or_reject():
    weak_model = decide(healthy(), burst=3.0, legacy_probability=0.01, documents=DOCS)
    strong_model = decide(healthy(), burst=3.0, legacy_probability=0.99, documents=DOCS)
    assert weak_model.decision == strong_model.decision == ACCEPT
    lonely = state(hn=(OK, COMMUNITY))
    assert decide(lonely, legacy_probability=0.99, documents=DOCS).decision == LOW_EVIDENCE


# ---------- шлюз достаточности доказательства ----------

def test_single_family_is_low_evidence():
    d = decide(state(openalex=(OK, YOUNG_SCIENCE), github=(NO_RESULTS, {"gh_total": 0}),
                     news=(NO_RESULTS, {"news_1y": 0}), hn=(NO_RESULTS, {"hn_total": 0}),
                     wiki=(NO_RESULTS, NO_WIKI)), burst=3.0)
    assert d.decision == LOW_EVIDENCE
    assert f"из необходимых {MIN_CORROBORATING_FAMILIES}" in " ".join(d.decision_reasons)


def test_lead_only_evidence_is_rejected():
    """Класс D (сообщества, ранний отклик) сам по себе подтверждением не является."""
    d = decide(state(openalex=(NO_RESULTS, {"oa_total": 0}), github=(NO_RESULTS, {"gh_total": 0}),
                     news=(NO_RESULTS, {"news_1y": 0}), hn=(OK, COMMUNITY), wiki=(NO_RESULTS, NO_WIKI)),
               burst=5.0, legacy_probability=0.9)
    assert d.decision == LOW_EVIDENCE
    assert "первичный индикатор" in " ".join(d.decision_reasons)


def test_community_plus_one_family_still_needs_a_second_corroborating_family():
    d = decide(state(openalex=(OK, YOUNG_SCIENCE), github=(NO_RESULTS, {"gh_total": 0}),
                     news=(NO_RESULTS, {"news_1y": 0}), hn=(OK, COMMUNITY), wiki=(NO_RESULTS, NO_WIKI)), burst=3.0)
    assert d.decision == LOW_EVIDENCE


def test_two_corroborating_families_pass_the_gate():
    d = decide(state(openalex=(OK, YOUNG_SCIENCE), github=(OK, FRESH_CODE),
                     news=(NO_RESULTS, {"news_1y": 0}), hn=(NO_RESULTS, {"hn_total": 0}),
                     wiki=(NO_RESULTS, NO_WIKI)), burst=3.0, documents=DOCS)
    assert d.decision == ACCEPT and d.evidence_confidence >= 0.45


def test_evidence_confidence_is_reported_outside_the_model_probability():
    d = decide(healthy(), burst=3.0, legacy_probability=0.5, documents=DOCS)
    assert 0.0 <= d.evidence_confidence <= 1.0
    assert d.evidence_confidence != d.legacy_model_probability


# ---------- вето зрелости ----------

def test_mature_veto_on_old_wikipedia_article():
    d = decide(healthy(wiki=(OK, {"wiki_article": 1, "wiki_age": 11})), burst=3.0, legacy_probability=0.95)
    assert d.decision == MATURE and d.maturity_risk >= 0.6
    assert "Википедии" in " ".join(d.decision_reasons)


def test_mature_veto_on_large_old_literature():
    d = decide(healthy(openalex=(OK, {**YOUNG_SCIENCE, "oa_total": 9000, "oa_age": 12})), burst=3.0)
    assert d.decision == MATURE


# ---------- вето хайпа ----------

def test_hype_veto_on_press_release_share():
    # Доказательства есть: тест про ось хайпа, а не про их отсутствие (шлюз доказательства идёт раньше).
    d = decide(healthy(news=(OK, {**CLEAN_NEWS, "news_press_share": 0.8})), burst=3.0,
               legacy_probability=0.95, documents=DOCS)
    assert d.decision == HYPE and d.hype_risk >= 0.6
    assert "пресс-релиз" in " ".join(d.decision_reasons)


def test_hype_veto_on_media_without_substance():
    d = decide(state(openalex=(OK, {"oa_total": 4, "oa_growth": 2.0, "oa_age": 1, "oa_last_share": 0.9,
                                    "oa_preprint_share": 0.5, "oa_peak_ratio": 1.5}),
                     github=(OK, {"gh_total": 2, "gh_12m": 2, "gh_new_share": 1.0}),
                     news=(OK, {"news_1y": 80, "news_30d": 30, "news_30d_share": 0.37, "news_press_share": 0.2}),
                     hn=(OK, COMMUNITY), wiki=(NO_RESULTS, NO_WIKI)), burst=4.0, documents=DOCS)
    assert d.decision == HYPE


def test_faded_wave_is_hype_not_a_weak_signal():
    d = decide(healthy(openalex=(OK, {"oa_total": 400, "oa_growth": -0.4, "oa_age": 7, "oa_last_share": 0.05,
                                      "oa_preprint_share": 0.1, "oa_peak_ratio": 0.2, "oa_peak_year": 2021})),
               burst=0.2, documents=DOCS)
    assert d.decision in (HYPE, MATURE)


# ---------- шум ----------

def test_noise_reason_short_circuits_everything():
    d = decide(healthy(), burst=3.0, noise_reason="название продукта или компании, а не технологии",
               legacy_probability=0.99, documents=DOCS)
    assert d.decision == NOISE and d.decision_reasons == ["название продукта или компании, а не технологии"]


def test_no_measurable_emergence_is_noise():
    flat = {"oa_total": 900, "oa_growth": 0.0, "oa_age": 9, "oa_last_share": 0.05,
            "oa_preprint_share": 0.0, "oa_peak_ratio": 1.0}
    d = decide(state(openalex=(OK, flat), github=(OK, {"gh_total": 5, "gh_12m": 0, "gh_new_share": 0.0}),
                     news=(OK, {"news_1y": 3, "news_30d": 0, "news_30d_share": 0.0, "news_press_share": 0.0}),
                     hn=(NO_RESULTS, {"hn_total": 0}), wiki=(NO_RESULTS, NO_WIKI)), burst=0.0, documents=DOCS)
    assert d.decision in (NOISE, MATURE)


# ---------- порядок выдачи ----------

def test_rank_score_prefers_better_evidence_at_equal_emergence():
    strong = decide(healthy(), burst=3.0, documents=DOCS)
    thin = decide(state(openalex=(OK, YOUNG_SCIENCE), github=(OK, FRESH_CODE),
                        news=(NO_RESULTS, {"news_1y": 0}), hn=(ERROR, None), wiki=(NO_RESULTS, NO_WIKI)),
                  burst=3.0, documents=DOCS)
    assert strong.rank_score() > thin.rank_score()


# ---------- REPAIR 1: ACCEPT требует конкретного доказательства ----------

def test_accept_requires_at_least_one_concrete_ab_document():
    d = decide(healthy(), burst=3.0, legacy_probability=0.9, documents=[])
    assert d.decision == LOW_EVIDENCE
    assert d.decision_reason_codes == ["supporting_evidence_not_materialized"]
    assert d.supporting_document_count == 0


def test_press_release_and_community_documents_do_not_count_as_support():
    weak_docs = [_doc("PR", "https://www.prnewswire.com/x", "пресс-релиз", "PR Newswire"),
                 _doc("HN thread", "https://news.ycombinator.com/item?id=1", "сообщество", "Hacker News")]
    d = decide(healthy(), burst=3.0, documents=weak_docs)
    assert d.decision == LOW_EVIDENCE
    # Документы получены, но ни один не является первичным доказательством приемлемого качества.
    assert d.decision_reason_codes == ["no_acceptable_source_quality"]


def test_one_document_is_no_longer_enough_without_independent_corroboration():
    """R2-C: агрегатные счётчики больше не заменяют второе независимое конкретное свидетельство."""
    d = decide(healthy(), burst=3.0, documents=[PAPER])
    assert d.decision == LOW_EVIDENCE
    assert d.decision_reason_codes == ["no_independent_corroboration"]
    assert d.supporting_document_count == 1 and d.concrete_supporting_families == ["science"]


def test_accept_reports_which_families_the_concrete_documents_cover():
    d = decide(healthy(), burst=3.0, documents=DOCS)
    assert d.supporting_document_count == 2
    assert d.supporting_document_families == ["code", "science"]
    assert "accepted" in d.decision_reason_codes


def test_single_family_everywhere_is_not_independent_corroboration():
    """Одно семейство и в агрегате, и в документах: подтверждения нет даже при конкретном документе."""
    only_science = state(openalex=(OK, YOUNG_SCIENCE), github=(NO_RESULTS, {"gh_total": 0}),
                         news=(NO_RESULTS, {"news_1y": 0}), hn=(NO_RESULTS, {"hn_total": 0}),
                         wiki=(NO_RESULTS, NO_WIKI))
    d = decide(only_science, burst=3.0, documents=[PAPER])
    assert d.decision == LOW_EVIDENCE
    assert d.decision_reason_codes in (["insufficient_corroborating_families"], ["no_independent_corroboration"])


def test_every_rejection_carries_a_machine_readable_code():
    for d in (decide(healthy(), burst=3.0, documents=[]),
              decide(healthy(wiki=(OK, {"wiki_article": 1, "wiki_age": 11})), burst=3.0, documents=DOCS),
              decide(healthy(news=(OK, {**CLEAN_NEWS, "news_press_share": 0.8})), burst=3.0, documents=DOCS),
              decide(healthy(), burst=3.0, noise_reason="продукт", documents=DOCS)):
        assert d.decision_reason_codes and all(isinstance(c, str) for c in d.decision_reason_codes)
