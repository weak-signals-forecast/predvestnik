"""Шлюз качества доказательства, трассируемость утверждений и честная деградация без модели
(R2-A, R2-C, R2-D).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import evidence_quality as eq  # noqa: E402
from wsignals import relevance, trust  # noqa: E402
from wsignals.decision import (ACCEPT, DETERMINISTIC_ONLY, LOW_EVIDENCE,  # noqa: E402
                               SEMANTIC_UNCERTAIN, SEMANTICALLY_VERIFIED, decide)
from wsignals.evidence import NO_RESULTS, OK, EvidenceState, Observation  # noqa: E402
from wsignals.schema import Document  # noqa: E402

YOUNG = {"oa_total": 40, "oa_growth": 1.8, "oa_age": 2, "oa_last_share": 0.45,
         "oa_preprint_share": 0.3, "oa_peak_ratio": 1.2, "oa_fields": 2, "oa_prior": 3}
CODE = {"gh_total": 30, "gh_12m": 25, "gh_new_share": 0.8}
NEWS = {"news_1y": 8, "news_30d": 3, "news_30d_share": 0.37, "news_press_share": 0.1}
HN = {"hn_total": 6, "hn_12m": 5, "hn_growth": 1.1}
NO_WIKI = {"wiki_article": 0, "wiki_age": 0}


def healthy() -> EvidenceState:
    s = EvidenceState()
    for name, values in (("openalex", YOUNG), ("github", CODE), ("news", NEWS), ("hn", HN)):
        s.add(Observation(name, OK, values))
    s.add(Observation("wiki", NO_RESULTS, NO_WIKI))
    return s


def doc(title, url, source_type, name=""):
    _d = trust.assess(Document(title=title, url=url, source_name=name, source_type=source_type,
                                 published="2026-03-01", language="en")).with_provenance()
    # P1-C. Признак претензионной релевантности читается строго: подтверждает только явное
    # True. Эта фикстура ПО ПОСТРОЕНИЮ представляет документ про рассматриваемого кандидата,
    # поэтому помечается явно. Неразмеченные документы проверяются отдельно тестом,
    # доказывающим, что они не открывают ACCEPT.
    _d.claim_relevant = True
    return _d


PAPER = doc("Agent attestation", "https://doi.org/10.1145/1", "научная публикация", "ACM")
PAPER_OTHER = doc("Attestation survey", "https://doi.org/10.1016/2", "научная публикация", "Elsevier")
REPO = doc("acme/agent-attestation", "https://github.com/acme/x", "репозиторий", "GitHub")
STANDARD = doc("Attestation profile", "https://www.iso.org/standard/1", "реестр", "ISO")
PR = doc("Acme launches", "https://www.prnewswire.com/a", "пресс-релиз", "PR Newswire")
THREAD = doc("thread", "https://news.ycombinator.com/item?id=1", "сообщество", "Hacker News")
UNKNOWN_DOC = doc("news", "https://unlisted-outlet.test/a", "новости", "Unlisted Outlet")
AGGREGATOR = doc("Acme launches", "https://aggregator.test/a", "агрегатор", "Aggregator")


def verdict(documents, verification=SEMANTICALLY_VERIFIED, **kw):
    return decide(healthy(), burst=3.0, documents=documents, verification=verification, **kw)


# ---------- R2-C: шлюз перестал быть тавтологией ----------

def test_no_documents_at_all():
    d = verdict([])
    assert d.decision == LOW_EVIDENCE and d.decision_reason_codes == ["supporting_evidence_not_materialized"]


@pytest.mark.parametrize("docs,label", [
    ([UNKNOWN_DOC], "неопознанное издание"),
    ([PR], "только пресс-релиз"),
    ([THREAD], "только сообщество"),
    ([AGGREGATOR, PR], "агрегатор и пресс-релиз"),
    ([UNKNOWN_DOC, AGGREGATOR, THREAD], "три источника без приемлемого качества"),
])
def test_documents_without_acceptable_quality_are_rejected(docs, label):
    d = verdict(docs)
    assert d.decision == LOW_EVIDENCE, label
    assert d.decision_reason_codes == ["no_acceptable_source_quality"], label
    assert d.supporting_document_count == 0


def test_two_documents_from_the_same_family_and_origin_are_not_independent():
    same_publisher = [PAPER, doc("Second", "https://doi.org/10.1145/2", "научная публикация", "ACM")]
    d = verdict(same_publisher)
    assert d.decision == LOW_EVIDENCE and d.decision_reason_codes == ["no_independent_corroboration"]


@pytest.mark.parametrize("docs,label", [
    ([PAPER, REPO], "наука и независимая реализация"),
    ([STANDARD, REPO], "реестр и техническое свидетельство"),
    ([PAPER, PAPER_OTHER], "две работы разных издателей"),
])
def test_genuinely_independent_evidence_passes(docs, label):
    d = verdict(docs)
    assert d.decision == ACCEPT, (label, d.decision_reasons)
    assert d.supporting_document_count >= 2


def test_low_quality_documents_do_not_help_a_single_good_one():
    """Один хороший документ плюс сколько угодно промо не дают независимого подтверждения."""
    d = verdict([PAPER, PR, THREAD, UNKNOWN_DOC, AGGREGATOR])
    assert d.decision == LOW_EVIDENCE
    assert d.decision_reason_codes == ["no_independent_corroboration"]
    assert d.supporting_document_count == 1


def test_evidence_gate_precedes_hype_so_promo_only_gets_a_precise_reason():
    """Отсутствие приемлемого доказательства — это причина про доказательство, а не про хайп."""
    d = verdict([AGGREGATOR, PR])
    assert d.decision == LOW_EVIDENCE
    assert d.decision_reason_codes == ["no_acceptable_source_quality"]


def test_hype_still_fires_when_evidence_exists_but_promo_dominates():
    promo = [PAPER, REPO] + [doc(f"Acme launches {i}", f"https://www.prnewswire.com/{i}", "пресс-релиз",
                                 "PR Newswire") for i in range(5)]
    d = verdict(promo)
    assert d.decision == "HYPE", d.decision_reasons
    assert d.hype_risk >= 0.6


def test_gate_codes_are_distinct_and_specific():
    codes = {tuple(verdict(d).decision_reason_codes) for d in ([], [PR], [PAPER])}
    assert codes == {("supporting_evidence_not_materialized",), ("no_acceptable_source_quality",),
                     ("no_independent_corroboration",)}


# ---------- R2-D: трассируемость утверждений ----------

def test_aggregate_and_concrete_families_are_reported_separately():
    d = verdict([PAPER, REPO])
    assert "media" in d.measured_aggregate_families          # счётчики новостей есть
    assert "media" not in d.concrete_supporting_families     # а документа из медиа нет
    assert d.evidence_quality["aggregate_only_families"] == ["media"]


def test_accept_wording_never_claims_a_family_without_a_document():
    d = verdict([PAPER, REPO])
    text = " ".join(d.decision_reasons)
    assert "подтверждено конкретными документами: 2 (code, science)" in text
    assert "активность по счётчикам без полученных документов: media" in text


def test_every_claim_with_a_family_has_at_least_one_document():
    q = eq.assess_documents([PAPER, REPO], aggregate_families=["science", "code", "media"])
    for claim in q.claims:
        if claim.aggregate_only:
            assert claim.documents == []
        else:
            assert claim.documents, claim.claim
            for ref in claim.documents:
                assert ref["url"] and ref["family"] in claim.families


def test_claim_documents_carry_an_inspectable_url_and_tier():
    q = eq.assess_documents([PAPER, REPO])
    refs = [r for c in q.claims for r in c.documents]
    assert refs and all(r["url"].startswith("http") for r in refs)
    assert all(r["trust"] in trust.SUPPORTING_TIERS for r in refs)


def test_media_document_makes_the_media_claim_concrete():
    media = doc("Хабр про attestation", "https://habr.com/ru/articles/1", "новости", "Хабр")
    d = verdict([PAPER, media])
    assert "media" in d.concrete_supporting_families
    assert "media" not in d.evidence_quality["aggregate_only_families"]


# ---------- R2-A: честная деградация без семантической нормализации ----------

def assess(phrase, seeds=frozenset({"security", "ai"})):
    return relevance.assess(phrase, set(seeds), set(), channels={"литература": 3, "репозитории": 2}, lit_docs=3)


def test_lexically_clean_off_domain_phrase_is_not_confirmed_without_semantics():
    """Главный дефект аудита: чистая по лексике, но бессмысленная фраза не должна становиться
    подтверждённым сигналом только из-за сильных счётчиков."""
    a = assess("short drama")
    assert not a.rejected                                     # высокий реколл сохранён
    assert not relevance.deterministic_high_certainty(a)      # но подтверждать её нечем
    d = verdict([PAPER, REPO], verification=DETERMINISTIC_ONLY, high_certainty=False)
    assert d.decision == LOW_EVIDENCE
    assert d.decision_reason_codes == ["semantic_verification_pending"]


def test_morphology_alone_no_longer_confers_acceptance(): 
    """R2-B: морфологический признак остаётся сигналом, но права на окончательное подтверждение
    у него нет — иначе перевёрнутые n-граммы и обрывки получали бы ACCEPT."""
    a = assess("rag poisoning")
    assert relevance.deterministic_high_certainty(a)          # признак по-прежнему вычисляется
    d = verdict([PAPER, REPO], verification=DETERMINISTIC_ONLY, high_certainty=True)
    assert d.decision == LOW_EVIDENCE
    assert d.decision_reason_codes == ["semantic_verification_pending"]
    assert "морфология указывает" in " ".join(d.decision_reasons)
    # доказательства при этом собраны и остаются видимыми
    assert d.supporting_document_count == 2


def test_ambiguous_phrase_stays_uncertain_even_with_strong_evidence():
    a = assess("key management")
    assert not a.rejected and a.ambiguous
    assert not relevance.deterministic_high_certainty(a)
    d = verdict([PAPER, REPO], verification=SEMANTIC_UNCERTAIN, high_certainty=False)
    assert d.decision == LOW_EVIDENCE
    assert "лексика кандидата неоднозначна" in " ".join(d.decision_reasons)


def test_semantic_normalization_success_confirms_without_the_deterministic_rule():
    d = verdict([PAPER, REPO], verification=SEMANTICALLY_VERIFIED, high_certainty=False)
    assert d.decision == ACCEPT and d.verification == SEMANTICALLY_VERIFIED


def test_strong_counters_alone_never_confirm():
    """Счётчики могут быть любыми: без семантики и без детерминированного признака — не подтверждено."""
    d = decide(healthy(), burst=99.0, documents=[PAPER, REPO], legacy_probability=0.99,
               verification=DETERMINISTIC_ONLY, high_certainty=False)
    assert d.decision == LOW_EVIDENCE and d.decision_reason_codes == ["semantic_verification_pending"]


def test_unverified_candidate_keeps_its_evidence_visible_for_the_watchlist():
    """Полезность при отказе модели: доказательства собраны и видны, просто статус другой."""
    d = verdict([PAPER, REPO], verification=DETERMINISTIC_ONLY, high_certainty=False)
    assert d.supporting_document_count == 2
    assert d.concrete_supporting_families == ["code", "science"]
    assert d.evidence_quality["independent_supports"] == 2


@pytest.mark.parametrize("phrase,expected", [
    ("rag poisoning", True), ("post-quantum cryptography", True), ("instruction tuning", True),
    ("quantum sensing", True), ("token compression", True),
    ("short drama", False), ("local first", False), ("model context protocol", False),
    ("agent identity", False), ("desktop app", False),
])
def test_deterministic_high_certainty_rule_is_general(phrase, expected):
    assert relevance.deterministic_high_certainty(assess(phrase)) is expected, phrase
