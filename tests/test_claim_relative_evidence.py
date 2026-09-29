"""Претензионно-относительная релевантность документа и таксономия статусов новостного корпуса.

Сеть не используется. Контрпримеры взяты из ТРЁХ РЕАЛЬНЫХ холодных прогонов владельца на d948:
заголовки документов воспроизведены дословно, чтобы тест ловил ровно тот дефект, который был измерен.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import claim_relevance as cr  # noqa: E402
from wsignals import evidence_quality as eq  # noqa: E402
from wsignals import http, query, relevance, trust  # noqa: E402
from wsignals.decision import ACCEPT, LOW_EVIDENCE, SEMANTICALLY_VERIFIED, decide  # noqa: E402
from wsignals.evidence import NO_RESULTS, OK, EvidenceState, Observation  # noqa: E402
from wsignals.schema import Document  # noqa: E402


def doc(title, url, source_type, source_name="Источник", snippet=""):
    d = Document(title=title, url=url, source_name=source_name, source_type=source_type,
                 published="2026-06-01", language="en", snippet=snippet)
    return trust.assess(d.with_provenance(retrieved_at="2026-09-22T00:00:00+00:00"))


def cand(phrase, merged_from=(), canonical=None):
    return {"phrase": phrase, "original_phrase": phrase,
            "canonical_label": canonical or phrase, "merged_from": list(merged_from)}


def state(news=OK):
    s = EvidenceState()
    s.add(Observation("openalex", OK, {"oa_total": 63, "oa_growth": 1.9, "oa_last_share": 0.45,
                                       "oa_age": 2, "oa_fields": 2, "oa_prior": 3,
                                       "oa_preprint_share": 0.3}))
    s.add(Observation("github", OK, {"gh_corpus_repos": 2}))
    s.add(Observation("wiki", NO_RESULTS, {}))
    s.add(Observation("news", news, {"news_1y": 12, "news_30d": 4, "news_30d_share": 0.3,
                                     "news_press_share": 0.15, "news_sources": 4}))
    return s


def verdict(candidate, documents):
    return decide(state(), burst=1.2, documents=documents, verification=SEMANTICALLY_VERIFIED,
                  legacy_probability=0.76, candidate=candidate)


# ============ документы из живых прогонов ============

AI_AGENTS_PAPER = doc("AI Agents and Agentic AI–navigating a plethora of concepts for future manufacturing",
                      "https://doi.org/10.1016/j.jmsy.2025.08.017", "научная публикация", "J Manuf Syst")
AI_LITERACY_REVIEW = doc("Artificial intelligence literacy education in primary schools: a review",
                         "https://doi.org/10.1000/aild.2025.01", "научная публикация", "Education Review")
CHAI_REPO = doc("NIHAR-SARKAR/CHAI: Cyber Host Artificial Intelligence (C.H.A.I) - mcp server",
                "https://github.com/NIHAR-SARKAR/CHAI", "репозиторий", "GitHub")
BROSEIDON_REPO = doc("interamps/broseidon: Cutting edge cloud AI technology put into a Bleeding blunt "
                     "decade old server", "https://github.com/interamps/broseidon", "репозиторий", "GitHub")

KYC_REPO = doc("vyayasan/kyc-analyst: Open-source KYC/AML compliance automation. 17 human-in-the-loop "
               "checkpoints", "https://github.com/vyayasan/kyc-analyst", "репозиторий", "GitHub")
KYC_LIST_REPO = doc("FinAuth-SDK/awesome-identity-verification: A curated list of identity verification, "
                    "KYC, AML and biometric authentication resources",
                    "https://github.com/FinAuth-SDK/awesome-identity-verification", "репозиторий", "GitHub")
CBDC_THESIS = doc("An Ethereum-Based Blockchain Infrastructure for a Retail CBDC: A Demonstration Aligned "
                  "With ECB Requirements", "https://doi.org/10.1000/cbdc", "научная публикация", "Ledger")
FRAUD_PAPER = doc("Multi-Agent AI Systems for Secure, Transparent, and Compliant Fraud Surveillance in "
                  "Cross-Border Payments", "https://doi.org/10.1000/fraud", "научная публикация", "J Fin Tech",
                  snippet="the heterogeneous regulatory environments, varying kyc/aml standards, and speed "
                          "of digital finance innovation complicate oversight")
ROMANIA_PREPRINT = doc("Governing Financial Innovation Through Institutional Learning: Lessons from "
                       "Romania's Fintech Innovation Hub", "https://doi.org/10.1000/ro", "препринт", "SSRN")

SOIL_PAPER = doc("Soil moisture retrieval from Sentinel-1 over Bavarian grassland",
                 "https://doi.org/10.1000/soil", "научная публикация", "Remote Sensing of Environment")
TODO_REPO = doc("alice/todo-list: my first todo app", "https://github.com/alice/todo-list",
                "репозиторий", "GitHub")


# ============ 1. обязательные контрпримеры правила ============

@pytest.mark.parametrize("document,expected", [
    (AI_AGENTS_PAPER, True),          # работа НАЗВАНА про агентов
    (AI_LITERACY_REVIEW, False),      # совпадает только общее «artificial intelligence»
    (BROSEIDON_REPO, False),          # общий репозиторий «cloud AI», к агентам отношения не имеет
    (CHAI_REPO, False),
    (SOIL_PAPER, False),
    (TODO_REPO, False),
])
def test_improved_ai_agents_counterexamples(document, expected):
    """Контракт V3 — фразовый: заголовок обязан содержать САМО название кандидата.

    «improved ai agents» не встречается ни в одном из этих заголовков, поэтому подтверждения нет
    даже у работы про AI-агентов: точность предпочтена полноте, LOW_EVIDENCE — приемлемый исход."""
    ok, _name, reason = cr.match(document, cand("improved ai agents"))
    assert ok is False, (document.title, reason)


@pytest.mark.parametrize("document,expected", [
    (KYC_REPO, True),
    (KYC_LIST_REPO, True),
    (CBDC_THESIS, False),             # тезис про розничный CBDC: KYC/AML в названии нет
    (FRAUD_PAPER, False),             # «kyc/aml» есть только в аннотации, одним упоминанием
    (ROMANIA_PREPRINT, False),
])
def test_kyc_aml_counterexamples(document, expected):
    ok, _name, reason = cr.match(document, cand("kyc aml", merged_from=["aml kyc"]))
    assert ok is expected, (document.title, reason)


def test_mention_in_the_abstract_is_not_aboutness():
    """Упоминание технологии в аннотации подтверждением не является: измерено на живом прогоне."""
    assert "kyc/aml" in FRAUD_PAPER.snippet.lower()
    ok, _n, _r = cr.match(FRAUD_PAPER, cand("kyc aml"))
    assert ok is False


def test_absorbed_alias_is_not_authoritative_without_proven_equivalence():
    """Правило B: попадание в merged_from равнозначности НЕ означает.

    «mcp server» и «model context protocol» — непересекающиеся множества токенов. Сокращение может
    стать авторитетным только по ЯВНОМУ происхождению аббревиатуры, установленному до сопоставления;
    такого источника в системе нет, поэтому псевдоним остаётся происхождением поиска."""
    c = cand("model context protocol", merged_from=["mcp server"])
    ok, _name, reason = cr.match(doc("someorg/mcp-server: reference MCP server implementation",
                                     "https://github.com/someorg/mcp-server", "репозиторий"), c)
    assert ok is False, reason
    assert cr.ALIAS_NOT_PROVEN_EQUIVALENT in [st for _, _, st in cr.identity_report(c) if st]


def test_permutation_alias_is_authoritative():
    """Разрешённая равнозначность: то же мультимножество токенов при ином порядке."""
    c = cand("kyc aml", merged_from=["aml kyc"])
    assert cr.equivalent("aml kyc", "kyc aml") is True
    assert cr.match(doc("AML KYC automation for banks", "https://x/1", "научная публикация"), c)[0] is True


def test_generic_identity_is_matched_as_its_own_phrase():
    """Общее название берётся КАК ЕСТЬ: никакого отсева и никакой лесенки."""
    assert cr.identity_tokens("artificial intelligence") == (["artificial", "intelligence"], None)
    assert cr.match(AI_LITERACY_REVIEW, cand("artificial intelligence"))[0] is True
    assert cr.match(SOIL_PAPER, cand("artificial intelligence"))[0] is False


def test_unknown_relevance_is_not_positive_support():
    """Fail-closed: нет личности кандидата или нет заголовка — подтверждения нет."""
    assert cr.match(AI_AGENTS_PAPER, None)[0] is False
    assert cr.match(AI_AGENTS_PAPER, cand(""))[0] is False
    assert cr.match(doc("", "https://x/1", "научная публикация"), cand("improved ai agents"))[0] is False
    assert cr.is_relevant(doc("x", "https://x/2", "научная публикация")) is False   # не размечен


# ============ 2. ложное подтверждение не открывает ACCEPT ============

SYNTHETIC_FALSE = [SOIL_PAPER, TODO_REPO]


def test_false_support_cannot_satisfy_the_family_gate():
    d = verdict(cand("improved ai agents"), SYNTHETIC_FALSE)
    assert d.supporting_document_count == 0
    assert d.concrete_supporting_families == []
    assert d.decision_reason_codes == ["no_claim_relevant_evidence"]


def test_false_support_cannot_enable_accept():
    """Ровно та связка, что воспроизводила ложный ACCEPT до ремонта: наука + код, но не про это."""
    assert verdict(cand("improved ai agents"), SYNTHETIC_FALSE).decision != ACCEPT
    assert verdict(cand("improved ai agents"),
                   [SOIL_PAPER, doc("A review of 14th-century Byzantine liturgical manuscripts",
                                    "https://doi.org/10.1000/byz", "научная публикация", "J Med Studies"),
                    TODO_REPO, doc("bob/dotfiles: personal shell config",
                                   "https://github.com/bob/dotfiles", "репозиторий")]).decision != ACCEPT


def test_false_support_cannot_raise_evidence_confidence():
    """Качество доказательства считается по агрегатным семействам и от документов не растёт."""
    empty = decide(state(), burst=1.2, documents=[], verification=SEMANTICALLY_VERIFIED,
                   candidate=cand("improved ai agents"))
    false_support = verdict(cand("improved ai agents"), SYNTHETIC_FALSE)
    assert false_support.evidence_confidence <= empty.evidence_confidence


def test_live_ai_candidate_is_no_longer_corroborated_by_unrelated_documents():
    """Живой кандидат d948: из четырёх «подтверждений» правилу отвечает только одно."""
    # Контракт V3 фразовый: подтверждает только САМО название кандидата. Живая фраза,
    # которая действительно стоит в заголовке, — «ai agents»; «improved ai agents» не
    # подтверждается ничем и проверяется отдельным контрпримером выше.
    d = verdict(cand("ai agents"),
                [AI_AGENTS_PAPER, AI_LITERACY_REVIEW, CHAI_REPO, BROSEIDON_REPO])
    assert d.supporting_document_count == 1
    assert d.decision != ACCEPT
    assert d.evidence_quality["claim_irrelevant_document_count"] == 3


def test_relevant_evidence_still_accepts():
    """Ремонт не должен закрывать допуск вообще: настоящее подтверждение продолжает работать."""
    d = verdict(cand("lattice coupler"),
                [doc("Federated learning for cryogenic lattice couplers", "https://doi.org/10.1/f1",
                     "научная публикация", "Nature"),
                 doc("someorg/lattice-coupler: cryogenic lattice coupler toolkit",
                     "https://github.com/someorg/lattice-coupler", "репозиторий")])
    assert d.decision == ACCEPT, d.decision_reasons
    assert d.concrete_supporting_families == ["code", "science"]


# ============ 3. пересчёт после склейки и сохранность происхождения ============

def _merged_pair():
    """Живой случай d948: «improved ai agents» поглотил более широкий «ai agent» по ключу модели.

    Широкий вариант принёс с собой обзор преподавания ИИ в начальной школе — документ, который на
    живом прогоне засчитался кандидату в подтверждение."""
    llm = {"source": "llm", "is_technology": True, "domain_relevance": 0.9}
    lead = {"phrase": "ai agents", "original_phrase": "ai agents",
            "canonical_label": "ai agents", "merge_key": "agent", "assessment": dict(llm),
            "burst": 9.0, "channels": {}, "recent_docs": 1, "prior_docs": 0,
            "examples": [{"id": "W1", "doi": "https://doi.org/10.1016/j.jmsy.2025.08.017",
                          "title": "AI Agents and Agentic AI–navigating a plethora of concepts",
                          "type": "article", "publication_date": "2026-05-01", "language": "en",
                          "primary_location": {"source": {"display_name": "J Manuf Syst"}}}],
            "repos": []}
    broad = {"phrase": "ai agent", "original_phrase": "ai agent", "canonical_label": "ai agent",
             "merge_key": "agent", "assessment": dict(llm),
             "burst": 5.0, "channels": {}, "recent_docs": 1, "prior_docs": 0,
             "examples": [{"id": "W9", "doi": "https://doi.org/10.1000/aild.2025.01",
                           "title": "Artificial intelligence literacy education in primary schools: a review",
                           "type": "article", "publication_date": "2026-05-01", "language": "en",
                           "primary_location": {"source": {"display_name": "Education Review"}}}],
             "repos": []}
    return lead, broad


def test_relevance_is_recomputed_against_the_surviving_identity_after_collapse():
    """Документ, пришедший с ПОГЛОЩЁННЫМ широким вариантом, не наследует подтверждение."""
    lead_in, broad = _merged_pair()
    kept, merged = relevance.collapse([lead_in, broad])
    lead = kept[0]
    assert lead["phrase"] == "ai agents"
    assert "ai agent" in (lead.get("merged_from") or []), lead.get("merged_from")
    assert len(lead["examples"]) == 2, "склейка обязана сохранить происхождение обоих вариантов"

    docs = query.concrete_documents(lead, "2026-09-22T00:00:00+00:00")
    assert len(docs) == 2, "документ поглощённого варианта обязан остаться видимым"
    by_title = {d.title: d for d in docs}
    own = next(d for t, d in by_title.items() if "AI Agents" in t)
    inherited = next(d for t, d in by_title.items() if "literacy" in t)
    assert own.claim_relevant is True
    assert inherited.claim_relevant is False, "подтверждение не наследуется от широкого варианта"

    q = eq.assess_documents(docs, candidate=lead)
    assert [d.title for d in q.supporting] == [own.title]
    assert q.independent_supports == 1


def test_absorbed_variant_cannot_complete_the_family_gate():
    """Ровно тот эффект, что открывал ложный ACCEPT: чужой документ добирал второе семейство."""
    lead_in, broad = _merged_pair()
    broad["repos"] = [{"full_name": "interamps/broseidon",
                       "description": "Cutting edge cloud AI technology put into a Bleeding blunt server",
                       "html_url": "https://github.com/interamps/broseidon",
                       "created_at": "2026-06-01T00:00:00Z", "stargazers_count": 3}]
    lead = relevance.collapse([lead_in, broad])[0][0]
    d = verdict(lead, query.concrete_documents(lead, "2026-09-22T00:00:00+00:00"))
    assert d.concrete_supporting_families == ["science"], d.concrete_supporting_families
    assert d.decision != ACCEPT
    assert d.decision_reason_codes == ["no_independent_corroboration"]


def test_irrelevant_documents_stay_visible_as_retrieved_evidence():
    """Документ не удаляется из происхождения: он получен, показан и объяснён, но не засчитан."""
    d = verdict(cand("ai agents"), [AI_AGENTS_PAPER, AI_LITERACY_REVIEW])
    q = d.evidence_quality
    assert q["materialized_document_count"] == 2
    assert q["claim_irrelevant_document_count"] == 1
    ref = q["claim_irrelevant_documents"][0]
    assert ref["title"].startswith("Artificial intelligence literacy")
    assert ref["claim_relevant"] is False and ref["claim_relevance_reason"]


def test_supporting_documents_carry_the_matched_name():
    docs = [AI_AGENTS_PAPER]
    eq.assess_documents(docs, candidate=cand("ai agents", merged_from=["ai agent"]))
    assert docs[0].claim_relevant is True
    assert docs[0].claim_match and docs[0].claim_relevance_reason


def test_claims_never_cite_an_unrelated_document():
    q = eq.assess_documents([AI_AGENTS_PAPER, AI_LITERACY_REVIEW, TODO_REPO],
                            aggregate_families=["science", "code"],
                            candidate=cand("ai agents"))
    cited = {d["title"] for claim in q.claims for d in claim.documents}
    assert all("literacy" not in t and "todo" not in t for t in cited), cited


def test_production_materialization_always_marks_relevance():
    """На живом пути неразмеченных документов не бывает: разметка стоит в точке материализации."""
    c = {"phrase": "lattice coupler", "original_phrase": "lattice coupler",
         "canonical_label": "lattice coupler", "merged_from": [],
         "examples": [{"id": "W1", "doi": "https://doi.org/10.1/a", "title": "Lattice coupler calibration",
                       "type": "article", "publication_date": "2026-05-01", "language": "en",
                       "primary_location": {"source": {"display_name": "Nature"}}},
                      {"id": "W2", "doi": "https://doi.org/10.1/b", "title": "Soil moisture over grassland",
                       "type": "article", "publication_date": "2026-05-01", "language": "en",
                       "primary_location": {"source": {"display_name": "RSE"}}}],
         "repos": [{"full_name": "org/lattice-coupler", "description": "toolkit",
                    "html_url": "https://github.com/org/lattice-coupler",
                    "created_at": "2026-06-01T00:00:00Z", "stargazers_count": 5}]}
    docs = query.concrete_documents(c, "2026-09-22T00:00:00+00:00")
    assert len(docs) == 3
    assert all(d.claim_relevant is not None for d in docs)
    assert sum(d.claim_relevant for d in docs) == 2          # почва не про соединитель


def test_same_work_merge_regression_is_untouched():
    """Склейка версий одной работы (P1-A) продолжает считаться одним свидетельством."""
    a = doc("Lattice coupler calibration", "https://doi.org/10.1/x", "препринт", "arXiv")
    b = doc("Lattice coupler calibration", "https://doi.org/10.1/x", "научная публикация", "Nature")
    q = eq.assess_documents([a, b], candidate=cand("lattice coupler"))
    assert q.independent_supports == 1, [d.title for d in q.supporting]


# ============ 4. таксономия статусов новостного корпуса ============

def _seeds():
    return ["alpha technology", "beta technology"]


def _titles_with(monkeypatch, behaviour):
    monkeypatch.setattr(query.news, "search", behaviour)
    return query._news_titles(_seeds())


def test_news_corpus_reports_ok_and_no_results_only_for_a_parsed_feed(monkeypatch):
    titles, errors = _titles_with(monkeypatch, lambda *a, **kw: [
        Document(title="Alpha raises seed round", url="https://x/1", source_name="Reuters",
                 source_type="новости", published="2026-08-01", language="en")])
    assert titles and errors == []

    titles, errors = _titles_with(monkeypatch, lambda *a, **kw: [])
    assert titles == [] and errors == [], "разобранная пустая лента это ответ источника"


@pytest.mark.parametrize("failure,mark", [
    (http.FetchError("HTTP 500", http.HTTP_ERROR, 500), "http_error"),
    (http.FetchError("HTTP 429", http.RATE_LIMIT, 429), "rate_limit"),
    (http.FetchError("connection reset", http.NETWORK), "network"),
    (http.FetchError("исчерпан общий бюджет запроса", http.DEADLINE), "deadline"),
    (ET.ParseError("syntax error: line 1, column 0"), "parse"),
])
def test_news_corpus_failures_are_never_reported_as_no_results(monkeypatch, failure, mark):
    """HTTP, 429, сеть, дедлайн и страница вместо RSS — это ERROR, а не «источник ответил ноль»."""
    def boom(*_a, **_kw):
        raise failure

    titles, errors = _titles_with(monkeypatch, boom)
    assert titles == []
    assert errors and all(mark in e for e in errors), errors


def test_news_corpus_status_mapping(monkeypatch):
    """Статус адаптера собирается ровно по правилу PHASE 1: отказ никогда не становится NO_RESULTS."""
    def status(headlines, news_errors):
        # То же выражение, что в extract_candidates, проверенное отдельно от сетевого сбора.
        if news_errors:
            return "ERROR" if not headlines else "PARTIAL"
        return "NO_RESULTS" if not headlines else "OK"

    assert status(["t"], []) == "OK"
    assert status([], []) == "NO_RESULTS"
    assert status([], ["q: parse"]) == "ERROR"
    assert status([], ["q: deadline"]) == "ERROR"
    assert status(["t"], ["q: rate_limit"]) == "PARTIAL"


def test_html_consent_page_is_a_parse_failure_not_an_empty_feed(monkeypatch):
    """Страница согласия или блокировки вместо ленты разбирается как отказ, а не как ноль новостей."""
    from wsignals.sources import news as news_mod

    monkeypatch.setattr(news_mod, "get", lambda *a, **kw: "<!doctype html><html><body>Before you "
                                                          "continue to Google</body></html>")
    with pytest.raises(ET.ParseError):
        news_mod._items("alpha", "1y", "en")


def test_compound_spelling_of_a_multiword_name_is_the_same_name():
    """«vibecoding» это «vibe coding» без пробела: измерено на живом прогоне (AI, «vibe coding»)."""
    repo = doc("bobvibes/Awesome-Vibecoding-Guide: A compendium drawn from real commercial projects",
               "https://github.com/bobvibes/Awesome-Vibecoding-Guide", "репозиторий")
    assert cr.match(repo, cand("vibe coding"))[0] is True


def test_compound_rule_never_applies_to_a_single_word_name():
    """Иначе правило выродилось бы в поиск подстроки: «bot» нашёлся бы в «robot»."""
    robot = doc("acme/robotics-kit: industrial robot arm controller",
                "https://github.com/acme/robotics-kit", "репозиторий")
    assert cr.match(robot, cand("bot"))[0] is False
    assert cr.match(robot, cand("trading bot"))[0] is False
