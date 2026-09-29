"""Правда идентичности: через какую личность документ вообще может подтвердить технологию.

Четыре измеренных дефекта предыдущего claim-relative ремонта:

  P1-A личность строилась из МНОЖЕСТВА токенов: сначала сжималась до одного широкого слова
       («ai security» → «security»), затем лесенка V2 возвращала контекст — и порождала ложное
       подтверждение «общее слово × существительное» («Food security … with AI»). Обе механики
       заменены ФРАЗОВЫМ контрактом V3;
  P1-B поглощённый склейкой псевдоним наследовал авторитет без доказанной равнозначности;
  P1-C неразмеченный документ (`claim_relevant is None`) засчитывался в подтверждение, хотя
       отпечаток конфигурации объявлял обратное;
  P1-D токенизатор [a-z0-9]+ делал кириллический заголовок пустым и сообщал неправду «нет заголовка».
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import claim_relevance as CR, evidence_quality as EQ, provenance, relevance, trust  # noqa: E402
from wsignals.decision import ACCEPT, SEMANTICALLY_VERIFIED, decide  # noqa: E402
from wsignals.evidence import NO_RESULTS, OK, EvidenceState, Observation  # noqa: E402
from wsignals.schema import Document  # noqa: E402


def doc(title, url="https://x.test/a", source_type="научная публикация", name="J"):
    return trust.assess(Document(title=title, url=url, source_name=name, source_type=source_type,
                                 published="2026-01-01", language="en")).with_provenance()


# ---------- A/C: фразовый контракт ----------

FALSE_SUPPORT_PROBES = [
    ("ai security", "Food security forecasting with AI across sub-Saharan farms"),
    ("quantum computing", "Quantum chemistry workloads on exascale high performance computing clusters"),
    ("model compression", "Lossy compression of global climate model output using wavelet transforms"),
    ("governance digital", "Public Policy and Governance in The Digital Economy of India"),
    ("edge caching", "Cutting-edge caching strategies for PostgreSQL"),
    ("intelligent ledger", "Digital ledger portfolio management overview"),
    ("improved ai agents", "Agent-based modelling of malaria transmission"),
    ("ai security", "acme/home-cctv-security: camera recorder"),
]


@pytest.mark.parametrize("candidate,title", FALSE_SUPPORT_PROBES)
def test_tokens_scattered_through_a_title_are_not_support(candidate, title):
    """Все слова где-то в заголовке — это не «заголовок называет технологию»."""
    ok, name, reason = CR.match(doc(title), candidate)
    assert ok is False, (candidate, title, name, reason)


@pytest.mark.parametrize("candidate,title", [
    ("ai security", "AI security posture management for cloud workloads"),
    ("quantum computing", "Quantum computing for combinatorial optimisation"),
    ("model compression", "Model compression via structured pruning"),
    ("edge caching", "Edge caching policies for video delivery"),
    ("multi-agent orchestration", "Multi agent orchestration for large language models"),
    ("federated learning", "Federated learnings in hospital networks"),
    ("rag poisoning", "RAG poisoning: attacks on retrieval pipelines"),
])
def test_phrase_named_in_the_title_supports(candidate, title):
    ok, name, reason = CR.match(doc(title), candidate)
    assert ok is True, (candidate, title, reason)


def test_hyphen_compound_does_not_open_a_phrase_boundary():
    assert CR.match(doc("Cutting-edge caching strategies"), "edge caching")[0] is False
    assert CR.match(doc("Edge caching strategies"), "edge caching")[0] is True


def test_identity_is_taken_as_written_without_vocabulary_filtering():
    assert CR.identity_tokens("ai security") == (["ai", "security"], None)
    assert CR.identity_tokens("artificial intelligence") == (["artificial", "intelligence"], None)
    assert CR.identity_tokens("the of and")[1] == CR.INSUFFICIENT_IDENTITY_SPECIFICITY


def test_identity_ladder_is_gone():
    for removed in ("meaningful_tokens", "alias_is_safe", "EDITORIAL", "GENERIC_TECH",
                    "GENERIC_DOMAIN", "IGNORED"):
        assert not hasattr(CR, removed), removed


def test_no_phrase_specific_blacklist_in_the_rule():
    import ast

    tree = ast.parse((ROOT / "wsignals" / "claim_relevance.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    code = ast.unparse(tree).lower()
    for leaked in ("ai security", "improved ai agents", "governance digital", "intelligent ledger",
                   "model context protocol", "rag poisoning"):
        assert leaked not in code, leaked


# ---------- B: равнозначность псевдонима ----------

def test_permutation_alias_is_authoritative():
    kyc = {"phrase": "kyc aml", "canonical_label": "kyc aml", "original_phrase": "kyc aml",
           "merged_from": ["aml kyc"]}
    ok, name, _ = CR.match(doc("AML KYC automation for banks"), kyc)
    assert ok is True and name


def test_plural_and_punctuation_only_variation_is_authoritative():
    assert CR.equivalent("ai agents", "ai agent") is True
    assert CR.equivalent("multi-agent orchestration", "multi agent orchestration") is True


@pytest.mark.parametrize("canonical,alias,title", [
    ("model context protocol", "mcp", "acme/mcp-server: MCP tooling"),
    ("multi-agent orchestration", "agent", "Tutorial on building your first agent"),
    ("multi-agent orchestration", "customer relationship management",
     "acme/crm: customer relationship management suite"),
])
def test_unproven_alias_never_grants_support(canonical, alias, title):
    """Непересекающиеся, более широкие и «модель склеила» псевдонимы авторитета не дают."""
    cand = {"phrase": canonical, "canonical_label": canonical, "original_phrase": canonical,
            "merged_from": [alias]}
    ok, name, reason = CR.match(doc(title, source_type="репозиторий", name="GitHub"), cand)
    assert ok is False, (alias, name, reason)
    states = [st for _, _, st in CR.identity_report(cand) if st]
    assert CR.ALIAS_NOT_PROVEN_EQUIVALENT in states


def test_malformed_collapse_cannot_grant_code_family_support():
    """Модель канонизировала CRM вместе с оркестрацией агентов — семейство «код» не выдаётся."""
    cand = {"phrase": "multi-agent orchestration", "canonical_label": "multi-agent orchestration",
            "original_phrase": "multi-agent orchestration",
            "merged_from": ["customer relationship management"]}
    repo = doc("acme/crm: customer relationship management suite",
               "https://github.com/acme/crm", "репозиторий", "GitHub")
    q = EQ.assess_documents([repo], candidate=cand)
    assert q.supporting == [] and q.concrete_supporting_families == []


# ---------- D: происхождение привязки ----------

def test_merged_attachment_must_be_an_authoritative_equivalent():
    """P1-B: поглощённый псевдоним не наследует авторитет — когда заголовок сам НЕ называет выжившего.

    Прежняя редакция этого теста брала заголовок «Multi agent orchestration for large language
    models», который выжившего кандидата называет напрямую и фразовый контракт проходит. То есть
    она фиксировала не защиту P1-B, а ложноотрицательный случай: документ ровно про технологию
    кандидата отбрасывался из-за маршрута привязки. Защита проверяется заголовком, который про
    другую технологию, — сам предмет правила от этого не меняется."""
    survivor = {"phrase": "multi-agent orchestration", "canonical_label": "multi-agent orchestration",
                "original_phrase": "multi-agent orchestration", "merged_from": ["agent"]}
    d = doc("A conversational agent for hospital appointment scheduling")
    d.attachment_phrase, d.attachment_source = "agent", "merged"
    ok, _, reason = CR.match(d, survivor)
    assert ok is False and CR.ATTACHMENT_NOT_AUTHORITATIVE in reason


def test_merged_attachment_does_not_veto_a_title_that_names_the_survivor():
    """Тот же кандидат и тот же маршрут привязки, но заголовок называет выжившего сам."""
    survivor = {"phrase": "multi-agent orchestration", "canonical_label": "multi-agent orchestration",
                "original_phrase": "multi-agent orchestration", "merged_from": ["agent"]}
    d = doc("Multi agent orchestration for large language models")
    d.attachment_phrase, d.attachment_source = "agent", "merged"
    ok, name, reason = CR.match(d, survivor)
    assert ok is True and name == "multi-agent orchestration"
    assert CR.ATTACHMENT_NOT_AUTHORITATIVE not in reason


def test_merged_attachment_with_an_equivalent_phrase_is_allowed():
    survivor = {"phrase": "ai agents", "canonical_label": "ai agents",
                "original_phrase": "ai agents", "merged_from": ["ai agent"]}
    d = doc("AI agents for industrial scheduling")
    d.attachment_phrase, d.attachment_source = "ai agent", "merged"
    assert CR.match(d, survivor)[0] is True


def test_own_attachment_is_not_gated():
    d = doc("Edge caching policies for video delivery")
    d.attachment_phrase, d.attachment_source = "edge caching", "own"
    assert CR.match(d, "edge caching")[0] is True


def test_absorbed_evidence_records_its_attachment_phrase():
    lead = {"phrase": "multi-agent orchestration", "burst": 3.0, "examples": [], "repos": []}
    other = {"phrase": "agent", "burst": 9.0,
             "examples": [{"id": "W1", "title": "Tutorial on building your first agent"}],
             "repos": [{"full_name": "a/b", "html_url": "https://github.com/a/b"}]}
    relevance._absorb(lead, other)
    assert lead["examples"][0]["attached_via"] == "agent"
    assert lead["repos"][0]["attached_via"] == "agent"


def test_attachment_provenance_reaches_serialization():
    d = doc("Edge caching policies for video delivery")
    d.attachment_phrase, d.attachment_normalized, d.attachment_source = "edge caching", "edge caching", "own"
    q = EQ.assess_documents([d], candidate="edge caching")
    ref = q.claims[0].documents[0]
    assert ref["attachment_phrase"] == "edge caching"
    assert ref["attachment_source"] == "own"


# ---------- P1-C: неизвестная релевантность закрыта ----------

def _state():
    s = EvidenceState()
    s.add(Observation("openalex", OK, {"oa_total": 40, "oa_growth": 1.8, "oa_age": 2,
                                       "oa_last_share": 0.45, "oa_preprint_share": 0.3,
                                       "oa_peak_ratio": 1.2, "oa_fields": 2, "oa_prior": 3}))
    s.add(Observation("github", OK, {"gh_total": 30, "gh_12m": 25, "gh_new_share": 0.8}))
    s.add(Observation("news", NO_RESULTS, {"news_1y": 0, "news_30d": 0, "news_30d_share": 0.0,
                                           "news_press_share": 0.0}))
    s.add(Observation("hn", OK, {"hn_total": 6, "hn_12m": 5, "hn_growth": 1.1}))
    s.add(Observation("wiki", NO_RESULTS, {"wiki_article": 0, "wiki_age": 0}))
    return s


UNANNOTATED = [doc("Lattice calibration for cryogenic interconnects", "https://doi.org/10.1038/a"),
               doc("someorg/lattice-cal: toolkit", "https://github.com/someorg/lattice-cal",
                   "репозиторий", "GitHub")]


def test_unknown_relevance_never_counts_as_support():
    for d in UNANNOTATED:
        assert getattr(d, "claim_relevant", None) is None
        assert EQ.counts_as_support(d) is False


def test_decision_path_rejects_unannotated_documents():
    """Прямая регрессия по пути решения: неразмеченные документы не открывают ACCEPT."""
    d = decide(_state(), burst=0.8, documents=UNANNOTATED, verification=SEMANTICALLY_VERIFIED)
    assert d.decision != ACCEPT
    assert d.supporting_document_count == 0


def test_annotated_relevant_documents_still_accept():
    annotated = [doc("Lattice calibration for cryogenic interconnects", "https://doi.org/10.1038/a"),
                 doc("someorg/lattice-calibration: toolkit",
                     "https://github.com/someorg/lattice-calibration", "репозиторий", "GitHub")]
    cand = {"phrase": "lattice calibration", "canonical_label": "lattice calibration",
            "original_phrase": "lattice calibration"}
    d = decide(_state(), burst=0.8, documents=annotated, verification=SEMANTICALLY_VERIFIED,
               candidate=cand)
    assert d.decision == ACCEPT, d.decision_reason_codes


def test_declared_and_implemented_behaviour_agree():
    cfg = provenance.behavioral_config()["evidence"]["claim_relevance"]
    assert cfg["unknown_counts_as_support"] is False
    assert EQ.counts_as_support(doc("anything")) is False


# ---------- P1-D: юникод и межъязыковое состояние ----------

def test_cyrillic_title_is_tokenized_rather_than_empty():
    toks = CR.tokens("Постквантовая криптография в банковских системах")
    assert toks and all(t.strip() for t in toks)


def test_russian_candidate_matches_russian_evidence():
    ok, name, _ = CR.match(doc("Постквантовая криптография в банковских системах"),
                           "постквантовая криптография")
    assert ok is True and name


def test_cross_language_is_reported_honestly_not_as_a_missing_title():
    ok, _, reason = CR.match(doc("Постквантовая криптография в банковских системах"),
                             "post-quantum cryptography")
    assert ok is False
    assert CR.CROSS_LANGUAGE_UNVERIFIED in reason
    assert "нет заголовка" not in reason


def test_genuinely_empty_title_still_says_so():
    ok, _, reason = CR.match(doc(""), "post-quantum cryptography")
    assert ok is False and "нет заголовка" in reason


# ---------- A: канон модели не является авторитетом доказательства ----------
#
# Измеренная атака: свободное семантическое переименование делало подтверждением документы про
# СОВСЕМ ДРУГУЮ технологию. Канон годится для показа и кластеризации, но не для истины
# доказательства: LOW_EVIDENCE предпочтительнее подтверждения через переименование.

RENAMED = {"phrase": "quantum key distribution", "original_phrase": "quantum key distribution",
           "canonical_label": "homomorphic encryption", "merged_from": []}
RENAMED_DOCS = [
    doc("Homomorphic encryption for privacy preserving analytics", "https://doi.org/10.1038/a",
         "научная публикация", "Nature"),
    doc("someorg/homomorphic-encryption: HE toolkit", "https://github.com/someorg/he",
         "репозиторий", "GitHub"),
]


def test_free_semantic_rename_is_not_claim_evidence_authority():
    ok, name, reason = CR.match(RENAMED_DOCS[0], RENAMED)
    assert ok is False, (name, reason)
    states = {n: st for n, _, st in CR.identity_report(RENAMED) if st}
    assert states["homomorphic encryption"] == CR.NON_AUTHORITATIVE_FOR_CLAIM_EVIDENCE
    assert [n for n, _ in CR.identity_variants(RENAMED)] == ["quantum key distribution"]


def test_renamed_candidate_cannot_satisfy_families_or_accept():
    q = EQ.assess_documents(RENAMED_DOCS, candidate=RENAMED)
    assert q.supporting == [] and q.concrete_supporting_families == []
    assert q.independent_supports == 0
    d = decide(_state(), burst=0.8, documents=RENAMED_DOCS, verification=SEMANTICALLY_VERIFIED,
               candidate=RENAMED)
    assert d.decision != ACCEPT
    assert d.supporting_document_count == 0


def test_originating_identity_still_supports_the_same_candidate():
    """Ремонт не закрывает допуск: по СВОЕМУ названию кандидат подтверждается как прежде."""
    own = [doc("Quantum key distribution over metropolitan fibre", "https://doi.org/10.1038/b",
                "научная публикация", "Nature"),
           doc("someorg/quantum-key-distribution: QKD stack",
                "https://github.com/someorg/qkd", "репозиторий", "GitHub")]
    d = decide(_state(), burst=0.8, documents=own, verification=SEMANTICALLY_VERIFIED,
               candidate=RENAMED)
    assert d.decision == ACCEPT, d.decision_reason_codes


@pytest.mark.parametrize("original,canonical,title", [
    ("kyc aml", "aml kyc", "AML KYC automation for banks"),
    ("ai agents", "ai agent", "AI agent orchestration for industry"),
    ("multi-agent orchestration", "multi agent orchestration", "Multi-agent orchestration for LLMs"),
])
def test_deterministically_equivalent_canonical_keeps_authority(original, canonical, title):
    cand = {"phrase": original, "original_phrase": original, "canonical_label": canonical,
            "merged_from": []}
    assert CR.match(doc(title, "https://x/1", "научная публикация", "J"), cand)[0] is True


def test_modifier_dropping_canonicalization_gains_no_authority():
    """«improved ai agents» → «ai agents» — семантическое сокращение, а не равнозначность.

    Матчер при этом НЕ ослабляется: правило равнозначности остаётся прежним, а результат
    LOW_EVIDENCE принимается как есть."""
    assert CR.equivalent("improved ai agents", "ai agents") is False
    cand = {"phrase": "improved ai agents", "original_phrase": "improved ai agents",
            "canonical_label": "ai agents", "merged_from": []}
    ok, _, _ = CR.match(doc("AI agents for industrial scheduling", "https://x/2",
                             "научная публикация", "J"), cand)
    assert ok is False


def test_merged_attachment_gate_uses_authoritative_variants_only():
    """Привязка через поглощённый вариант сверяется с АВТОРИТЕТНЫМИ названиями, не с переименованием."""
    d = doc("Homomorphic encryption for privacy preserving analytics", "https://doi.org/10.1038/a",
             "научная публикация", "Nature")
    d.attachment_phrase, d.attachment_source = "homomorphic encryption", "merged"
    assert CR.authoritative_attachment(d, RENAMED) is False
