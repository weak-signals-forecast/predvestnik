"""Качество склейки семейств: что должно слипаться, что обязано остаться раздельным (REPAIR D).

Правило приоритета: ложная склейка хуже двух отдельных кандидатов. Поэтому склеиваются только варианты
одной и той же фразы (число, порядок слов, общие и причастные определения), а разные определения при
общем головном слове семьёй не считаются.

Фикстуры собраны по трём направлениям задания — ИИ, финтех, кибербезопасность — и повторяют формы,
которые реально встречались в выдаче, но проверяют общее правило, а не конкретные фразы.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals.relevance import collapse, duplicate_rate, merge_key, token_similarity  # noqa: E402


def cand(phrase, burst=1.0, **kw):
    c = {"phrase": phrase, "original_phrase": phrase, "canonical_label": phrase, "burst": burst,
         "channels": {"литература": 2, "репозитории": 1}, "recent_docs": 2, "prior_docs": 0,
         "examples": [], "repos": []}
    c.update(kw)
    return c


def collapse_phrases(phrases):
    kept, merged = collapse([cand(p, burst=len(phrases) - i) for i, p in enumerate(phrases)])
    return [c["phrase"] for c in kept], [c["phrase"] for c in merged]


# ---------- что обязано слипаться ----------

@pytest.mark.parametrize("variants", [
    ["mcp server", "mcp servers"],                       # число
    ["rag poisoning", "poisoning of rag"],               # порядок слов
    ["agent identity", "ai agent identity"],             # общее определение
    ["ai-powered fraud", "ai-driven fraud"],             # причастное определение
    ["kyc aml", "aml kyc"],                              # перестановка
    ["zero-knowledge proof", "zero knowledge proofs"],   # дефис и число
])
def test_variants_of_one_phenomenon_collapse(variants):
    kept, merged = collapse_phrases(variants)
    assert len(kept) == 1, (variants, kept)
    assert len(merged) == len(variants) - 1


# ---------- что обязано остаться раздельным ----------

@pytest.mark.parametrize("distinct", [
    ["rag security", "llm security", "mcp security"],        # разные определения, одно головное слово
    ["stablecoin payments", "agentic payments", "cross-border payments"],
    ["token compression", "context compression"],
    ["fraud detection", "anomaly detection"],
    ["quantum sensing", "quantum computing"],
])
def test_different_technologies_never_collapse(distinct):
    kept, merged = collapse_phrases(distinct)
    assert len(kept) == len(distinct), (distinct, kept)
    assert merged == []


def test_product_and_technology_stay_separate():
    kept, _ = collapse_phrases(["agent skills", "claude skills"])
    assert len(kept) == 2


# ---------- склейка по ключу модели проходит детерминированную проверку (REPAIR B) ----------

def llm_cand(phrase, canonical, key, burst=1.0):
    c = cand(phrase, burst=burst)
    c["canonical_label"] = canonical
    c["merge_key"] = key
    c["assessment"] = {"source": "llm", "canonical_name": canonical, "merge_key": key}
    return c


def test_llm_merge_is_accepted_when_canonical_names_agree():
    kept, merged = collapse([llm_cand("agent identity", "agent identity", "agent-identity", 3.0),
                             llm_cand("machine credentials", "agent identity", "agent-identity", 1.0)])
    assert len(kept) == 1 and merged[0]["merged_into"] == "agent identity"


def test_llm_merge_is_refused_when_nothing_deterministic_supports_it():
    """Модель назвала один ключ двум разным технологиям — склейка необратима, поэтому она отклоняется."""
    kept, merged = collapse([llm_cand("quantum sensing", "quantum sensing", "same-key", 3.0),
                             llm_cand("payment rails", "payment rails", "same-key", 1.0)])
    assert len(kept) == 2 and merged == []
    rejected = [c for c in kept if c.get("merge_notes")]
    assert rejected and "отклонена детерминированной проверкой" in rejected[0]["merge_notes"][0]


def test_llm_merge_is_accepted_when_phrases_are_similar_enough():
    kept, merged = collapse([llm_cand("rag poisoning", "rag poisoning attack", "rag-poison", 3.0),
                             llm_cand("rag poisoning attacks", "rag poisoning attack", "rag-poison", 1.0)])
    assert len(kept) == 1 and len(merged) == 1


def test_similarity_is_symmetric_and_bounded():
    for a, b in (("rag poisoning", "poisoning rag"), ("a b", "c d"), ("x", "x")):
        s1, s2 = token_similarity(a, b), token_similarity(b, a)
        assert s1 == s2 and 0.0 <= s1 <= 1.0


# ---------- лineage ----------

def test_collapse_preserves_lineage_and_evidence():
    lead = cand("mcp server", burst=3.0, examples=[{"id": "W1"}], repos=[{"full_name": "a/b"}])
    other = cand("mcp servers", burst=1.0, examples=[{"id": "W2"}], repos=[{"full_name": "c/d"}])
    kept, merged = collapse([lead, other])
    s = kept[0]
    assert s["original_phrase"] == "mcp server" and s["merged_from"] == ["mcp servers"]
    assert {w["id"] for w in s["examples"]} == {"W1", "W2"}
    assert {r["full_name"] for r in s["repos"]} == {"a/b", "c/d"}
    assert s["recent_docs"] == 4 and s["merge_key"] == merge_key("mcp server")
    assert merged[0]["merged_into"] == "mcp server"


# ---------- доля дублей до и после ----------

FIXTURES = {
    "ИИ": ["ai-agent skill", "ai agent skills", "token compression", "context compression",
           "instruction tuning", "instruction-tuning", "secure ai agent", "secure ai agents",
           "chain of thought", "chain-of-thought"],
    "финтех": ["stablecoin payments", "stablecoin payment", "ai-powered fraud", "ai-driven fraud",
               "aml kyc", "kyc aml", "x402 payments", "agentic payments", "tokenized deposits"],
    "кибербезопасность": ["rag poisoning", "poisoning of rag", "mcp security", "mcp securities",
                          "llm security", "rag security", "ai red team", "ai red teams",
                          "prompt injection", "prompt injections"],
}


@pytest.mark.parametrize("domain", sorted(FIXTURES))
def test_duplicate_rate_drops_to_zero_after_collapse(domain):
    phrases = FIXTURES[domain]
    before = duplicate_rate(phrases)
    kept, _ = collapse_phrases(phrases)
    after = duplicate_rate(kept)
    assert before > 0, (domain, "фикстура обязана содержать варианты")
    assert after == 0.0, (domain, kept)
    assert len(kept) < len(phrases)


def test_collapse_does_not_over_merge_the_fixtures():
    """Контроль на переусердствование: различимые технологии внутри фикстур обязаны выжить."""
    kept, _ = collapse_phrases(FIXTURES["кибербезопасность"])
    survivors = set(kept)
    for distinct in ("mcp security", "llm security", "rag security"):
        assert distinct in survivors, (distinct, survivors)
