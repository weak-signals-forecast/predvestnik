"""Релевантность кандидатов, канонизация и минимальное происхождение (PHASE 5, PHASE 7, PHASE 8)."""
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import relevance  # noqa: E402
from wsignals.schema import Document, canonicalize_url  # noqa: E402
from wsignals.trust import assess  # noqa: E402

SEED_TOKENS = {"cybersecurity", "security", "ai"}


# ---------- детерминированные фильтры ----------

@pytest.mark.parametrize("phrase", ["claude skills", "openai operator", "code codex", "chatgpt plugins"])
def test_unambiguous_product_phrases_are_rejected(phrase):
    """Отвергаются коины без обычного технического значения. Слова-омонимы («meta», «swift», «visa»,
    «gemini») проверяются отдельно: они лишь понижают специфичность."""
    assert "продукт" in relevance.deterministic_reason(phrase, SEED_TOKENS, set())


@pytest.mark.parametrize("phrase", ["awesome agents", "starter template", "agent cookbook"])
def test_repo_furniture_phrases_are_rejected(phrase):
    assert relevance.deterministic_reason(phrase, SEED_TOKENS, set()) is not None


@pytest.mark.parametrize("phrase", ["cli mcp", "mcp server"])
def test_soft_plumbing_phrases_survive_with_a_flag(phrase):
    """REPAIR A: обвязка понижает специфичность, но не отвергает кандидата."""
    r = relevance.assess(phrase, SEED_TOKENS, set(), channels={"литература": 2, "репозитории": 1}, lit_docs=1)
    assert not r.rejected and r.ambiguous, (phrase, r.reject_reason)


@pytest.mark.parametrize("phrase", ["ai security", "advanced technology", "emerging solutions"])
def test_restatements_of_the_query_are_rejected(phrase):
    a = relevance.assess(phrase, SEED_TOKENS, set(), channels={"литература": 2, "репозитории": 1}, lit_docs=3)
    assert a.rejected and a.is_generic_phrase and not a.is_technology


@pytest.mark.parametrize("phrase", ["agent identity", "rag poisoning", "confidential computing",
                                    "tokenized deposits", "on-policy distillation"])
def test_real_technology_phrases_survive(phrase):
    assert relevance.deterministic_reason(phrase, SEED_TOKENS, set()) is None, phrase


def test_plan_exclusions_are_honoured():
    assert relevance.deterministic_reason("neural pipeline", SEED_TOKENS, {"neural"}) is not None
    assert relevance.deterministic_reason("neural pipeline", SEED_TOKENS, set()) is None


def test_no_expected_technology_list_is_hardcoded():
    """Ни один список ожидаемых ТОП-технологий не зашит: фильтры работают только на отказ."""
    source = (ROOT / "wsignals" / "relevance.py").read_text(encoding="utf-8")
    for leaked in ("agentic payments", "rag poisoning", "agent identity", "on-policy distillation"):
        assert leaked not in source


def _cand(phrase):
    return {"phrase": phrase, "channels": {"литература": 3, "репозитории": 2}, "recent_docs": 3, "burst": 1.0}


def test_filter_splits_candidates_and_keeps_reasons():
    keep, drop = relevance.filter_candidates([_cand("agent identity"), _cand("claude skills")], SEED_TOKENS, set())
    assert [c["phrase"] for c in keep] == ["agent identity"]
    assert drop[0]["noise_reason"]
    assert keep[0]["original_phrase"] == "agent identity" and keep[0]["canonical_label"] == "agent identity"
    assert keep[0]["assessment"]["is_technology"] is True


# ---------- канонизация ----------

def test_canonicalization_is_skipped_without_a_configured_model(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    keep, dropped = relevance.canonicalize([_cand("agent identity")])
    assert keep[0]["phrase"] == "agent identity" and dropped == []


def _yandex(monkeypatch, text):
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")
    monkeypatch.setenv("YANDEX_QUERY_MODEL", "yandexgpt-5-lite")

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"alternatives": [{"message": {"text": text}}]}}

    monkeypatch.setattr(requests, "post", lambda *a, **kw: R())


def test_canonicalization_keeps_the_original_phrase(monkeypatch):
    _yandex(monkeypatch, '[{"i": 0, "canonical_name": "agent identity management", "is_technology": true,'
                         ' "domain_relevance": 0.9, "is_product_or_brand": false, "is_generic_phrase": false,'
                         ' "merge_key": "agent identity", "reject_reason": null}]')
    cands = [_cand("agent identity")]
    cands[0]["original_phrase"] = "agent identity"
    keep, dropped = relevance.canonicalize(cands)
    assert keep[0]["original_phrase"] == "agent identity"
    assert keep[0]["canonical_label"] == "agent identity management"
    assert keep[0]["assessment"]["source"] == "llm"


def test_canonicalization_cannot_invent_candidates(monkeypatch):
    _yandex(monkeypatch, '[{"i": 99, "canonical_name": "quantum teleportation banking", "is_technology": true,'
                         ' "domain_relevance": 1.0, "is_product_or_brand": false, "is_generic_phrase": false,'
                         ' "merge_key": null, "reject_reason": null}]')
    keep, dropped = relevance.canonicalize([_cand("agent identity")])
    assert len(keep) == 1 and keep[0]["phrase"] == "agent identity"
    assert "canonical_source" not in keep[0]          # выдуманный индекс не тронул ни одного кандидата


def test_canonicalization_failure_is_not_fatal(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")
    monkeypatch.setattr(requests, "post", lambda *a, **kw: (_ for _ in ()).throw(requests.ConnectionError("down")))
    keep, dropped = relevance.canonicalize([_cand("agent identity")])
    assert keep[0]["phrase"] == "agent identity" and dropped == []


# ---------- происхождение ----------

def test_canonical_url_strips_tracking_and_rejects_aggregator_redirects():
    assert canonicalize_url("https://www.nature.com/articles/x/?utm_source=nl&ref=a") == ("https://nature.com/articles/x", True)
    assert canonicalize_url("https://news.google.com/rss/articles/abc?oc=5") == (None, False)
    assert canonicalize_url("") == (None, False)


def test_document_provenance_fields_are_filled():
    d = Document(title="Agent identity for autonomous payments", url="https://doi.org/10.1/abc",
                 source_name="Nature", source_type="научная публикация", published="2026-03-01",
                 language="en", snippet="text").with_provenance(retrieved_at="2026-09-20T10:00:00+00:00")
    out = d.to_dict()
    assert out["canonical_url"] == "https://doi.org/10.1/abc"
    assert out["published_at"] == "2026-03-01" and out["retrieved_at"] == "2026-09-20T10:00:00+00:00"
    assert out["source_family"] == "science" and out["is_primary"] is True
    assert out["adapter_status"] == "OK" and len(out["content_hash"]) == 32
    assert out["language"] == "en"


def test_aggregator_document_is_not_primary_and_has_no_canonical_url():
    d = Document(title="t", url="https://news.google.com/rss/articles/abc?oc=5", source_name="RBC",
                 source_type="новости", published="2026-03-01", language="ru").with_provenance()
    assert d.canonical_url is None and d.is_primary is False and d.source_family == "media"


def test_stale_cache_status_is_visible_on_the_document():
    d = Document(title="t", url="https://github.com/x/y", source_name="GitHub", source_type="репозиторий",
                 published=None, language="en").with_provenance(retrieved_at="2026-09-01T00:00:00+00:00",
                                                                adapter_status="STALE_CACHE")
    assert d.adapter_status == "STALE_CACHE" and d.retrieved_at == "2026-09-01T00:00:00+00:00"


def test_trust_assessment_still_works_on_the_extended_document():
    d = assess(Document(title="t", url="https://www.prnewswire.com/x", source_name="PR Newswire",
                        source_type="новости", published=None, language="en")).with_provenance()
    assert d.trust == "пониженный" and d.source_family == "media"


def test_content_hash_is_stable_and_distinguishes_documents():
    def doc(title):
        return Document(title=title, url="https://example.com/a", source_name="s", source_type="новости",
                        published=None, language="en").with_provenance().content_hash

    assert doc("one") == doc("one") and doc("one") != doc("two")
