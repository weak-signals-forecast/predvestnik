"""Fail-closed доверие и независимость свидетельств (R2-B, R2-E)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import independence, trust  # noqa: E402
from wsignals.schema import Document  # noqa: E402


def doc(title="t", url="https://example.test/a", name="", source_type="новости"):
    return trust.assess(Document(title=title, url=url, source_name=name, source_type=source_type,
                                 published="2026-03-01", language="en")).with_provenance()


# ---------- R2-B: неизвестное остаётся неизвестным ----------

def test_unclassified_outlet_is_explicitly_unknown_not_medium():
    d = doc(url="https://some-unlisted-outlet.test/a", name="Some Unlisted Outlet")
    assert d.trust == trust.TIER_UNKNOWN
    assert "не опознано" in d.trust_reason


def test_unknown_tier_is_not_supporting_evidence():
    assert not trust.is_supporting(doc(url="https://some-unlisted-outlet.test/a", name="X"))
    assert trust.TIER_UNKNOWN not in trust.SUPPORTING_TIERS


@pytest.mark.parametrize("url,name,source_type,tier", [
    ("https://www.nature.com/articles/x", "Nature", "научная публикация", trust.TIER_A),
    ("https://arxiv.org/abs/1", "arXiv", "препринт", trust.TIER_A),
    ("https://habr.com/ru/articles/1", "Хабр", "новости", trust.TIER_B),
    ("https://github.com/a/b", "GitHub", "репозиторий", trust.TIER_B),
    ("https://www.prnewswire.com/x", "PR Newswire", "пресс-релиз", trust.TIER_C),
    ("https://news.ycombinator.com/item?id=1", "Hacker News", "сообщество", trust.TIER_C),
])
def test_known_classes_keep_their_tier(url, name, source_type, tier):
    assert doc(url=url, name=name, source_type=source_type).trust == tier


def test_tier_order_is_total_and_places_unknown_lowest():
    assert trust.TIER_ORDER[trust.TIER_UNKNOWN] < trust.TIER_ORDER[trust.TIER_C]
    assert trust.TIER_ORDER[trust.TIER_C] < trust.TIER_ORDER[trust.TIER_B] < trust.TIER_ORDER[trust.TIER_A]


# ---------- R2-B: claim-относительное доверие ----------

def test_code_is_primary_for_implementation_but_not_for_adoption():
    repo = doc(title="acme/x", url="https://github.com/acme/x", name="GitHub", source_type="репозиторий")
    assert trust.supports_claim(repo, trust.CLAIM_IMPLEMENTATION)
    assert trust.supports_claim(repo, trust.CLAIM_EXISTENCE)
    assert not trust.supports_claim(repo, trust.CLAIM_ADOPTION)
    assert not trust.supports_claim(repo, trust.CLAIM_REGULATION)


def test_science_is_primary_for_research_not_for_implementation():
    paper = doc(url="https://doi.org/10.1145/1", name="ACM", source_type="научная публикация")
    assert trust.supports_claim(paper, trust.CLAIM_RESEARCH)
    assert not trust.supports_claim(paper, trust.CLAIM_IMPLEMENTATION)


def test_registry_is_primary_for_regulation():
    std = doc(url="https://www.iso.org/standard/1", name="ISO", source_type="реестр")
    assert trust.supports_claim(std, trust.CLAIM_REGULATION)


def test_no_family_is_primary_for_adoption_yet():
    """Внедрение на рынке подтвердить нечем: провайдеров такого класса в системе нет, и притворяться
    обратным нельзя."""
    for url, name, t in (("https://doi.org/10.1145/1", "ACM", "научная публикация"),
                         ("https://github.com/a/b", "GitHub", "репозиторий"),
                         ("https://habr.com/x", "Хабр", "новости")):
        assert not trust.supports_claim(doc(url=url, name=name, source_type=t), trust.CLAIM_ADOPTION)


def test_press_and_community_are_primary_for_nothing():
    for url, name, t in (("https://www.prnewswire.com/x", "PR Newswire", "пресс-релиз"),
                         ("https://news.ycombinator.com/item?id=1", "Hacker News", "сообщество")):
        assert trust.claims_supported(doc(url=url, name=name, source_type=t)) == []


# ---------- R2-E: независимость ----------

def test_syndicated_reprints_count_as_one_evidence_item():
    docs = [doc("Acme launches agent attestation", "https://www.prnewswire.com/a", "PR Newswire", "пресс-релиз"),
            doc("Acme launches agent attestation", "https://finance.yahoo.test/b", "Yahoo", "новости"),
            doc("Acme launches agent attestation!", "https://other-feed.test/c", "Other", "новости")]
    keys = {independence.independence_key(d) for d in docs}
    assert len(keys) == 1
    assert len(independence.independent_documents(docs)) == 1


def test_independent_documents_keeps_the_best_tier_in_a_group():
    docs = [doc("Same headline", "https://unlisted.test/a", "Unlisted", "новости"),
            doc("Same headline", "https://habr.com/b", "Хабр", "новости")]
    kept = independence.independent_documents(docs)
    assert len(kept) == 1 and kept[0].trust == trust.TIER_B


def test_same_publisher_different_papers_stay_distinct():
    docs = [doc("First study", "https://doi.org/10.1145/1", "ACM", "научная публикация"),
            doc("Second study", "https://doi.org/10.1145/2", "ACM", "научная публикация")]
    assert len(independence.independent_documents(docs)) == 2
    assert len({independence.origin(d) for d in docs}) == 1      # но происхождение одно


def test_repo_origin_is_the_owner_not_the_host():
    a = doc("acme/one", "https://github.com/acme/one", "GitHub", "репозиторий")
    b = doc("acme/two", "https://github.com/acme/two", "GitHub", "репозиторий")
    c = doc("beta/three", "https://github.com/beta/three", "GitHub", "репозиторий")
    assert independence.origin(a) == independence.origin(b) == "github.com:acme"
    assert independence.origin(c) != independence.origin(a)


def test_subdomains_collapse_to_the_registrable_domain():
    assert independence.registrable_domain("https://ir.reuters.com/x") == "reuters.com"
    assert independence.registrable_domain("https://www.reuters.com/y") == "reuters.com"
    assert independence.registrable_domain("https://news.bbc.co.uk/z") == "bbc.co.uk"


def test_syndication_and_concentration_rates():
    docs = [doc("A", "https://doi.org/10.1145/1", "ACM", "научная публикация"),
            doc("Same", "https://x.test/a", "X", "новости"),
            doc("Same", "https://y.test/b", "Y", "новости")]
    assert independence.syndication_rate(docs) > 0
    assert 0.0 < independence.origin_concentration(docs) <= 1.0
    assert independence.syndication_rate([]) == 0.0 and independence.origin_concentration([]) == 0.0
