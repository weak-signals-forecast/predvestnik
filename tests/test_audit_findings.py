"""Наблюдения аудита.

НАБЛЮДЕНИЕ 1 (доверие к научным публикациям грубо) зафиксировано КАК ЕСТЬ: политика площадок
в этой задаче не меняется и уходит в научный red-team. Тест описывает текущее поведение, а не
одобряет его: когда рубрику поменяют, он упадёт и потребует явного обновления.

НАБЛЮДЕНИЕ 2 (препринт и журнальная версия одной работы) ИСПРАВЛЕНО в P1-A: одна научная работа
больше не может выступать двумя независимыми подтверждениями. Тесты ниже закрепляют исправленное
поведение.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import independence  # noqa: E402
from wsignals.evidence_quality import assess_documents, gate  # noqa: E402
from wsignals.schema import Document  # noqa: E402
from wsignals.trust import CLAIM_EXISTENCE, TIER_A, assess, supports_claim  # noqa: E402


def doc(title, url, source_type, name):
    _d = assess(Document(title=title, url=url, source_name=name, source_type=source_type,
                           published="2026-03-01", language="en")).with_provenance()
    # P1-C. Признак претензионной релевантности читается строго: подтверждает только явное
    # True. Эта фикстура ПО ПОСТРОЕНИЮ представляет документ про рассматриваемого кандидата,
    # поэтому помечается явно. Неразмеченные документы проверяются отдельно тестом,
    # доказывающим, что они не открывают ACCEPT.
    _d.claim_relevant = True
    return _d


# ---------- НАБЛЮДЕНИЕ 1: шкала доверия к научным публикациям груба ----------

@pytest.mark.parametrize("venue,url", [
    ("Nature", "https://doi.org/10.1038/x"),
    ("ACM", "https://doi.org/10.1145/x"),
    ("Unknown Proceedings of Somewhere", "https://doi.org/10.9999/x"),
    ("Obscure Open Journal", "https://example-journal.test/a"),
    ("No venue at all", "https://random-host.test/paper.pdf"),
])
def test_finding_any_scientific_publication_gets_tier_a_regardless_of_venue(venue, url):
    """ТЕКУЩЕЕ ПОВЕДЕНИЕ: разряд определяется ТИПОМ источника, площадка не учитывается.

    Это наблюдение аудита, а не утверждение о правильности. Белого списка журналов здесь нет
    и в этой задаче не вводится."""
    d = doc("paper", url, "научная публикация", venue)
    assert d.trust == TIER_A
    assert supports_claim(d, CLAIM_EXISTENCE) is True


def test_finding_venue_quality_is_not_an_input_to_trust_at_all():
    weak = doc("paper", "https://random-host.test/p.pdf", "научная публикация", "No venue at all")
    strong = doc("paper", "https://doi.org/10.1038/x", "научная публикация", "Nature")
    assert weak.trust == strong.trust, "площадка сейчас ни на что не влияет"


# ---------- НАБЛЮДЕНИЕ 2 (ИСПРАВЛЕНО, P1-A): препринт и журнальная версия одной работы ----------

SAME_TITLE = "Agent attestation for autonomous payments"
PREPRINT = doc(SAME_TITLE, "https://doi.org/10.48550/arXiv.2601.00001", "препринт", "arXiv")
JOURNAL = doc(SAME_TITLE, "https://doi.org/10.1145/3716628", "научная публикация", "ACM")


def test_publisher_prefix_still_differs_but_no_longer_implies_independence():
    """Ключ независимости по-прежнему включает издателя — но сам по себе он больше не решает.

    Работа опознаётся отдельно, поэтому разные префиксы DOI не делают одну работу двумя."""
    assert independence.independence_key(PREPRINT) != independence.independence_key(JOURNAL)
    assert independence.origin(PREPRINT) != independence.origin(JOURNAL)
    assert independence.same_scientific_work(PREPRINT, JOURNAL)


def test_same_work_no_longer_satisfies_independent_corroboration():
    """ИСПРАВЛЕНО (P1-A): препринт плюс его журнальная версия — одно свидетельство, а не два."""
    q = assess_documents([PREPRINT, JOURNAL], aggregate_families=["science"])
    assert q.independent_supports == 1 and q.distinct_origins == 1
    assert q.same_work_merges == 1
    passed, code, _ = gate(q)
    assert passed is False and code == "no_independent_corroboration"


def test_finding_does_not_extend_to_news_where_syndication_already_collapses():
    """Контроль: в новостных семействах перепечатка уже считается одним свидетельством —
    наблюдение касается именно научного семейства."""
    a = doc("Acme launches attestation", "https://www.prnewswire.com/a", "пресс-релиз", "PR Newswire")
    b = doc("Acme launches attestation", "https://finance.yahoo.test/b", "новости", "Yahoo")
    assert independence.independence_key(a) == independence.independence_key(b)
