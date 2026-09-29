"""Инвариантные проверки внутреннего состязательного бенчмарка (R2-I).

Бенчмарк — инженерный, а не эталон организаторов. Здесь проверяются только те его результаты, которые
являются жёсткими инвариантами системы и не зависят от субъективной разметки:

  * подтверждение только по классу D или по неизвестному качеству невозможно;
  * утверждение без материализованного документа невозможно;
  * при недоступной семантической нормализации очевидный шум не подтверждается;
  * бренды и широкие зрелые категории не подтверждаются никогда;
  * список наблюдения не пустует и несёт доказательства.

Субъективные случаи помечены на разметку эксперту и в точность не засчитываются.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_quality import (AREAS, MODE_COMPETENT, MODE_NAIVE, MODE_UNAVAILABLE,  # noqa: E402
                                       benchmark, cases)

MODES = (MODE_UNAVAILABLE, MODE_COMPETENT, MODE_NAIVE)
REPORTS = {m: benchmark(m) for m in MODES}


def test_benchmark_covers_all_six_organizer_areas():
    assert set(REPORTS[MODE_UNAVAILABLE]["areas_covered"]) == set(AREAS)


def test_benchmark_covers_every_required_adversarial_class():
    classes = set(REPORTS[MODE_UNAVAILABLE]["by_class"])
    for required in ("не по теме", "бренд или продукт", "широкая зрелая", "пресс-релизы и хайп",
                     "родственная семья", "узкая зарождающаяся", "частичный отказ провайдеров",
                     "только D или неизвестное качество", "зависимые свидетельства",
                     "независимое подтверждение"):
        assert required in classes, required


@pytest.mark.parametrize("mode", MODES)
def test_no_acceptance_on_class_d_or_unknown_quality(mode):
    m = REPORTS[mode]
    assert m["D_only_accept_count"] == 0, mode
    assert m["unknown_trust_accept_count"] == 0, mode


@pytest.mark.parametrize("mode", MODES)
def test_no_confirmed_claim_lacks_a_document(mode):
    assert REPORTS[mode]["claim_without_document_count"] == 0, mode


def test_nothing_is_confirmed_when_the_normalizer_is_unavailable():
    """R2-B: при отказе нормализатора окончательных подтверждений нет вообще, а не «немного и точных»."""
    m = REPORTS[MODE_UNAVAILABLE]
    assert m["confirmed_count"] == 0, m["rows"] and [r["phrase"] for r in m["rows"] if r["decision"] == "ACCEPT"]
    assert m["obvious_noise_confirmed_count"] == 0
    # точность на пустом множестве не определена и 1.0 выдаваться не должна
    assert m["confirmed_precision_by_expert_review"] is None


def test_obvious_noise_is_never_confirmed_with_a_competent_normalizer():
    m = REPORTS[MODE_COMPETENT]
    assert m["obvious_noise_confirmed_count"] == 0, m["obvious_noise_confirmed"]


@pytest.mark.parametrize("mode", MODES)
def test_brands_and_mature_categories_are_never_confirmed(mode):
    rows = REPORTS[mode]["rows"]
    for r in rows:
        if r["class"] in ("бренд или продукт", "широкая зрелая"):
            assert r["decision"] != "ACCEPT", (mode, r["phrase"], r["decision"])


@pytest.mark.parametrize("mode", MODES)
def test_dependent_evidence_is_never_confirmed(mode):
    for r in REPORTS[mode]["rows"]:
        if r["class"] == "зависимые свидетельства":
            assert r["decision"] != "ACCEPT", (mode, r["phrase"])


@pytest.mark.parametrize("mode", MODES)
def test_only_d_or_unknown_quality_is_never_confirmed(mode):
    for r in REPORTS[mode]["rows"]:
        if r["class"] == "только D или неизвестное качество":
            assert r["decision"] != "ACCEPT", (mode, r["phrase"])


def test_narrow_emerging_technologies_are_confirmed_with_a_working_normalizer():
    """Система обязана оставаться полезной: узкие зарождающиеся технологии подтверждаются."""
    rows = [r for r in REPORTS[MODE_COMPETENT]["rows"] if r["class"] == "узкая зарождающаяся"]
    confirmed = [r for r in rows if r["decision"] == "ACCEPT"]
    assert len(confirmed) == len(rows), [r["phrase"] for r in rows if r["decision"] != "ACCEPT"]


def test_system_stays_useful_without_the_normalizer():
    """Отказ модели — рабочее состояние продукта: выдача не пустеет и не падает, кандидаты уходят
    в список наблюдения с приложенными доказательствами."""
    m = REPORTS[MODE_UNAVAILABLE]
    assert m["watchlist_count"] >= 10, m["watchlist_count"]
    assert m["watchlist_usefulness"] >= 0.5, m["watchlist_usefulness"]
    # вето зрелости, хайпа и шума продолжают работать и в отказном режиме
    by_class = m["by_class"]
    assert by_class["широкая зрелая"].get("MATURE", 0) == by_class["широкая зрелая"]["total"]
    assert by_class["бренд или продукт"].get("NOISE", 0) == by_class["бренд или продукт"]["total"]


@pytest.mark.parametrize("mode", MODES)
def test_confirmed_top_has_no_duplicate_families(mode):
    assert REPORTS[mode]["duplicate_rate"] == 0.0, mode


@pytest.mark.parametrize("mode", MODES)
def test_sibling_concentration_stays_bounded(mode):
    """Родственники не склеиваются, но и не заполняют выдачу целиком."""
    assert REPORTS[mode]["sibling_concentration"] <= 0.35, (mode, REPORTS[mode]["sibling_concentration"])


def test_partial_provider_failure_does_not_crash_or_fabricate():
    for mode in MODES:
        rows = [r for r in REPORTS[mode]["rows"] if r["class"] == "частичный отказ провайдеров"]
        assert rows
        for r in rows:
            assert r["decision"] in ("ACCEPT", "LOW_EVIDENCE", "MATURE", "HYPE", "NOISE")
            if r["decision"] == "ACCEPT":
                assert r["supporting"] >= 2, r["phrase"]


def test_ambiguous_cases_are_queued_for_expert_review_not_auto_labelled():
    unlabelled = [c for c in cases() if c.label is None]
    assert unlabelled, "спорные случаи должны существовать и не получать автоматических меток"
    queue = REPORTS[MODE_UNAVAILABLE]["expert_review_queue"]
    assert queue and set(queue) <= {c.phrase for c in unlabelled}


def test_normalizer_quality_is_the_dominant_precision_factor():
    """Честное сравнение границ: наивный нормализатор роняет точность, и это видно в отчёте."""
    competent = REPORTS[MODE_COMPETENT]["confirmed_precision_by_expert_review"]
    naive = REPORTS[MODE_NAIVE]["confirmed_precision_by_expert_review"]
    assert competent > naive
