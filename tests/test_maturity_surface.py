"""Модель зрелости v2: поверхность отклика, монотонность и поведение на границах (REPAIR C).

Прежняя модель была неработоспособна по построению: непрерывная ветка упиралась в 0.50, средний разряд
Википедии давал 0.55, а порог вето равен 0.60 — пересечь его могли только два дискретных условия.

Здесь проверяется, что непрерывная ветка способна пересечь порог, что умеренные признаки складываются,
что отсутствие данных не снижает риск и что риск не убывает ни по одному входу. Пороги под эти тесты
не подбирались: тесты читают MATURITY_VETO из кода и не задают собственных чисел.
"""
import itertools
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals.decision import (MATURITY_VETO, MATURITY_WEIGHTS, UNKNOWN_MATURITY_RISK,  # noqa: E402
                               maturity, maturity_components)
from wsignals.evidence import ERROR, OK, EvidenceState, Observation  # noqa: E402

NO_ARTICLE = {"wiki_article": 0, "wiki_age": 0}


def state(*, oa=None, wiki=NO_ARTICLE, gh=None, wiki_error=False, oa_error=False) -> EvidenceState:
    s = EvidenceState()
    if oa_error:
        s.add(Observation("openalex", ERROR, error_type="timeout"))
    elif oa is not None:
        s.add(Observation("openalex", OK, oa))
    if wiki_error:
        s.add(Observation("wiki", ERROR, error_type="timeout"))
    elif wiki is not None:
        s.add(Observation("wiki", OK, wiki))
    if gh is not None:
        s.add(Observation("github", OK, gh))
    return s


def science(total, age, fields, prior):
    return {"oa_total": total, "oa_age": age, "oa_fields": fields, "oa_prior": prior}


def article(age):
    return {"wiki_article": 1, "wiki_age": age}


def risk(**kw) -> float:
    return maturity(state(**kw))[0]


# ---------- поверхность отклика ----------

VOLUME = {"низкий": 20, "средний": 700, "высокий": 4000}
AGE = {"молодой": 2, "средний": 6, "старый": 11}
BREADTH = {"узкая": 1, "широкая": 6}
ENCYCLOPEDIA = {"нет": NO_ARTICLE, "молодая": article(1), "старая": article(6)}


def surface() -> dict:
    out = {}
    for (vn, v), (an, ag), (bn, b), (en, w) in itertools.product(
            VOLUME.items(), AGE.items(), BREADTH.items(), ENCYCLOPEDIA.items()):
        prior = int(v * 0.2)
        out[(vn, an, bn, en)] = risk(oa=science(v, ag, b, prior), wiki=w)
    return out


SURFACE = surface()


def test_surface_covers_the_whole_grid():
    assert len(SURFACE) == len(VOLUME) * len(AGE) * len(BREADTH) * len(ENCYCLOPEDIA)


def test_low_evidence_young_niche_does_not_become_mature_from_absence():
    """Ничего не найдено и статьи нет — это не зрелость, а отсутствие следа."""
    for age in ("молодой", "средний"):
        for breadth in BREADTH:
            r = SURFACE[("низкий", age, breadth, "нет")]
            assert r < MATURITY_VETO, (age, breadth, r)


def test_strong_combinations_cross_the_veto():
    """Несколько сильных признаков вместе обязаны давать вето."""
    assert SURFACE[("высокий", "старый", "широкая", "старая")] >= MATURITY_VETO
    assert SURFACE[("высокий", "старый", "широкая", "нет")] >= MATURITY_VETO
    assert SURFACE[("высокий", "средний", "широкая", "молодая")] >= MATURITY_VETO


def test_moderate_indicators_combine_into_a_veto_without_any_discrete_fact():
    """Ни одного дискретного признака: ни статьи старше пяти лет, ни 3000 работ за 8 лет."""
    v = maturity(state(oa=science(800, 6, 4, 150), wiki=article(3)))
    assert v[0] >= MATURITY_VETO
    assert not any("Википедии существует" in r and "лет" in r and "5" in r for r in v[1])


def test_continuous_branch_alone_can_cross_the_veto():
    """Непрерывная ветка ниже дискретных условий (2900 работ < 3000, статьи нет) всё равно пересекает порог."""
    r = risk(oa=science(2900, 7, 8, 500), wiki=NO_ARTICLE)
    assert r >= MATURITY_VETO, r


def test_large_broad_established_is_mature_even_when_young():
    """Требование 5: крупная и широкая область получает высокий риск даже моложе шести лет."""
    young_big = risk(oa=science(5000, 4, 6, 500), wiki=NO_ARTICLE)
    young_small = risk(oa=science(30, 4, 1, 5), wiki=NO_ARTICLE)
    assert young_big >= MATURITY_VETO > young_small


def test_high_growth_cannot_erase_maturity():
    """Требование 6: темпа роста в формуле нет, поэтому он ничего не стирает."""
    slow = risk(oa={**science(4000, 9, 6, 800), "oa_growth": 0.0}, wiki=NO_ARTICLE)
    fast = risk(oa={**science(4000, 9, 6, 800), "oa_growth": 5.0}, wiki=NO_ARTICLE)
    assert slow == fast


# ---------- монотонность ----------

@pytest.mark.parametrize("field,low,high", [
    ("oa_total", 50, 2500), ("oa_age", 2, 9), ("oa_fields", 1, 6), ("oa_prior", 0, 250),
])
def test_risk_never_decreases_when_an_input_increases(field, low, high):
    for total, age, fields, prior in itertools.product((50, 700), (2, 7), (1, 5), (0, 100)):
        base = science(total, age, fields, prior)
        a = risk(oa={**base, field: low}, wiki=NO_ARTICLE)
        b = risk(oa={**base, field: high}, wiki=NO_ARTICLE)
        assert b >= a, (field, base, a, b)


def test_risk_never_decreases_with_encyclopedia_evidence():
    base = science(400, 5, 3, 80)
    none_ = risk(oa=base, wiki=NO_ARTICLE)
    young = risk(oa=base, wiki=article(1))
    old = risk(oa=base, wiki=article(6))
    assert none_ <= young <= old


def test_risk_never_decreases_with_adoption_evidence():
    base = science(400, 5, 3, 80)
    small = risk(oa=base, gh={"gh_total": 10})
    large = risk(oa=base, gh={"gh_total": 4000})
    assert large >= small


def test_every_weight_is_positive_so_no_component_can_lower_the_score():
    assert all(w > 0 for w in MATURITY_WEIGHTS.values())


# ---------- отсутствие данных ----------

def test_absent_component_is_excluded_from_the_denominator_not_counted_as_zero():
    s_ok = state(oa=science(600, 5, 4, 100), wiki=NO_ARTICLE)
    s_err = state(oa=science(600, 5, 4, 100), wiki_error=True)
    assert maturity_components(s_ok)["encyclopedia"] == 0.0
    assert maturity_components(s_err)["encyclopedia"] is None
    assert maturity(s_err)[0] >= maturity(s_ok)[0]        # отсутствие не снижает риск


def test_unknown_alone_cannot_produce_a_veto():
    """Неизвестность остаётся неизвестностью: вето должно опираться на измерение."""
    weak = science(300, 4, 2, 40)
    assert maturity(state(oa=weak, wiki_error=True))[0] < MATURITY_VETO


def test_measured_maturity_still_vetoes_when_another_source_is_unknown():
    strong = science(2900, 12, 8, 500)
    assert maturity(state(oa=strong, wiki_error=True))[0] >= MATURITY_VETO


def test_unknown_wikipedia_keeps_the_conservative_floor_and_says_so():
    r, reasons, unknown = maturity(state(oa=science(5, 1, 1, 0), wiki_error=True))
    assert unknown and r >= 0.5
    assert any("Википедия не ответила" in x for x in reasons)


def test_all_science_unknown_leaves_only_the_encyclopedia_signal():
    s = state(oa_error=True, wiki=article(7))
    assert maturity(s)[0] >= MATURITY_VETO          # статья старше пяти лет остаётся дискретным признаком
    s2 = state(oa_error=True, wiki=NO_ARTICLE)
    assert maturity(s2)[0] < MATURITY_VETO


# ---------- границы ----------

def test_risk_is_bounded_to_the_unit_interval():
    extreme = risk(oa=science(10 ** 6, 40, 30, 10 ** 5), wiki=article(30), gh={"gh_total": 10 ** 6})
    assert 0.0 <= extreme <= 1.0


def test_empty_evidence_is_unknown_not_immature_and_not_a_veto():
    """Пустое состояние это «ничего не известно»: консервативный floor, но не вето."""
    r, reasons, unknown = maturity(EvidenceState())
    assert unknown and r == UNKNOWN_MATURITY_RISK and r < MATURITY_VETO
