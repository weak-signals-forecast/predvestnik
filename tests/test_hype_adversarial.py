"""Состязательная проверка логики хайпа (REPAIR 8).

На трёх живых запросах R1 вето HYPE не сработало ни разу. Это не доказывает ни того, что логика
работает, ни того, что она сломана: кандидаты с такими профилями просто не дошли до третьей ступени.

Здесь построены синтетические профили, каждый из которых по построению описывает свой сценарий
маркетингового шума, и контрольные профили, которые сработать не должны. Пороги под эти фикстуры
НЕ подбирались: правила и их числа зафиксированы в `decision.py` до написания этого файла, а сам
файл не вводит новых порогов.

Это проверка логики на синтетике. **Живой калибровкой она не является** и о частоте хайпа в реальной
выдаче ничего не говорит.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals.decision import (ACCEPT, HYPE, HYPE_VETO, LOW_EVIDENCE, MATURE,  # noqa: E402
                               SEMANTICALLY_VERIFIED, decide as _decide)


def decide(state, **kw):
    kw.setdefault("verification", SEMANTICALLY_VERIFIED)
    return _decide(state, **kw)
from wsignals.evidence import NO_RESULTS, OK, EvidenceState, Observation  # noqa: E402
from wsignals.schema import Document  # noqa: E402
from wsignals.trust import assess as assess_trust  # noqa: E402


def _doc(title, url, source_type, source_name="s"):
    _d = assess_trust(Document(title=title, url=url, source_name=source_name, source_type=source_type,
                                 published="2026-03-01", language="en")).with_provenance()
    # P1-C. Признак претензионной релевантности читается строго: подтверждает только явное
    # True. Эта фикстура ПО ПОСТРОЕНИЮ представляет документ про рассматриваемого кандидата,
    # поэтому помечается явно. Неразмеченные документы проверяются отдельно тестом,
    # доказывающим, что они не открывают ACCEPT.
    _d.claim_relevant = True
    return _d


DOCS = [_doc("paper", "https://doi.org/10.1/x", "научная публикация", "Nature"),
        _doc("repo", "https://github.com/a/b", "репозиторий", "GitHub")]


def state(openalex, github, news, hn, wiki) -> EvidenceState:
    s = EvidenceState()
    for name, values in (("openalex", openalex), ("github", github), ("news", news), ("hn", hn), ("wiki", wiki)):
        s.add(Observation(name, OK if any(values.values()) else NO_RESULTS, values))
    return s


YOUNG = {"oa_total": 40, "oa_growth": 1.8, "oa_age": 2, "oa_last_share": 0.45,
         "oa_preprint_share": 0.3, "oa_peak_ratio": 1.2}
CODE = {"gh_total": 30, "gh_12m": 25, "gh_new_share": 0.8}
QUIET_NEWS = {"news_1y": 8, "news_30d": 3, "news_30d_share": 0.37, "news_press_share": 0.1}
CHATTER = {"hn_total": 6, "hn_12m": 5, "hn_growth": 1.1}
NO_WIKI = {"wiki_article": 0, "wiki_age": 0}


# ---------- пять состязательных профилей хайпа ----------

HYPE_FIXTURES = {
    # 1. Поток пресс-релизов: почти все новости написаны самими компаниями.
    "press_release_flood": state(
        YOUNG, CODE, {"news_1y": 40, "news_30d": 20, "news_30d_share": 0.5, "news_press_share": 0.85},
        CHATTER, NO_WIKI),
    # 2. Внимание медиа без научного и инженерного следа: говорят все, делает никто.
    "media_without_substance": state(
        {"oa_total": 4, "oa_growth": 2.0, "oa_age": 1, "oa_last_share": 0.9, "oa_preprint_share": 0.5,
         "oa_peak_ratio": 1.5},
        {"gh_total": 2, "gh_12m": 2, "gh_new_share": 1.0},
        {"news_1y": 90, "news_30d": 40, "news_30d_share": 0.44, "news_press_share": 0.2},
        CHATTER, NO_WIKI),
    # 3. Угасшая волна: пик пройден несколько лет назад, публикаций втрое меньше пикового года.
    "faded_wave": state(
        {"oa_total": 400, "oa_growth": -0.4, "oa_age": 7, "oa_last_share": 0.05, "oa_preprint_share": 0.1,
         "oa_peak_ratio": 0.2, "oa_peak_year": 2021},
        CODE, QUIET_NEWS, CHATTER, NO_WIKI),
    # 4. Запуск на пресс-релизах: половина ленты это анонсы.
    "launch_announcement_wave": state(
        YOUNG, CODE, {"news_1y": 60, "news_30d": 35, "news_30d_share": 0.58, "news_press_share": 0.55},
        CHATTER, NO_WIKI),
    # 5. Угасание плюс маркетинг: обе причины сразу.
    "faded_and_marketed": state(
        {"oa_total": 200, "oa_growth": -0.2, "oa_age": 6, "oa_last_share": 0.08, "oa_preprint_share": 0.1,
         "oa_peak_ratio": 0.3, "oa_peak_year": 2022},
        CODE, {"news_1y": 50, "news_30d": 25, "news_30d_share": 0.5, "news_press_share": 0.6},
        CHATTER, NO_WIKI),
}

# Контроль: профили, на которых вето хайпа срабатывать не должно.
CONTROL_FIXTURES = {
    "healthy_emerging": (state(YOUNG, CODE, QUIET_NEWS, CHATTER, NO_WIKI), ACCEPT),
    "mature_technology": (state({**YOUNG, "oa_total": 9000, "oa_age": 12}, CODE, QUIET_NEWS, CHATTER,
                                {"wiki_article": 1, "wiki_age": 11}), MATURE),
    "quiet_science_only": (state(YOUNG, {"gh_total": 0, "gh_12m": 0, "gh_new_share": 0.0},
                                 {"news_1y": 0, "news_30d": 0, "news_30d_share": 0.0, "news_press_share": 0.0},
                                 {"hn_total": 0, "hn_12m": 0, "hn_growth": 0.0}, NO_WIKI), LOW_EVIDENCE),
}


@pytest.mark.parametrize("name", sorted(HYPE_FIXTURES))
def test_hype_fixture_is_detected(name):
    d = decide(HYPE_FIXTURES[name], burst=3.0, documents=DOCS, legacy_probability=0.9)
    assert d.decision == HYPE, (name, d.decision, d.decision_reasons)
    assert d.hype_risk >= HYPE_VETO
    assert d.decision_reason_codes == ["hype_veto"]
    assert d.decision_reasons, name


@pytest.mark.parametrize("name", sorted(CONTROL_FIXTURES))
def test_control_fixture_is_not_called_hype(name):
    st, expected = CONTROL_FIXTURES[name]
    d = decide(st, burst=3.0, documents=DOCS, legacy_probability=0.9)
    assert d.decision != HYPE, (name, d.decision_reasons)
    assert d.decision == expected, (name, d.decision)


def test_hype_risk_is_monotone_in_press_release_share():
    """Ось хайпа не бинарная: она растёт вместе с долей пресс-релизов."""
    risks = []
    for share in (0.0, 0.2, 0.4, 0.6, 0.9):
        st = state(YOUNG, CODE, {"news_1y": 20, "news_30d": 8, "news_30d_share": 0.4, "news_press_share": share},
                   CHATTER, NO_WIKI)
        risks.append(decide(st, burst=3.0, documents=DOCS).hype_risk)
    assert risks == sorted(risks) and risks[0] < risks[-1]


def test_hype_is_not_inferred_when_the_news_adapter_failed():
    """Отказ ленты новостей не превращается ни в хайп, ни в его отсутствие."""
    st = EvidenceState()
    st.add(Observation("openalex", OK, YOUNG))
    st.add(Observation("github", OK, CODE))
    st.add(Observation("news", "ERROR", error_type="timeout"))
    st.add(Observation("hn", OK, CHATTER))
    st.add(Observation("wiki", NO_RESULTS, NO_WIKI))
    d = decide(st, burst=3.0, documents=DOCS)
    assert d.hype_risk == 0.0 and d.decision != HYPE


def test_fixtures_cover_distinct_rules():
    """Пять профилей должны срабатывать по разным причинам, а не по одной и той же."""
    reasons = {name: decide(st, burst=3.0, documents=DOCS).decision_reasons[0]
               for name, st in HYPE_FIXTURES.items()}
    assert len(set(reasons.values())) >= 3, reasons
