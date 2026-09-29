"""Тесты сервиса слабых сигналов: без сети и без данных.

    python -m pytest -q tests/test_wsignals.py
"""
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import trust  # noqa: E402
from wsignals.dataset import stage_to_ordinal  # noqa: E402
from wsignals.features import FEATURES, NAMES_RU, derive  # noqa: E402
from wsignals.schema import Document  # noqa: E402
from wsignals.timemachine import YEARS, ladder, oa_at, pit_features  # noqa: E402


# ---------- машина времени: признаки на дату не видят будущее ----------

def _row(series, **kw):
    base = {"oa_years": " ".join(map(str, series)), "gh_total": 50, "gh_12m": 20, "hn_12m": 3, "hn_prior24m": 2, "hn_total": 9,
            "wiki_article": 0, "wiki_age": 0, "news_1y": 5}
    base.update({f"gh_before_{y}": v for y, v in zip([2016, 2019, 2021, 2022, 2024], [0, 2, 5, 9, 30])})
    base.update({f"hn_before_{y}": v for y, v in zip([2012, 2016, 2019, 2020, 2021, 2022, 2024], [0, 0, 1, 1, 2, 4, 6])})
    base.update(kw)
    return pd.Series(base)


def test_oa_features_at_freeze_ignore_later_years():
    a = [0, 0, 0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    b = a[: YEARS.index(2021) + 1] + [0] * (len(YEARS) - YEARS.index(2021) - 1)
    assert oa_at(a, 2021) == oa_at(b, 2021)


def test_pit_features_ignore_post_freeze_counts():
    r1 = _row([1] * len(YEARS))
    r2 = _row([1] * len(YEARS), gh_before_2024=99999, hn_before_2024=99999, gh_total=10**6, hn_12m=500)
    assert pit_features(r1, 2021) == pit_features(r2, 2021)


def test_wikipedia_article_created_after_freeze_does_not_exist_then():
    r = _row([5] * len(YEARS), wiki_article=1, wiki_age=2)          # статья создана в 2024
    assert pit_features(r, 2021)["wiki_article"] == 0.0
    assert pit_features(r, 2026)["wiki_article"] == 1.0


def test_ladder_counts_layers():
    r = _row([0] * 12 + [4, 6, 9, 12, 20], wiki_article=0)
    lad = ladder(r)
    assert lad["first_наука"] == 2022 and lad["first_энциклопедия"] is None
    assert 1 <= lad["ladder_step"] <= 4


# ---------- разметка ----------

def test_stage_normalization_handles_free_text():
    assert stage_to_ordinal("Концепция/Исследование") == 1
    assert stage_to_ordinal("Прототип/PoC → Пилот") == 2.5
    assert stage_to_ordinal("Пилот → раннее внедрение") == 3.5
    assert stage_to_ordinal("Раннее внедрение (у лидера) / Прототип (у остальных)") == 3.0
    assert math.isnan(stage_to_ordinal("непонятно"))


def test_all_features_have_russian_names_and_derive_is_total():
    assert set(FEATURES) <= set(NAMES_RU)
    out = derive({})
    assert set(FEATURES) <= set(out) and all(math.isfinite(v) for v in out.values())


# ---------- доверие источникам ----------

def doc(url, source_name="", source_type="новости"):
    return trust.assess(Document(title="t", url=url, source_name=source_name, source_type=source_type, published=None, language="en"))


def test_trust_levels():
    assert doc("https://arxiv.org/abs/1", source_type="препринт").trust == "высокий"
    assert doc("https://www.nature.com/articles/x").trust == "высокий"
    assert doc("https://www.cbr.ru/press/").trust == "высокий"
    assert doc("https://habr.com/ru/articles/1/", "Хабр").trust == "средний"
    assert doc("https://www.prnewswire.com/news/x", "PR Newswire").trust == "пониженный"
    assert doc("https://news.ycombinator.com/item?id=1", "Hacker News", "сообщество").trust == "пониженный"


# ---------- открытый запрос: разбор и фразы ----------

def test_parse_query_maps_russian_directions(monkeypatch):
    from wsignals import query
    monkeypatch.setattr(query.llm, "translate_query", lambda t: f"<{t}>")
    assert query.parse_query("слабые сигналы в кибербезопасности")[:2] == ["cybersecurity", "AI security"]
    assert "fintech" in query.parse_query("перспективные решения в финтехе")
    assert query.parse_query("quantum sensing") == ["quantum sensing"]


def test_phrases_skip_stopword_edges_and_generic_heads():
    from wsignals.query import _phrases
    ph = _phrases("A new framework for agent identity and scoped permissions in security systems")
    assert "agent identity" in ph and "scoped permissions" in ph
    assert not any(p.startswith("a ") or p.endswith(" security") for p in ph)


def test_russian_plural_forms():
    from wsignals.query import plural
    assert plural(1, "работа", "работы", "работ") == "1 работа"
    assert plural(3, "работа", "работы", "работ") == "3 работы"
    assert plural(11, "работа", "работы", "работ") == "11 работ"
    assert plural(22, "работа", "работы", "работ") == "22 работы"


def test_standards_are_recognized():
    from wsignals.query import STANDARD
    for ph in ("owasp llm top", "nist ai rmf", "eu ai act", "pci dss", "iso 20022", "mitre atlas"):
        assert STANDARD.search(ph), ph
    for ph in ("agentic payments", "mcp security", "rag poisoning"):
        assert not STANDARD.search(ph), ph


# ---------- русские названия ----------

def test_russian_names_are_composed_with_agreement():
    from wsignals.ru import technology_name
    cases = {"rag poisoning": "отравление RAG", "agent skills": "навыки агентов", "context compression": "сжатие контекста",
             "quantum sensing": "квантовая сенсорика", "quantum computing": "квантовые вычисления",
             "agentic payments": "агентные платежи", "decentralized kyc": "децентрализованный KYC"}
    for eng, ru_name in cases.items():
        got = technology_name(eng)
        assert got["ru"] == ru_name, (eng, got["ru"])
        assert got["original"] == eng


def test_russian_name_keeps_original_when_unknown():
    from wsignals.ru import technology_name
    got = technology_name("zzz qqq")
    assert got["ru"] == "zzz qqq" and got["source"].startswith("название оставлено")


def test_glossary_match_rejects_different_head_word():
    from wsignals.ru import technology_name
    # в таблице есть «agent memory poisoning»: это другая технология, подставлять её название нельзя
    assert technology_name("agent memory")["ru"] == "память агентов"


# ---------- предсказатель взлёта и фильтр штампов ----------

def test_takeoff_label_uses_only_future_window():
    from wsignals.takeoff import label
    flat = " ".join(["10"] * len(YEARS))
    assert label({"oa_years": flat}, 2019)[0] == 0
    growth = [0] * len(YEARS)
    for i, y in enumerate(YEARS):
        growth[i] = 2 if y <= 2019 else 40
    lab, ratio = label({"oa_years": " ".join(map(str, growth))}, 2019)
    assert lab == 1 and ratio > 2


def test_takeoff_label_requires_minimum_volume():
    from wsignals.takeoff import label
    tiny = [0] * len(YEARS)
    for i, y in enumerate(YEARS):
        tiny[i] = 0 if y <= 2019 else 2      # рост «в бесконечность», но всего 6 работ
    assert label({"oa_years": " ".join(map(str, tiny))}, 2019)[0] == 0


def test_keyness_separates_stamps_from_terms():
    from collections import Counter
    from wsignals.query import KEYNESS_MIN, keyness
    bg = Counter({"have explored": 60, "agent memory": 0})
    assert keyness("have explored", 20, 600, 1800, bg) < KEYNESS_MIN      # так же часто, как в фоне
    assert keyness("agent memory", 12, 600, 1800, bg) > KEYNESS_MIN       # в фоне почти не встречается
