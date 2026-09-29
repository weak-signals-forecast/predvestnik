"""Признаки технологии по открытым источникам. Одна функция для обучения и для открытого запроса.

Каждая группа признаков отвечает на свой вопрос ТЗ:
  наука     OpenAlex: есть ли исследовательская база, растёт ли, прошла ли пик, кто пишет (компании, препринты)
  медиа     Google News: насколько тема уже в новостях, доля пресс-релизов (маркетинговый шум)
  сообщество Hacker News: интерес разработчиков сейчас против двух предыдущих лет
  код       GitHub: есть ли реализация, насколько свежая
  зрелость  Википедия: есть ли отдельная статья, её возраст и популярность (признак мейнстрима)
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

from .sources import github, hackernews, news, openalex, wikipedia

SOURCES = {"openalex": openalex.stats, "news": news.stats, "hn": hackernews.stats,
           "github": github.stats, "wiki": wikipedia.stats}

FEATURES = [
    "oa_total_log", "oa_recent_log", "oa_growth", "oa_age", "oa_peak_ratio", "oa_last_share",
    "oa_preprint_share", "oa_company_share", "oa_fields",
    "news_1y", "news_30d_share", "news_sources", "news_press_share",
    "hn_12m_log", "hn_growth",
    "gh_total_log", "gh_new_share", "gh_max_stars_log",
    "wiki_article", "wiki_age", "wiki_views_log", "wiki_mentions_log",
]

NAMES_RU = {
    "oa_total_log": "объём научных публикаций", "oa_recent_log": "публикаций за 2024–2026",
    "oa_growth": "рост публикаций к 2019–2021", "oa_age": "лет с появления в науке",
    "oa_peak_ratio": "близость к пику публикаций", "oa_last_share": "доля публикаций последнего года",
    "oa_preprint_share": "доля препринтов", "oa_company_share": "доля авторов из компаний",
    "oa_fields": "число научных областей", "news_1y": "новостей за год",
    "news_30d_share": "доля свежих новостей за месяц", "news_sources": "число изданий",
    "news_press_share": "доля пресс-релизов", "hn_12m_log": "обсуждений разработчиков за год",
    "hn_growth": "рост обсуждений разработчиков", "gh_total_log": "число репозиториев",
    "gh_new_share": "доля репозиториев за последний год", "gh_max_stars_log": "популярность главного репозитория",
    "wiki_article": "есть статья в Википедии", "wiki_age": "возраст статьи в Википедии",
    "wiki_views_log": "просмотры статьи в Википедии", "wiki_mentions_log": "упоминания в Википедии",
}


def derive(raw: dict) -> dict:
    """Сырые счётчики источников -> признаки модели (логарифмы, доли, темпы)."""
    g = lambda k, d=0.0: raw.get(k, d) if raw.get(k) is not None else d
    return {
        "oa_total_log": math.log1p(g("oa_total")), "oa_recent_log": math.log1p(g("oa_recent")),
        "oa_growth": g("oa_growth"), "oa_age": g("oa_age"), "oa_peak_ratio": g("oa_peak_ratio", 1.0),
        "oa_last_share": g("oa_last_share"), "oa_preprint_share": g("oa_preprint_share"),
        "oa_company_share": g("oa_company_share"), "oa_fields": g("oa_fields"),
        "news_1y": g("news_1y"), "news_30d_share": g("news_30d") / max(1.0, g("news_1y")),
        "news_sources": g("news_sources"), "news_press_share": g("news_press_share"),
        "hn_12m_log": math.log1p(g("hn_12m")),
        "hn_growth": math.log((g("hn_12m") + 1) / (g("hn_prior24m") / 2 + 1)),
        "gh_total_log": math.log1p(g("gh_total")), "gh_new_share": g("gh_new_share"),
        "gh_max_stars_log": g("gh_max_stars_log"),
        "wiki_article": g("wiki_article"), "wiki_age": g("wiki_age"),
        "wiki_views_log": g("wiki_views_log"), "wiki_mentions_log": g("wiki_mentions_log"),
    }


def collect_state(query: str):
    """Опрашивает все источники параллельно и возвращает (сырые счётчики, состояние доказательства).

    Статус каждого источника явный (PHASE 1): OK, NO_RESULTS или ERROR. Значения отказавшего источника
    в счётчики не попадают — ноль вместо ошибки запрещён, потому что для модели «нет следа» это довод
    в пользу слабого сигнала."""
    from .evidence import EvidenceState, observe

    state = EvidenceState()
    with ThreadPoolExecutor(len(SOURCES)) as ex:
        futs = {name: ex.submit(observe, name, fn, query) for name, fn in SOURCES.items()}
        for name, f in futs.items():
            state.add(f.result())
    return state.values(), state


def collect(query: str) -> dict:
    """Признаки технологии. Ошибка одного источника не роняет сбор: она сохраняется в колонках
    `*_status` и `*_error`, а его счётчики в данные не попадают вовсе."""
    raw, state = collect_state(query)
    raw = dict(raw)
    years = raw.pop("oa_years", None)
    out = {**raw, **derive(raw)}
    for name, obs in state.observations.items():
        out[f"{name}_status"] = obs.status
        if obs.status == "ERROR":
            out[f"{name}_error"] = f"{obs.error_type}: {obs.error}"[:200]
    if years:
        out["oa_years"] = " ".join(map(str, years))
    return out
