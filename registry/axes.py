"""Оси слабости сигнала: измеримые признаки поверх рядов реестра (counts.npz, totals.json, business_series.npz).

    python -m registry.axes                   # -> <registry>/axes.parquet для всех фраз реестра
    REGISTRY_CUTOFF=2021 GPB_REGISTRY=~/gpb_corpus/registry_2021 python -m registry.axes

Зачем. Модели реестра учат «растёт ли фраза и не массовая ли она». Эксперты заказчика хотят другого: «новое ли это
знание, а не улучшение известного». Оси ниже измеряют именно это, числами из тех же рядов, без языковой модели:

  родитель (parent_*)     зрелый термин, от которого фраза произошла: хвост фразы или фраза без первого/последнего
                          слова, найденный в таблице фраз, появившийся не позже cutoff−PARENT_AGE и с объёмом
                          ≥ PARENT_MIN в окне среза. incremental_lex = log2(объём родителя / объём фразы):
                          чем больше, тем вероятнее «улучшение известного». На срезе 2021 кандидаты с зрелым
                          родителем взлетали к 2024–2026 в 2–3 раза реже (проверки/2026-10-18_ось_родитель.md).
                          Ограничение: однословных терминов в таблице фраз почти нет, такие родители не находятся.
  стадия S-кривой         accel — ускорение роста доли (разность годовых log-приростов), peak_ratio — доля
  (stage_*)               последнего года к пику (< 1: тема прошла пик), recent_share — какая часть всех работ фразы
                          пришлась на последние три года (1 — всё новое), docs_now — объём в окне среза.
  переход слоёв           год первого следа в науке, на Hacker News, у стартапов YC и в прессе; лаги между слоями;
  (layer_*)               only_science — есть в науке и больше нигде (самый ранний наблюдаемый момент).
  стандарт (is_standard)  правило по форме названия: версия, протокол, спецификация, консорциум. По ТЗ стандарты
                          и протоколы — не слабые сигналы.

Все окна считаются от года среза (машина времени): признаки на срез 2021 не видят данных после 2021.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import CORPUS, CUTOFF, REGISTRY, SOURCES, YEARS

PARENT_MIN, PARENT_AGE = 200, 4
AXES = ["incremental_lex", "parent_docs_log", "stage_accel", "stage_peak_ratio", "stage_recent_share",
        "layer_count", "layer_only_science", "layer_lag_hn", "layer_lag_market", "is_standard"]
LABELS = {"incremental_lex": "Зрелость родителя (улучшение известного)", "parent_docs_log": "Объём родительской темы",
          "stage_accel": "Ускорение роста (стадия S-кривой)", "stage_peak_ratio": "Отношение к пику (угасание)",
          "stage_recent_share": "Доля работ за последние три года", "layer_count": "Сколько слоёв достигнуто",
          "layer_only_science": "Только в науке (самая ранняя стадия)", "layer_lag_hn": "Лаг наука → сообщество, лет",
          "layer_lag_market": "Лаг наука → рынок (YC, пресса), лет", "is_standard": "Стандарт или протокол"}
STANDARD = re.compile(
    r"\b(v?\d+\.\d+|version \d|rfc ?\d+|ieee ?\d|iso ?\d|iec ?\d|3gpp|etsi|w3c|oasis|ietf)\b|"
    r"\b(protocol|protocols|standard|standards|specification|spec|interconnect express|ethernet|wi-?fi \d|"
    r"bluetooth|usb|hdmi|pcie|pci express|nvlink|cxl|ucie|uefi|onnx|openapi|a2a|mcp)\b", re.I)


def _share_series(df: np.ndarray, eids: np.ndarray, totals: dict, years: list[int]) -> np.ndarray:
    """Доля документов (OpenAlex + arXiv) по годам, [фраза, год]."""
    oa, ax = SOURCES.index("openalex"), SOURCES.index("arxiv")
    yi = {y: i for i, y in enumerate(YEARS)}
    out = np.zeros((len(eids), len(years)), dtype=np.float64)
    for j, y in enumerate(years):
        n = totals.get(f"openalex|{y}", 0) + totals.get(f"arxiv|{y}", 0)
        out[:, j] = (df[eids, oa, yi[y]] + df[eids, ax, yi[y]]) / max(n, 1)
    return out


def _docs(df: np.ndarray, eids: np.ndarray, years) -> np.ndarray:
    oa, ax = SOURCES.index("openalex"), SOURCES.index("arxiv")
    yi = {y: i for i, y in enumerate(YEARS)}
    return sum(df[eids, oa, yi[y]] + df[eids, ax, yi[y]] for y in years)


def parent_axis(a: pd.DataFrame, docs_now: np.ndarray, cutoff: int) -> pd.DataFrame:
    """Лексический родитель каждой фразы. a — таблица фраз с колонками phrase, first_year; docs_now — объём в окне."""
    by_phrase = {p: i for i, p in enumerate(a.phrase.values)}
    fy = a.first_year.values
    mature = (fy <= cutoff - PARENT_AGE) & (docs_now >= PARENT_MIN)

    def find(phrase: str):
        w = phrase.split()
        if len(w) < 2:
            return -1
        cands = [" ".join(w[-k:]) for k in (1, 2) if len(w) > k] + [" ".join(w[1:]), " ".join(w[:-1])]
        best = -1
        for c in dict.fromkeys(cands):
            j = by_phrase.get(c, -1)
            if j >= 0 and mature[j] and (best < 0 or docs_now[j] > docs_now[best]):
                best = j
        return best

    par = np.fromiter((find(p) for p in a.phrase.values), dtype=np.int64, count=len(a))
    has = par >= 0
    pdocs = np.where(has, docs_now[np.where(has, par, 0)], 0).astype(float)
    return pd.DataFrame({
        "parent": np.where(has, a.phrase.values[np.where(has, par, 0)], None),
        "parent_docs_log": np.where(has, np.log1p(pdocs), np.nan),
        "incremental_lex": np.where(has, np.log2((pdocs + 1) / (docs_now + 1)), np.nan),
    }, index=a.index)


def stage_axis(share: np.ndarray, docs_all: np.ndarray, docs_now: np.ndarray) -> pd.DataFrame:
    """share — доли по годам до среза включительно (последний столбец = год среза)."""
    s = share + 1e-7
    g1 = np.log2(s[:, -1] / s[:, -2])                      # прирост последнего года
    g2 = np.log2(s[:, -2] / s[:, -3])                      # прирост предыдущего
    peak = share[:, :-1].max(axis=1)                       # пик по завершённым годам
    return pd.DataFrame({
        "stage_accel": g1 - g2,
        "stage_peak_ratio": np.where(peak > 0, share[:, -1] / np.where(peak > 0, peak, 1), np.nan),
        "stage_recent_share": np.where(docs_all > 0, docs_now / np.maximum(docs_all, 1), np.nan),
    })


def layer_axis(keys: pd.Series, sci_first: np.ndarray, reg: Path, cutoff: int) -> pd.DataFrame:
    """Первый год в каждом слое по бизнес-рядам (hn, yc, press) не позже среза; лаги относительно науки."""
    p = reg / "business_series.npz"
    if not p.exists():                                  # у реестра-среза своих бизнес-рядов нет: берём полные и режем по году
        p = CORPUS / "registry_v3" / "business_series.npz"
    n = len(keys)
    out = {"layer_count": np.ones(n), "layer_only_science": np.ones(n), "layer_lag_hn": np.full(n, np.nan),
           "layer_lag_market": np.full(n, np.nan)}
    if not p.exists():
        return pd.DataFrame(out, index=keys.index)
    b = np.load(p, allow_pickle=True)
    idx = {k: i for i, k in enumerate(b["keys"])}
    src = [str(s) for s in b["sources"]]
    years = np.array([int(y) for y in b["years"]])
    ok_years = years <= cutoff
    df = b["df"][:, :, ok_years]
    ys = years[ok_years]

    def first_year(row: np.ndarray) -> float:
        nz = np.nonzero(row)[0]
        return float(ys[nz[0]]) if len(nz) else np.nan

    rows = [idx.get(k) for k in keys]
    for n_, (i, sf) in enumerate(zip(rows, sci_first)):
        if i is None:
            continue
        hn = first_year(df[i, src.index("hn")])
        yc = first_year(df[i, src.index("yc")])
        pr = first_year(df[i, src.index("press")])
        reached = [x for x in (hn, yc, pr) if x == x]
        out["layer_count"][n_] = 1 + len(reached)
        out["layer_only_science"][n_] = float(not reached)
        if hn == hn and sf == sf:
            out["layer_lag_hn"][n_] = hn - sf
        market = [x for x in (yc, pr) if x == x]
        if market and sf == sf:
            out["layer_lag_market"][n_] = min(market) - sf
    return pd.DataFrame(out, index=keys.index)


def compute(reg: Path = REGISTRY, cutoff: int = CUTOFF) -> pd.DataFrame:
    a = pd.read_parquet(reg / "all_phrases.parquet", columns=["phrase", "key", "eid", "first_year"])
    df = np.load(reg / "counts.npz")["df"]
    totals = json.loads((reg / "totals.json").read_text(encoding="utf-8"))
    eids = a.eid.values.astype(int)
    years = [y for y in YEARS if y <= cutoff]
    now = tuple(range(cutoff - 2, cutoff + 1))
    docs_now = _docs(df, eids, now).astype(float)
    docs_all = _docs(df, eids, years).astype(float)
    share = _share_series(df, eids, totals, years)
    out = pd.concat([a[["key", "phrase"]], parent_axis(a, docs_now, cutoff),
                     stage_axis(share, docs_all, docs_now).set_index(a.index),
                     layer_axis(a.key, a.first_year.values.astype(float), reg, cutoff)], axis=1)
    out["is_standard"] = a.phrase.map(lambda p: float(bool(STANDARD.search(p.replace("_", " "))))).values
    out["docs_now"] = docs_now
    return out


def load(reg: Path = REGISTRY) -> pd.DataFrame:
    """Оси для склейки с таблицами моделей по key (считаются один раз, python -m registry.axes)."""
    p = reg / "axes.parquet"
    if not p.exists():
        compute(reg).to_parquet(p)
    return pd.read_parquet(p).drop_duplicates("key")


def main() -> int:
    out = compute()
    out.to_parquet(REGISTRY / "axes.parquet")
    c = out
    print(f"оси для {len(c):,} фраз -> {REGISTRY / 'axes.parquet'}".replace(",", " "))
    print(f"  с зрелым родителем {c.parent.notna().mean():.1%}; стандартов {c.is_standard.mean():.2%}; "
          f"только в науке {c.layer_only_science.mean():.1%}; прошли пик (peak_ratio < 0,5) "
          f"{(c.stage_peak_ratio < 0.5).mean():.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
