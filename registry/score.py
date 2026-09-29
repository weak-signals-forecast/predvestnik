"""Сборка лексического реестра, шаг 2: из рядов по годам — кандидаты в слабые сигналы.

    python -m registry.score          # -> <corpus>/registry/registry.parquet и registry_top.csv

Все пороги ниже зафиксированы ДО первого сравнения со списком организаторов и подбирались только по
общему виду выдачи. Сравнение с 100 сигналами — отдельный скрипт registry/evaluate.py.

Считаем ДОЛИ, а не штуки: старые годы OpenAlex скачаны не полностью, поэтому частота фразы в году делится
на число прочитанных документов того же источника и года. Источники считаются отдельно (OpenAlex и arXiv
частично дублируют друг друга) и объединяются уже на уровне показателей.

Окна: свежее R = 2024–2026, база B = 2019–2022, ранее E = 2012–2018.

Показатели фразы:
  volume_recent   документов в R (OpenAlex + arXiv)
  growth          средний по источникам log2 роста доли R к B, со сглаживанием +1 документ
  first_year      первый год, когда фраза встретилась хотя бы в 2 документах
  max_rate_recent наибольшая по источникам доля документов R с этой фразой (мера массовости)
  spread          нормированная энтропия частоты фразы по группам (подобласти OpenAlex, категории arXiv)
                  за R; близко к 1 — фраза размазана по всем темам: штамп или общее слово
  sources_recent  в скольких источниках (OpenAlex, arXiv, HF Papers) фраза есть в R

Кандидат в слабые сигналы:
  volume_recent >= MIN_VOLUME              есть на что опереться
  growth >= MIN_GROWTH или first_year >= NEW_SINCE    растёт или появилась недавно
  max_rate_recent <= MAX_RATE              ещё не массовая
  spread <= MAX_SPREAD                     не штамп
  sources_recent >= MIN_SOURCES            подтверждена хотя бы двумя независимыми источниками
  фраза не является обрывком более длинной (C-value): нет надфразы, в которой она встречается почти всегда
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import CORPUS, CUTOFF, SOURCES, YEARS, REGISTRY

REG = REGISTRY
# окна считаются от года среза (машина времени): при срезе 2026 — R 2024–2026, B 2019–2022
R = tuple(range(CUTOFF - 2, CUTOFF + 1))
B = tuple(range(CUTOFF - 7, CUTOFF - 3))
E = tuple(range(2012, CUTOFF - 7))
MIN_VOLUME = 5
import os
GROWTH_SOURCES = tuple(os.environ.get("REGISTRY_GROWTH", "openalex,arxiv").split(","))
MIN_GROWTH = 1.0          # доля выросла минимум вдвое
NEW_SINCE = CUTOFF - 4
MAX_RATE = 1e-3           # фраза есть больше чем в 0,1 % свежих работ источника — уже массовая тема
MAX_SPREAD = 0.80
MIN_SOURCES = 2
WEAK_VOLUME = 200         # выше этого объёма тема уже скорее «горячая», чем «слабая»: мягкий штраф
MIN_SPREAD_DOCS = 20      # энтропию считаем, только когда документов по группам достаточно
FRAGMENT_SHARE = 0.8      # обрывок: надфраза покрывает не меньше 80 % его документов
MIN_GROUP_DOCS = 2000     # группы меньше этого размера в энтропию не берём: шумят

# Общие слова: фраза целиком из них — не название технологии.
GENERIC = set("""
ai artificial intelligence machine learning deep neural network networks model models system systems data method methods
approach framework algorithm algorithms learning based analysis application applications technology technologies research
study results performance proposed paper information computer science computing process processing management design
development problem problems task tasks time real use using case new novel different various high low large small
""".split())


def _years_idx(years):
    return [YEARS.index(y) for y in years]


def load():
    d = np.load(REG / "counts.npz")
    g = np.load(REG / "groups.npz")
    vocab = [l for l in (REG / "entities.txt").read_text(encoding="utf-8").split("\n") if l]   # названия сущностей
    totals = json.loads((REG / "totals.json").read_text(encoding="utf-8"))
    return vocab, d["df"], g["grp"], [str(x) for x in g["groups"]], totals


def score() -> pd.DataFrame:
    vocab, df, grp, groups, totals = load()
    V = len(vocab)
    N = np.array([[totals.get(f"{s}|{y}", 0) for y in YEARS] for s in SOURCES], dtype=np.float64)  # [S, Y]
    iR, iB = _years_idx(R), _years_idx(B)
    oa, ax, hf = SOURCES.index("openalex"), SOURCES.index("arxiv"), SOURCES.index("hf")

    dR = df[:, :, iR].sum(axis=2).astype(np.float64)       # [V, S]
    dB = df[:, :, iB].sum(axis=2).astype(np.float64)
    nR, nB = N[:, iR].sum(axis=1), N[:, iB].sum(axis=1)     # [S]

    growth_s = np.full((V, len(SOURCES)), np.nan)
    for s in (oa, ax):
        if nR[s] > 0 and nB[s] > 0:
            g = np.log2(((dR[:, s] + 1) / nR[s]) / ((dB[:, s] + 1) / nB[s]))
            measured = (dR[:, s] + dB[:, s]) >= 3
            growth_s[measured, s] = g[measured]
    # по каким источникам считать рост: старые годы OpenAlex скачаны неравномерно по подобластям, и смена
    # состава между окнами выдаёт себя за рост (реестр-2021 без этого — сплошь финансы и экономика здоровья);
    # arXiv скачан целиком с 2012 года, его состав стабилен
    gsrc = [SOURCES.index(x) for x in GROWTH_SOURCES]
    growth = np.nanmean(growth_s[:, gsrc], axis=1)             # предупреждение numpy о пустых строках безвредно

    rate_R = np.zeros((V, len(SOURCES)))
    for s in range(len(SOURCES)):
        if nR[s] > 0:
            rate_R[:, s] = dR[:, s] / nR[s]
    max_rate = rate_R[:, [oa, ax]].max(axis=1)
    volume = dR[:, oa] + dR[:, ax]

    both = df[:, oa, :] + df[:, ax, :]                         # [V, Y]
    both = both * (np.array(YEARS) <= CUTOFF)                  # после года среза данных «ещё нет»
    has2 = both >= 2
    first_year = np.where(has2.any(axis=1), np.array(YEARS)[has2.argmax(axis=1)], 0)
    sources_recent = (dR[:, [oa, ax, hf]] > 0).sum(axis=1)

    # энтропия по группам: частоты, нормированные на размер группы, чтобы большие группы не перевешивали
    gsize = np.array([sum(totals.get(f"{s}|{y}|{g}", 0) for s in SOURCES for y in R) for g in groups], dtype=np.float64)
    keep_g = gsize >= MIN_GROUP_DOCS
    rates = grp[keep_g].T.astype(np.float64) / gsize[keep_g]     # [V, G]
    tot = rates.sum(axis=1, keepdims=True)
    p = np.divide(rates, tot, out=np.zeros_like(rates), where=tot > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(np.where(p > 0, p * np.log(p), 0)).sum(axis=1) / math.log(keep_g.sum())
    spread = np.where(grp[keep_g].sum(axis=0) >= MIN_SPREAD_DOCS, H, np.nan)

    t = pd.DataFrame({"phrase": vocab, "volume_recent": volume.astype(int), "growth": growth,
                      "first_year": first_year, "max_rate_recent": max_rate, "spread": spread,
                      "sources_recent": sources_recent, "oa_recent": dR[:, oa].astype(int), "ax_recent": dR[:, ax].astype(int),
                      "hf_recent": dR[:, hf].astype(int), "oa_base": dB[:, oa].astype(int), "ax_base": dB[:, ax].astype(int)})
    t["eid"] = np.arange(V)
    t["key"] = [l for l in (REG / "entity_keys.txt").read_text(encoding="utf-8").split("\n") if l]
    t["n_words"] = t.phrase.str.count(" ") + 1
    t["generic"] = t.phrase.map(lambda ph: all(w in GENERIC for w in ph.split()))

    # C-value: обрывок более длинной фразы
    vol = dict(zip(t.phrase, t.volume_recent))
    best_super: dict[str, int] = {}
    for ph, v in vol.items():
        w = ph.split()
        for n in range(1, len(w)):
            for i in range(len(w) - n + 1):
                sub = " ".join(w[i:i + n])
                if sub in vol and v > best_super.get(sub, 0):
                    best_super[sub] = v
    t["super_share"] = [best_super.get(ph, 0) / v if v else 0.0 for ph, v in zip(t.phrase, t.volume_recent)]

    cand = ((t.volume_recent >= MIN_VOLUME) & ((t.growth >= MIN_GROWTH) | (t.first_year >= NEW_SINCE))
            & (t.max_rate_recent <= MAX_RATE) & ~(t.spread > MAX_SPREAD) & ~t.generic & (t.super_share < FRAGMENT_SHARE) & (t.sources_recent >= MIN_SOURCES))
    t["candidate"] = cand
    # ранжирующий балл (порядок в реестре, не вероятность): рост + новизна + подтверждение источниками,
    # минус размазанность по темам и минус объём сверх «слабого» (крупная тема — уже не слабый сигнал)
    g = t.growth.fillna(0).clip(0, 6)
    t["score"] = (g + 1.0 * (t.first_year >= NEW_SINCE) + 0.5 * (t.sources_recent - 1).clip(0, 2)
                  - 1.0 * t.spread.fillna(0.5) - 0.5 * np.log10((t.volume_recent / WEAK_VOLUME).clip(lower=1)))
    return t


def main() -> int:
    t = score()
    t.to_parquet(REG / "all_phrases.parquet", index=False)
    reg = t[t.candidate].sort_values("score", ascending=False).reset_index(drop=True)
    reg.to_parquet(REG / "registry.parquet", index=False)
    reg.head(3000).to_csv(REG / "registry_top.csv", index=False)
    print(f"фраз всего {len(t):,}, в реестре {len(reg):,}".replace(",", " "))
    print(reg.head(60)[["phrase", "volume_recent", "growth", "first_year", "spread", "sources_recent", "score"]].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
