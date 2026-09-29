"""Исторический двойник: какая технология в конце 2021 года выглядела так же, как эта сейчас, — и что с ней стало.

    python -m registry.twins          # -> twins.parquet (для карточек базы)

Признаки — те же, что у модели взлёта на истории (registry/history_model.py): рост, объём, темп, размазанность,
источники, возраст темы, форма названия, Hacker News, YC, пресса. Они не зависят от абсолютного года, поэтому
технология сейчас (срез 2026) и технологии 2021 года сравнимы. Признаки стандартизуются по 2021 году, двойник —
ближайший сосед среди технологий 2021 года (не обрывок, ≥ 3 работ в arXiv), кроме самой этой технологии.
Исход двойника — по правилу машины времени: рост доли работ к 2024–2026, взлёт, выход в стартапы и прессу.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .history_model import FEATS, PAST, NOW, REG_NOW, REG_PAST, features, future_label

SEM_K = 50
COMBO_SEM_MIN = 0.72
GENERIC = {"ai", "artificial", "intelligence", "intelligent", "agent", "agents", "model", "models", "learning", "for"}


def main() -> int:
    from .signal_base import is_fragment, is_umbrella
    from .time_machine import outcome
    past = features(REG_PAST, PAST)
    past["takeoff"] = future_label(past).values
    past = pd.concat([past, outcome(past)], axis=1)
    past = past[~past.phrase.map(is_fragment) & ~past.phrase.map(is_umbrella) & (past.docs_then >= 3)].reset_index(drop=True)
    now = features(REG_NOW, NOW)
    base = pd.read_parquet(REG_NOW / "signal_base.parquet", columns=["key"])
    now = now[now.key.isin(set(base.key))].reset_index(drop=True)
    med = past[FEATS].median()
    mu, sd = past[FEATS].fillna(med).mean(), past[FEATS].fillna(med).std().replace(0, 1)
    P = ((past[FEATS].fillna(med) - mu) / sd).values
    Q = ((now[FEATS].fillna(med) - mu) / sd).values
    # двойник — среди SEM_K технологий 2021 года, ближайших по смыслу термина (иначе у «профилирования GPU»
    # двойником выходила «политическая реклама»): похожая тема, похожая траектория
    from sentence_transformers import SentenceTransformer
    import torch
    from .search import MODEL
    m = SentenceTransformer(MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")
    m.max_seq_length = 32
    Ep = m.encode(past.phrase.str.replace("_", " ").tolist(), batch_size=256, normalize_embeddings=True)
    En = m.encode(now.phrase.str.replace("_", " ").tolist(), batch_size=256, normalize_embeddings=True)
    rows = []
    for n, k in enumerate(now.key):
        sem = Ep @ En[n]
        sem[past.key.values == k] = -np.inf             # не сама же технология
        near = np.argpartition(-sem, SEM_K)[:SEM_K]
        d = ((P[near] - Q[n]) ** 2).sum(axis=1)
        j = int(near[np.argmin(d)])
        d = ((P - Q[n]) ** 2).sum(axis=1)
        t = past.iloc[j]
        rows.append({"key": k, "twin": t.phrase.replace("_", " "), "twin_takeoff": bool(t.takeoff),
                     "twin_growth": round(float(t.future_growth), 1), "twin_docs_then": int(t.docs_then),
                     "twin_docs_later": int(t.docs_later), "twin_yc_after": int(t.yc_after),
                     "twin_press_after": int(t.press_after), "distance": round(float(np.sqrt(d[j])), 2)})
    # составные сигналы «метод × объект»: признаков реестра у них нет — двойник по смыслу связки и по трём признакам
    # траектории, которые у связки есть (объём свежих работ, рост, возраст), в рангах внутри своей совокупности
    comp_p = REG_NOW / "composites.parquet"
    combos = pd.read_parquet(REG_NOW / "signal_base.parquet", columns=["key"])
    combos = combos[combos.key.str.startswith("combo:")]
    if comp_p.exists() and len(combos):
        comp = pd.read_parquet(comp_p)
        comp["key"] = "combo:" + comp.method + "|" + comp.object
        comp = comp[comp.key.isin(set(combos.key))].reset_index(drop=True)
        rk = lambda v: pd.Series(v).rank(pct=True).values
        Pr = np.c_[rk(past.volume_recent.fillna(0)), rk(past.growth.fillna(past.growth.median())), rk(past.age)]
        Cr = np.c_[rk(comp.docs_recent), rk(np.log(comp.growth.clip(1e-3, 1e3))), rk(NOW - comp.first_year.fillna(NOW))]
        Ec = m.encode(comp.phrase.str.replace("_", " ").tolist(), batch_size=64, normalize_embeddings=True)
        for n, k in enumerate(comp.key):
            sem = Ep @ Ec[n]
            near = np.argpartition(-sem, 5)[:5]             # в словаре 2021 года связок почти нет: только 5 ближайших
            near = near[sem[near] >= COMBO_SEM_MIN]          # и только действительно близкие по смыслу
            if not len(near):
                continue
            # и с общим предметным словом (не «ai», «agent», «model»): иначе близость ловит «human ai», «ai empowered»
            mine = set(comp.phrase[n].replace("_", " ").split()) - GENERIC
            near = np.array([i for i in near if set(past.phrase.iat[i].replace("_", " ").split()) & mine], dtype=int)
            if not len(near):
                continue
            d = ((Pr[near] - Cr[n]) ** 2).sum(axis=1)
            j = int(near[np.argmin(d)])
            t = past.iloc[j]
            rows.append({"key": k, "twin": t.phrase.replace("_", " "), "twin_takeoff": bool(t.takeoff),
                         "twin_growth": round(float(t.future_growth), 1), "twin_docs_then": int(t.docs_then),
                         "twin_docs_later": int(t.docs_later), "twin_yc_after": int(t.yc_after),
                         "twin_press_after": int(t.press_after), "distance": round(float(np.sqrt(d.min())), 2)})
    out = pd.DataFrame(rows)
    out.to_parquet(REG_NOW / "twins.parquet")
    print(f"двойники: {len(out)}; из них взлетели к 2024–2026: {out.twin_takeoff.mean():.0%}; "
          f"медиана расстояния {out.distance.median():.2f}")
    print(out.sample(8, random_state=1)[["key", "twin", "twin_growth", "twin_takeoff"]].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
