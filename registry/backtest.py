"""Машина времени для реестра: что из вершины реестра, собранного «как будто сейчас конец 2021 года», выросло к 2026.

    REGISTRY_CUTOFF=2021 GPB_REGISTRY=~/gpb_corpus/registry_2021 python -m registry.build all
    REGISTRY_CUTOFF=2021 GPB_REGISTRY=~/gpb_corpus/registry_2021 python -W ignore -m registry.score
    REGISTRY_CUTOFF=2021 GPB_REGISTRY=~/gpb_corpus/registry_2021 python -m registry.backtest

Проверка без списка организаторов и без утечки: при отборе кандидатов данные после года среза не используются
(словарь — из заголовков 2018–2021, окна 2019–2021 и 2014–2017). Будущее смотрится только здесь.

Правила (зафиксированы до замера):
  доля   — работы OpenAlex + arXiv с фразой / все прочитанные работы этих источников за окно;
  взлёт  — доля 2024–2026 не меньше чем в TAKEOFF раз выше доли окна среза, и в 2024–2026 не меньше MIN_FUTURE работ;
  массовая к 2026 — доля 2024–2026 выше 0,1 % (как порог «массовости» в score.py).
Сравнение: первые K кандидатов реестра по ручному баллу против случайной выборки кандидатов того же реестра и
против всех фраз словаря с ≥ 5 работами в окне среза (базовая частота).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from .corpus import CUTOFF, REGISTRY, SOURCES, YEARS

REG = REGISTRY
FUTURE = (2024, 2025, 2026)
TAKEOFF = 3.0
MIN_FUTURE = 30
MASS = 1e-3
KS = (50, 100, 500)
SEED = 11


def main() -> int:
    t = pd.read_parquet(REG / "all_phrases.parquet")
    df = np.load(REG / "counts.npz")["df"]
    totals = json.loads((REG / "totals.json").read_text(encoding="utf-8"))
    oa, ax = SOURCES.index("openalex"), SOURCES.index("arxiv")
    yi = {y: i for i, y in enumerate(YEARS)}
    now = tuple(range(CUTOFF - 2, CUTOFF + 1))

    import os
    srcs = os.environ.get("REGISTRY_GROWTH", "openalex,arxiv").split(",")   # те же источники, что и для роста в score.py

    def share(years):
        n = sum(totals.get(f"{s}|{y}", 0) for s in srcs for y in years)
        d = sum(df[:, SOURCES.index(s), yi[y]] for s in srcs for y in years)
        return d, d / max(n, 1)

    d_now, s_now = share(now)
    d_fut, s_fut = share(FUTURE)
    t["docs_now"], t["docs_future"] = d_now, d_fut
    t["growth_future"] = (s_fut + 1e-9) / (s_now + 1e-9)
    t["takeoff"] = (t.growth_future >= TAKEOFF) & (t.docs_future >= MIN_FUTURE)
    t["mass_2026"] = s_fut > MASS
    t["success"] = t.takeoff | t.mass_2026        # успех прогноза: взлетела или стала массовой (в 2021 массовой не была)

    cand = t[t.candidate].sort_values("score", ascending=False).reset_index(drop=True)
    base = t[(t.docs_now >= 5)]
    rng = np.random.default_rng(SEED)
    print(f"источники роста: {', '.join(srcs)}")
    print(f"срез {CUTOFF}: кандидатов реестра {len(cand):,}, фраз с ≥5 работами в {now[0]}–{now[-1]}: {len(base):,}".replace(",", " "))
    print(f"базовая частота взлёта (все такие фразы): {base.takeoff.mean():.1%}; среди кандидатов реестра: {cand.takeoff.mean():.1%}")
    for k in KS:
        top = cand.head(k)
        rnd = cand.iloc[rng.choice(len(cand), size=min(k, len(cand)), replace=False)]
        rnd_big = cand.sample(min(20 * k, len(cand)), random_state=SEED)
        print(f"  первые {k:>3}: взлетели или стали массовыми {top.success.mean():.0%} "
              f"(случайные кандидаты: {rnd_big.success.mean():.0%}, все фразы: {base.success.mean():.0%})")
        print(f"  первые {k:>3}: взлетели {top.takeoff.mean():.0%} (случайные {k} кандидатов: {rnd.takeoff.mean():.0%}); "
              f"стали массовыми к 2026: {top.mass_2026.mean():.0%}; медиана роста доли ×{top.growth_future.median():.1f}")
    cols = ["phrase", "docs_now", "docs_future", "growth_future", "takeoff", "mass_2026", "score"]
    cand[cols].head(500).to_csv(REG / f"backtest_{CUTOFF}.csv", index=False)
    print("\nпервые 40 реестра-{}: что с ними стало к 2024–2026".format(CUTOFF))
    for _, r in cand.head(40).iterrows():
        mark = "ВЗЛЁТ" if r.takeoff else ("массовая" if r.mass_2026 else "")
        print(f"  {r.phrase[:40]:40} {int(r.docs_now):>5} → {int(r.docs_future):>6} работ  ×{r.growth_future:5.1f}  {mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
