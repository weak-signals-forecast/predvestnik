"""Машина времени для витрины: что система выдала бы в конце 2021 года — и что с этим стало к 2024–2026.

    python -m registry.time_machine          # -> web/data/time_machine.json

Честность: балл 2021 года для каждой технологии — прогноз модели взлёта (registry/history_model.py), которая эту
технологию НЕ видела при обучении (предсказания на отложенных частях 5-кратной кросс-валидации). Исход берётся из
будущего по правилу, зафиксированному до замера (машина времени, registry/backtest.py): доля работ arXiv в 2024–2026
не меньше чем втрое выше доли 2019–2021 и не меньше 30 работ. Отдельно — пришла ли тема в стартапы и прессу после 2021.
Сравнение — с долей взлетевших среди всех технологий 2021 года; интервал для подъёма — бутстрэп.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .history_model import FEATS, FUTURE, PAST, REG_NOW, REG_PAST, features, future_label, model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "data" / "time_machine.json"
TOP = 30


def outcome(r: pd.DataFrame) -> pd.DataFrame:
    from .corpus import SOURCES, YEARS
    df = np.load(REG_PAST / "counts.npz")["df"]
    totals = json.loads((REG_PAST / "totals.json").read_text(encoding="utf-8"))
    ax = SOURCES.index("arxiv")
    yi = {y: i for i, y in enumerate(YEARS)}
    e = r.eid.astype(int).values

    def docs(years):
        return sum(df[e, ax, yi[y]] for y in years)

    def share(years):
        return docs(years) / max(sum(totals.get(f"arxiv|{y}", 0) for y in years), 1)

    now_y = range(PAST - 2, PAST + 1)
    out = pd.DataFrame({"docs_then": docs(now_y), "docs_later": docs(FUTURE),
                        "future_growth": (share(FUTURE) + 1e-9) / (share(now_y) + 1e-9)}, index=r.index)
    b = np.load(REG_NOW / "business_series.npz", allow_pickle=True)
    idx = {k: i for i, k in enumerate(b["keys"])}
    src = [str(s) for s in b["sources"]]
    years = [int(y) for y in b["years"]]
    after = [i for i, y in enumerate(years) if y > PAST]
    before = [i for i, y in enumerate(years) if y <= PAST]
    for name in ("hn", "yc", "press"):
        s = src.index(name)
        out[f"{name}_before"] = [int(b["df"][idx[k], s, before].sum()) if k in idx else 0 for k in r.key]
        out[f"{name}_after"] = [int(b["df"][idx[k], s, after].sum()) if k in idx else 0 for k in r.key]
    return out


def main() -> int:
    from sklearn.model_selection import StratifiedKFold
    from .signal_base import is_fragment, is_umbrella
    past = features(REG_PAST, PAST)
    y = future_label(past).values.astype(int)
    X = past[FEATS].astype(float)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        oof[te] = model().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    past["p"], past["takeoff"] = oof, y.astype(bool)
    past = pd.concat([past, outcome(past)], axis=1)
    ok = ~past.phrase.map(is_fragment) & ~past.phrase.map(is_umbrella) & (past.docs_then >= 3)
    pool = past[ok]
    top = pool.nlargest(TOP, "p")
    base_rate, top_rate = float(pool.takeoff.mean()), float(top.takeoff.mean())
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):                                  # бутстрэп подъёма: переcэмпл верхних и всего пула
        t = rng.choice(top.takeoff.values, len(top))
        b = rng.choice(pool.takeoff.values, len(pool))
        boot.append(t.mean() / max(b.mean(), 1e-9))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    items = [{"term": r.phrase.replace("_", " "), "p": round(float(r.p), 3), "takeoff": bool(r.takeoff),
              "docs_then": int(r.docs_then), "docs_later": int(r.docs_later), "growth": round(float(r.future_growth), 1),
              "market_after": {"hn": int(r.hn_after), "yc": int(r.yc_after), "press": int(r.press_after)},
              "market_before": {"hn": int(r.hn_before), "yc": int(r.yc_before), "press": int(r.press_before)}}
             for r in top.itertuples()]
    data = {"cutoff": PAST, "future": f"{FUTURE[0]}–{FUTURE[-1]}", "pool": int(len(pool)), "top": TOP,
            "base_rate": round(base_rate, 3), "top_rate": round(top_rate, 3),
            "lift": round(top_rate / base_rate, 1), "lift_ci": [round(float(lo), 1), round(float(hi), 1)],
            "rule": "взлёт — доля работ arXiv в 2024–2026 не меньше чем втрое выше доли 2019–2021 и не меньше 30 работ",
            "items": items}
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"2021: технологий {len(pool)}, взлетели {base_rate:.1%}; из верхних {TOP} по прогнозу — {top_rate:.0%} "
          f"(подъём ×{top_rate / base_rate:.1f}, 95 % интервал {lo:.1f}–{hi:.1f}) -> {OUT}")
    for r in top.head(15).itertuples():
        print(f"  {'✓' if r.takeoff else '·'} {r.phrase.replace('_', ' ')[:40]:40} {int(r.docs_then):>4} → {int(r.docs_later):>5} "
              f"×{r.future_growth:5.1f}  YC {int(r.yc_after)}  пресса {int(r.press_after)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
