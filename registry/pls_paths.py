"""Модель путей со скрытыми факторами (PLS-SEM): что толкает технологию к взлёту и через что.

    GPB_CORPUS=<корпус> GPB_REGISTRY=<текущий реестр> python -m registry.pls_paths <реестр 2017>:2017 <реестр 2021>:2021 \
        [--n 20000] [--out pls.json]

Зачем. Модели реестра предсказывают, SHAP раскладывает балл по признакам, но не показывает связи между группами
признаков. Здесь признаки собраны в скрытые факторы и считается сила путей между ними (PLS path modeling, схема
path, пакет plspm):

  Масштаб  — объём работ в окне среза, наибольшая доля в источнике, число научных источников;
  Импульс  — рост доли науки, ускорение, доля работ за последние три года;
  Рынок    — Hacker News (объём, рост), стартапы YC, пресса — строго до среза;
  Зрелость — родитель на порядок объёмнее (incremental_lex), объём родителя, возраст термина;
  Взлёт    — log2 роста доли работ arXiv в окне cutoff+3..cutoff+5 к окну среза (будущее, только как исход).

Пути: Зрелость → Импульс, Масштаб → Рынок, Импульс → Рынок, Масштаб/Импульс/Рынок/Зрелость → Взлёт.
Устойчивость (против подгонки): одна и та же схема оценивается отдельно на срезе 2017 и на срезе 2021; выводы — только
о путях, которые совпадают по знаку и близки по силе на обоих срезах. Бутстреп — доверительные интервалы путей.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import SOURCES, YEARS
from .history_model import features

BLOCKS = {
    "Масштаб": ["log_volume", "max_rate_recent", "sources_recent"],
    "Импульс": ["growth", "stage_accel", "stage_recent_share"],
    "Рынок": ["hn_recent", "hn_growth", "yc_total", "press_recent"],
    "Зрелость": ["incremental_lex", "parent_docs_log", "age"],
    "Взлёт": ["future_growth"],
}
PATHS = [("Зрелость", "Импульс"), ("Масштаб", "Рынок"), ("Импульс", "Рынок"),
         ("Масштаб", "Взлёт"), ("Импульс", "Взлёт"), ("Рынок", "Взлёт"), ("Зрелость", "Взлёт")]


def future_growth(reg: Path, eid: np.ndarray, cutoff: int) -> np.ndarray:
    df = np.load(reg / "counts.npz")["df"]
    totals = json.loads((reg / "totals.json").read_text(encoding="utf-8"))
    ax = SOURCES.index("arxiv")
    yi = {y: i for i, y in enumerate(YEARS)}

    def share(years):
        n = sum(totals.get(f"arxiv|{y}", 0) for y in years)
        return (sum(df[eid, ax, yi[y]] for y in years) + 1.0) / max(n, 1)

    return np.log2(share(range(cutoff + 3, cutoff + 6)) / share(range(cutoff - 2, cutoff + 1)))


def table(reg: Path, cutoff: int, n: int, seed: int = 0) -> pd.DataFrame:
    r = features(reg, cutoff, with_axes=True)
    r = r.sample(min(n, len(r)), random_state=seed).reset_index(drop=True)
    r["log_volume"] = np.log1p(r.volume_recent)
    for c in ("incremental_lex", "parent_docs_log", "stage_accel"):
        r[c] = r[c].fillna(0.0)
    r["future_growth"] = future_growth(reg, r.eid.astype(int).values, cutoff)
    cols = [c for b in BLOCKS.values() for c in b]
    return r[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit(data: pd.DataFrame, boot: int) -> dict:
    import plspm.config as c
    from plspm.mode import Mode
    from plspm.plspm import Plspm
    from plspm.scheme import Scheme
    s = c.Structure()
    for a, b in PATHS:
        s.add_path([a], [b])
    cfg = c.Config(s.path(), scaled=True)
    for lv, mvs in BLOCKS.items():
        cfg.add_lv(lv, Mode.A, *[c.MV(m) for m in mvs])
    p = Plspm(data, cfg, Scheme.PATH, bootstrap=boot > 0, bootstrap_iterations=max(boot, 1), processes=4)
    pc = p.path_coefficients()
    out = {"paths": {f"{a} → {b}": round(float(pc.loc[b, a]), 3) for a, b in PATHS},
           "r2": {k: round(float(v), 3) for k, v in p.inner_summary()["r_squared"].items() if v > 0},
           "loadings": {k: round(float(v), 3) for k, v in p.outer_model()["loading"].items()}}
    eff = p.effects()
    out["effects"] = {str(i): {k: round(float(v), 3) for k, v in row.items() if isinstance(v, (int, float))}
                      for i, row in eff.iterrows()}
    if boot:
        bp = p.bootstrap().paths()
        out["paths_ci"] = {str(i): [round(float(row["perc.025"]), 3), round(float(row["perc.975"]), 3)]
                           for i, row in bp.iterrows()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slices", nargs="+")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {}
    for spec in a.slices:
        reg, cut = Path(spec.rsplit(":", 1)[0]), int(spec.rsplit(":", 1)[1])
        data = table(reg, cut, a.n)
        res[str(cut)] = {"n": len(data), **fit(data, a.boot)}
        print(f"срез {cut}: n={len(data)}", flush=True)
        print(json.dumps(res[str(cut)]["paths"], ensure_ascii=False, indent=1), flush=True)
        print("R²:", res[str(cut)]["r2"], flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
