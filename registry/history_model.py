"""Модель взлёта, обученная на истории: признаки на конец 2021 года → взлетела ли тема к 2024–2026.

    python -m registry.history_model [--axes] [--label v1|v2] [--suffix _axes]

Метки из будущего, без чьей-либо разметки и без языковых моделей.
  v1 — «взлёт»: доля работ arXiv с фразой в 2024–2026 не меньше TAKEOFF раз выше доли 2019–2021 и в 2024–2026
       не меньше MIN_FUTURE работ (правило машины времени, registry/backtest.py, зафиксировано до замера);
  v2 — «слабый сигнал, который взлетел»: то же, и на дату среза тема была малой (не больше SMALL_MAX работ за
       три года) и не была улучшением зрелого родителя (ось incremental_lex < INC_MAX, registry/axes.py).
       Так модель учится находить «малое, новое и выросшее», а не просто «выросшее».

Признаки считаются только по данным до года среза и не зависят от абсолютного года, поэтому модель, обученная
на срезе 2021, применяется к признакам на срез 2026:
  наука (registry.parquet): объём, рост, темп, размазанность, источники, возраст, форма названия;
  рынок (business_series.npz, ряды по годам): Hacker News за 3 года и его рост, стартапы YC к году среза, пресса;
  --axes: оси слабости (axes.parquet): родитель, S-кривая, переход слоёв, стандарт.
Проверка: кросс-валидация на срезе 2021; независимо — на экспертной разметке 2026; и «что попало бы в топ-30
на срезе 2021»: доля взлетевших и доля улучшений известного.

Пути: корпус GPB_CORPUS (по умолчанию ~/gpb_corpus), реестр среза HIST_PAST (по умолчанию <корпус>/registry_2021),
текущий реестр GPB_REGISTRY (по умолчанию <корпус>/registry).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .axes import AXES, LABELS as AXES_LABELS
from .corpus import CORPUS, REGISTRY

REG_PAST = Path(os.environ.get("HIST_PAST", CORPUS / "registry_2021"))
REG_NOW = REGISTRY
PAST, NOW = int(os.environ.get("HIST_PAST_YEAR", "2021")), int(os.environ.get("REGISTRY_CUTOFF", "2026"))
TAKEOFF, MIN_FUTURE = 3.0, 30
FUTURE = (2024, 2025, 2026)
SMALL_MAX, INC_MAX = 60, 3.0

SCI = ["volume_recent", "growth", "max_rate_recent", "spread", "sources_recent", "oa_share", "age", "n_words",
       "generic", "super_share"]
BIZ = ["hn_recent", "hn_growth", "yc_total", "press_recent"]
FEATS = SCI + BIZ
LABELS = {"volume_recent": "Объём научных работ", "growth": "Рост доли в науке",
          "max_rate_recent": "Доля среди всех научных работ", "spread": "Охват разных научных областей",
          "sources_recent": "Число научных источников", "oa_share": "Доля журнальных статей (против препринтов)",
          "age": "Возраст темы", "n_words": "Длина названия (конкретность)", "generic": "Общие слова в названии",
          "super_share": "Часть более длинных терминов", "hn_recent": "Обсуждения на Hacker News",
          "hn_growth": "Рост обсуждений на Hacker News", "yc_total": "Стартапы Y Combinator",
          "press_recent": "Публикации в прессе", **AXES_LABELS}


def business(keys: pd.Series, cutoff: int) -> pd.DataFrame:
    """Бизнес-признаки строго по годам ≤ cutoff (ряды по годам посчитаны на полном корпусе)."""
    p = REG_NOW / "business_series.npz"
    if not p.exists():
        p = CORPUS / "registry_v3" / "business_series.npz"
    b = np.load(p, allow_pickle=True)
    idx = {k: i for i, k in enumerate(b["keys"])}
    src = [str(s) for s in b["sources"]]
    years = [int(y) for y in b["years"]]
    yi = {y: i for i, y in enumerate(years)}
    df, tot = b["df"], b["totals"]
    rec = [yi[y] for y in range(cutoff - 2, cutoff + 1) if y in yi]
    base = [yi[y] for y in range(cutoff - 5, cutoff - 2) if y in yi]
    upto = [yi[y] for y in years if y <= cutoff]
    hn, yc, pr = src.index("hn"), src.index("yc"), src.index("press")
    out = np.zeros((len(keys), 4))
    for n, k in enumerate(keys):
        i = idx.get(k)
        if i is None:
            continue
        h_rec = df[i, hn, rec].sum() / max(tot[hn, rec].sum(), 1)
        h_base = df[i, hn, base].sum() / max(tot[hn, base].sum(), 1)
        out[n] = [np.log1p(df[i, hn, rec].sum()), np.log((h_rec + 1e-7) / (h_base + 1e-7)),
                  np.log1p(df[i, yc, upto].sum()), np.log1p(df[i, pr, rec].sum())]
    return pd.DataFrame(out, columns=BIZ, index=keys.index)


def features(reg: Path, cutoff: int, with_axes: bool = False) -> pd.DataFrame:
    from . import axes
    r = pd.read_parquet(reg / "registry.parquet")
    r = r[r.candidate].drop_duplicates("key").reset_index(drop=True)
    r["oa_share"] = r.oa_recent / (r.oa_recent + r.ax_recent).clip(lower=1)
    r["age"] = cutoff - r.first_year
    r = pd.concat([r, business(r.key, cutoff)], axis=1)
    if with_axes:
        a = axes.load(reg)[["key"] + AXES + ["parent", "docs_now"]]
        r = r.merge(a, on="key", how="left")
        r["layer_lag_hn"] = r.layer_lag_hn.fillna(-1)          # нет следа в слое — отдельное значение, не пропуск
        r["layer_lag_market"] = r.layer_lag_market.fillna(-1)
    return r


def future_label(r: pd.DataFrame, version: str = "v1") -> pd.Series:
    """Метка из будущего по arXiv (правило машины времени); v2 — только для малых и не инкрементальных на срезе."""
    from .corpus import SOURCES, YEARS
    df = np.load(REG_PAST / "counts.npz")["df"]
    totals = json.loads((REG_PAST / "totals.json").read_text(encoding="utf-8"))
    ax = SOURCES.index("arxiv")
    yi = {y: i for i, y in enumerate(YEARS)}
    e = r.eid.astype(int).values

    def share(years):
        n = sum(totals.get(f"arxiv|{y}", 0) for y in years)
        d = sum(df[e, ax, yi[y]] for y in years)
        return d, d / max(n, 1)

    _, s_now = share(range(PAST - 2, PAST + 1))
    d_fut, s_fut = share(FUTURE)
    y = ((s_fut + 1e-9) / (s_now + 1e-9) >= TAKEOFF) & (d_fut >= MIN_FUTURE)
    if version == "v2":
        inc = r.incremental_lex if "incremental_lex" in r else pd.Series(np.nan, index=r.index)
        y &= (r.volume_recent <= SMALL_MAX) & ~(inc.fillna(-np.inf) >= INC_MAX)
    return pd.Series(y, index=r.index)


def model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400, min_samples_leaf=40,
                                          l2_regularization=1.0, class_weight="balanced", random_state=0)


def top_report(past: pd.DataFrame, p: np.ndarray, y_v1: np.ndarray, k: int = 30) -> str:
    """Что попало бы в топ-k на срезе: доля взлетевших (v1) и доля улучшений известного."""
    top = past.assign(p=p).nlargest(k, "p")
    inc = (top.incremental_lex.fillna(-np.inf) >= INC_MAX).mean() if "incremental_lex" in top else float("nan")
    return (f"топ-{k} на срезе {PAST}: взлетели {y_v1[top.index].mean():.0%}, улучшений известного {inc:.0%}, "
            f"медианный объём {int(top.volume_recent.median())}")


def main() -> int:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", action="store_true")
    ap.add_argument("--label", default="v1", choices=["v1", "v2"])
    ap.add_argument("--suffix", default=None, help="суффикс файлов результата (по умолчанию '' или '_axes')")
    a = ap.parse_args()
    feats = FEATS + (AXES if a.axes else [])
    suffix = a.suffix if a.suffix is not None else ("_axes" if a.axes else "")
    past = features(REG_PAST, PAST, a.axes)
    y_v1 = future_label(past, "v1").values.astype(int)
    y = future_label(past, a.label).values.astype(int)
    X = past[feats].astype(float)
    print(f"срез {PAST}: кандидатов {len(past)}, метка {a.label}: положительных {int(y.sum())} ({y.mean():.1%}); "
          f"признаков {len(feats)}{' (с осями)' if a.axes else ''}")
    auc, ap_, oof = [], [], np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        p = model().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        auc.append(roc_auc_score(y[te], p))
        ap_.append(average_precision_score(y[te], p))
    print(f"кросс-валидация: ROC AUC {np.mean(auc):.3f}, PR AUC {np.mean(ap_):.3f} (база {y.mean():.3f})")
    print(top_report(past, oof, y_v1, 30))
    print(top_report(past, oof, y_v1, 100))
    m = model().fit(X, y)

    now = features(REG_NOW, NOW, a.axes)
    now["history_p"] = m.predict_proba(now[feats].astype(float))[:, 1]
    now[["key", "phrase", "history_p"] + feats].to_parquet(REG_NOW / f"history_scored{suffix}.parquet")
    joblib.dump({"model": m, "features": feats, "label": a.label}, REG_NOW / f"history_model{suffix}.joblib")

    lab_p = REG_NOW / "expert_labels.csv"
    if lab_p.exists():
        lab = pd.read_csv(lab_p)[["key", "y"]].drop_duplicates("key")
        t = lab.merge(now[["key", "history_p"]], on="key")
        if len(t) and t.y.nunique() == 2:
            print(f"экспертная разметка ({len(t)}, сигналов {int(t.y.sum())}): ROC AUC {roc_auc_score(t.y, t.history_p):.3f}, "
                  f"PR AUC {average_precision_score(t.y, t.history_p):.3f} (база {t.y.mean():.3f})")
    base_p = REG_NOW / "base_candidates.parquet"
    if base_p.exists():
        import shap
        base = set(pd.read_parquet(base_p).key)
        sub = now[now.key.isin(base)].reset_index(drop=True)
        ex = shap.TreeExplainer(m)
        sv = ex.shap_values(sub[feats].astype(float))
        sv = sv[1] if isinstance(sv, list) else sv
        out = pd.DataFrame(sv, columns=feats)
        out.insert(0, "key", sub.key.values)
        out.to_parquet(REG_NOW / f"history_shap{suffix}.parquet")
        Xv = sub[feats].astype(float)
        glob = []
        for j, f in enumerate(feats):
            v = Xv[f].values
            ok = ~np.isnan(v)
            c = float(np.corrcoef(v[ok], sv[ok, j])[0, 1]) if ok.sum() > 2 and np.std(v[ok]) > 0 else 0.0
            glob.append({"feature": f, "label": LABELS[f], "mean_abs": float(np.abs(sv[:, j]).mean()), "direction": c})
        glob.sort(key=lambda x: -x["mean_abs"])
        (REG_NOW / f"history_shap_global{suffix}.json").write_text(json.dumps(glob, ensure_ascii=False, indent=1), encoding="utf-8")
        print("главные признаки:", ", ".join(f"{g['label']} {'↑' if g['direction'] >= 0 else '↓'}" for g in glob[:8]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
