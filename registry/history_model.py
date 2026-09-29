"""Модель взлёта, обученная на истории: признаки на конец 2021 года → взлетела ли тема к 2024–2026.

    python -m registry.history_model

Метки из будущего, без чьей-либо разметки и без языковых моделей. Правило «взлёта» — то же, что в машине времени
(registry/backtest.py), зафиксировано до замера: доля работ arXiv с фразой в 2024–2026 не меньше TAKEOFF раз выше
доли 2019–2021 и в 2024–2026 не меньше MIN_FUTURE работ. Рост — по arXiv: состав OpenAlex по годам смещён.

Признаки считаются только по данным до года среза и не зависят от абсолютного года (возраст темы, а не год
появления), поэтому модель, обученная на срезе 2021, применяется к признакам на срез 2026:
  наука (registry_*/registry.parquet): объём, рост, темп, размазанность, источники, возраст, форма названия;
  рынок (business_series.npz, ряды по годам): Hacker News за 3 года и его рост, стартапы YC к году среза, пресса.
Проверка: кросс-валидация на срезе 2021 и — независимо — на экспертной разметке и сигналах организаторов (срез 2026).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CORPUS = Path(os.path.expanduser("~/gpb_corpus"))
REG_PAST = CORPUS / "registry_2021"
REG_NOW = CORPUS / "registry_v3"
PAST, NOW = 2021, 2026
TAKEOFF, MIN_FUTURE = 3.0, 30
FUTURE = (2024, 2025, 2026)

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
          "press_recent": "Публикации в прессе"}


def business(keys: pd.Series, cutoff: int) -> pd.DataFrame:
    """Бизнес-признаки строго по годам ≤ cutoff (ряды по годам посчитаны на полном корпусе)."""
    b = np.load(REG_NOW / "business_series.npz", allow_pickle=True)
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


def features(reg: Path, cutoff: int) -> pd.DataFrame:
    r = pd.read_parquet(reg / "registry.parquet")
    r = r[r.candidate].drop_duplicates("key").reset_index(drop=True)
    r["oa_share"] = r.oa_recent / (r.oa_recent + r.ax_recent).clip(lower=1)
    r["age"] = cutoff - r.first_year
    r = pd.concat([r, business(r.key, cutoff)], axis=1)
    return r


def future_label(r: pd.DataFrame) -> pd.Series:
    """Взлёт к 2024–2026 по arXiv (правило машины времени)."""
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
    return pd.Series(((s_fut + 1e-9) / (s_now + 1e-9) >= TAKEOFF) & (d_fut >= MIN_FUTURE), index=r.index)


def model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400, min_samples_leaf=40,
                                          l2_regularization=1.0, class_weight="balanced", random_state=0)


def main() -> int:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    past = features(REG_PAST, PAST)
    y = future_label(past).values.astype(int)
    X = past[FEATS].astype(float)
    print(f"срез {PAST}: кандидатов {len(past)}, взлетели к 2024–2026: {int(y.sum())} ({y.mean():.1%})")
    auc, ap = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        p = model().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        auc.append(roc_auc_score(y[te], p))
        ap.append(average_precision_score(y[te], p))
    print(f"кросс-валидация на срезе {PAST}: ROC AUC {np.mean(auc):.3f}, PR AUC {np.mean(ap):.3f} (база {y.mean():.3f})")
    m = model().fit(X, y)

    now = features(REG_NOW, NOW)
    now["history_p"] = m.predict_proba(now[FEATS].astype(float))[:, 1]
    now[["key", "phrase", "history_p"] + FEATS].to_parquet(REG_NOW / "history_scored.parquet")
    joblib.dump({"model": m, "features": FEATS}, REG_NOW / "history_model.joblib")

    # независимая проверка на срезе 2026: экспертная разметка и сигналы организаторов
    lab = pd.read_csv(REG_NOW / "expert_labels.csv")[["key", "y"]].drop_duplicates("key")
    t = lab.merge(now[["key", "history_p"]], on="key")
    print(f"экспертная разметка ({len(t)}, сигналов {int(t.y.sum())}): ROC AUC {roc_auc_score(t.y, t.history_p):.3f}, "
          f"PR AUC {average_precision_score(t.y, t.history_p):.3f} (база {t.y.mean():.3f})")
    oof = pd.read_csv(REG_NOW / "expert_oof.csv").merge(t[["key"]], on="key")
    if len(oof) and oof.y.nunique() == 2:
        print(f"  для сравнения, экспертная модель вне обучения на тех же: ROC AUC {roc_auc_score(oof.y, oof.oof):.3f}")

    import shap
    base = set(pd.read_parquet(REG_NOW / "base_candidates.parquet").key)
    sub = now[now.key.isin(base)].reset_index(drop=True)
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(sub[FEATS].astype(float))
    sv = sv[1] if isinstance(sv, list) else sv
    out = pd.DataFrame(sv, columns=FEATS)
    out.insert(0, "key", sub.key.values)
    out.to_parquet(REG_NOW / "history_shap.parquet")
    Xv = sub[FEATS].astype(float)
    glob = []
    for j, f in enumerate(FEATS):
        v = Xv[f].values
        ok = ~np.isnan(v)
        c = float(np.corrcoef(v[ok], sv[ok, j])[0, 1]) if ok.sum() > 2 and np.std(v[ok]) > 0 else 0.0
        glob.append({"feature": f, "label": LABELS[f], "mean_abs": float(np.abs(sv[:, j]).mean()), "direction": c})
    glob.sort(key=lambda x: -x["mean_abs"])
    (REG_NOW / "history_shap_global.json").write_text(json.dumps(glob, ensure_ascii=False, indent=1), encoding="utf-8")
    print("главные признаки:", ", ".join(f"{g['label']} {'↑' if g['direction'] >= 0 else '↓'}" for g in glob[:6]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
