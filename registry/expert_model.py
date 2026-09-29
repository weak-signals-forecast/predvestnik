"""Модель слабого сигнала по экспертной разметке: градиентный бустинг и разложение балла по признакам (SHAP).

    python -m registry.expert_model          # сравнение с логистической регрессией, обучение, SHAP

Итоговая модель — градиентный бустинг (по качеству наравне с регрессией: ROC AUC 0,78 против 0,80 — в пределах
разброса), потому что его балл раскладывается по признакам точно (SHAP, TreeExplainer): вклад каждого признака в
балл конкретной технологии и общая картина по всем кандидатам.

Разметка: expert_labels.csv (≈1 000 кандидатов из базы, из вычищенного правилами и из пула; 1 — эксперт признал бы
слабым сигналом, 0 — нет) + известные сигналы (совпадения с сигналами организаторов и прежняя разметка выдачи).
Признаки: 19 признаков реестра (наука, бизнес-слои, форма названия) + год первого появления, доля теоретических и
академических рубрик. Сравнение моделей — повторённая стратифицированная 5-кратная кросс-валидация, ROC AUC и
средняя точность (PR AUC).
"""
from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd

from .corpus import REGISTRY as REG
from .ranker import FEATURES

EXTRA = ["first_year", "theory_share", "academic_share"]
ALL = FEATURES + EXTRA


def table() -> pd.DataFrame:
    from .search import first_year
    from .signal_base import ACADEMIC_RUBRICS, rubric_share, theory_share
    r = pd.read_parquet(REG / "ranked.parquet")
    r = r[r.candidate].drop_duplicates("key").reset_index(drop=True)
    r["first_year"] = first_year(r.key)
    r["theory_share"] = theory_share(r.key)
    r["academic_share"] = rubric_share(r.key, ACADEMIC_RUBRICS)
    return r


def labels(r: pd.DataFrame) -> pd.DataFrame:
    from .signal_base import _labels
    e = pd.read_csv(REG / "expert_labels.csv")[["key", "y"]]
    k = pd.read_parquet(REG / "all_phrases.parquet", columns=["key", "phrase"])
    old = _labels()
    old = old[old.y == 1].merge(k, on="phrase", how="inner")[["key", "y"]]
    y = pd.concat([e, old]).groupby("key").y.max().reset_index()
    return r.merge(y, on="key", how="inner")


def models():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "логистическая регрессия": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                 LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000)),
        "градиентный бустинг": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                              min_samples_leaf=15, l2_regularization=1.0,
                                                              class_weight="balanced", random_state=0),
    }


def main() -> int:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    r = table()
    d = labels(r)
    X, y = d[ALL].astype(float), d.y.values.astype(int)
    print(f"размечено {len(d)}: сигналов {int(y.sum())}, не сигналов {int((y == 0).sum())}")
    res = {}
    for name, m in models().items():
        auc, ap = [], []
        for rep in range(5):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
                m.fit(X.iloc[tr], y[tr])
                p = m.predict_proba(X.iloc[te])[:, 1]
                auc.append(roc_auc_score(y[te], p))
                ap.append(average_precision_score(y[te], p))
        res[name] = (np.mean(auc), np.mean(ap))
        print(f"{name:26} ROC AUC {np.mean(auc):.3f} ± {np.std(auc):.3f}   PR AUC {np.mean(ap):.3f} (доля сигналов {y.mean():.3f})")
    best = "градиентный бустинг"
    # предсказания вне обучения (для выбора порога чистки базы): каждая строка предсказана моделью, не видевшей её
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        oof[te] = models()[best].fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    d[["key", "phrase", "y"]].assign(oof=oof).to_csv(REG / "expert_oof.csv", index=False)
    m = models()[best].fit(X, y)
    joblib.dump({"model": m, "features": ALL, "name": best}, REG / "expert_model.joblib")
    r["expert_p"] = m.predict_proba(r[ALL].astype(float))[:, 1]
    r[["key", "phrase", "expert_p"] + ALL].to_parquet(REG / "expert_scored.parquet")
    shap_values(m, r)
    print(f"итоговая: {best} -> expert_model.joblib, expert_scored.parquet ({len(r)} кандидатов), expert_shap.parquet")
    return 0


def shap_values(m, r: pd.DataFrame) -> None:
    """SHAP для кандидатов пула (вклад признаков в логит балла) + общая картина: средний |вклад| и направление."""
    import json
    import shap
    from .signal_base import CAND
    keys = set(pd.read_parquet(CAND).key)
    sub = r[r.key.isin(keys)].reset_index(drop=True)
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(sub[ALL].astype(float))
    sv = sv[1] if isinstance(sv, list) else sv
    out = pd.DataFrame(sv, columns=ALL)
    out.insert(0, "key", sub.key.values)
    out["base_value"] = float(np.ravel(ex.expected_value)[-1])
    out.to_parquet(REG / "expert_shap.parquet")
    Xv = sub[ALL].astype(float)
    glob = []
    for j, f in enumerate(ALL):
        v = Xv[f].values
        ok = ~np.isnan(v)
        corr = float(np.corrcoef(v[ok], sv[ok, j])[0, 1]) if ok.sum() > 2 and np.std(v[ok]) > 0 else 0.0
        glob.append({"feature": f, "mean_abs": float(np.abs(sv[:, j]).mean()), "direction": corr})
    glob.sort(key=lambda x: -x["mean_abs"])
    (REG / "expert_shap_global.json").write_text(json.dumps(glob, ensure_ascii=False, indent=1), encoding="utf-8")
    try:                                                  # картинка для презентации
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .export_web import LABELS
        shap.summary_plot(sv, Xv.rename(columns=lambda c: LABELS.get(c, c)), show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(REG / "expert_shap_beeswarm.png", dpi=160)
        plt.close()
    except Exception as e:
        print("картинка SHAP не построена:", str(e)[:120])


if __name__ == "__main__":
    sys.exit(main())
