"""Кредитный рейтинг технологии: вероятность и срок взлёта по истории корпуса (модель времени до события).

    GPB_CORPUS=<корпус> GPB_REGISTRY=<текущий реестр> python -m registry.rating validate \
        --train <реестр 2017>:2017 --test <реестр 2021>:2021
    GPB_CORPUS=<корпус> GPB_REGISTRY=<текущий реестр> python -m registry.rating apply \
        --train <реестр 2017>:2017,<реестр 2021>:2021 --bundle artifacts/registry_v5

Зачем. Реестр отвечает «что» растёт. Банк спрашивает ещё «когда» и «с какой вероятностью» — как у заёмщика:
вероятность события за горизонт и рейтинговая шкала. Здесь это считается по истории самого корпуса: для каждой фразы
среза (2017, 2021) видно, через сколько лет она взлетела или не взлетела к 2026.

Событие «взлёт» в год Y — то же правило, что в машине времени (registry/backtest.py, history_model.py): доля работ
arXiv с фразой за три года Y−2..Y не меньше TAKEOFF раз выше доли окна среза (cutoff−2..cutoff), и за эти три года
не меньше MIN_FUTURE работ. Время до события T = Y − cutoff; без события до 2026 — цензурирование.

Модель — дискретная модель риска (как модели дефолта по периодам): строка на «фразу × год после среза», пока событие
не наступило, признак t — номер года, метка — событие в этом году. Признаки фразы — те же, что у исторической модели
(history_model.features: объём, рост, охват, возраст, бизнес-слои строго до среза, оси). Классификатор без
перевзвешивания классов, чтобы вероятности оставались вероятностями. P(T ≤ k) = 1 − Π_{t ≤ k} (1 − h_t).

Шкала AAA…CCC — по доле кандидатов, как у рейтинговых агентств: границы — квантили P(T ≤ 3) на обучающем срезе,
фиксируются до проверки. validate: обучение на срезе train с горизонтом до года среза test (без заглядывания
в будущее test), проверка на срезе test: AUC, калибровка по децилям, доля взлетевших по рейтингам.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .axes import AXES
from .corpus import SOURCES, YEARS
from .history_model import FEATS, features

TAKEOFF, MIN_FUTURE = 3.0, 30
LAST = 2026
GRADES = ("AAA", "AA", "A", "BBB", "BB", "B", "CCC")
GRADE_SHARE = (0.02, 0.03, 0.05, 0.10, 0.20, 0.30)     # верхние 2 %, следующие 3 % … остальное CCC
HORIZON_GRADE = 3


def _slices(spec: str) -> list[tuple[Path, int]]:
    out = []
    for part in spec.split(","):
        p, y = part.rsplit(":", 1)
        out.append((Path(p), int(y)))
    return out


def event_time(reg: Path, eid: np.ndarray, cutoff: int, last: int) -> np.ndarray:
    """Год взлёта минус cutoff (1..last−cutoff) или 0, если до last взлёта нет."""
    df = np.load(reg / "counts.npz")["df"]
    totals = json.loads((reg / "totals.json").read_text(encoding="utf-8"))
    ax = SOURCES.index("arxiv")
    yi = {y: i for i, y in enumerate(YEARS)}

    def share(years):
        n = sum(totals.get(f"arxiv|{y}", 0) for y in years)
        d = sum(df[eid, ax, yi[y]] for y in years).astype(float)
        return d, d / max(n, 1)

    _, s0 = share(range(cutoff - 2, cutoff + 1))
    t = np.zeros(len(eid), int)
    for y in range(cutoff + 1, last + 1):
        d, s = share(range(y - 2, y + 1))
        hit = (t == 0) & ((s + 1e-9) / (s0 + 1e-9) >= TAKEOFF) & (d >= MIN_FUTURE)
        t[hit] = y - cutoff
    return t


def load_slice(reg: Path, cutoff: int, last: int) -> tuple[pd.DataFrame, list[str]]:
    r = features(reg, cutoff, with_axes=True)
    feats = [f for f in FEATS + AXES if f in r.columns]
    r["T"] = event_time(reg, r.eid.astype(int).values, cutoff, last)
    r["H"] = last - cutoff
    return r, feats


def expand(r: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Строки «фраза × год риска»: годы 1..T (событие в T) или 1..H (цензура)."""
    X, y = [], []
    base = r[feats].astype(float).values
    for t in range(1, int(r.H.max()) + 1):
        at_risk = ((r.T == 0) & (r.H >= t)) | (r.T >= t)
        m = at_risk.values
        X.append(np.column_stack([base[m], np.full(m.sum(), t)]))
        y.append((r.T.values[m] == t).astype(int))
    return np.vstack(X), np.concatenate(y)


def fit(train: list[tuple[pd.DataFrame, list[str]]]):
    from sklearn.ensemble import HistGradientBoostingClassifier
    feats = train[0][1]
    Xs, ys = zip(*(expand(r, feats) for r, _ in train))
    X, y = np.vstack(Xs), np.concatenate(ys)
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=200,
                                       l2_regularization=1.0, random_state=0)
    m.fit(X, y)
    print(f"обучение: строк {len(y):,}, событий {int(y.sum()):,}".replace(",", " "), flush=True)
    return m, feats


def cumulative(m, r: pd.DataFrame, feats: list[str], horizon: int = 5) -> np.ndarray:
    """P(T ≤ k) для k = 1..horizon, массив [n, horizon]."""
    base = r[feats].astype(float).values
    surv = np.ones(len(r))
    out = np.zeros((len(r), horizon))
    for t in range(1, horizon + 1):
        h = m.predict_proba(np.column_stack([base, np.full(len(r), t)]))[:, 1]
        surv *= 1 - h
        out[:, t - 1] = 1 - surv
    return out


def grade_edges(p: np.ndarray) -> np.ndarray:
    q = np.cumsum(GRADE_SHARE)
    return np.quantile(p, 1 - q)


def grade(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(-edges, -p, side="right")     # edges по убыванию
    return np.array(GRADES)[np.minimum(idx, len(GRADES) - 1)]


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def validate(a) -> dict:
    (tr_reg, tr_cut), = _slices(a.train)
    (te_reg, te_cut), = _slices(a.test)
    train = load_slice(tr_reg, tr_cut, te_cut)           # обучение видит события только до года среза test
    test, _ = load_slice(te_reg, te_cut, LAST)
    m, feats = fit([train])
    P_tr = cumulative(m, train[0], feats, HORIZON_GRADE)
    edges = grade_edges(P_tr[:, HORIZON_GRADE - 1])
    H = LAST - te_cut
    P = cumulative(m, test, feats, H)
    res = {"train": f"{tr_cut} (события до {te_cut})", "test": f"{te_cut} (события до {LAST})",
           "n_train": len(train[0]), "n_test": len(test), "edges_p3": [round(float(e), 4) for e in edges], "horizons": []}
    for k in range(1, H + 1):
        yk = ((test["T"] > 0) & (test["T"] <= k)).values.astype(int)
        res["horizons"].append({"k": k, "base_rate": round(float(yk.mean()), 4), "mean_pred": round(float(P[:, k - 1].mean()), 4),
                                "auc": round(auc(yk, P[:, k - 1]), 3)})
    y5 = ((test["T"] > 0) & (test["T"] <= H)).values.astype(int)
    dec = pd.qcut(P[:, H - 1], 10, labels=False, duplicates="drop")
    res["calibration"] = [{"decile": int(d), "pred": round(float(P[dec == d, H - 1].mean()), 4),
                           "obs": round(float(y5[dec == d].mean()), 4), "n": int((dec == d).sum())} for d in np.unique(dec)]
    g = grade(P[:, HORIZON_GRADE - 1], edges)
    y3 = ((test["T"] > 0) & (test["T"] <= HORIZON_GRADE)).values.astype(int)
    res["grades"] = [{"grade": G, "n": int((g == G).sum()), "pred_p3": round(float(P[g == G, HORIZON_GRADE - 1].mean()), 4)
                      if (g == G).any() else None, "obs_3y": round(float(y3[g == G].mean()), 4) if (g == G).any() else None,
                      f"obs_{H}y": round(float(y5[g == G].mean()), 4) if (g == G).any() else None} for G in GRADES]
    test = test.assign(p1=P[:, 0], p3=P[:, min(2, H - 1)], pH=P[:, H - 1], grade=g)
    res["top_test"] = test.sort_values("p3", ascending=False).head(25)[["phrase", "grade", "p3", "pH", "T"]] \
        .round(3).to_dict("records")
    return res


def apply(a) -> None:
    from .corpus import REGISTRY
    train = [load_slice(reg, cut, LAST) for reg, cut in _slices(a.train)]
    m, feats = fit(train)
    edges = grade_edges(cumulative(m, train[0][0], feats, HORIZON_GRADE)[:, HORIZON_GRADE - 1])
    now = features(REGISTRY, int(os.environ.get("REGISTRY_CUTOFF", "2026")), with_axes=True)
    P = cumulative(m, now, feats, 5)
    cut = int(os.environ.get("REGISTRY_CUTOFF", "2026"))
    half = (P >= 0.5).argmax(axis=1) + 1
    rt = pd.DataFrame({"key": now.key, "rating": grade(P[:, HORIZON_GRADE - 1], edges), "p_takeoff_1y": P[:, 0],
                       "p_takeoff_3y": P[:, 2], "p_takeoff_5y": P[:, 4],
                       "takeoff_year": np.where(P[:, 4] >= 0.5, cut + half, 0)})
    rt.to_parquet(REGISTRY / "rating.parquet", index=False)
    print(rt.rating.value_counts().reindex(GRADES).to_dict())
    if a.bundle:
        b = Path(a.bundle)
        ax = pd.read_parquet(b / "axes.parquet")
        ax = ax.drop(columns=[c for c in rt.columns if c != "key" and c in ax.columns]).merge(rt, on="key", how="left")
        ax.to_parquet(b / "axes.parquet")
        with gzip.open(b / "cards.json.gz", "rt", encoding="utf-8") as fh:
            cards = json.load(fh)
        idx = rt.set_index("key")
        n = 0
        for k, c in cards.items():
            if k not in idx.index:
                continue
            x = idx.loc[k]
            c["rating"] = {"grade": x.rating, "p1": round(float(x.p_takeoff_1y), 3), "p3": round(float(x.p_takeoff_3y), 3),
                           "p5": round(float(x.p_takeoff_5y), 3), "year": int(x.takeoff_year) or None}
            yr = f", ожидаемо к {int(x.takeoff_year)}" if x.takeoff_year else ""
            fact = (f"Рейтинг {x.rating}: вероятность взлёта за 3 года {x.p_takeoff_3y:.0%}, за 5 лет {x.p_takeoff_5y:.0%}{yr} "
                    f"(модель времени до взлёта, проверена на срезе 2021)")
            c["facts"] = [f for f in (c.get("facts") or []) if not f.startswith("Рейтинг ")] + [fact]
            n += 1
        with gzip.open(b / "cards.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(cards, fh, ensure_ascii=False)
        print(f"набор {b}: рейтинг у {n} из {len(cards)} карточек; по рейтингам "
              f"{ax.rating.value_counts().reindex(GRADES).fillna(0).astype(int).to_dict()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "apply"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", default="")
    ap.add_argument("--bundle", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "validate":
        res = validate(a)
        print(json.dumps({k: v for k, v in res.items() if k != "top_test"}, ensure_ascii=False, indent=1))
        print(pd.DataFrame(res["top_test"]).to_string(index=False))
        if a.out:
            Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        apply(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
