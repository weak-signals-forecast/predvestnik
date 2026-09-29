"""Предсказатель взлёта: вероятность, что технология станет заметно популярнее через 3–5 лет.

    python -m wsignals.takeoff train     # обучение и временная проверка: reports/takeoff.md, models/takeoff_model.joblib
    python -m wsignals.takeoff score     # проверить реестр прогнозов по текущим данным

Модель этапа 1 отвечает на вопрос «ранняя это стадия или зрелая». Этот модуль отвечает на другой вопрос:
«вырастет ли технология». Идея перенесена из исследовательской версии проекта, где предсказатель обучался
на заморозках прошлых лет и проверялся на годах, которых не видел.

Схема:
  - заморозки 2019, 2020, 2021: признаки технологии восстанавливаются на конец года только по данным до него
    (те же, что в машине времени: публикации, репозитории, обсуждения разработчиков, статья в Википедии);
  - метка из будущего: взлёт, если среднегодовое число публикаций через 3–5 лет минимум втрое выше, чем
    за три года до заморозки, и за эти годы набралось не меньше 15 работ;
  - честная проверка: обучение на 2019–2020, тест на 2021, пятикратно по группам технологий, так что ни одна
    технология теста не встречается в обучении ни в каком году;
  - прогноз на сегодня фиксируется в реестре с датой и критерием, чтобы через три года его можно было проверить.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .timemachine import COUNTS, DATA, PIT_FEATURES, YEARS, pit_features

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "takeoff_model.joblib"
REPORT = ROOT / "reports" / "takeoff.md"
# Путь реестра переопределяется переменной окружения: живой запрос не должен дописывать файл,
# отслеживаемый git, и тем более файл внутри контейнера без тома.
REGISTRY = Path(os.getenv("WSIGNALS_FORECAST_REGISTRY", ROOT / "data" / "forecasts" / "registry.jsonl"))
FREEZES, TEST_FREEZE = [2019, 2020, 2021], 2021
HORIZON = (3, 5)
MIN_OUTCOME_DOCS = 15
# Порог роста выбран по проверке: при удвоении взлетала почти вся литература по ИИ 2021–2026 годов (база 63 %),
# при утроении база около половины, а модель различает лучше (AUC 0,78 против 0,76).
GROWTH = 3
YI = {y: i for i, y in enumerate(YEARS)}
ZF = [f"z_{k}" for k in PIT_FEATURES]
CRITERION = ("среднегодовое число научных публикаций через 3–5 лет минимум втрое выше, чем за три года до даты "
             f"прогноза, и не меньше {MIN_OUTCOME_DOCS} работ за эти годы")


def partial_factor(today: date | None = None) -> float:
    """Текущий год неполный: во сколько раз досчитать его до годового темпа."""
    today = today or date.today()
    return 12 / max(1.0, today.month - 1 + today.day / 30)


def series(row) -> list[float]:
    raw = row.get("oa_years")
    if not isinstance(raw, str):
        return [0.0] * len(YEARS)
    s = [float(x) for x in raw.split()]
    s[YI[2026]] *= partial_factor()
    return s


def label(row, F: int) -> tuple[int, float] | None:
    """Метка взлёта для заморозки F и отношение будущего темпа к прошлому."""
    s = series(row)
    base = sum(s[YI[F - 2]: YI[F] + 1]) / 3
    years = [y for y in range(F + HORIZON[0], F + HORIZON[1] + 1) if y in YI]
    if not years:
        return None
    future = [s[YI[y]] for y in years]
    ratio = (sum(future) / len(future) + 0.5) / (base + 0.5)
    return int(ratio >= GROWTH and sum(future) >= MIN_OUTCOME_DOCS), ratio


def frames() -> pd.DataFrame:
    df = pd.read_csv(DATA).merge(pd.read_csv(COUNTS), on="tech_id", how="left")
    rows = []
    for _, r in df.iterrows():
        s = series(r)
        for F in FREEZES:
            if sum(s[: YI[F] + 1]) < 3:
                continue                              # к дате заморозки технология ещё не видна в литературе
            lab = label(r, F)
            if lab is None:
                continue
            f = pit_features(r, F)
            rows.append({"tech_id": r["tech_id"], "group": r["group"], "name": r["name"], "class": r["label"],
                         "freeze": F, "y": lab[0], "ratio": lab[1], **{k: f[k] for k in PIT_FEATURES}})
    d = pd.DataFrame(rows)
    # Признаки относительно всех технологий того же года. GitHub, обсуждения и публикации в целом растут год от года,
    # и без этого модель, обученная на 2019–2021, видела бы «взлёт» почти у любой технологии 2026 года.
    for F, g in d.groupby("freeze"):
        mu, sd = g[PIT_FEATURES].mean(), g[PIT_FEATURES].std().replace(0, 1)
        d.loc[g.index, ZF] = ((g[PIT_FEATURES] - mu) / sd).values
    return d


def reference_2026() -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Среднее и разброс признаков на 2026 год по всем технологиям датасета: база для прогноза на сегодня."""
    df = pd.read_csv(DATA).merge(pd.read_csv(COUNTS), on="tech_id", how="left")
    rows = []
    for _, r in df.iterrows():
        if sum(series(r)) < 3:
            continue
        rr = r.copy()
        rr["oa_years"] = " ".join(str(x) for x in series(r))
        f = pit_features(rr, 2026)
        rows.append({k: f[k] for k in PIT_FEATURES})
    ref = pd.DataFrame(rows)
    return ref.mean(), ref.std().replace(0, 1), ref


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000))


def precision_at(y, p, k):
    order = np.argsort(-p)[:k]
    return float(np.mean(np.asarray(y)[order]))


def bootstrap(y, p, k, n=500, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    lifts = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        base = y[ix].mean()
        if base == 0:
            continue
        lifts.append(precision_at(y[ix], p[ix], k) / base)
    return float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5))


def train():
    d = frames()
    # Честная схема: технологии делятся на 5 групп. Модель обучается на заморозках 2019–2020 технологий из четырёх
    # групп и предсказывает заморозку 2021 для пятой. Каждая технология теста не видна модели ни в каком году,
    # а все признаки обучения старше даты теста.
    from sklearn.model_selection import GroupKFold
    test = d[d["freeze"] == TEST_FREEZE].reset_index(drop=True)
    p = np.zeros(len(test))
    n_train = []
    for tr_groups, te_idx in ((set(test["group"].iloc[tr]), te) for tr, te in
                              GroupKFold(5).split(test, groups=test["group"])):
        train_ = d[(d["freeze"] < TEST_FREEZE) & d["group"].isin(tr_groups)]
        n_train.append(len(train_))
        m = model().fit(train_[ZF].values, train_["y"].values)
        p[te_idx] = m.predict_proba(test.loc[te_idx, ZF].values)[:, 1]
    base = test["y"].mean()
    auc = roc_auc_score(test["y"], p) if test["y"].nunique() > 1 else float("nan")

    # Для сравнения: без разделения технологий (та же технология в 2019 году в обучении и в 2021 в тесте)
    old = d[d["freeze"] < TEST_FREEZE]
    p2 = model().fit(old[ZF].values, old["y"].values).predict_proba(test[ZF].values)[:, 1]
    auc2 = roc_auc_score(test["y"], p2) if test["y"].nunique() > 1 else float("nan")
    train_ = old

    L = ["# Предсказатель взлёта: временная проверка\n",
         f"Взлёт: {CRITERION}. Признаки на дату заморозки восстанавливаются только по данным до неё и берутся "
         "относительно всех технологий того же года, чтобы общий рост GitHub и публикаций не выдавался за взлёт.\n",
         "| Заморозка | Технологий видно | Доля взлетевших |\n|---|---|---|"]
    for F, g in d.groupby("freeze"):
        L.append(f"| {F} | {len(g)} | {g['y'].mean():.0%} |")
    L.append(f"\n## Проверка на невиданном году\n")
    L.append(f"Обучение на заморозках 2019–2020, тест на заморозке 2021 ({len(test)} технологий), пятикратно по группам технологий: "
             f"в каждом прогоне около {int(np.mean(n_train))} обучающих примеров, и ни одна технология теста не встречается в обучении.\n")
    L.append("| k | Точность топ-k | База | Подъём | 95 % интервал подъёма |\n|---|---|---|---|---|")
    for k in (10, 20, 30):
        if k > len(test):
            continue
        pk = precision_at(test["y"], p, k)
        lo, hi = bootstrap(test["y"], p, k)
        L.append(f"| {k} | {pk:.0%} | {base:.0%} | ×{pk / base:.1f} | ×{lo:.1f}–{hi:.1f} |")
    L.append(f"\nAUC {auc:.2f}. Если не разделять технологии между обучением и тестом, AUC {auc2:.2f}: близкие значения "
             "означают, что модель опирается на общий профиль роста, а не на знание конкретных технологий.\n")

    by_class = test.assign(p=p).groupby("class")[["p", "y"]].mean()
    L.append("## Как прогноз соотносится с классами разметки\n")
    L.append("| Класс | Средняя вероятность взлёта в 2021 | Реально взлетели к 2024–2026 |\n|---|---|---|")
    names = {"weak_signal": "слабые сигналы 2026", "mainstream": "зрелые технологии", "faded_hype": "угасший хайп"}
    for c, r in by_class.iterrows():
        L.append(f"| {names.get(c, c)} | {r['p']:.0%} | {r['y']:.0%} |")

    top = test.assign(p=p).sort_values("p", ascending=False).head(15)
    L.append("\n## Что модель предсказала бы в конце 2021 года\n")
    L.append("| Технология | Вероятность взлёта | Рост публикаций к 2024–2026 | Исход |\n|---|---|---|---|")
    for _, r in top.iterrows():
        L.append(f"| {r['name'][:70]} | {r['p']:.0%} | ×{r['ratio']:.1f} | {'взлетела' if r['y'] else 'нет'} |")

    final = model().fit(d[ZF].values, d["y"].values)
    coef = pd.Series(final.named_steps["logisticregression"].coef_[0], index=PIT_FEATURES).sort_values()
    L.append("\n## Что предсказывает взлёт\n")
    L.append("Стандартизованные коэффициенты итоговой модели, обученной на всех заморозках.\n")
    L.append("| Признак | Коэффициент |\n|---|---|")
    from .features import NAMES_RU
    for k, v in coef.iloc[::-1].items():
        L.append(f"| {NAMES_RU.get(k, k)} | {v:+.2f} |")
    L.append("\n## Ограничения\n")
    L.append("- Метка измеряет рост научных публикаций, а не рынок или деньги. Это видно по таблице выше: о NFT-маркетплейсах "
             "и play-to-earn научных работ стало больше, хотя как рынок это угасший хайп. Поэтому прогноз взлёта показывается "
             "рядом с оценкой стадии, а не вместо неё.\n"
             "- Выборка небольшая, интервалы широкие: прогноз надо читать как ранжирование, а не как точную вероятность.\n"
             "- Модель обучена на 2019–2021 годах, а применяется к 2026: объёмы GitHub и публикаций с тех пор выросли, "
             "это может смещать вероятности.")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    mu, sd, ref = reference_2026()
    ref_p = np.sort(final.predict_proba(((ref - mu) / sd)[PIT_FEATURES].values)[:, 1])
    joblib.dump({"model": final, "features": PIT_FEATURES, "ref_mean": mu.to_dict(), "ref_std": sd.to_dict(),
                 "ref_probabilities": ref_p.tolist(),
                 "trained": date.today().isoformat(), "criterion": CRITERION, "test_auc": auc}, MODEL)
    print("\n".join(L[:20]))


# ---------- прогноз для открытого запроса и реестр ----------

_bundle = None


def predict(features: dict) -> dict | None:
    """Вероятность взлёта к 2029–2031 для признаков, собранных открытым запросом."""
    global _bundle
    if _bundle is None:
        if not MODEL.exists():
            return None
        _bundle = joblib.load(MODEL)
    row = pd.Series({**features})
    if not isinstance(row.get("oa_years"), str):
        return None
    s = series(row)
    row["oa_years"] = " ".join(str(x) for x in s)
    f = pit_features(row, 2026)
    mu, sd = _bundle["ref_mean"], _bundle["ref_std"]
    z = [(f[k] - mu[k]) / (sd[k] or 1) for k in _bundle["features"]]
    p = float(_bundle["model"].predict_proba(np.array([z]))[0, 1])
    base = sum(s[YI[2024]: YI[2026] + 1]) / 3
    ref = _bundle.get("ref_probabilities") or []
    pct = float(np.searchsorted(ref, p, side="right") / len(ref)) if ref else None
    return {"probability": round(p, 3), "percentile": round(pct, 2) if pct is not None else None, "horizon": "2029–2031",
            "baseline_per_year": round(base, 1), "criterion": CRITERION}


def register(query: str, cards: list[dict]) -> None:
    """Реестр прогнозов: дата, запрос, технология, вероятность, базовый темп и критерий успеха."""
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", encoding="utf-8") as fh:
        for c in cards:
            t = c.get("takeoff")
            if not t:
                continue
            fh.write(json.dumps({"date": date.today().isoformat(), "query": query, "technology": c["technology"],
                                 "original": c.get("technology_original"), "probability": t["probability"],
                                 "baseline_per_year": t["baseline_per_year"], "horizon": t["horizon"],
                                 "criterion": t["criterion"]}, ensure_ascii=False) + "\n")


def score():
    """Проверить прогнозы реестра по текущим данным OpenAlex. Имеет смысл с 2029 года."""
    from .sources import openalex
    if not REGISTRY.exists():
        print("реестр пуст")
        return
    rows = [json.loads(line) for line in REGISTRY.read_text(encoding="utf-8").splitlines() if line.strip()]
    this_year = date.today().year
    for r in rows:
        made = int(r["date"][:4])
        years = [y for y in range(made + HORIZON[0], made + HORIZON[1] + 1) if y < this_year]
        if not years:
            print(f"{r['technology']}: прогноз от {r['date']}, проверка возможна с {made + HORIZON[0] + 1} года")
            continue
        counts = openalex.stats(r["original"])["oa_years"]
        now = sum(counts[YEARS.index(y)] for y in years if y in YEARS) / len(years)
        ok = now >= GROWTH * r["baseline_per_year"]
        print(f"{r['technology']}: вероятность {r['probability']:.0%}, темп был {r['baseline_per_year']}, стал {now:.1f}: "
              f"{'сбылось' if ok else 'не сбылось'}")


if __name__ == "__main__":
    {"train": train, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "train"]()
