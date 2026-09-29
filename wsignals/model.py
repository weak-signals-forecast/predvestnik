"""Модель этапа 1: слабый сигнал против зрелой технологии и угасшего хайпа.

    python -m wsignals.model            # обучение, оценка, отчёт reports/model_report.md, модель models/signal_model.joblib

Выбор метода. Две модели на одних признаках:
  логистическая регрессия   прозрачна: вклад каждого признака в решение считается как коэффициент × значение;
  градиентный бустинг       ловит нелинейности (например, «рост высокий, но статья в Википедии уже есть»).
Итоговая модель выбирается по F1 класса «слабый сигнал» на кросс-валидации, при равенстве берётся более простая.

Честная оценка:
  - повторённая стратифицированная кросс-валидация по группам (5 фолдов × 10 повторов): почти одинаковые
    технологии из таблицы всегда в одном фолде;
  - кросс-валидация «одна область на тест»: модель не видит, например, финтех при обучении и проверяется на нём.
    Это ближе всего к открытому запросу жюри по новому направлению;
  - 95% интервалы метрик по повторам.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, NAMES_RU

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset" / "technologies.csv"
REPORT = ROOT / "reports" / "model_report.md"
MODEL = ROOT / "models" / "signal_model.joblib"
CONFIDENT = 0.75


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["y"] = (df["label"] == "weak_signal").astype(int)
    df[FEATURES] = df[FEATURES].astype(float).fillna(0.0)
    return df


def candidates() -> dict:
    return {
        "Логистическая регрессия": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000)),
        "Градиентный бустинг": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250,
                                                              min_samples_leaf=10, class_weight="balanced", random_state=0),
    }


def metrics(y, p, thr=0.5) -> dict:
    yhat = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0), "f1": f1_score(y, yhat, zero_division=0)}


def repeated_cv(model, df, repeats=10) -> tuple[pd.DataFrame, np.ndarray]:
    rows, oof_sum = [], np.zeros(len(df))
    X, y, g = df[FEATURES].values, df["y"].values, df["group"].values
    for r in range(repeats):
        oof = np.zeros(len(df))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, y, g):
            m = clone(model).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        rows.append(metrics(y, oof))
        oof_sum += oof
    return pd.DataFrame(rows), oof_sum / repeats


def area_cv(model, df) -> pd.DataFrame:
    rows = []
    for area in sorted(df["area"].unique()):
        tr, te = df[df["area"] != area], df[df["area"] == area]
        m = clone(model).fit(tr[FEATURES].values, tr["y"].values)
        p = m.predict_proba(te[FEATURES].values)[:, 1]
        rows.append({"область": area, "n": len(te), **metrics(te["y"].values, p)})
    return pd.DataFrame(rows)


def lr_contributions(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Вклад признаков в логит для каждой строки: коэффициент × стандартизованное значение."""
    scaler, lr = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregression"]
    Z = (X[FEATURES].values - scaler.mean_) / scaler.scale_
    return pd.DataFrame(Z * lr.coef_[0], columns=FEATURES, index=X.index)


def ci(s: pd.Series) -> str:
    return f"{s.mean():.1%} [{s.quantile(0.025):.1%}; {s.quantile(0.975):.1%}]"


def main():
    df = load()
    L = [f"# Отчёт о модели этапа 1\n",
         f"Дата снимка признаков: {df['snapshot'].iloc[0]}. Технологий {len(df)}: "
         + ", ".join(f"{k} {v}" for k, v in df["label"].value_counts().items())
         + ". Положительный класс: 100 технологий из таблицы организаторов. Отрицательные: зрелые технологии и угасший хайп "
           "по тем же шести областям. Признаки собраны одинаково для всех классов из OpenAlex, Google News, Hacker News, "
           "GitHub и Википедии.\n"]
    results, oofs = {}, {}
    for name, model in candidates().items():
        res, oof = repeated_cv(model, df)
        results[name], oofs[name] = res, oof
    L.append("## Кросс-валидация: 5 фолдов по группам × 10 повторов\n")
    L.append("Среднее и 95 % интервал по повторам. Класс «слабый сигнал» положительный, порог 0,5.\n")
    L.append("| Модель | Accuracy | Precision | Recall | F1 |\n|---|---|---|---|---|")
    for name, res in results.items():
        L.append(f"| {name} | {ci(res['accuracy'])} | {ci(res['precision'])} | {ci(res['recall'])} | {ci(res['f1'])} |")
    best = max(results, key=lambda k: (round(results[k]["f1"].mean(), 2), k == "Логистическая регрессия"))
    model = candidates()[best]
    L.append(f"\nВыбрана модель: **{best}** (лучший F1; при равенстве выбирается более прозрачная).\n")

    y, oof = df["y"].values, oofs[best]
    cm = confusion_matrix(y, (oof >= 0.5).astype(int))
    L.append("## Матрица ошибок (усреднённые вне-фолдовые вероятности)\n")
    L.append(f"| | Предсказано: не сигнал | Предсказано: слабый сигнал |\n|---|---|---|\n"
             f"| Факт: не сигнал | {cm[0,0]} | {cm[0,1]} |\n| Факт: слабый сигнал | {cm[1,0]} | {cm[1,1]} |\n")
    by_class = df.assign(p=oof).groupby("label")["p"].agg(["mean", lambda s: (s >= 0.5).mean()])
    L.append("Средняя вероятность «слабого сигнала» и доля отнесённых к нему, по исходным классам:\n")
    L.append("| Класс | Средняя вероятность | Доля ≥ 0,5 |\n|---|---|---|")
    for lab, r in by_class.iterrows():
        L.append(f"| {lab} | {r['mean']:.2f} | {r.iloc[1]:.0%} |")

    ws = df["label"] == "weak_signal"
    L.append("\n## Если проверка идёт только по списку организаторов\n")
    L.append("Организаторы подтвердили, что в их таблице все строки это слабые сигналы: отрицательных примеров не будет. "
             "Тогда метрика на скрытой выборке это доля сигналов, которые модель узнала. Ниже вне-фолдовая доля на 100 "
             "технологиях таблицы при разных порогах, рядом цена такого порога на наших отрицательных примерах.\n")
    L.append("| Порог | Узнано сигналов организаторов | Ложно принято: зрелые | Ложно принято: угасший хайп |\n|---|---|---|---|")
    for thr in (0.5, 0.4, 0.35, 0.3):
        L.append(f"| {thr:.2f} | {(oof[ws.values] >= thr).mean():.0%} | {(oof[(df['label'] == 'mainstream').values] >= thr).mean():.0%} | "
                 f"{(oof[(df['label'] == 'faded_hype').values] >= thr).mean():.0%} |")
    L.append("\nПорог 0,5 оставлен рабочим: он даёт высокую долю узнанных сигналов и при этом отсекает большинство зрелых "
             "технологий, что требуется в открытом запросе по ТЗ.\n")

    L.append("\n## Перенос на новую область: одна область в тесте, пять в обучении\n")
    L.append("Самая строгая проверка: так модель работает на открытом запросе жюри по направлению, которого не видела.\n")
    ac = area_cv(model, df)
    L.append("| Область | n | Accuracy | Precision | Recall | F1 |\n|---|---|---|---|---|---|")
    for _, r in ac.iterrows():
        L.append(f"| {r['область']} | {r['n']} | {r['accuracy']:.0%} | {r['precision']:.0%} | {r['recall']:.0%} | {r['f1']:.0%} |")
    L.append(f"| **среднее** | | {ac['accuracy'].mean():.0%} | {ac['precision'].mean():.0%} | {ac['recall'].mean():.0%} | {ac['f1'].mean():.0%} |")

    final = clone(model).fit(df[FEATURES].values, y)
    imp = permutation_importance(final, df[FEATURES].values, y, scoring="f1", n_repeats=20, random_state=0)
    order = np.argsort(-imp.importances_mean)
    L.append("\n## Интерпретируемость: какие признаки решают\n")
    L.append("Permutation importance: насколько падает F1, если перемешать признак.\n")
    L.append("| Признак | Падение F1 |\n|---|---|")
    for i in order[:12]:
        L.append(f"| {NAMES_RU[FEATURES[i]]} | {imp.importances_mean[i]:+.3f} |")
    lr = candidates()["Логистическая регрессия"].fit(df[FEATURES].values, y)
    coef = pd.Series(lr.named_steps["logisticregression"].coef_[0], index=FEATURES).sort_values()
    L.append("\nНаправление влияния (логистическая регрессия, стандартизованные коэффициенты): "
             "минус толкает к «зрелое или хайп», плюс к «слабый сигнал».\n")
    L.append("| Толкает к «не сигнал» | коэф. | Толкает к «слабый сигнал» | коэф. |\n|---|---|---|---|")
    neg, pos = coef.head(6), coef.tail(6)[::-1]
    for (fn, cn), (fp, cp) in zip(neg.items(), pos.items()):
        L.append(f"| {NAMES_RU[fn]} | {cn:+.2f} | {NAMES_RU[fp]} | {cp:+.2f} |")

    pos_df = df[df["y"] == 1].assign(p=oof[df["y"].values == 1])
    rho = pos_df[["p", "org_score", "org_stage"]].corr(method="spearman")
    L.append("\n## Согласованность с оценкой организаторов\n")
    r_score, r_stage = rho.loc["p", "org_score"], rho.loc["p", "org_stage"]
    verdict = ("связи практически нет" if max(abs(r_score), abs(r_stage)) < 0.2 else
               "связь положительная" if r_score > 0 else "связь отрицательная")
    L.append(f"Среди 100 слабых сигналов ранговая корреляция вероятности модели с баллом «стадия + тренд» из таблицы: "
             f"{r_score:+.2f}, со стадией развития: {r_stage:+.2f}, {verdict}. Модель отвечает на вопрос «ранняя это стадия "
             "или зрелая», а балл таблицы ранжирует технологии внутри класса слабых сигналов. Это разные шкалы, поэтому "
             "балл организаторов не используется ни в признаках, ни в разметке.\n")
    words = df.assign(w=df["query"].str.split().str.len()).groupby("label")["w"].mean()
    L.append(f"Проверка смещения разметки: средняя длина поисковой фразы у слабых сигналов {words.get('weak_signal', 0):.1f} слова, "
             f"у зрелых {words.get('mainstream', 0):.1f}, у угасшего хайпа {words.get('faded_hype', 0):.1f}. Длинная фраза находит меньше "
             "документов, поэтому часть ложных срабатываний это зрелые технологии, заданные узкой фразой. Длина фразы как признак "
             "не улучшила кросс-валидацию и в модель не добавлена, чтобы модель не училась на манере формулировать запросы.\n")

    errors = df.assign(p=oof).query("(y == 1 and p < 0.5) or (y == 0 and p >= 0.5)").sort_values("p")
    L.append("## Разбор ошибок\n")
    L.append("| Технология | Класс | Вероятность |\n|---|---|---|")
    for _, r in errors.iterrows():
        L.append(f"| {r['name'][:80]} | {r['label']} | {r['p']:.2f} |")
    L.append("\n## Ограничения\n")
    L.append("- Отрицательные примеры составлены командой по явным критериям (массовое внедрение, стандарт, прошедший пик хайпа); "
             "список и критерии открыты в `data/labels/negatives.yaml`.\n"
             "- Признаки сняты на одну дату, модель отличает раннюю стадию от зрелой, но не прогнозирует будущее.\n"
             "- Малый след в источниках характерен и для слабых сигналов, и для случайных фраз. Поэтому на открытом запросе "
             "модель применяется только к кандидатам, подтверждённым минимум двумя независимыми источниками.")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    # Интервал уверенности: та же модель на 50 бутстрэп-выборках. Узкий интервал значит, что оценка не держится
    # на паре обучающих примеров.
    rng = np.random.default_rng(0)
    bag = []
    for _ in range(50):
        idx = rng.integers(0, len(df), len(df))
        if df["y"].values[idx].min() == df["y"].values[idx].max():
            continue
        bag.append(clone(model).fit(df[FEATURES].values[idx], y[idx]))
    joblib.dump({"model": final, "explainer": lr, "bag": bag, "features": FEATURES, "name": best,
                 "trained": date.today().isoformat(), "confident": CONFIDENT}, MODEL)
    summary = {k: {m: round(float(v[m].mean()), 3) for m in ["accuracy", "precision", "recall", "f1"]} for k, v in results.items()}
    summary["best"] = best
    summary["area_cv_mean"] = {m: round(float(ac[m].mean()), 3) for m in ["accuracy", "precision", "recall", "f1"]}
    (ROOT / "reports" / "model_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
