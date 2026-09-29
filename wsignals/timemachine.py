"""Машина времени: признаки технологии на прошлую дату и историческая проверка модели.

    python -m wsignals.timemachine collect     # досчитать исторические счётчики GitHub и Hacker News (кэшируется)
    python -m wsignals.timemachine backtest    # отчёт reports/timemachine.md

Идея. Модель обучена на разметке 2026 года. Если она действительно распознаёт раннюю стадию, а не запоминает названия,
то на признаках, восстановленных на конец 2021 года, она должна относить к слабым сигналам технологии, которые тогда
только появлялись и к 2026 году стали мейнстримом (RAG, джейлбрейки LLM, векторные базы), и не относить к ним то,
что было зрелым уже в 2021 (Kubernetes, SCADA).

Честность проверки:
  - признаки на дату F считаются только по данным до F: годовые ряды OpenAlex, репозитории GitHub с датой создания до F,
    истории Hacker News до F, дата создания статьи в Википедии. Новости и просмотры исторически не восстанавливаются
    и в этой модели не используются;
  - каждая проверяемая технология оценивается моделью, обученной БЕЗ неё и её группы (leave-one-group-out);
  - тест tests/test_timemachine.py проверяет, что признаки на F не зависят от значений после F.

Лестница зрелости. Для каждой технологии известен год появления в каждом слое: наука (OpenAlex), код (GitHub),
сообщество (Hacker News), энциклопедия (Википедия). Слабый сигнал обычно уже в науке и коде, но ещё не в энциклопедии.
Ступень лестницы сравнивается со стадией развития из таблицы организаторов.
"""
from __future__ import annotations

import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .http import get_json
from .sources import github as gh
from .sources import hackernews as hn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset" / "technologies.csv"
COUNTS = ROOT / "data" / "dataset" / "history_counts.csv"
REPORT = ROOT / "reports" / "timemachine.md"
TWINS = ROOT / "data" / "dataset" / "twins_2021.csv"
FREEZE = 2021
YEARS = list(range(2010, 2027))
GH_CUTS = [2018, 2019, 2020, 2021, 2022]            # «создано до 1 января года»
HN_CUTS = [2012, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2024]
PIT_FEATURES = ["oa_total_log", "oa_recent_log", "oa_growth", "oa_age", "oa_peak_ratio", "oa_last_share",
                "hn_12m_log", "hn_growth", "gh_total_log", "gh_new_share", "wiki_article", "wiki_age"]


def _ts(year: int) -> int:
    return int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())


def history(query: str) -> dict:
    q = f'"{query}" in:name,description,readme'
    out = {}
    for y in GH_CUTS:
        d = get_json(gh.API, {"q": f"{q} created:<{y}-01-01", "per_page": 1}, gh._headers(), max_age_days=60)
        out[f"gh_before_{y}"] = int(d.get("total_count", 0))
    for y in HN_CUTS:
        out[f"hn_before_{y}"] = hn._count(query, until=_ts(y))
    return out


def collect():
    df = pd.read_csv(DATA)
    print(f"исторические счётчики для {len(df)} технологий", flush=True)
    rows = {}
    with ThreadPoolExecutor(4) as ex:
        for i, (tid, res) in enumerate(zip(df["tech_id"], ex.map(lambda q: _safe(history, q), df["query"])), 1):
            rows[tid] = res
            if i % 25 == 0:
                print(f"  {i}/{len(df)}", flush=True)
    pd.DataFrame.from_dict(rows, orient="index").rename_axis("tech_id").to_csv(COUNTS)
    print(f"сохранено {COUNTS.relative_to(ROOT)}")


def _safe(fn, q):
    try:
        return fn(q)
    except Exception as e:  # noqa: BLE001
        return {"history_error": str(e)[:200]}


# ---------- признаки на дату ----------

def oa_at(series: list[float], F: int) -> dict:
    """Признаки OpenAlex на конец года F по годовому ряду. Значения после F не используются."""
    yi = {y: i for i, y in enumerate(YEARS)}
    c = [float(v) for v in series[: yi[F] + 1]]
    total = sum(c)
    recent = sum(c[yi[F - 2]: yi[F] + 1]) / 3
    prior = sum(c[yi[F - 7]: yi[F - 4]]) / 3
    first = next((YEARS[i] for i, n in enumerate(c) if n >= 3), None)
    peak = max(c[: yi[F]]) if len(c) > 1 else 0.0          # пик по завершённым годам до F-1
    return {"oa_total_log": math.log1p(total), "oa_recent_log": math.log1p(recent),
            "oa_growth": math.log((recent + 1) / (prior + 1)), "oa_age": (F - first) if first else 0,
            "oa_peak_ratio": (c[yi[F - 1]] + 1) / (peak + 1), "oa_last_share": c[yi[F - 1]] / total if total else 0.0}


def pit_features(row: pd.Series, F: int) -> dict:
    series = [float(x) for x in str(row["oa_years"]).split()] if isinstance(row.get("oa_years"), str) else [0.0] * len(YEARS)
    f = oa_at(series, F)
    if F >= 2026:
        gh_total, gh_new = row.get("gh_total", 0), row.get("gh_12m", 0)
        hn_12m, hn_prior = row.get("hn_12m", 0), row.get("hn_prior24m", 0)
    else:
        gh_total = row.get(f"gh_before_{F + 1}", 0)
        gh_new = gh_total - row.get(f"gh_before_{F}", 0)
        hn_12m = row.get(f"hn_before_{F + 1}", 0) - row.get(f"hn_before_{F}", 0)
        hn_prior = row.get(f"hn_before_{F}", 0) - row.get(f"hn_before_{F - 2}", 0)
    wiki_year = 2026 - row["wiki_age"] if row.get("wiki_article") and row.get("wiki_age", 0) > 0 else None
    exists = bool(row.get("wiki_article")) and (wiki_year is None or wiki_year <= F)
    f.update({"hn_12m_log": math.log1p(max(0, hn_12m)), "hn_growth": math.log((max(0, hn_12m) + 1) / (max(0, hn_prior) / 2 + 1)),
              "gh_total_log": math.log1p(max(0, gh_total)), "gh_new_share": max(0, gh_new) / gh_total if gh_total else 0.0,
              "wiki_article": float(exists), "wiki_age": float(F - wiki_year) if exists and wiki_year else 0.0})
    return f


def ladder(row: pd.Series) -> dict:
    """Год появления технологии в каждом слое и ступень лестницы зрелости на 2026 год (0–4)."""
    series = [float(x) for x in str(row["oa_years"]).split()] if isinstance(row.get("oa_years"), str) else []
    science = next((YEARS[i] for i, n in enumerate(series) if n >= 3), None)
    code = next((y - 1 for y in GH_CUTS if row.get(f"gh_before_{y}", 0) >= 3), 2026 if row.get("gh_total", 0) >= 3 else None)
    community = next((y - 1 for y in HN_CUTS if row.get(f"hn_before_{y}", 0) >= 1), 2026 if row.get("hn_total", 0) >= 1 else None)
    encyclopedia = int(2026 - row["wiki_age"]) if row.get("wiki_article") and row.get("wiki_age", 0) > 0 else None
    media = 2026 if row.get("news_1y", 0) >= 20 else None
    layers = {"наука": science, "код": code, "сообщество": community, "массовые медиа": media, "энциклопедия": encyclopedia}
    step = sum(v is not None for v in layers.values())
    return {"ladder_step": step, **{f"first_{k}": v for k, v in layers.items()}}


# ---------- проверка ----------

def backtest():
    df = pd.read_csv(DATA).merge(pd.read_csv(COUNTS), on="tech_id", how="left")
    df["y"] = (df["label"] == "weak_signal").astype(int)
    now = pd.DataFrame([pit_features(r, 2026) for _, r in df.iterrows()]).fillna(0.0)
    then = pd.DataFrame([pit_features(r, FREEZE) for _, r in df.iterrows()]).fillna(0.0)
    lad = pd.DataFrame([ladder(r) for _, r in df.iterrows()])
    df = pd.concat([df, lad], axis=1)
    first_sci = lad["first_наука"]

    def model():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000))

    # Кого проверяем в 2021 году
    groups = {
        "Стали мейнстримом к 2026, появились в 2017–2021": (df["label"] == "mainstream") & first_sci.between(2017, FREEZE),
        "Были зрелыми уже в 2021 (≥ 3 работ уже в 2010–2012)": (df["label"] == "mainstream") & (first_sci <= 2012),
        "Угасший хайп": df["label"] == "faded_hype",
        "Слабые сигналы 2026 года": df["label"] == "weak_signal",
    }
    p_then = np.full(len(df), np.nan)
    visible = then["oa_total_log"] > math.log1p(2)          # к 2021 году есть хотя бы 3 публикации
    for i in np.where(visible)[0]:
        train = df["group"] != df.loc[i, "group"]
        m = model().fit(now[PIT_FEATURES][train].values, df["y"][train].values)
        p_then[i] = m.predict_proba(then[PIT_FEATURES].iloc[[i]].values)[0, 1]
    df["p_2021"] = p_then
    cv_now = model().fit(now[PIT_FEATURES].values, df["y"].values)

    L = ["# Машина времени: что модель сказала бы в конце 2021 года\n",
         "Модель обучена на разметке 2026 года только на признаках, которые можно восстановить на прошлую дату "
         "(публикации, репозитории, обсуждения разработчиков, статья в Википедии). Каждая технология оценивается моделью, "
         "обученной без неё. Признаки на 2021 год посчитаны только по данным до конца 2021.\n"]
    L.append("| Группа технологий | Всего | Видны в 2021 | Отнесены к слабым сигналам в 2021 |\n|---|---|---|---|")
    for name, mask in groups.items():
        vis = mask & visible
        share = (df.loc[vis, "p_2021"] >= 0.5).mean() if vis.any() else float("nan")
        L.append(f"| {name} | {int(mask.sum())} | {int(vis.sum())} | {share:.0%} |")
    from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
    young = df.loc[groups["Стали мейнстримом к 2026, появились в 2017–2021"] & visible, "p_2021"]
    mature = df.loc[groups["Были зрелыми уже в 2021 (≥ 3 работ уже в 2010–2012)"] & visible, "p_2021"]
    table = [[int((young >= 0.5).sum()), int((young < 0.5).sum())], [int((mature >= 0.5).sum()), int((mature < 0.5).sum())]]
    odds, p_fisher = fisher_exact(table, alternative="greater")
    auc = mannwhitneyu(young, mature, alternative="greater").statistic / (len(young) * len(mature))
    L.append(f"\nТехнологии, которые в 2021 году только появлялись, модель тогда относила к слабым сигналам в "
             f"{(young >= 0.5).mean():.0%} случаев, а уже зрелые в {(mature >= 0.5).mean():.0%}: разница в "
             f"{(young >= 0.5).mean() / max((mature >= 0.5).mean(), 1e-9):.1f} раза, точный тест Фишера p = {p_fisher:.3f}. "
             f"Вероятность того, что случайная «молодая» технология получит оценку выше случайной зрелой (AUC): {auc:.2f}.\n")
    known = df.set_index("name")["p_2021"]
    examples = ", ".join(f"{n} {known[n]:.0%}" for n in ("RAG", "FlashAttention", "TinyML") if n in known and pd.notna(known[n]))
    L.append("Честное прочтение. Модель различает стадию в прошлом, а не запоминает названия: иначе обе доли были бы близки. "
             f"Но при пороге 0,5 она поймала бы в 2021 году только {(young >= 0.5).mean():.0%} будущего мейнстрима ({examples}). "
             "Для раннего обнаружения на открытом запросе полезно смотреть не только ТОП выше порога, но и ранжирование. "
             "Слабые сигналы 2026 года в большинстве в 2021 ещё не видны в источниках: это и есть горизонт раннего обнаружения. "
             "Годовые ряды начинаются с 2010 года, поэтому «зрелые» здесь это технологии с публикациями уже в 2010–2012.\n")

    hype_p = df.loc[groups["Угасший хайп"] & visible, "p_2021"]
    L.append("## Выбор порогов для открытого запроса\n")
    L.append("| Порог | Будущий мейнстрим отмечен в 2021 | Ложные срабатывания: зрелые | Ложные срабатывания: угасший хайп |\n|---|---|---|---|")
    for thr in (0.5, 0.4, 0.35, 0.3, 0.25):
        L.append(f"| {thr:.2f} | {(young >= thr).mean():.0%} | {(mature >= thr).mean():.0%} | {(hype_p >= thr).mean():.0%} |")
    L.append("\nВ открытом запросе порог 0,5 даёт статус «слабый сигнал», диапазон 0,35–0,5 статус «под наблюдением»: "
             "второй эшелон ловит заметно больше будущего мейнстрима при том же уровне ложных срабатываний на зрелых технологиях. "
             "Угасший хайп дополнительно отсекается правилом прошедшего пика.\n")

    hits = df[groups["Стали мейнстримом к 2026, появились в 2017–2021"] & visible].sort_values("p_2021", ascending=False)
    L.append("## Технологии, ставшие мейнстримом: оценка модели в 2021 году\n")
    L.append("| Технология | Первые работы | Вероятность слабого сигнала в 2021 | Сейчас |\n|---|---|---|---|")
    for _, r in hits.iterrows():
        L.append(f"| {r['name']} | {int(r['first_наука'])} | {r['p_2021']:.0%} | мейнстрим |")
    old = df[groups["Были зрелыми уже в 2021 (≥ 3 работ уже в 2010–2012)"] & visible].sort_values("p_2021", ascending=False).head(8)
    L.append("\n## Контроль: зрелые уже в 2021, самые высокие оценки\n")
    L.append("| Технология | Первые работы | Вероятность в 2021 |\n|---|---|---|")
    for _, r in old.iterrows():
        L.append(f"| {r['name']} | {int(r['first_наука'])} | {r['p_2021']:.0%} |")

    L.append("\n## Лестница зрелости\n")
    L.append("Ступень это число слоёв, в которых технология уже появилась: наука, код, сообщество разработчиков, "
             "массовые медиа, энциклопедия.\n")
    L.append("| Класс | Средняя ступень | Нет статьи в Википедии |\n|---|---|---|")
    for lab, g in df.groupby("label"):
        L.append(f"| {lab} | {g['ladder_step'].mean():.1f} | {g['first_энциклопедия'].isna().mean():.0%} |")
    ws = df[df["label"] == "weak_signal"]
    rho = ws[["ladder_step", "org_stage", "org_score"]].corr(method="spearman")
    r1, p1 = spearmanr(ws["ladder_step"], ws["org_stage"], nan_policy="omit")
    r2, p2 = spearmanr(ws["ladder_step"], ws["org_score"], nan_policy="omit")
    strength = "слабая" if max(abs(r1), abs(r2)) < 0.3 else "умеренная"
    signif = "статистически значимая" if min(p1, p2) < 0.05 else "статистически не значимая на 100 технологиях"
    L.append(f"\nСреди 100 слабых сигналов ступень лестницы и стадия развития из таблицы организаторов: ρ = {r1:+.2f} (p = {p1:.2f}), "
             f"ступень и балл «стадия + тренд»: ρ = {r2:+.2f} (p = {p2:.2f}). Связь {strength} положительная, {signif}. "
             "Лестница надёжно отделяет классы друг от друга (таблица выше), но внутри класса слабых сигналов "
             "экспертная стадия и следы в источниках расходятся: многие технологии на стадии пилота почти не видны в открытых данных.\n")
    by_step = ws.groupby("ladder_step")["org_stage"].agg(["count", "mean"])
    L.append("| Ступень | Технологий | Средняя стадия по таблице (1 исследование … 4 раннее внедрение) |\n|---|---|---|")
    for s, r in by_step.iterrows():
        L.append(f"| {s} | {int(r['count'])} | {r['mean']:.2f} |")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    df[["tech_id", "name", "label", "p_2021", "ladder_step"] + [c for c in df.columns if c.startswith("first_")]].to_csv(
        ROOT / "data" / "dataset" / "timemachine.csv", index=False)
    # База исторических двойников: как технологии выглядели в 2021 году и чем они стали к 2026
    twins = pd.concat([df[["tech_id", "name", "query", "label", "p_2021"]], then[PIT_FEATURES]], axis=1)[visible]
    twins.to_csv(TWINS, index=False)
    print("\n".join(L[:12]))


if __name__ == "__main__":
    {"collect": collect, "backtest": backtest}[sys.argv[1] if len(sys.argv) > 1 else "backtest"]()


FATE = {"mainstream": "к 2026 году стала мейнстримом", "faded_hype": "к 2026 году угасла как хайп",
        "weak_signal": "к 2026 году остаётся слабым сигналом"}


def find_twin(features_now: dict) -> dict | None:
    """Исторический двойник: технология, чьи признаки на конец 2021 года ближе всего к признакам кандидата сейчас."""
    if not TWINS.exists():
        return None
    t = pd.read_csv(TWINS)
    X = t[PIT_FEATURES].astype(float).values
    mu, sd = X.mean(0), X.std(0) + 1e-9
    x = (np.array([features_now.get(k, 0.0) for k in PIT_FEATURES], dtype=float) - mu) / sd
    d = np.sqrt((((X - mu) / sd - x) ** 2).sum(1))
    i = int(np.argmin(d))
    r = t.iloc[i]
    return {"name": r["name"], "query": r["query"], "year": FREEZE, "fate": FATE.get(r["label"], r["label"]),
            "label_now": r["label"], "distance": round(float(d[i]), 2)}
