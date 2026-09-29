"""Сборка размеченного датасета этапа 1.

    python -m wsignals.dataset            # собрать признаки для всех технологий (кэшируется)
    python -m wsignals.dataset --limit 6  # быстрая проверка

Положительный класс: 100 технологий из таблицы организаторов «100 слабых технологических сигналов».
Отрицательные классы: зрелые технологии и угасший хайп по тем же шести областям (data/labels/negatives.yaml).

Признаки считаются ОДИНАКОВО для всех классов и только по открытым источникам на дату снимка.
Колонки таблицы организаторов (стадия, тренд, балл) в признаки НЕ идут: у отрицательных примеров их нет,
а на открытом запросе их тоже не будет. Они используются только для проверки согласованности модели.
Выход: data/dataset/technologies.csv
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from .features import collect

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "вводные данные" / "100_слабых_технологических_сигналов_сентябрь_2026.xlsx"
POS_QUERIES = ROOT / "data" / "labels" / "positives_queries.csv"
NEGATIVES = ROOT / "data" / "labels" / "negatives.yaml"
OUT = ROOT / "data" / "dataset" / "technologies.csv"
STAGE_SCALE = [("концепц", 1), ("исследован", 1), ("прототип", 2), ("poc", 2), ("пилот", 3), ("ранн", 4)]


def stage_to_ordinal(text: str) -> float:
    """47 свободных формулировок стадии -> порядковая шкала 1–4.
    «A → B» это переход: берём середину. Косая черта внутри части это синонимы («Прототип/PoC») или альтернативы
    для разных игроков («Пилот / раннее внедрение у гиперскейлеров»): берём среднее по различным уровням."""
    levels = []
    for part in str(text).lower().split("→"):
        found = {v for k, v in STAGE_SCALE if k in part}
        if found:
            levels.append(sum(found) / len(found))
    return sum(levels) / len(levels) if levels else float("nan")


def labels() -> pd.DataFrame:
    src = pd.read_excel(XLSX, header=1).iloc[:, 1:]
    q = pd.read_csv(POS_QUERIES)
    pos = src.merge(q, left_on="№", right_on="id")
    pos = pd.DataFrame({
        "tech_id": "S" + pos["№"].astype(str).str.zfill(3), "name": pos["Технология (слабый сигнал)"], "query": pos["query"],
        "area": pos["Область"], "label": "weak_signal", "group": pos["group"],
        "org_stage": pos["Стадия развития"].map(stage_to_ordinal), "org_score": pos["Балл (стадия+тренд)"],
        "org_sources": pos["Источники"],
    })
    rows = []
    for area, classes in yaml.safe_load(NEGATIVES.read_text(encoding="utf-8")).items():
        for label, items in classes.items():
            for i, (query, name) in enumerate(items, 1):
                rows.append({"tech_id": f"{'M' if label == 'mainstream' else 'H'}-{area[:3]}-{i:02d}", "name": name,
                             "query": query, "area": area, "label": label})
    df = pd.concat([pos, pd.DataFrame(rows)], ignore_index=True)
    # Почти одинаковые технологии таблицы (например, №44 и №84) держим в одной группе, чтобы они не попали
    # в разные фолды кросс-валидации и не завышали метрику.
    df["group"] = df["group"].fillna(df["tech_id"])
    return df


def refresh_openalex_derived(path: Path = OUT) -> None:
    """Пересчитать пик и связанные признаки по сохранённому годовому ряду (без запросов к API).
    Нужно после исправления формулы: пик считается только по завершённым годам."""
    from .features import derive
    df = pd.read_csv(path)
    years = list(range(2010, 2027))
    yi = {y: i for i, y in enumerate(years)}
    for i, row in df.iterrows():
        if not isinstance(row.get("oa_years"), str):
            continue
        counts = [float(x) for x in row["oa_years"].split()]
        done = counts[: yi[2025] + 1]
        peak = max(done)
        df.at[i, "oa_peak_ratio"] = (counts[yi[2025]] + 1) / (peak + 1)
        df.at[i, "oa_peak_year"] = years[done.index(peak)] if peak else None
    for i, row in df.iterrows():
        d = derive({k: (None if pd.isna(v) else v) for k, v in row.items()})
        for k, v in d.items():
            df.at[i, k] = v
    df.to_csv(path, index=False)
    print(f"пересчитаны признаки OpenAlex для {len(df)} технологий")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--refresh-derived", action="store_true", help="пересчитать производные признаки без запросов")
    args = ap.parse_args()
    if args.refresh_derived:
        return refresh_openalex_derived()
    df = labels()
    if args.limit:
        df = df.groupby("label", group_keys=False).head(args.limit // 3 or 1)
    print(f"технологий {len(df)}: {df['label'].value_counts().to_dict()}", flush=True)
    feats, done = {}, 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(collect, q): tid for tid, q in zip(df["tech_id"], df["query"])}
        for f in as_completed(futs):
            tid = futs[f]
            feats[tid] = f.result()
            done += 1
            if done % 20 == 0 or done == len(df):
                print(f"  собрано {done}/{len(df)}", flush=True)
    F = pd.DataFrame.from_dict(feats, orient="index")
    out = df.merge(F, left_on="tech_id", right_index=True)
    out["snapshot"] = date.today().isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    errors = out.filter(like="_error").notna().sum()
    print(f"сохранено {OUT.relative_to(ROOT)}: {len(out)} строк; ошибки источников: {errors[errors > 0].to_dict()}")


if __name__ == "__main__":
    sys.exit(main())
