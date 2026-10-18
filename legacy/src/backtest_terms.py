"""Ретроспективная проверка лексического детектора: замораживаем данные на конец года F,
находим термины-кандидаты только по данным до F, затем смотрим, что с ними стало через 3-5 лет.

    python src/backtest_terms.py

Утечек нет: концентрация термина считается по темам OpenAlex (поле topic из ингеста), а не по нашей
кластеризации, и только по документам до года заморозки.
Успех = частота термина (на 1000 документов года) в окне исхода выросла минимум вдвое и ≥ 15 документов.
База сравнения = все термины, заметные в окне до заморозки, без требования новизны.
Выход: data/signals/backtest.md, backtest.csv
"""
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "signals"
STOP = set(ENGLISH_STOP_WORDS) | {"study", "results", "based", "using", "used", "method", "paper", "proposed", "analysis",
                                  "model", "data", "approach", "new", "review", "high", "low", "effect", "effects",
                                  "performance", "process", "system", "systems", "research", "present", "significant",
                                  "https", "doi", "org", "abstract", "images", "fig", "figure", "table", "et", "al"}
FREEZES = [  # (заморозка, окно "до", окно "сейчас", окно исхода)
    (2020, [2015, 2016, 2017], [2018, 2019, 2020], [2023, 2024, 2025]),
    (2022, [2017, 2018, 2019], [2020, 2021, 2022], [2024, 2025, 2026]),
]


def load():
    rows = []
    for p in glob.glob(str(ROOT / "data/openalex/*.jsonl")):
        with open(p, encoding="utf-8") as f:
            rows.extend(json.loads(l) for l in f)
    df = pd.DataFrame(rows).drop_duplicates("id")
    df = df[df["year"].between(2015, 2026) & df["language"].fillna("en").eq("en")].reset_index(drop=True)
    df["text"] = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).str.lower()
    return df


def run_freeze(df, freeze, prior, recent, outcome, min_recent=8, max_prior=3):
    years = df["year"].to_numpy()
    cv = CountVectorizer(ngram_range=(1, 3), stop_words=list(STOP), min_df=5, max_df=0.2, binary=True,
                         token_pattern=r"(?u)\b[a-z][a-z0-9\-]{2,}\b")
    M = cv.fit_transform(df["text"]).tocsc()
    terms = cv.get_feature_names_out()
    all_years = list(range(2015, 2027))
    docs_per_year = np.array([(years == y).sum() for y in all_years], dtype=float)
    docs_per_year[-1] *= 12 / 8
    counts = np.stack([np.asarray(M[years == y].sum(axis=0)).ravel() for y in all_years], axis=1).astype(float)
    counts[:, -1] *= 12 / 8
    rate = counts / np.maximum(docs_per_year, 1) * 1000
    yi = {y: i for i, y in enumerate(all_years)}
    idx = lambda ys: [yi[y] for y in ys]
    prior_n, recent_n, out_n = counts[:, idx(prior)].sum(1), counts[:, idx(recent)].sum(1), counts[:, idx(outcome)].sum(1)
    recent_r, out_r = rate[:, idx(recent)].mean(1), rate[:, idx(outcome)].mean(1)

    # концентрация по темам OpenAlex, только документы до заморозки (без утечки из будущего)
    past = np.where(years <= freeze)[0]
    topics = df["topic"].fillna("—").to_numpy()
    tcodes, tinv = np.unique(topics[past], return_inverse=True)
    Mp = M[past].tocsc()
    entropy = np.ones(len(terms))
    for j in range(len(terms)):
        rows_ = Mp[:, j].nonzero()[0]
        if len(rows_) < 3:
            continue
        cnt = np.bincount(tinv[rows_], minlength=len(tcodes))
        pr = cnt[cnt > 0] / cnt.sum()
        entropy[j] = -(pr * np.log(pr)).sum() / math.log(max(2, len(tcodes)))

    t = pd.DataFrame({"term": terms, "prior_docs": prior_n, "recent_docs": recent_n, "outcome_docs": out_n,
                      "recent_rate": recent_r, "outcome_rate": out_r, "entropy": entropy})
    t["success"] = (t["outcome_rate"] >= 2 * t["recent_rate"]) & (t["outcome_docs"] >= 15)
    t["alive"] = (t["outcome_rate"] >= t["recent_rate"]) & (t["outcome_docs"] >= 15)
    base = t[t["recent_docs"] >= min_recent]
    flagged = base[(base["prior_docs"] <= max_prior) & (base["entropy"] <= 0.6)].copy()
    flagged["score"] = np.log((flagged["recent_rate"] + 0.5) / (flagged["prior_docs"] / len(prior) / 1 + 0.5)) * np.log(flagged["recent_docs"])
    flagged = flagged.sort_values("score", ascending=False)
    # вложенные n-граммы с той же поддержкой
    keep, seen = [], []
    for _, r in flagged.iterrows():
        if any((r.term in s or s in r.term) and abs(r.recent_docs - n) <= 0.25 * max(r.recent_docs, n) for s, n in seen):
            continue
        keep.append(r.name); seen.append((r.term, r.recent_docs))
    flagged = flagged.loc[keep]
    return base, flagged


def main():
    df = load()
    L = ["# Ретроспективная проверка лексического детектора\n",
         "Данные замораживаются на конец года, кандидаты выбираются только по прошлому, "
         "затем измеряется, что с ними стало. Успех: частота термина в окне исхода выросла не менее чем вдвое "
         "и набрала ≥ 15 документов. База: все заметные термины окна без требования новизны.\n"]
    rows = []
    for freeze, prior, recent, outcome in FREEZES:
        base, flagged = run_freeze(df, freeze, prior, recent, outcome)
        top = flagged.head(30)
        p_base, p_flag, p_top = base["success"].mean(), flagged["success"].mean(), top["success"].mean()
        a_base, a_top = base["alive"].mean(), top["alive"].mean()
        rows.append(dict(freeze=freeze, base_terms=len(base), base_success=round(p_base, 3), flagged=len(flagged),
                         flagged_success=round(p_flag, 3), top30_success=round(p_top, 3), lift_top30=round(p_top / max(p_base, 1e-9), 1),
                         base_alive=round(a_base, 3), top30_alive=round(a_top, 3), lift_alive=round(a_top / max(a_base, 1e-9), 1)))
        L.append(f"\n## Заморозка на конец {freeze}: смотрим {recent[0]}-{recent[-1]}, проверяем в {outcome[0]}-{outcome[-1]}\n")
        L.append(f"| | Терминов | Выросли ещё ≥2x | Удержались или выросли |\n|---|---|---|---|\n"
                 f"| База: все заметные термины | {len(base)} | {p_base:.0%} | {a_base:.0%} |\n"
                 f"| Кандидаты детектора | {len(flagged)} | {p_flag:.0%} | {flagged['alive'].mean():.0%} |\n"
                 f"| Топ-30 детектора | {len(top)} | {p_top:.0%} | {a_top:.0%} |\n")
        L.append(f"Подъём относительно базы: рост ещё вдвое **×{p_top / max(p_base, 1e-9):.1f}**, "
                 f"удержание **×{a_top / max(a_base, 1e-9):.1f}**.\n")
        L.append(f"Что детектор показал бы в конце {freeze} года и что случилось:\n")
        L.append("| Термин | до | тогда | потом | исход |\n|---|---|---|---|---|")
        for _, r in top.iterrows():
            L.append(f"| {r.term} | {int(r.prior_docs)} | {int(r.recent_docs)} | {int(r.outcome_docs)} | "
                     f"{'✅ вырос ×%.1f' % (r.outcome_rate / max(r.recent_rate, 1e-9)) if r.success else ('➖ остался' if r.outcome_rate >= 0.7 * r.recent_rate else '❌ угас')} |")
        flagged.assign(freeze=freeze).to_csv(OUT / f"backtest_{freeze}.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "backtest.csv", index=False)
    (OUT / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
