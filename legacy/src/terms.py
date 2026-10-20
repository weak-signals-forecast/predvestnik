"""Детектор всплеска терминов (лексический слой). Дополняет эмбеддинговые детекторы:
ловит новый метод или объект внутри старой темы, который семантически близок к ней
и потому невидим для кластеризации (спутниковый мониторинг внутри "выбросов метана").

    python src/terms.py

Для каждой n-граммы (1-3 слова) считается число документов по годам, нормированное на объём года.
Кандидат в слабые сигналы: почти отсутствовал до 2020, заметен в последние три года, растёт.
Выход: data/clusters/terms.csv и секция в report.md.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster import OUT, STOP, YEARS, PARTIAL_YEAR_FACTOR  # noqa: E402

PRIOR = [2017, 2018, 2019]
RECENT = [2023, 2024, 2025]


def main():
    df = pd.read_parquet(OUT / "doc_topics.parquet")
    text = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).str.lower()
    cv = CountVectorizer(ngram_range=(1, 3), stop_words=list(STOP), min_df=8, max_df=0.2, binary=True,
                         token_pattern=r"(?u)\b[a-z][a-z0-9\-]{2,}\b")
    M = cv.fit_transform(text).tocsc()
    terms = cv.get_feature_names_out()
    years = df["year"].to_numpy()
    year_idx = {y: np.where(years == y)[0] for y in YEARS}
    docs_per_year = np.array([len(year_idx[y]) for y in YEARS], dtype=float)
    counts = np.stack([np.asarray(M[year_idx[y]].sum(axis=0)).ravel() for y in YEARS], axis=1).astype(float)
    counts[:, -1] *= PARTIAL_YEAR_FACTOR
    docs_per_year[-1] *= PARTIAL_YEAR_FACTOR
    rate = counts / np.maximum(docs_per_year, 1) * 1000  # документов на 1000 в год

    yi = {y: i for i, y in enumerate(YEARS)}
    prior_n = counts[:, [yi[y] for y in PRIOR]].sum(1)
    recent_n = counts[:, [yi[y] for y in RECENT]].sum(1)
    prior_r = rate[:, [yi[y] for y in PRIOR]].mean(1)
    recent_r = rate[:, [yi[y] for y in RECENT]].mean(1)
    growth = np.log((recent_r + 0.5) / (prior_r + 0.5))
    cum = np.cumsum(counts, axis=1)
    first_year = np.array([YEARS[int(np.argmax(c >= 5))] if c[-1] >= 5 else YEARS[-1] for c in cum])
    lc = np.log1p(counts)
    accel = lc[:, yi[2025]] - 2 * lc[:, yi[2024]] + lc[:, yi[2023]]

    # Концентрация термина по темам. Академические штампы ("valuable insights", "underscores")
    # тоже всплеснули после 2023, но размазаны по всем темам: нормированная энтропия > 0.65.
    # Содержательные термины концентрируются в одной-двух темах.
    topic_of_term, entropy = [], []
    labels = df["topic"].to_numpy()
    n_topics = len(set(labels[labels >= 0]))
    for j in range(M.shape[1]):
        rows = M[:, j].nonzero()[0]
        lab = labels[rows]
        lab = lab[lab >= 0]
        if len(lab) == 0:
            topic_of_term.append(-1)
            entropy.append(1.0)
            continue
        cnt = np.bincount(lab)
        pr = cnt[cnt > 0] / cnt.sum()
        topic_of_term.append(int(cnt.argmax()))
        entropy.append(float(-(pr * np.log(pr)).sum() / math.log(max(2, n_topics))))

    t = pd.DataFrame({
        "term": terms, "first_year": first_year, "prior_docs": prior_n.astype(int), "recent_docs": recent_n.astype(int),
        "growth": growth, "accel": accel, "topic": topic_of_term, "topic_entropy": entropy,
        "counts": [" ".join(str(int(round(v))) for v in c) for c in counts],
    })
    # Слабый сигнал: почти не было до 2020, есть сейчас, растёт. Вес на объём логарифмический.
    cand = t[(t.prior_docs <= 5) & (t.recent_docs >= 12) & (t.topic_entropy <= 0.6)].copy()
    cand["score"] = cand["growth"] * np.log(cand["recent_docs"]) * (1 - cand["topic_entropy"])
    # Убираем n-граммы, вложенные в более длинную с почти той же поддержкой (дубли вида "swing" / "moisture swing").
    cand = cand.sort_values("score", ascending=False)
    keep, seen = [], []
    for _, r in cand.iterrows():
        if any((r.term in s or s in r.term) and abs(r.recent_docs - n) <= 0.25 * max(r.recent_docs, n) for s, n in seen):
            continue
        keep.append(r.name)
        seen.append((r.term, r.recent_docs))
    cand = cand.loc[keep]
    t.to_csv(OUT / "terms_all.csv", index=False)
    cand.to_csv(OUT / "terms.csv", index=False)

    topics = pd.read_csv(OUT / "topics.csv").set_index("topic")["name"].to_dict()
    lines = ["\n## Всплески терминов (лексический детектор)\n",
             "Термины, которых почти не было в 2017-2019 (≤5 документов), заметны в 2023-2025 (≥12) и сконцентрированы "
             "в немногих темах (штампы вроде 'valuable insights' размазаны по всем темам и отсеяны). "
             "Указана большая тема, внутри которой живёт термин.\n"]
    for _, r in cand.head(40).iterrows():
        lines.append(f"- **{r.term}** с {r.first_year}: {r.prior_docs} → {r.recent_docs} документов, рост {r.growth:+.2f}, "
                     f"концентрация {1 - r.topic_entropy:.2f}, "
                     f"внутри «{topics.get(r.topic, 'шум')[:45]}»  \n  по годам: {r.counts}")
    rep = OUT / "report.md"
    body = rep.read_text(encoding="utf-8")
    marker = "\n## Всплески терминов"
    body = body.split(marker)[0]
    rep.write_text(body + "\n".join(lines), encoding="utf-8")
    print(f"терминов {len(t)}, кандидатов {len(cand)}, топ-40 добавлен в report.md")
    return cand


if __name__ == "__main__":
    main()
