"""Кластеризация корпуса, признаки тем и детекторы слабых сигналов.

    python src/cluster.py [--min-cluster 30]

Что считается:
  1. Глобальные темы: UMAP(5) + HDBSCAN по всем документам. Имена тем через c-TF-IDF.
  2. Динамика темы по годам и классические признаки: рост, ускорение, год рождения, объём.
  3. Три нестандартных детектора:
     - precedent_gap: доля ближайших соседей документа, опубликованных НЕ раньше него.
       Если у свежих статей темы нет ранних прецедентов, тема новая по содержанию, а не по названию.
     - drift: косинусное расстояние между центроидом темы до 2020 и после 2023.
       Тема, которая "поехала", меняет предмет: например, из лаборатории в промышленность.
     - lineage: кластеризация по двухлетним окнам с привязкой к предыдущему окну.
       Кластер без родителя = рождение темы. Это и есть кандидат в слабые сигналы.
  4. cross_domain: доля соседей из другого направления. Мост между областями.

Выход в data/clusters/: topics.csv, doc_topics.parquet, windows.csv, umap.png, report.md
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "embeddings" / "bge-base.npz"
OUT = Path(os.environ.get("CLUSTERS_DIR", ROOT / "data" / "clusters"))
YEAR_MIN, YEAR_MAX = 2015, date.today().year
YEARS = list(range(YEAR_MIN, YEAR_MAX + 1))
PARTIAL_YEAR_FACTOR = 12 / max(1, date.today().month - 1)  # текущий год неполный, досчитываем до годового темпа
WINDOWS = [(y, y + 1) for y in range(YEAR_MIN, YEAR_MAX + 1, 2)]
SEED = 42

STOP = set(ENGLISH_STOP_WORDS) | {
    "study", "studies", "results", "result", "based", "using", "used", "use", "method", "methods",
    "paper", "proposed", "propose", "analysis", "model", "models", "data", "approach", "new", "review",
    "high", "low", "different", "effect", "effects", "performance", "process", "processes", "system",
    "systems", "research", "present", "presented", "provide", "provides", "significant", "important",
    "development", "developed", "investigated", "investigate", "shows", "shown", "show", "et", "al",
}


# ---------- загрузка ----------

def load() -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    for pattern in ("data/openalex/*.jsonl", "data/patents/*.jsonl"):
        for p in sorted(glob.glob(str(ROOT / pattern))):
            if ".v1." in p:
                continue
            with open(p, encoding="utf-8") as f:
                rows.extend(json.loads(l) for l in f)
    df = pd.DataFrame(rows)
    doms = df.groupby("id")["domain"].agg(lambda x: sorted(set(x)))
    df = df.drop_duplicates("id").set_index("id")
    df["domains"] = doms
    df = df.reset_index()
    n0 = len(df)
    # Самиздат: препринт без аффилиаций, с одним автором и без цитирований. Многоавторские
    # препринты без организаций это обычные работы на SSRN и arXiv, их оставляем.
    junk = ((df["doc_type"] == "preprint") & (df["orgs"].map(len) == 0)
            & (df["authors"].map(len) <= 1) & (df["cited_by"].fillna(0) == 0))
    df = df[~junk]
    # Только английский: OpenAlex размечает язык, пропуски считаем английскими. Не-английские статьи
    # (индонезийские, китайские) дают ложные "всплески терминов" вроде "dilakukan dengan".
    n_lang = int((df["language"].fillna("en") != "en").sum())
    df = df[df["language"].fillna("en") == "en"]
    norm_title = df["title"].fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    # Пустой заголовок не считается дублем, такие документы просто выбрасываем: без заголовка эмбеддинг ненадёжен.
    df = df[(norm_title != "") & ~norm_title.duplicated(keep="first")].reset_index(drop=True)
    print(f"фильтр: самиздат {int(junk.sum())}, не-английские {n_lang}, дубли заголовков {n0 - int(junk.sum()) - n_lang - len(df)}, осталось {len(df)}")
    z = np.load(CACHE, allow_pickle=False)
    vec = dict(zip(z["ids"].tolist(), z["vectors"]))
    df = df[df["id"].isin(vec) & df["year"].between(YEAR_MIN, YEAR_MAX)].reset_index(drop=True)
    X = np.stack([vec[i] for i in df["id"]]).astype(np.float32)
    df["text"] = df["title"].fillna("") + ". " + df["abstract"].fillna("")
    return df, X


# ---------- кластеризация ----------

def reduce(X: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    u5 = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=seed).fit_transform(X)
    u2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(X)
    return u5, u2


def cluster(u5: np.ndarray, min_cluster: int, min_samples: int | None = None) -> np.ndarray:
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster, min_samples=min_samples or max(5, min_cluster // 3),
                           cluster_selection_method="eom").fit_predict(u5)


def ctfidf_terms(texts: pd.Series, labels: np.ndarray, top: int = 6) -> dict[int, str]:
    """Имена кластеров: термины, характерные для кластера относительно остальных (c-TF-IDF)."""
    ids = sorted(l for l in set(labels) if l >= 0)
    if not ids:
        return {}
    docs_per_class = [" ".join(texts[labels == l]) for l in ids]
    cv = CountVectorizer(ngram_range=(1, 2), stop_words=list(STOP), min_df=2, max_features=60000)
    tf = cv.fit_transform(docs_per_class).astype(np.float64)
    words = np.array(cv.get_feature_names_out())
    row_sums = np.asarray(tf.sum(axis=1)).ravel()
    tf_norm = tf.multiply(1 / np.maximum(row_sums, 1)[:, None]).tocsr()
    avg_len = row_sums.mean()
    freq = np.asarray(tf.sum(axis=0)).ravel()
    idf = np.log(1 + avg_len / np.maximum(freq, 1))
    scores = tf_norm.multiply(idf).tocsr()
    out = {}
    for i, l in enumerate(ids):
        row = scores.getrow(i).toarray().ravel()
        best = row.argsort()[::-1][:top]
        out[l] = ", ".join(words[best])
    return out


# ---------- детекторы на уровне документов ----------

def knn_signals(X: np.ndarray, years: np.ndarray, domains: np.ndarray, k: int = 10, chunk: int = 2000):
    """Для каждого документа: доля соседей не раньше него (precedent_gap) и доля соседей из других направлений."""
    n = len(X)
    gap = np.zeros(n, dtype=np.float32)
    cross = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        sims = X[s:s + chunk] @ X.T
        for i in range(sims.shape[0]):
            sims[i, s + i] = -1  # исключаем сам документ
        nn = np.argpartition(-sims, k, axis=1)[:, :k]
        for i in range(sims.shape[0]):
            g = s + i
            nb = nn[i]
            gap[g] = np.mean(years[nb] >= years[g])
            cross[g] = np.mean(domains[nb] != domains[g])
    return gap, cross


# ---------- признаки тем ----------

def entropy(counter: Counter) -> float:
    tot = sum(counter.values())
    return -sum(c / tot * math.log(c / tot) for c in counter.values()) if tot else 0.0


def year_counts(years: pd.Series) -> np.ndarray:
    c = years.value_counts().reindex(YEARS, fill_value=0).to_numpy(dtype=float)
    c[-1] *= PARTIAL_YEAR_FACTOR
    return c


def topic_features(df: pd.DataFrame, X: np.ndarray, labels: np.ndarray, names: dict[int, str]) -> pd.DataFrame:
    rows = []
    yi = {y: i for i, y in enumerate(YEARS)}
    for l in sorted(set(labels)):
        if l < 0:
            continue
        m = labels == l
        sub = df[m]
        c = year_counts(sub["year"])
        lc = np.log1p(c)
        recent = c[yi[2023]:yi[2025] + 1].mean()
        prior = c[yi[2019]:yi[2021] + 1].mean()
        growth = math.log((recent + 1) / (prior + 1))
        accel = lc[yi[2025]] - 2 * lc[yi[2024]] + lc[yi[2023]]
        birth = next((y for y, n in zip(YEARS, c) if n >= 3), YEARS[0])
        recent_mask = sub["year"] >= 2023
        orgs = Counter(o for os in sub["orgs"] for o in os)
        countries = Counter(cc for cs in sub["countries"] for cc in cs)
        c_recent = Counter(cc for cs in sub[recent_mask]["countries"] for cc in cs)
        c_prior = Counter(cc for cs in sub[~recent_mask]["countries"] for cc in cs)
        cn_recent = c_recent["CN"] / max(1, sum(c_recent.values()))
        cn_prior = c_prior["CN"] / max(1, sum(c_prior.values()))
        early, late = X[m & (df["year"] <= 2020).to_numpy()], X[m & (df["year"] >= 2023).to_numpy()]
        drift = float(1 - early.mean(0) @ late.mean(0) / (np.linalg.norm(early.mean(0)) * np.linalg.norm(late.mean(0)))) \
            if len(early) >= 10 and len(late) >= 10 else np.nan
        dom = Counter(sub["domain"])
        examples = sub.sort_values(["year", "cited_by"], ascending=[False, False])["title"].head(3).tolist()
        rows.append({
            "topic": l, "name": names.get(l, ""), "n": int(m.sum()),
            "domain": dom.most_common(1)[0][0], "domain_share": dom.most_common(1)[0][1] / m.sum(),
            "birth_year": birth, "growth": growth, "accel": accel,
            "recent_share": float(recent_mask.mean()),
            "precedent_gap": float(sub.loc[recent_mask, "precedent_gap"].mean()) if recent_mask.any() else np.nan,
            "cross_domain": float(sub["cross_domain"].mean()),
            "drift": drift,
            "org_diversity": min(1.0, len(orgs) / max(1, m.sum())),
            "country_entropy": entropy(countries),
            "china_share_recent": cn_recent, "china_delta": cn_recent - cn_prior,
            "top_orgs": "; ".join(o for o, _ in orgs.most_common(3)),
            "top_countries": "; ".join(f"{k}:{v}" for k, v in countries.most_common(4)),
            "counts": " ".join(f"{int(round(v))}" for v in c),
            "examples": " | ".join(examples),
        })
    t = pd.DataFrame(rows)
    t["log_n"] = np.log(t["n"])

    def z(col):
        s = t[col].fillna(t[col].median())
        return (s - s.mean()) / (s.std() + 1e-9)

    # Слабый сигнал: растёт, ускоряется, без прецедентов, подхвачен разными акторами, но пока небольшой.
    t["weak_signal_score"] = (1.0 * z("growth") + 0.5 * z("accel") + 1.0 * z("precedent_gap")
                              + 0.5 * z("org_diversity") + 0.5 * z("country_entropy") + 0.3 * z("cross_domain")
                              - 0.75 * z("log_n"))
    # Сильный тренд: растёт и уже большой.
    t["strong_trend_score"] = z("growth") + z("log_n")
    return t.sort_values("weak_signal_score", ascending=False).reset_index(drop=True)


# ---------- окна и родословная ----------

def precedent_similarity(X: np.ndarray, members: np.ndarray, earlier: np.ndarray, chunk: int = 4000) -> float:
    """Среднее по документам кластера максимальное сходство с любым документом из более ранних окон.
    Низкое значение: у темы нет содержательных предшественников, а не просто нет похожего кластера."""
    if len(earlier) == 0:
        return np.nan
    best = np.full(len(members), -1.0, dtype=np.float32)
    for s in range(0, len(earlier), chunk):
        sims = X[members] @ X[earlier[s:s + chunk]].T
        best = np.maximum(best, sims.max(axis=1))
    return float(best.mean())


def windows_lineage(df: pd.DataFrame, X: np.ndarray, u5: np.ndarray, labels: np.ndarray,
                    min_cluster: int) -> pd.DataFrame:
    """Кластеризация по окнам. Рождение определяется без порога, относительно:
    кластер рождён, если он похож на лучшего кандидата в родители из прошлого окна не сильнее,
    чем на ближайшую соседнюю тему в своём же окне (соседи по построению разные темы).
    Дополнительно считается precedent_sim на уровне документов для ранжирования рождений."""
    rows, prev = [], []  # prev: [(wid, centroid)]
    years = df["year"].to_numpy()
    for (a, b) in WINDOWS:
        idx = np.where((years >= a) & (years <= b))[0]
        earlier = np.where(years < a)[0]
        if len(idx) < 3 * min_cluster:
            continue
        wl = cluster(u5[idx], min_cluster=min_cluster)
        names = ctfidf_terms(df["text"].iloc[idx].reset_index(drop=True), wl)
        cids = [l for l in sorted(set(wl)) if l >= 0]
        cens = {}
        for l in cids:
            cen = X[idx[wl == l]].mean(0)
            cens[l] = cen / np.linalg.norm(cen)
        cur = []
        for l in cids:
            members = idx[wl == l]
            cen = cens[l]
            parent, parent_sim = None, 0.0
            for pid, pcen in prev:
                sim = float(cen @ pcen)
                if sim > parent_sim:
                    parent, parent_sim = pid, sim
            sibling_sim = max((float(cen @ cens[o]) for o in cids if o != l), default=0.0)
            born = bool(prev) and parent_sim <= sibling_sim
            wid = f"{a}-{b}#{l}"
            rows.append({"window": f"{a}-{b}", "wid": wid, "n": len(members), "name": names.get(l, ""),
                         "parent": None if born else parent, "parent_sim": round(parent_sim, 3),
                         "sibling_sim": round(sibling_sim, 3), "born": born,
                         "precedent_sim": round(precedent_similarity(X, members, earlier), 3),
                         "global_topic": int(Counter(labels[members]).most_common(1)[0][0]),
                         "example": df["title"].iloc[members[0]],
                         "members": " ".join(df["id"].iloc[members])})
            cur.append((wid, cen))
        prev = cur
    return pd.DataFrame(rows)


def orphan_births(df: pd.DataFrame, X: np.ndarray, u5: np.ndarray, labels: np.ndarray,
                  orphan_quantile: float = 0.2, min_cluster: int = 8, chunk: int = 4000) -> pd.DataFrame:
    """Детектор рождений на уровне документов. В каждом окне документ-сирота это статья, у которой
    максимальное сходство с любым документом прошлых окон попадает в нижние orphan_quantile окна.
    Кластер сирот = подтема без предшественников, даже если она сидит внутри большой темы,
    которую оконная кластеризация не разрезает (спутниковый мониторинг внутри 'methane emissions')."""
    rows = []
    years = df["year"].to_numpy()
    for (a, b) in WINDOWS:
        idx = np.where((years >= a) & (years <= b))[0]
        earlier = np.where(years < a)[0]
        if len(earlier) < 500 or len(idx) < 100:
            continue
        best = np.full(len(idx), -1.0, dtype=np.float32)
        for s in range(0, len(earlier), chunk):
            best = np.maximum(best, (X[idx] @ X[earlier[s:s + chunk]].T).max(axis=1))
        cut = np.quantile(best, orphan_quantile)
        orphans = idx[best <= cut]
        ol = hdbscan.HDBSCAN(min_cluster_size=min_cluster, min_samples=3, cluster_selection_method="leaf").fit_predict(u5[orphans])
        names = ctfidf_terms(df["text"].iloc[orphans].reset_index(drop=True), ol)
        for l in sorted(set(ol)):
            if l < 0:
                continue
            members = orphans[ol == l]
            gt = Counter(labels[members]).most_common(1)[0][0]
            orgs = Counter(o for os in df["orgs"].iloc[members] for o in os)
            rows.append({"window": f"{a}-{b}", "n": len(members), "name": names.get(l, ""),
                         "precedent_sim": round(float(best[np.isin(idx, members)].mean()), 3),
                         "window_cut": round(float(cut), 3), "global_topic": int(gt),
                         "global_topic_size": int((labels == gt).sum()) if gt >= 0 else 0,
                         "top_orgs": "; ".join(o for o, _ in orgs.most_common(3)),
                         "example": df["title"].iloc[members[0]],
                         "members": " ".join(df["id"].iloc[members])})
    return pd.DataFrame(rows)


# ---------- отчёт ----------

def plot(u2: np.ndarray, df: pd.DataFrame, labels: np.ndarray, topics: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 11))
    doms = sorted(df["domain"].unique())
    for d, col in zip(doms, ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]):
        m = (df["domain"] == d).to_numpy()
        ax.scatter(u2[m, 0], u2[m, 1], s=3, alpha=0.35, c=col, label=d, linewidths=0)
    for _, t in topics.head(12).iterrows():
        m = labels == t["topic"]
        cx, cy = u2[m].mean(0)
        ax.annotate(t["name"].split(", ")[0] + f"\n(n={t['n']})", (cx, cy), fontsize=8, ha="center",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(markerscale=6)
    ax.set_title("Карта корпуса: подписаны 12 тем с наибольшим weak_signal_score")
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def report(topics: pd.DataFrame, win: pd.DataFrame, orph: pd.DataFrame, n_docs: int, n_noise: int, path: Path):
    L = [f"# Первый прогон детектора слабых сигналов\n",
         f"Документов {n_docs}, тем {len(topics)}, шум HDBSCAN {n_noise} ({n_noise / n_docs:.0%}). "
         f"Счётчики по годам {YEARS[0]}-{YEARS[-1]}, последний год досчитан до годового темпа.\n"]
    for dom in sorted(topics["domain"].unique()):
        L.append(f"\n## {dom}: топ-6 слабых сигналов\n")
        for _, t in topics[topics["domain"] == dom].head(6).iterrows():
            L.append(f"- **{t['name']}** (тема {t['topic']}, n={t['n']}, рождение {t['birth_year']}, "
                     f"score {t['weak_signal_score']:.2f})  \n"
                     f"  рост {t['growth']:+.2f}, ускорение {t['accel']:+.2f}, без прецедентов {t['precedent_gap']:.2f}, "
                     f"дрейф {t['drift']:.2f}, кросс-домен {t['cross_domain']:.2f}, Китай {t['china_share_recent']:.0%} "
                     f"({t['china_delta']:+.0%})  \n"
                     f"  по годам: {t['counts']}  \n"
                     f"  кто: {t['top_orgs']}  \n"
                     f"  пример: {t['examples'].split(' | ')[0]}")
    L.append("\n## Сильные тренды (для контраста)\n")
    for _, t in topics.sort_values("strong_trend_score", ascending=False).head(8).iterrows():
        L.append(f"- {t['name']} (n={t['n']}, рост {t['growth']:+.2f})")
    L.append("\n## Темы, рождённые в последних окнах (без родителя в предыдущем окне)\n")
    born = win[win["born"] & (win["n"] >= 15) & win["window"].isin([f"{a}-{b}" for a, b in WINDOWS[-3:]])]
    L.append("Критерий: сходство с лучшим родителем из прошлого окна не выше сходства с ближайшей соседней темой "
             "своего окна. Отсортировано по precedent_sim: чем ниже, тем меньше у темы содержательных предшественников.\n")
    for _, w in born.sort_values("precedent_sim").iterrows():
        L.append(f"- {w['window']}: **{w['name']}** (n={w['n']}, родитель {w['parent_sim']:.2f} vs сосед {w['sibling_sim']:.2f}, "
                 f"прецедент {w['precedent_sim']:.2f}, глобальная тема {w['global_topic']})  \n  пример: {w['example']}")
    L.append("\n## Подтемы-сироты: кластеры документов без предшественников (детектор уровня документов)\n")
    L.append("В каждом окне взяты 20% документов с наименьшим сходством с прошлым, среди них найдены плотные группы. "
             "Указана большая тема, внутри которой сидит подтема.\n")
    for w_name, g in orph[orph["window"].isin([f"{a}-{b}" for a, b in WINDOWS[-3:]])].groupby("window"):
        L.append(f"\n### {w_name}\n")
        for _, o in g.sort_values("n", ascending=False).head(10).iterrows():
            L.append(f"- **{o['name']}** (n={o['n']}, прецедент {o['precedent_sim']:.2f}, "
                     f"внутри темы {o['global_topic']} размером {o['global_topic_size']})  \n"
                     f"  кто: {o['top_orgs']}  \n  пример: {o['example']}")
    L.append("\n## Темы с наибольшим дрейфом (сменили предмет)\n")
    for _, t in topics.dropna(subset=["drift"]).sort_values("drift", ascending=False).head(6).iterrows():
        L.append(f"- {t['name']} (n={t['n']}, дрейф {t['drift']:.3f})")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cluster", type=int, default=30)
    ap.add_argument("--seed", type=int, default=SEED, help="зерно UMAP; разные зёрна дают разные границы кластеров")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df, X = load()
    print(f"документов с эмбеддингами: {len(df)}")
    years, doms = df["year"].to_numpy(), df["domain"].to_numpy()
    df["precedent_gap"], df["cross_domain"] = knn_signals(X, years, doms)
    print("kNN-признаки готовы")
    u5, u2 = reduce(X, args.seed)
    print("UMAP готов")
    labels = cluster(u5, args.min_cluster)
    n_noise = int((labels < 0).sum())
    print(f"HDBSCAN: тем {labels.max() + 1}, шум {n_noise}")
    names = ctfidf_terms(df["text"], labels)
    topics = topic_features(df, X, labels, names)
    win = windows_lineage(df, X, u5, labels, min_cluster=max(15, args.min_cluster // 2))
    for w, g in win.groupby("window"):
        print(f"  окно {w}: кластеров {len(g)}, рождений {int(g['born'].sum())}")
    orph = orphan_births(df, X, u5, labels)
    print("сироты: " + ", ".join(f"{w}: {len(g)} кластеров" for w, g in orph.groupby("window")))
    print(f"окон {win['window'].nunique()}, кластеров в окнах {len(win)}, рождений {int(win['born'].sum())}")

    df.assign(topic=labels, x=u2[:, 0], y=u2[:, 1], **{f"u{i}": u5[:, i] for i in range(5)}).drop(columns=["text"]).to_parquet(OUT / "doc_topics.parquet")
    topics.to_csv(OUT / "topics.csv", index=False)
    win.to_csv(OUT / "windows.csv", index=False)
    orph.to_csv(OUT / "orphan_births.csv", index=False)
    plot(u2, df, labels, topics, OUT / "umap.png")
    report(topics, win, orph, len(df), n_noise, OUT / "report.md")
    print(f"результаты в {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
