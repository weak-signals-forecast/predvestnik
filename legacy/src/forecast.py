"""Обучаемый предсказатель взлёта с временной валидацией, прогнозом на сегодня и проверкой в будущем.

    python src/forecast.py                       # валидация + прогноз (термины и темы OpenAlex)
    python src/forecast.py --score data/forecast/forecast_terms.csv   # проверить старый прогноз по текущим данным

Две единицы прогноза:
  термины  - n-граммы 1-3 слова, встречающиеся в заголовках (иначе это служебная лексика);
  темы     - primary topic OpenAlex, курируемая таксономия, грубее, но без лексического шума.
Для каждого года заморозки F признаки считаются только по документам <= F. Метка: через 3-5 лет
частота (на 1000 документов года) выросла минимум вдвое и набрала >= 15 документов.
Головная проверка временная: обучение на 2017-2019, тест на 2020-2021. Дополнительно перекрёстная
проверка по годам (каждый год предсказывается моделью, обученной на остальных).
Модель: ансамбль градиентного бустинга и k ближайших исторических соседей.
Выход: data/forecast/validation.md, forecast.md, forecast_terms.csv, forecast_topics.csv, forecast.json
"""
import argparse
import glob
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "forecast"
YEARS = list(range(2015, 2027))
YI = {y: i for i, y in enumerate(YEARS)}
PARTIAL = 12 / max(1, date.today().month - 1)
ALL_FREEZES, TRAIN_FREEZES, TEST_FREEZES, NOW = [2017, 2018, 2019, 2020, 2021], [2017, 2018, 2019], [2020, 2021], 2026
STOP = set(ENGLISH_STOP_WORDS) | {"study", "studies", "results", "result", "based", "using", "used", "use", "method",
    "methods", "paper", "proposed", "propose", "analysis", "model", "models", "data", "approach", "new", "review",
    "high", "low", "different", "effect", "effects", "performance", "process", "processes", "system", "systems",
    "research", "present", "presented", "provide", "provides", "significant", "important", "development",
    "developed", "investigated", "investigate", "shows", "shown", "show", "et", "al", "https", "doi", "org",
    "abstract", "images", "fig", "figure", "table", "version", "started", "urgent", "segments", "band", "note",
    "prospects", "enabled", "unprecedented", "roadmap", "rigorous", "insufficiently", "formal", "exclusive", "legacy",
    "novel", "comprehensive", "critical", "assessment", "highlights", "insights", "underscores", "determination",
    "predicting", "sequential", "discrimination", "techno", "blast", "pilot-scale", "ternary", "clusters"}
INDUSTRY = ["sinopec", "petrochina", "cnpc", "china national petroleum", "cnooc", "aramco", "shell", "totalenergies",
            "exxon", "chevron", "equinor", "eni ", "gazprom", "rosneft", "lukoil", "novatek", "baker hughes",
            "schlumberger", "halliburton", "linde", "air liquide", "air products", "siemens", "general electric",
            "mitsubishi", "petrobras", "adnoc", "qatarenergy", "petronas", "woodside", "repsol", "omv", "conocophillips",
            "occidental", "engie", "snam", "national grid", "enbridge", "honeywell", "johnson matthey", "topsoe",
            "toyota", "hyundai", "kawasaki", "chiyoda", "technip", "saipem", "climeworks", "ghgsat", "sumitomo"]
FEATURES = ["recent_docs", "recent_rate", "prior_rate", "growth", "accel", "last_share", "first_year_age", "dormancy",
            "active_years", "topic_entropy", "n_topics", "org_per_doc", "n_countries", "china_share", "industry_share",
            "elite_share", "preprint_share", "title_share", "domain_spread", "venue_per_doc"]
TWIN_F = ["recent_docs", "growth", "accel", "last_share", "first_year_age", "dormancy", "topic_entropy",
          "org_per_doc", "n_countries", "elite_share", "industry_share", "title_share", "preprint_share"]
NAMES = {"recent_docs": "документов за 3 года", "recent_rate": "частота сейчас", "prior_rate": "частота раньше",
         "growth": "рост частоты", "accel": "ускорение", "last_share": "доля последнего года",
         "first_year_age": "лет с первого упоминания", "dormancy": "годы дремоты", "active_years": "активных лет",
         "topic_entropy": "размазанность по темам", "n_topics": "число тем", "org_per_doc": "организаций на документ",
         "n_countries": "число стран", "china_share": "доля Китая", "industry_share": "доля индустрии",
         "elite_share": "доля топ-100 организаций", "preprint_share": "доля препринтов",
         "title_share": "доля в заголовках", "domain_spread": "число направлений", "venue_per_doc": "площадок на документ"}


# ---------- корпус и матрицы ----------

def load():
    rows = []
    for p in glob.glob(str(ROOT / "data/openalex/*.jsonl")):
        with open(p, encoding="utf-8") as f:
            rows.extend(json.loads(l) for l in f)
    df = pd.DataFrame(rows)
    doms = df.groupby("id")["domain"].agg(lambda x: sorted(set(x)))
    df = df.drop_duplicates("id").set_index("id")
    df["domains"] = doms
    df = df.reset_index()
    df = df[df["year"].between(YEARS[0], YEARS[-1]) & df["language"].fillna("en").eq("en")]
    df = df[df["title"].fillna("").str.strip() != ""].reset_index(drop=True)
    df["text"] = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).str.lower()
    df["title_l"] = df["title"].fillna("").str.lower()
    return df


def onehot(lists):
    cnt = Counter(x for xs in lists for x in xs)
    keys = {k: i for i, k in enumerate(cnt)}
    rows, cols = [], []
    for r, xs in enumerate(lists):
        for x in set(xs):
            rows.append(r); cols.append(keys[x])
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(lists), len(keys))), list(keys)


def context(df):
    """Матрицы документ × категория, общие для обеих единиц прогноза."""
    T, tnames = onehot([[t or "—"] for t in df["topic"]])
    O, _ = onehot(df["orgs"])
    C, _ = onehot(df["countries"])
    V, _ = onehot([[v or "—"] for v in df["venue"]])
    D, _ = onehot(df["domains"])
    flags = {
        "industry": np.array([any(k in o.lower() for o in os for k in INDUSTRY) for os in df["orgs"]], dtype=float),
        "preprint": (df["doc_type"] == "preprint").to_numpy(dtype=float),
        "china": np.array(["CN" in cs for cs in df["countries"]], dtype=float),
    }
    years = df["year"].to_numpy()
    docs_per_year = np.array([(years == y).sum() for y in YEARS], dtype=float)
    docs_per_year[-1] *= PARTIAL
    return dict(T=T, tnames=tnames, O=O, C=C, V=V, D=D, flags=flags, years=years, docs_per_year=docs_per_year)


def year_counts(M, years):
    counts = np.stack([np.asarray(M[years == y].sum(axis=0)).ravel() for y in YEARS], axis=1).astype(float)
    counts[:, -1] *= PARTIAL
    return counts


def unit_terms(df, ctx):
    cv = CountVectorizer(ngram_range=(1, 3), stop_words=list(STOP), min_df=8, max_df=0.2, binary=True,
                         token_pattern=r"(?u)\b[a-z][a-z0-9\-]{2,}\b")
    M = cv.fit_transform(df["text"]).tocsr()
    Mt = CountVectorizer(vocabulary=cv.vocabulary_, ngram_range=(1, 3), binary=True,
                         token_pattern=r"(?u)\b[a-z][a-z0-9\-]{2,}\b").transform(df["title_l"]).tocsr()
    counts = year_counts(M, ctx["years"])
    return dict(kind="термины", names=np.array(cv.get_feature_names_out()), M=M, Mt=Mt, counts=counts,
                rate=counts / np.maximum(ctx["docs_per_year"], 1) * 1000,
                pop=dict(recent_min=10, prior_max=15, title_docs_min=3, title_share_min=0.25))


def unit_topics(df, ctx):
    M = ctx["T"].tocsr()
    counts = year_counts(M, ctx["years"])
    return dict(kind="темы OpenAlex", names=np.array(ctx["tnames"]), M=M, Mt=M, counts=counts,
                rate=counts / np.maximum(ctx["docs_per_year"], 1) * 1000,
                pop=dict(recent_min=10, prior_max=40, title_docs_min=0, title_share_min=0.0))


# ---------- признаки на момент заморозки ----------

def spread(Mp, X, past):
    """Для каждой единицы: число категорий, сумма, энтропия распределения по категориям."""
    S = (Mp.T @ X[past]).tocsr()
    S.sum_duplicates()
    nnz = np.diff(S.indptr)
    rs = np.asarray(S.sum(axis=1)).ravel()
    row_of = np.repeat(np.arange(S.shape[0]), nnz)
    pr = S.data / np.maximum(rs[row_of], 1)
    ent = -np.bincount(row_of, weights=pr * np.log(np.maximum(pr, 1e-12)), minlength=S.shape[0])
    return nnz, rs, ent


def features_at(U, ctx, F):
    years, counts, rate = ctx["years"], U["counts"], U["rate"]
    past = years <= F
    Mp = U["M"][past]
    n_docs = np.asarray(Mp.sum(axis=0)).ravel()
    ri = [YI[y] for y in (F - 2, F - 1, F) if y in YI]
    pi = [YI[y] for y in (F - 5, F - 4, F - 3) if y in YI]
    recent_docs, recent_rate = counts[:, ri].sum(1), rate[:, ri].mean(1)
    prior_docs = counts[:, pi].sum(1) if pi else np.zeros(len(recent_docs))
    prior_rate = rate[:, pi].mean(1) if pi else np.zeros(len(recent_rate))
    lc = np.log1p(counts)
    upto = counts[:, :YI[F] + 1] > 0
    first_idx = np.where(upto.any(1), upto.argmax(1), YI[F])
    first_year_age = YI[F] - first_idx
    active_years = upto.sum(1)
    n_topics, _, ent_t = spread(Mp, ctx["T"], past)
    n_orgs, _, _ = spread(Mp, ctx["O"], past)
    n_countries, _, _ = spread(Mp, ctx["C"], past)
    n_venues, _, _ = spread(Mp, ctx["V"], past)
    n_domains, _, _ = spread(Mp, ctx["D"], past)
    fl = {k: np.asarray(Mp.T @ v[past]).ravel() / np.maximum(n_docs, 1) for k, v in ctx["flags"].items()}
    # Топ-100 организаций и число тем берутся только по документам до заморозки: иначе будущее протекает в признаки.
    org_past = np.asarray(ctx["O"][past].sum(axis=0)).ravel()
    top_orgs = np.argsort(-org_past)[:100]
    elite_doc = (np.asarray(ctx["O"][:, top_orgs].sum(axis=1)).ravel() > 0).astype(float)
    fl["elite"] = np.asarray(Mp.T @ elite_doc[past]).ravel() / np.maximum(n_docs, 1)
    n_topics_past = int((np.asarray(ctx["T"][past].sum(axis=0)).ravel() > 0).sum())
    title_docs = np.asarray(U["Mt"][past].sum(axis=0)).ravel()
    X = pd.DataFrame({
        "term": U["names"], "freeze": F, "n_docs": n_docs, "recent_docs": recent_docs, "recent_rate": recent_rate,
        "prior_docs": prior_docs, "prior_rate": prior_rate,
        "growth": np.log((recent_rate + 0.5) / (prior_rate + 0.5)),
        "accel": lc[:, YI[F]] - 2 * lc[:, YI[F - 1]] + lc[:, YI[F - 2]],
        "last_share": counts[:, YI[F]] / np.maximum(recent_docs, 1),
        "first_year_age": first_year_age, "dormancy": np.maximum(0, first_year_age + 1 - active_years),
        "active_years": active_years,
        "topic_entropy": np.nan_to_num(ent_t / np.log(np.maximum(2, np.minimum(n_docs, n_topics_past)))),
        "n_topics": n_topics, "org_per_doc": n_orgs / np.maximum(n_docs, 1), "n_countries": n_countries,
        "china_share": fl["china"], "industry_share": fl["industry"], "elite_share": fl["elite"],
        "preprint_share": fl["preprint"], "title_docs": title_docs, "title_share": title_docs / np.maximum(n_docs, 1),
        "domain_spread": n_domains, "venue_per_doc": n_venues / np.maximum(n_docs, 1),
    })
    p = U["pop"]
    X = X[(X["recent_docs"] >= p["recent_min"]) & (X["prior_docs"] <= p["prior_max"])
          & (X["title_docs"] >= p["title_docs_min"]) & (X["title_share"] >= p["title_share_min"])]
    if p["title_docs_min"] > 0:  # только для терминов, не для тем OpenAlex
        X = X[~X["term"].map(is_generic)]
    return X.reset_index(drop=True)


def label_at(U, X, F):
    oy = [YI[y] for y in (F + 3, F + 4, F + 5) if y in YI]
    idx = pd.Index(U["names"]).get_indexer(X["term"])
    X = X.copy()
    X["outcome_rate"] = U["rate"][idx][:, oy].mean(1)
    X["outcome_docs"] = U["counts"][idx][:, oy].sum(1)
    X["ratio"] = X["outcome_rate"] / np.maximum(X["recent_rate"], 1e-9)
    X["takeoff"] = ((X["ratio"] >= 2) & (X["outcome_docs"] >= 15)).astype(int)
    return X


# ---------- модель ----------

def fit(train):
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                        min_samples_leaf=25, random_state=42).fit(train[FEATURES], train["takeoff"])
    sc = StandardScaler().fit(train[TWIN_F])
    knn = NearestNeighbors(n_neighbors=min(25, len(train))).fit(sc.transform(train[TWIN_F]))
    return dict(gb=gb, sc=sc, knn=knn, train=train.reset_index(drop=True))


def predict(model, X):
    p_gb = model["gb"].predict_proba(X[FEATURES])[:, 1]
    d, ix = model["knn"].kneighbors(model["sc"].transform(X[TWIN_F]))
    tr = model["train"]
    p_knn = tr["takeoff"].to_numpy()[ix].mean(1)
    nearest = ix[:, 0]
    return pd.DataFrame({"p_gb": p_gb, "p_knn": p_knn, "p": 0.5 * (p_gb + p_knn),
                         "twin": tr["term"].to_numpy()[nearest], "twin_freeze": tr["freeze"].to_numpy()[nearest],
                         "twin_ratio": tr["ratio"].to_numpy()[nearest], "twin_takeoff": tr["takeoff"].to_numpy()[nearest]},
                        index=X.index)


GENERIC = {"circularity", "strategic", "deployment", "spatially", "explicit", "energy-efficient", "direct", "efficient",
           "interpretable", "recurrent", "engineered", "predicting", "resilience", "throughput", "architectures",
           "symbiosis", "surrogate", "system-level", "scalable", "robust", "integrated", "optimal", "advanced", "smart",
           "intelligent", "emerging", "sustainable", "green", "hybrid", "multi-scale", "multiscale", "data-driven",
           "holistic", "systematic", "quantitative", "qualitative", "comparative", "preliminary", "experimental",
           "numerical", "theoretical", "practical", "potential", "future", "current", "recent", "global", "regional",
           "national", "local", "large-scale", "small-scale", "long-term", "short-term", "real-time", "cost-effective",
           "low-cost", "high-performance", "state-of-the-art", "cutting-edge", "next-generation", "promising"}
ADJ_SUFFIX = ("able", "ible", "ent", "ant", "ive", "ous", "ed", "ing", "ly", "ity", "ness")


def is_generic(term: str) -> bool:
    """Общие слова без предметного содержания: одиночные прилагательные и причастия по морфологии,
    и фразы, целиком состоящие из оценочной лексики. Названия технологий содержат существительное-предмет."""
    words = term.split()
    if len(words) == 1:
        if "-" in term:  # дефисные составные слова почти всегда предметные: amine-grafted, off-grid, ai-driven
            return term in GENERIC
        return term in GENERIC or term.endswith(ADJ_SUFFIX)
    return all(w in GENERIC or w.endswith(("ly",)) for w in words)


def hand_score(X):
    return X["growth"] * np.log(X["recent_docs"]) * (1 - X["topic_entropy"]) * (X["prior_docs"] <= 3)


def dedupe(df, col):
    """Склеивает фразы с общими словами и похожей поддержкой: 'air carbon capture' и 'direct air carbon' один сигнал."""
    df = df.sort_values(col, ascending=False)
    keep, seen = [], []
    for _, r in df.iterrows():
        w = set(r.term.split())
        dup = False
        for s, n in seen:
            ws = set(s.split())
            shared = len(w & ws) / max(1, min(len(w), len(ws)))
            if (r.term in s or s in r.term or shared >= 0.5) and abs(r.recent_docs - n) <= 0.5 * max(r.recent_docs, n):
                dup = True; break
        if not dup:
            keep.append(r.name); seen.append((r.term, r.recent_docs))
    return df.loc[keep]


def precision_at(df, col, k=30):
    return dedupe(df, col).head(k)["takeoff"].mean()


def bootstrap_lift(df, col, k=30, n_boot=500, seed=0):
    """Подъём точности@k над базовой долей с 95% интервалом: бутстрэп по единицам тестовой выборки.
    Склейка вложенных фраз делается один раз, ресэмплируются уже уникальные единицы."""
    d = dedupe(df, col).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    y, s_ = d["takeoff"].to_numpy(), d[col].to_numpy()
    lifts = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(d), len(d))
        yb, sb = y[ix], s_[ix]
        base = yb.mean()
        top = yb[np.argsort(-sb)[:k]].mean()
        lifts.append(top / base if base > 0 else np.nan)
    lifts = np.array(lifts)
    point = precision_at(df, col, k) / max(df["takeoff"].mean(), 1e-9)
    return point, float(np.nanpercentile(lifts, 2.5)), float(np.nanpercentile(lifts, 97.5))


# ---------- проверка старого прогноза ----------

def score(path, df, ctx):
    old = pd.read_csv(path)
    F = int(old["freeze"].iloc[0])
    oy = [y for y in (F + 3, F + 4, F + 5) if y in YI and y <= date.today().year]
    if not oy:
        print(f"Прогноз на {F}: окно исхода {F + 3}-{F + 5} ещё не наступило, проверять нечего."); return
    U = unit_terms(df, ctx) if old["kind"].iloc[0] == "термины" else unit_topics(df, ctx)
    idx = pd.Index(U["names"]).get_indexer(old["term"])
    ok = idx >= 0
    out_rate = np.where(ok, U["rate"][np.maximum(idx, 0)][:, [YI[y] for y in oy]].mean(1), 0)
    old["realized_ratio"] = out_rate / np.maximum(old["recent_rate"], 1e-9)
    old["realized"] = (old["realized_ratio"] >= 2)
    print(f"Прогноз от {F}, доступные годы исхода {oy}: из топ-30 сбылось {old.head(30)['realized'].mean():.0%}, "
          f"из всех {len(old)} кандидатов {old['realized'].mean():.0%}")
    print(old.head(30)[["term", "p", "realized_ratio", "realized"]].to_string(index=False))


# ---------- основной сценарий ----------

def run_unit(U, ctx, L, P, forecast_rows, publish=True):
    kind = U["kind"]
    tabs = {F: label_at(U, features_at(U, ctx, F), F) for F in ALL_FREEZES}
    for F, X in tabs.items():
        X["hand"] = hand_score(X)
    train = pd.concat([tabs[F] for F in TRAIN_FREEZES], ignore_index=True)
    test = pd.concat([tabs[F] for F in TEST_FREEZES], ignore_index=True)
    model = fit(train)
    test = pd.concat([test, predict(model, test)], axis=1)

    L.append(f"\n# {kind.capitalize()}\n" + ("" if U["kind"] == "термины" else
             "Единица не прошла валидацию: положительных примеров слишком мало (база 3%), подъём около единицы. "
             "В прогноз не включена, оставлена как отрицательный результат.\n"))
    L.append("Популяция по заморозкам: " + ", ".join(f"{F}: {len(X)} ({X['takeoff'].mean():.0%} взлетели)" for F, X in tabs.items()) + "\n")
    L.append("## Временная проверка: обучение 2017–2019, тест 2020–2021\n")
    L.append("| Заморозка | Единиц | База | Ручной скор @30 | Бустинг @30 | Соседи @30 | Ансамбль @30 | Ансамбль @10 | AUC |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for F in TEST_FREEZES + ["все"]:
        t = test if F == "все" else test[test["freeze"] == F]
        base = t["takeoff"].mean()
        ph, pg, pk, pe = precision_at(t, "hand"), precision_at(t, "p_gb"), precision_at(t, "p_knn"), precision_at(t, "p")
        pe10 = precision_at(t, "p", 10)
        auc = roc_auc_score(t["takeoff"], t["p"]) if t["takeoff"].nunique() > 1 else float("nan")
        L.append(f"| {F} | {len(t)} | {base:.0%} | {ph:.0%} (×{ph / base:.1f}) | {pg:.0%} (×{pg / base:.1f}) | "
                 f"{pk:.0%} (×{pk / base:.1f}) | **{pe:.0%} (×{pe / base:.1f})** | {pe10:.0%} (×{pe10 / base:.1f}) | {auc:.2f} |")
        rows.append(dict(kind=kind, freeze=F, n=len(t), base=base, hand=ph, gb=pg, knn=pk, ens=pe, ens10=pe10, auc=auc))
    lift_all = rows[-1]["ens"] / max(rows[-1]["base"], 1e-9)
    ci = {k: bootstrap_lift(test, "p", k) for k in (10, 30)}
    L.append(f"\nБутстрэп по единицам тестовой выборки (500 повторов), подъём ансамбля над базой с 95% интервалом: "
             f"топ-30 ×{ci[30][0]:.1f} [{ci[30][1]:.1f}; {ci[30][2]:.1f}], топ-10 ×{ci[10][0]:.1f} [{ci[10][1]:.1f}; {ci[10][2]:.1f}]. "
             "Интервалы широкие, потому что в тестовых заморозках 340–470 единиц и 30–45 взлётов; это честная цена малого корпуса.\n")
    for r in rows:
        r["lift30_lo"], r["lift30_hi"] = round(ci[30][1], 2), round(ci[30][2], 2)
        r["lift10_lo"], r["lift10_hi"] = round(ci[10][1], 2), round(ci[10][2], 2)

    L.append("\n## Перекрёстная проверка по годам: каждый год предсказан моделью, обученной на остальных\n")
    L.append("| Заморозка | База | Ансамбль @30 | Ансамбль @10 | AUC |\n|---|---|---|---|---|")
    cv_rows = []
    for F in ALL_FREEZES:
        tr = pd.concat([tabs[g] for g in ALL_FREEZES if g != F], ignore_index=True)
        te = pd.concat([tabs[F], predict(fit(tr), tabs[F])], axis=1)
        base, pe, pe10 = te["takeoff"].mean(), precision_at(te, "p"), precision_at(te, "p", 10)
        auc = roc_auc_score(te["takeoff"], te["p"]) if te["takeoff"].nunique() > 1 else float("nan")
        L.append(f"| {F} | {base:.0%} | {pe:.0%} (×{pe / base:.1f}) | {pe10:.0%} (×{pe10 / base:.1f}) | {auc:.2f} |")
        cv_rows.append(dict(base=base, ens=pe, ens10=pe10))
    cv = pd.DataFrame(cv_rows)
    L.append(f"| среднее | {cv.base.mean():.0%} | {cv.ens.mean():.0%} (×{cv.ens.mean() / cv.base.mean():.1f}) | "
             f"{cv.ens10.mean():.0%} (×{cv.ens10.mean() / cv.base.mean():.1f}) | |")

    if kind == "термины":
        imp = permutation_importance(model["gb"], test[FEATURES], test["takeoff"], scoring="roc_auc", n_repeats=10, random_state=0)
        L.append("\n## Что предсказывает взлёт (важность признаков бустинга на тесте)\n")
        L.append("| Признак | Вклад в AUC |\n|---|---|")
        for i in np.argsort(-imp.importances_mean)[:10]:
            L.append(f"| {NAMES[FEATURES[i]]} | {imp.importances_mean[i]:+.3f} |")

    for F in TEST_FREEZES:
        tt = dedupe(test[test["freeze"] == F], "p").head(20)
        L.append(f"\n## {kind.capitalize()}: что модель предсказала бы в конце {F} и что случилось к {F + 3}–{F + 5}\n")
        L.append("| Единица | p | тогда | потом | исход |\n|---|---|---|---|---|")
        for _, r in tt.iterrows():
            res = "✅ ×%.1f" % r.ratio if r.takeoff else ("➖ ×%.1f" % r.ratio if r.ratio >= 0.8 else "❌ ×%.1f" % r.ratio)
            L.append(f"| {r.term} | {r.p:.2f} | {int(r.recent_docs)} | {int(r.outcome_docs)} | {res} |")

    # ---- прогноз на сегодня: модель на всех размеченных годах ----
    full = pd.concat([tabs[F] for F in ALL_FREEZES], ignore_index=True)
    model = fit(full)
    now = features_at(U, ctx, NOW)
    now = pd.concat([now, predict(model, now)], axis=1)
    idx = pd.Index(U["names"]).get_indexer(now["term"])
    now["counts"] = [" ".join(str(int(round(v))) for v in U["counts"][i]) for i in idx]
    now["kind"] = kind
    now = dedupe(now, "p").reset_index(drop=True)
    fname = "forecast_terms.csv" if kind == "термины" else "forecast_topics.csv"
    now.to_csv(OUT / fname, index=False)
    top = now.head(30)
    if not publish:
        print(f"[{kind}] тест: база {rows[-1]['base']:.0%}, ансамбль@30 {rows[-1]['ens']:.0%} (×{lift_all:.1f}), "
              f"CV ×{cv.ens.mean() / cv.base.mean():.1f}; в прогноз не включены")
        return rows
    P.append(f"\n## {kind.capitalize()}: что взлетит к {NOW + 3}–{NOW + 5}\n")
    P.append(f"На временной проверке ансамбль дал точность топ-30 ×{lift_all:.1f} над базой, "
             f"в перекрёстной проверке по годам ×{cv.ens.mean() / cv.base.mean():.1f}. "
             f"Проверка прогноза после {NOW + 3}: `python src/forecast.py --score data/forecast/{fname}`.\n")
    P.append("| # | Единица | p(взлёт) | докум. 2024–26 | Исторический двойник | Судьба двойника | Динамика 2015–2026 |")
    P.append("|---|---|---|---|---|---|---|")
    for i, r in now.head(60).iterrows():
        fate = f"вырос ×{r.twin_ratio:.1f}" if r.twin_takeoff else (f"удержался ×{r.twin_ratio:.1f}" if r.twin_ratio >= 0.8 else f"угас ×{r.twin_ratio:.1f}")
        if i < 30:
            P.append(f"| {i + 1} | **{r.term}** | {r.p:.2f} | {int(r.recent_docs)} | {r.twin} ({r.twin_freeze}) | {fate} | `{r.counts}` |")
        forecast_rows.append(dict(kind=kind, rank=i + 1, term=r.term, p=round(float(r.p), 3), p_gb=round(float(r.p_gb), 3),
                                  p_knn=round(float(r.p_knn), 3), recent_docs=int(r.recent_docs), growth=round(float(r.growth), 2),
                                  twin=r.twin, twin_freeze=int(r.twin_freeze), twin_ratio=round(float(r.twin_ratio), 2),
                                  twin_takeoff=int(r.twin_takeoff), counts=[int(c) for c in r.counts.split()],
                                  industry_share=round(float(r.industry_share), 2), elite_share=round(float(r.elite_share), 2),
                                  china_share=round(float(r.china_share), 2)))
    print(f"[{kind}] тест: база {rows[-1]['base']:.0%}, ансамбль@30 {rows[-1]['ens']:.0%} (×{lift_all:.1f}), "
          f"CV ×{cv.ens.mean() / cv.base.mean():.1f}; прогноз топ-10: " + ", ".join(top.head(10)["term"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", help="проверить старый прогноз (csv) по текущим данным")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    ctx = context(df)
    if args.score:
        return score(args.score, df, ctx)
    print(f"документов {len(df)}")
    L = ["# Валидация предсказателя взлёта\n",
         "Взлёт: через 3–5 лет частота единицы (на 1000 документов года) выросла минимум вдвое и набрала ≥ 15 документов. "
         "Ручной скор это формула текущего лексического детектора. Ансамбль это среднее вероятности бустинга и доли "
         "взлетевших среди 25 ближайших исторических соседей. Точность @k считается после склейки вложенных фраз.\n"]
    P = [f"# Прогноз: что взлетит к {NOW + 3}–{NOW + 5}\n",
         f"Зафиксировано {date.today().isoformat()}. Критерий успеха: частота единицы в {NOW + 3}–{NOW + 5} минимум вдвое выше, "
         f"чем в {NOW - 2}–{NOW}, и ≥ 15 документов. Двойник это ближайший по признакам термин прошлого; его судьба "
         "показывает, чем обычно заканчивалась такая траектория.\n"]
    rows, forecast_rows = [], []
    rows += run_unit(unit_terms(df, ctx), ctx, L, P, forecast_rows)
    rows += run_unit(unit_topics(df, ctx), ctx, L, P, forecast_rows, publish=False)
    (OUT / "validation.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "forecast.md").write_text("\n".join(P), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "validation.csv", index=False)
    (OUT / "forecast.json").write_text(json.dumps(dict(date=date.today().isoformat(), horizon=[NOW + 3, NOW + 5],
                                                        validation=rows, forecast=forecast_rows), ensure_ascii=False, indent=1, default=float),
                                       encoding="utf-8")


if __name__ == "__main__":
    main()
