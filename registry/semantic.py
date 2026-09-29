"""Оценка реестра по смыслу и сравнение моделей поиска (запрос → запись реестра).

    python -m registry.semantic [модель ...]    # по умолчанию все из MODELS, у кого есть веса локально
    -> <corpus>/registry/semantic_candidates.csv  (кандидаты каждой модели)
       <corpus>/registry/semantic_pairs.csv       (объединённые уникальные пары для разметки)

Зачем. Лексическая оценка (evaluate.py) сравнивает слова командных английских формулировок со словами
реестра и ошибается в обе стороны. Здесь для каждого из 100 сигналов каждая модель даёт TOP_K ближайших
записей из всех слоёв реестра (А — фразы науки, А2 — сочетания, В — бизнес-слой). Кандидаты всех моделей
объединяются и размечаются один раз, поэтому разметка не подыгрывает ни одной модели, а для каждой модели
видно, сколько настоящих совпадений она нашла.

Запросы на двух языках: русские названия из таблицы организаторов (так будет писать жюри, реестр при этом
английский) и английские формулировки команды. Лексическая база TF-IDF по буквосочетаниям — только для
английских запросов (русский запрос с английским реестром она сравнить не может).

Правило разметки пары (колонка same), зафиксировано до замера:
  совпадение  — запись называет ту же технологию, её синоним или прямой вариант (другое написание, число,
                более узкая формулировка той же вещи);
  нет         — более общее понятие («ai agents» для «AI agent identity»), только составная часть,
                соседняя или другая технология.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import CORPUS, REGISTRY

REG = REGISTRY
ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "labels" / "semantic_pairs.csv"   # разметка пар хранится в репозитории, не в папке сборки
XLSX = ROOT / "вводные данные" / "100_слабых_технологических_сигналов_сентябрь_2026.xlsx"
TOP_K = 5
MODELS = {
    "bge-base-en": {"name": "BAAI/bge-base-en-v1.5", "q": "Represent this sentence for searching relevant passages: ", "p": "", "ml": False},
    "bge-m3": {"name": "BAAI/bge-m3", "q": "", "p": "", "ml": True},
    "me5-large": {"name": "intfloat/multilingual-e5-large", "q": "query: ", "p": "passage: ", "ml": True},
    "user-bge-m3": {"name": "deepvk/USER-bge-m3", "q": "", "p": "", "ml": True},
}


def layers() -> pd.DataFrame:
    a = pd.read_parquet(REG / "registry.parquet", columns=["phrase"]).assign(layer="А")
    c = pd.read_parquet(REG / "combos.parquet", columns=["a", "b"])
    c = pd.DataFrame({"phrase": c.a + " × " + c.b, "layer": "А2"})
    b = pd.read_parquet(REG / "business.parquet", columns=["phrase"]).assign(layer="В")
    out = []
    for t in (a, c, b):
        t = t.reset_index(drop=True)
        t["layer_rank"] = np.arange(1, len(t) + 1)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def queries() -> list[dict]:
    import openpyxl
    rows = list(openpyxl.load_workbook(XLSX, read_only=True).worksheets[0].iter_rows(values_only=True))
    hi = next(i for i, r in enumerate(rows) if r and "Источники" in [str(c) for c in r])
    hdr = [str(c) for c in rows[hi]]
    ru = {str(r[hdr.index("№")]): str(r[hdr.index("Технология (слабый сигнал)")]) for r in rows[hi + 1:] if r[hdr.index("№")]}
    out = []
    for q in csv.DictReader(open(ROOT / "data/labels/positives_queries.csv", encoding="utf-8")):
        out.append({"id": int(q["id"]), "en": q["query"], "ru": ru.get(str(q["id"]), "")})
    return out


def texts(reg: pd.DataFrame) -> list[str]:
    return [p.replace("_", " ").replace(" × ", " and ") for p in reg.phrase]


def encode_registry(key: str, reg: pd.DataFrame, model) -> np.ndarray:
    path = REG / f"emb_{key}.npy"
    if path.exists() and np.load(path, mmap_mode="r").shape[0] == len(reg):
        return np.load(path, mmap_mode="r")
    cfg = MODELS[key]
    tx = [cfg["p"] + t for t in texts(reg)]
    parts, t0, step = [], time.time(), 20_000
    for s in range(0, len(tx), step):
        parts.append(model.encode(tx[s:s + step], batch_size=256, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False).astype(np.float16))
        print(f"  {key}: {min(s + step, len(tx)):,}/{len(tx):,}, {time.time() - t0:.0f} с".replace(",", " "), flush=True)
    E = np.concatenate(parts)
    np.save(path, E)
    return E


def top_k(E: np.ndarray, qv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sims = np.empty(E.shape[0], dtype=np.float32)
    for s in range(0, E.shape[0], 100_000):
        sims[s:s + 100_000] = np.asarray(E[s:s + 100_000], dtype=np.float32) @ qv
    top = np.argpartition(-sims, TOP_K)[:TOP_K]
    top = top[np.argsort(-sims[top])]
    return top, sims[top]


def run_model(key: str, reg: pd.DataFrame, qs: list[dict]) -> list[dict]:
    from sentence_transformers import SentenceTransformer
    import torch
    cfg = MODELS[key]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = SentenceTransformer(cfg["name"], device=dev)
    m.max_seq_length = 32          # записи реестра — короткие фразы
    E = encode_registry(key, reg, m)
    rows = []
    for lang in (("ru", "en") if cfg["ml"] else ("en",)):
        Q = m.encode([cfg["q"] + q[lang] for q in qs], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        for qi, q in enumerate(qs):
            idx, sc = top_k(E, Q[qi])
            rows += [{"id": q["id"], "lang": lang, "model": key, "k": k + 1, "candidate": reg.phrase[j],
                      "layer": reg.layer[j], "layer_rank": int(reg.layer_rank[j]), "cos": round(float(s), 3)}
                     for k, (j, s) in enumerate(zip(idx, sc))]
    return rows


def run_tfidf(reg: pd.DataFrame, qs: list[dict]) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(texts(reg))
    Q = vec.transform([q["en"].lower() for q in qs])
    S = (X @ Q.T).tocsc()
    rows = []
    for qi, q in enumerate(qs):
        col = S[:, qi].toarray().ravel()
        top = np.argpartition(-col, TOP_K)[:TOP_K]
        top = top[np.argsort(-col[top])]
        rows += [{"id": q["id"], "lang": "en", "model": "tfidf-char", "k": k + 1, "candidate": reg.phrase[j],
                  "layer": reg.layer[j], "layer_rank": int(reg.layer_rank[j]), "cos": round(float(col[j]), 3)}
                 for k, j in enumerate(top)]
    return rows


def report() -> None:
    """Итог по размеченным парам: покрытие реестра и сравнение моделей."""
    c = pd.read_csv(REG / "semantic_candidates.csv")
    lab = pd.read_csv(LABELS)[["id", "candidate", "same"]]
    c = c.merge(lab, on=["id", "candidate"], how="left")
    todo = c.same.isna().sum()
    if todo:
        print(f"ВНИМАНИЕ: не размечено пар: {todo} — цифры по новым моделям неполные")
    c["split"] = np.where(c.id % 2 == 1, "dev", "test")
    best = lab.groupby("id").same.max()
    print("\nПокрытие реестра (сигнал найден хотя бы одним методом среди его кандидатов):")
    for sp, ids in (("dev", best.index[best.index % 2 == 1]), ("test", best.index[best.index % 2 == 0]), ("все", best.index)):
        b = best.loc[ids]
        print(f"  {sp:5} точно {int((b >= 1).sum())}/{len(b)}   ядро или точно {int((b >= 0.5).sum())}/{len(b)}")
    print("\nМодели: сигналов, у которых среди 5 ближайших есть совпадение (точно / ядро или точно):")
    g = c.groupby(["model", "lang", "id"]).same.max().reset_index()
    for (m, lg), t in g.groupby(["model", "lang"]):
        print(f"  {m:12} {lg}:  {int((t.same >= 1).sum()):>3} / {int((t.same >= 0.5).sum()):>3}")
    hit = c[c.same >= 0.5].sort_values("layer_rank").drop_duplicates(["id", "layer"])
    print("\nВ каком слое найдены совпадения (сигналов):", hit.groupby("layer").id.nunique().to_dict())
    top = c[c.same >= 0.5].groupby("id").layer_rank.min()
    print("Лучшая позиция совпавшей записи внутри своего слоя, медиана:", int(top.median()) if len(top) else None,
          "; в первой тысяче своего слоя:", int((top <= 1000).sum()), "сигналов")


def _key(candidate: str) -> str:
    from .text import entity_key
    if " × " in candidate:
        a, b = candidate.split(" × ", 1)
        return " × ".join(sorted([entity_key(a), entity_key(b)]))
    return entity_key(candidate)


def compare(dirs: list[str]) -> None:
    """Сравнение версий реестра без повторной разметки: метки пар переносятся по ключу сущности
    (число и порядок слов не важны). Для каждой версии — покрытие и места совпавших записей в своих слоях."""
    lab = pd.read_csv(LABELS)
    lab = lab[lab.same.notna()].copy()
    lab["key"] = lab.candidate.map(_key)
    for d in dirs:
        d = Path(d).expanduser()
        ranks = {}
        a = pd.read_parquet(d / "registry.parquet", columns=["phrase"])
        ranks["А"] = {k: i + 1 for i, k in reversed(list(enumerate(a.phrase.map(_key))))}
        c = pd.read_parquet(d / "combos.parquet", columns=["a", "b"])
        ranks["А2"] = {k: i + 1 for i, k in reversed(list(enumerate((c.a + " × " + c.b).map(_key))))}
        b = pd.read_parquet(d / "business.parquet", columns=["phrase"])
        ranks["В"] = {k: i + 1 for i, k in reversed(list(enumerate(b.phrase.map(_key))))}
        sizes = {"А": len(a), "А2": len(c), "В": len(b)}
        rows = []
        for _, r in lab.iterrows():
            for layer, rk in ranks.items():
                if r.key in rk:
                    rows.append({"id": r.id, "same": r.same, "layer": layer, "rank": rk[r.key]})
        t = pd.DataFrame(rows)
        best = t.groupby("id").same.max() if len(t) else pd.Series(dtype=float)
        print(f"\n{d.name}: записей А {sizes['А']:,}, А2 {sizes['А2']:,}, В {sizes['В']:,}".replace(",", " "))
        for sp, m in (("dev", 1), ("test", 0)):
            bb = best[best.index % 2 == m]
            print(f"  {sp}: точно {int((bb >= 1).sum())}/50, ядро или точно {int((bb >= 0.5).sum())}/50")
        hit = t[t.same >= 0.5].groupby("id")["rank"].min()
        for layer in ("А", "А2", "В"):
            h = t[(t.same >= 0.5) & (t.layer == layer)].groupby("id")["rank"].min()
            if len(h):
                print(f"  слой {layer}: сигналов {len(h)}, место в слое — медиана {int(h.median()):,}, в первой 1000: {int((h <= 1000).sum())}, в первых 100: {int((h <= 100).sum())}".replace(",", " "))


def main() -> int:
    if sys.argv[1:] == ["report"]:
        report()
        return 0
    if sys.argv[1:2] == ["compare"]:
        compare(sys.argv[2:])
        return 0
    reg = layers()
    qs = queries()
    want = sys.argv[1:] or ["tfidf-char", *MODELS]
    out = REG / "semantic_candidates.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    for key in want:
        t0 = time.time()
        rows = run_tfidf(reg, qs) if key == "tfidf-char" else run_model(key, reg, qs)
        if len(old):
            old = old[old.model != key]
        old = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
        old.to_csv(out, index=False)
        print(f"{key}: готово за {time.time() - t0:.0f} с", flush=True)
    pairs = old.drop_duplicates(["id", "candidate"])[["id", "candidate", "layer", "layer_rank"]].copy()
    pairs["split"] = np.where(pairs.id % 2 == 1, "dev", "test")
    lab = LABELS
    if lab.exists():   # разметку уже размеченных пар не теряем
        prev = pd.read_csv(lab)[["id", "candidate", "same"]]
        pairs = pairs.merge(prev, on=["id", "candidate"], how="left")
    else:
        pairs["same"] = ""
    pairs.sort_values(["id", "candidate"]).to_csv(lab, index=False)
    print(f"уникальных пар для разметки: {len(pairs)} -> {lab}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
