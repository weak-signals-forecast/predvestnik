"""Проверка поиска без ручной разметки: найдёт ли сервис запись по заголовку её собственной работы.

    REGISTRY_BUNDLE=artifacts/registry_v5d python scripts/eval_retrieval.py [--n 2000] [--rerank] [--out ...json]

Запрос — заголовок источника записи, который НЕ участвовал в построении её вектора (вектор строится по названию и первым
mini_bundle.EMB_TITLES заголовкам; берётся следующий). Цель — сама запись. Метрики: Recall@1/10/50 и MRR среди
всех записей набора, отдельно по областям. Это проверка поиска «запрос → нужная технология», а не качества сигналов:
модели слабости в ней не участвуют. --rerank: ближайшие TOPN по вектору переупорядочиваются кросс-энкодером
(registry_service.rerank), чтобы сравнить порядок до и после на одних и тех же запросах.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from registry.mini_bundle import EMB_TITLES  # noqa: E402

BUNDLE = Path(os.getenv("REGISTRY_BUNDLE", "artifacts/registry"))
TOPN = 80


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--queries", default="", help="jsonl {query, key}: свои запросы (scripts/make_ru_queries.py) вместо заголовков")
    a = ap.parse_args()
    sig = pd.read_parquet(BUNDLE / "signals.parquet")
    with gzip.open(BUNDLE / "cards.json.gz", "rt", encoding="utf-8") as fh:
        cards = json.load(fh)
    E = np.load(BUNDLE / "emb.npy").astype(np.float32)
    rows = []
    for i, k in enumerate(sig.key):
        ts = [(d.get("title") or "").strip() for d in (cards.get(k) or {}).get("sources") or [] if d.get("title")]
        if len(ts) > EMB_TITLES:
            rows.append((i, ts[EMB_TITLES]))
    if a.queries:
        pos = {k: i for i, k in enumerate(sig.key)}
        rows = [(pos[x["key"]], x["query"]) for x in map(json.loads, open(a.queries, encoding="utf-8")) if x["key"] in pos]
    rng = np.random.default_rng(0)
    if len(rows) > a.n:
        rows = [rows[j] for j in sorted(rng.choice(len(rows), a.n, replace=False))]
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("deepvk/USER-bge-m3", device="cpu")
    m.max_seq_length = 64                                   # как в сервисе
    t0 = time.time()
    Q = m.encode([q for _, q in rows], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    S = Q @ E.T
    print(f"запросов {len(rows)}, векторы за {time.time() - t0:.0f} с", flush=True)
    target = np.array([i for i, _ in rows])
    ranks = (S > S[np.arange(len(rows)), target][:, None]).sum(1) + 1
    res = {"n": len(rows), "bundle": str(BUNDLE), "vector": _metrics(ranks)}
    if a.rerank:
        from registry_service.rerank import Reranker
        rr = Reranker()
        texts = [_doc_text(sig, cards, j) for j in range(len(sig))]
        rk = []
        t0 = time.time()
        for q_i, (i, q) in enumerate(rows):
            top = np.argsort(-S[q_i])[:TOPN]
            if i not in set(top.tolist()):
                rk.append(ranks[q_i]); continue            # вне ТОП-N переранжирование не помогает: ранг прежний
            sc = rr.scores(q, [texts[j] for j in top])
            order = top[np.argsort(-sc)]
            rk.append(int(np.where(order == i)[0][0]) + 1)
        res["rerank"] = _metrics(np.array(rk))
        res["rerank_sec_per_query"] = round((time.time() - t0) / len(rows), 2)
    areas = sig.areas.map(lambda l: list(l)[0] if len(l) else "").values
    res["by_area_vector"] = {ar: _metrics(ranks[areas[target] == ar]) for ar in sorted(set(areas[target]))}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def _doc_text(sig, cards, j) -> str:
    c = cards.get(sig.key[j]) or {}
    ts = [(d.get("title") or "") for d in (c.get("sources") or [])][:EMB_TITLES]
    name = sig.name_ru[j]
    desc = sig.description_ru[j] if isinstance(sig.description_ru[j], str) else ""
    return ". ".join([x for x in [name, sig.phrase[j].replace("_", " "), desc] + ts if x])


def _metrics(r: np.ndarray) -> dict:
    if not len(r):
        return {}
    return {"n": int(len(r)), "recall@1": round(float((r <= 1).mean()), 3), "recall@10": round(float((r <= 10).mean()), 3),
            "recall@50": round(float((r <= 50).mean()), 3), "mrr": round(float((1 / r).mean()), 3),
            "median_rank": int(np.median(r))}


if __name__ == "__main__":
    sys.exit(main())
