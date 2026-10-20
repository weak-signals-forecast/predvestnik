"""Эмбеддинги документов с кэшем. Считает только те id, которых ещё нет в кэше.

    python src/embed.py            # все data/openalex/*.jsonl и data/patents/*.jsonl

Кэш: data/embeddings/<model>.npz (ids, vectors). Модель BAAI/bge-base-en-v1.5, 768 измерений,
векторы нормированы, косинус = скалярное произведение.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "BAAI/bge-base-en-v1.5"
CACHE = ROOT / "data" / "embeddings" / "bge-base.npz"
MAX_CHARS = 2000  # ~400 токенов: заголовок плюс начало аннотации


def load_docs() -> list[dict]:
    docs = []
    for pattern in ("data/openalex/*.jsonl", "data/patents/*.jsonl"):
        for p in sorted(glob.glob(str(ROOT / pattern))):
            if ".v1." in p:
                continue
            with open(p, encoding="utf-8") as f:
                docs.extend(json.loads(line) for line in f)
    return docs


def doc_text(d: dict) -> str:
    return (d["title"].strip() + ". " + d.get("abstract", "").strip())[:MAX_CHARS]


def load_cache() -> dict[str, np.ndarray]:
    if not CACHE.exists():
        return {}
    z = np.load(CACHE, allow_pickle=False)
    return dict(zip(z["ids"].tolist(), z["vectors"]))


def save_cache(cache: dict[str, np.ndarray]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    ids = np.array(list(cache), dtype=str)
    vecs = np.stack(list(cache.values())).astype(np.float32)
    np.savez(CACHE, ids=ids, vectors=vecs)


def main():
    docs = load_docs()
    cache = load_cache()
    todo = [d for d in docs if d["id"] not in cache]
    print(f"документов {len(docs)}, в кэше {len(cache)}, считать {len(todo)}")
    if not todo:
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=device)
    model.max_seq_length = 384
    texts = [doc_text(d) for d in todo]
    vecs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    for d, v in zip(todo, vecs):
        cache[d["id"]] = v
    save_cache(cache)
    print(f"сохранено {len(cache)} векторов в {CACHE.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
