"""Переранжирование ближайших кандидатов кросс-энкодером (флаг REGISTRY_RERANK=1).

Векторный поиск (USER-bge-m3) сравнивает запрос и запись по отдельности посчитанным векторам: русский запрос и
английская запись с короткими названиями часто расходятся («мемристоры» рядом с «memory experts»). Кросс-энкодер читает
пару «запрос — текст записи» целиком и даёт балл соответствия; им переупорядочиваются TOPN ближайших по вектору.
Модель BAAI/bge-reranker-v2-m3 (многоязычная, локальная с Hugging Face), CPU. Эффект измеряется
scripts/eval_retrieval.py --rerank на одних и тех же запросах (docs/EVAL.md).
"""
from __future__ import annotations

import os

import numpy as np

MODEL = os.getenv("REGISTRY_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
MAX_LEN = 192


class Reranker:
    def __init__(self, name: str = MODEL):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    def scores(self, query: str, docs: list[str], batch: int = 32) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(docs), batch):
                enc = self.tok([query] * len(docs[i:i + batch]), docs[i:i + batch], padding=True, truncation=True,
                               max_length=MAX_LEN, return_tensors="pt")
                out.append(self.model(**enc).logits.view(-1).float().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)
