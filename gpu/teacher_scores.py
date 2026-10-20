#!/usr/bin/env python3
"""Teacher scores for distillation: cross-encoder BAAI/bge-reranker-v2-m3 over (query, candidate record) pairs.

    HF_HOME=... python3 teacher_scores.py --dir .../distill --out .../distill/teacher.npz

Input (built on the VPS by scripts/distill_adapter.py inputs): pairs.jsonl {qid, query, cands[K]}, docs.jsonl {i, text}.
Output: teacher.npz with scores [n_queries, K] in the order of cands. torch + transformers only (container).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    d = Path(a.dir)
    pairs = [json.loads(l) for l in open(d / "pairs.jsonl", encoding="utf-8")]
    docs = {}
    for l in open(d / "docs.jsonl", encoding="utf-8"):
        x = json.loads(l)
        docs[x["i"]] = x["text"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    K = len(pairs[0]["cands"])
    flat = [(p["query"], docs[c]) for p in pairs for c in p["cands"]]
    print(f"queries {len(pairs)}, pairs {len(flat)}, device {dev}", flush=True)
    out = np.zeros(len(flat), np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(flat), a.batch):
            q, t = zip(*flat[s:s + a.batch])
            enc = tok(list(q), list(t), padding=True, truncation=True, max_length=a.max_len, return_tensors="pt").to(dev)
            out[s:s + a.batch] = model(**enc).logits.view(-1).float().cpu().numpy()
            if (s // a.batch) % 100 == 0:
                print(f"  {s + len(q)}/{len(flat)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    np.savez_compressed(a.out, scores=out.reshape(len(pairs), K), qid=np.array([p["qid"] for p in pairs]))
    print(f"done -> {a.out}; {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
