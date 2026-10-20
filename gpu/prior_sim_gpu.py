#!/usr/bin/env python3
"""GPU-heavy half of predecessors.py: for every "recent" document (cutoff-2..cutoff) the best
cosine similarity to any "prior" document (<= cutoff-4), plus the similarity graph among the
top-5% least similar ("orphans"). torch + numpy only, so it runs inside the container.

    DOC_EMB=/data/doc_emb python prior_sim_gpu.py --cutoff 2026 2017

Output: ${DOC_EMB}/../prior_sim_<cutoff>.npz with recent (indices in the sorted-npz order used by
predecessors.load_all), best, arg (index into prior), orph, edge_r, edge_c. predecessors.py picks it up
through PRIOR_SIM=<file> and then only needs numpy/scipy/pandas (CPU, minutes).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

EMB = Path(os.environ["DOC_EMB"])
ORPHAN_TOP = 0.05
LINK_SIM = 0.85
CHUNK = 8192


def load_all():
    years, embs = [], []
    for f in sorted(EMB.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        if len(z["year"]) == 0:
            continue
        years.append(z["year"]); embs.append(z["emb"])
    return np.concatenate(years).astype(int), np.concatenate(embs)


def max_sim(q: np.ndarray, base: np.ndarray, dev, torch):
    B = torch.from_numpy(base).to(dev)
    best = np.zeros(len(q), np.float32); arg = np.zeros(len(q), np.int64)
    # the score block (rows x len(base), fp16) must fit in GPU memory: ~2 GB per block
    rows = max(64, min(CHUNK, int(2e9 / (2 * max(len(base), 1)))))
    for i in range(0, len(q), rows):
        Q = torch.from_numpy(q[i:i + rows]).to(dev)
        s = Q @ B.T
        v, j = s.max(dim=1)
        best[i:i + rows] = v.float().cpu().numpy(); arg[i:i + rows] = j.cpu().numpy()
        del s
    del B
    return best, arg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, nargs="+", default=[2026, 2017])
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    year, emb = load_all()
    print(f"documents {len(year):,}, dims {emb.shape[1]}, device {dev}", flush=True)
    for cutoff in a.cutoff:
        out = EMB.parent / f"prior_sim_{cutoff}.npz"
        recent = np.where((year >= cutoff - 2) & (year <= cutoff))[0]
        prior = np.where(year <= cutoff - 4)[0]
        print(f"cutoff {cutoff}: recent {len(recent):,}, prior {len(prior):,}", flush=True)
        if not len(recent) or not len(prior):
            print("  nothing to do"); continue
        q = emb[recent]
        budget = 0.3 * torch.cuda.get_device_properties(0).total_memory if dev == "cuda" else 8e9
        step = max(100_000, int(budget / (emb.shape[1] * 2)))
        best = np.full(len(recent), -1.0, np.float32); arg = np.zeros(len(recent), np.int64)
        for s in range(0, len(prior), step):
            b, g = max_sim(q, emb[prior[s:s + step]], dev, torch)
            upd = b > best
            best[upd], arg[upd] = b[upd], g[upd] + s
        novelty = 1.0 - best
        thr = np.quantile(novelty, 1 - ORPHAN_TOP)
        orph = recent[novelty >= thr]
        O = torch.from_numpy(emb[orph]).to(dev)
        rows, cols = [], []
        for i in range(0, len(orph), CHUNK):
            s = (O[i:i + CHUNK] @ O.T) >= LINK_SIM
            r, c = torch.nonzero(s, as_tuple=True)
            rows.append((r + i).cpu().numpy()); cols.append(c.cpu().numpy())
        del O
        np.savez_compressed(out, cutoff=np.array(cutoff), recent=recent, prior=prior, best=best, arg=arg,
                            orph=orph, edge_r=np.concatenate(rows), edge_c=np.concatenate(cols))
        print(f"  novelty median {np.median(novelty):.3f}, p95 {np.percentile(novelty, 95):.3f}; orphans {len(orph):,}; "
              f"-> {out}; {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
