#!/usr/bin/env python3
"""Per-document novelty for the VPS, independent of which corpus folders the VPS has.

    DOC_EMB=.../doc_emb python novelty_export.py --cutoff 2026

Reads ${DOC_EMB}/*.npz (ids, year; same sorted order as prior_sim_gpu.py) and ${DOC_EMB}/../prior_sim_<cutoff>.npz.
Writes ${DOC_EMB}/../novelty_<cutoff>.npz with one row per recent document (cutoff-2..cutoff):
  ids, year, novelty (1 - best cosine to any document <= cutoff-4), nearest (id of that earlier document),
  cluster (-1 or component id among the 5% most novel "orphans", links at cosine >= 0.85),
  cluster_size, cluster_growing (size in the cutoff year > size two years earlier).
numpy only (union-find in python), so it runs inside the container. The VPS matches rows by document id.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

EMB = Path(os.environ["DOC_EMB"])


def load_ids():
    ids, years = [], []
    for f in sorted(EMB.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        if len(z["year"]) == 0:
            continue
        ids.append(z["ids"]); years.append(z["year"])
    return np.concatenate(ids), np.concatenate(years).astype(int)


def components(n: int, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(r.tolist(), c.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    _, lab = np.unique(roots, return_inverse=True)
    return lab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=2026)
    a = ap.parse_args()
    ids, year = load_ids()
    z = np.load(EMB.parent / f"prior_sim_{a.cutoff}.npz", allow_pickle=False)
    recent, prior, best, arg = z["recent"], z["prior"], z["best"], z["arg"]
    assert len(recent) == int(((year >= a.cutoff - 2) & (year <= a.cutoff)).sum()), "doc_emb changed after prior_sim"
    orph, er, ec = z["orph"], z["edge_r"], z["edge_c"]
    lab = components(len(orph), er, ec)
    sizes = np.bincount(lab)
    yr = year[orph]
    now = np.bincount(lab[yr == a.cutoff], minlength=len(sizes))
    before = np.bincount(lab[yr == a.cutoff - 2], minlength=len(sizes))
    cluster = np.full(len(recent), -1, np.int64)
    pos = np.searchsorted(recent, orph)
    cluster[pos] = lab
    csize = np.where(cluster >= 0, sizes[np.maximum(cluster, 0)], 0)
    grow = np.where(cluster >= 0, (now > before)[np.maximum(cluster, 0)], False)
    out = EMB.parent / f"novelty_{a.cutoff}.npz"
    np.savez_compressed(out, ids=ids[recent], year=year[recent].astype(np.int16), novelty=(1.0 - best).astype(np.float32),
                        nearest=ids[prior[arg]], cluster=cluster, cluster_size=csize.astype(np.int32),
                        cluster_growing=grow)
    print(f"recent {len(recent):,}, orphans {len(orph):,}, clusters >= 5: {int((sizes >= 5).sum())}, "
          f"growing >= 5: {int(((sizes >= 5) & (now > before)).sum())} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
