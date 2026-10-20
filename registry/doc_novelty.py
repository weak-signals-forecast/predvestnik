"""Новизна документов → признак записи набора: насколько свежие работы технологии не похожи ни на одну прошлую.

    GPB_CORPUS=<корпус> GPB_REGISTRY=<реестр набора> python -m registry.doc_novelty --novelty novelty_2026.npz \
        --bundle artifacts/registry_v5d

Вход — файл кластера (for_cesga/server/novelty_export.py): для каждого документа 2024–2026 novelty = 1 − наибольшая
косинусная близость (bge-base) к любой работе ≤ 2022. Здесь по корпусу ищутся свежие документы, в тексте которых есть
фраза записи (тот же разбор, что при сборке реестра), и по их novelty считаются:
  doc_novelty_mean   средняя новизна свежих работ записи;
  doc_novelty_p90    90-й процентиль;
  doc_orphan_share   доля работ из 5 % самых новых по всему корпусу («без предшественников»);
  doc_n_recent       сколько свежих работ нашлось.
Колонки дописываются в axes.parquet набора; сервис рисует полоску «новизна» (rank.axes_view, doc_novelty_mean / 0,5).
Документы сопоставляются по id (W… OpenAlex без префикса URL, arXiv:…), поэтому корпус кластера и VPS могут различаться.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import REGISTRY, docs, files
from .text import candidates

_NOV: dict[str, float] = {}
_WANT: dict[str, str] = {}


def _norm_id(s: str) -> str:
    s = str(s)
    return s.rsplit("/", 1)[-1] if s.startswith("http") else s


def _scan(args):
    source, path = args
    out = defaultdict(list)
    for d in docs(source, path):
        if not d["year"] or d["year"] < 2024:
            continue
        v = _NOV.get(_norm_id(d["id"]))
        if v is None:
            continue
        for k in {k for ph in candidates(d["text"]) if (k := _WANT.get(ph)) is not None}:
            out[k].append(v)
    return dict(out)


def main() -> int:
    global _NOV, _WANT
    ap = argparse.ArgumentParser()
    ap.add_argument("--novelty", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    z = np.load(a.novelty, allow_pickle=False)
    nov = z["novelty"].astype(float)
    thr = float(np.quantile(nov, 0.95))
    _NOV = {_norm_id(i): float(v) for i, v in zip(z["ids"], nov)}
    print(f"документов с новизной {len(_NOV):,}; порог «без предшественников» {thr:.3f}".replace(",", " "), flush=True)

    b = Path(a.bundle)
    sig = pd.read_parquet(b / "signals.parquet", columns=["key"])
    ap_ = pd.read_parquet(REGISTRY / "all_phrases.parquet", columns=["key", "eid"]).drop_duplicates("key")
    eid2key = {int(e): k for k, e in zip(ap_.key, ap_.eid) if k in set(sig.key)}
    phrases = [l for l in (REGISTRY / "vocab.txt").read_text(encoding="utf-8").split("\n") if l]
    p2e = np.load(REGISTRY / "phrase2entity.npy")
    _WANT = {ph: eid2key[int(e)] for ph, e in zip(phrases, p2e) if int(e) in eid2key}
    print(f"записей набора {len(sig)}, вариантов фраз {len(_WANT):,}".replace(",", " "), flush=True)

    todo = [(s, p) for s, p in files() if s == "arxiv" or (s == "openalex" and "year=202" in p and
                                                            any(f"year={y}" in p for y in (2024, 2025, 2026)))]
    acc = defaultdict(list)
    with ProcessPoolExecutor(a.workers) as ex:              # fork: _NOV и _WANT наследуются без копирования по сети
        for n, got in enumerate(ex.map(_scan, todo, chunksize=4), 1):
            for k, vs in got.items():
                acc[k] += vs
            if n % 200 == 0:
                print(f"  файлов {n}/{len(todo)}", flush=True)
    rows = []
    for k, vs in acc.items():
        v = np.array(vs)
        rows.append({"key": k, "doc_novelty_mean": float(v.mean()), "doc_novelty_p90": float(np.quantile(v, 0.9)),
                     "doc_orphan_share": float((v >= thr).mean()), "doc_n_recent": int(len(v))})
    t = pd.DataFrame(rows, columns=["key", "doc_novelty_mean", "doc_novelty_p90", "doc_orphan_share", "doc_n_recent"])
    axp = b / "axes.parquet"
    ax = pd.read_parquet(axp)
    ax = ax.drop(columns=[c for c in t.columns if c != "key" and c in ax.columns]).merge(t, on="key", how="left")
    ax.to_parquet(axp)
    cov = ax.doc_novelty_mean.notna().mean()
    print(f"новизна посчитана у {ax.doc_novelty_mean.notna().sum()} из {len(ax)} записей ({cov:.0%}); "
          f"медиана {ax.doc_novelty_mean.median():.3f} (корпус {np.median(nov):.3f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
