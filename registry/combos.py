"""Вариант А2: новые сочетания устоявшихся терминов («атипичные сочетания», Uzzi et al., 2013).

    python -m registry.combos      # -> <corpus>/registry/combos.parquet

Многие слабые сигналы — не новое слово, а старая технология в новом контексте: оптическая коммутация
в дата-центрах для ИИ, кредитный скоринг в блокчейне, фонды денежного рынка в токенизированном виде.
Одиночная фраза такого роста не показывает: оба термина давно есть и по отдельности не растут. Растёт
их совместная встречаемость.

Термин (зафиксировано до замера): фраза словаря, встреченная не меньше чем в TERM_MIN_DF работах OpenAlex +
arXiv за все годы, впервые — не позже TERM_BORN_BY, не массовая (доля свежих работ ≤ TERM_MAX_RATE),
не штамп (spread ≤ 0.8), не общая и не обрывок. То есть «устоявшийся, но не общий» термин.

Пара (a, b): термины без общих слов, встреченные в одной работе (заголовок + аннотация + ключевые слова).
Кандидат: в свежем окне R не меньше PAIR_MIN_R работ, доля пары выросла к базе B минимум в 2^PAIR_MIN_GROWTH
раз (сглаживание +1 работа). Источники OpenAlex и arXiv складываются; доли — на число прочитанных работ окна.
"""
from __future__ import annotations

import collections
import gzip
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import CORPUS, SOURCES, YEARS, docs, files, REGISTRY
from .text import candidates

REG = REGISTRY
R, B = (2024, 2025, 2026), (2019, 2020, 2021, 2022)
TERM_MIN_DF = 50
TERM_BORN_BY = 2020
TERM_MAX_RATE = 5e-3
PAIR_MIN_R = 5
PAIR_MIN_GROWTH = 2.0      # в 4 раза
MAX_TERMS_PER_DOC = 25     # в работе с большим числом терминов берём самые специфичные (редкие)
BUCKETS = 64
WORKERS = int(os.environ.get("REGISTRY_WORKERS", "6"))

_T: dict[str, int] = {}
_DF: np.ndarray | None = None
_WORDS: list[frozenset] = []
_KEEP: set[int] | None = None
_NT = 0   # число терминов (не фраз: несколько вариантов фразы указывают на один термин)


def terms() -> pd.DataFrame:
    t = pd.read_parquet(REG / "all_phrases.parquet")
    d = np.load(REG / "counts.npz")["df"]
    oa, ax = SOURCES.index("openalex"), SOURCES.index("arxiv")
    t["df_all"] = (d[:, oa, :] + d[:, ax, :]).sum(axis=1)
    sel = ((t.df_all >= TERM_MIN_DF) & (t.first_year > 0) & (t.first_year <= TERM_BORN_BY)
           & (t.max_rate_recent <= TERM_MAX_RATE) & ~(t.spread > 0.8) & ~t.generic & (t.super_share < 0.8))
    return t[sel].reset_index(drop=True)


def _init(term_list: list[str], df_all: list[int], keep: set[int] | None, term_eids: list[int] | None = None) -> None:
    """Любой вариант фразы из словаря, принадлежащий сущности-термину, указывает на этот термин."""
    global _T, _DF, _WORDS, _KEEP, _NT
    _NT = len(term_list)
    e2t = {e: i for i, e in enumerate(term_eids)}
    phrases = [l for l in (REG / "vocab.txt").read_text(encoding="utf-8").split("\n") if l]
    p2e = np.load(REG / "phrase2entity.npy")
    _T = {ph: e2t[int(e)] for ph, e in zip(phrases, p2e) if int(e) in e2t}
    _DF = np.asarray(df_all)
    _WORDS = [frozenset(ph.replace("_", " ").split()) for ph in term_list]
    _KEEP = keep


def _doc_pairs(text: str):
    ids = list({i for ph in candidates(text) if (i := _T.get(ph)) is not None})
    if len(ids) < 2:
        return
    if len(ids) > MAX_TERMS_PER_DOC:
        ids = sorted(ids, key=lambda i: _DF[i])[:MAX_TERMS_PER_DOC]
    ids.sort()
    T = _NT
    for a, b in combinations(ids, 2):
        if _WORDS[a] & _WORDS[b]:
            continue        # общие слова: «tool poisoning» и «poisoning attacks» — один термин, не пара
        yield a * T + b


def _pass_recent(args):
    """Проход по свежим работам: пары раскладываются по корзинам на диск (иначе не хватит памяти)."""
    source, path, tmp = args
    outs = {}
    n = 0
    for d in docs(source, path):
        if d["year"] not in R:
            continue
        n += 1
        for key in _doc_pairs(d["text"]):
            b = zlib.crc32(key.to_bytes(8, "little")) % BUCKETS
            if b not in outs:
                outs[b] = gzip.open(Path(tmp) / f"b{b:02d}_{os.getpid()}.txt.gz", "at")
            outs[b].write(f"{key}\n")
    for o in outs.values():
        o.close()
    return n


def _pass_base(args):
    """Проход по базовому окну: считаем только пары, уже отобранные по свежему окну."""
    source, path = args
    c = collections.Counter()
    n = 0
    for d in docs(source, path):
        if d["year"] not in B:
            continue
        n += 1
        for key in _doc_pairs(d["text"]):
            if key in _KEEP:
                c[key] += 1
    return n, c


def _window_files(window) -> list[tuple[str, str]]:
    import re
    out = []
    for s, p in files():
        if s == "hf":
            continue
        if s == "openalex" and int(re.search(r"year=(\d+)", p).group(1)) not in window:
            continue
        out.append((s, p))
    return out


def build() -> pd.DataFrame:
    t = terms()
    tl, dfl, el = t.phrase.tolist(), t.df_all.astype(int).tolist(), t.eid.astype(int).tolist()
    T = len(tl)
    print(f"устоявшихся терминов: {T:,}".replace(",", " "), flush=True)
    tmp = REG / "tmp_pairs"
    tmp.mkdir(exist_ok=True)
    for f in tmp.glob("*.gz"):
        f.unlink()
    t0 = time.time()
    todo = [(s, p, str(tmp)) for s, p in _window_files(R)]
    nR = 0
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(tl, dfl, None, el)) as ex:
        for i, n in enumerate(ex.map(_pass_recent, todo, chunksize=2), 1):
            nR += n
            if i % 200 == 0:
                print(f"  свежее окно: файлов {i}/{len(todo)}, {time.time() - t0:.0f} с", flush=True)
    cR = {}
    for b in range(BUCKETS):
        c = collections.Counter()
        for f in tmp.glob(f"b{b:02d}_*.txt.gz"):
            with gzip.open(f, "rt") as fh:
                c.update(int(l) for l in fh)
        cR.update({k: v for k, v in c.items() if v >= PAIR_MIN_R})
    for f in tmp.glob("*.gz"):
        f.unlink()
    tmp.rmdir()
    keep = set(cR)
    print(f"пар с ≥{PAIR_MIN_R} свежими работами: {len(keep):,}; {time.time() - t0:.0f} с".replace(",", " "), flush=True)

    cB = collections.Counter()
    nB = 0
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(tl, dfl, keep, el)) as ex:
        for n, c in ex.map(_pass_base, _window_files(B), chunksize=2):
            nB += n
            cB.update(c)
    keys = np.fromiter(cR.keys(), dtype=np.int64)
    r = np.fromiter((cR[k] for k in keys), dtype=np.float64)
    b = np.fromiter((cB.get(k, 0) for k in keys), dtype=np.float64)
    growth = np.log2(((r + 1) / nR) / ((b + 1) / nB))
    out = pd.DataFrame({"a": [tl[k // T] for k in keys], "b": [tl[k % T] for k in keys],
                        "pair_recent": r.astype(int), "pair_base": b.astype(int), "growth": growth})
    out = out[out.growth >= PAIR_MIN_GROWTH]
    # ранжирование: рост пары, но без награды за объём — слабый сигнал маленький
    out["score"] = out.growth.clip(upper=8) - 0.5 * np.log10((out.pair_recent / 200).clip(lower=1))
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["phrase"] = out.a + " × " + out.b
    print(f"свежих работ {nR:,}, базовых {nB:,}; новых сочетаний: {len(out):,}; {time.time() - t0:.0f} с".replace(",", " "))
    return out


def main() -> int:
    out = build()
    out.to_parquet(REG / "combos.parquet", index=False)
    print(out.head(40)[["phrase", "pair_recent", "pair_base", "growth"]].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
