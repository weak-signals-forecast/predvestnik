"""Сборка лексического реестра, шаг 1: словарь фраз и ряды по годам.

    python -m registry.build vocab     # проход 1: словарь фраз из заголовков 2023–2026
    python -m registry.build entities  # склейка вариантов фраз в сущности (kv cache = kv caches)
    python -m registry.build counts    # проход 2: документная частота фраз по источникам, годам и группам
    python -m registry.build all

Проход 1. Фразы-кандидаты берутся из заголовков (у HF — ещё из ключевых слов) свежих лет: название новой
технологии почти всегда попадает в заголовок работы, которая её вводит. Чтобы уложиться в память, фразы
раскладываются по 32 корзинам по хешу и считаются по корзинам. В словарь идёт фраза с документной
частотой в свежих заголовках не ниже VOCAB_MIN_DF.

Проход 2. По всем годам и полному тексту (заголовок, аннотация, ключевые слова) для каждой фразы словаря
считается, в скольких документах она встречается: по источнику и году, и отдельно по группам (подобласть
OpenAlex, категория arXiv) за свежие годы — для фильтра штампов. Плюс знаменатели: сколько документов
каждого источника и года вообще прочитано, потому что старые годы OpenAlex скачаны не полностью.

Результат в <corpus>/registry/: vocab.txt, counts.npz (df[фраза, источник, год]), groups.npz, totals.json.
100 сигналов организаторов здесь не используются никак.
"""
from __future__ import annotations

import collections
import gzip
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .corpus import CORPUS, CUTOFF, SOURCES, YEARS, docs, files, REGISTRY
from .text import candidates

OUT = REGISTRY
VOCAB_YEARS = tuple(range(CUTOFF - 3, CUTOFF + 1))          # 2023–2026 при срезе 2026
VOCAB_MIN_DF = 3          # зафиксировано до первого замера
RECENT_GROUP_YEARS = tuple(range(CUTOFF - 2, CUTOFF + 1))
BUCKETS = 32
WORKERS = int(os.environ.get("REGISTRY_WORKERS", "6"))


def _bucket(ph: str) -> int:
    return zlib.crc32(ph.encode()) % BUCKETS


# ---------- проход 1 ----------

def _vocab_file(args) -> tuple[str, int]:
    source, path, tmpdir = args
    outs = [gzip.open(Path(tmpdir) / f"b{b:02d}_{os.getpid()}.txt.gz", "at", encoding="utf-8") for b in range(BUCKETS)]
    n = 0
    for d in docs(source, path):
        if d["year"] not in VOCAB_YEARS:
            continue
        n += 1
        for ph in candidates(d["title"]):
            outs[_bucket(ph)].write(ph + "\n")
    for o in outs:
        o.close()
    return path, n


def build_vocab() -> None:
    tmp = OUT / "tmp_vocab"
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("*.gz"):
        f.unlink()
    todo = [(s, p, str(tmp)) for s, p in files() if s == "hf" or _maybe_recent(s, p)]
    t0, n_docs = time.time(), 0
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, (_, n) in enumerate(ex.map(_vocab_file, todo, chunksize=4), 1):
            n_docs += n
            if i % 200 == 0:
                print(f"  словарь: файлов {i}/{len(todo)}, заголовков {n_docs:,}, {time.time() - t0:.0f} с".replace(",", " "), flush=True)
    vocab: dict[str, int] = {}
    for b in range(BUCKETS):
        c = collections.Counter()
        for f in tmp.glob(f"b{b:02d}_*.txt.gz"):
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                c.update(l.rstrip("\n") for l in fh)
        vocab.update({ph: k for ph, k in c.items() if k >= VOCAB_MIN_DF})
    phrases = sorted(vocab)
    (OUT / "vocab.txt").write_text("\n".join(phrases) + "\n", encoding="utf-8")
    np.save(OUT / "vocab_df.npy", np.array([vocab[ph] for ph in phrases], dtype=np.int32))
    for f in tmp.glob("*.gz"):
        f.unlink()
    tmp.rmdir()
    print(f"словарь: {len(vocab):,} фраз из {n_docs:,} свежих заголовков, {time.time() - t0:.0f} с".replace(",", " "))


def build_entities() -> None:
    """Склейка вариантов фразы в сущность по ключу (набор основ слов): число и порядок слов не важны.
    Название сущности — самый частый в свежих заголовках вариант. Считать документы дальше будем по сущностям,
    поэтому документ с «kv cache» и «kv caches» засчитывается сущности один раз."""
    from .text import entity_key
    phrases = [l for l in (OUT / "vocab.txt").read_text(encoding="utf-8").split("\n") if l]
    df = np.load(OUT / "vocab_df.npy")
    best: dict[str, tuple[int, str]] = {}
    for ph, k in zip(phrases, df):
        key = entity_key(ph)
        if key not in best or k > best[key][0]:
            best[key] = (int(k), ph)
    keys = sorted(best)
    kid = {k: i for i, k in enumerate(keys)}
    (OUT / "entities.txt").write_text("\n".join(best[k][1] for k in keys) + "\n", encoding="utf-8")
    (OUT / "entity_keys.txt").write_text("\n".join(keys) + "\n", encoding="utf-8")
    np.save(OUT / "phrase2entity.npy", np.array([kid[entity_key(ph)] for ph in phrases], dtype=np.int32))
    print(f"сущности: {len(keys):,} из {len(phrases):,} фраз".replace(",", " "), flush=True)


def _maybe_recent(source: str, path: str) -> bool:
    """OpenAlex разложен по годам в пути — старые годы в проход 1 не читаем. arXiv читаем весь (год внутри)."""
    if source == "openalex":
        import re
        return int(re.search(r"year=(\d+)", path).group(1)) in VOCAB_YEARS
    return True


# ---------- проход 2 ----------

_VOCAB: dict[str, int] = {}
_GROUPS: dict[str, int] = {}


_NENT = 0


def _init(vocab_path: str, groups: list[str]) -> None:
    """Фраза словаря -> id сущности."""
    global _VOCAB, _GROUPS, _NENT
    phrases = [l for l in Path(vocab_path).read_text(encoding="utf-8").split("\n") if l]
    p2e = np.load(Path(vocab_path).parent / "phrase2entity.npy")
    _VOCAB = {ph: int(e) for ph, e in zip(phrases, p2e)}
    _NENT = int(p2e.max()) + 1
    _GROUPS = {g: i for i, g in enumerate(groups)}


def _count_file(args):
    """Для одного файла: {(источник, год): массив id фраз по документам}, пары (группа, id) за свежие годы, число документов."""
    source, path = args
    per_year: dict[int, list[int]] = collections.defaultdict(list)
    grp_keys: list[int] = []
    n_docs: collections.Counter = collections.Counter()
    V = _NENT
    for d in docs(source, path):
        y = d["year"]
        if y not in YEARS:
            continue
        n_docs[(d["group"], y)] += 1
        ids = list({i for ph in candidates(d["text"]) if (i := _VOCAB.get(ph)) is not None})   # сущность — раз на документ
        if not ids:
            continue
        per_year[y].extend(ids)
        if y in RECENT_GROUP_YEARS and (g := _GROUPS.get(d["group"], _GROUPS.get("ax:other"))) is not None:
            grp_keys.extend(g * V + i for i in ids)
    # разреженно: (id фраз, число документов) по годам — плотные массивы на весь словарь слишком тяжелы для передачи
    out = {y: np.unique(np.asarray(v, dtype=np.int64), return_counts=True) for y, v in per_year.items()}
    gk = np.unique(np.asarray(grp_keys, dtype=np.int64), return_counts=True) if grp_keys else (np.array([], np.int64), np.array([], np.int64))
    return source, out, gk, dict(n_docs)


def _all_groups() -> list[str]:
    """Группы заранее: 14+ подобластей OpenAlex из путей и категории arXiv из быстрого прохода по файлам."""
    import re
    g = {f"oa:{re.search(r'subfield=(\d+)', p).group(1)}" for s, p in files() if s == "openalex"}
    g.add("hf")
    cats: set[str] = set()
    for s, p in files():
        if s != "arxiv":
            continue
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for k, l in enumerate(fh):
                if k > 200:
                    break
                c = json.loads(l).get("categories")
                from .corpus import _arxiv_cats
                cs = _arxiv_cats(c)
                if cs:
                    cats.add("ax:" + cs[0])
    return sorted(g | cats | {"ax:other"})


def build_counts() -> None:
    vocab_path = OUT / "vocab.txt"
    V = sum(1 for l in (OUT / "entities.txt").read_text(encoding="utf-8").split("\n") if l)
    groups = _all_groups()
    si = {s: k for k, s in enumerate(SOURCES)}
    yi = {y: k for k, y in enumerate(YEARS)}
    df = np.zeros((V, len(SOURCES), len(YEARS)), dtype=np.int32)
    grp = np.zeros((len(groups), V), dtype=np.int32)
    totals: collections.Counter = collections.Counter()
    todo = files()
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(str(vocab_path), groups)) as ex:
        for i, (source, per_year, (gk, gc), n_docs) in enumerate(ex.map(_count_file, todo, chunksize=2), 1):
            for y, (ids, cnt) in per_year.items():
                df[ids, si[source], yi[y]] += cnt.astype(np.int32)
            if len(gk):
                np.add.at(grp, (gk // V, gk % V), gc.astype(np.int32))
            for (g, y), n in n_docs.items():
                totals[f"{source}|{y}"] += n
                totals[f"{source}|{y}|{g}"] += n
            if i % 100 == 0:
                print(f"  ряды: файлов {i}/{len(todo)}, {time.time() - t0:.0f} с", flush=True)
    np.savez_compressed(OUT / "counts.npz", df=df, sources=np.array(SOURCES), years=np.array(YEARS))
    np.savez_compressed(OUT / "groups.npz", grp=grp, groups=np.array(groups), years=np.array(RECENT_GROUP_YEARS))
    (OUT / "totals.json").write_text(json.dumps(dict(totals), ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"ряды: {V:,} фраз × {len(SOURCES)} источника × {len(YEARS)} лет, групп {len(groups)}, {time.time() - t0:.0f} с".replace(",", " "))


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    if what in ("vocab", "all"):
        build_vocab()
    if what in ("entities", "all"):
        build_entities()
    if what in ("counts", "all"):
        build_counts()
    return 0


if __name__ == "__main__":
    sys.exit(main())
