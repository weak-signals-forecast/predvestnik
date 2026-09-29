"""Кто стоит за сигналом: институты и компании из авторов работ, стартапы YC, раунды Form D.

    python -m registry.actors          # -> actors.parquet (по технологиям базы), institutions.json (названия)

Один проход по работам OpenAlex 2023–2026: у каждой работы, где в заголовке или аннотации есть термин технологии
(тот же разбор, что при сборке реестра), берутся институты авторов — идентификатор, тип (компания, университет, …)
и страна. Названия институтов запрашиваются в OpenAlex пакетами только для верхних институтов каждой технологии.
Стартапы YC — из бизнес-слоя (компании, в описании которых есть термин), раунды — из SEC Form D (funding.parquet).
Признаки распространения: сколько независимых организаций подхватили тему, сколько из них компании, сколько стран.
"""
from __future__ import annotations

import collections
import glob
import gzip
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .corpus import CORPUS, REGISTRY as REG
from .text import candidates

TOP_INST = 6
_P2K: dict[str, str] = {}
_COMBO: list[tuple[str, "re.Pattern", "re.Pattern"]] = []


def _init(keys: list[str]) -> None:
    global _P2K, _COMBO
    want = set(keys)
    # составные сигналы «метод × объект»: работа относится к связке, если в ней есть и метод, и объект
    from .composites import METHODS, OBJECTS
    _COMBO = [(k, re.compile(METHODS[k[6:].split("|")[0]][0]), re.compile(OBJECTS[k[6:].split("|")[1]][0]))
              for k in keys if k.startswith("combo:")]
    phrases = [l for l in (REG / "vocab.txt").read_text(encoding="utf-8").split("\n") if l]
    p2e = np.load(REG / "phrase2entity.npy")
    ek = [l for l in (REG / "entity_keys.txt").read_text(encoding="utf-8").split("\n") if l]
    _P2K = {ph: ek[int(e)] for ph, e in zip(phrases, p2e) if int(e) >= 0 and ek[int(e)] in want}


def _file(path: str):
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for l in fh:
            r = json.loads(l)
            text = (r.get("title") or "") + " . " + (r.get("abstract") or "")
            ks = {k for ph in candidates(text) if (k := _P2K.get(ph)) is not None}
            if _COMBO:
                low = text.lower()
                ks |= {k for k, mre, ore in _COMBO if mre.search(low) and ore.search(low)}
            if not ks:
                continue
            inst = {}
            for a in r.get("authorships") or []:
                for i in a.get("institutions") or []:
                    if i.get("id"):
                        inst[i["id"]] = (i.get("type") or "", i.get("country") or "")
            if inst:
                out.append((sorted(ks), inst))
    return out


def scan(keys: list[str]) -> dict[str, collections.Counter]:
    files = sorted(glob.glob(str(CORPUS / "openalex/subfield=*/year=202[3-6]/*.jsonl.gz")))
    per_key: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    info: dict[str, tuple[str, str]] = {}
    with ProcessPoolExecutor(6, initializer=_init, initargs=(keys,)) as ex:
        for hits in ex.map(_file, files, chunksize=2):
            for ks, inst in hits:
                info.update(inst)
                for k in ks:
                    per_key[k].update(inst.keys())
    return per_key, info


def names(ids: set[str]) -> dict[str, str]:
    """Названия институтов из OpenAlex, пакетами по 50 (ключ из ~/.openalex_key, если есть)."""
    import requests
    cache_p = REG / "institutions.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    todo = sorted(i for i in ids if i not in cache)
    key_file = os.path.expanduser("~/.openalex_key")
    api_key = open(key_file).read().strip() if os.path.exists(key_file) else None
    for n in range(0, len(todo), 50):
        part = todo[n:n + 50]
        params = {"filter": "openalex:" + "|".join(part), "per-page": 50, "select": "id,display_name"}
        ok = False
        for use_key in ([True, False] if api_key else [False]):   # лимит ключа исчерпан — пробуем без ключа
            q = dict(params, api_key=api_key) if use_key else params
            try:
                r = requests.get("https://api.openalex.org/institutions", params=q, timeout=60)
                if r.status_code == 429:
                    continue
                r.raise_for_status()
                for x in r.json().get("results", []):
                    cache[x["id"].rsplit("/", 1)[-1]] = x.get("display_name") or ""
                ok = True
                break
            except Exception as e:
                print("  ошибка:", str(e)[:80], flush=True)
        if not ok:
            print(f"  OpenAlex не отдаёт названия (лимит); пропущено {len(todo) - n} институтов — повторить позже",
                  flush=True)
            break
    cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


def main() -> int:
    from .export_web import business_evidence
    base = pd.read_parquet(REG / "signal_base.parquet", columns=["key", "phrase"])
    keys = base.key.tolist()
    t0 = time.time()
    import pickle
    scan_cache = REG / "actors_scan.pkl"             # разбор работ — на диск: повтор не начинается с нуля
    if scan_cache.exists() and pickle.loads(scan_cache.read_bytes())["keys"] == keys:
        per_key, info = pickle.loads(scan_cache.read_bytes())["data"]
    else:
        per_key, info = scan(keys)
        scan_cache.write_bytes(pickle.dumps({"keys": keys, "data": (dict(per_key), info)}))
    print(f"работ OpenAlex 2023–2026 разобрано за {time.time() - t0:.0f} с; технологий с институтами: {len(per_key)}",
          flush=True)
    top_ids = {i for c in per_key.values() for i, _ in c.most_common(TOP_INST)}
    nm = names({i.rsplit("/", 1)[-1] for i in top_ids})
    biz = business_evidence(set(keys))
    fund = pd.read_parquet(REG / "funding.parquet").set_index("key") if (REG / "funding.parquet").exists() else None
    rows = []
    for k in keys:
        c = per_key.get(k, collections.Counter())
        types = collections.Counter(info[i][0] for i in c)
        countries = collections.Counter(info[i][1] for i in c if info[i][1])
        top = [{"name": nm.get(i.rsplit("/", 1)[-1]) or i, "type": info[i][0], "country": info[i][1], "works": int(n)}
               for i, n in c.most_common(TOP_INST)]
        yc = [{"name": d.get("title"), "url": d.get("url")} for d in biz.get(k, []) if d.get("layer") == "yc"][:8]
        f = fund.loc[k] if fund is not None and k in fund.index else None
        rows.append({"key": k, "orgs": len(c), "companies": int(types.get("company", 0)),
                     "universities": int(types.get("education", 0)), "countries": len(countries),
                     "cn_share": round(countries.get("CN", 0) / max(1, sum(countries.values())), 3),
                     "top_orgs": json.dumps(top, ensure_ascii=False),
                     "yc": json.dumps(yc, ensure_ascii=False), "yc_count": len(yc),
                     "funded": int(f.funded_recent) if f is not None else 0,
                     "raised": float(f.raised_recent) if f is not None else 0.0})
    a = pd.DataFrame(rows)
    a.to_parquet(REG / "actors.parquet")
    print(f"-> actors.parquet: {len(a)} технологий; с институтами {int((a.orgs > 0).sum())}, "
          f"со стартапами YC {int((a.yc_count > 0).sum())}; медиана организаций {int(a.orgs.median())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
