"""Проекты ЕС (CORDIS: Horizon 2020 и Horizon Europe) — слой промышленного внедрения для технологий реестра.

    python -m registry.cordis          # -> cordis.parquet (по ключам технологий) и cordis_docs.parquet (проекты)

У промышленных и аппаратных технологий рыночный след — не Hacker News и стартапы YC, а отраслевые консорциумы и
проекты с участием компаний. CORDIS — открытый реестр проектов ЕС: название, цели, ключевые слова, дата старта,
вклад ЕС. Термины технологий ищутся в названии, целях и ключевых словах тем же разбором, что при сборке реестра
(registry.text.candidates → словарь фраз → сущность). По каждой технологии: число проектов по годам старта,
вклад ЕС за 2022–2026, первый год, до 5 свежих проектов со ссылками (для карточки).
"""
from __future__ import annotations

import collections
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .corpus import CORPUS, REGISTRY as REG
from .text import candidates

ZIPS = [CORPUS / "cordis" / "cordis-h2020projects-json.zip", CORPUS / "cordis" / "cordis-HORIZONprojects-json.zip"]
YEARS = list(range(2014, 2027))
_P2K: dict[str, str] = {}


def _init() -> None:
    global _P2K
    phrases = [l for l in (REG / "vocab.txt").read_text(encoding="utf-8").split("\n") if l]
    p2e = np.load(REG / "phrase2entity.npy")
    ek = [l for l in (REG / "entity_keys.txt").read_text(encoding="utf-8").split("\n") if l]
    _P2K = {ph: ek[int(e)] for ph, e in zip(phrases, p2e) if int(e) >= 0}


def _chunk(args):
    path, names = args
    out = []
    with zipfile.ZipFile(path) as z:
        for n in names:
            try:
                d = json.loads(z.read(n))
            except Exception:
                continue
            y = str(d.get("startDate") or "")[:4]
            if not y.isdigit():
                continue
            text = " . ".join(str(d.get(f) or "") for f in ("title", "objective", "keywords"))
            ks = {k for ph in candidates(text) if (k := _P2K.get(ph)) is not None}
            if not ks:
                continue
            try:
                eu = float(d.get("ecMaxContribution") or 0)
            except (TypeError, ValueError):
                eu = 0.0
            out.append((sorted(ks), {"id": str(d.get("id")), "acronym": d.get("acronym") or "",
                                     "title": (d.get("title") or "")[:200], "year": int(y), "eu": eu}))
    return out


def main() -> int:
    jobs = []
    for p in ZIPS:
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.endswith(".json") and n.startswith("project-")]
        jobs += [(p, names[i:i + 400]) for i in range(0, len(names), 400)]
    per_key: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    eu_recent: dict[str, float] = collections.defaultdict(float)
    projects: dict[str, dict] = {}
    key_projects: dict[str, list] = collections.defaultdict(list)
    n_proj = 0
    with ProcessPoolExecutor(6, initializer=_init) as ex:
        for hits in ex.map(_chunk, jobs):
            for ks, pr in hits:
                n_proj += 1
                projects[pr["id"]] = pr
                for k in ks:
                    per_key[k][pr["year"]] += 1
                    if pr["year"] >= 2022:
                        eu_recent[k] += pr["eu"]
                    key_projects[k].append(pr["id"])
    rows = []
    for k, c in per_key.items():
        recent = sum(c[y] for y in range(2022, 2027))
        base = sum(c[y] for y in range(2016, 2021))
        first = min((y for y, n in c.items() if n > 0), default=None)
        top = sorted((projects[i] for i in key_projects[k]), key=lambda p: (-p["year"], -p["eu"]))[:5]
        rows.append({"key": k, "cordis_total": int(sum(c.values())), "cordis_recent": int(recent),
                     "cordis_base": int(base), "cordis_first_year": first,
                     "cordis_series": json.dumps({str(y): int(c[y]) for y in YEARS}),
                     "eu_recent_meur": round(eu_recent[k] / 1e6, 2),
                     "cordis_projects": json.dumps([{**p, "url": f"https://cordis.europa.eu/project/id/{p['id']}"}
                                                    for p in top], ensure_ascii=False)})
    out = pd.DataFrame(rows)
    out.to_parquet(REG / "cordis.parquet")
    print(f"проектов с терминами реестра: {n_proj}; технологий с проектами ЕС: {len(out)}; "
          f"из них с проектами 2022–2026: {int((out.cordis_recent > 0).sum())}")
    base = pd.read_parquet(REG / "signal_base.parquet", columns=["key", "areas"])
    b = base.merge(out, on="key", how="left")
    for a in ("Индустриальный ИИ", "Инфраструктура ИИ", "Edge", "Защита ИИ", "Финтех", "Роботы"):
        m = b.areas.map(lambda l, a=a: a in list(l))
        print(f"  база, {a:18}: с проектами ЕС 2022–2026 {(b[m].cordis_recent.fillna(0) > 0).mean():.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
