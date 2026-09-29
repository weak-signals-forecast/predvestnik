"""Пул кандидатов для точечной проверки моделью — общий для llm_assess.py и stage2_pool.py.

Пул = общий топ GLOBAL_N по обученному баллу  ∪  для каждой области топ PER_AREA_N по баллу среди технологий,
у которых заметная доля свежих работ выходит в рубриках этой области (AREA_GROUPS_POOL). Всё — только данные,
без модели. Зачем по областям: общий балл выше всего у ИИ/LLM-тем, и в общем топе-20 тыс. было ~300 технологий
финтеха — выдаче по финтеху и промышленности не из чего было выбирать.

Рубрики — классификация arXiv и подобласти OpenAlex; с 27.09 добавлено докачанное железо: 2208 (электроника) —
edge и инфраструктура, 2209 (производство) — промышленность.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GLOBAL_N = 20000
PER_AREA_N = 3000
MIN_AREA_SHARE = 0.30
MIN_AREA_DOCS = 5
AREA_GROUPS_POOL = {
    "Edge": ("ax:cs.NI", "ax:cs.AR", "ax:cs.ET", "oa:1705", "oa:1708", "oa:2208"),
    "Защита ИИ": ("ax:cs.CR",),
    "Индустриальный ИИ": ("ax:eess.SY", "oa:2207", "oa:2209"),
    "Инфраструктура ИИ": ("ax:cs.DC", "ax:cs.AR", "ax:cs.PF", "ax:cs.ET", "oa:1708", "oa:2208"),
    "Роботы": ("ax:cs.RO", "oa:2207"),
    "Финтех": ("q-fin", "econ", "oa:2003"),
}


def candidates(reg) -> pd.DataFrame:
    from .stage2 import GENERIC_END, GENERIC_START
    r = pd.read_parquet(reg / "ranked.parquet")
    c = r[r.candidate].copy()
    ph = c.phrase.str.replace("_", " ")
    return c[~(ph.str.startswith(GENERIC_START) | ph.str.endswith(GENERIC_END))].reset_index(drop=True)


def area_shares(reg, c: pd.DataFrame) -> pd.DataFrame:
    """Доля свежих работ сущности в рубриках каждой области (0, если работ по группам меньше MIN_AREA_DOCS)."""
    g = np.load(reg / "groups.npz")
    groups = [str(x) for x in g["groups"]]
    grp = g["grp"]
    sci = pd.read_parquet(reg / "all_phrases.parquet", columns=["key", "eid"])
    eid = c.key.map(dict(zip(sci.key, sci.eid)))
    has = eid.notna().values
    idx = eid[has].astype(int).values
    tot = np.zeros(len(c))
    tot[has] = grp[:, idx].sum(axis=0)
    out = pd.DataFrame(index=c.index)
    for area, prefixes in AREA_GROUPS_POOL.items():
        rows = [i for i, name in enumerate(groups)
                if any(name == p or name.startswith("ax:" + p + ".") or name.startswith(p + ".") for p in prefixes)]
        a = np.zeros(len(c))
        if rows:
            a[has] = grp[np.ix_(rows, idx)].sum(axis=0)
        out[area] = np.where(tot >= MIN_AREA_DOCS, a / np.maximum(tot, 1), 0.0)
    return out


def select(reg, global_n: int = GLOBAL_N, per_area_n: int = PER_AREA_N) -> pd.DataFrame:
    """Пул с колонкой source: 'global' или название области (первая, через которую технология вошла)."""
    c = candidates(reg)
    sh = area_shares(reg, c)
    parts = [c.nlargest(global_n, "learned").assign(source="global")]
    for area in AREA_GROUPS_POOL:
        m = sh[area] >= MIN_AREA_SHARE
        parts.append(c[m].nlargest(per_area_n, "learned").assign(source=area))
    p = pd.concat(parts).drop_duplicates("key")
    return p
