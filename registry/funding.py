"""Деньги в бизнес-слое: раунды SEC Form D у стартапов, чьи технологии известны из описаний Y Combinator.

    python -m registry.funding      # -> <registry>/funding.parquet (признаки по ключу сущности)

Form D — уведомление о частном размещении в США: название компании, дата, сумма, отрасль (грубо). Описаний
технологий в нём нет, поэтому источником названий технологий он быть не может. Связка:
  описание компании YC → фразы → ключи сущностей бизнес-слоя → название компании → её подачи Form D.
Название сравнивается после нормализации (регистр, пунктуация, юридические формы Inc/LLC/Corp/Ltd/Co).
Поправки (ISAMENDMENT) и пулы инвестфондов не считаются. Сумма — TOTALAMOUNTSOLD (продано), если есть.

Признаки сущности: число компаний YC с этой технологией в описании; из них — с раундом по Form D в
2023–2026; сумма этих раундов; год первого раунда.
"""
from __future__ import annotations

import glob
import io
import json
import re
import sys
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

from .corpus import CORPUS, REGISTRY
from .text import candidates, entity_key

REG = REGISTRY
RECENT = (2023, 2024, 2025, 2026)
LEGAL = re.compile(r"\b(inc|incorporated|llc|l\.l\.c|corp|corporation|co|company|ltd|limited|plc|pbc|lp|holdings)\b\.?")


def norm_name(s: str) -> str:
    s = LEGAL.sub(" ", (s if isinstance(s, str) else "").lower())
    return " ".join(re.findall(r"[a-z0-9]+", s))


def load_formd() -> pd.DataFrame:
    rows = []
    for z in sorted(glob.glob(str(CORPUS / "sec_formd/*_d.zip"))):
        zf = zipfile.ZipFile(z)
        name = lambda suf: next(n for n in zf.namelist() if n.upper().endswith(suf))
        rd = lambda suf, cols: pd.read_csv(io.BytesIO(zf.read(name(suf))), sep="\t", usecols=cols, dtype=str,
                                           on_bad_lines="skip", encoding_errors="replace")
        iss = rd("ISSUERS.TSV", ["ACCESSIONNUMBER", "ENTITYNAME", "IS_PRIMARYISSUER_FLAG"])
        off = rd("OFFERING.TSV", ["ACCESSIONNUMBER", "INDUSTRYGROUPTYPE", "ISAMENDMENT", "TOTALAMOUNTSOLD",
                                  "ISPOOLEDINVESTMENTFUNDTYPE"])
        sub = rd("FORMDSUBMISSION.TSV", ["ACCESSIONNUMBER", "FILING_DATE"])
        d = iss[iss.IS_PRIMARYISSUER_FLAG == "YES"].merge(off, on="ACCESSIONNUMBER").merge(sub, on="ACCESSIONNUMBER")
        d = d[(d.ISAMENDMENT.str.lower() != "true") & (d.ISPOOLEDINVESTMENTFUNDTYPE.str.lower() != "true")]
        rows.append(d)
    f = pd.concat(rows, ignore_index=True)
    f["year"] = pd.to_datetime(f.FILING_DATE, format="%d-%b-%Y", errors="coerce").dt.year
    f["amount"] = pd.to_numeric(f.TOTALAMOUNTSOLD, errors="coerce").fillna(0)
    f["name"] = f.ENTITYNAME.map(norm_name)
    return f[f.year.notna()]


def main() -> int:
    biz = pd.read_parquet(REG / "all_business.parquet", columns=["key"])
    keys = set(biz.key)
    f = load_formd()
    by_name = f.groupby("name")
    print(f"подач Form D (без поправок и фондов): {len(f):,}; компаний: {f.name.nunique():,}".replace(",", " "))
    yc = []
    for p in glob.glob(str(CORPUS / "yc/*.json")):
        yc += json.load(open(p, encoding="utf-8"))
    comp_keys = {}
    for r in yc:
        text = " . ".join(str(r.get(x) or "") for x in ("one_liner", "long_description", "tags"))
        ks = {entity_key(ph) for ph in candidates(text)} & keys
        if ks:
            comp_keys[norm_name(r.get("name"))] = ks
    matched = {n for n in comp_keys if n in by_name.groups}
    print(f"компаний YC с технологиями в описании: {len(comp_keys):,}; найдены в Form D: {len(matched):,}".replace(",", " "))
    agg = defaultdict(lambda: {"yc_companies": 0, "funded_recent": 0, "raised_recent": 0.0, "first_raise": np.nan})
    for n, ks in comp_keys.items():
        g = by_name.get_group(n) if n in matched else None
        rec = g[g.year.isin(RECENT)] if g is not None else None
        for k in ks:
            a = agg[k]
            a["yc_companies"] += 1
            if rec is not None and len(rec):
                a["funded_recent"] += 1
                a["raised_recent"] += float(rec.amount.sum())
            if g is not None and len(g):
                a["first_raise"] = np.nanmin([a["first_raise"], g.year.min()])
    out = pd.DataFrame([{"key": k, **v} for k, v in agg.items()])
    out.to_parquet(REG / "funding.parquet", index=False)
    top = out.sort_values("raised_recent", ascending=False).head(15)
    print(top.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
