"""Применить обученные модели (ранжировщик, экспертная и историческая с осями) к новому реестру без переобучения.

    GPB_CORPUS=~/gpb_corpus GPB_REGISTRY=~/gpb_corpus/registry_v5 python -m registry.apply_models [--models registry_v3] [--show graphene]

Зачем. Ранжировщик `registry.ranker` обучается на `semantic_pairs.csv`, которого в репозитории нет, а экспертная
и историческая модели — на разметке и срезе 2021. Для реестра, собранного на новом корпусе (v5, cond-mat, срез 2017),
модели не переобучаются, а применяются как есть: признаки считаются по тому же коду, что при обучении.

Результат: <реестр>/scored.parquet — key, phrase, learned (ранжировщик), expert_p, history_p, blend
(0,7 · expert_p + 0,3 · history_p), оси; плюс печать верхних записей и записей по подстроке --show.
Без бизнес-слоя (all_business.parquet, wiki.parquet, funding.parquet) бизнес-признаки равны нулю — это честно
отражает, что таких наблюдений нет.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import axes as axes_mod
from .axes import AXES
from .corpus import CORPUS, CUTOFF, REGISTRY


def ranker_table(reg: Path, cutoff: int) -> pd.DataFrame:
    """Таблица признаков ранжировщика (registry/ranker.py: table) для реестра, возможно без бизнес-слоя."""
    sci = pd.read_parquet(reg / "all_phrases.parquet")
    t = pd.DataFrame({"key": sci.key, "phrase": sci.phrase, "candidate": sci.candidate, "eid": sci.eid,
                      "sci_log_volume": np.log1p(sci.volume_recent), "sci_growth": sci.growth,
                      "sci_age": np.where(sci.first_year > 0, cutoff - sci.first_year, np.nan),
                      "sci_log_rate": np.log10(sci.max_rate_recent + 1e-7), "sci_spread": sci.spread,
                      "sci_sources": sci.sources_recent,
                      "sci_oa_share": sci.oa_recent / (sci.oa_recent + sci.ax_recent).replace(0, np.nan),
                      "first_year": sci.first_year})
    bp = reg / "all_business.parquet"
    if bp.exists():
        biz = pd.read_parquet(bp)
        b = pd.DataFrame({"key": biz.key, "biz_log_hn": np.log1p(biz.hn_recent), "biz_hn_growth": biz.growth,
                          "biz_log_yc": np.log1p(biz.yc_recent), "biz_log_press": np.log1p(biz.press_recent),
                          "biz_log_launches": np.log1p(biz.hn_launches)})
        t = t.merge(b, on="key", how="left")
    for c in ("biz_log_hn", "biz_hn_growth", "biz_log_yc", "biz_log_press", "biz_log_launches"):
        t[c] = t[c].fillna(0.0) if c in t else 0.0
    wp, fp = reg / "wiki.parquet", reg / "funding.parquet"
    t["wiki_present"] = t.key.isin(set(pd.read_parquet(wp).key)).astype(int) if wp.exists() else 0
    if fp.exists():
        f = pd.read_parquet(fp)[["key", "raised_recent", "funded_recent"]]
        t = t.merge(f, on="key", how="left")
        t["biz_log_raised"] = np.log1p(t.raised_recent.fillna(0) / 1e6)
        t["biz_funded"] = t.funded_recent.fillna(0)
    else:
        t["biz_log_raised"], t["biz_funded"] = 0.0, 0
    t["layers"] = (t.sci_log_volume.fillna(0) > 0).astype(int) + (t.biz_log_hn > 0) + (t.biz_log_yc > 0) + (t.biz_log_press > 0)
    t["n_words"] = t.phrase.str.count(" ") + 1
    t["name_like"] = t.phrase.str.contains(r"\d").astype(int)
    wc = collections.Counter(w for ph in (reg / "vocab.txt").read_text(encoding="utf-8").split("\n") if ph
                             for w in set(ph.split()))
    t["specificity"] = t.phrase.map(lambda ph: np.log10(min((wc.get(w, 1) for w in ph.split()), default=1)))
    return t


def rubric_shares(reg: Path, keys: pd.Series, eids: pd.Series) -> pd.DataFrame:
    """theory_share и academic_share как в registry/signal_base.py, но без привязки к REGISTRY модуля."""
    from .signal_base import ACADEMIC_RUBRICS
    g = np.load(reg / "groups.npz")
    names = [str(x) for x in g["groups"]]
    grp = g["grp"]
    th = [i for i, n in enumerate(names)
          if (n.startswith("ax:math") and n != "ax:math.OC") or n.startswith(("ax:econ", "ax:hep", "ax:astro", "ax:nucl"))
          or n in ("ax:stat.ME", "ax:stat.ST", "ax:stat.AP", "ax:stat.CO", "ax:stat.OT", "ax:math-ph", "ax:funct-an",
                   "ax:alg-geom", "ax:gr-qc", "ax:q-fin.MF", "ax:q-fin.PR", "ax:q-fin.ST", "ax:q-fin.PM",
                   "ax:q-fin.RM", "ax:q-fin.CP")]
    ac = [names.index(r) for r in ACADEMIC_RUBRICS if r in names]
    idx = eids.astype(int).values
    tot = grp[:, idx].sum(axis=0)
    def share(rows):
        return np.where(tot >= 5, grp[rows][:, idx].sum(axis=0) / np.maximum(tot, 1), np.nan) if rows else np.full(len(idx), np.nan)
    return pd.DataFrame({"theory_share": share(th), "academic_share": share(ac)}, index=keys.index)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="registry_v3", help="папка реестра с ranker.joblib, expert_model_axes.joblib, history_model_axes.joblib")
    ap.add_argument("--show", default="", help="подстрока фразы для печати (например graphene)")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    reg, mdir = REGISTRY, CORPUS / a.models
    t = ranker_table(reg, CUTOFF)
    ax = axes_mod.load(reg)[["key"] + AXES + ["parent", "docs_now"]]
    t = t.merge(ax, on="key", how="left")
    t["layer_lag_hn"] = t.layer_lag_hn.fillna(-1)
    t["layer_lag_market"] = t.layer_lag_market.fillna(-1)
    t = pd.concat([t, rubric_shares(reg, t.key, t.eid)], axis=1)

    rk = joblib.load(mdir / "ranker.joblib")
    t["learned"] = rk["model"].predict_proba(t[rk["features"]].astype(float))[:, 1]
    em = joblib.load(mdir / "expert_model_axes.joblib")
    t["expert_p"] = em["model"].predict_proba(t[em["features"]].astype(float))[:, 1]
    # историческая: признаки registry.history_model.features (volume_recent, growth, … + оси)
    from . import history_model as hm
    hm.REG_NOW = reg
    h = hm.features(reg, CUTOFF, with_axes=True)
    hmod = joblib.load(mdir / "history_model_axes.joblib")
    h["history_p"] = hmod["model"].predict_proba(h[hmod["features"]].astype(float))[:, 1]
    t = t.merge(h[["key", "history_p"]], on="key", how="left")
    t["blend"] = 0.7 * t.expert_p + 0.3 * t.history_p.fillna(t.expert_p)
    cand = t[t.candidate.fillna(False).astype(bool)].copy()
    out = reg / "scored.parquet"
    cand.to_parquet(out)
    print(f"реестр {reg.name}: фраз {len(t):,}, кандидатов {len(cand):,} -> {out}".replace(",", " "))
    cols = ["phrase", "learned", "expert_p", "history_p", "blend", "first_year", "parent", "incremental_lex"]
    pd.set_option("display.width", 200)
    print("\nверх по смеси моделей:")
    print(cand.nlargest(a.top, "blend")[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if a.show:
        sub = cand[cand.phrase.str.contains(a.show, case=False, regex=False)]
        print(f"\nкандидаты с «{a.show}»: {len(sub)}")
        print(sub.nlargest(a.top, "blend")[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
