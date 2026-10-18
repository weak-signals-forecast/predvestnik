"""Машина времени по теме: что система показала бы на срезе по заданной области, и на сколько лет раньше массовости.

    GPB_CORPUS=~/gpb_corpus python -m registry.time_machine_topic --cutoff 2021 --topic nlp
    GPB_CORPUS=~/gpb_corpus python -m registry.time_machine_topic --cutoff 2017 --topic nlp --model registry_v3

Пример экспертов заказчика: «увидеть в 2017 году будущий генеративный ИИ за три года до GPT-3». Здесь это проверка
на цифрах, без списка ожидаемых ответов:
  1. берётся реестр, собранный «как будто сейчас конец года среза» (<корпус>/registry_<cutoff>: словарь из
     заголовков до среза, ряды до среза), признаки и оси считаются только по данным до среза;
  2. тема задаётся группами корпуса (категории arXiv, подобласти OpenAlex), а не списком слов: кандидат по теме,
     если не меньше TOPIC_SHARE его свежих работ лежит в этих группах;
  3. балл — историческая модель с осями. На срезе, на котором она обучена (2021), берутся предсказания вне
     обучения (5-кратная кросс-валидация); на другом срезе (2017) — модель 2021 как есть: честный перенос назад;
  4. исход — из полного реестра (registry_v3, годы до 2026): взлёт по правилу машины времени (доля в окне
     cutoff+3..cutoff+5 втрое выше доли окна среза и ≥ MIN_FUTURE работ) и год массовости — первый год, когда доля
     работ источника ≥ MASS (как порог массовости в score.py). Опережение = год массовости − год среза.

Результат: web/data/time_machine_<cutoff>_<topic>.json (тот же контракт, что time_machine.json, плюс lead_years и
mass_year у записей и mean_lead в целом) и строка в web/data/time_machine_index.json для витрины.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .corpus import CORPUS, SOURCES, YEARS

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "web" / "data"
TOPICS = {
    "nlp": {"title": "Обработка естественного языка и языковые модели", "groups": ["ax:cs.CL"]},
    "cv": {"title": "Компьютерное зрение", "groups": ["ax:cs.CV", "oa:1707"]},
    "materials": {"title": "Материалы и конденсированное состояние", "groups": ["ax:cond-mat.mtrl-sci", "ax:cond-mat.mes-hall",
                                                                                  "ax:cond-mat.supr-con", "oa:2505", "oa:2504", "oa:3104"]},
    "security": {"title": "Безопасность", "groups": ["ax:cs.CR"]},
    "robotics": {"title": "Робототехника", "groups": ["ax:cs.RO"]},
}
TOPIC_SHARE = 0.4
TAKEOFF, MIN_FUTURE, MASS = 3.0, 30, 1e-3
TOP = 15
INC_MAX = 3.0


def topic_mask(reg: Path, eids: np.ndarray, groups: list[str]) -> np.ndarray:
    g = np.load(reg / "groups.npz", allow_pickle=True)
    names = [str(x) for x in g["groups"]]
    grp = g["grp"]
    idx = [names.index(x) for x in groups if x in names]
    if not idx:
        raise SystemExit(f"в {reg} нет групп {groups}; есть, например: {names[:12]}")
    total = grp[:, eids].sum(axis=0)
    share = grp[idx][:, eids].sum(axis=0) / np.maximum(total, 1)
    return share >= TOPIC_SHARE


def outcomes(keys: pd.Series, cutoff: int, now_reg: Path) -> pd.DataFrame:
    """Исход по полному реестру: взлёт, год массовости, работы по годам."""
    a = pd.read_parquet(now_reg / "all_phrases.parquet", columns=["key", "eid"]).drop_duplicates("key")
    df = np.load(now_reg / "counts.npz")["df"]
    tot = json.loads((now_reg / "totals.json").read_text(encoding="utf-8"))
    oa, ax = SOURCES.index("openalex"), SOURCES.index("arxiv")
    yi = {y: i for i, y in enumerate(YEARS)}
    eid = keys.map(dict(zip(a.key, a.eid)))
    out = {"docs_then": [], "docs_later": [], "takeoff": [], "mass_year": [], "lead_years": []}
    fut = [y for y in range(cutoff + 3, cutoff + 6) if y in yi]
    now = [y for y in range(cutoff - 2, cutoff + 1) if y in yi]

    def share(e, years, src):
        n = sum(tot.get(f"{SOURCES[src]}|{y}", 0) for y in years)
        return sum(df[e, src, yi[y]] for y in years) / max(n, 1)

    for e in eid:
        if e != e:
            for k in out:
                out[k].append(np.nan if k != "takeoff" else False)
            continue
        e = int(e)
        d_then = int(sum(df[e, ax, yi[y]] + df[e, oa, yi[y]] for y in now))
        d_fut = int(sum(df[e, ax, yi[y]] + df[e, oa, yi[y]] for y in fut))
        s_now, s_fut = share(e, now, ax), share(e, fut, ax)
        take = bool((s_fut + 1e-9) / (s_now + 1e-9) >= TAKEOFF and d_fut >= MIN_FUTURE)
        mass = next((y for y in YEARS if y > cutoff and max(share(e, [y], ax), share(e, [y], oa)) >= MASS), None)
        out["docs_then"].append(d_then); out["docs_later"].append(d_fut); out["takeoff"].append(take)
        out["mass_year"].append(mass if mass else np.nan); out["lead_years"].append((mass - cutoff) if mass else np.nan)
    return pd.DataFrame(out, index=keys.index)


def main() -> int:
    from .history_model import FEATS, features, future_label, model
    from .axes import AXES
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=2021)
    ap.add_argument("--topic", default="nlp", choices=sorted(TOPICS))
    ap.add_argument("--model", default="registry_v3", help="папка реестра с history_model_axes.joblib и полными рядами")
    ap.add_argument("--top", type=int, default=TOP)
    a = ap.parse_args()
    reg = CORPUS / f"registry_{a.cutoff}"
    now_reg = CORPUS / a.model
    os.environ.setdefault("GPB_REGISTRY", str(reg))
    feats = FEATS + AXES
    past = features(reg, a.cutoff, with_axes=True)
    X = past[feats].astype(float)
    bundle = joblib.load(now_reg / "history_model_axes.joblib")
    if a.cutoff == 2021:                      # срез обучения: предсказания вне обучения
        from sklearn.model_selection import StratifiedKFold
        import registry.history_model as hm
        hm.REG_PAST = reg
        y = future_label(past, "v1").values.astype(int)
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            p[te] = model().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        how = "предсказания вне обучения (5-кратная кросс-валидация на срезе обучения)"
    else:
        p = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]
        how = f"модель, обученная на срезе 2021, применена к признакам на срез {a.cutoff}"
    past["p"] = p
    inc = past.incremental_lex.fillna(-np.inf) >= INC_MAX
    past["score"] = past.p * np.where(inc, 0.85, 1.0)
    t = TOPICS[a.topic]
    from .signal_base import is_fragment, is_umbrella
    mask = topic_mask(reg, past.eid.values.astype(int), t["groups"])
    clean = ~past.phrase.map(is_fragment) & ~past.phrase.map(is_umbrella) & ~past.generic.astype(bool) & (past.n_words >= 2)
    pool = past[mask & clean & (past.volume_recent >= 5)].copy()
    res = pd.concat([pool, outcomes(pool.key, a.cutoff, now_reg)], axis=1)
    top = res.nlargest(a.top, "score")
    base_rate = float(res.takeoff.mean()) if len(res) else float("nan")
    top_rate = float(top.takeoff.mean())
    lead = top.lead_years.dropna()
    items = [{"term": r.phrase.replace("_", " "), "p": round(float(r.p), 3), "takeoff": bool(r.takeoff),
              "docs_then": int(r.docs_then) if r.docs_then == r.docs_then else 0,
              "docs_later": int(r.docs_later) if r.docs_later == r.docs_later else 0,
              "growth": round(float((r.docs_later + 1) / (r.docs_then + 1)), 1),
              "mass_year": int(r.mass_year) if r.mass_year == r.mass_year else None,
              "lead_years": int(r.lead_years) if r.lead_years == r.lead_years else None,
              "incremental": bool(inc.get(r.Index, False)), "parent": (r.parent if isinstance(r.parent, str) else None),
              "market_after": {"hn": 0, "yc": 0, "press": 0}, "market_before": {"hn": 0, "yc": 0, "press": 0}}
             for r in top.itertuples()]
    data = {"cutoff": a.cutoff, "future": f"{a.cutoff + 3}–{a.cutoff + 5}", "topic": a.topic, "title": t["title"],
            "groups": t["groups"], "pool": int(len(res)), "top": a.top, "base_rate": round(base_rate, 3),
            "top_rate": round(top_rate, 3), "lift": round(top_rate / base_rate, 1) if base_rate else None,
            "lift_ci": None, "how": how,
            "rule": f"взлёт — доля работ arXiv в {a.cutoff + 3}–{a.cutoff + 5} не меньше чем втрое выше доли "
                    f"{a.cutoff - 2}–{a.cutoff} и не меньше {MIN_FUTURE} работ; массовость — доля ≥ 0,1 % работ источника за год",
            "mean_lead": round(float(lead.mean()), 1) if len(lead) else None, "n_mass": int(len(lead)), "items": items}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"time_machine_{a.cutoff}_{a.topic}.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    idx_p = OUT_DIR / "time_machine_index.json"
    idx = json.loads(idx_p.read_text(encoding="utf-8")) if idx_p.exists() else {"items": []}
    idx["items"] = [x for x in idx["items"] if x["file"] != out.name] + [{"file": out.name, "cutoff": a.cutoff,
                                                                            "topic": a.topic, "title": t["title"]}]
    idx["items"].sort(key=lambda x: (x["cutoff"], x["topic"]))
    idx_p.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"срез {a.cutoff}, тема «{t['title']}»: в теме {len(res)} кандидатов, взлетели {base_rate:.1%}; "
          f"из топ-{a.top} взлетели {top_rate:.0%}; стали массовыми {len(lead)}, среднее опережение {data['mean_lead']} лет -> {out}")
    for r in items:
        print(f"  {'✓' if r['takeoff'] else '·'} {r['term'][:40]:40} p={r['p']:.2f} {r['docs_then']:>4} → {r['docs_later']:>5}"
              f"  масс.: {r['mass_year'] or '—'}  {'улучшение: ' + r['parent'] if r['incremental'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
