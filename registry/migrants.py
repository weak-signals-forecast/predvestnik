"""Технологии-мигранты: термин, который годами жил в одной науке и за последние три года пришёл в другую.

    GPB_CORPUS=<корпус> python -m registry.migrants scan <реестр>:<срез> [<реестр>:<срез> ...]
    GPB_CORPUS=<корпус> python -m registry.migrants check <реестр 2017>:2017 [<реестр 2021>:2021]
    GPB_CORPUS=<корпус> python -m registry.migrants now <реестр>:2026 [--bundle artifacts/registry_v5]

Зачем. Новые направления часто рождаются переносом метода из одной науки в другую: диффузионные модели — из
статистической физики в машинное обучение, языковые модели — в биологию белков, графовые сети — в материаловедение.
Рост внутри своей науки реестр уже видит; переход границы — отдельный измеримый признак.

scan: один проход по корпусу; для кандидатов каждого реестра — число документов по «науке» (раздел arXiv или подобласть
OpenAlex основной категории документа → одна из SCIENCES) и году. Результат <реестр>/sci_series.npz.

Мигрант на срез c (правило зафиксировано до проверки): у термина не меньше HIST_MIN документов за годы ≤ c−3, «домашняя»
наука — где их больше всего; в окне c−2..c есть наука-цель (не домашняя), где термин встречался в годы ≤ c−3 не больше
TARGET_HIST_MAX раз, а в окне — не меньше TARGET_MIN раз и не меньше TARGET_SHARE всех документов окна.

check: на исторических срезах доля взлетевших (правило registry/rating.py) среди мигрантов и прочих кандидатов,
в целом и внутри групп по объёму в окне среза (чтобы не спутать с эффектом размера).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import YEARS, docs, files
from .text import candidates

SCIENCES = (   # первое совпадение по группе документа
    ("Биология и медицина", r"^ax:(q-bio|physics\.(bio|med)-ph)"),
    ("Финансы и экономика", r"^(ax:(q-fin|econ)|oa:(1404|2003))"),
    ("Квантовая физика", r"^ax:quant-ph"),
    ("Материалы и конденсированное состояние", r"^ax:cond-mat"),
    ("Оптика и прикладная физика", r"^(ax:physics\.(optics|app-ph|ins-det|acc-ph)|oa:3107)"),
    ("Другая физика", r"^ax:(physics|hep|astro|nucl|gr-qc|atom-ph|nlin|chao-dyn|math-ph)"),
    ("Электротехника и связь", r"^(ax:eess|oa:2208)"),
    ("Инженерия и управление", r"^oa:(2207|2209)"),
    ("Статистика и математика", r"^ax:(stat\.(ME|AP|CO|OT|TH)|math)"),
    ("ИИ и информатика", r"."),
)
HIST_MIN, TARGET_HIST_MAX, TARGET_MIN, TARGET_SHARE = 5, 1, 5, 0.2
WORKERS = 7

_MAPS: list[dict[str, int]] = []
_SCI_CACHE: dict[str, int] = {}


def science(group: str) -> int:
    s = _SCI_CACHE.get(group)
    if s is None:
        s = next(i for i, (_, rx) in enumerate(SCIENCES) if re.search(rx, group))
        _SCI_CACHE[group] = s
    return s


def _init(maps):
    global _MAPS
    _MAPS = maps


def _scan_file(args):
    source, path = args
    yi = {y: i for i, y in enumerate(YEARS)}
    out = [collections.Counter() for _ in _MAPS]
    for d in docs(source, path):
        y = yi.get(d["year"])
        if y is None:
            continue
        s = science(d.get("group") or "")
        ph = candidates(d["text"])
        for j, mp in enumerate(_MAPS):
            for r in {mp[p] for p in ph if p in mp}:
                out[j][(r, s, y)] += 1
    return out


def _cand(reg: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Кандидаты реестра (строка = сущность) и отображение фраза словаря -> номер строки."""
    r = pd.read_parquet(reg / "registry.parquet", columns=["key", "phrase", "eid", "candidate"])
    r = r[r.candidate].drop_duplicates("eid").reset_index(drop=True)
    row = {int(e): i for i, e in enumerate(r.eid)}
    phrases = [l for l in (reg / "vocab.txt").read_text(encoding="utf-8").split("\n") if l]
    p2e = np.load(reg / "phrase2entity.npy")
    mp = {p: row[int(e)] for p, e in zip(phrases, p2e) if int(e) in row}
    return r, mp


def scan(specs: list[tuple[Path, int]]) -> None:
    regs = [_cand(reg) for reg, _ in specs]
    maps = [mp for _, mp in regs]
    arrs = [np.zeros((len(r), len(SCIENCES), len(YEARS)), np.int32) for r, _ in regs]
    todo = [(s, p) for s, p in files() if s in ("openalex", "arxiv", "hf")]
    print(f"файлов {len(todo)}; кандидатов: " + ", ".join(str(len(r)) for r, _ in regs), flush=True)
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(maps,)) as ex:
        for n, outs in enumerate(ex.map(_scan_file, todo, chunksize=2), 1):
            for a, cnt in zip(arrs, outs):
                for (r, s, y), v in cnt.items():
                    a[r, s, y] += v
            if n % 200 == 0:
                print(f"  файлов {n}/{len(todo)}", flush=True)
    for (reg, _), (r, _), a in zip(specs, regs, arrs):
        np.savez_compressed(reg / "sci_series.npz", series=a, keys=r.key.values.astype(str), eid=r.eid.values,
                            sciences=np.array([s for s, _ in SCIENCES]), years=np.array(YEARS))
        print(f"{reg}: {a.sum():,} упоминаний -> sci_series.npz".replace(",", " "), flush=True)


def migrants(reg: Path, cutoff: int) -> pd.DataFrame:
    z = np.load(reg / "sci_series.npz", allow_pickle=True)
    a, years = z["series"].astype(float), [int(y) for y in z["years"]]
    hist = [i for i, y in enumerate(years) if y <= cutoff - 3]
    rec = [i for i, y in enumerate(years) if cutoff - 2 <= y <= cutoff]
    H, R = a[:, :, hist].sum(2), a[:, :, rec].sum(2)
    home = H.argmax(1)
    ok_hist = H.sum(1) >= HIST_MIN
    new_t = (H <= TARGET_HIST_MAX) & (R >= TARGET_MIN) & (R >= TARGET_SHARE * R.sum(1, keepdims=True).clip(min=1))
    new_t[np.arange(len(a)), home] = False
    score = np.where(new_t, R, 0)
    target = score.argmax(1)
    mig = ok_hist & new_t.any(1)
    sci = [str(s) for s in z["sciences"]]
    return pd.DataFrame({"key": z["keys"], "eid": z["eid"], "migrant": mig,
                         "home": [sci[i] for i in home], "target": [sci[i] if m else "" for i, m in zip(target, mig)],
                         "target_recent": np.where(mig, score.max(1), 0), "recent_total": R.sum(1), "hist_total": H.sum(1)})


def check(specs: list[tuple[Path, int]]) -> dict:
    from .rating import event_time
    res = {}
    for reg, cut in specs:
        m = migrants(reg, cut)
        T = event_time(reg, m.eid.astype(int).values, cut, 2026)
        m["took_off"] = T > 0
        m = m[m.hist_total >= HIST_MIN]                       # сравнение только среди терминов с историей
        m["vol_bin"] = pd.qcut(np.log1p(m.recent_total), 5, labels=False, duplicates="drop")
        by = m.groupby(["vol_bin", "migrant"]).took_off.agg(["mean", "size"]).unstack()
        out = {"cutoff": cut, "n": len(m), "migrants": int(m.migrant.sum()),
               "rate_migrant": round(float(m[m.migrant].took_off.mean()), 4),
               "rate_other": round(float(m[~m.migrant].took_off.mean()), 4),
               "by_volume": {int(b): {"migrant": [round(float(by.loc[b, ("mean", True)]), 4) if ("mean", True) in by.columns else None,
                                                  int(by.loc[b, ("size", True)]) if ("size", True) in by.columns else 0],
                                      "other": [round(float(by.loc[b, ("mean", False)]), 4), int(by.loc[b, ("size", False)])]}
                             for b in by.index},
               "examples": m[m.migrant & m.took_off].sort_values("target_recent", ascending=False)
                           .head(20)[["key", "home", "target", "target_recent"]].to_dict("records")}
        res[str(cut)] = out
        print(json.dumps(out, ensure_ascii=False, indent=1, default=str), flush=True)
    return res


def now(reg: Path, cut: int, bundle: str) -> None:
    m = migrants(reg, cut)
    mm = m[m.migrant]
    flows = mm.groupby(["home", "target"]).size().sort_values(ascending=False)
    print(f"срез {cut}: мигрантов {len(mm)} из {int((m.hist_total >= HIST_MIN).sum())} кандидатов с историей")
    print(flows.head(25).to_string())
    m.to_parquet(reg / "migrants.parquet", index=False)
    (reg / "migrant_flows.json").write_text(json.dumps(
        [{"from": h, "to": t, "n": int(n)} for (h, t), n in flows.items()], ensure_ascii=False, indent=1), encoding="utf-8")
    if bundle:
        b = Path(bundle)
        with gzip.open(b / "cards.json.gz", "rt", encoding="utf-8") as fh:
            cards = json.load(fh)
        idx = mm.set_index("key")
        n = 0
        for k, c in cards.items():
            if k in idx.index:
                x = idx.loc[k]
                c["migrant"] = {"from": x.home, "to": x.target, "recent_in_target": int(x.target_recent)}
                fact = f"Мигрант: термин из области «{x.home}» за три года пришёл в «{x.target}» ({int(x.target_recent)} работ)"
                c["facts"] = [f for f in (c.get("facts") or []) if not f.startswith("Мигрант:")] + [fact]
                n += 1
        with gzip.open(b / "cards.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(cards, fh, ensure_ascii=False)
        ax = pd.read_parquet(b / "axes.parquet")
        ax = ax.drop(columns=[c for c in ("migrant", "migrant_from", "migrant_to") if c in ax.columns])
        ax = ax.merge(m[["key", "migrant", "home", "target"]].rename(columns={"home": "migrant_from", "target": "migrant_to"}),
                      on="key", how="left")
        ax.to_parquet(b / "axes.parquet")
        print(f"набор {b}: мигрантов среди карточек {n} из {len(cards)}")


def _specs(xs: list[str]) -> list[tuple[Path, int]]:
    return [(Path(x.rsplit(":", 1)[0]), int(x.rsplit(":", 1)[1])) for x in xs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "check", "now"])
    ap.add_argument("regs", nargs="+")
    ap.add_argument("--bundle", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    specs = _specs(a.regs)
    if a.cmd == "scan":
        scan(specs)
    elif a.cmd == "check":
        res = check(specs)
        if a.out:
            Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    else:
        now(specs[0][0], specs[0][1], a.bundle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
