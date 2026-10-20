"""Замер выдачи на отложенных запросах: прогон через сервис базы сигналов, JSON на запрос и таблица для ручной оценки.

    python scripts/eval_queries.py --queries проверки/злые_запросы.csv --out проверки/до [--llm] [--url http://localhost:8010]
    python scripts/eval_queries.py --compare проверки/до проверки/после1      # что изменилось в ТОП-10

Ручная оценка: в <out>/оценка.csv для каждой строки ТОП-10 эксперт ставит verdict: S — слабый сигнал по теме,
I — улучшение известного, M — зрелое, P — стандарт/протокол, X — не по теме / мусор. Сводка считается по этой колонке:
precision@10 = доля S; доля I, M, P, X отдельно.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests

TOP = 10


def run(queries: Path, out: Path, url: str, llm: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(queries.open(encoding="utf-8")))
    table = []
    for r in rows:
        q = r["query"]
        t = time.time()
        d = requests.post(f"{url}/api/registry/search", json={"query": q, "use_llm": llm}, timeout=600).json()
        took = time.time() - t
        (out / f"{int(r['id']):02d}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        sig = d.get("signals", [])[:TOP]
        print(f"{r['id']:>2}. {q:40s} область={d.get('area')!s:18s} режим={d.get('mode')!s:26s} {took:5.1f} с  "
              f"топ: {', '.join(s['term_en'][:30] for s in sig[:3])}")
        for s in sig:
            table.append({"id": r["id"], "query": q, "rank": s["rank"], "term_en": s["term_en"], "name": s["name"],
                          "score": s.get("score"), "verdict": ""})
        if not sig:
            table.append({"id": r["id"], "query": q, "rank": 0, "term_en": "", "name": "(пусто)", "score": "", "verdict": "X"})
    with (out / "оценка.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "query", "rank", "term_en", "name", "score", "verdict"])
        w.writeheader()
        w.writerows(table)
    print(f"-> {out}/оценка.csv: {len(table)} строк для ручной оценки (verdict: S/I/M/P/X)")


def summary(out: Path) -> dict:
    rows = list(csv.DictReader((out / "оценка.csv").open(encoding="utf-8")))
    scored = [r for r in rows if r["verdict"].strip()]
    if not scored:
        return {}
    n = len(scored)
    byq: dict[str, list] = {}
    for r in scored:
        byq.setdefault(r["query"], []).append(r["verdict"].strip().upper())
    p10 = [sum(v == "S" for v in vs) / max(len(vs), 1) for vs in byq.values()]
    return {"запросов оценено": len(byq), "precision@10 средняя": round(sum(p10) / len(p10), 3),
            **{f"доля {k}": round(sum(r["verdict"].strip().upper() == k for r in scored) / n, 3) for k in "SIMPX"}}


def compare(a: Path, b: Path) -> None:
    for name, p in (("до", a), ("после", b)):
        s = summary(p)
        print(name, json.dumps(s, ensure_ascii=False) if s else "(оценка не заполнена)")
    for fa in sorted(a.glob("*.json")):
        fb = b / fa.name
        if not fb.exists():
            continue
        da, db = json.loads(fa.read_text(encoding="utf-8")), json.loads(fb.read_text(encoding="utf-8"))
        ta = [s["term_en"] for s in da.get("signals", [])[:TOP]]
        tb = [s["term_en"] for s in db.get("signals", [])[:TOP]]
        same = len(set(ta) & set(tb))
        print(f"{da['query'][:38]:40s} общих в ТОП-10: {same:2d}; ушли: {', '.join(x[:22] for x in ta if x not in tb)[:90]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="проверки/злые_запросы.csv")
    ap.add_argument("--out", default="проверки/до")
    ap.add_argument("--url", default="http://localhost:8010")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(Path(a.compare[0]), Path(a.compare[1]))
        return 0
    run(Path(a.queries), Path(a.out), a.url, a.llm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
