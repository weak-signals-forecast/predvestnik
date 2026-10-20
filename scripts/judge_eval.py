"""Качество ТОП-10 на узких запросах: оценка независимой моделью-судьёй (вслепую, без знания конфигурации).

    python scripts/judge_eval.py run --queries q.jsonl --url http://localhost:8012 --label v5d --out runs/v5d.jsonl
    python scripts/judge_eval.py prompts --runs runs/*.jsonl --out prompts_j0 --map judge_map.json
    python scripts/judge_eval.py score --answers answers_j0.jsonl --map judge_map.json [--out judge.json]

run: запросы через API сервиса (без языковой модели в сервисе), сохраняется ТОП-10: термин, название, описание,
два заголовка источников. prompts: на каждую пару «запрос, конфигурация» — один промпт со списком из 10 технологий в
случайном порядке; конфигурация в промпт не попадает. Судья (for_cesga/server/llm_local.py, Qwen2.5-14B-Instruct —
крупнее и независимее модели, писавшей названия) ставит каждой вердикт:
  S — конкретная технология по теме запроса на ранней стадии (слабый сигнал);
  I — по теме, но улучшение известной технологии;   M — по теме, но зрелая массовая технология;
  X — не по теме запроса или не технология.
score: precision@10 = доля S, on_topic@10 = доля S+I+M, по конфигурациям и по типу запроса.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

PROMPT = """Ты эксперт по технологическому скаутингу. Пользователь ищет ранние технологические сигналы по запросу:
«{query}»

Ниже 10 технологий из выдачи поиска (английский термин, русское название, описание, заголовки работ-источников).
Для КАЖДОЙ поставь ровно один вердикт:
  S — конкретная технология по теме запроса, новая, на ранней стадии (слабый сигнал);
  I — по теме запроса, но это улучшение или вариант уже известной технологии;
  M — по теме запроса, но это зрелая, массовая технология или общее направление;
  X — не по теме запроса, или это не технология (обрывок фразы, общие слова, название продукта).
Оценивай строго по смыслу запроса. Верни СТРОГО JSON-массив: [{{"i": <номер>, "verdict": "S|I|M|X"}}, ...]

{items}"""


def run(a) -> None:
    out = open(a.out, "w", encoding="utf-8")
    t0 = time.time()
    qs = [json.loads(l) for l in open(a.queries, encoding="utf-8") if l.strip()]
    for n, q in enumerate(qs, 1):
        d = requests.post(f"{a.url}/api/registry/search", json={"query": q["query"], "use_llm": False}, timeout=600).json()
        top = []
        for s in d.get("signals", [])[:10]:
            src = [x.get("title") or "" for x in (s.get("sources") or [])][:2]
            top.append({"term": s.get("term_en"), "name": s.get("name"), "desc": s.get("description") or "", "src": src})
        out.write(json.dumps({"label": a.label, "query": q["query"], "kind": q.get("kind", ""), "top": top},
                             ensure_ascii=False) + "\n")
        if n % 25 == 0:
            print(f"  {a.label}: {n}/{len(qs)} {(time.time() - t0) / n:.1f} с/запрос", flush=True)
    print(f"{a.label}: {len(qs)} запросов -> {a.out}")


def prompts(a) -> None:
    rnd = random.Random(0)
    rows, mp = [], {}
    files = sorted(f for g in a.runs for f in glob.glob(g))
    for f in files:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if not r["top"]:
                continue
            order = list(range(len(r["top"])))
            rnd.shuffle(order)
            items = "\n".join(
                f"{j}. {r['top'][k]['term']} — {r['top'][k]['name']}: {r['top'][k]['desc'][:200]}"
                + "".join(f"\n   · {t[:160]}" for t in r["top"][k]["src"]) for j, k in enumerate(order, 1))
            pid = f"j:{len(rows)}"
            rows.append({"id": pid, "prompt": PROMPT.replace("{query}", r["query"]).replace("{items}", items), "max_new": 400})
            mp[pid] = {"label": r["label"], "query": r["query"], "kind": r["kind"], "order": order, "n": len(r["top"])}
    rnd.shuffle(rows)                                       # конфигурации вперемешку и в файле
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(a.map).write_text(json.dumps(mp, ensure_ascii=False), encoding="utf-8")
    print(f"промптов {len(rows)} из {len(files)} прогонов -> {a.out}")


def score(a) -> None:
    mp = json.loads(Path(a.map).read_text(encoding="utf-8"))
    agg = defaultdict(lambda: defaultdict(int))
    per_query = []
    bad = 0
    for l in open(a.answers, encoding="utf-8"):
        x = json.loads(l)
        m = mp.get(x["id"])
        if not m:
            continue
        mt = re.search(r"\[.*\]", x["raw"], re.S)
        try:
            arr = json.loads(mt.group(0)) if mt else []
        except json.JSONDecodeError:
            arr = []
        v = {int(e["i"]): str(e.get("verdict", "")).strip().upper()[:1] for e in arr
             if isinstance(e, dict) and isinstance(e.get("i"), int) and 1 <= e["i"] <= m["n"]}
        if len(v) < m["n"]:
            bad += 1
            continue
        verdicts = [v[j] for j in range(1, m["n"] + 1)]
        for key in (m["label"], f"{m['label']}|{m['kind']}"):
            agg[key]["lists"] += 1
            agg[key]["items"] += len(verdicts)
            for c in "SIMX":
                agg[key][c] += verdicts.count(c)
        per_query.append({"label": m["label"], "query": m["query"], "kind": m["kind"],
                          "verdicts": [verdicts[m["order"].index(k)] for k in range(m["n"])]})
    res = {"unparsed_lists": bad, "configs": {}}
    for key, c in sorted(agg.items()):
        n = max(c["items"], 1)
        res["configs"][key] = {"lists": c["lists"], "precision@10": round(c["S"] / n, 3),
                               "on_topic@10": round((c["S"] + c["I"] + c["M"]) / n, 3),
                               "I": round(c["I"] / n, 3), "M": round(c["M"] / n, 3), "X": round(c["X"] / n, 3)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps({**res, "per_query": per_query}, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "prompts", "score"])
    ap.add_argument("--queries"); ap.add_argument("--url", default="http://localhost:8012")
    ap.add_argument("--label"); ap.add_argument("--out", default="")
    ap.add_argument("--runs", nargs="+"); ap.add_argument("--map"); ap.add_argument("--answers")
    a = ap.parse_args()
    {"run": run, "prompts": prompts, "score": score}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
