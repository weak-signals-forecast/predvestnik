"""Русские запросы для проверки поиска: заголовок работы записи → короткий запрос аналитика на русском (открытая модель).

    REGISTRY_BUNDLE=artifacts/registry_v5d python scripts/make_ru_queries.py prompts --out prompts_q0 --targets targets_q.json
    python scripts/make_ru_queries.py collect --answers answers_q0.jsonl --targets targets_q.json --out queries_ru.jsonl

Берутся те же отложенные заголовки, что в scripts/eval_retrieval.py (источник, не участвовавший в векторе записи).
Модель не видит название записи — только заголовок работы, поэтому запрос не повторяет название дословно.
Промпты считаются на кластере (for_cesga/server/llm_local.py), ответы собираются в jsonl {query, key}.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from registry.mini_bundle import EMB_TITLES  # noqa: E402

BUNDLE = Path(os.getenv("REGISTRY_BUNDLE", "artifacts/registry"))
PROMPT = ("Ниже заголовок научной работы. Напиши, как руководитель направления R&D в банке коротко набрал бы в поиске "
          "запрос о технологии из этой работы: 2–5 слов по-русски, без кавычек, без пояснений, одной строкой.\n\n"
          "Заголовок: {title}\n\nЗапрос:")


def held_out(n: int):
    sig = pd.read_parquet(BUNDLE / "signals.parquet")
    with gzip.open(BUNDLE / "cards.json.gz", "rt", encoding="utf-8") as fh:
        cards = json.load(fh)
    rows = []
    for i, k in enumerate(sig.key):
        ts = [(d.get("title") or "").strip() for d in (cards.get(k) or {}).get("sources") or [] if d.get("title")]
        if len(ts) > EMB_TITLES:
            rows.append((k, ts[EMB_TITLES]))
    rng = np.random.default_rng(0)
    if len(rows) > n:
        rows = [rows[j] for j in sorted(rng.choice(len(rows), n, replace=False))]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prompts", "collect"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--answers", default="")
    a = ap.parse_args()
    if a.cmd == "prompts":
        rows = held_out(a.n)
        with open(a.out, "w", encoding="utf-8") as fh:
            for j, (k, t) in enumerate(rows):
                fh.write(json.dumps({"id": f"q:{j}", "prompt": PROMPT.replace("{title}", t[:300]), "max_new": 32},
                                    ensure_ascii=False) + "\n")
        Path(a.targets).write_text(json.dumps({f"q:{j}": {"key": k, "title": t} for j, (k, t) in enumerate(rows)},
                                              ensure_ascii=False), encoding="utf-8")
        print(f"промптов {len(rows)} -> {a.out}")
    else:
        tg = json.loads(Path(a.targets).read_text(encoding="utf-8"))
        n = 0
        with open(a.out, "w", encoding="utf-8") as fh:
            for l in open(a.answers, encoding="utf-8"):
                x = json.loads(l)
                q = re.split(r"[\n\r]", x["raw"].strip())[0].strip(" «»\"'.:")
                if x["id"] in tg and q and re.search(r"[а-яё]", q, re.I) and len(q.split()) <= 8:
                    fh.write(json.dumps({"query": q, **tg[x["id"]]}, ensure_ascii=False) + "\n")
                    n += 1
        print(f"запросов {n} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
