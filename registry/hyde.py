"""HyDE для поиска по базе: YandexGPT пишет гипотетические описания ранних технологий под запрос, поиск идёт по ним.

    python -m registry.hyde               # сгенерировать описания для 18 оценочных запросов (1 вызов на запрос, кэш)
    python -m registry.hyde --eval        # сравнить поиск с HyDE и без: сколько известных сигналов в 40 кандидатах

Короткий общий запрос («новое железо для нейросетей») плохо совпадает с конкретными терминами базы. Модель пишет
5 коротких описаний конкретных ранних технологий по теме запроса — «как выглядел бы идеальный ответ»; их эмбеддинги
добавляются к эмбеддингу темы. Модель влияет только на поиск: всё, что попадает в выдачу, — записи базы с данными
корпуса; сами описания нигде не показываются.
"""
from __future__ import annotations

import json
import os
import re
import sys

import requests

from .corpus import REGISTRY as REG

CACHE = REG / "hyde.json"
PROMPT = """Запрос аналитика банка: «{query}».
Назови 5 РАЗНЫХ конкретных технологий ранней стадии (появились в 2023–2026, ещё не массовые), которые отвечают на
этот запрос. Для каждой — одна строка: русское название, английский термин в скобках, одно предложение о сути.
Без вступления и нумерации, только 5 строк."""


def generate(queries: list[str]) -> dict[str, list[str]]:
    from wsignals.decompose import RUNTIME_YANDEX_MODEL
    from .llm_assess import load_env
    load_env()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    uri = f"gpt://{os.environ['YANDEX_FOLDER_ID']}/{RUNTIME_YANDEX_MODEL}"
    calls = tokens = 0
    for q in queries:
        if q in cache:
            continue
        for attempt in range(2):
            calls += 1
            try:
                r = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion", timeout=90,
                                  headers={"Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}"},
                                  json={"modelUri": uri, "completionOptions": {"temperature": 0.3, "maxTokens": 800},
                                        "messages": [{"role": "user", "text": PROMPT.replace("{query}", q)}]})
                r.raise_for_status()
                res = r.json()["result"]
                tokens += int((res.get("usage") or {}).get("totalTokens", 0))
                lines = [re.sub(r"^[\s\-–—•\d.)]+", "", l).strip()
                         for l in res["alternatives"][0]["message"]["text"].splitlines()]
                lines = [l for l in lines if len(l) > 15][:5]
                if lines:
                    cache[q] = lines
                    break
            except Exception as e:
                print("  повтор:", str(e)[:100], flush=True)
        print(f"  {q[:60]}: {len(cache.get(q, []))} описаний", flush=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"вызовов {calls}, токенов {tokens} -> {CACHE}")
    return cache


def evaluate() -> None:
    """Известные сигналы в 40 кандидатах: экспертная разметка (y=1) и сигналы организаторов нужной области."""
    import pandas as pd
    from .search import Searcher, NEW_QUERIES, FRESH_QUERIES
    from .semantic import LABELS
    from .stage2 import AREAS, area_of_signals
    from .text import entity_key
    pos = set(pd.read_csv(REG / "expert_labels.csv").query("y == 1").key)
    lab = pd.read_csv(LABELS)
    lab = lab[lab.same >= 0.5]
    sa = area_of_signals()
    org_area: dict[str, set] = {}
    for _, x in lab.iterrows():
        org_area.setdefault(entity_key(x.candidate), set()).add(sa.get(int(x.id)))
    qs = [q for q, _ in AREAS.values()] + NEW_QUERIES + FRESH_QUERIES
    s = Searcher(base=True)
    hy = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    tot = {False: [0, 0], True: [0, 0]}
    for q in qs:
        row = []
        for use in (False, True):
            s.hyde = hy.get(q) if use else None
            out, area = s.search(q)
            a = sum(k in pos for k in out.key)
            b = sum(area in org_area.get(k, ()) for k in out.key) if area else 0
            tot[use][0] += a
            tot[use][1] += b
            row.append(f"{a:>2}/{b:<2}")
        print(f"{q[:55]:55} без HyDE {row[0]}   с HyDE {row[1]}   (размеченные сигналы / сигналы организаторов)")
    print(f"ИТОГО без HyDE {tot[False][0]}/{tot[False][1]}, с HyDE {tot[True][0]}/{tot[True][1]}")


def main() -> int:
    if "--eval" in sys.argv:
        evaluate()
        return 0
    from .search import FRESH_QUERIES, NEW_QUERIES
    from .stage2 import AREAS
    generate([q for q, _ in AREAS.values()] + NEW_QUERIES + FRESH_QUERIES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
