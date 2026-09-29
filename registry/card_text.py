"""Преимущество и кейс для карточек выдачи — YandexGPT Pro по заголовкам источников карточки (поля ТЗ).

    python -m registry.card_text          # по web/data/demo.json -> card_text.jsonl и те же поля в demo.json
    python -m registry.card_text --all    # по всей базе (artifacts/registry/cards.json.gz) — для ответов контейнера

Модель видит только название, описание и заголовки источников карточки (научные работы, истории Hacker News,
стартапы YC, пресса — всё из корпуса) и пишет:
  advantage — чем технология лучше существующих решений, до 25 слов;
  case      — где и как её применяют или пробуют, до 30 слов, только то, что видно из заголовков.
Запрещено добавлять компании, цифры и события, которых нет в заголовках. Ответ принимается только с дословным
повтором фразы. Кэш по фразе: карточка одной технологии в разных запросах получает один и тот же текст.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

from .adjudicate import _echo_ok, _norm
from .corpus import REGISTRY as REG
from .llm_assess import load_env

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "web" / "data" / "demo.json"
BUNDLE_CARDS = ROOT / "artifacts" / "registry" / "cards.json.gz"
CACHE = REG / "card_text.jsonl"
MODEL = "yandexgpt/rc"
BATCH = 10
TITLES = 6

PROMPT = """Ты аналитик технологического радара банка. Ниже карточки технологий: название, описание и заголовки
источников (научные работы, обсуждения Hacker News, стартапы Y Combinator, пресса). Для КАЖДОЙ карточки напиши
по-русски:
advantage — чем эта технология лучше существующих решений (до 25 слов);
case — где и как её уже применяют или пробуют (до 30 слов).
Опирайся ТОЛЬКО на описание и заголовки источников карточки. Не добавляй компании, продукты, цифры и события,
которых там нет. Если из заголовков не видно применения — напиши, в какой области его исследуют.
Верни СТРОГО JSON-массив, по объекту на номер:
[{"i": <номер>, "phrase": "<английская фраза дословно>", "advantage": "<...>", "case": "<...>"}]

Карточки:
{items}"""


def ask(items: list[dict], uri: str, usage: dict) -> dict[str, dict]:
    listing = "\n\n".join(
        f"{i}. {c['phrase']} — {c['name']}: {c['description']}\n   Источники:\n" +
        "\n".join(f"   - {t}" for t in c["titles"]) for i, c in enumerate(items, 1))
    for attempt in range(2):
        usage["calls"] += 1
        try:
            r = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion", timeout=120,
                              headers={"Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}"},
                              json={"modelUri": uri, "completionOptions": {"temperature": 0.2, "maxTokens": 4000},
                                    "messages": [{"role": "user", "text": PROMPT.replace("{items}", listing)}]})
            r.raise_for_status()
            res = r.json()["result"]
            usage["tokens"] += int((res.get("usage") or {}).get("totalTokens", 0))
            raw = res["alternatives"][0]["message"]["text"]
            m = re.search(r"\[.*\]", raw, re.S)
            try:
                rows = json.loads(m.group(0)) if m else []
            except json.JSONDecodeError:
                rows = []
            out = {}
            for x in rows:
                i = x.get("i") if isinstance(x, dict) else None
                if not isinstance(i, int) or isinstance(i, bool) or not 1 <= i <= len(items):
                    continue
                c = items[i - 1]
                if not _echo_ok(x.get("phrase"), c["phrase"]):
                    continue
                adv, case = str(x.get("advantage") or "").strip(), str(x.get("case") or "").strip()
                if adv and case:
                    out[c["phrase"]] = {"advantage": adv[:300], "case": case[:400]}
            return out
        except Exception as e:
            print("  повтор:", str(e)[:100], flush=True)
            time.sleep(5 * (attempt + 1))
    return {}


def main() -> int:
    load_env()
    uri = f"gpt://{os.environ['YANDEX_FOLDER_ID']}/{MODEL}"
    all_base = "--all" in sys.argv                  # по всей базе: карточки набора данных контейнера
    if all_base:
        import gzip
        with gzip.open(BUNDLE_CARDS, "rt", encoding="utf-8") as fh:
            bundle = json.load(fh)
        data = {"results": {"__bundle__": {"cards": list(bundle.values())}}}
    else:
        data = json.loads(DEMO.read_text(encoding="utf-8"))
    cache = {}
    if CACHE.exists():
        for l in CACHE.read_text(encoding="utf-8").splitlines():
            x = json.loads(l)
            cache[x["phrase"]] = x
    cards = {}
    for res in data["results"].values():
        for c in res["cards"]:
            ph = c["name"].lower()
            if ph not in cards:
                cards[ph] = {"phrase": ph, "name": c.get("name_ru") or c["name"],
                             "description": ((c.get("description") or {}).get("text") or "")[:300],
                             "titles": [s["title"][:160] for s in (c.get("sources") or [])[:TITLES] if s.get("title")]}
    todo = [c for ph, c in cards.items() if ph not in cache]
    print(f"карточек {len(cards)}, без текста {len(todo)} -> вызовов ~{-(-len(todo) // BATCH)}", flush=True)
    usage = {"calls": 0, "tokens": 0}
    for n in range(0, len(todo), BATCH):
        got = ask(todo[n:n + BATCH], uri, usage)
        with open(CACHE, "a", encoding="utf-8") as fh:
            for ph, v in got.items():
                cache[ph] = {"phrase": ph, **v}
                fh.write(json.dumps(cache[ph], ensure_ascii=False) + "\n")
        print(f"  пачка {n // BATCH + 1}: принято {len(got)}/{len(todo[n:n + BATCH])}", flush=True)
    filled = 0
    for res in data["results"].values():
        for c in res["cards"]:
            v = cache.get(c["name"].lower())
            if v:
                c["advantage"] = {"text": v["advantage"], "generated": True}
                c["case"] = {"text": v["case"], "generated": True}
                filled += 1
    if all_base:
        import gzip
        with gzip.open(BUNDLE_CARDS, "wt", encoding="utf-8") as fh:
            json.dump({c["key"]: c for c in data["results"]["__bundle__"]["cards"]}, fh, ensure_ascii=False)
        import shutil
        shutil.copy(CACHE, BUNDLE_CARDS.parent / "cache" / "card_text.jsonl")
    else:
        DEMO.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"вызовов {usage['calls']}, токенов {usage['tokens']:,}; заполнено карточек {filled} -> {DEMO}".replace(",", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
