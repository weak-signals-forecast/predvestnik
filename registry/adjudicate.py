"""Проверка верхушки выдачи сильной моделью (YandexGPT Pro 5.1, `yandexgpt/rc`): ≤ 5 вызовов на запрос.

    python -m registry.adjudicate            # по search_top40.csv -> adjudicated_top15.csv

По каждому из 40 кандидатов модель отвечает ОТДЕЛЬНО по четырём условиям метрики (конъюнкция условий — это и есть
«верная строка»), глядя не только на название, но и на ИЗМЕРЕНИЯ из корпуса:
  specific — конкретная технология (метод, устройство, архитектура, протокол), а не задача/область, не общая
             рубрика, не продукт одной компании, не обрывок фразы;
  on_topic — относится к теме запроса;
  early    — ранняя стадия: растёт в последние годы и ещё не массовая/устоявшаяся (по ряду работ по годам,
             году первого появления в arXiv, числу компаний, прессе, статье в Википедии);
  dup_of   — номер более раннего пункта, если это та же технология другими словами.
Ответ принимается, только если модель дословно повторила фразу. Кэш — по (запрос, фраза) и версии промпта.
Версия 2: в промпт добавлены размеченные примеры (верные строки и отказы с причиной) — ТОЛЬКО из выдачи шести
запросов по областям (stage2_pool_top15.csv, ru_mix). Поэтому точность на запросах по областям после v2 завышена
и не засчитывается; честный замер — новые и свежие запросы. Без примеров (v1) Pro пропускала общие рубрики
(«кластеры ИИ», «управляемый ИИ»): 18/90 и 15/90 против 29/90 без неё.
ТОП-15: сначала прошедшие все четыре условия в порядке поиска, затем — если их меньше 15 — прошедшие все, кроме
«early» (они помечаются). Модель ничего не добавляет от себя: только отсеивает кандидатов из корпуса.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .corpus import REGISTRY, SOURCES, YEARS
from .llm_assess import load_env

REG = REGISTRY
PROMPT_VERSION = 2
CACHE = REG / ("adjudicate_cache.jsonl" if PROMPT_VERSION == 1 else f"adjudicate_cache_v{PROMPT_VERSION}.jsonl")
EXAMPLES = REG / "stage2_pool_top15.csv"
FAIL_RU = {"umbrella": ("specific=false", "общая рубрика, а не технология"),
           "not_tech": ("specific=false", "задача или показатель, а не технология"),
           "fragment": ("specific=false", "обрывок фразы"),
           "product": ("specific=false", "продукт одной компании"),
           "mature": ("early=false", "устоявшийся метод, работ много давно")}
MODEL = "yandexgpt/rc"
BATCH = 10
MAX_CALLS_PER_QUERY = 5

PROMPT = """Ты аналитик технологических слабых сигналов в банке. Запрос пользователя: «{query}».
Ниже кандидаты, найденные в корпусе научных работ, историй Hacker News, стартапов Y Combinator и прессы, и измерения
по каждому. По КАЖДОМУ кандидату реши четыре условия НЕЗАВИСИМО. Опирайся на измерения; ничего не добавляй от себя.

specific: true — конкретная технология: метод, устройство, архитектура, протокол, класс решений с устойчивым
  названием. false — задача или прикладная область («оформление кредита», «манипуляция с длинным горизонтом»),
  общая рубрика («ИИ-инфраструктура», «производственные системы ИИ»), продукт одной компании, обрывок фразы.
on_topic: true — прямо относится к теме запроса.
early: true — ранняя стадия: работ мало, но их число растёт в последние 2–3 года, появилась недавно, компаний и
  прессы немного, статьи в Википедии нет. false — устоявшаяся или массовая технология (много работ давно, есть
  статья в Википедии, много компаний) или угасшая (пик давно прошёл).
dup_of: номер более раннего кандидата из этого списка, если это та же технология другими словами, иначе null.
Будь строгим: слабый сигнал — редкость, обычно проходит меньшая часть кандидатов. Если сомневаешься, что это
конкретная технология, а не широкая тема «ИИ + отрасль/инфраструктура», ставь specific=false.

Размеченные примеры (из других запросов):
{examples}

Верни СТРОГО JSON-массив, по объекту на номер:
[{"i": <номер>, "phrase": "<английская фраза кандидата дословно>", "specific": true|false, "on_topic": true|false,
  "early": true|false, "dup_of": <номер или null>, "reason": "<до 12 слов по-русски>"}]

Кандидаты:
{items}"""


def measurements(reg: Path) -> dict:
    """Ключ сущности -> строка измерений для промпта."""
    sci = pd.read_parquet(reg / "all_phrases.parquet", columns=["key", "eid"])
    eid = dict(zip(sci.key, sci.eid))
    df = np.load(reg / "counts.npz")["df"]
    ax, oa = SOURCES.index("arxiv"), SOURCES.index("openalex")
    biz = pd.read_parquet(reg / "all_business.parquet", columns=["key", "hn_recent", "yc_recent", "press_recent"]).set_index("key")
    wiki = set(pd.read_parquet(reg / "wiki.parquet").key) if (reg / "wiki.parquet").exists() else set()
    yi = {y: i for i, y in enumerate(YEARS)}

    def line(k: str) -> str:
        parts = []
        if k in eid:
            e = int(eid[k])
            series = ", ".join(f"{y}: {int(df[e, ax, yi[y]] + df[e, oa, yi[y]])}" for y in range(2018, 2027))
            first = [y for y in YEARS if df[e, ax, yi[y]] >= 2]
            parts.append(f"работ по годам — {series}")
            parts.append(f"в arXiv впервые: {first[0] if first else 'нет'}")
        if k in biz.index:
            b = biz.loc[k]
            parts.append(f"Hacker News 2024–26: {int(b.hn_recent)}, стартапов YC: {int(b.yc_recent)}, пресса: {int(b.press_recent)}")
        parts.append("статья в Википедии: " + ("есть" if k in wiki else "нет"))
        return "; ".join(parts)
    return {"line": line}


def examples() -> str:
    """Примеры для промпта: по 2 верных строки на область и до 3 отказов на каждую причину (не «вне темы»/«дубль»,
    они зависят от запроса). Только запросы по областям, порядок детерминированный."""
    if PROMPT_VERSION == 1 or not EXAMPLES.exists():
        return ""
    t = pd.read_csv(EXAMPLES)
    t = t[(t.method == "ru_mix") & t.weak_signal.notna()].sort_values(["area", "pos"])
    lines = [f"+ {x.phrase.replace('_', ' ')} — {x.name_ru}: проходит все условия"
             for x in t[t.weak_signal == 1].groupby("area").head(2).itertuples()]
    for reason, (flag, why) in FAIL_RU.items():
        for x in t[(t.weak_signal == 0) & (t.fail_reason == reason)].head(5 if reason == "umbrella" else 3).itertuples():
            lines.append(f"− {x.phrase.replace('_', ' ')} — {x.name_ru}: {flag}, {why}")
    return "\n".join(lines)


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _echo_ok(echo, phrase: str) -> bool:
    """Модель повторила фразу кандидата. Pro часто возвращает всю строку «фраза — название: описание» —
    принимаем, если до первого тире стоит ровно эта фраза."""
    e = str(echo or "")
    head = re.split(r"\s[—–-]\s", e, maxsplit=1)[0]
    return _norm(head) == _norm(phrase) or _norm(e) == _norm(phrase)


def ask(query: str, items: list[dict], model_uri: str, usage: dict) -> dict[str, dict]:
    listing = "\n".join(f"{i}. {c['phrase']} — {c['name_ru']}: {c['description_ru']}\n   {c['meas']}"
                        for i, c in enumerate(items, 1))
    for attempt in range(3):
        if usage["calls_query"] >= MAX_CALLS_PER_QUERY:
            return {}
        usage["calls_query"] += 1
        usage["calls"] += 1
        try:
            r = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion", timeout=90,
                              headers={"Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}"},
                              json={"modelUri": model_uri, "completionOptions": {"temperature": 0.1, "maxTokens": 3000},
                                    "messages": [{"role": "user", "text": PROMPT.replace("{query}", query).replace("{examples}", EX)
                                                                         .replace("{items}", listing)}]})
            r.raise_for_status()
            res = r.json()["result"]
            usage["tokens"] += int((res.get("usage") or {}).get("totalTokens", 0))
            m = re.search(r"\[.*\]", res["alternatives"][0]["message"]["text"], re.S)
            rows = json.loads(m.group(0)) if m else []
            out = {}
            for x in rows:
                i = x.get("i") if isinstance(x, dict) else None
                if not isinstance(i, int) or isinstance(i, bool) or not 1 <= i <= len(items):
                    continue
                c = items[i - 1]
                if not _echo_ok(x.get("phrase"), c["phrase"]):
                    continue                                   # ответ не про эту фразу
                flags = [x.get(f) for f in ("specific", "on_topic", "early")]
                if not all(isinstance(f, bool) for f in flags):
                    continue
                d = x.get("dup_of")
                dup = items[d - 1]["phrase"] if isinstance(d, int) and not isinstance(d, bool) and 1 <= d < i else None
                out[c["phrase"]] = {"specific": flags[0], "on_topic": flags[1], "early": flags[2], "dup_of": dup,
                                    "reason": str(x.get("reason") or "")[:120]}
            return out
        except Exception:
            time.sleep(5 * (attempt + 1))
    return {}


EX = ""


def main() -> int:
    global EX
    EX = examples()
    load_env()
    model_uri = f"gpt://{os.environ['YANDEX_FOLDER_ID']}/{MODEL}"
    cand = pd.read_csv(REG / "search_top40.csv")
    meas = measurements(REG)
    cache = {}
    if CACHE.exists():
        for l in CACHE.read_text(encoding="utf-8").splitlines():
            x = json.loads(l)
            cache[(x["query"], x["phrase"])] = x
    usage = {"calls": 0, "tokens": 0}
    out = []
    for q, g in cand.groupby("query", sort=False):
        g = g.sort_values("rank")
        items = [{"phrase": p.replace("_", " "), "name_ru": n, "description_ru": d, "meas": meas["line"](k)}
                 for p, n, d, k in zip(g.phrase, g.name_ru, g.description_ru, g.key)]
        todo = [it for it in items if (q, it["phrase"]) not in cache]
        usage["calls_query"] = 0
        for i in range(0, len(todo), BATCH):
            got = ask(q, todo[i:i + BATCH], model_uri, usage)
            with open(CACHE, "a", encoding="utf-8") as fh:
                for ph, v in got.items():
                    cache[(q, ph)] = {"query": q, "phrase": ph, **v}
                    fh.write(json.dumps(cache[(q, ph)], ensure_ascii=False) + "\n")
        g = g.assign(ph=g.phrase.str.replace("_", " "))
        v = g.ph.map(lambda p: cache.get((q, p)))
        g["judged"] = v.notna()
        for f in ("specific", "on_topic", "early"):
            g[f] = v.map(lambda x: x.get(f) if x else None)
        g["dup_of"] = v.map(lambda x: x.get("dup_of") if x else None)
        g["reason"] = v.map(lambda x: x.get("reason") if x else "")
        ok = g.judged & g.specific.eq(True) & g.on_topic.eq(True) & g.early.eq(True) & g.dup_of.isna()
        near = g.judged & g.specific.eq(True) & g.on_topic.eq(True) & g.early.eq(False) & g.dup_of.isna()
        top = pd.concat([g[ok], g[near]]).head(15).copy()
        top["tier"] = np.where(top.index.isin(g[ok].index), "все условия", "кроме ранней стадии")
        top["pos"] = range(1, len(top) + 1)
        out.append(top)
        print(f"{q[:60]:60} прошли все условия: {int(ok.sum()):>2} из {int(g.judged.sum())} проверенных; в ТОП-15: {len(top)}")
    res = pd.concat(out)
    prev = REG / "adjudicated_top15.csv"
    if prev.exists():                                          # разметку выдачи не теряем
        p = pd.read_csv(prev)
        if "weak_signal" in p:
            lab = {(a, b): c for a, b, c in zip(p["query"], p["phrase"], p["weak_signal"]) if pd.notna(c)}
            res["weak_signal"] = [lab.get((a, b), np.nan) for a, b in zip(res["query"], res["phrase"])]
    res.to_csv(prev, index=False)
    print(f"вызовов {usage['calls']}, токенов {usage['tokens']:,} -> {prev}".replace(",", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
