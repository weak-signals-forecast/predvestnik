"""Точечная смысловая проверка пула кандидатов разрешённой моделью (YandexGPT) — один раз, офлайн, с кэшем.

    python -m registry.llm_assess --pool 1000 --max-calls 100

Схема: пул кандидатов формируется ДАННЫМИ, без модели (реестр → фильтры общих оборотов → обученный балл
сигнальности → верхние N). Модель спрашивается только по этому пулу и только один раз: результат кэшируется
по фразе. На запросе пользователя модель не вызывается вовсе — запрос сравнивается эмбеддингами с русскими
названиями и описаниями из пула.

Что возвращает модель по каждой технологии (строгий JSON, ответ не по схеме отбрасывается):
  is_technology, is_product_or_brand, is_generic_phrase — отсев не-технологий;
  name_ru, description_ru — русское название и одно предложение описания ПО ЗАГОЛОВКАМ работ из корпуса,
  которые подаются вместе с фразой (модель не приносит фактов от себя);
  areas — какие из шести областей ТЗ затрагивает.
Модель НЕ оценивает, слабый ли это сигнал: это делает обученная модель по данным корпуса.

Каждое обращение к API, включая повторы после обрывов, считается в --max-calls; расход токенов берётся из
ответа API (поле usage) и печатается в конце. Ключ — из окружения или .env проекта, в вывод не попадает.
Результат: <registry>/llm_assess.jsonl (кэш), llm_usage.json (вызовы и токены).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .corpus import REGISTRY, docs, files
from .text import candidates

REG = REGISTRY
ROOT = Path(__file__).resolve().parents[1]
CACHE = REG / "llm_assess.jsonl"
USAGE = REG / "llm_usage.json"
BATCH = 10
WORKERS = 3
TIMEOUT = 60.0
TITLES = 3
AREAS = ["Edge", "Защита ИИ", "Индустриальный ИИ", "Инфраструктура ИИ", "Роботы", "Финтех", "другое"]

PROMPT = """Ты технологический аналитик. Ниже — фразы, найденные в научных работах, и заголовки работ, где они
встречаются. По КАЖДОЙ фразе ответь только на смысловые вопросы; опирайся на заголовки, ничего не придумывай.
Ты НЕ оцениваешь, перспективна ли технология.

Верни СТРОГО JSON-массив, по одному объекту на каждый номер:
[{"i": <номер>, "phrase": "<фраза под этим номером, дословно>", "is_technology": true|false, "is_product_or_brand": true|false, "is_generic_phrase": true|false,
  "name_ru": "<русское название технологии, 2–6 слов>", "description_ru": "<одно предложение: что это, по заголовкам>",
  "areas": [<из списка: "Edge", "Защита ИИ", "Индустриальный ИИ", "Инфраструктура ИИ", "Роботы", "Финтех", "другое">],
  "reject_reason": "<кратко по-русски, если это не конкретная технология, иначе null>"}]

areas — только если технология прямо относится к области; общие методы ИИ и машинного обучения (архитектуры
моделей, способы обучения, рекомендательные системы) и всё вне шести областей — ["другое"]. Области:
Edge — ИИ на устройствах и периферии (смартфоны, IoT, микроконтроллеры, NPU);
Защита ИИ — атаки на ИИ-системы и защита от них, безопасность агентов и моделей;
Индустриальный ИИ — ИИ в промышленности, производстве, энергетике, управлении оборудованием;
Инфраструктура ИИ — чипы, память, сети, дата-центры, системы обучения и инференса;
Роботы — роботы и управление ими; Финтех — платежи, банки, криптоактивы, финансовые рынки и регулирование.

is_technology = true только для конкретного метода, устройства, архитектуры или протокола. Общая рубрика
(«ai security», «financial analytics»), обрывок фразы, название продукта, компании, бенчмарка, датасета или
события — false.

Фразы:
{items}"""

_lock = threading.Lock()
_calls = 0
_tokens = {"input": 0, "completion": 0, "total": 0}


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def load_cache() -> dict[str, dict]:
    out = {}
    if CACHE.exists():
        for l in CACHE.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                out[r["phrase"]] = r
    return out


# ---------- заголовки работ для каждой технологии пула (из корпуса, без модели) ----------

_P2E: dict[str, int] = {}


def _init(want: list[int]) -> None:
    global _P2E
    ws = set(want)
    phrases = [l for l in (REG / "vocab.txt").read_text(encoding="utf-8").split("\n") if l]
    p2e = np.load(REG / "phrase2entity.npy")
    _P2E = {ph: int(e) for ph, e in zip(phrases, p2e) if int(e) in ws}


def _titles_file(args):
    source, path = args
    got = defaultdict(list)
    for d in docs(source, path):
        if d["year"] < 2023:
            continue
        for e in {e for ph in candidates(d["title"]) if (e := _P2E.get(ph)) is not None}:
            if len(got[e]) < TITLES:
                got[e].append(d["title"].strip()[:160])
    return dict(got)


def pool_titles(eids: list[int]) -> dict[int, list[str]]:
    todo = [(s, p) for s, p in files() if s == "arxiv" or (s == "openalex" and re.search(r"year=202[3-6]", p))]
    out = defaultdict(list)
    with ProcessPoolExecutor(6, initializer=_init, initargs=(eids,)) as ex:
        for got in ex.map(_titles_file, todo, chunksize=8):
            for e, ts in got.items():
                if len(out[e]) < TITLES:
                    out[e] += ts[:TITLES - len(out[e])]
    return out


# ---------- вызовы модели ----------

def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _validate(rows, n: int, phrases: list[str] | None = None) -> dict[int, dict]:
    """Ответ принимается, только если модель дословно повторила фразу своего номера: сдвиг нумерации
    (первая версия принимала ответ по номеру, модель нумеровала с 1 — названия уехали на соседние фразы)
    так становится невозможен."""
    ok = {}
    counts = defaultdict(int)
    for r in rows if isinstance(rows, list) else []:
        if isinstance(r, dict) and isinstance(r.get("i"), int) and not isinstance(r.get("i"), bool):
            counts[r["i"]] += 1
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict) or not isinstance(r.get("i"), int) or isinstance(r.get("i"), bool):
            continue
        i = r["i"] - 1                                                # в списке нумерация с 1
        if not 0 <= i < n or counts[r["i"]] != 1:
            continue                                                  # чужой или повторный номер — не верим
        if phrases is not None and _norm(r.get("phrase")) != _norm(phrases[i]):
            continue                                                  # фраза не совпала — ответ не про неё
        flags = [r.get("is_technology"), r.get("is_product_or_brand"), r.get("is_generic_phrase")]
        if not all(isinstance(f, bool) for f in flags):
            continue
        name, desc = r.get("name_ru"), r.get("description_ru")
        if not isinstance(name, str) or not isinstance(desc, str) or not name.strip():
            continue
        areas = [a for a in (r.get("areas") or []) if a in AREAS]
        reason = r.get("reject_reason") if isinstance(r.get("reject_reason"), str) else None
        keep = flags[0] and not flags[1] and not flags[2]
        ok[i] = {"is_technology": flags[0], "is_product_or_brand": flags[1], "is_generic_phrase": flags[2],
                 "name_ru": name.strip()[:80], "description_ru": desc.strip()[:300], "areas": areas,
                 "reject_reason": None if keep else (reason or "не конкретная технология"), "keep": keep}
    return ok


def _call(items: list[tuple[str, list[str]]], model_uri: str, max_calls: int) -> dict[int, dict] | None:
    """Один вызов API. None — вызов не состоялся (лимит вызовов или сетевая ошибка)."""
    global _calls
    with _lock:
        if _calls >= max_calls:
            return None
        _calls += 1
    listing = "\n".join(f"{i}. {ph}" + "".join(f"\n   — {t}" for t in ts) for i, (ph, ts) in enumerate(items, 1))
    try:
        r = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion", timeout=TIMEOUT,
                          headers={"Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}"},
                          json={"modelUri": model_uri,
                                "completionOptions": {"temperature": 0.1, "maxTokens": 2500,
                                                      "reasoningOptions": {"mode": "DISABLED"}},
                                "messages": [{"role": "user", "text": PROMPT.replace("{items}", listing)}]})
        r.raise_for_status()
        res = r.json()["result"]
        u = res.get("usage") or {}
        with _lock:
            _tokens["input"] += int(u.get("inputTextTokens", 0))
            _tokens["completion"] += int(u.get("completionTokens", 0))
            _tokens["total"] += int(u.get("totalTokens", 0))
        text = res["alternatives"][0]["message"]["text"]
        m = re.search(r"\[.*\]", text, re.S)
        return _validate(json.loads(m.group(0)) if m else [], len(items), [ph for ph, _ in items])
    except Exception:
        return None


def _batch(items, model_uri, max_calls):
    """Порция с повторами (каждый повтор — отдельный учтённый вызов)."""
    for attempt in range(3):
        got = _call(items, model_uri, max_calls)
        if got is not None:
            return got
        with _lock:
            if _calls >= max_calls:
                return {}
        time.sleep(5 * 2 ** attempt)
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=1000, help="общий топ N кандидатов по обученному баллу")
    ap.add_argument("--per-area", type=int, default=0, help="плюс топ N внутри рубрик каждой области (registry/pool.py)")
    ap.add_argument("--max-calls", type=int, default=100, help="жёсткий предел обращений к API, с повторами")
    ap.add_argument("--workers", type=int, default=WORKERS, help="параллельных вызовов")
    ap.add_argument("--keys-file", default=None, help="проверить технологии из списка ключей (по строке на ключ) "
                                                       "вместо пула — например, промышленные кандидаты")
    a = ap.parse_args()
    load_env()
    from wsignals import decompose
    from .stage2 import GENERIC_START, GENERIC_END
    model_uri, why = decompose.configured_model()
    if not model_uri:
        raise SystemExit(f"YandexGPT не сконфигурирован: {why}")

    from . import pool as pool_mod
    if a.keys_file:                                       # заданный список технологий (registry/industrial.py)
        want = {l.strip() for l in open(a.keys_file, encoding="utf-8") if l.strip()}
        r = pd.read_parquet(REG / "ranked.parquet", columns=["key", "phrase", "candidate"])
        pool = r[r.key.isin(want)].drop_duplicates("key")
    else:
        pool = pool_mod.select(REG, a.pool, a.per_area)  # общий топ + топ внутри рубрик каждой области
    have = load_cache()
    pool = pool[~pool.phrase.str.replace("_", " ").isin(have)]   # в кэше фразы с пробелами вместо «_»
    sci = pd.read_parquet(REG / "all_phrases.parquet", columns=["key", "eid"])
    eid = dict(zip(sci.key, sci.eid))
    t0 = time.time()
    titles = pool_titles([int(eid[k]) for k in pool.key if k in eid])
    print(f"пул: общий топ {a.pool} + по {a.per_area} на область; новых к проверке {len(pool)}; заголовки собраны за {time.time() - t0:.0f} с", flush=True)

    items = [(p.replace("_", " "), titles.get(int(eid[k]), []) if k in eid else []) for p, k in zip(pool.phrase, pool.key)]
    chunks = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    done = kept = 0
    with ThreadPoolExecutor(a.workers) as ex, open(CACHE, "a", encoding="utf-8") as fh:
        for n, (ch, got) in enumerate(zip(chunks, ex.map(lambda ch: _batch(ch, model_uri, a.max_calls), chunks)), 1):
            for i, v in got.items():
                fh.write(json.dumps({"phrase": ch[i][0], **v}, ensure_ascii=False) + "\n")
                done += 1
                kept += v["keep"]
            fh.flush()
            if n % 50 == 0 or n == len(chunks):
                print(f"  порций {n}/{len(chunks)} · вызовов {_calls} · проверено {done}, оставлено {kept} · "
                      f"токенов {_tokens['total']:,} · {time.time() - t0:.0f} с".replace(",", " "), flush=True)
    usage = {"calls": _calls, **_tokens, "checked": done, "kept": kept, "model": model_uri.rsplit("/", 1)[-1],
             "time_s": round(time.time() - t0)}
    prev = json.loads(USAGE.read_text()) if USAGE.exists() else []
    USAGE.write_text(json.dumps(prev + [usage], ensure_ascii=False, indent=1))
    print(f"вызовов API: {_calls} (предел {a.max_calls}); проверено технологий: {done}, оставлено: {kept}, "
          f"отсеяно: {done - kept}")
    print(f"токены: вход {_tokens['input']:,}, ответ {_tokens['completion']:,}, всего {_tokens['total']:,}".replace(",", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
