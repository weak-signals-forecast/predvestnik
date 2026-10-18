"""LLM как измерительный прибор: извлечение проверяемых фактов о сигнале из его источников — один раз, офлайн, с кэшем.

    python -m registry.llm_axes --dry-run 3            # показать промпт и ожидаемый ответ, без вызова модели
    python -m registry.llm_axes --max-calls 120        # YandexGPT Pro по записям базы без ответа в кэше
    python -m registry.llm_axes --stats                 # сколько записей покрыто, сколько абстрактов с цитатой

Зачем. Модель не спрашивают «слабый ли это сигнал» — это решают измерения и обученные модели. Модель просят
извлечь из описания и источников карточки четыре факта, каждый из которых дальше проверяется или измеряется:
  parent       — родительская технология (1–3 слова по-английски): по ней считается зрелость родителя (OpenAlex /
                 таблица фраз), а не по мнению модели;
  mechanism    — что именно нового, одно предложение, с ДОСЛОВНОЙ цитатой из заголовка или описания (цитата
                 проверяется программно; без цитаты абстракт не печатается);
  first_work   — самая ранняя работа из списка источников, где механизм уже есть (название, год);
  kind         — тип: метод / архитектура / устройство / материал / класс решений / стандарт или протокол /
                 задача / рубрика. «стандарт или протокол» и «рубрика» — основания для исключения из выдачи.
Плюс abstract — 3–4 предложения для карточки «почему в выдаче»: что ново, чем отличается от родителя, первая
работа и кто её сделал, цифры из измерений. И why_not_mature — три коротких довода «почему это не зрелая
технология и не хайп», каждый опирается на источник или измерение.

Кэш: artifacts/registry/cache/llm_axes.jsonl (по фразе). Сервис подхватывает его в карточку (поле abstract).
Ключ и папка YandexGPT — из окружения или .env проекта (registry/llm_assess.load_env). Токены считаются из ответа API.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.getenv("REGISTRY_BUNDLE", ROOT / "artifacts" / "registry"))
CACHE = BUNDLE / "cache" / "llm_axes.jsonl"
MODEL = os.getenv("LLM_AXES_MODEL", "yandexgpt/rc")
BATCH = 8
KINDS = ("метод", "архитектура", "устройство", "материал", "класс решений", "стандарт или протокол", "задача", "рубрика")

PROMPT = """Ты аналитик технологического радара. Ниже технологии-кандидаты из базы слабых сигналов: английский термин, русское
название и описание, измерения из корпуса (рост, объём, год появления, слои) и заголовки источников с датами.

Для КАЖДОГО кандидата извлеки факты СТРОГО ПО ПРИВЕДЁННЫМ источникам и измерениям. Ничего не добавляй от себя:
если факта в источниках нет — пустая строка. Поля:
  parent        родительская технология, от которой кандидат произошёл: 1–3 слова по-английски, устоявшийся термин
                («quantization», «language models», «graphene»). Если родителя нет — "".
  mechanism     что именно нового в кандидате относительно родителя: одно предложение по-русски, механизм, а не область.
  quote         ДОСЛОВНЫЙ фрагмент (4–15 слов) одного заголовка источника или описания, подтверждающий mechanism.
  first_work    название самой ранней работы из списка источников, где этот механизм уже есть, и её год: "название — год".
  kind          ровно одно из: {kinds}.
                «стандарт или протокол» — спецификация, интерфейс, консорциум, версия; «рубрика» — область, а не технология.
  abstract      3–4 предложения по-русски для эксперта, почему эта технология в выдаче: (1) что нового, (2) чем отличается
                от родителя, (3) первая работа и кто её сделал, (4) цифры из измерений (рост, объём, слои).
  why_not_mature три коротких довода (до 15 слов каждый), почему это не зрелая технология и не маркетинговый шум;
                каждый довод опирается на измерение или источник из списка.
Верни СТРОГО JSON-массив, по объекту на номер:
[{"i": <номер>, "phrase": "<английский термин дословно>", "parent": "...", "mechanism": "...", "quote": "...",
  "first_work": "...", "kind": "...", "abstract": "...", "why_not_mature": ["...", "...", "..."]}]

Кандидаты:
{items}"""


def _norm(s) -> str:
    return " ".join(re.findall(r"[a-zа-я0-9]+", str(s or "").lower()))


def load_env() -> None:
    try:
        from .llm_assess import load_env as _le
        _le()
    except Exception:  # noqa: BLE001 — без .env ключ может быть уже в окружении
        pass


def base_items() -> list[dict]:
    """Записи базы с описанием, измерениями и источниками (из artifacts/registry)."""
    sig = pd.read_parquet(BUNDLE / "signals.parquet")
    with gzip.open(BUNDLE / "cards.json.gz", "rt", encoding="utf-8") as fh:
        cards = json.load(fh)
    out = []
    for r in sig.itertuples():
        c = cards.get(r.key) or {}
        src = [{"title": (d.get("title") or "")[:180], "date": str(d.get("published") or "")[:10],
                "type": d.get("source_type") or d.get("source_name") or ""} for d in (c.get("sources") or []) if d.get("title")][:6]
        out.append({"key": r.key, "phrase": r.phrase.replace("_", " "), "name_ru": r.name_ru,
                    "description_ru": r.description_ru, "meas": getattr(r, "meas", "") or "", "sources": src})
    return out


def listing(items: list[dict]) -> str:
    return "\n".join(
        f"{j}. {it['phrase']} — {it['name_ru']}: {it['description_ru']}\n   измерения: {it['meas']}\n   источники:\n" +
        "\n".join(f"   - [{d['type']}, {d['date']}] {d['title']}" for d in it["sources"])
        for j, it in enumerate(items, 1))


def prompt_for(items: list[dict]) -> str:
    return PROMPT.replace("{kinds}", " / ".join(KINDS)).replace("{items}", listing(items))


def parse(raw: str, items: list[dict]) -> list[dict]:
    """Ответ модели -> записи кэша. Принимаются только строки с дословным повтором фразы и допустимым типом;
    цитата проверяется по заголовкам источников и описанию — без неё абстракт помечается непроверенным."""
    m = re.search(r"\[.*\]", raw or "", re.S)
    try:
        rows = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    out = []
    for x in rows:
        j = x.get("i") if isinstance(x, dict) else None
        if not isinstance(j, int) or isinstance(j, bool) or not 1 <= j <= len(items):
            continue
        it = items[j - 1]
        if _norm(x.get("phrase")) != _norm(it["phrase"]):
            continue
        kind = str(x.get("kind") or "").strip().lower()
        if kind not in KINDS:
            kind = ""
        q = _norm(x.get("quote"))
        hay = [_norm(d["title"]) for d in it["sources"]] + [_norm(it["description_ru"])]
        quote_ok = bool(q) and len(q.split()) >= 3 and any(q in h for h in hay)
        wnm = x.get("why_not_mature")
        out.append({"key": it["key"], "phrase": it["phrase"], "parent": str(x.get("parent") or "").strip().lower()[:60],
                    "mechanism": str(x.get("mechanism") or "")[:300], "quote": str(x.get("quote") or "")[:200],
                    "quote_ok": quote_ok, "first_work": str(x.get("first_work") or "")[:200], "kind": kind,
                    "abstract": str(x.get("abstract") or "")[:900],
                    "why_not_mature": [str(w)[:120] for w in wnm][:3] if isinstance(wnm, list) else [],
                    "model": MODEL})
    return out


def complete(text: str, usage: dict, max_tokens: int = 4000) -> str | None:
    usage["calls"] += 1
    for attempt in range(3):
        try:
            r = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion", timeout=120,
                              headers={"Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}"},
                              json={"modelUri": f"gpt://{os.environ['YANDEX_FOLDER_ID']}/{MODEL}",
                                    "completionOptions": {"temperature": 0.1, "maxTokens": max_tokens},
                                    "messages": [{"role": "user", "text": text}]})
            r.raise_for_status()
            res = r.json()["result"]
            usage["tokens"] += int((res.get("usage") or {}).get("totalTokens", 0))
            return res["alternatives"][0]["message"]["text"]
        except Exception as e:  # noqa: BLE001
            usage["errors"].append(str(e)[:200])
            time.sleep(5 * (attempt + 1))
    return None


def load_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    out = {}
    for l in CACHE.read_text(encoding="utf-8").splitlines():
        if l.strip():
            x = json.loads(l)
            out[x["key"]] = x
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-calls", type=int, default=50)
    ap.add_argument("--dry-run", type=int, default=0, help="показать N кандидатов в промпте и не звать модель")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    items = base_items()
    cache = load_cache()
    if a.stats:
        n = len(cache)
        print(f"в кэше {n} из {len(items)} записей базы; с цитатой {sum(1 for x in cache.values() if x.get('quote_ok'))}; "
              f"типы: {pd.Series([x.get('kind') for x in cache.values()]).value_counts().to_dict()}")
        return 0
    todo = [it for it in items if it["key"] not in cache]
    if a.dry_run:
        part = todo[: a.dry_run]
        print(prompt_for(part))
        print("\n--- ожидаемый ответ (схема) ---")
        print(json.dumps([{"i": 1, "phrase": part[0]["phrase"], "parent": "", "mechanism": "", "quote": "", "first_work": "",
                           "kind": "метод", "abstract": "", "why_not_mature": ["", "", ""]}], ensure_ascii=False, indent=1))
        return 0
    load_env()
    if not (os.getenv("YANDEX_API_KEY") and os.getenv("YANDEX_FOLDER_ID")):
        print("нет YANDEX_API_KEY / YANDEX_FOLDER_ID: положите ключ в .env проекта", file=sys.stderr)
        return 2
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    usage = {"calls": 0, "tokens": 0, "errors": []}
    got = 0
    for n in range(0, len(todo), BATCH):
        if usage["calls"] >= a.max_calls:
            break
        part = todo[n:n + BATCH]
        raw = complete(prompt_for(part), usage)
        rows = parse(raw, part) if raw else []
        with open(CACHE, "a", encoding="utf-8") as fh:
            for rec in rows:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        got += len(rows)
        print(f"  вызовов {usage['calls']}, токенов {usage['tokens']:,}, принято {got}/{min(n + BATCH, len(todo))}"
              .replace(",", " "), flush=True)
    print(f"итого: вызовов {usage['calls']}, токенов {usage['tokens']:,}, записей {got} -> {CACHE}".replace(",", " "))
    if usage["errors"]:
        print("ошибки:", usage["errors"][:3])
    return 0


if __name__ == "__main__":
    sys.exit(main())
