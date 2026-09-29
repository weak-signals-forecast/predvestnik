"""Выбор на запросе: YandexGPT Pro отбирает слабые сигналы из 40 кандидатов базы (≤ 4 вызовов на запрос).

    python -m registry.select              # по search_top40_base.csv -> selected_top15.csv

Модель смотрит глазами экспертов заказчика: в промпте — сигналы организаторов нужной области как образец того,
что эксперты признают слабым сигналом (сдвиг на рынке, «X вместо Y», «X как отдельный рынок», «X выходит из
лаборатории», свежесть 2024–2026), а не как список ответов. По каждому кандидату модель решает: слабый ли это
сигнал и по теме ли он, и пишет формулировку сигнала и объяснение — только по описанию и измерениям из корпуса,
без фактов от себя. Ответ принимается только с дословным повтором фразы. Кэш по (запрос, фраза).
ТОП: одобренные моделью, без дублей (все слова одной фразы входят в другую, кроме «ai/llm»), в порядке экспертного
балла базы (проверен на отложенных сигналах организаторов); не больше 15; меньше — выдаётся меньше.
    python -m registry.select --offline    # собрать выдачу только из кэша, без вызовов модели
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import pandas as pd
import requests

from .adjudicate import _echo_ok, _norm, measurements
from .corpus import REGISTRY
from .llm_assess import load_env

REG = REGISTRY
CACHE = REG / "select_cache.jsonl"
MODEL = "yandexgpt/rc"
BATCH = 10
MAX_CALLS_PER_QUERY = 4
TOP = 15

PROMPT = """Ты эксперт по технологическому форсайту в банке. Запрос пользователя: «{query}».
Слабый сигнал — ранний признак изменения, которое может стать значимым для бизнеса: новая технология, класс
продуктов, бизнес-модель или требование, которые только появляются (2024–2026), растут и ещё не стали массовыми.
Так выглядят сигналы, которые эксперты признали слабыми сигналами:
{examples}
НЕ слабый сигнал: тема научной статьи без выхода к рынку; общая область («ИИ-агенты», «безопасность LLM»);
устоявшаяся технология; прикладная задача без нового способа; продукт одной компании; обрывок фразы.

Ниже кандидаты из корпуса научных работ, Hacker News, стартапов YC и прессы: описание, измерения и заголовки
их источников со ссылками. По КАЖДОМУ ответь ПО ИСТОЧНИКАМ, а не по своим знаниям:
on_topic — подходит ли он под запрос (true/false);
early — что конкретно в источниках подтверждает раннюю стадию (до 20 слов);
mass — признаки стадии: "массовая" (уже массовая технология), "исследование" (только исследовательская тема) или
  "ранний рынок" (первые продукты, пилоты, проекты);
sources_confirm — подтверждают ли источники именно этот сигнал, а не соседнюю тему (true/false);
signal — признал бы эксперт это слабым сигналом (true/false);
quote — ДОСЛОВНЫЙ фрагмент одного из заголовков источников этого кандидата (3–15 слов);
url — ссылка этого источника из списка (или "", если у него нет ссылки);
headline — формулировка сигнала по-русски, до 14 слов, с английским термином в скобках;
why — почему это слабый сигнал, до 25 слов: новизна, рост, переход в рынок, значение для банка или отрасли.
Если источники не подтверждают сигнал — sources_confirm=false и signal=false. Не добавляй фактов от себя.
Верни СТРОГО JSON-массив, по объекту на номер:
[{"i": <номер>, "phrase": "<английская фраза дословно>", "on_topic": true|false, "early": "<...>",
  "mass": "массовая|исследование|ранний рынок", "sources_confirm": true|false, "signal": true|false,
  "quote": "<...>", "url": "<...>", "headline": "<...>", "why": "<...>"}]

Кандидаты:
{items}"""


def examples(area: str | None, k: int = 16) -> str:
    """Сигналы организаторов области (все 100 — описание цели заказчиком; для запроса без области — по 3 из каждой)."""
    from .semantic import queries
    from .stage2 import area_of_signals
    sa = area_of_signals()
    qs = queries()
    if area:
        pick = [q for q in qs if sa.get(q["id"]) == area][:k]
    else:
        per: dict[str, list] = {}
        for q in qs:
            per.setdefault(sa.get(q["id"], ""), []).append(q)
        pick = [q for v in per.values() for q in v[:3]][:k]
    return "\n".join(f"— {q['ru']}" for q in pick)


def ask(query: str, ex: str, items: list[dict], uri: str, usage: dict) -> dict[str, dict]:
    listing = "\n".join(f"{i}. {c['phrase']} — {c['name_ru']}: {c['description_ru']}\n   {c['meas']}\n   источники:\n" +
                        "\n".join(f"   - [{d['type']}, {d['date']}] {d['title']}" + (f" <{d['url']}>" if d['url'] else "")
                                  for d in c.get("sources", []))
                        for i, c in enumerate(items, 1))
    text = PROMPT.replace("{query}", query).replace("{examples}", ex).replace("{items}", listing)
    for attempt in range(2):
        if usage["calls_query"] >= MAX_CALLS_PER_QUERY:
            return {}
        usage["calls_query"] += 1
        usage["calls"] += 1
        try:
            r = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion", timeout=120,
                              headers={"Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}"},
                              json={"modelUri": uri, "completionOptions": {"temperature": 0.1, "maxTokens": 4000},
                                    "messages": [{"role": "user", "text": text}]})
            r.raise_for_status()
            res = r.json()["result"]
            usage["tokens"] += int((res.get("usage") or {}).get("totalTokens", 0))
            raw = res["alternatives"][0]["message"]["text"]
            m = re.search(r"\[.*\]", raw, re.S)
            out = {}
            try:
                rows = json.loads(m.group(0)) if m else []
            except json.JSONDecodeError:
                rows = []
            for x in rows:
                i = x.get("i") if isinstance(x, dict) else None
                if not isinstance(i, int) or isinstance(i, bool) or not 1 <= i <= len(items):
                    continue
                c = items[i - 1]
                if not _echo_ok(x.get("phrase"), c["phrase"]):
                    continue
                if not isinstance(x.get("signal"), bool) or not isinstance(x.get("on_topic"), bool):
                    continue
                src = c.get("sources", [])
                qn = _normt(x.get("quote"))
                quote_ok = bool(qn) and len(qn.split()) >= 3 and any(qn in _normt(d["title"]) for d in src)
                url = str(x.get("url") or "")
                out[c["phrase"]] = {"signal": x["signal"], "on_topic": x["on_topic"],
                                    "headline": str(x.get("headline") or "")[:160],
                                    "why": str(x.get("why") or "")[:300], "verified": True,
                                    "sources_confirm": x.get("sources_confirm") is True, "quote_ok": quote_ok,
                                    "early": str(x.get("early") or "")[:200], "mass": str(x.get("mass") or "")[:40],
                                    "quote": str(x.get("quote") or "")[:200],
                                    "url": url if url in {d["url"] for d in src} else ""}
            if len(out) < len(items):                        # что отбраковано — в лог, чтобы видеть причину
                with open(REG / "select_rejected.log", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"query": query, "asked": [c["phrase"] for c in items],
                                         "accepted": list(out), "raw": raw[:4000]}, ensure_ascii=False) + "\n")
            return out
        except Exception as e:
            print("  повтор:", str(e)[:100], flush=True)
            time.sleep(5 * (attempt + 1))
    return {}


_ENC = {}


def emb_sim(a: str, b: str) -> float:
    """Близость фраз по смыслу (USER-bge-m3, как в поиске) — для склейки дублей в выдаче."""
    if "m" not in _ENC:
        from sentence_transformers import SentenceTransformer
        from .search import MODEL
        _ENC["m"] = SentenceTransformer(MODEL, device="cpu")
    for p in (a, b):
        if p not in _ENC:
            _ENC[p] = _ENC["m"].encode([p.replace("_", " ")], normalize_embeddings=True)[0]
    return float(_ENC[a] @ _ENC[b])


def _normt(s) -> str:
    return " ".join(re.findall(r"[a-zа-я0-9]+", str(s or "").lower()))


def main() -> int:
    offline = "--offline" in sys.argv
    import gzip
    from pathlib import Path
    bundle = Path(__file__).resolve().parents[1] / "artifacts" / "registry" / "cards.json.gz"
    cards = json.load(gzip.open(bundle, "rt", encoding="utf-8")) if bundle.exists() else {}

    def sources_of(k: str, n: int = 3) -> list[dict]:
        c = cards.get(k) or {}
        if not c and str(k).startswith(("combo:", "mkt:")):
            from .passports import CAND
            pc = pd.read_parquet(CAND).set_index("id")
            docs = json.loads(pc.loc[k].docs) if k in pc.index else []
            return [{"title": (d.get("title") or "")[:180], "url": d.get("url") or "", "type": d.get("source") or "",
                     "date": str(d.get("year") or "")} for d in docs][:n]
        return [{"title": (d.get("title") or "")[:180], "url": d.get("url") or "",
                 "type": d.get("source_type") or d.get("source_name") or "", "date": str(d.get("published") or "")[:10]}
                for d in (c.get("sources") or []) if d.get("title")][:n]
    load_env()
    uri = f"gpt://{os.environ['YANDEX_FOLDER_ID']}/{MODEL}"
    cand = pd.read_csv(REG / "search_top40_base.csv")
    expert = pd.read_parquet(REG / "signal_base.parquet", columns=["key", "expert", "tier"])
    line = measurements(REG)["line"]
    cache = {}
    if CACHE.exists():
        for l in CACHE.read_text(encoding="utf-8").splitlines():
            x = json.loads(l)
            cache[(x["query"], x["phrase"])] = x
    usage = {"calls": 0, "tokens": 0}
    out = []
    for q, g in cand.groupby("query", sort=False):
        g = g.sort_values("rank").assign(ph=lambda d: d.phrase.str.replace("_", " "))
        g_extra = g[g.extra.fillna(False).astype(bool)] if "extra" in g else g.iloc[:0]
        g = g[~g.index.isin(g_extra.index)]              # модель проверяет только 40 кандидатов
        area = g.area_detected.dropna().iloc[0] if g.area_detected.notna().any() else None
        from registry_service.service import AREA_GUIDE
        ex = (AREA_GUIDE.get(area, "") + "\n" if area else "") + examples(area)
        items = [{"phrase": p, "name_ru": n, "description_ru": d, "meas": line(k), "sources": sources_of(k)}
                 for p, n, d, k in zip(g.ph, g.name_ru, g.description_ru, g.key)]
        # ответы без проверки по источникам (старый формат) спрашиваются заново
        todo = [it for it in items if not (cache.get((q, it["phrase"])) or {}).get("verified")]
        usage["calls_query"] = 0
        for i in range(0, 0 if offline else len(todo), BATCH):
            got = ask(q, ex, todo[i:i + BATCH], uri, usage)
            with open(CACHE, "a", encoding="utf-8") as fh:
                for ph, v in got.items():
                    cache[(q, ph)] = {"query": q, "phrase": ph, **v}
                    fh.write(json.dumps(cache[(q, ph)], ensure_ascii=False) + "\n")
        v = g.ph.map(lambda p: cache.get((q, p)))
        for f in ("signal", "on_topic", "headline", "why", "verified", "sources_confirm", "quote_ok", "early", "mass",
                  "quote", "url"):
            g[f] = v.map(lambda x, f=f: x.get(f) if x else None)
        from registry_service.service import accepted, _dup
        ok = v.map(accepted).astype(bool)
        cand_ok = g[ok].merge(expert, on="key", how="left").sort_values(["tier", "expert"], ascending=[True, False])
        keep: list[int] = []
        for i, ph in zip(cand_ok.index, cand_ok.ph):
            if not any(_dup(ph, cand_ok.ph[j], emb_sim(ph, cand_ok.ph[j])) for j in keep):
                keep.append(i)
        top = cand_ok.loc[keep].copy()                 # ТОП-15 — первые TOP; остальные принятые — «ещё по теме»
        if len(g_extra):                                # и ещё по теме без проверки моделью — в конец списка
            ge = g_extra.merge(expert, on="key", how="left")
            add = [i for i in ge.index if not any(_dup(ge.ph[i], p, emb_sim(ge.ph[i], p)) for p in top.ph)]
            top = pd.concat([top, ge.loc[add].assign(unverified=True)], ignore_index=True)
        top["pos"] = range(1, len(top) + 1)
        out.append(top)
        print(f"{q[:60]:60} проверено {int(v.notna().sum())}/{len(g)}, сигналов по теме {int(ok.sum())}, в выдаче {len(top)}")
    res = pd.concat(out)
    dst = REG / "selected_top15.csv"
    if dst.exists():                                           # ручную разметку не теряем
        p = pd.read_csv(dst)
        if "weak_signal" in p:
            lab = {(a, b): c for a, b, c in zip(p["query"], p["phrase"], p["weak_signal"]) if pd.notna(c)}
            res["weak_signal"] = [lab.get((a, b)) for a, b in zip(res["query"], res["phrase"])]
    res.to_csv(dst, index=False)
    print(f"вызовов {usage['calls']}, токенов {usage['tokens']:,} -> {dst}".replace(",", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
