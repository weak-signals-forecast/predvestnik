"""Паспорта сигналов: «технология + новое применение + наблюдаемое событие», подтверждённые источниками корпуса.

    python -m registry.passports candidates            # без модели: кандидаты и их источники -> passport_candidates.parquet
    python -m registry.passports ask --max-calls 400   # YandexGPT Lite заполняет паспорта (кэш passports.jsonl)
    python -m registry.passports stats                 # сколько подтверждено, по областям
    python -m registry.passports subject               # без модели YandexGPT: о самом ли сигнале цитата -> passport_subject.json
    python -m registry.passports records               # подтверждённые новые сигналы -> passport_records.parquet

Кандидаты двух видов:
  составной — пара «метод ИИ × объект» из registry/composites.py (свежая и растущая);
  рыночный  — фраза, которая с 2023 года появилась в Hacker News, стартапах YC и прессе и растёт там
              (бизнес-слой реестра), с технологическим словом, не версия продукта и не новость.
Простая комбинация без подтверждающего события кандидатом не становится: YandexGPT Lite читает 3–6 свежих
источников кандидата и отвечает, есть ли в них конкретный механизм и наблюдаемое событие (пилот, продукт,
прототип, проект ЕС, стандарт, раунд). Паспорт: что нового, кто делает, событие с датой, почему ещё рано, что
вызывает сомнение, какой следующий факт усилит сигнал, дословная цитата источника и его ссылка. Цитата обязана
совпасть с одним из выданных заголовков, ссылка — с его ссылкой; иначе паспорт не подтверждён. Дословная цитата
может быть о соседней работе из тех же источников («foundation model for predictive maintenance» с цитатой про
федеративное обучение) — поэтому цитата ещё проверяется на предмет (subject): у составного сигнала в ней метод или
объект либо близость по смыслу ≥ 0,50, у термина базы — близость ≥ 0,40 или половина его слов.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from .corpus import REGISTRY as REG

CAND = REG / "passport_candidates.parquet"
CACHE = REG / "passports.jsonl"
BATCH = 5
DOCS = 6
MARKET_LIMIT = 1500
AREAS = ("Edge", "Защита ИИ", "Индустриальный ИИ", "Инфраструктура ИИ", "Роботы", "Финтех")

TECH_WORD = re.compile(
    r"\b(protocol|standard|agent|agents|agentic|model|models|inference|chip|chips|npu|gpu|sensor|sensors|payment|"
    r"payments|token|tokens|tokenized|wallet|stablecoin|stablecoins|robot|robots|robotics|humanoid|drone|drones|"
    r"battery|batteries|reactor|satellite|orbital|identity|auth|authentication|security|attack|attacks|compliance|"
    r"insurance|lending|credit|fraud|kyc|aml|marketplace|api|runtime|compiler|memory|cache|fine.?tuning|rag|"
    r"embedding|embeddings|mcp|a2a|llm|llms|vlm|world model|digital twin|quantum|photonic|optical|cooling|"
    r"interconnect|datacenter|data center|edge|on.device|local|federated|watermark|deepfake|provenance|browser|"
    r"copilot|assistant|automation|manufacturing|factory|warehouse|logistics|grid|energy|fusion|nuclear)\b")
NOT_TECH = re.compile(
    r"\b(trump|musk|biden|vance|bessent|mamdani|kirk|epstein|iran|israel|gaza|ukraine|russia|china tariff|pope|"
    r"election|senate|congress|lawsuit|sues|says|ceo|layoffs?|stock|stocks|earnings|ipo|deal|boom|bubble|"
    r"spending|demand|slop|hype|war|strikes?|ceasefire|files)\b")
VERSION = re.compile(r"\d")

PROMPT = """Ты аналитик технологического радара банка. Ниже кандидаты в слабые сигналы и свежие источники по
каждому (заголовки научных работ, проектов ЕС, обсуждений Hacker News, стартапов YC, прессы; дата и ссылка).
Слабый сигнал = конкретная технология + новое применение + НАБЛЮДАЕМОЕ СОБЫТИЕ в источниках: пилот, продукт,
прототип, проект, стандарт, раунд, первое внедрение. Общая тема без события — не сигнал.
Для КАЖДОГО кандидата реши по источникам (не по своим знаниям):
confirmed — есть ли в источниках конкретный механизм и наблюдаемое событие (true/false);
signal — формулировка сигнала по-русски, до 14 слов, конкретно: что и где применяют (с английским термином в скобках);
what_new — что нового: какой механизм появился (до 20 слов);
who — кто делает: организация и тип (продукт, прототип, исследование, проект ЕС, стартап) — только из источников;
event — наблюдаемое событие с датой из источника (до 20 слов);
why_early — почему ещё рано: признаки стадии и пробелы (до 20 слов);
doubt — что вызывает сомнение (до 15 слов);
next_fact — какой следующий факт усилит сигнал (до 15 слов);
quote — ДОСЛОВНЫЙ фрагмент одного из заголовков-источников этого кандидата (3–15 слов);
url — ссылка ИМЕННО этого источника из списка (или "" если у него нет ссылки);
area — одна из: Edge, Защита ИИ, Индустриальный ИИ, Инфраструктура ИИ, Роботы, Финтех, другое.
Верни СТРОГО JSON-массив, по объекту на кандидата:
[{"i": <номер>, "confirmed": true|false, "signal": "...", "what_new": "...", "who": "...", "event": "...",
  "why_early": "...", "doubt": "...", "next_fact": "...", "quote": "...", "url": "...", "area": "..."}]

Кандидаты:
{items}"""


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-zа-я0-9]+", (s or "").lower()))


def candidates() -> pd.DataFrame:
    from .export_web import business_evidence
    from .signal_base import is_fragment, is_umbrella
    rows = []
    comp = pd.read_parquet(REG / "composites.parquet")
    for r in comp[comp.signal].itertuples():
        docs = json.loads(r.examples)[:DOCS]
        rows.append({"id": f"combo:{r.method}|{r.object}", "kind": "составной", "phrase": r.phrase,
                     "hint": r.name_ru, "area_hint": getattr(r, "area", None) or "", "growth": float(r.growth),
                     "volume": int(r.docs_recent), "docs": json.dumps(docs, ensure_ascii=False)})
    b = pd.read_parquet(REG / "all_business.parquet")
    have = {json.loads(l)["phrase"] for l in (REG / "llm_assess.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    ph = b.phrase.str.replace("_", " ")
    m = (b.candidate & (b.first_year >= 2023) & (b.volume_recent >= 5) & (b.growth >= 1.5) & ~b.generic.astype(bool)
         & ph.map(lambda p: bool(TECH_WORD.search(p)) and not NOT_TECH.search(p) and not VERSION.search(p))
         & ~b.phrase.map(is_fragment) & ~b.phrase.map(is_umbrella) & ~ph.isin(have))
    mk = b[m].assign(rank=lambda d: d.volume_recent.rank(pct=True) + d.growth.rank(pct=True)).nlargest(MARKET_LIMIT, "rank")
    ev = business_evidence(set(mk.key))
    for r in mk.itertuples():
        docs = sorted(ev.get(r.key, []), key=lambda d: str(d.get("published") or ""), reverse=True)[:DOCS]
        docs = [{"title": d.get("title"), "url": d.get("url"), "year": str(d.get("published") or "")[:10],
                 "source": {"hn": "Hacker News", "yc": "Y Combinator", "press": "пресса"}.get(d.get("layer"), d.get("layer"))}
                for d in docs if d.get("title")]
        if len(docs) >= 2:
            rows.append({"id": f"mkt:{r.key}", "kind": "рыночный", "phrase": r.phrase.replace("_", " "), "hint": "",
                         "area_hint": "", "growth": float(r.growth), "volume": int(r.volume_recent),
                         "docs": json.dumps(docs, ensure_ascii=False)})
    # сигналы базы — тот же паспорт по их источникам (проверка: подтверждают ли источники именно этот сигнал)
    import gzip
    from pathlib import Path
    bundle = Path(__file__).resolve().parents[1] / "artifacts" / "registry" / "cards.json.gz"
    if bundle.exists():
        with gzip.open(bundle, "rt", encoding="utf-8") as fh:
            cards = json.load(fh)
        base = pd.read_parquet(REG / "signal_base.parquet", columns=["key", "phrase", "name_ru"])
        for r in base.itertuples():
            c = cards.get(r.key) or {}
            docs = [{"title": s.get("title"), "url": s.get("url"), "year": str(s.get("published") or "")[:10],
                     "source": s.get("source_type") or s.get("source_name")} for s in (c.get("sources") or [])
                    if s.get("title")][:DOCS]
            if docs:
                rows.append({"id": f"base:{r.key}", "kind": "база", "phrase": r.phrase.replace("_", " "),
                             "hint": r.name_ru, "area_hint": "", "growth": 0.0, "volume": 0,
                             "docs": json.dumps(docs, ensure_ascii=False)})
    t = pd.DataFrame(rows)
    t.to_parquet(CAND)
    print(f"кандидатов в паспорта: {len(t)} (составных {int((t.kind == 'составной').sum())}, "
          f"рыночных {int((t.kind == 'рыночный').sum())}, сигналов базы {int((t.kind == 'база').sum())}) "
          f"-> вызовов Lite ~{-(-len(t) // BATCH)}")
    print("рыночные, примеры:", ", ".join(t[t.kind == "рыночный"].phrase.head(40)))
    return t


def _ask(items: list[dict], uri: str, usage: dict) -> list[dict]:
    listing = "\n\n".join(
        f"{i}. {c['phrase']}" + (f" ({c['hint']})" if c["hint"] else "") + "\n" +
        "\n".join(f"   - [{d.get('source', '')}, {d.get('year', '')}] {d.get('title', '')}"
                  + (f" <{d['url']}>" if d.get("url") else "") for d in c["docs"])
        for i, c in enumerate(items, 1))
    for attempt in range(2):
        usage["calls"] += 1
        try:
            r = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion", timeout=120,
                              headers={"Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}"},
                              json={"modelUri": uri, "completionOptions": {"temperature": 0.1, "maxTokens": 4000},
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
            out = []
            for x in rows:
                i = x.get("i") if isinstance(x, dict) else None
                if not isinstance(i, int) or isinstance(i, bool) or not 1 <= i <= len(items):
                    continue
                c = items[i - 1]
                titles = [_norm(d.get("title")) for d in c["docs"]]
                urls = {d.get("url") for d in c["docs"] if d.get("url")}
                q = _norm(x.get("quote"))
                quote_ok = bool(q) and len(q.split()) >= 3 and any(q in t for t in titles)
                url = str(x.get("url") or "")
                if not url.startswith("http"):               # вместо ссылки модель пишет подпись источника
                    url = ""
                url_ok = url in urls or not url              # ссылки нет — подтверждение даёт дословная цитата
                conf = bool(x.get("confirmed") is True and quote_ok and url_ok)
                out.append({"id": c["id"], "kind": c["kind"], "phrase": c["phrase"], "confirmed": conf,
                            "llm_confirmed": x.get("confirmed") is True, "quote_ok": quote_ok, "url_ok": url_ok,
                            **{f: str(x.get(f) or "")[:300] for f in ("signal", "what_new", "who", "event", "why_early",
                                                                      "doubt", "next_fact", "quote", "url", "area")}})
            return out
        except Exception as e:
            usage["errors"] += 1
            print("  повтор:", str(e)[:100], flush=True)
            time.sleep(5 * (attempt + 1))
    return []


def ask(max_calls: int, workers: int = 3) -> None:
    from wsignals.decompose import RUNTIME_YANDEX_MODEL
    from .llm_assess import load_env
    load_env()
    uri = f"gpt://{os.environ['YANDEX_FOLDER_ID']}/{RUNTIME_YANDEX_MODEL}"
    t = pd.read_parquet(CAND)
    done = {json.loads(l)["id"] for l in CACHE.read_text(encoding="utf-8").splitlines() if l.strip()} if CACHE.exists() else set()
    order = {"составной": 0, "рыночный": 1, "база": 2}              # новые кандидаты — первыми, потом проверка базы
    t = t[~t.id.isin(done)].assign(o=lambda d: d.kind.map(order)).sort_values(["o", "growth"], ascending=[True, False])
    items = [{"id": r.id, "kind": r.kind, "phrase": r.phrase, "hint": r.hint, "docs": json.loads(r.docs)}
             for r in t.itertuples()]
    chunks = [items[i:i + BATCH] for i in range(0, len(items), BATCH)][:max_calls]
    usage = {"calls": 0, "tokens": 0, "errors": 0}
    got = conf = 0
    with ThreadPoolExecutor(workers) as ex, open(CACHE, "a", encoding="utf-8") as fh:
        for n, res in enumerate(ex.map(lambda ch: _ask(ch, uri, usage), chunks), 1):
            for x in res:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
                got += 1
                conf += x["confirmed"]
            fh.flush()
            if n % 20 == 0 or n == len(chunks):
                print(f"  пачек {n}/{len(chunks)} · вызовов {usage['calls']} · паспортов {got}, подтверждено {conf} · "
                      f"токенов {usage['tokens']:,}".replace(",", " "), flush=True)
    print(f"итого: вызовов {usage['calls']}, токенов {usage['tokens']:,}, паспортов {got}, подтверждено {conf}".replace(",", " "))


EVENT_WORD = re.compile(r"представ|выпуст|запуст|анонс|пилот|прототип|проект|внедр|стандарт|раунд|инвест|открыл|"
                        r"разработ|создал|предлож|опубликовал[аи]? (набор|датасет|бенчмарк)|launch|releas|introduc|"
                        r"announc|pilot|prototype|deploy|funding|raised")
NOT_EVENT = re.compile(r"^(обсуждени|упоминани|публикаци|статьи|исследования по|научные работы)")
RECORDS = REG / "passport_records.parquet"
UNCONFIRMED_FACTOR = 0.7      # сигнал базы, чьи источники не подтвердили событие, — балл ×0,7 (понижение, не исключение)


SUBJECT = REG / "passport_subject.json"
SUBJ_SIM_COMBO, SUBJ_SIM_BASE = 0.50, 0.40
_FILL = {"ai", "llm", "llms", "for", "of", "the", "and", "based", "model", "models", "system", "systems", "learning",
         "large", "language"}


def _stem(w: str) -> str:
    for suf in ("ies", "es", "s", "ing", "ed"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def subject() -> None:
    """Для каждого подтверждённого паспорта: цитата — о самом сигнале? Близость USER-bge-m3 и совпадение слов."""
    from sentence_transformers import SentenceTransformer
    from .composites import METHODS, OBJECTS
    from .search import MODEL
    ps = [x for x in load(raw=True).values() if x.get("confirmed")]
    m = SentenceTransformer(MODEL, device="cpu")
    A = m.encode([x["phrase"].replace("_", " ") for x in ps], normalize_embeddings=True, batch_size=128)
    Q = m.encode([x.get("quote") or "" for x in ps], normalize_embeddings=True, batch_size=128)
    out = {}
    for x, a, q in zip(ps, A, Q):
        sim, quote = float(a @ q), (x.get("quote") or "").lower()
        if x["id"].startswith("combo:"):
            me, ob = x["id"][6:].split("|")
            ok = bool(re.search(METHODS[me][0], quote) or re.search(OBJECTS[ob][0], quote)) or sim >= SUBJ_SIM_COMBO
        else:
            ws = {_stem(w) for w in re.findall(r"[a-z0-9]+", x["phrase"].lower().replace("_", " "))} - _FILL
            qs = {_stem(w) for w in re.findall(r"[a-z0-9]+", quote)}
            ok = sim >= SUBJ_SIM_BASE or not ws or len(ws & qs) >= max(1, (len(ws) + 1) // 2)
        out[x["id"]] = ok
    SUBJECT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    bad = [x for x in ps if not out[x["id"]]]
    print(f"подтверждённых {len(ps)}; цитата не о сигнале у {len(bad)} -> {SUBJECT.name}")
    for x in bad[:15]:
        print(f"  {x['phrase']} | {(x.get('quote') or '')[:90]}")


def load(raw: bool = False) -> dict[str, dict]:
    """Паспорта по id. Подтверждение снимается, если цитата не о самом сигнале (subject, passport_subject.json)."""
    if not CACHE.exists():
        return {}
    subj = json.loads(SUBJECT.read_text(encoding="utf-8")) if SUBJECT.exists() and not raw else {}
    out = {}
    for l in CACHE.read_text(encoding="utf-8").splitlines():
        if l.strip():
            x = json.loads(l)
            if x.get("confirmed") and subj.get(x["id"]) is False:
                x = {**x, "confirmed": False, "subject_ok": False}
            out[x["id"]] = x
    return out


def records() -> pd.DataFrame:
    """Подтверждённые составные и рыночные сигналы — записи базы (колонки как у поиска по базе).
    Балл по правилу (у них нет признаков экспертной модели): 0,55 + 0,4 × ранг роста среди подтверждённых —
    подтверждённое событие ставит их в верхнюю половину базы, порядок внутри — по росту."""
    from .signal_base import our_areas
    ps = load()
    cand = pd.read_parquet(CAND).set_index("id")
    rows = []
    for pid, x in ps.items():
        if not x.get("confirmed") or x["kind"] == "база" or pid not in cand.index:
            continue
        # строгий фильтр поверх подтверждения моделью: событие — действие (продукт, пилот, проект, запуск),
        # а не «обсуждение» или «публикации»; область — по нашему словарю, а не по метке модели
        if not EVENT_WORD.search((x.get("event") or "").lower()) or NOT_EVENT.search((x.get("event") or "").lower()):
            continue
        c = cand.loc[pid]
        text_row = {"phrase": x["phrase"], "name_ru": x.get("signal"), "description_ru": x.get("what_new"), "areas": []}
        ours = our_areas(text_row)
        area = c.area_hint if c.area_hint in AREAS else (ours[0] if ours else None)
        if x["kind"] == "рыночный":                     # рыночные не берём: даже после фильтра это в основном
            continue                                    # новости и одиночные продукты (проверено выборкой глазами)
        rows.append({"key": pid, "phrase": x["phrase"], "name_ru": x["signal"] or c.hint or x["phrase"],
                     "description_ru": x["what_new"], "areas": [area] if area else [], "kind": x["kind"],
                     "growth_rule": float(c.growth), "volume": int(c.volume), "docs": c.docs})
    t = pd.DataFrame(rows)
    if len(t):
        t["expert"] = 0.55 + 0.4 * t.growth_rule.rank(pct=True)
        for a in AREAS:
            t["share:" + a] = t.areas.map(lambda l, a=a: 1.0 if a in l else 0.0)
        t = t.assign(learned=t.expert, expert_p=None, history_p=None, org_sim=0.0, biz=0, aliases=[[]] * len(t),
                     first_year=None, sci_log_volume=0.0, biz_log_hn=0.0)
    t.to_parquet(RECORDS)
    print(f"подтверждённых новых сигналов: {len(t)}" + (f"; по областям: " + ", ".join(
        f"{a} {n}" for a, n in pd.Series([a for l in t.areas for a in l] or ["—"]).value_counts().items()) if len(t) else ""))
    return t


def passport_view(x: dict | None) -> dict | None:
    """Паспорт для карточки: что появилось → чем подтверждено → почему ещё рано → сомнение → следующий факт."""
    if not x:
        return None
    return {"confirmed": bool(x.get("confirmed")), "what_new": x.get("what_new"), "who": x.get("who"),
            "event": x.get("event"), "why_early": x.get("why_early"), "doubt": x.get("doubt"),
            "next_fact": x.get("next_fact"), "quote": x.get("quote"), "url": x.get("url")}


def _combo_rank(key: str) -> float | None:
    """Ранг роста составного сигнала среди подтверждённых (как в балле по правилу: 0,55 + 0,4 × ранг)."""
    if not RECORDS.exists():
        return None
    r = pd.read_parquet(RECORDS, columns=["key", "growth_rule"])
    r["rank"] = r.growth_rule.rank(pct=True)
    x = r[r.key == key]
    return float(x["rank"].iloc[0]) if len(x) else None


def _plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    return few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many


def passport_card(key: str, strength: float, ps: dict[str, dict]) -> dict:
    """Карточка составного или рыночного сигнала (у них нет признаков реестра): паспорт, источники, ряды по годам,
    факты из рядов, лестница по слоям (наука → проекты ЕС → рынок) и разбор балла по правилу."""
    x = ps.get(key) or {}
    cand = pd.read_parquet(CAND).set_index("id")
    c = cand.loc[key] if key in cand.index else None
    docs = json.loads(c.docs) if c is not None else []
    kind = x.get("kind") or (c.kind if c is not None else "")
    series, facts, ladder, explain = None, [], [], []
    event = x.get("event") or ""
    if key.startswith("combo:"):
        from .composites import METHODS, OBJECTS
        comp = pd.read_parquet(REG / "composites.parquet")
        m, o = key[6:].split("|", 1)
        row = comp[(comp.method == m) & (comp.object == o)]
        if len(row):
            ser = {int(k): int(v) for k, v in json.loads(row.iloc[0].series).items()}
            cser = {int(k): int(v) for k, v in json.loads(row.iloc[0].cordis_series).items()}
            ys = sorted(ser)
            series = {"years": ys, "lines": [
                {"label": "наука (записей с методом и объектом)", "color": "science", "values": [ser[y] for y in ys]},
                {"label": "проекты ЕС (штук)", "color": "yc", "values": [cser.get(y, 0) for y in ys]}]}
            first = next((y for y in ys if ser[y] >= 2), None)
            recent = sum(ser.get(y, 0) for y in (2024, 2025, 2026))
            before = sum(ser.get(y, 0) for y in (2020, 2021, 2022))
            eu = sum(cser.get(y, 0) for y in (2022, 2023, 2024, 2025, 2026))
            if first:
                facts.append(f"Новая связка: в науке заметна с {first} года")
            if before:
                facts.append(f"Записей в научных источниках с методом и объектом: {before} за 2020–2022 → {recent} за 2024–2026 "
                             f"(×{recent / before:.0f})".replace(".", ","))
            elif recent:
                facts.append(f"{recent} {_plural(recent, 'запись', 'записи', 'записей')} в научных источниках "
                             f"за 2024–2026, до 2023 года — ни одной")
            if eu:
                facts.append(f"{eu} {_plural(eu, 'проект', 'проекта', 'проектов')} ЕС (CORDIS) в 2022–2026 — "
                             f"выход из лаборатории во внедрение")
            facts.append("Связки нет ни в Википедии, ни среди стартапов YC")
            ladder = [{"layer": "Наука", "present": recent > 0, "value": f"{recent} записей" if recent else "нет"},
                      {"layer": "Проекты ЕС", "present": eu > 0,
                       "value": f"{eu} {_plural(eu, 'проект', 'проекта', 'проектов')}" if eu else "нет"},
                      {"layer": "Стартапы YC", "present": False, "value": "нет"},
                      {"layer": "Пресса", "present": None, "value": "нет данных"},
                      {"layer": "Энциклопедия", "present": False, "value": "нет"}]
            # событие показываем, только если оно о самом сигнале (в нём метод или объект)
            ev = event.lower()
            if event and not (re.search(METHODS[m][0], ev) or re.search(OBJECTS[o][0], ev)
                              or any(w in ev for w in (METHODS[m][1].lower().split()[0], OBJECTS[o][1].lower().split()[0]))):
                event = ""
        rank = _combo_rank(key)
        if rank is not None:
            explain = [{"label": "Рост работ с методом и объектом (ранг среди составных сигналов)",
                        "value": round(0.4 * (rank - 0.5), 3), "detail": f"ранг роста: {rank:.2f}"},
                       {"label": "Событие подтверждено цитатой из источников (паспорт)", "value": 0.05,
                        "detail": "база балла 0,55 против 0,5 у средней записи"}]
    if event:
        facts.append(event)
    who = (x.get("who") or "").strip()
    if who and not re.fullmatch(r"(исследовани[яе]|проекты|учёные|ученые)[^а-я]*(\(.*\))?( и проекты ес)?", who.lower()):
        facts.append(f"Кто делает: {who}")          # «Исследования (openalex)» — не ответ на «кто», не показываем
    return {"key": key, "name": x.get("phrase") or key, "name_ru": x.get("signal") or x.get("phrase"),
            "strength": round(float(strength), 3), "strength_registry": None, "facts": facts, "ladder": ladder,
            "series": series or {"years": [], "lines": []}, "explain": explain, "explain_history": None,
            "explain_rule": "Балл составного сигнала — по правилу: 0,55 за подтверждённое событие в источниках и "
                            "до 0,4 за рост работ, где встречаются и метод, и объект (ранг среди составных). "
                            "Признаков модели реестра у связок нет — поэтому разбор по правилу, а не SHAP.",
            "description": {"text": x.get("what_new"), "generated": True} if x.get("what_new") else None,
            "why": {"text": x.get("why_early"), "generated": True} if x.get("why_early") else None,
            "advantage": None, "case": {"text": event, "generated": True} if event else None,
            "passport": passport_view(x), "actors": None, "twin": None, "kind": kind,
            "sources": [{"title": d.get("title"), "url": d.get("url") or "", "source_name": d.get("source"),
                         "published": d.get("year"), "source_type": d.get("source"), "language": "en",
                         "trust": "средний", "trust_reason": "источник корпуса"} for d in docs]}


def stats() -> None:
    rows = [json.loads(l) for l in CACHE.read_text(encoding="utf-8").splitlines() if l.strip()]
    t = pd.DataFrame(rows).drop_duplicates("id", keep="last")
    print(f"паспортов {len(t)}; модель подтвердила {int(t.llm_confirmed.sum())}; прошли проверку цитаты и ссылки "
          f"{int(t.confirmed.sum())}")
    print(t[t.confirmed].groupby(["kind", "area"]).size().to_string())
    for r in t[t.confirmed].head(12).itertuples():
        print(f"  [{r.kind}] {r.signal} | {r.event} | «{r.quote}»")


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "candidates":
        candidates()
    elif cmd == "ask":
        n = int(sys.argv[sys.argv.index("--max-calls") + 1]) if "--max-calls" in sys.argv else 50
        ask(n)
    elif cmd == "stats":
        stats()
    elif cmd == "subject":
        subject()
    elif cmd == "records":
        records()
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
