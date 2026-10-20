"""Ответ на запрос по базе слабых сигналов: поиск по базе → выбор YandexGPT → ТОП-15 карточек в формате ТЗ.

Работает только на наборе данных artifacts/registry (registry/bundle.py) и векторной модели USER-bge-m3; корпус
не нужен. Интернет нужен только для YandexGPT (HyDE — 1 вызов Lite, выбор — до 4 вызовов Pro); без ключа сервис
отвечает без модели: ТОП-15 по порядку поиска.

Шаги (те же, что проверены офлайн в registry/search.py, registry/hyde.py, registry/select.py):
  1. тема запроса без слов-намерений; область ТЗ — ближайшее название области, подтверждённое тегами ближайших
     сигналов базы;
  2. HyDE: 5 гипотетических описаний ранних технологий под запрос (YandexGPT Lite) — только для поиска;
  3. 80 ближайших по смеси «смысл запроса + доля работ в рубриках области» → порядок: 50 % близость, 50 % балл
     моделей → склейка дублей по смыслу → 40 кандидатов;
  4. YandexGPT Pro по каждому кандидату: слабый ли сигнал и по теме ли, формулировка и «почему это сигнал» — только
     по описанию и измерениям из корпуса;
  5. одобренные — без дублей, по баллу моделей, не больше 15; карточка — из набора данных.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from wsignals import intent

from . import rank

BUNDLE = Path(os.getenv("REGISTRY_BUNDLE", Path(__file__).resolve().parents[1] / "artifacts" / "registry"))
CACHE_DIR = Path(os.getenv("REGISTRY_CACHE", BUNDLE / "cache"))
MODEL = os.getenv("REGISTRY_EMB_MODEL", "deepvk/USER-bge-m3")
PRO, LITE = "yandexgpt/rc", "yandexgpt-5-lite"
TOPIC, KEEP, TOP, DUP_SIM = 80, 40, 15, 0.88
EXTRA = 60                     # «ещё по теме» без проверки моделью — чтобы список листался дальше
DUP_NEAR = 0.80                # близость для склейки фраз с общим словом в итоговой выдаче
AREA_NAMED, VOTE_NN, MIN_AREA_SHARE = 0.75, 100, 0.30
# Версия 2 порядка и фильтров (registry_service/rank.py): оси слабости, исключения с причиной, порядок по запросу,
# группировка подсигналов. По умолчанию выключена: демонстрационный путь остаётся прежним байт в байт.
RANK_V2 = os.getenv("REGISTRY_RANK_V2", "0") == "1"
DROP_MATURE = os.getenv("REGISTRY_DROP_MATURE", "1" if RANK_V2 else "0") == "1"
SEM_TOP = 5                  # столько ближайших по смыслу проходят мимо фильтра области
# переранжирование ближайших кросс-энкодером (registry_service/rerank.py); эффект — docs/EVAL.md, scripts/eval_retrieval.py
RERANK = os.getenv("REGISTRY_RERANK", "0") == "1"
RERANK_MIN = 0.1             # вероятность соответствия ниже — кандидат не по теме и не идёт в выдачу, пока хватает других
# абстракт открытой модели не печатается, если в нём чужая письменность (иероглифы, арабица) или склейка
# кириллицы с латиницей внутри слова («автoregressive»): так модель иногда срывается на другой язык
_BROKEN = re.compile(r"[\u0600-\u06ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]|[а-яё][A-Za-z]|[A-Za-z][а-яё]", re.I)

VOTE_STRONG, NAME_MID, VOTE_MID = 0.35, 0.42, 0.20
BATCH, MAX_CALLS = 10, 4
AREA_NAMES = {"Edge": "Edge AI: ИИ на устройствах и периферийные вычисления",
              "Защита ИИ": "Защита ИИ: безопасность систем искусственного интеллекта",
              "Индустриальный ИИ": "Индустриальный ИИ: ИИ в промышленности и производстве",
              "Инфраструктура ИИ": "Инфраструктура ИИ: чипы, память, сети и дата-центры для ИИ",
              "Роботы": "Роботы и робототехника",
              "Финтех": "Финтех: платежи, банки, кредитование, страхование и оценка рисков, криптоактивы и финансовые рынки"}

# что считается слабым сигналом в каждой области — для промпта выбора (наши формулировки по названиям областей ТЗ).
# Без них модель принимала для промышленности общий корпоративный ИИ, а для Edge — сети связи 6G.
AREA_GUIDE = {
    "Индустриальный ИИ": "Сигнал в этой области — ИИ внутри производства и OT-стека: ПЛК, SCADA, MES, генерация "
                         "управляющего кода и технологической документации, контроль качества без разметки под каждый "
                         "дефект, промышленные временные ряды, новые способы предиктивного обслуживания, роботы на "
                         "заводе. НЕ сигнал: общий ИИ для предприятий (агенты, RAG, ассистенты, кодинг), классические "
                         "методы управления и диагностики, давно известные задачи поиска дефектов.",
    "Инфраструктура ИИ": "Сигнал в этой области — новый класс железа и инфраструктуры для ИИ: чипы и ускорители, память "
                         "(HBM, CXL, флеш для инференса), интерконнект и оптика, охлаждение и питание дата-центров, "
                         "системы обслуживания моделей нового типа. НЕ сигнал: общие «ИИ-дата-центры», «кластеры ИИ», "
                         "обычные GPU, программные фреймворки общего назначения.",
    "Edge": "Сигнал в этой области — ИИ на устройстве без облака: NPU в смартфонах, ПК и микроконтроллерах, квантизация и "
            "дообучение на устройстве, локальные языковые модели, вычисления в сенсоре, носимые ИИ-устройства. "
            "НЕ сигнал: сети связи 6G (метаповерхности, MIMO), облачные сервисы, общие темы про агентов.",
    "Роботы": "Сигнал в этой области — новые способы обучать и строить роботов: фундаментальные модели роботов, данные "
              "телеуправления, ловкие кисти, гуманоиды в работе, перенос из симуляции. НЕ сигнал: классическое "
              "планирование пути и управление.",
    "Защита ИИ": "Сигнал в этой области — новые атаки на ИИ и защита от них: агенты, инструменты и протоколы агентов, "
                 "отравление данных и памяти, водяные знаки, мониторинг рассуждений, конфиденциальный инференс. "
                 "НЕ сигнал: общая кибербезопасность без ИИ, общие слова «безопасность ИИ».",
    "Финтех": "Сигнал в этой области — новые финансовые технологии и модели: платежи ИИ-агентов, токенизация активов, "
              "стейблкоины, новые способы скоринга и антифрода, цифровая идентичность, регулирование. НЕ сигнал: "
              "эконометрика и финансовая математика, давно известные рыночные механизмы.",
}


SELECT_PROMPT = """Ты эксперт по технологическому форсайту в банке. Запрос пользователя: «{query}».
Слабый сигнал — ранний признак изменения, которое может стать значимым для бизнеса: новая технология, класс
продуктов, бизнес-модель или требование, которые только появляются (2024–2026), растут и ещё не стали массовыми.
Примеры слабых сигналов этой области:
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

HYDE_PROMPT = """Запрос аналитика банка: «{query}».
Назови 5 РАЗНЫХ конкретных технологий ранней стадии (появились в 2023–2026, ещё не массовые), которые отвечают на
этот запрос. Для каждой — одна строка: русское название, английский термин в скобках, одно предложение о сути.
Без вступления и нумерации, только 5 строк."""


def _norm(s: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _echo_ok(echo, phrase: str) -> bool:
    """Модель повторила фразу кандидата. Pro часто возвращает всю строку «фраза — название: описание» —
    принимаем, если до первого тире стоит ровно эта фраза."""
    e = str(echo or "")
    head = re.split(r"\s[—–-]\s", e, maxsplit=1)[0]
    return _norm(head) == _norm(phrase) or _norm(e) == _norm(phrase)


def _normt(s) -> str:
    return " ".join(re.findall(r"[a-zа-я0-9]+", str(s or "").lower()))


def accepted(j: dict | None) -> bool:
    """Строка идёт в выдачу: сигнал и по теме; после проверки по источникам — ещё и источники подтверждают
    (цитата найдена дословно) и это не массовая технология. Нет подтверждения — исключение."""
    if not j or not j.get("signal") or not j.get("on_topic"):
        return False
    if j.get("verified"):
        return bool(j.get("sources_confirm") and j.get("quote_ok") and j.get("mass") != "массовая")
    return True


_FILLER = {"ai", "llm", "llms", "based", "for", "of", "the", "and"}


def _stem(w: str) -> str:
    """Основа слова для склейки дублей: agentic/agent, payments/payment, systems/system."""
    for suf in ("ic", "ies", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _dup(a: str, b: str, sim: float = 0.0) -> bool:
    """Одна тема в выдаче дважды: слова одной фразы входят в другую; или общие ≥ 2/3 слов и то же главное слово
    («bimanual mobile / dexterous manipulation»); или есть общее слово и близость по смыслу ≥ 0,80 («dexterous grasp
    synthesis / generation», «kv cache offloading / transfer»); или близость ≥ 0,88."""
    wa, wb = a.replace("_", " ").split(), b.replace("_", " ").split()
    x = {_stem(_stem(w)) for w in wa} - {_stem(f) for f in _FILLER}
    y = {_stem(_stem(w)) for w in wb} - {_stem(f) for f in _FILLER}
    if not x or not y:
        return False
    if x <= y or y <= x:
        return True
    c = len(x & y)
    same_head = _stem(_stem(wa[-1])) == _stem(_stem(wb[-1]))
    return (c >= 2 and c >= 2 / 3 * min(len(x), len(y)) and same_head) or (c >= 1 and sim >= DUP_NEAR) or sim >= DUP_SIM


class Registry:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.sig = pd.read_parquet(BUNDLE / "signals.parquet")
        self.E = np.load(BUNDLE / "emb.npy").astype(np.float32)
        with gzip.open(BUNDLE / "cards.json.gz", "rt", encoding="utf-8") as fh:
            self.cards = json.load(fh)
        # адаптер запроса: кросс-энкодер, дистиллированный в матрицу над вектором запроса (scripts/distill_adapter.py)
        ad = BUNDLE / "query_adapter.npy"
        self.adapter = np.load(ad).astype(np.float32) if ad.exists() and os.getenv("REGISTRY_ADAPTER", "1") == "1" else None
        self.reranker = None
        if RERANK:
            from .rerank import Reranker
            self.reranker = Reranker()
        self.axes = None
        ax_p = BUNDLE / "axes.parquet"
        if RANK_V2 and ax_p.exists():
            self.axes = pd.read_parquet(ax_p).drop_duplicates("key")
        if DROP_MATURE and ax_p.exists():            # записи, которые экспертная проверка базы признала зрелыми («M»)
            verdict = pd.read_parquet(ax_p, columns=["key", "verdict"]).drop_duplicates("key").set_index("key").verdict
            keep = ~self.sig.key.map(verdict).eq("M").values
            self.sig, self.E = self.sig[keep].reset_index(drop=True), self.E[keep]
        self.examples = json.loads((BUNDLE / "examples.json").read_text(encoding="utf-8"))
        self.model_global = json.loads((BUNDLE / "model_global.json").read_text(encoding="utf-8"))
        self.m = SentenceTransformer(MODEL, device="cpu")
        self.m.max_seq_length = 64
        self.N = {a: self.m.encode([t], normalize_embeddings=True)[0] for a, t in AREA_NAMES.items()}
        self.area_n = {a: max(1, int(self.sig.areas.map(lambda l, a=a: a in list(l)).sum())) for a in AREA_NAMES}
        self.ph = self.sig.phrase.str.replace("_", " ")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        seed = BUNDLE / "cache"                          # ответы YandexGPT из набора данных — в рабочий кэш
        if seed.exists() and seed.resolve() != CACHE_DIR.resolve():
            import shutil
            for f in seed.iterdir():
                if not (CACHE_DIR / f.name).exists():
                    shutil.copy(f, CACHE_DIR / f.name)
        self.lock = threading.Lock()
        self.select_cache = self._load_jsonl("select_cache.jsonl", lambda x: (x["query"], x["phrase"]))
        # извлечённые моделью факты по записям базы (registry/llm_axes.py): родитель, механизм, тип, абстракт
        self.llm_axes = self._load_jsonl("llm_axes.jsonl", lambda x: x["key"]) if RANK_V2 else {}
        cal_p = BUNDLE / "calibration.json"                 # балл → калиброванная вероятность (registry/calibrate.py)
        self.calibration = json.loads(cal_p.read_text(encoding="utf-8")) if RANK_V2 and cal_p.exists() else None
        hy = CACHE_DIR / "hyde.json"
        self.hyde_cache = json.loads(hy.read_text(encoding="utf-8")) if hy.exists() else {}

    def _load_jsonl(self, name, key):
        p, out = CACHE_DIR / name, {}
        if p.exists():
            for l in p.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    x = json.loads(l)
                    out[key(x)] = x
        return out

    # ---------- YandexGPT ----------
    @staticmethod
    def _llm_ready() -> bool:
        return bool(os.getenv("YANDEX_API_KEY") and os.getenv("YANDEX_FOLDER_ID"))

    @staticmethod
    def _complete(model: str, text: str, max_tokens: int, usage: dict) -> str | None:
        usage["calls"] += 1
        try:
            r = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion", timeout=120,
                              headers={"Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}"},
                              json={"modelUri": f"gpt://{os.environ['YANDEX_FOLDER_ID']}/{model}",
                                    "completionOptions": {"temperature": 0.1, "maxTokens": max_tokens},
                                    "messages": [{"role": "user", "text": text}]})
            r.raise_for_status()
            res = r.json()["result"]
            usage["tokens"] += int((res.get("usage") or {}).get("totalTokens", 0))
            return res["alternatives"][0]["message"]["text"]
        except Exception as e:  # noqa: BLE001
            usage["errors"].append(str(e)[:200])
            return None

    def hyde(self, query: str, usage: dict) -> list[str] | None:
        if query in self.hyde_cache:
            return self.hyde_cache[query]
        if not self._llm_ready():
            return None
        raw = self._complete(LITE, HYDE_PROMPT.replace("{query}", query), 800, usage)
        if not raw:
            return None
        lines = [re.sub(r"^[\s\-–—•\d.)]+", "", l).strip() for l in raw.splitlines()]
        lines = [l for l in lines if len(l) > 15][:5]
        if lines:
            with self.lock:
                self.hyde_cache[query] = lines
                (CACHE_DIR / "hyde.json").write_text(json.dumps(self.hyde_cache, ensure_ascii=False), encoding="utf-8")
        return lines or None

    def select(self, query: str, area: str | None, cand: pd.DataFrame, usage: dict) -> None:
        # ответы без проверки по источникам (старый формат) спрашиваются заново
        todo = [i for i in cand.index if not (self.select_cache.get((query, self.ph[i])) or {}).get("verified")]
        if not todo or not self._llm_ready():
            return
        ex = self.examples.get(area) if area else None
        if not ex:
            ex = [x for v in self.examples.values() for x in v[:3]]
        ex_text = (AREA_GUIDE.get(area, "") + "\n" if area else "") + "\n".join(f"— {x}" for x in ex[:16])
        calls = 0
        for n in range(0, len(todo), BATCH):
            if calls >= MAX_CALLS:
                break
            part = todo[n:n + BATCH]
            listing = "\n".join(f"{j}. {self.ph[i]} — {self.sig.name_ru[i]}: {self.sig.description_ru[i]}\n   "
                                f"{self.sig.meas[i]}\n   источники:\n" +
                                "\n".join(f"   - [{d['type']}, {d['date']}] {d['title']}" + (f" <{d['url']}>" if d['url'] else "")
                                          for d in self._sources(i))
                                for j, i in enumerate(part, 1))
            calls += 1
            raw = self._complete(PRO, SELECT_PROMPT.replace("{query}", query).replace("{examples}", ex_text)
                                 .replace("{items}", listing), 4000, usage)
            m = re.search(r"\[.*\]", raw or "", re.S)
            try:
                rows = json.loads(m.group(0)) if m else []
            except json.JSONDecodeError:
                rows = []
            new = []
            for x in rows:
                j = x.get("i") if isinstance(x, dict) else None
                if not isinstance(j, int) or isinstance(j, bool) or not 1 <= j <= len(part):
                    continue
                i = part[j - 1]
                if not _echo_ok(x.get("phrase"), self.ph[i]):
                    continue
                if not isinstance(x.get("signal"), bool) or not isinstance(x.get("on_topic"), bool):
                    continue
                src = self._sources(i)
                q = _normt(x.get("quote"))
                quote_ok = bool(q) and len(q.split()) >= 3 and any(q in _normt(d["title"]) for d in src)
                url = str(x.get("url") or "")
                rec = {"query": query, "phrase": self.ph[i], "signal": x["signal"], "on_topic": x["on_topic"],
                       "headline": str(x.get("headline") or "")[:160], "why": str(x.get("why") or "")[:300],
                       "verified": True, "sources_confirm": x.get("sources_confirm") is True, "quote_ok": quote_ok,
                       "early": str(x.get("early") or "")[:200], "mass": str(x.get("mass") or "")[:40],
                       "quote": str(x.get("quote") or "")[:200],
                       "url": url if url in {d["url"] for d in src} else ""}
                new.append(rec)
            with self.lock:
                with open(CACHE_DIR / "select_cache.jsonl", "a", encoding="utf-8") as fh:
                    for rec in new:
                        self.select_cache[(query, rec["phrase"])] = rec
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _doc_text(self, i: int) -> str:
        """Текст записи для кросс-энкодера: русское название, термин, описание и заголовки первых источников."""
        c = self.cards.get(self.sig.key[i]) or {}
        desc = self.sig.description_ru[i] if isinstance(self.sig.description_ru[i], str) else ""
        ts = [(d.get("title") or "") for d in (c.get("sources") or [])][:3]
        return ". ".join(x for x in [self.sig.name_ru[i], self.ph[i], desc] + ts if x)

    def _sources(self, i: int, n: int = 3) -> list[dict]:
        c = self.cards.get(self.sig.key[i]) or {}
        return [{"title": (d.get("title") or "")[:180], "url": d.get("url") or "",
                 "type": d.get("source_type") or d.get("source_name") or "", "date": str(d.get("published") or "")[:10]}
                for d in (c.get("sources") or []) if d.get("title")][:n]

    # ---------- поиск ----------
    def area_of(self, qv: np.ndarray) -> str | None:
        sims = {a: float(self.N[a] @ qv) for a in self.N}
        a = max(sims, key=sims.get)
        sim = self.E @ qv
        vote = dict.fromkeys(self.N, 0.0)
        for i in np.argsort(-sim)[:VOTE_NN]:
            for t in self.sig.areas.iat[i]:
                if t in vote:
                    vote[t] += sim[i] / self.area_n[t] ** 0.5
        # голоса по базе крупнее, чем по полному пулу (база плотнее в темах ИИ): пороги подобраны на базе, на 23 запросах
        ok = (sims[a] >= AREA_NAMED or vote[a] >= VOTE_STRONG or (sims[a] >= NAME_MID and vote[a] >= VOTE_MID))
        return a if ok else None

    def candidates(self, query: str, usage: dict, use_hyde: bool = True) -> tuple[pd.DataFrame, str | None]:
        topic = intent.split(query).domain or query
        qv = self.m.encode([topic], normalize_embeddings=True)[0]
        area = self.area_of(qv)
        if self.adapter is not None:
            qv = self.adapter @ qv
            qv = qv / np.linalg.norm(qv)
        rel = self.E @ qv
        hy = self.hyde(query, usage) if use_hyde else None
        if hy:
            H = self.m.encode(hy, normalize_embeddings=True)
            rel = 0.5 * rel + 0.5 * (self.E @ H.T).max(axis=1)
        t = self.sig.assign(rel_query=rel)
        if area:
            qr = t.rel_query.rank(pct=True)
            t["relevance"] = 0.5 * qr + 0.5 * t["share:" + area].rank(pct=True)
            # самые близкие по смыслу не отсекаются областью: область могла определиться неточно или сигнал на стыке
            sem = t.index.isin(t.rel_query.nlargest(SEM_TOP).index)
            t.loc[sem, "relevance"] = np.maximum(t.relevance[sem], qr[sem])
            if "areas_our" in t:     # своя принадлежность к области (signal_base.our_areas): рубрики + предметное слово
                t = t[t.areas_our.map(lambda l: area in list(l)) | sem]
            else:
                t = t[(t["share:" + area] >= MIN_AREA_SHARE) | t.areas.map(lambda l: area in list(l)) | sem]
        else:
            t["relevance"] = t.rel_query
        near = t.nlargest(TOPIC, "relevance")
        if self.reranker is not None:
            # близость к запросу — по кросс-энкодеру (пара «запрос — текст записи»), а не по векторам; явно чужие темы
            # (вероятность < RERANK_MIN) отходят, если по теме остаётся хотя бы KEEP // 4 кандидатов
            p = 1 / (1 + np.exp(-self.reranker.scores(query, [self._doc_text(i) for i in near.index])))
            near = near.assign(rel_query=p)
            if (p >= RERANK_MIN).sum() >= KEEP // 4:
                near = near[p >= RERANK_MIN]
        near = near.assign(order=self._order(near)).sort_values("order", ascending=False)
        picked: list[int] = []
        for i in near.index:
            if picked and float((self.E[picked] @ self.E[i]).max()) >= DUP_SIM:
                continue
            picked.append(i)
            if len(picked) >= KEEP:
                break
        out = t.loc[picked].copy()
        out["strength"] = out.expert                 # итоговый балл (смесь моделей) — та же шкала, что на странице
        # «ещё по теме» за пределами 40 кандидатов: следующие по тому же порядку, без дублей; моделью не проверяются
        far = t.nlargest(TOPIC * 3, "relevance")
        far = far.assign(order=self._order(far)).sort_values("order", ascending=False)
        extra: list[int] = []
        for i in far.index:
            if i in picked or float((self.E[picked + extra] @ self.E[i]).max()) >= DUP_SIM:
                continue
            extra.append(i)
            if len(extra) >= EXTRA:
                break
        ext = t.loc[extra].copy()
        ext["strength"] = ext.expert
        self._extra = ext
        return out, area

    def _order(self, t: pd.DataFrame) -> pd.Series:
        """Порядок кандидатов. v1: половина близости, половина балла, ярус 1 разметки первым.
        v2: близость × слабость (оси, без эшелона) — registry_service/rank.py."""
        if RANK_V2:                                   # близость именно к запросу, без примеси доли рубрик области
            return rank.order(t, t.rel_query if "rel_query" in t else t.relevance, self.axes)
        tier = t.tier.fillna(2) - 1 if "tier" in t else 0
        return 0.5 * t.relevance.rank(pct=True) + 0.5 * t.expert - tier

    def _row(self, i: int) -> dict:
        """Запись базы вместе с её осями (для фильтров и карточки v2)."""
        r = {k: self.sig.at[i, k] for k in ("key", "phrase", "kind", "first_year", "expert")}
        if self.axes is not None:
            a = self.axes[self.axes.key == r["key"]]
            if len(a):
                r.update(a.iloc[0].to_dict())
        return r

    # ---------- ответ ----------
    def answer(self, query: str, use_llm: bool = True) -> dict:
        t0 = time.time()
        usage = {"calls": 0, "tokens": 0, "errors": []}
        cand, area = self.candidates(query, usage, use_hyde=use_llm)
        if use_llm:
            self.select(query, area, cand, usage)
        judged = cand.index.map(lambda i: self.select_cache.get((query, self.ph[i])))
        mode = "выбор YandexGPT" if use_llm and any(j is not None for j in judged) else "без модели: порядок поиска"
        excluded: list[dict] = []
        subs: dict[int, list[int]] = {}
        if mode.startswith("выбор"):
            ok = [i for i, j in zip(cand.index, judged) if accepted(j)]
        else:
            ok = list(cand.index)
        if RANK_V2:
            kept = []
            for i in ok:                              # исключения с причиной: стандарт, зрелое, зрелое по проверке базы
                ex = rank.exclusion(self._row(i))
                if ex:
                    excluded.append({"term_en": self.ph[i], "name": self.sig.name_ru[i], "code": ex[0], "reason": ex[1]})
                else:
                    kept.append(i)
            order = cand.order if "order" in cand else self._order(cand)
            ok = sorted(kept, key=lambda i: -float(order[i]))
            grouped = rank.group(ok, lambda i, j: _dup(self.ph[i], self.ph[j], float(self.E[i] @ self.E[j])))
            top = [i for i, _ in grouped]
            subs = {i: s for i, s in grouped}
        else:
            ok = sorted(ok, key=lambda i: (float(self.sig.tier[i]) if "tier" in self.sig else 1.0,
                                           -float(self.sig.expert[i]))) if mode.startswith("выбор") else ok
            top = []
            for i in ok:                              # ТОП-15 и «ещё по теме»: все принятые без дублей
                if not any(_dup(self.ph[i], self.ph[j], float(self.E[i] @ self.E[j])) for j in top):
                    top.append(i)
        signals = [self.card(query, i, rank_, cand, subs.get(i)) for rank_, i in enumerate(top[:TOP], 1)]
        more = [self.card(query, i, rank_, cand, subs.get(i)) for rank_, i in enumerate(top[TOP:], TOP + 1)]
        ext = getattr(self, "_extra", None)
        if ext is not None and len(ext):
            rejected = {i for i, j in zip(cand.index, judged) if j is not None and not accepted(j)}
            shown = list(top)
            for i in ext.index:
                if i in rejected or any(_dup(self.ph[i], self.ph[j], float(self.E[i] @ self.E[j])) for j in shown):
                    continue
                shown.append(i)
                more.append({**self.card(query, i, len(shown), ext), "unverified": True})
        confident = sum(1 for s_ in signals + more if (s_.get("confidence") or 0) > 0.75)
        return {"query": query, "area": area, "mode": mode, "signals": signals, "more": more,
                "excluded": excluded, "ranking": "v2" if RANK_V2 else "v1",
                "stats": {"candidates": len(cand), "excluded": len(excluded), "shown": len(signals) + len(more),
                          "confident_75": confident, "sources": sum(len(s_.get("sources") or []) for s_ in signals + more)},
                "meta": {"candidates": len(cand), "base_size": len(self.sig), "llm_calls": usage["calls"],
                         "llm_tokens": usage["tokens"], "llm_errors": usage["errors"][:3],
                         "took_s": round(time.time() - t0, 1)}}

    def card(self, query: str, i: int, rank_: int, cand: pd.DataFrame, subs: list[int] | None = None) -> dict:
        k = self.sig.key[i]
        c = self.cards.get(k, {})
        sel = self.select_cache.get((query, self.ph[i])) or {}
        strength = float(cand.strength.get(i, 0.0))
        v2: dict = {}
        if RANK_V2:
            row = self._row(i)
            strength = rank.weakness(row)
            v2 = {"axes": rank.axes_view(row), "parent_note": rank.parent_note(row),
                  "subsignals": [{"term_en": self.ph[j], "name": self.sig.name_ru[j]} for j in (subs or [])],
                  "confidence": rank.confidence(row, self.calibration)}
            la = self.llm_axes.get(k) or {}
            if (la.get("abstract") and la.get("quote_ok")      # абстракт печатается только с проверенной цитатой
                    and not _BROKEN.search(la["abstract"] + " " + (la.get("mechanism") or ""))):
                v2["abstract"] = {"text": la["abstract"], "mechanism": la.get("mechanism"), "quote": la.get("quote"),
                                  "first_work": la.get("first_work"), "parent_llm": la.get("parent"),
                                  "kind_llm": la.get("kind"),
                                  "why_not_mature": [w for w in la.get("why_not_mature") or [] if not _BROKEN.search(w)],
                                  "generated": True, "model": la.get("model")}
        explain = c.get("explain") or []
        num = lambda v: ("+" if v >= 0 else "−") + f"{abs(v):.2f}".replace(".", ",")
        explanation = (f"Сила сигнала {round(strength * 100)} из 100 (итоговый балл двух моделей; кандидатов по запросу: {len(cand)}). "
                       f"Главные вклады в балл экспертной модели (SHAP): " +
                       "; ".join(f"{x['label'][:1].lower() + x['label'][1:]} {num(x['value'])}" for x in explain[:5]) + ".") if explain else ""
        txt = lambda f: (c.get(f) or {}).get("text") if isinstance(c.get(f), dict) else None
        return {
            "rank": rank_,
            **v2,
            "name": sel.get("headline") or c.get("name_ru") or self.sig.name_ru[i],
            "term_en": c.get("name") or self.ph[i],
            "description": txt("description") or self.sig.description_ru[i],
            "advantage": txt("advantage"),
            "case": txt("case"),
            "why_signal": sel.get("why"),
            "score": round(strength, 3),
            "explanation": explanation,
            "sources": [{"name": s.get("source_name"), "title": s.get("title"), "url": s.get("url"),
                         "date": s.get("published"), "type": s.get("source_type"), "language": s.get("language"),
                         "trust": s.get("trust")} for s in (c.get("sources") or [])],
            "series": c.get("series"), "ladder": c.get("ladder"), "facts": c.get("facts"),
            "shap": {"expert": explain, "history": c.get("explain_history")},
            "actors": c.get("actors"),
            "twin": c.get("twin"),
            "passport": c.get("passport"), "kind": c.get("kind"), "explain_rule": c.get("explain_rule"),
            "verification": ({k: sel.get(k) for k in ("sources_confirm", "quote_ok", "early", "mass", "quote", "url")}
                             if sel.get("verified") else None),
        }


_REG: Registry | None = None


def registry() -> Registry:
    global _REG
    if _REG is None:
        _REG = Registry()
    return _REG
