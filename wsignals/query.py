"""Открытый запрос: свободный текст -> ТОП-15 слабых сигналов с объяснением, источниками и причинами исключений.

    python -m wsignals.query "слабые сигналы в кибербезопасности"
    python -m wsignals.query "перспективные решения в финтехе" --top 15 --candidates 30 --json out.json

Конвейер:
  1. Разбор запроса: план поиска от явно заданной модели YandexGPT либо детерминированный словарь направлений.
  2. Сбор корпуса запроса: свежие работы OpenAlex 2025–2026 и базовый срез 2019–2022.
  3. Кандидаты: фразы 2–4 слова, которые резко участились в свежем срезе (лексический детектор всплеска),
     затем отбраковка нерелевантных и канонизация названия.
  4. Наблюдения пяти открытых источников с явным статусом OK / NO_RESULTS / ERROR.
  5. Решение по четырём осям: появление, зрелость, качество доказательства, риск хайпа.
     Вероятность модели этапа 1 остаётся как предиктор, но больше не решает вопрос включения.
  6. Карточка сигнала: описание, преимущество, кейс, источники с происхождением и уровнем доверия.

Весь живой путь ограничен общим бюджетом времени: частичный успех лучше, чем отказ всего запроса.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import claim_relevance, decompose, evidence, intent, llm, relevance, ru, takeoff, timemachine, trust  # noqa: F401
from . import provenance
from .decision import (ACCEPT, DETERMINISTIC_ONLY, HYPE, LOW_EVIDENCE, MATURE, MIN_EMERGENCE, NOISE,
                       SEMANTIC_UNCERTAIN, SEMANTICALLY_VERIFIED, decide)
from . import evidence
from .evidence import EvidenceState, Observation, observe
from .features import FEATURES, NAMES_RU
from .schema import Document
from .sources import github, hackernews, news, openalex
from .http import (DEADLINE, MIN_INTERVAL, FetchError, adopt_context, current_context, fetch_context, get_json,
                   phase, time_left, too_late_to_start as http_too_late)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "signal_model.joblib"
# Порог второго эшелона выбран по машине времени (reports/timemachine.md): при 0,35 модель в 2021 году отметила бы
# 39 % будущего мейнстрима при 11 % ложных срабатываний на уже зрелых технологиях; при 0,5 только 25 %.
# Пороги вероятности модели этапа 1. С ремонтом они больше не решают вопрос включения: решение принимает
# слой decision.py по четырём осям, а вероятность модели остаётся отдельным предиктором в карточке.
SIGNAL, WATCH = 0.5, 0.35
# Живой путь: цель 60 с, жёсткая стена 90 с по общему времени ответа (REPAIR E).
#
#     hard_wall_budget_s   = что видит пользователь; верхняя граница всего ответа
#     finalization_margin_s = резерв на сборку карточек, запись и сериализацию — без сети
#     network_deadline_s   = hard_wall − margin; после него ни один запрос не стартует
#
# Раньше дедлайн равнялся всему бюджету, сборка карточек шла уже сверх него, а таймаут запроса имел
# пол в 0,5 с — поэтому 90 с не были гарантией. Теперь сеть заканчивается заранее, а финализация
# укладывается в резерв и сама сетевой работы не начинает.
TARGET_SECONDS = float(os.getenv("WSIGNALS_QUERY_TARGET", "60"))
HARD_BUDGET_SECONDS = float(os.getenv("WSIGNALS_QUERY_BUDGET", "90"))
FINALIZATION_MARGIN_S = float(os.getenv("WSIGNALS_FINALIZATION_MARGIN", "8"))


def budget_plan(hard_wall: float | None = None) -> dict:
    """Три числа бюджета. Документация не должна обещать больше, чем даёт код, поэтому они считаются
    в одном месте и отдаются наружу как есть."""
    wall = HARD_BUDGET_SECONDS if hard_wall is None else hard_wall
    margin = min(FINALIZATION_MARGIN_S, max(1.0, wall * 0.2))
    return {"hard_wall_budget_s": round(wall, 2), "finalization_margin_s": round(margin, 2),
            "network_deadline_s": round(max(1.0, wall - margin), 2)}
# R2-A. Единственная авторитетная терминальная классификация — `decision`. Человекочитаемые `tier`
# и `status` ВЫВОДЯТСЯ из неё и второго классификатора в себе не несут. Раньше `status` считался по
# собственному порогу, и в полосе появления [MIN_EMERGENCE, WATCHLIST_EMERGENCE) выдача противоречила
# сама себе: decision=ACCEPT, tier=подтверждённый, status=под наблюдением.
TERMINAL_STATE = {
    ACCEPT:       ("подтверждённый слабый сигнал", "слабый сигнал"),
    LOW_EVIDENCE: ("требует дополнительных доказательств", "под наблюдением"),
    MATURE:       ("исключён как зрелая технология", "исключён"),
    HYPE:         ("исключён как маркетинговый шум", "исключён"),
    NOISE:        ("исключён как шум", "исключён"),
}
WEAK_SIGNAL_STATUS = TERMINAL_STATE[ACCEPT][1]
WATCHLIST_STATUS = TERMINAL_STATE[LOW_EVIDENCE][1]
# Порог подачи, оставленный как объявленная настройка в отпечатке конфигурации. Терминальную
# классификацию он больше НЕ определяет и в текущем коде ни на что не влияет: порядок выдачи задаётся
# rank_score и слоем разнообразия. Значение не меняется (R2-A: пороги не тюнятся).
WATCHLIST_EMERGENCE = 0.30


def terminal_state(decision: str) -> tuple[str, str]:
    """(tier, status) по авторитетному решению. Другого источника терминального статуса нет."""
    return TERMINAL_STATE.get(decision, ("исключён", "исключён"))
# Доли общего бюджета по этапам. Решающий третий этап получает остаток, но не меньше 30 %:
# без этого сбор корпуса и дешёвые проверки съедают весь бюджет и выдача оказывается пустой.
BUDGET_SHARE = {"corpus": 0.32, "stage1": 0.18, "stage2": 0.16}
# Ширина каскада на живом пути. Узкое место — поиск GitHub: 30 запросов в минуту, то есть 2,2 с на запрос
# и 2 запроса на кандидата. Шире этих чисел третий этап в 90 секунд физически не помещается.
LIVE_KEEP_WIKI = int(os.getenv("WSIGNALS_LIVE_KEEP_WIKI", "24"))
LIVE_KEEP_FULL = int(os.getenv("WSIGNALS_LIVE_KEEP_FULL", "14"))


def stage3_width(top: int) -> int:
    """Сколько кандидатов ОДНОГО запроса допускается к третьему этапу.

    Третий этап — единственное место, где измеряются новости, а без них хайп не измерен и ACCEPT
    закрыт. Каскад работает по ВСЕМУ запросу одним списком (поисковые фразы сливаются в один
    корпус, отдельных рабочих наборов по темам нет), поэтому LIVE_KEEP_FULL был потолком числа
    подтверждённых на весь запрос: при 14 < 15 ТОП-15 был недостижим даже при 15 и более готовых
    кандидатах. Ширина не может быть меньше запрошенного ТОПа. Время по-прежнему ограничивает
    `_pass`: порция не начинается, если на неё не осталось бюджета."""
    return max(LIVE_KEEP_FULL, int(top))


# Сколько секунд резервируется на одного кандидата третьего этапа перед запуском следующей порции.
STAGE3_SECONDS_PER_CANDIDATE = float(os.getenv("WSIGNALS_STAGE3_SECONDS", "3"))
# Счётчики поиска GitHub стоят 2 запроса на кандидата при лимите 30 запросов в минуту: 4,4 с на кандидата.
# На живом пути они выключены, а семейство «код» измеряется по репозиториям, уже полученным при сборе
# корпуса. Включается для офлайн-сбора и для сверки с моделью этапа 1.
LIVE_GITHUB_STATS = os.getenv("WSIGNALS_LIVE_GITHUB_STATS", "0") == "1"
# Сколько принятых кандидатов дообогащаются русскими и англоязычными новостями (по 2 запроса, 1 с каждый).
ENRICH_TOP = int(os.getenv("WSIGNALS_ENRICH_TOP", "8"))

# Термины направлений: русский -> английские поисковые фразы. Покрывают формулировки ТЗ и области таблицы.
DIRECTIONS = {
    r"\bии\b|искусственн\w* интеллект|нейросет|машинн\w* обучени|генеративн": ["artificial intelligence", "large language models", "AI agents", "generative AI"],
    r"кибербез|информационн\w* безопасн|защит\w* ии|безопасн\w* ии": ["cybersecurity", "AI security"],
    r"финтех|финанс|банк|платеж|платёж": ["fintech", "payments", "banking", "KYC"],
    r"робот": ["robotics"],
    r"\bedge\b|периферийн|граничн\w* вычислен|на устройств": ["edge AI", "on-device AI"],
    r"инфраструктур\w* ии|цод|дата-центр|центр\w* обработки данных|вычислительн\w* инфраструктур": ["AI infrastructure", "data center"],
    r"промышлен|производств|индустриальн|завод": ["industrial AI", "manufacturing"],
    r"энергет|электросет|аккумулятор|водород": ["energy technology", "energy storage"],
    r"медицин|здравоохран|биотех": ["medical AI", "biotechnology"],
    r"квант": ["quantum computing", "quantum technology"],
    r"телеком|связ\w+ 6g|\b6g\b|5g": ["telecommunications", "6G"],
    r"блокчейн|крипт|токениз": ["blockchain", "tokenization"],
    r"космос|спутник": ["space technology", "satellite"],
    r"агент": ["AI agents"],
}
FILLER = re.compile(r"(слаб\w+ сигнал\w*|перспективн\w+ решени\w*|зарождающ\w+ тренд\w*|технологи\w+ в|технологи\w+|"
                    r"решени\w+ в|в области|в сфере|направлени\w+|новые|новых|тренды|тренд)", re.I)
INNER_STOP = set("""integrates integrating integrated leveraging leverages actionable insights contains repository capabilities
systematic mapping meta analysis literature review reviews including includes enables enabling provides providing using uses based
significant potential various several key novel proposed existing current recent state art paper study studies findings results
with and or for built using across by from in on of the to via official implementation open source real
time complex tasks dynamic environments professionals track anonymously curated awesome collection list simple easy""".split())
RU_STOP = set("в во и для на по о об с со к ко у из за от до при про над под а но или как что это".split())
BACKGROUND = ROOT / "data" / "labels" / "background_phrases.json"
BACKGROUND_SEEDS = ["clinical trial", "materials science", "education", "agriculture", "urban planning", "tourism",
                    "psychology", "hydrology", "public policy", "marketing", "linguistics", "construction engineering"]
STOP = set("""more most less before after they them your my our their you we us i me his her its false true zero a an the of for and or in on to with by from at as is are was were be been being this that these those it its
into via using based towards toward over under between within without new novel approach approaches study analysis
method methods framework frameworks model models system systems paper review survey results case use role impact
large language data learning deep machine artificial intelligence ai llm llms towards through their our we can
application applications challenges opportunities future perspective perspectives comprehensive toward how what why""".split())
GENERIC_HEADS = {"security", "technology", "technologies", "solution", "solutions", "research", "development", "issues",
                 "performance", "evaluation", "management", "detection", "network", "networks", "algorithm", "algorithms"}
STANDARD = re.compile(r"\b(owasp|nist|iso|iec|ieee|rfc|mitre att|att ck|mitre atlas|pci dss|gdpr|hipaa|soc 2|fedramp|ai act|dora|nis2|"
                      r"csf|cis controls|swift csp|psd2|iso 20022|basel)\b", re.I)
ADVANTAGE = re.compile(r"\b(reduc\w+|improv\w+|enabl\w+|lower\w*|faster|outperform\w*|increas\w+|without|cheaper|"
                       r"accurac\w+|efficien\w+|scal\w+|protect\w+|prevent\w+|automat\w+)\b", re.I)


# ---------- 1. разбор запроса ----------

def parse_query(text: str) -> list[str]:
    """Детерминированный разбор. Работает по ПРЕДМЕТНОЙ ОБЛАСТИ запроса, а не по словам о задаче
    поиска: тот же контракт, что и у модельного пути (REPAIR A)."""
    t = intent.split(text).domain.lower()
    seeds = [s for pat, terms in DIRECTIONS.items() if re.search(pat, t) for s in terms]
    rest = FILLER.sub(" ", t)
    for pat in DIRECTIONS:
        rest = re.sub(rf"\w*(?:{pat})\w*", " ", rest)
    rest = re.sub(r"[^\w\s-]", " ", rest)
    words = [w for w in rest.split() if w not in RU_STOP]
    rest = " ".join(words)
    if words and re.search(r"[a-z]", rest) and not re.search(r"[а-я]", rest):
        seeds.append(rest)
    elif any(len(w) >= 4 for w in words):
        seeds.append(llm.translate_query(rest))
    return list(dict.fromkeys(s for s in seeds if s and len(s) > 1))[:4]


# ---------- 2–3. корпус запроса и кандидаты ----------

TECH_DOMAINS = "3"                 # OpenAlex Physical Sciences: информатика, инженерия, материалы, энергетика
HEALTH_DOMAINS = "1|4"             # Life Sciences и Health Sciences для медицинских запросов


def _works(seed: str, years: str, n: int, domains: str = TECH_DOMAINS) -> list[dict]:
    p = {"filter": f'title_and_abstract.search:"{seed}",publication_year:{years},primary_topic.domain.id:{domains}', "per-page": n,
         "sort": "relevance_score:desc", "select": "id,doi,title,publication_date,abstract_inverted_index,primary_location,type"}
    return get_json(openalex.API, {**openalex._base(seed), **p}).get("results", [])


def _text(w: dict) -> str:
    inv = w.get("abstract_inverted_index") or {}
    abstract = " ".join(t for _, t in sorted((i, t) for t, idx in inv.items() for i in idx))
    return f"{w.get('title') or ''}. {abstract}"


def _phrases(text: str) -> set[str]:
    toks = re.findall(r"[a-z][a-z0-9\-]+", text.lower())
    out = set()
    for n in (2, 3, 4):
        for i in range(len(toks) - n + 1):
            g = toks[i:i + n]
            if g[0] in STOP or g[-1] in STOP or g[-1] in GENERIC_HEADS or any(w in INNER_STOP for w in g):
                continue
            if sum(w not in STOP for w in g) < 2:
                continue
            out.add(" ".join(g))
    return out


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()[:120]


def _dedupe(works: list[dict]) -> list[dict]:
    seen, out = set(), []
    for w in works:
        k = _norm_title(w.get("title"))
        if k and k not in seen:
            seen.add(k)
            out.append(w)
    return out


def background_phrases() -> set[str]:
    """Штампы научного языка: фразы, частые в работах по несвязанным областям. Считается один раз и кэшируется в файл."""
    if BACKGROUND.exists():
        return set(json.loads(BACKGROUND.read_text(encoding="utf-8")))
    df, n = Counter(), 0
    for seed in BACKGROUND_SEEDS:
        for w in _dedupe(_works(seed, "2024-2026", 150)):
            n += 1
            df.update(_phrases(_text(w)))
    stamps = sorted(ph for ph, c in df.items() if c / max(1, n) >= 0.004)
    BACKGROUND.write_text(json.dumps(stamps, ensure_ascii=False, indent=0), encoding="utf-8")
    return set(stamps)


GENERIC_MODIFIERS = set("""ai emerging advanced enhancing enhanced ai-driven ai-powered ai-based ai-enabled intelligent smart modern
future next-generation effective robust novel innovative threats threat challenges risks risk trends landscape techniques strategies
solutions practices framework systems approach overview towards digital""".split())
TOPIC_STOP = set("""ai llm llms ml machine-learning deep-learning artificial-intelligence python javascript typescript go golang rust java
react nextjs docker kubernetes openai chatgpt gpt claude anthropic gemini langchain pytorch tensorflow api cli tool tools framework
library awesome awesome-list list security cybersecurity hacking hacktoberfest data-science nlp computer-vision generative-ai genai
agents agent ai-agents llm-agents automation research tutorial course book dataset benchmark mcp-server server web app application
open-source opensource self-hosted chatbot bot assistant copilot rag vector-database""".split())


def _new_repos(seed: str, pages: int = 2) -> tuple[list[dict], str | None]:
    """Новые репозитории за последний год. Возвращает (репозитории, класс ошибки или None).

    Отказ здесь больше не молчит: семейство «код» в решении измеряется именно по этой выборке,
    поэтому «GitHub не ответил» и «репозиториев нет» обязаны различаться."""
    since = (date.today().replace(year=date.today().year - 1)).isoformat()
    items, error = [], None
    for page in range(1, pages + 1):
        try:
            # P1-A. Запрос идёт через канонический адаптер, а не напрямую через get_json.
            #
            # Живая проверка владельца показала цену прямого вызова: переменная окружения содержала
            # просроченный токен, GitHub ответил 401, и весь корпус кода отпадал целиком — хотя
            # неавторизованный доступ к тому же поиску отвечал 200. Повтор без токена уже был
            # реализован в `github.fetch`, но этот путь его не использовал, и на живом пути
            # (`WSIGNALS_LIVE_GITHUB_STATS=0`) `fetch` не вызывался вообще. Второй реализации
            # повтора здесь нет и быть не должно.
            d = github.fetch({"q": f'"{seed}" created:>{since}', "sort": "stars", "per_page": 100,
                              "page": page}, max_age_days=3)
        except FetchError as e:
            error = e.error_type
            break
        except Exception as e:  # noqa: BLE001
            error = type(e).__name__
            break
        items += d.get("items", [])
        if len(d.get("items", [])) < 100:
            break
    return items, error


def _news_titles(seeds: list[str]) -> tuple[list[str], list[str]]:
    """Заголовки о запусках и раундах и СПИСОК ОТКАЗОВ. Запросов намеренно мало: лента Google News
    троттлится по секунде на запрос, а на живом пути это прямой вычет из бюджета решающего этапа.

    Отказы возвращаются наружу наравне с заголовками. Раньше здесь стоял глухой `except Exception:
    return []`, и пустой результат означал одновременно «лента ответила и ничего не нашла» и «ленту
    не удалось получить или разобрать». Живой прогон показал цену: три широких запроса подряд дали
    `news = NO_RESULTS`, то есть ОТВЕТ ИСТОЧНИКА, хотя ни один ответ не был ни получен, ни разобран.
    Это прямо противоречит правилу PHASE 1 «ERROR это не ноль и не ответ»."""
    ctx = current_context()
    queries = [f"{s} {suffix}" for s in seeds[:2] for suffix in ("startup raises", "unveils")]

    def fetch(q):
        with adopt_context(ctx):
            try:
                return [d.title for d in news.search(q, lang="en", when="6m", limit=100, exact=False)], None
            except FetchError as e:
                # Отказ транспорта: HTTP, 429, сеть, исчерпанный дедлайн. Класс ошибки уже типизирован.
                return [], f"{q}: {e.error_type}"
            except ET.ParseError:
                # Лента получена, но это не RSS: страница блокировки, согласия на cookie или ошибки.
                return [], f"{q}: parse"
            except Exception as e:  # noqa: BLE001 — сбор корпуса не должен валить запрос
                return [], f"{q}: {type(e).__name__}"

    titles, errors = [], []
    with ThreadPoolExecutor(min(4, len(queries) or 1)) as ex:
        for found, error in ex.map(fetch, queries):
            titles += found
            if error:
                errors.append(error)
    return list(dict.fromkeys(titles)), errors


BACKGROUND_COUNTS = ROOT / "data" / "labels" / "background_counts.json"
KEYNESS_MIN = math.log(3)        # фраза должна встречаться в корпусе запроса минимум втрое чаще, чем в фоне


def background_counts() -> tuple[int, Counter]:
    """Частоты фраз в работах по несвязанным областям: медицина, образование, гидрология, лингвистика и другие.
    Считается один раз и кэшируется в файл."""
    if BACKGROUND_COUNTS.exists():
        d = json.loads(BACKGROUND_COUNTS.read_text(encoding="utf-8"))
        return d["n_docs"], Counter(d["df"])
    df, n = Counter(), 0
    for seed in BACKGROUND_SEEDS:
        for w in _dedupe(_works(seed, "2024-2026", 150)):
            n += 1
            df.update(_phrases(_text(w)))
    df = Counter({k: v for k, v in df.items() if v >= 2})
    BACKGROUND_COUNTS.write_text(json.dumps({"n_docs": n, "df": df}, ensure_ascii=False), encoding="utf-8")
    return n, df


def keyness(ph: str, df_query: int, n_query: int, n_bg: int, df_bg: Counter) -> float:
    """Логарифм отношения частоты фразы в корпусе запроса к частоте в фоновом корпусе. Развитие идеи энтропийного
    фильтра из исследовательской версии: термин направления сконцентрирован в своей области, а штамп научного
    языка («have explored», «unlike traditional») встречается с одинаковой частотой где угодно."""
    return math.log((df_query / max(1, n_query) + 1e-4) / (df_bg.get(ph, 0) / max(1, n_bg) + 1e-4))


def extract_candidates(seeds: list[str], k: int = 30) -> tuple[list[dict], dict]:
    """Кандидаты из нескольких каналов появления технологии. Слабый сигнал редко виден в одном канале целиком:
    его называют в тегах новых репозиториев, в заголовках препринтов, в новостях о раундах. Появление в разных
    каналах даёт бонус, фраза из одного канала должна быть в нём частой.

    PHASE 6: отказ OpenAlex на этой стадии больше не роняет весь запрос. Каждый срез берётся отдельно,
    отказ записывается в журнал адаптеров и виден в статистике запроса."""
    recent, prior = [], []
    adapters = {"openalex": "OK", "github": "OK", "news": "OK"}
    oa_errors: list[str] = []
    domains = HEALTH_DOMAINS if any(re.search(r"medical|bio|health|clinical", s, re.I) for s in seeds) else TECH_DOMAINS
    # Срезы корпуса берутся параллельно. Последовательный цикл был скрытым узким местом: при повторах
    # (429 от OpenAlex) восемь запросов подряд выбирали весь бюджет этапа и корпус получался пустым.
    # Троттлинг по хосту всё равно соблюдается: он общий для всех потоков.
    ctx = current_context()
    jobs = [(s, window) for s in seeds for window in ("2025-2026", "2019-2022")]

    def fetch(job):
        seed, window = job
        with adopt_context(ctx):
            try:
                return window, _works(seed, window, 150, domains), None
            except FetchError as e:
                return window, [], f"{seed} {window}: {e.error_type}"
            except Exception as e:  # noqa: BLE001 — сбор корпуса не должен валить запрос
                return window, [], f"{seed} {window}: {type(e).__name__}"

    def collect_works():
        with ThreadPoolExecutor(min(8, len(jobs))) as ex:
            return list(ex.map(fetch, jobs))

    def collect_repos():
        found, errors = [], []
        for seed in seeds[:3]:
            try:
                items, error = _new_repos(seed, pages=1 if len(seeds) > 2 else 2)
            except Exception as e:  # noqa: BLE001 — сбор корпуса не должен валить запрос
                items, error = [], type(e).__name__
            found += items
            if error:
                errors.append(f"{seed}: {error}")
        return found, errors

    # Три источника корпуса опрашиваются одновременно и на разных хостах. Раньше они шли подряд,
    # и мёртвый OpenAlex выбирал весь бюджет этапа, не оставляя ни репозиториев, ни новостей.
    with ThreadPoolExecutor(3) as ex:
        f_works = ex.submit(lambda: _with_ctx(ctx, collect_works))
        f_repos = ex.submit(lambda: _with_ctx(ctx, collect_repos))
        f_news = ex.submit(lambda: _with_ctx(ctx, lambda: _news_titles(seeds)))
        for window, works, error in f_works.result():
            (recent if window.startswith("2025") else prior).extend(works)
            if error:
                oa_errors.append(error)
        repos, repo_errors = f_repos.result()
        headlines, news_errors = f_news.result()
    if oa_errors:
        adapters["openalex"] = "ERROR" if not recent else "PARTIAL"
    recent, prior = _dedupe(recent), _dedupe(prior)
    stamps = background_phrases()
    seed_tokens = {t for s in seeds for t in re.findall(r"[a-z0-9]+", s.lower())} - {"ai"}

    channels: dict[str, Counter] = {"литература": Counter(), "препринты": Counter(), "репозитории": Counter(),
                                    "новости": Counter()}
    examples = defaultdict(list)
    df_p = Counter()
    for w in prior:
        df_p.update(_phrases(_text(w)))
    in_title = Counter()
    for w in recent:
        in_title.update(_phrases(w.get("title") or ""))
        for ph in _phrases(_text(w)):
            channels["литература"][ph] += 1

            if len(examples[ph]) < 3:
                examples[ph].append(w)
        if w.get("type") == "preprint":
            channels["препринты"].update(_phrases(w.get("title") or ""))
    repos = list({r["full_name"]: r for r in repos}.values())
    if repo_errors:
        adapters["github"] = "ERROR" if not repos else "PARTIAL"
    elif not repos:
        adapters["github"] = "NO_RESULTS"
    repo_examples = defaultdict(list)
    for r in repos:
        topics = {t.replace("-", " ") for t in r.get("topics", []) if t not in TOPIC_STOP and "-" in t
                  and not any(w in INNER_STOP for w in t.split("-"))}
        found = topics | _phrases(r.get("description") or "")
        channels["репозитории"].update(found)
        for ph in found:
            if len(repo_examples[ph]) < 2:
                repo_examples[ph].append(r)
    # Статус новостного среза корпуса по тем же правилам, что у остальных адаптеров (PHASE 1):
    # NO_RESULTS присваивается ТОЛЬКО когда лента была получена и разобрана, а валидных записей в
    # ней не оказалось. Любой отказ транспорта или разбора — это ERROR (или PARTIAL, если часть
    # запросов всё же принесла заголовки), и ответом источника он не считается.
    if news_errors:
        adapters["news"] = "ERROR" if not headlines else "PARTIAL"
    elif not headlines:
        adapters["news"] = "NO_RESULTS"
    for t in headlines:
        channels["новости"].update(_phrases(t))

    n_r, n_p = max(1, len(recent)), max(1, len(prior))
    phrases = set().union(*channels.values())
    n_bg, bg_df = background_counts()
    stamped = 0

    # REPAIR B. Порядок исправлен: сначала собираются ВСЕ прошедшие дешёвые фильтры фразы, затем
    # варианты одного явления склеиваются, и только после этого применяется порог по числу каналов.
    # Раньше порог стоял до склейки и убивал раздробленное свидетельство: «mcp server» и «mcp servers»
    # по отдельности не набирали трёх упоминаний и отсеивались оба, хотя вместе набирали.
    raw: list[dict] = []
    for ph in phrases:
        toks = ph.split()
        lit_n = channels["литература"].get(ph, 0)
        if lit_n >= 3 and bg_df.get(ph, 0) >= 2 and keyness(ph, lit_n, n_r, n_bg, bg_df) < KEYNESS_MIN:
            stamped += 1
            continue                                  # штамп: так же част в работах по несвязанным областям
        if ph in stamps or set(toks) <= seed_tokens | GENERIC_MODIFIERS:
            continue                                  # пересказ самой темы: «enhancing cybersecurity», «ai-driven security»
        present = {name: c[ph] for name, c in channels.items() if c.get(ph, 0) > 0}
        lit = channels["литература"].get(ph, 0)
        families = len({"наука" if n in ("литература", "препринты") else n for n in present})
        if lit >= 3 and in_title.get(ph, 0) == 0 and families == 1:
            continue                                  # фраза только из аннотаций: чаще штамп, чем технология
        burst_raw = math.log((lit / n_r + 0.005) / (df_p.get(ph, 0) / n_p + 0.005)) if lit else 0.0
        raw.append({"phrase": ph, "original_phrase": ph, "canonical_label": ph,
                    "burst": round(max(burst_raw, 0.0), 3), "recent_docs": lit,
                    "prior_docs": df_p.get(ph, 0), "channels": present,
                    "examples": examples.get(ph, []), "repos": repo_examples.get(ph, [])})

    families_before = len(raw)
    raw, fragmented = relevance.collapse(raw)          # склейка вариантов ДО порога по каналам
    survived_gate: list[dict] = []
    for c in raw:
        present = c.get("channels") or {}
        families = len({"наука" if n in ("литература", "препринты") else n for n in present if present[n] > 0})
        if sum(present.values()) < 3 and families < 2:
            continue                                  # слабый след даже после объединения вариантов
        lit = c.get("recent_docs", 0)
        burst = c.get("burst", 0.0)
        score = (burst * math.log1p(lit) + 1.2 * math.log1p(present.get("репозитории", 0))
                 + 0.8 * math.log1p(present.get("препринты", 0))
                 + 0.5 * math.log1p(present.get("новости", 0))) * (1 + 0.5 * (families - 1))
        if score > 0:
            c["burst"] = round(score, 2)
            survived_gate.append(c)
    survived_gate.sort(key=lambda c: -c["burst"])
    scored = survived_gate
    picked: list[dict] = []
    norm = lambda x: x.replace("-", " ")                # «on-policy distillation» и «on policy distillation» один кандидат
    for c in survived_gate:
        ph = c["phrase"]
        if any(norm(ph) in norm(q["phrase"]) or norm(q["phrase"]) in norm(ph) for q in picked):
            continue
        picked.append(c)
        if len(picked) >= k:
            break
    stats = {"работ за 2025–2026": len(recent), "работ за 2019–2022": len(prior), "новых репозиториев": len(repos),
             "заголовков новостей": len(headlines), "фраз-кандидатов": len(scored), "отсеяно как штампы": stamped,
             "фраз до склейки": families_before, "склеено вариантов до порога": len(fragmented),
             "адаптеры корпуса": adapters,
             # Класс каждого отказа корпуса (строки отказов выше оканчиваются на «: <класс>»). Нужен,
             # чтобы исчерпанный собственный бюджет (deadline) не выдавался за отказ источника.
             "типы отказов корпуса": {name: sorted({e.rsplit(": ", 1)[-1] for e in errs})
                                      for name, errs in (("openalex", oa_errors), ("github", repo_errors),
                                                         ("news", news_errors)) if errs}}
    if oa_errors:
        stats["отказы OpenAlex на сборе корпуса"] = oa_errors[:10]
    if repo_errors:
        stats["отказы GitHub на сборе корпуса"] = repo_errors[:10]
    if news_errors:
        stats["отказы новостей на сборе корпуса"] = news_errors[:10]
    return picked, stats


# ---------- 4–5. скоринг и фильтр мейнстрима ----------

def novelty(r: dict, channels: dict) -> float:
    """Новизна кандидата по глобальному годовому ряду публикаций: молодой возраст, рост, небольшой, но ненулевой объём,
    плюс след в новых репозиториях. Зрелые и пустые фразы получают низкий балл.

    Считается только по ответившим источникам: если OpenAlex не ответил, пустой словарь сюда не подставляется."""
    total, age, growth = r.get("oa_total", 0), r.get("oa_age", 0), r.get("oa_growth", 0.0)
    code = math.log1p(channels.get("репозитории", 0))
    size = -abs(math.log1p(total) - math.log1p(150))           # «золотая середина» около 150 работ
    return 1.5 * max(growth, 0) - 0.25 * age + 0.4 * size + 0.8 * code + (0.5 if 0 < total < 3000 else 0)


def _with_ctx(ctx, fn):
    """Выполняет функцию в рабочем потоке, унаследовав контекст запроса (дедлайн, журнал происхождения)."""
    with adopt_context(ctx):
        return fn()


# Коды отказа, которые могут измениться, если получить больше конкретных документов: именно таких
# кандидатов имеет смысл дообогащать новостями до окончательного решения (R2-C).
ENRICHABLE_CODES = {"supporting_evidence_not_materialized", "no_acceptable_source_quality",
                    "no_independent_corroboration",
                    # Документы есть, но ни один не про эту технологию: адресный запрос новостей
                    # по самой фразе кандидата способен изменить исход, поэтому кандидат обогащается.
                    "no_claim_relevant_evidence"}
# R2-G. Насколько понижается балл кандидата за каждого уже выбранного «родственника» по головному слову.
DIVERSITY_PENALTY = float(os.getenv("WSIGNALS_DIVERSITY_PENALTY", "0.55"))


def verification_status(assessment: dict | None) -> str:
    """Состояние семантической проверки кандидата (R2-A).

    Модель подтвердила технологию и связь с направлением — SEMANTICALLY_VERIFIED.
    Модели не было, но лексика чистая — DETERMINISTIC_ONLY.
    Модели не было и лексика неоднозначна — SEMANTIC_UNCERTAIN."""
    a = assessment or {}
    if a.get("source") == "llm" and a.get("is_technology") and not a.get("reject_reason"):
        if float(a.get("domain_relevance", 0.0)) >= relevance.MIN_DOMAIN_RELEVANCE:
            return SEMANTICALLY_VERIFIED
    return SEMANTIC_UNCERTAIN if a.get("ambiguity_flags") else DETERMINISTIC_ONLY


def sibling_key(cand: dict) -> str:
    """Ключ «родственной семьи» для разнообразия выдачи: нормализованное головное слово.

    Это НЕ ключ склейки. Родственники остаются разными технологиями и никогда не сливаются;
    ключ используется только при ранжировании, чтобы ТОП не состоял из вариаций одного слова."""
    words = re.findall(r"[a-z0-9]+", (cand.get("canonical_label") or cand["phrase"]).lower())
    return relevance._stem(words[-1]) if words else ""


def diversify(rows: list[dict], score_of) -> list[dict]:
    """Переупорядочивает принятых так, чтобы ТОП не был занят родственниками по головному слову (R2-G).

    Исходный балл не меняется: он остаётся в `rank_score`, а место в выдаче — в `diversity_rank`.
    Кандидаты не выдумываются и не выбрасываются, меняется только порядок."""
    remaining = sorted(rows, key=lambda r: -score_of(r))
    picked: list[dict] = []
    seen: dict[str, int] = {}
    while remaining:
        best, best_adjusted = None, None
        for r in remaining:
            adjusted = score_of(r) * (DIVERSITY_PENALTY ** seen.get(sibling_key(r["cand"]), 0))
            if best_adjusted is None or adjusted > best_adjusted:
                best, best_adjusted = r, adjusted
        remaining.remove(best)
        key = sibling_key(best["cand"])
        best["diversity_rank"] = len(picked) + 1
        best["diversity_adjusted_score"] = round(float(best_adjusted), 4)
        best["sibling_key"] = key
        seen[key] = seen.get(key, 0) + 1
        picked.append(best)
    return picked


def sibling_concentration(cands: list[dict]) -> float:
    """Доля кандидатов, чьё головное слово уже встречалось в списке. 0.0 — родственников нет."""
    if not cands:
        return 0.0
    keys = [sibling_key(c) for c in cands]
    return round(1 - len(set(keys)) / len(keys), 3)


def _corpus_code_observation(cand: dict, corpus_adapters: dict) -> Observation:
    """Семейство «код» по репозиториям, уже полученным при сборе корпуса запроса.

    `_new_repos` берёт только репозитории, созданные за последние 12 месяцев, поэтому их число по фразе
    это прямое измерение свежего инженерного следа. Дополнительных запросов не делается: на живом пути
    поиск GitHub ограничен 30 запросами в минуту и является главным узким местом по времени.

    Если сбор репозиториев в корпусе отказал, это ERROR, а не ноль."""
    status = corpus_adapters.get("github", "OK")
    if status == "ERROR":
        return Observation("github", evidence.ERROR, error_type="corpus_fetch_failed",
                           error="сбор новых репозиториев для корпуса запроса не удался")
    n = len(cand.get("repos") or []) or int((cand.get("channels") or {}).get("репозитории", 0))
    return Observation("github", evidence.OK if n else evidence.NO_RESULTS, {"gh_corpus_repos": n})


def _attachment(item: dict, cand: dict) -> dict:
    """Происхождение привязки документа к кандидату (P1-B).

    `attached_via` проставляет склейка тому свидетельству, которое пришло от поглощённого варианта.
    Своё свидетельство помечается собственной фразой кандидата. Дальше правило претензионной
    релевантности решает, даёт ли этот вариант право на подтверждение: более широкий поглощённый
    псевдоним его не даёт."""
    own = cand.get("phrase", "") or ""
    via = item.get("attached_via") or own
    return {"attachment_phrase": via,
            "attachment_normalized": " ".join(claim_relevance.tokens(via)),
            "attachment_source": "own" if via == own else "merged"}


def concrete_documents(cand: dict, retrieved_at: str | None, stale: bool = False) -> list[Document]:
    """Конкретные документы, подтверждающие кандидата. Все они уже получены при сборе корпуса запроса,
    поэтому проверка инварианта «ACCEPT требует документа» не стоит ни одного дополнительного запроса.

    Здесь же каждому документу проставляется претензионно-относительная релевантность. Точка выбрана
    намеренно: материализация идёт ПОСЛЕ склейки, поэтому документ, пришедший вместе с поглощённым
    широким вариантом, проверяется против ВЫЖИВШЕЙ канонической личности, а не против той фразы,
    по которой был когда-то найден."""
    papers = [Document(title=w.get("title") or "", url=w.get("doi") or w["id"],
                       source_name=((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "OpenAlex",
                       source_type="препринт" if w.get("type") == "preprint" else "научная публикация",
                       published=w.get("publication_date"), language=w.get("language") or "en",
                       snippet=_text(w)[:700],
                       **_attachment(w, cand)) for w in cand.get("examples", [])]
    repos = [Document(title=f"{r['full_name']}: {r.get('description') or ''}"[:200], url=r["html_url"],
                      source_name="GitHub", source_type="репозиторий",
                      published=(r.get("created_at") or "")[:10] or None, language="en",
                      extra={"stars": r.get("stargazers_count", 0)},
                      **_attachment(r, cand)) for r in cand.get("repos", [])]
    status = "STALE_CACHE" if stale else "OK"
    docs = [trust.assess(d).with_provenance(retrieved_at=retrieved_at, adapter_status=status)
            for d in papers + repos]
    return claim_relevance.annotate(docs, cand)


def _mapper(ctx, fn):
    """Оборачивает работу потока пула: thread-local контекст запроса (дедлайн, журнал) в пул не наследуется."""
    def run(arg):
        with adopt_context(ctx):
            return fn(arg)
    return run


def _by_host_cost(sources: dict[str, tuple]) -> dict:
    """Источники в порядке возрастания минимального интервала их хоста.

    Троттлинг в `http` — ГЛОБАЛЬНЫЙ сериализатор на хост: k-й запрос к хосту не может стартовать
    раньше, чем через (k−1)·MIN_INTERVAL. Поэтому внутри одного кандидата дешёвый хост обязан идти
    первым: иначе дорогой занимает окно, а дешёвый — и вместе с ним критичное для допуска измерение
    зрелости — не начинается вовсе. Значения берутся из уже объявленных интервалов, отдельной
    таблицы стоимостей и предметных исключений здесь нет."""
    return {name: fn for name, (fn, host) in
            sorted(sources.items(), key=lambda kv: MIN_INTERVAL.get(kv[1][1], 0.3))}


def cascade_collect(cands: list[dict], workers: int = 8, log=print, keep_wiki: int = 70, keep_full: int = 45,
                    budget: dict | None = None, corpus_adapters: dict | None = None,
                    with_github_stats: bool = True, timings: dict | None = None,
                    admission_workset: int | None = None):
    """Каскадная проверка кандидатов, от дешёвых признаков зрелости к дорогим.
    Ступень 1: годовой ряд OpenAlex (один быстрый запрос). Отсекает сформированные направления.
    Ступень 2: Википедия для лучших по новизне. Отсекает технологии с давней статьёй.
    Ступень 3: полные признаки (OpenAlex, новости, Hacker News, GitHub) для оставшихся.

    Каждое обращение к источнику возвращает наблюдение с явным статусом (PHASE 1). Отказ источника
    не превращается в ноль: он не даёт вето, не даёт подтверждения и снижает качество доказательства.
    """
    from .features import derive
    from .sources import wikipedia

    budget = budget or {}
    corpus_adapters = corpus_adapters or {}
    timings = timings if timings is not None else {}
    ctx = current_context()
    t_stage = time.perf_counter()
    # C. Терминальные кандидаты не получают дорогих запросов.
    #
    # Измерено: ступень 1 запрашивала OpenAlex для КАЖДОГО кандидата, включая уже отбракованных
    # детерминированно — семантический отказ и склеенные варианты приходят сюда в общем списке.
    # Вернуться к допуску они не могут ни при каком ответе источника, поэтому запрос по ним —
    # чистая трата бюджета, которого потом не хватает на критичные для допуска измерения.
    #
    # Пропускаются ТОЛЬКО те, чьё состояние уже не может измениться: отбракованные семантикой и
    # отраслевые стандарты. LOW_EVIDENCE и SEMANTIC_UNCERTAIN не пропускаются никогда.
    terminal = [bool(c.get("noise_reason")) or bool(STANDARD.search(c["phrase"])) for c in cands]
    evaluated = [c for c, skip in zip(cands, terminal) if not skip]
    skipped = sum(terminal)
    with phase(budget.get("stage1")), ThreadPoolExecutor(workers) as ex:
        s1 = list(ex.map(_mapper(ctx, lambda q: observe("openalex", openalex.stats, q, light=True)),
                         [c["phrase"] for c in evaluated]))
    by_phrase = {id(c): obs for c, obs in zip(evaluated, s1)}
    rows = []
    for c, skip in zip(cands, terminal):
        state = EvidenceState()
        # У пропущенного кандидата наблюдения НЕТ вовсе: отсутствие измерения нельзя путать ни
        # с нулём, ни с отказом провайдера (F).
        if not skip:
            state.add(by_phrase[id(c)])
        row = {"cand": c, "state": state}
        # Лёгкий запрос ступени 1 даёт годовой ряд, но НЕ даёт полей широты, доли препринтов и
        # доли компаний: они возвращаются нулями, неотличимыми от измеренных. Пока полный опрос
        # OpenAlex не выполнен, зрелость измерена не до конца (P1).
        row["openalex_full"] = False
        if skip:
            row["not_evaluated"] = "terminal_before_retrieval"
        rows.append(row)
    timings["stage1_skipped_terminal"] = skipped
    for row in rows:
        state, cand = row["state"], row["cand"]
        v = state.values()
        if cand.get("noise_reason"):
            row["stage"], row["decision"] = 0, NOISE
            row["reasons"], row["reason_code"] = [cand["noise_reason"]], "semantic_rejection"
        elif STANDARD.search(cand["phrase"]):
            row["stage"], row["decision"] = 1, NOISE
            row["reasons"] = ["отраслевой стандарт или регуляторная рамка: по ТЗ не включается в слабые сигналы"]
            row["reason_code"] = "industry_standard"
        elif state.ok("openalex") and v.get("oa_total", 0) >= 3000 and v.get("oa_age", 0) >= 8:
            row["stage"], row["decision"] = 1, MATURE
            row["reasons"] = [f"сформированное научное направление: {int(v['oa_total'])} публикаций, "
                              f"первые работы {int(v['oa_age'])} лет назад"]
            row["reason_code"] = "maturity_veto"
        elif (state.ok("openalex") and v.get("oa_peak_ratio", 1) < 0.5 and (v.get("oa_peak_year") or 9999) <= 2023
              and v.get("oa_age", 0) >= 5 and v.get("oa_total", 0) >= 50):
            row["stage"], row["decision"] = 1, HYPE
            row["reasons"] = ["угасающая тема: публикаций в 2025 году меньше половины от пикового года"]
            row["reason_code"] = "hype_veto"
        row["novelty"] = novelty(v, cand.get("channels", {})) if state.ok("openalex") else 0.0
    alive = sorted((r for r in rows if "decision" not in r), key=lambda r: -r["novelty"])
    for r in alive[keep_wiki:]:
        r["stage"], r["decision"] = 1, LOW_EVIDENCE
        r["reasons"] = ["не проверен полностью: не вошёл в бюджет интерактивного запроса по новизне"]
        r["reason_code"] = "not_evaluated_novelty_cut"
    alive = alive[:keep_wiki]
    timings["stage1_s"] = round(time.perf_counter() - t_stage, 2)
    t_stage = time.perf_counter()
    # F. Провайдер отказал — это не то же самое, что «запрос не начинали». Пропущенные терминальные
    # кандидаты в счётчик отказов адаптера не попадают.
    failed_1 = sum(1 for r in rows if not r.get("not_evaluated") and not r["state"].ok("openalex"))
    log(f"Ступень 1, публикации: из {len(rows)} кандидатов дальше {len(alive)}"
        + (f"; пропущено терминальных без запроса: {skipped}" if skipped else "")
        + (f"; OpenAlex не ответил по {failed_1} кандидатам" if failed_1 else ""))

    with phase(budget.get("stage2")), ThreadPoolExecutor(workers) as ex:
        s2 = list(ex.map(_mapper(ctx, lambda row: observe("wiki", wikipedia.stats, row["cand"]["phrase"])), alive))
    for row, obs in zip(alive, s2):
        row["state"].add(obs)
        w = obs.values
        if obs.succeeded and w.get("wiki_article") and w.get("wiki_age", 0) >= 5:
            row["stage"], row["decision"] = 2, MATURE
            row["reasons"] = [f"зрелая технология: отдельная статья в Википедии существует {int(w['wiki_age'])} лет"]
            row["reason_code"] = "maturity_veto"
    alive = [r for r in alive if "decision" not in r]
    # Не вошедшие в ширину третьего этапа не опрашивались вовсе. Без явной причины они доходили до
    # решения с кодом «хайп нечем измерить: источник не ответил», хотя источник никто не спрашивал.
    for r in alive[keep_full:]:
        r["stage"], r["decision"] = 2, LOW_EVIDENCE
        r["reasons"] = ["не проверен полностью: не вошёл в ширину третьего этапа интерактивного запроса"]
        r["reason_code"] = "not_evaluated_stage3_width"
    alive = alive[:keep_full]
    timings["stage2_s"] = round(time.perf_counter() - t_stage, 2)
    t_stage = time.perf_counter()
    log(f"Ступень 2, Википедия: дальше {len(alive)}; полные признаки по новостям, Hacker News"
        + (", GitHub и публикациям" if with_github_stats else " и публикациям"))

    # Семейство «код» берётся из уже полученных репозиториев корпуса, дополнительных запросов нет.
    for row in rows:
        row["state"].add(_corpus_code_observation(row["cand"], corpus_adapters))

    # D. Критичное для ДОПУСКА измеряется РАНЬШЕ необязательного обогащения.
    #
    # Измерено на холодных прогонах: ступени 1 и 2 съедали около 27 с, до сетевого дедлайна
    # оставалось 9–11 с, и новости по кандидатам не успевали начаться ни для одного из них. Но без
    # измеренного новостного канала хайп объявляется неизмеримым, и ACCEPT закрыт fail-closed —
    # то есть весь бюджет уходил на обогащение, которое допуск открыть не может.
    #
    # Порядок работ переставлен, новых работ не добавлено и дедлайны не выросли: сначала новости по
    # ограниченному набору, затем всё остальное на остатке.
    # Полный опрос OpenAlex КРИТИЧЕН ДЛЯ ДОПУСКА наравне с новостями.
    #
    # Измерено: у одного и того же кандидата полный OpenAlex давал зрелость 0,74 и вето MATURE,
    # а отложенный — 0,58 и ACCEPT. Разница целиком в полях, которые лёгкий запрос ступени 1
    # возвращает нулями: широта областей, доля препринтов, доля компаний. Отложенное обогащение
    # таким образом ОСЛАБЛЯЛО вето зрелости, то есть отсутствие работы делало допуск легче.
    #
    # Hacker News и счётчики GitHub остаются необязательными и идут после критичного прохода.
    #
    # G. Внутри одного кандидата критичные источники опрашиваются в порядке ВОЗРАСТАНИЯ стоимости
    # хоста, а не в порядке записи словаря.
    #
    # Измерено на трёх владельческих живых прогонах (37da4c3): новости шли первыми, и одного
    # зависшего запроса хватало, чтобы съесть всё окно третьего этапа — таймаут запроса равен
    # min(policy, остаток), то есть ровно остатку. Полный OpenAlex после него не начинался НИ У
    # ОДНОГО кандидата: stage3_admission_critical=8, stage3_maturity_incomplete=8,
    # stage3_admission_complete=0. При этом сам по себе хост новостей на порядок дороже: 1,0 с
    # против 0,12 с у OpenAlex, то есть восемь кандидатов требуют 8 с только на выдачу запросов,
    # а всего третьему этапу оставалось 5,8–7,2 с.
    #
    # Порядок берётся из уже объявленных интервалов троттлинга, а не зашит списком: добавление
    # источника не требует правки этого места, и предметных исключений здесь нет. Ни один источник
    # не пропускается и не становится необязательным — меняется только очередь внутри кандидата.
    critical = _by_host_cost({"news": (news.stats, "news.google.com"),
                              "openalex": (openalex.stats, "api.openalex.org")})
    optional = {"hn": hackernews.stats}
    if with_github_stats:
        optional["github"] = github.stats

    def _pass(targets, sources, label):
        """Один проход по кандидатам порциями. Возвращает, скольких успели обработать.

        Пул, троттлинг по хостам и дедлайны — существующие. Новых очередей и исполнителей нет:
        внутри одного кандидата источники опрашиваются подряд, между кандидатами — как и раньше,
        параллельно, поэтому разные хосты своих лимитов друг другу не занимают."""
        done = 0
        if not sources or not targets:
            return 0
        work = lambda row: [observe(name, fn, row["cand"]["phrase"]) for name, fn in sources.items()]
        with ThreadPoolExecutor(workers) as ex:
            for start in range(0, len(targets), workers):
                chunk = targets[start:start + workers]
                left = time_left()
                if left is not None and left < STAGE3_SECONDS_PER_CANDIDATE:
                    break
                for row, observations in zip(chunk, ex.map(_mapper(ctx, work), chunk)):
                    for obs in observations:
                        row["state"].add(obs)
                        # Полная зрелость считается измеренной, только когда полный опрос
                        # действительно ответил. Отказ и недоезд полным измерением не являются.
                        if obs.source == "openalex" and obs.succeeded:
                            row["openalex_full"] = True
                    row[label] = True
                done += len(chunk)
        return done

    # E. Набор для дорогой поздней работы ограничен уже вычисленным порядком новизны: дальше него
    # кандидат в TOP-15 или список наблюдения попасть всё равно не может. Правило общее, зависит
    # только от ранжирования и от размера выдачи, ни к предметной области, ни к эталону не привязано.
    workset = alive[:max(1, admission_workset or len(alive))]
    critical_done = _pass(workset, critical, "admission_measured")
    for row in alive[:critical_done]:
        row["stage"] = 3
    enrich_done = _pass(alive[:critical_done], optional, "enriched")
    done = critical_done
    timings["stage3_admission_critical"] = critical_done
    timings["stage3_enriched"] = enrich_done
    timings["stage3_workset"] = len(workset)
    for row in alive[done:]:
        row["stage"], row["decision"] = 3, LOW_EVIDENCE
        row["reasons"] = ["не проверен полностью: исчерпан бюджет интерактивного запроса"]
        row["reason_code"] = "budget_exhausted"

    # B. Кандидат считается готовым к окончательному решению, только когда критичная для допуска
    # работа ПО НЕМУ разрешена. Если полный OpenAlex не стартовал, не доехал или отказал, лёгкие
    # нули ступени 1 остаются в состоянии и выглядят как измеренная зрелость — принимать по ним
    # нельзя. Закрываемся существующим механизмом принудительного решения каскада: пороги зрелости
    # и слой решения не трогаются.
    incomplete = 0
    for row in alive[:done]:
        if row.get("openalex_full") or "decision" in row:
            continue
        incomplete += 1
        row["decision"] = LOW_EVIDENCE
        row["reasons"] = ["зрелость измерена не полностью: полный опрос OpenAlex не выполнен, "
                          "а лёгкие значения первой ступени измерением широты и состава публикаций "
                          "не являются. Принимать кандидата по ним нельзя"]
        row["reason_code"] = "maturity_not_fully_measured"
    timings["stage3_maturity_incomplete"] = incomplete
    timings["stage3_admission_complete"] = sum(
        1 for r in alive[:done] if r.get("openalex_full") and r.get("admission_measured"))
    timings["stage3_s"] = round(time.perf_counter() - t_stage, 2)
    timings["stage3_enriched"] = done
    if done < len(alive):
        log(f"Ступень 3: полные признаки собраны по {done} из {len(alive)} кандидатов, дальше бюджет исчерпан")
    for row in rows:
        raw = dict(row["state"].values())
        years = raw.pop("oa_years", None)
        # Признаки модели этапа 1 требуют полного вектора. Отказавший источник даёт здесь ноль,
        # но этот вектор используется ТОЛЬКО как legacy_model_probability и помечается как ненадёжный.
        row["features"] = {**raw, **derive(raw)}
        if years:
            row["features"]["oa_years"] = " ".join(map(str, years))
    return rows


def legacy_probability(bundle, rows: list[dict]) -> None:
    """Вероятность модели этапа 1 для строк, дошедших до полных признаков. Отдельный предиктор, не право вето."""
    full = [r for r in rows if r.get("stage") == 3]
    if not full:
        return
    X = pd.DataFrame([{k: r["features"].get(k, 0.0) for k in FEATURES} for r in full]).astype(float).fillna(0.0)
    for r, p in zip(full, bundle["model"].predict_proba(X.values)[:, 1]):
        r["p"] = float(p)


def key_predictors(bundle, f: dict, top: int = 4) -> list[str]:
    lr = bundle["explainer"]
    x = pd.DataFrame([{k: f.get(k, 0.0) for k in FEATURES}])
    sc, m = lr.named_steps["standardscaler"], lr.named_steps["logisticregression"]
    contrib = ((x.values - sc.mean_) / sc.scale_ * m.coef_[0])[0]
    order = np.argsort(-np.abs(contrib))[:top]
    return [f"{'↑' if contrib[i] > 0 else '↓'} {NAMES_RU[FEATURES[i]]}: {_fmt(FEATURES[i], f.get(FEATURES[i], 0))}" for i in order]


def _fmt(name: str, v: float) -> str:
    if name.endswith("_log"):
        return f"{math.expm1(v):,.0f}".replace(",", " ")
    if name.endswith("_share"):
        return f"{v:.0%}"
    if name == "oa_growth":
        return f"×{math.exp(v):.1f}"
    if name == "wiki_article":
        return "да" if v else "нет"
    return f"{v:.2f}".rstrip("0").rstrip(".")


# ---------- 6. карточка ----------

def plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} {one}"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} {few}"
    return f"{n} {many}"


def build_card(cand: dict, f: dict, p: float | None, bundle, base_docs: list[Document],
               verdict, state: EvidenceState, diversity_rank: int | None = None) -> dict:
    retrieved = state.retrieved_at()
    docs = list(base_docs)
    papers = [d for d in docs if d.source_family == "science"]
    sentences = [s.strip() for d in papers for s in re.split(r"(?<=[.!?])\s+", d.snippet) if cand["phrase"].split()[0] in s.lower()]
    advantage_en = next((s for s in sentences if ADVANTAGE.search(s) and len(s) > 60), "")
    case = next((d for d in docs if d.source_type in {"новости", "репозиторий"}), docs[0] if docs else None)
    gen = llm.summarize(cand["phrase"], [d.to_dict() for d in docs[:6]])
    growth = math.exp(f.get("oa_growth", 0))
    name = ru.technology_name(cand.get("canonical_label") or cand["phrase"])
    interval = None
    if p is not None and bundle.get("bag"):
        x = pd.DataFrame([{k: f.get(k, 0.0) for k in FEATURES}]).astype(float).values
        bag = [m.predict_proba(x)[0, 1] for m in bundle["bag"]]
        interval = [round(float(np.percentile(bag, 5)), 3), round(float(np.percentile(bag, 95)), 3)]
    # Прогноз взлёта и исторический двойник считаются по PIT_FEATURES, куда входят счётчики GitHub
    # (gh_total, gh_12m) и Hacker News. На живом пути счётчики GitHub не запрашиваются
    # (LIVE_GITHUB_STATS=0), а Hacker News — необязательное обогащение, поэтому эти признаки
    # приходят нулями, неотличимыми от измеренных, — в том числе самый сильный признак модели
    # взлёта. Число по такому вектору не подкреплено проверкой из reports/takeoff.md: оно не
    # показывается и не пишется в реестр прогнозов.
    forward_supported = bool(verdict.legacy_model_reliable and state.ok("hn") and "gh_total" in f)
    row = pd.Series({**f, "oa_years": f.get("oa_years", "")})
    twin = (timemachine.find_twin(timemachine.pit_features(row, 2026))
            if forward_supported and isinstance(f.get("oa_years"), str) else None)
    gen_description = (gen or {}).get("description")
    gen_advantage = (gen or {}).get("advantage")
    gen_case = (gen or {}).get("case")
    # Откуда взят текст преимущества: пересказ модели, дословный фрагмент аннотации научной работы
    # на языке оригинала или ничего. Пометка «цитата» допустима только во втором случае.
    advantage_source = "generated" if gen_advantage else "abstract_quote" if advantage_en else "none"
    description = gen_description or (
        f"«{name['ru']}» ({cand['phrase']}): {plural(cand['recent_docs'], 'свежая работа', 'свежие работы', 'свежих работ')} по запросу за 2025–2026 годы "
        f"против {cand['prior_docs']} за 2019–2022. В открытых источниках: "
        f"{plural(f.get('oa_total', 0), 'научная публикация', 'научные публикации', 'научных публикаций')} "
        f"(рост ×{growth:.1f} к 2019–2021), {plural(f.get('news_1y', 0), 'новость', 'новости', 'новостей')} за год, "
        f"{plural(int(f.get('gh_total') or f.get('gh_corpus_repos') or 0), 'репозиторий', 'репозитория', 'репозиториев')}. "
        + ("Отдельной статьи в Википедии нет." if not f.get("wiki_article") else "Статья в Википедии появилась недавно."))
    card = {
        "technology": name["ru"], "technology_original": cand["phrase"],
        "original_phrase": cand.get("original_phrase", cand["phrase"]),
        "canonical_label": cand.get("canonical_label", cand["phrase"]),
        "canonical_source": cand.get("canonical_source"), "name_source": name["source"],
        "merge_key": cand.get("merge_key"), "merged_from": cand.get("merged_from", []),
        "assessment": cand.get("assessment"),
        "specificity": round(float((cand.get("assessment") or {}).get("specificity", 1.0)), 3),
        "ambiguity_flags": list((cand.get("assessment") or {}).get("ambiguity_flags") or []),
        "high_certainty": bool(cand.get("high_certainty")),
        "diversity_rank": diversity_rank, "sibling_key": cand.get("sibling_key"),
        # tier и status выводятся из авторитетного решения и только из него (R2-A).
        "tier": terminal_state(verdict.decision)[0],
        "status": terminal_state(verdict.decision)[1],
        # `score` в БД это ранжирующий балл, а не уверенность модели: см. REPAIR 4.
        "score": verdict.rank_score(),
        "legacy_model_interval_90": interval,
        "historical_twin": twin,
        "forward_looking_supported": forward_supported,
        "description": description, "description_generated": bool(gen_description),
        "generator": llm.model_name() if gen else None,
        # Языки источников, по которым модель писала русский пересказ: пересказ англоязычной работы
        # не является ни цитатой, ни русскоязычным источником.
        "summary_source_languages": sorted({d.language for d in docs[:6] if d.language}) if gen else [],
        "advantage": gen_advantage or advantage_en or "В найденных источниках преимущество явно не сформулировано.",
        "advantage_source": advantage_source,
        "advantage_is_original_en": advantage_source == "abstract_quote",
        "case": gen_case or (case.to_dict() if case else None), "case_generated": bool(gen_case),
        "why_weak_signal": (key_predictors(bundle, f) if verdict.legacy_model_reliable else verdict.decision_reasons[:4]),
        "takeoff": takeoff.predict(f) if forward_supported else None,
        "evidence": {"burst": cand["burst"], "recent_docs": cand["recent_docs"], "prior_docs": cand["prior_docs"],
                     **state.to_dict()},
        "features": {k: round(float(f.get(k, 0)), 3) for k in FEATURES},
        "sources": [d.to_dict() for d in docs],
        "sources_note": None,
    }
    card.update(verdict.to_dict())
    card["retrieved_at"] = retrieved
    card["from_cache"] = state.uses_stale_cache()
    return card


COMPLETE, INCOMPLETE_SEARCH, PROVIDER_ERROR = "COMPLETE", "INCOMPLETE_SEARCH", "PROVIDER_ERROR"
# Коды решения, которые говорят не «кандидат плох», а «кандидата не успели или не смогли проверить».
INCOMPLETE_CODES = {
    "budget_exhausted": "не проверены из-за исчерпанного бюджета времени",
    "not_evaluated_novelty_cut": "не вошли в бюджет проверки по новизне",
    "not_evaluated_stage3_width": "не вошли в ширину третьего этапа",
    "maturity_not_fully_measured": "зрелость измерена не полностью",
    "semantic_verification_pending": "ждут семантической проверки",
    "hype_evidence_unavailable": "без измерения хайпа: новостной канал не дал измерения",
}


# Отказы, причина которых — НАШ бюджет времени, а не источник: общий или фазовый дедлайн запроса,
# исчерпанный бюджет семантической фазы, поздний ответ, отброшенный по нашему же сроку. Запрос в
# таком случае либо не начинался, либо был прерван нами — утверждать «источник не ответил» нельзя.
INTERNAL_ERROR_TYPES = frozenset({DEADLINE})
INTERNAL_SEMANTIC_FAILURES = frozenset({"semantic_budget_exhausted", "late_budget_exhausted",
                                        "late_result_after_deadline"})
# Производное наблюдение: семейство «код» по кандидату отказывает, когда отказал сбор репозиториев
# корпуса. Причина уже учтена на уровне корпуса, второй раз её не считаем.
DERIVED_ERROR_TYPES = frozenset({"corpus_fetch_failed"})


def search_completeness(corpus_adapters: dict, corpus_error_types: dict, candidate_failures: dict,
                        reason_codes: Counter, semantic: dict, late: dict, accepted: int) -> dict:
    """Насколько полон поиск. Отделяет настоящее «ничего не найдено» от отказа провайдера и от
    недопроверенного запроса: пустая выдача при отказе источника — это не ответ «сигналов нет».

    Приоритет детерминированный:
      PROVIDER_ERROR    — хотя бы один внешний запрос к источнику или модели РЕАЛЬНО был начат и
                          отказал по внешней причине (HTTP, сеть, таймаут транспорта, лимит, разбор);
      INCOMPLETE_SEARCH — внешних отказов нет, но работа не завершена по нашей собственной причине:
                          дедлайн запроса или фазы, исчерпанный бюджет семантики, срез по ширине,
                          работа не начата из-за нехватки времени, модель не настроена;
      COMPLETE          — ни того, ни другого: пустая выдача означает, что проверки не прошёл никто.
    При PROVIDER_ERROR недопроверка не теряется: она дописывается в сообщение.

    `corpus_error_types` — {адаптер: [классы отказов корпуса]}; `candidate_failures` — {адаптер:
    {класс отказа: число кандидатов}}."""
    provider, incomplete = [], []
    for name, status in sorted(corpus_adapters.items()):
        if status not in ("ERROR", "PARTIAL"):
            continue
        types = set(corpus_error_types.get(name) or []) or {"unknown"}
        external = sorted(types - INTERNAL_ERROR_TYPES)
        if external:
            provider.append(f"{name} при сборе корпуса: {status} ({', '.join(external)})")
        if types & INTERNAL_ERROR_TYPES:
            incomplete.append(f"сбор корпуса {name} не завершён: исчерпан бюджет времени запроса")
    for name, by_type in sorted(candidate_failures.items()):
        external = sum(n for t, n in by_type.items()
                       if t not in INTERNAL_ERROR_TYPES and t not in DERIVED_ERROR_TYPES)
        internal = sum(n for t, n in by_type.items() if t in INTERNAL_ERROR_TYPES)
        if external:
            provider.append(f"{name}: отказ по {plural(external, 'кандидату', 'кандидатам', 'кандидатам')}")
        if internal:
            incomplete.append(f"{name}: по {plural(internal, 'кандидату', 'кандидатам', 'кандидатам')} "
                              "опрос не начат или прерван — исчерпан бюджет времени запроса")

    configured = bool(semantic.get("configured"))
    failure = semantic.get("failure_type")
    # Модель не настроена — это отсутствие конфигурации (причина всегда названа), а не отказ модели.
    # Пустой список кандидатов тоже даёт configured=False, но без причины: тогда модель недоступной
    # не называется.
    semantic_available = configured or not failure
    early_calls_failed = int(semantic.get("batches_attempted") or 0) > int(semantic.get("batches_succeeded") or 0)
    late_calls_made = int(late.get("attempt_count") or 0) > 0
    late_failure = late.get("failure_type")
    if (configured and semantic.get("outcome") == "failed" and early_calls_failed
            and failure not in INTERNAL_SEMANTIC_FAILURES and not late.get("verified_count")):
        provider.append(f"модель семантической проверки не ответила ({failure or 'отказ'})")
    elif configured and semantic.get("outcome") == "failed" and not early_calls_failed:
        incomplete.append("семантическая проверка не начата: исчерпан бюджет её фазы")
    if (late_calls_made and late_failure and late_failure not in INTERNAL_SEMANTIC_FAILURES
            and not late.get("verified_count")):
        provider.append(f"модель поздней семантической проверки не ответила ({late_failure})")
    elif late_failure in INTERNAL_SEMANTIC_FAILURES:
        incomplete.append("поздняя семантическая проверка не завершена: исчерпан её бюджет времени")
    incomplete += [f"{plural(n, 'кандидат', 'кандидата', 'кандидатов')} {INCOMPLETE_CODES[code]}"
                   for code, n in sorted(reason_codes.items()) if code in INCOMPLETE_CODES and n]
    if not semantic_available:
        incomplete.insert(0, f"семантическая проверка недоступна: модель не настроена ({failure}). "
                             "Без неё ни один кандидат не может быть подтверждён")
    state = PROVIDER_ERROR if provider else INCOMPLETE_SEARCH if incomplete else COMPLETE
    if state == COMPLETE:
        message = ("Поиск выполнен полностью." if accepted else
                   "Поиск выполнен полностью: все источники ответили, но ни один кандидат не прошёл все "
                   "проверки — достаточность доказательства, конкретный документ уровня A/B, независимое "
                   "подтверждение и семантическую проверку.")
    else:
        if state == PROVIDER_ERROR:
            message = "Часть источников не ответила: " + "; ".join(provider) + "."
            if incomplete:
                message += " Кроме того, поиск выполнен не полностью: " + "; ".join(incomplete) + "."
        else:
            message = "Поиск выполнен не полностью: " + "; ".join(incomplete) + "."
        message += (" Отсутствие подтверждённых сигналов в этом случае не означает, что их нет."
                    if not accepted else " Выдача может быть неполной.")
    return {"state": state, "provider_errors": provider, "incomplete_reasons": incomplete,
            "semantic_verification_available": semantic_available, "message": message}


def run(query: str, top: int = 15, n_candidates: int = 160, workers: int = 8, log=print,
        budget_s: float | None = None) -> dict:
    """Полный проход открытого запроса под общим бюджетом времени.

    Отказ одного адаптера не роняет запрос: он снижает качество доказательства по конкретному кандидату.
    Исчерпание бюджета тоже не роняет запрос: возвращается то, что успели подтвердить."""
    t0 = time.time()
    timings: dict = {}
    run_meta = provenance.run_metadata(query)
    plan_budget = budget_plan(budget_s)
    budget = plan_budget["hard_wall_budget_s"]
    network_budget = plan_budget["network_deadline_s"]
    bundle = joblib.load(MODEL)
    with fetch_context(budget_s=network_budget) as ctx:
        t = time.perf_counter()
        plan = decompose.decompose(query, parse_query)
        seeds = plan.seeds()
        timings["query_decomposition_s"] = round(time.perf_counter() - t, 2)
        # REPAIR F. Источник плана фиксируется сразу после разбора: раньше метаданные собирались
        # до него, и в выдаче стояло query_plan_source = null при plan.source = llm.
        run_meta["query_plan_source"] = plan.source
        run_meta["query_plan_model"] = plan.model
        run_meta["query_intent"] = plan.intent
        log(f"Запрос: «{query}». Разбор: {plan.source}"
            + (f" ({plan.model})" if plan.model else f" ({plan.fallback_reason})")
            + f". Поисковые фразы: {seeds}")
        if not seeds:
            seeds = [query.strip()]

        t = time.perf_counter()
        with phase(network_budget * BUDGET_SHARE["corpus"]):
            cands, corpus = extract_candidates(seeds, n_candidates)
        timings["candidate_generation_s"] = round(time.perf_counter() - t, 2)

        # --- семантическая нормализация и склейка вариантов до дорогого обогащения (REPAIR 2, 3) ---
        t = time.perf_counter()
        seed_tokens = {tok for s in seeds for tok in re.findall(r"[a-z0-9]+", s.lower())}
        cands, dropped = relevance.filter_candidates(cands, seed_tokens, set(plan.exclusions))
        semantic_report: dict = {}
        cands, llm_dropped = relevance.canonicalize(cands, domain=plan.domain, report=semantic_report)
        run_meta["semantic_normalizer"] = {**run_meta.get("semantic_normalizer", {}), **semantic_report}
        dropped += llm_dropped
        cands, merged = relevance.collapse(cands)
        dropped += merged
        timings["semantic_filter_s"] = round(time.perf_counter() - t, 2)
        log(f"Корпус: { {k: v for k, v in corpus.items() if k != 'адаптеры корпуса'} }. "
            f"Кандидатов: {len(cands)}, отбраковано семантически: {len(dropped) - len(merged)}, "
            f"склеено вариантов: {len(merged)}")

        # E. Дорогая поздняя работа ограничена тем, что вообще может попасть в выдачу: размер TOP
        # плюс столько же на список наблюдения. Ни предметной области, ни эталона здесь нет.
        # H. Под адресную позднюю проверку резервируется окно [T−8, T] из ОБЩЕГО дедлайна.
        #
        # Резерв берётся не у третьего этапа и не у финализации: доля семантики как была 18 с, так и
        # осталась, она лишь разделена на раннюю фазу (10 с) и позднюю (8 с). Каскад работает до
        # T−8 вместо T, то есть теряет ровно те секунды, которые ранняя фаза больше не занимает.
        #
        # Если модель не настроена, резерв равен нулю: отнимать у каскада время под невозможный
        # вызов нельзя, деградированное детерминированное поведение остаётся прежним.
        late_reserve = relevance.late_reserve_seconds()
        timings["late_semantic_reserve_s"] = late_reserve
        pre_late = time_left()
        pre_late = None if pre_late is None else max(0.0, pre_late - late_reserve)
        timings["stage3_keep_full"] = stage3_width(top)
        with phase(pre_late):
            rows = cascade_collect(cands + dropped, workers, log, keep_wiki=LIVE_KEEP_WIKI,
                                   keep_full=stage3_width(top),
                                   admission_workset=max(1, top * 2),
                                   budget={k: network_budget * v for k, v in BUDGET_SHARE.items()},
                                   corpus_adapters=corpus.get("адаптеры корпуса", {}),
                                   with_github_stats=LIVE_GITHUB_STATS, timings=timings)
            legacy_probability(bundle, rows)

        t = time.perf_counter()

        def verdict_for(row, documents, verification=None):
            """Решение по кандидату при данном наборе конкретных документов.

            `verification` позволяет вычислить решение при КОНТРФАКТИЧЕСКОМ состоянии семантической
            проверки, ничего не записывая в кандидата. Ровно этим отбираются цели поздней проверки:
            пороги здесь не дублируются, используется тот же `decide`."""
            assessment = row["cand"].get("assessment") or {}
            forced = row.get("decision")
            v = decide(row["state"], burst=row["cand"].get("burst", 0.0),
                       noise_reason=(row["reasons"][0] if forced == NOISE else row["cand"].get("noise_reason")),
                       legacy_probability=row.get("p"), documents=documents,
                       noise_code="semantic_rejection",
                       verification=verification or verification_status(assessment),
                       high_certainty=bool(row["cand"].get("high_certainty")),
                       candidate=row["cand"])
            if forced and forced != NOISE:
                v.decision, v.decision_reasons = forced, row["reasons"]
                v.decision_reason_codes = [row.get("reason_code", "cascade_veto")]
            return v

        # Проход 1: решение по документам, уже полученным при сборе корпуса.
        for r in rows:
            r["documents"] = concrete_documents(r["cand"], r["state"].retrieved_at(),
                                               r["state"].uses_stale_cache())
            r["specificity"] = float((r["cand"].get("assessment") or {}).get("specificity", 1.0))
            r["verdict"] = verdict_for(r, r["documents"])
        timings["synthesis_s"] = round(time.perf_counter() - t, 2)

        # --- дообогащение до окончательного решения (R2-C) ---
        # Новости добавляются тем кандидатам, у которых конкретных документов может не хватать: их
        # исход способен измениться. Дообогащение идёт до финального решения, поэтому полученные
        # документы реально участвуют в шлюзе качества, а не только украшают карточку.
        t = time.perf_counter()
        enrichable = [r for r in rows if r.get("stage") == 3 and (
            r["verdict"].accepted
            or set(r["verdict"].decision_reason_codes) & ENRICHABLE_CODES)]
        enrichable.sort(key=lambda r: -(r["verdict"].emergence_score * r.get("specificity", 1.0)))
        enriched = 0
        # Дообогащение — тоже работа ДО поздней проверки, поэтому оно обязано укладываться в T−8.
        pre_late_enrich = time_left()
        pre_late_enrich = None if pre_late_enrich is None else max(0.0, pre_late_enrich - late_reserve)
        with phase(pre_late_enrich):
            for r in enrichable[:ENRICH_TOP]:
                if http_too_late(4.0):
                    break
                try:
                    r["extra_docs"] = (news.search(r["cand"]["phrase"], lang="ru", limit=3)
                                       + news.search(r["cand"]["phrase"], lang="en", limit=3))
                    enriched += 1
                except Exception:  # noqa: BLE001 — карточка строится и без новостей
                    r["extra_docs"] = []
        timings["document_enrichment_s"] = round(time.perf_counter() - t, 2)
        timings["document_enrichment_count"] = enriched
        t_final = time.perf_counter()

        # --- адресная поздняя семантическая проверка (окно [T−8, T]) ---
        #
        # Кто проверяется. Только кандидат, для которого семантика — ПОСЛЕДНЕЕ препятствие: он ещё
        # не подтверждён, и при контрфактически подтверждённой семантике тот же самый `decide`
        # выдаёт ACCEPT. Проверка по тексту причины «semantic_verification_pending» была бы
        # недостаточна: измерено, что шесть таких кандидатов всё равно останавливаются на
        # hype_evidence_unavailable. Пороги здесь не дублируются и предметных списков нет.
        # R3. Дообогащение МАТЕРИАЛИЗУЕТСЯ ДО отбора целей.
        #
        # Раньше документы дообогащения приклеивались только во втором проходе, уже после выбора
        # целей. Кандидат, которому для достаточности не хватало ровно одного независимого
        # медийного документа, на момент отбора выглядел недостаточным по доказательствам — и
        # целью не становился, хотя после приклейки семантика оставалась его единственным
        # препятствием. Новых правил доказательства здесь нет: это перенос существующего блока
        # через тот же замороженный путь доверия и претензионной релевантности.
        retrieved_now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for r in rows:
            extra = r.get("extra_docs") or []
            if not extra:
                continue
            # Дообогащение проходит тот же шлюз релевантности: новость, найденная по фразе кандидата,
            # подтверждением не становится, если она не про эту технологию.
            r["documents"] = list(r["documents"]) + claim_relevance.annotate(
                [trust.assess(d).with_provenance(retrieved_at=r["state"].retrieved_at() or retrieved_now)
                 for d in extra], r["cand"])
            r["verdict"] = verdict_for(r, r["documents"])

        t_late = time.perf_counter()
        late_report: dict = {}
        qualified = []
        for r in rows:
            if r.get("stage") != 3 or r["verdict"].accepted:
                continue
            if verification_status(r["cand"].get("assessment") or {}) == SEMANTICALLY_VERIFIED:
                continue
            if verdict_for(r, r["documents"], verification=SEMANTICALLY_VERIFIED).accepted:
                qualified.append(r)
        # Ранжирование целей — существующим до-семантическим порядком выдачи, нового балла нет.
        qualified.sort(key=lambda r: -(r["verdict"].emergence_score * r.get("specificity", 1.0)))
        targets = qualified[:relevance.LATE_MAX_TARGETS]
        timings["late_semantic_qualified"] = len(qualified)
        timings["late_semantic_targets"] = len(targets)
        if targets:
            relevance.verify_late([r["cand"] for r in targets], domain=plan.domain,
                                  report=late_report, budget_s=late_reserve)
            for r in targets:
                r["verdict"] = verdict_for(r, r["documents"])
        timings["late_semantic_s"] = round(time.perf_counter() - t_late, 2)
        timings["late_semantic_verified"] = int(late_report.get("verified_count") or 0)
        run_meta["late_semantic"] = late_report
        # Поздняя проверка — тоже сетевая работа, поэтому сетевая фаза закрывается ПОСЛЕ неё:
        # иначе телеметрия занижала бы реально потраченное на сеть время.
        timings["network_phase_s"] = round(time.time() - t0, 2)

        accepted = [r for r in rows if r["verdict"].accepted]
        rejected = [r for r in rows if not r["verdict"].accepted]
        # R2-G: порядок выдачи разнообразится, исходный балл сохраняется.
        accepted = diversify(accepted, lambda r: r["verdict"].rank_score() * r.get("specificity", 1.0))
        # Список наблюдения: прошли вето зрелости, хайпа и шума, споткнулись только о доказательство
        # или о семантическую проверку. Подтверждённым слабым сигналом такой кандидат не называется.
        watchlist = [r for r in rejected
                     if r.get("stage") == 3 and r["verdict"].decision == LOW_EVIDENCE
                     and r["verdict"].emergence_score >= MIN_EMERGENCE]
        watchlist.sort(key=lambda r: -(r["verdict"].emergence_score * r.get("specificity", 1.0)))
        log(f"Решение: принято {len(accepted)}, в списке наблюдения {len(watchlist)}, отклонено {len(rejected)}")

        cards = [build_card(r["cand"], r["features"], r.get("p"), bundle, r["documents"],
                            r["verdict"], r["state"], diversity_rank=r.get("diversity_rank"))
                 for r in accepted[:top]]
        watch_cards = [build_card(r["cand"], r["features"], r.get("p"), bundle, r["documents"],
                                  r["verdict"], r["state"]) for r in watchlist[:top]]
        takeoff.register(query, cards)
        run_meta["provider_status"] = {
            "корпус": corpus.get("адаптеры корпуса", {}),
            "отказы адаптеров": dict(Counter(a for r in rows for a in r["state"].failed())),
        }
        run_meta["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # F. Усечение — это когда кандидаты РЕАЛЬНО остались непроверенными, а не просто когда
        # часы дошли до сетевого дедлайна: при полностью обработанном наборе усечения не было.
        truncated_rows = sum(1 for r in rows if r.get("reason_code") == "budget_exhausted")
        run_meta["budget_truncated"] = bool(truncated_rows) or bool(http_too_late(0.0))
        run_meta["not_evaluated_terminal"] = sum(1 for r in rows if r.get("not_evaluated"))
        run_meta["budget_truncated_candidates"] = truncated_rows
        by_decision = Counter(r["verdict"].decision for r in rejected)
        by_code = Counter(code for r in rejected for code in r["verdict"].decision_reason_codes)
        adapter_failures = Counter(a for r in rows for a in r["state"].failed())
        # Причина каждого отказа по кандидату: внешний отказ источника или наш собственный дедлайн.
        failure_types: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            for name, obs in r["state"].failed().items():
                failure_types[name][obs.error_type or "unknown"] += 1
        completeness = search_completeness(corpus.get("адаптеры корпуса", {}),
                                           corpus.get("типы отказов корпуса", {}), failure_types, by_code,
                                           run_meta.get("semantic_normalizer") or {}, late_report,
                                           len(accepted))
        elapsed = time.time() - t0
        timings["finalization_s"] = round(time.perf_counter() - t_final, 2)
        timings["budget_truncated"] = bool(truncated_rows) or bool(http_too_late(0.0))
        timings["budget_truncated_candidates"] = truncated_rows
        timings["total_wall_s"] = round(elapsed, 2)
        if timings["finalization_s"] > plan_budget["finalization_margin_s"]:
            log(f"Финализация заняла {timings['finalization_s']:.1f} с при резерве "
                f"{plan_budget['finalization_margin_s']:.1f} с: общее время вышло за стену")
        return {
            "query": query, "seeds": seeds, "query_plan": plan.to_dict(), "date": date.today().isoformat(),
            "model": bundle["name"],
            "stats": {**corpus, "план запроса": plan.to_dict(), "проверено кандидатов": len(rows),
                      "search_completeness": completeness,
                      "accepted_count": len(accepted), "watchlist_count": len(watchlist),
                      "rejected_count": len(rejected) - len(watchlist),
                      "принято (ACCEPT)": len(accepted),
                      "отклонено LOW_EVIDENCE": by_decision.get(LOW_EVIDENCE, 0),
                      "отклонено MATURE": by_decision.get(MATURE, 0),
                      "отклонено HYPE": by_decision.get(HYPE, 0),
                      "отклонено NOISE": by_decision.get(NOISE, 0),
                      "причины отклонения": dict(by_code),
                      "weak_signal_count": sum(c["status"] == WEAK_SIGNAL_STATUS for c in cards),
                      "слабых сигналов": sum(c["status"] == WEAK_SIGNAL_STATUS for c in cards),
                      "под наблюдением": len(watch_cards),
                      # Имя намеренно говорит про ДОСТУПНОСТЬ ИСТОЧНИКОВ: этот показатель не
                      # утверждает, что технология доказана на 75 % (P1 честности).
                      "с доступностью источников ≥ 75 %": sum(c["evidence_confidence"] >= 0.75 for c in cards),
                      "с достаточным претензионным доказательством": sum(
                          bool(c.get("claim_evidence_sufficient")) for c in cards),
                      "исключено": len(rejected),
                      "отказы адаптеров": dict(adapter_failures),
                      "источников на принятого кандидата": round(
                          sum(len(c["sources"]) for c in cards) / len(cards), 1) if cards else 0,
                      "минимум источников у принятого": min((len(c["sources"]) for c in cards), default=0),
                      "run_id": run_meta["run_id"], "commit_sha": run_meta["commit_sha"],
                      "config_hash": run_meta["config_hash"],
                      "семантическая нормализация": run_meta["semantic_normalizer"]["configured"],
                      "подтверждено семантически": sum(c["verification"] == "SEMANTICALLY_VERIFIED" for c in cards),
                      "подтверждено детерминированно": sum(c.get("high_certainty") for c in cards),
                      "родственная концентрация в топе": sibling_concentration([r["cand"] for r in accepted[:top]]),
                      "дублей в топе": relevance.duplicate_rate([c["technology_original"] for c in cards]),
                      "бюджет, с": budget, "секунд": round(elapsed, 1),
                      **plan_budget,
                      "бюджет исчерпан": elapsed >= budget, "тайминги": timings},
            "timings": timings, "budget": plan_budget, "run": run_meta,
            "signals": cards,
            "watchlist": watch_cards,
            "excluded": [{"technology": ru.technology_name(r["cand"].get("canonical_label") or r["cand"]["phrase"])["ru"],
                          "score": round(r["verdict"].rank_score(), 3),
                          "decision": r["verdict"].decision,
                          "code": (r["verdict"].decision_reason_codes or [""])[0],
                          "reason": "; ".join(r["verdict"].decision_reasons) or r["verdict"].decision}
                         for r in sorted(rejected, key=lambda r: (r.get("stage", 1), -r.get("novelty", 0)))],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--candidates", type=int, default=160)
    ap.add_argument("--json")
    ap.add_argument("--budget", type=float, default=None, help="общий бюджет запроса в секундах")
    a = ap.parse_args()
    res = run(a.query, a.top, a.candidates, budget_s=a.budget)
    print(f"\n{res['stats']}\n")
    print(f"Тайминги по этапам: {res['timings']}\n")

    def show(items, title):
        if not items:
            return
        print(f"\n{title}:")
        for i, s in enumerate(items, 1):
            print(f"{i:2d}. {s['technology']} ({s['technology_original']})  {s['decision']}")
            print(f"      появление {s['emergence_score']:.2f} · доступность источников {s['evidence_confidence']:.2f} · "
                  f"зрелость {s['maturity_risk']:.2f} · хайп {s['hype_risk']:.2f} · ранг {s['rank_score']:.3f}")
            legacy = s.get("legacy_model_probability")
            print("      базовый предиктор этапа 1: "
                  + (f"{legacy:.0%}" if legacy is not None else "не считался")
                  + (" (показателен)" if s.get("legacy_model_reliable") else " — экспериментальный, в решении не участвует"))
            ev = s["evidence"]
            print(f"      подтверждающих документов A/B: {s['supporting_document_count']} "
                  f"({', '.join(s['supporting_document_families']) or 'нет'}); "
                  f"семейства: {', '.join(ev['corroborating_families']) or 'нет'}; "
                  f"отказали: {', '.join(ev['failed_adapters']) or 'никто'}")
            if s.get("merged_from"):
                print(f"      склеено с: {', '.join(s['merged_from'])}")
            src = s["sources"][0] if s["sources"] else None
            if src:
                print(f"      источник: {src['source_name']}, {src['published']}, {src['source_type']}, "
                      f"доверие {src['trust']}: {src['url']}")

    show(res["signals"], "Подтверждённые слабые сигналы")
    show(res["watchlist"], "Список наблюдения (требуют дополнительных доказательств)")
    print("\nИсключены:")
    for e in res["excluded"][:15]:
        print(f"  - [{e['decision']}/{e['code']}] {e['technology']}: {e['reason']}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
