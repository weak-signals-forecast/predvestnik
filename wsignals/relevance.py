"""Семантическая нормализация кандидатов до дорогого обогащения (REPAIR 2, REPAIR 3).

Аудит R1: в живой выдаче оставались общие и не относящиеся к направлению фразы — «llm tools»,
«financial ai», «deploying ai agents», «ai-driven educational», «cloud dependency». Дорогое
обогащение тратилось на них, а они же занимали места в ТОП-15.

Порядок намеренный:
  1. дешёвые детерминированные правила — общие, а не список плохих примеров;
  2. если YandexGPT сконфигурирована — один пакетный структурированный вызов на всех выживших.

Правила общие и морфологические, а не перечень запрещённых фраз:
  * головное слово фразы называет категорию («tools», «solutions», «dependency», «ai») —
    это класс вещей, а не технология;
  * фраза начинается с герундия («deploying …») — это действие, а не название технологии;
  * фраза заканчивается прилагательным («… educational», «… ai-driven») — это не именная группа;
  * все токены фразы — общая лексика или слова самого запроса — это пересказ темы;
  * в фразе есть имя продукта или компании — это носитель технологии, а не технология.

Модель НЕ решает, слабый ли это сигнал. Её роль — семантическая связность, идентичность технологии
и отношение к направлению запроса. Схема ответа проверяется строго, по каждому элементу отдельно.

Схема оценки кандидата (одинаковая для детерминированного пути и для модели):

    original_phrase, canonical_name, is_technology, domain_relevance (0..1),
    is_product_or_brand, is_generic_phrase, merge_key, reject_reason
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field

import requests

from . import intent
from .decompose import COMPLETION_URL, MAX_LEN, TEMPERATURE, TIMEOUT, configured_model

log = logging.getLogger("wsignals.relevance")

# ---------- общая лексика ----------
#
# REPAIR A. Детерминированный префильтр обязан быть ВЫСОКОРЕКОЛЛЬНЫМ. Прежняя версия отвергала
# законные технологии только за головное слово («key management», «data integration», «spectral
# analysis», «protocol development») и за совпадение любого токена с названием бренда («meta»,
# «swift», «visa» в обычном техническом смысле).
#
# Поэтому словари разделены по силе улики, а широкие лексические правила из вето превращены
# в штраф специфичности и флаг неоднозначности. Жёстко отвергается только то, про что есть
# сильная улика: испорченная форма, очевидный обрывок, однозначно продуктовое имя, полностью
# общая фраза. Неоднозначное доходит до семантической нормализации и до доказательств.

# Коины: слова, у которых нет обычного технического значения. Совпадение по ним — сильная улика.
UNAMBIGUOUS_BRANDS = {"openai", "anthropic", "claude", "chatgpt", "gpt", "copilot", "codex", "llama",
                      "mistral", "deepseek", "qwen", "grok", "midjourney", "gigachat", "yandexgpt",
                      "huggingface", "langchain", "llamaindex", "ollama", "vllm", "pytorch", "tensorflow",
                      "kubernetes", "nvidia", "perplexity", "cursor", "prnewswire", "businesswire"}
# Обычные слова, которые заодно являются брендами. Совпадение по ним — не улика, а флаг.
AMBIGUOUS_BRANDS = {"meta", "swift", "visa", "apple", "amazon", "oracle", "spark", "go", "rust", "azure",
                    "gemini", "docker", "github", "gitlab", "stripe", "revolut", "binance", "coinbase",
                    "mastercard", "google", "microsoft", "sber", "yandex"}
BRANDS = UNAMBIGUOUS_BRANDS | AMBIGUOUS_BRANDS          # совместимость с прежним импортом

# Обстановка репозитория: сильная улика, что это не технология, а оформление.
REPO_FURNITURE = {"awesome", "boilerplate", "cheatsheet", "tutorial", "handbook", "roadmap", "curated",
                  "starter", "clone", "playground", "cookbook", "snippets", "template", "templates",
                  "examples", "example", "demo", "docs", "documentation", "list", "lists", "collection"}
# Техническая обвязка: слабая улика, только штраф.
SOFT_PLUMBING = {"server", "servers", "client", "clients", "api", "sdk", "cli", "app", "apps", "ui", "gui",
                 "dashboard", "plugin", "plugins", "extension", "extensions", "wrapper", "wrappers",
                 "toolkit", "bot", "chatbot", "repo", "repository", "port"}
PLUMBING = REPO_FURNITURE | SOFT_PLUMBING               # совместимость с прежним импортом

# Головные слова делового и обзорного дискурса: рынок и тренд не бывают технологией ни при каком
# определении, поэтому это по-прежнему вето.
DISCOURSE_HEADS = {"market", "markets", "industry", "industries", "company", "companies", "ecosystem",
                   "ecosystems", "landscape", "trend", "trends", "future", "era", "space", "domain",
                   "field", "area", "sector", "opportunity", "opportunities", "challenge", "challenges",
                   "issue", "issues", "perspective", "perspectives", "overview", "insight", "insights",
                   "strategy", "strategies", "initiative", "initiatives", "practice", "practices",
                   "adoption", "usage", "study", "studies", "survey", "review", "research",
                   "briefing", "briefings"}
# P1-B. Головные слова ДОКУМЕНТА или разговора о предмете. Сами по себе они вето НЕ дают: «numerical
# weather forecast», «structural health report», «cyber readiness» и «situational awareness» — это
# настоящие технические понятия. Независимый аудит показал, что жёсткое вето по этим словам выбивало
# 20 из 23 проверочных фраз, включая прогноз погоды и прогноз урожайности.
#
# Поэтому они лишь не считаются собственным техническим содержанием: фраза отбраковывается, только
# если КРОМЕ такого слова в ней нет ничего технического («technology trend report», «market outlook»).
DISCOURSE_ARTIFACT_HEADS = {"report", "reports", "forecast", "forecasts", "outlook",
                            "inequality", "awareness", "readiness", "maturity", "hype"}
# P1-B. Словарь ИССЛЕДОВАТЕЛЬСКОГО ДИСКУРСА: слова, которыми описывают саму работу, а не предмет.
# Используется ТОЛЬКО для проверки фразы ЦЕЛИКОМ: если каждое слово фразы взято из служебного,
# рубричного или дискурсивного словаря, фраза говорит о разговоре, а не о технологии. Ни одно из
# этих слов не запрещено само по себе: «scanning tunneling microscopy» и «work function» проходят.
RESEARCH_DISCOURSE_WORDS = {"horizon", "scanning", "foresight", "direction", "directions",
                            "section", "sections", "work", "works", "related", "literature",
                            "agenda", "gap", "gaps", "state", "art", "background", "introduction",
                            "conclusion", "conclusions", "discussion", "abstract", "references",
                            "appendix", "outlook", "prospects", "prospect"}
# Головные слова технических категорий: сами по себе общие, но с осмысленным определением называют
# настоящую технологию («key management», «data integration»). Только штраф, не вето.
TECHNICAL_CATEGORY_HEADS = {"technology", "technologies", "solution", "solutions", "system", "systems",
                            "platform", "platforms", "framework", "frameworks", "approach", "approaches",
                            "method", "methods", "tool", "tools", "service", "services", "product",
                            "products", "application", "applications", "capability", "capabilities",
                            "feature", "features", "factor", "factors", "aspect", "aspects",
                            "management", "integration", "analysis", "development", "evaluation",
                            "performance", "architecture", "infrastructure", "dependency", "dependencies"}
CATEGORY_HEADS = DISCOURSE_HEADS | TECHNICAL_CATEGORY_HEADS   # совместимость с прежним импортом
# Классы моделей как головное слово: «financial ai» это раздел. Штраф, не вето.
MODEL_CLASS_HEADS = {"ai", "ml", "llm", "llms", "dl", "nlp", "genai", "agi"}
# Общие определения без технического содержания. Намеренно НЕ включает слова с техническим смыслом
# («key», «core», «main»): именно такие включения и давали ложные отказы.
GENERIC_MODIFIERS = {"ai", "ai-driven", "ai-powered", "ai-based", "ai-enabled", "advanced", "emerging",
                     "modern", "novel", "innovative", "intelligent", "smart", "future", "next-generation",
                     "effective", "robust", "efficient", "scalable", "digital", "new", "improved",
                     "enhanced", "enhancing", "comprehensive", "generic", "various", "several", "strategic"}
# Причастные определения: в ключе склейки не участвуют, иначе «ai-powered fraud» и «ai-driven fraud»
# остаются разными кандидатами.
PARTICIPLE_MODIFIERS = {"driven", "powered", "based", "enabled", "aware", "ready", "centric", "oriented",
                        "assisted", "augmented", "native"}
# Отглагольные существительные: выглядят как герундий, но являются названиями.
GERUND_NOUNS = {"learning", "computing", "engineering", "networking", "sensing", "imaging", "printing",
                "monitoring", "routing", "caching", "training", "tuning", "poisoning", "watermarking",
                "fingerprinting", "mapping", "clustering", "embedding", "streaming", "scaling", "cooling",
                "charging", "manufacturing", "housing", "sharding", "pruning", "quantizing", "hedging",
                "onboarding", "provisioning", "sandboxing", "logging", "tracing", "profiling", "meshing",
                "planning", "reasoning", "understanding", "processing", "modeling", "modelling",
                "sequencing", "forecasting", "screening", "welding", "coating", "recycling", "harvesting",
                "breeding", "shielding", "grounding", "signing", "hashing", "encoding", "decoding",
                # отглагольные существительные отраслевой лексики: «banking», «trading», «mining»
                "banking", "accounting", "marketing", "advertising", "consulting", "trading", "lending",
                "borrowing", "underwriting", "leasing", "licensing", "staking", "mining", "farming",
                "drilling", "refining", "casting", "milling", "sintering", "doping", "etching",
                "bonding", "packaging", "testing", "dosing", "mixing", "blending", "reporting",
                "indexing", "ranking", "matching", "filtering", "sampling", "batching", "scheduling",
                "balancing", "provisioning", "partitioning", "compounding", "settling", "clearing"}
# Существительные на -al и подобные, которые прилагательным правилом задевать нельзя.
NOUN_EXCEPTIONS = {"signal", "signals", "material", "materials", "terminal", "terminals", "portal",
                   "portals", "crystal", "crystals", "metal", "metals", "capital", "journal", "canal",
                   "goal", "total", "manual", "arsenal", "protocol", "protocols", "interval", "channel",
                   "channels", "panel", "panels", "cell", "cells", "model", "models", "vehicle",
                   "vehicles", "module", "modules", "credential", "credentials", "professional",
                   "professionals", "individual", "individuals", "chemical", "chemicals", "mineral",
                   "minerals", "pedal", "spiral", "neural", "local", "global", "central", "digital"}
# Суффиксы, по которым слово почти наверняка прилагательное.
ADJECTIVE_SUFFIXES = ("ive", "ous", "able", "ible", "ful", "less", "ish", "ary", "ential", "ical",
                      "istic", "driven", "powered", "based", "enabled", "aware", "ready", "centric",
                      "friendly", "oriented")
FUNCTION_WORDS = {"of", "for", "in", "on", "with", "the", "a", "an", "and", "to", "by", "from", "at", "as"}

MIN_TOKENS, MAX_TOKENS_IN_PHRASE = 2, 4
MIN_DOMAIN_RELEVANCE = float(os.getenv("WSIGNALS_MIN_DOMAIN_RELEVANCE", "0.40"))
# Отказ по слабости требует, чтобы слабыми оказались ОБА признака сразу: и техническая специфичность,
# и связь с направлением. Одного общего головного слова для отказа недостаточно, как и одной слабой
# связи при полностью специфичной фразе — судить о такой должны доказательства, а не лексика.
MIN_SPECIFICITY = float(os.getenv("WSIGNALS_MIN_SPECIFICITY", "0.70"))
# Штрафы специфичности. Каждый снимает часть уверенности, но не отвергает кандидата.
PENALTY_TECHNICAL_HEAD = 0.35
PENALTY_MODEL_CLASS_HEAD = 0.35
PENALTY_SOFT_PLUMBING = 0.25
PENALTY_AMBIGUOUS_BRAND = 0.25


def _tokens(phrase: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", (phrase or "").lower())


def _words(phrase: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (phrase or "").lower())


def _stem(token: str) -> str:
    """Грубая нормализация числа. Точности морфологического анализатора здесь не нужно:
    ключ используется только для склейки вариантов одной и той же фразы."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


# Токены, не несущие различающего содержания: из ключа склейки они выбрасываются.
NON_DISTINGUISHING = GENERIC_MODIFIERS | PARTICIPLE_MODIFIERS | FUNCTION_WORDS


def merge_key(phrase: str) -> str:
    """Ключ склейки вариантов одной технологии: содержательные токены в нормальной форме, по алфавиту.

    Выбрасываются только слова без различающего содержания — общие определения и причастные модификаторы.
    Специфические токены не выбрасываются никогда: ложная склейка хуже двух отдельных кандидатов."""
    toks = [_stem(t) for t in _words(phrase) if t not in NON_DISTINGUISHING and len(t) > 1]
    return " ".join(sorted(set(toks)))


def _is_adjective(token: str) -> bool:
    word = token.replace("-", "")
    if token in NOUN_EXCEPTIONS or word in NOUN_EXCEPTIONS:
        return False
    if token.endswith(ADJECTIVE_SUFFIXES):
        return True
    return token.endswith("al") and token not in NOUN_EXCEPTIONS and len(token) > 5


def _is_gerund_verb(token: str) -> bool:
    """Герундий в роли действия. Причастия-определения («emerging», «enhancing») сюда не попадают:
    они общая лексика, и фразу отвергает другое правило, а не это."""
    return (token.endswith("ing") and token not in GERUND_NOUNS
            and token not in GENERIC_MODIFIERS and len(token) > 5)


# R2-A. Узкое детерминированное правило «это точно название технологии».
#
# Когда семантическая нормализация недоступна, лексически чистая фраза сама по себе НЕ доказывает,
# что это технология: «short drama» и «local first» проходят любой лексический фильтр. Признавать
# такие кандидаты подтверждёнными нельзя. Но полностью отказываться от подтверждений в отсутствие
# модели тоже нельзя — иначе сервис перестаёт быть полезным.
#
# Правило общее и морфологическое, словаря технологий здесь нет: головное слово должно быть
# ПРОЦЕССОМ или РЕЗУЛЬТАТОМ процесса — отглагольным существительным или существительным
# с продуктивным техническим суффиксом, — и при нём должно стоять содержательное определение.
# Именная группа с обычным бытовым головным словом такому описанию не отвечает и подтверждённой
# без семантической проверки не становится.
HIGH_CERTAINTY_HEAD_SUFFIXES = ("tion", "sion", "ization", "isation", "ography", "graphy", "ometry",
                                "metry", "onics", "scopy", "lysis", "thesis", "genesis", "therapy",
                                "ology", "istics")


def high_certainty_head(head: str) -> bool:
    """Головное слово называет процесс или его результат: отглагольное существительное либо
    существительное с продуктивным техническим суффиксом."""
    if head in GERUND_NOUNS:
        return True
    return len(head) > 6 and head.endswith(HIGH_CERTAINTY_HEAD_SUFFIXES)


def deterministic_high_certainty(assessment: "CandidateAssessment") -> bool:
    """Достаточно ли детерминированных оснований считать фразу названием технологии без модели.

    Требуется всё сразу: чистая оценка без флагов неоднозначности, головное слово-процесс и хотя бы
    одно содержательное определение при нём. Правило узкое намеренно: ошибочное подтверждение дороже
    отложенного."""
    if assessment.rejected or assessment.ambiguous:
        return False
    if assessment.is_product_or_brand or assessment.is_generic_phrase:
        return False
    words = _words(assessment.original_phrase)
    if len(words) < 2 or not high_certainty_head(words[-1]):
        return False
    return len(_words(assessment.canonical_name)) >= 2


def specific_tokens(phrase: str, seed_tokens: set[str]) -> list[str]:
    """Токены, несущие собственное техническое содержание: не общая лексика, не слова самого запроса,
    не бренд, не рубрика. Наличие хотя бы одного означает, что фраза не «просто общая»."""
    weak = (GENERIC_MODIFIERS | PARTICIPLE_MODIFIERS | FUNCTION_WORDS | TECHNICAL_CATEGORY_HEADS
            | DISCOURSE_HEADS | DISCOURSE_ARTIFACT_HEADS | MODEL_CLASS_HEADS | SOFT_PLUMBING
            | BRANDS | seed_tokens)
    return [t for t in _words(phrase) if t not in weak and len(t) > 2]


@dataclass
class CandidateAssessment:
    """Строгая схема оценки кандидата. Одинакова для детерминированного пути и для модели."""
    original_phrase: str
    canonical_name: str
    is_technology: bool
    domain_relevance: float
    is_product_or_brand: bool
    is_generic_phrase: bool
    merge_key: str | None
    reject_reason: str | None
    # Дополнительно к обязательному контракту: во сколько обошлись широкие лексические правила.
    specificity: float = 1.0
    ambiguity_flags: list[str] = field(default_factory=list)
    source: str = "deterministic"        # deterministic | llm

    @property
    def rejected(self) -> bool:
        return bool(self.reject_reason)

    @property
    def ambiguous(self) -> bool:
        return bool(self.ambiguity_flags)

    def to_dict(self) -> dict:
        return asdict(self)


def domain_relevance(phrase: str, seed_tokens: set[str], channels: dict | None, lit_docs: int) -> float:
    """Отношение фразы к направлению запроса по измеримым признакам, без эмбеддингов.

    База 0.5. Фраза, встреченная в корпусе запроса и сразу в нескольких каналах, ближе к направлению;
    фраза из одного канала и без общих слов с направлением — дальше."""
    channels = channels or {}
    present = sum(1 for v in channels.values() if v)
    score = 0.5
    if lit_docs >= 1:
        score += 0.20
    if present >= 2:
        score += 0.20
    if set(_words(phrase)) & seed_tokens:
        score += 0.10
    if present <= 1 and not (set(_words(phrase)) & seed_tokens) and lit_docs == 0:
        score -= 0.40
    return round(max(0.0, min(1.0, score)), 3)


def _is_meta_concept(toks: list[str]) -> bool:
    """Все слова фразы — служебные, рубричные или дискурсивные: технологии здесь нет.

    Правило смотрит на фразу целиком и потому не запрещает ни одного слова само по себе."""
    if not toks:
        return False
    vocabulary = (RESEARCH_DISCOURSE_WORDS | DISCOURSE_HEADS | DISCOURSE_ARTIFACT_HEADS
                  | TECHNICAL_CATEGORY_HEADS | MODEL_CLASS_HEADS | FUNCTION_WORDS
                  | intent.INTENT_WORDS)
    return all(t in vocabulary for t in toks)


def assess(phrase: str, seed_tokens: set[str], exclusions: set[str], *,
           channels: dict | None = None, lit_docs: int = 0) -> CandidateAssessment:
    """Детерминированная высокорекольная оценка одного кандидата.

    Жёсткий отказ — только при сильной улике. Широкие лексические совпадения понижают специфичность
    и поднимают флаг неоднозначности, а решение об отказе принимается по произведению специфичности
    и связи с направлением: слабыми должны оказаться оба признака сразу."""
    toks = _words(phrase)
    hyphenated = _tokens(phrase)
    rel = domain_relevance(phrase, seed_tokens, channels, lit_docs)
    specific = specific_tokens(phrase, seed_tokens)
    a = CandidateAssessment(original_phrase=phrase, canonical_name=phrase, is_technology=True,
                            domain_relevance=rel, is_product_or_brand=False, is_generic_phrase=False,
                            merge_key=merge_key(phrase), reject_reason=None)

    # ---- сильные улики: жёсткий отказ ----
    if not (MIN_TOKENS <= len(toks) <= MAX_TOKENS_IN_PHRASE):
        a.is_technology, a.reject_reason = False, "не похоже на название технологии: неподходящая длина фразы"
        return a
    if all(len(t) <= 2 for t in toks):
        a.is_technology, a.reject_reason = False, "слишком короткие токены: вероятно обрывок фразы"
        return a
    brand_hits = [t for t in toks if t in UNAMBIGUOUS_BRANDS]
    if brand_hits:
        a.is_product_or_brand, a.is_technology = True, False
        a.reject_reason = f"название продукта или компании («{brand_hits[0]}»), а не технологии"
        return a
    furniture = [t for t in toks if t in REPO_FURNITURE]
    if furniture:
        a.is_generic_phrase, a.is_technology = True, False
        a.reject_reason = f"оформление репозитория («{furniture[0]}»), а не технология"
        return a
    if _is_gerund_verb(toks[0]):
        a.is_technology = False
        a.reject_reason = f"фраза начинается с действия «{toks[0]}»: это не название технологии"
        return a
    if _is_adjective(hyphenated[-1]):
        a.is_technology = False
        a.reject_reason = f"фраза заканчивается прилагательным «{hyphenated[-1]}»: это не именная группа"
        return a
    head = toks[-1]
    if head in DISCOURSE_HEADS:
        a.is_generic_phrase, a.is_technology = True, False
        a.reject_reason = f"головное слово «{head}» относится к рынку или обзору, а не к технологии"
        return a
    ambiguous_brands = [t for t in toks if t in AMBIGUOUS_BRANDS]
    if ambiguous_brands and not [t for t in specific if t not in AMBIGUOUS_BRANDS]:
        # Контекст решает: бренд плюс только общая лексика — это название продукта, а не технология.
        a.is_product_or_brand, a.is_technology = True, False
        a.reject_reason = (f"в этом контексте «{ambiguous_brands[0]}» читается как название продукта: "
                           "других технических слов в фразе нет")
        return a
    if not specific:
        a.is_generic_phrase, a.is_technology = True, False
        a.reject_reason = "в фразе нет ни одного слова с собственным техническим содержанием"
        return a
    # P1-B. Мета-понятие отбраковывается по СТРУКТУРЕ ФРАЗЫ ЦЕЛИКОМ, а не по глобальному запрету
    # токенов.
    #
    # Первая версия ремонта искала маркеры намерения подстрокой и этим выбивала настоящие
    # технологии: «weak signal detection», «weak signal propagation reporter», «early warning
    # system», «early warning radar». Разбор ЗАПРОСА пользователя и ИДЕНТИЧНОСТЬ КАНДИДАТА — разные
    # задачи, и словарь первой не может быть чёрным списком для второй.
    #
    # Теперь фраза считается мета-понятием, только если КАЖДОЕ её слово взято из служебной,
    # рубричной или дискурсивной лексики: «horizon scanning», «future research directions»,
    # «related work section», «technology trend report». Стоит появиться хотя бы одному слову с
    # собственным техническим содержанием — фраза остаётся кандидатом.
    if _is_meta_concept(toks):
        a.is_generic_phrase, a.is_technology = True, False
        a.reject_reason = "фраза описывает разговор о предмете, а не саму технологию"
        return a
    if set(toks) & exclusions:
        a.reject_reason = "исключено планом запроса как не относящееся к направлению"
        return a

    # ---- слабые улики: штраф специфичности и флаг, но не отказ ----
    specificity = 1.0
    if head in TECHNICAL_CATEGORY_HEADS:
        specificity -= PENALTY_TECHNICAL_HEAD
        a.ambiguity_flags.append(f"общее головное слово «{head}»")
    if head in MODEL_CLASS_HEADS:
        specificity -= PENALTY_MODEL_CLASS_HEAD
        a.ambiguity_flags.append(f"головное слово «{head}» это класс моделей")
    soft = [t for t in toks if t in SOFT_PLUMBING]
    if soft:
        specificity -= PENALTY_SOFT_PLUMBING
        a.ambiguity_flags.append(f"техническая обвязка «{soft[0]}»")
    if ambiguous_brands:
        specificity -= PENALTY_AMBIGUOUS_BRAND
        a.ambiguity_flags.append(f"«{ambiguous_brands[0]}» может быть названием продукта")
    if len(specific) >= 2:
        specificity += 0.10
    a.specificity = round(max(0.0, min(1.0, specificity)), 3)
    a.is_generic_phrase = a.specificity < 0.5 and len(specific) < 2

    if a.specificity < MIN_SPECIFICITY and rel < MIN_DOMAIN_RELEVANCE:
        a.reject_reason = (f"слабо и по технической специфичности ({a.specificity:.2f} < {MIN_SPECIFICITY}), "
                           f"и по связи с направлением ({rel:.2f} < {MIN_DOMAIN_RELEVANCE})")
    return a


# ---------- совместимость с R1 ----------

def deterministic_reason(phrase: str, seed_tokens: set[str], exclusions: set[str]) -> str | None:
    """Причина отбраковки или None. Тонкая обёртка над `assess` для простых вызовов и тестов."""
    return assess(phrase, seed_tokens, exclusions, channels={"литература": 1, "репозитории": 1}, lit_docs=1).reject_reason


def filter_candidates(cands: list[dict], seed_tokens: set[str], exclusions: set[str] | None = None
                      ) -> tuple[list[dict], list[dict]]:
    """Делит кандидатов на прошедших и отбракованных, проставляя каждому полную схему оценки."""
    exclusions = {e.lower() for e in (exclusions or set())}
    keep, drop = [], []
    for c in cands:
        a = assess(c["phrase"], seed_tokens, exclusions, channels=c.get("channels"),
                   lit_docs=c.get("recent_docs", 0))
        c["assessment"] = a.to_dict()
        c.setdefault("original_phrase", c["phrase"])
        c["canonical_label"] = a.canonical_name
        c["merge_key"] = a.merge_key
        c["high_certainty"] = deterministic_high_certainty(a)
        if a.rejected:
            c["noise_reason"] = a.reject_reason
            drop.append(c)
        else:
            keep.append(c)
    return keep, drop


# ---------- пакетная семантическая нормализация моделью ----------

CANON_PROMPT = """Ты нормализуешь список фраз, извлечённых из научных работ и репозиториев по теме «{domain}».

Для каждой фразы определи ТОЛЬКО семантику: является ли она названием технологии, насколько она
относится к теме, не является ли она названием продукта или общей рубрикой, и какой у неё
канонический вид. Ты НЕ решаешь, слабый это сигнал или нет, и не оцениваешь перспективность.
Ничего не придумывай и не добавляй фраз, которых нет в списке.

Верни СТРОГО JSON-массив, по одному объекту на каждый номер из списка:
[{"i": <номер>, "canonical_name": "<каноническое название той же технологии>",
  "is_technology": true|false, "domain_relevance": <число от 0 до 1>,
  "is_product_or_brand": true|false, "is_generic_phrase": true|false,
  "merge_key": "<одинаковый ключ у фраз про одно и то же явление, иначе null>",
  "reject_reason": "<краткая причина по-русски, если фразу брать не стоит, иначе null>"}]

Список:
{items}"""

# REPAIR B. Семантическая проверка ограничена по времени и идёт порциями.
#
# Живая проверка показала архитектурный дефект: один пакетный вызов на все 60 кандидатов упирался
# в таймаут, возвращал НИЧЕГО, и весь набор оставался DETERMINISTIC_ONLY — то есть максимум ACCEPT
# был равен нулю при исправном контракте безопасности.
#
# Числа выбраны из общего рассуждения о бюджете, а не подгонкой под конкретные запросы:
#   * порция в 8 фраз — это примерно 8 x 40 = 320 токенов ответа, один вызов укладывается в секунды;
#   * короткий список в 24 кандидата — три порции, то есть единицы секунд при исправной модели;
#   * на фазу отводится доля сетевого бюджета, а каждый вызов ограничен её остатком;
#   * ниже минимального бюджета вызов не начинается вовсе.
# Измерено на холодных прогонах владельца (d948): порция из ВОСЬМИ фраз стабильно не укладывалась
# и отваливалась по таймауту около 10 с, а восстановительная порция из ЧЕТЫРЁХ отвечала за 5–6 с.
# Поэтому порция начинается сразу с четырёх: первая попытка становится той, которая на практике
# успевает, а не той, которая заведомо не успевает. Дальше работает прежний механизм — после успеха
# берётся следующая порция, после таймаута размер делится пополам, дедлайн фазы и резерв не растут.
LLM_BATCH = int(os.getenv("WSIGNALS_CANON_BATCH", "4"))
SEMANTIC_SHORTLIST = int(os.getenv("WSIGNALS_SEMANTIC_SHORTLIST", "24"))
SEMANTIC_PHASE_SHARE = float(os.getenv("WSIGNALS_SEMANTIC_PHASE_SHARE", "0.15"))
SEMANTIC_MIN_CALL_BUDGET = float(os.getenv("WSIGNALS_SEMANTIC_MIN_CALL", "3.0"))

# P1-C. Порции РАСТУТ от малого, а не начинаются с максимальной.
#
# Измеренный дефект: таймаут порции вычислялся как min(TIMEOUT, весь остаток фазы), поэтому ПЕРВЫЙ
# же вызов получал 100 % возможности семантической проверки. На живых прогонах владельца это дало
# ровно один вызов в каждом запуске: две трети запусков не проверили ничего, треть проверила восемь
# кандидатов. Машинерия частичного успеха существовала, но включиться не могла.
#
# Стратегия: начинать с маленькой порции, расширять её после успеха и сжимать после таймаута.
# Маленький первый вызов стоит дёшево и почти всегда укладывается, поэтому «ноль проверенных»
# перестаёт быть обычным исходом; после успеха размер растёт, и при быстрой модели проверка
# доходит до тех же объёмов. Ни один вызов не получает весь остаток фазы: доля ограничена, и после
# таймаута гарантированно остаётся бюджет на меньшую попытку.
#
# Числа выведены из архитектуры, а не подобраны под три наблюдавшихся запроса: порядок «начать с
# минимума, удваивать при успехе, делить пополам при таймауте» не зависит от конкретной задержки
# провайдера и сам подстраивается под неё.
SEMANTIC_MIN_BATCH = int(os.getenv("WSIGNALS_SEMANTIC_MIN_BATCH", "2"))
SEMANTIC_MAX_ATTEMPTS = int(os.getenv("WSIGNALS_SEMANTIC_MAX_ATTEMPTS", "8"))

# Провайдер имеет ФИКСИРОВАННУЮ часть задержки. Живое измерение владельца: порция из восьми фраз
# отвечает за 6–8 с, и эта величина почти не зависит от размера порции. Поэтому «начать с малого»
# не помогает: маленькая порция стоит столько же, сколько большая, а вот предельный таймаут вызова,
# срезанный долей остатка, оказывается НИЖЕ фиксированной задержки — и не успевает ничего.
#
# Первая версия растущих порций именно так и сломалась: доля 0,6 от восьмисекундной фазы давала
# 4,8 с на вызов, и при фиксированной задержке 5–7 с проверка не успевала вообще, тогда как один
# большой вызов до ремонта укладывался.
#
# Поэтому:
#   * фазе гарантируется ПОЛ, которого хватает и на один реалистичный вызов, и на восстановление;
#   * первый вызов идёт полной порцией и получает всё, кроме явного резерва восстановления;
#   * резерв рассчитан на повтор половинной порцией у провайдера с фиксированной задержкой.
#
# Пол ограничен долей остатка, чтобы на коротком бюджете фаза не съела всё оставшееся время.
SEMANTIC_PHASE_FLOOR = float(os.getenv("WSIGNALS_SEMANTIC_PHASE_FLOOR", "18.0"))
SEMANTIC_PHASE_MAX_SHARE = float(os.getenv("WSIGNALS_SEMANTIC_PHASE_MAX_SHARE", "0.35"))
SEMANTIC_RECOVERY_RESERVE = float(os.getenv("WSIGNALS_SEMANTIC_RECOVERY_RESERVE", "8.0"))


def _validate_row(row, by_index: dict) -> tuple[dict, CandidateAssessment] | None:
    """Строгая проверка одного элемента ответа модели. Любое нарушение схемы — элемент отбрасывается.

    bool в Python — подкласс int, поэтому `"i": true` раньше молча становился индексом 1, а
    `"domain_relevance": true` — оценкой 1.0. Ни то, ни другое не является ответом по схеме."""
    if not isinstance(row, dict) or isinstance(row.get("i"), bool) or not isinstance(row.get("i"), int):
        return None
    cand = by_index.get(row["i"])
    if cand is None:
        return None                                   # индекса не было во входе: выдумка
    canon = row.get("canonical_name")
    if not isinstance(canon, str) or not (1 < len(canon.strip()) <= MAX_LEN):
        canon = cand["phrase"]
    rel = row.get("domain_relevance")
    if isinstance(rel, bool) or not isinstance(rel, (int, float)) or not 0.0 <= float(rel) <= 1.0:
        return None
    flags = [row.get("is_technology"), row.get("is_product_or_brand"), row.get("is_generic_phrase")]
    if not all(isinstance(f, bool) for f in flags):
        return None
    key = row.get("merge_key")
    if key is not None and not isinstance(key, str):
        return None
    reason = row.get("reject_reason")
    if reason is not None and not isinstance(reason, str):
        return None
    is_tech, is_brand, is_generic = flags
    if not is_tech and not reason:
        reason = "модель не считает фразу названием технологии"
    if is_brand and not reason:
        reason = "название продукта или компании, а не технологии"
    if is_generic and not reason:
        reason = "общая рубрика, а не технология"
    if float(rel) < MIN_DOMAIN_RELEVANCE and not reason:
        reason = f"слабая связь с направлением запроса: {float(rel):.2f} < {MIN_DOMAIN_RELEVANCE}"
    return cand, CandidateAssessment(
        original_phrase=cand.get("original_phrase", cand["phrase"]),
        canonical_name=re.sub(r"\s+", " ", canon).strip(), is_technology=is_tech,
        domain_relevance=round(float(rel), 3), is_product_or_brand=is_brand, is_generic_phrase=is_generic,
        merge_key=(key.strip().lower() or None) if isinstance(key, str) else merge_key(cand["phrase"]),
        reject_reason=(reason.strip()[:200] if reason else None), source="llm")


def _validate_rows(rows, by_index: dict) -> list[tuple[dict, CandidateAssessment]]:
    """Все строки ответа через `_validate_row`, по одной оценке на кандидата.

    Если модель вернула для одного индекса больше одной строки, ответ по этому индексу
    неоднозначен: раньше последняя строка перезаписывала оценку, а причина отказа от первой
    оставалась, и кандидат одновременно числился подтверждённым и отбракованным, а счётчик
    проверенных рос дважды. Теперь все строки такого индекса отбрасываются — кандидат
    остаётся непроверенным (fail-closed), как при отсутствующей строке."""
    counts: dict[int, int] = {}
    for row in rows if isinstance(rows, list) else []:
        i = row.get("i") if isinstance(row, dict) else None
        if isinstance(i, int) and not isinstance(i, bool):
            counts[i] = counts.get(i, 0) + 1
    out = []
    for row in rows if isinstance(rows, list) else []:
        checked = _validate_row(row, by_index)
        if checked is not None and counts.get(row["i"], 0) == 1:
            out.append(checked)
    return out


def shortlist(cands: list[dict], limit: int = SEMANTIC_SHORTLIST) -> list[dict]:
    """Детерминированный, не зависящий от разметки короткий список для дорогой семантической проверки.

    Порядок задаётся уже вычисленным всплеском и специфичностью: никаких ожидаемых технологий,
    никаких списков фраз. Остальные кандидаты остаются непроверенными и подтверждены быть не могут."""
    def key(c: dict):
        a = c.get("assessment") or {}
        return (-float(c.get("burst", 0.0)) * float(a.get("specificity", 1.0)), c.get("phrase", ""))

    return sorted(cands, key=key)[:max(0, limit)]


# ---------- разделение семантической доли: ранняя фаза + адресная поздняя проверка ----------
#
# Общая семантическая доля остаётся ровно 18 с (SEMANTIC_PHASE_FLOOR). Измерено на владельческих
# живых прогонах: ранняя фаза выдавала 18 меток, и НИ ОДНА не попадала на кандидата, готового по
# доказательствам, — короткий список строится до сбора свидетельств и о готовности ничего не знает.
# Поэтому часть доли резервируется под проверку тех кандидатов, у которых семантика осталась
# ПОСЛЕДНИМ препятствием.
#
# Почему 10 + 8, а не 6 + 12. Замерено на живом провайдере: успешная порция отвечает не быстрее
# 6,27 с, а frozen `_attempt_timeout` немонотонен — он вычитает SEMANTIC_RECOVERY_RESERVE, поэтому
# на конверте 12 с первая попытка получает лишь 4 с и гарантированно не успевает, а на 8 с получает
# все 8 с. Конверты 11–15 с образуют мёртвую полосу. 8 с — единственный проверенный живой конверт.
EARLY_SEMANTIC_BUDGET = float(os.getenv("WSIGNALS_EARLY_SEMANTIC_BUDGET", "10.0"))
LATE_SEMANTIC_RESERVE = float(os.getenv("WSIGNALS_LATE_SEMANTIC_RESERVE", "8.0"))
# Адресная проверка стоит одного вызова на порцию: больше четырёх кандидатов в 8 с не проверить,
# а дробить порцию — значит потратить конверт на накладные расходы.
LATE_MAX_TARGETS = int(os.getenv("WSIGNALS_LATE_MAX_TARGETS", "4"))
# Нижняя граница таймаута: urllib3 отвергает неположительное значение.
LATE_MIN_TIMEOUT = 0.01


def late_reserve_seconds() -> float:
    """Сколько секунд резервируется под позднюю проверку. Ноль, если модель не настроена.

    Резервировать время под невозможный вызов нельзя: без провайдера это просто отнятый у каскада
    бюджет, а деградированное детерминированное поведение должно остаться прежним."""
    uri, _why = configured_model()
    return LATE_SEMANTIC_RESERVE if uri else 0.0


def _bounded_timeout(remaining: float):
    """Таймаут с ОБЩИМ пределом на соединение и чтение вместе.

    Что это действительно чинит. Скалярный таймаут requests ограничивает КАЖДУЮ фазу по
    отдельности, поэтому один вызов с номинальными 8 с законно длится вдвое дольше: живой замер дал
    8,73 с и 11,50 с. `urllib3.util.Timeout(total=...)` делает `read_timeout` убывающим на уже
    потраченное соединением, и этот перерасход закрывается. requests пропускает такой объект в
    адаптер без изменений (`elif isinstance(timeout, TimeoutSauce)`), новой зависимости не нужно.

    Чего это НЕ даёт, и утверждать обратное нельзя. Это НЕ абсолютная стена по времени на уровне
    процесса. Вне её остаются как минимум: ответ, который сервер отдаёт по капле (каждый отдельный
    фрагмент приходит внутри таймаута чтения, а тело целиком — нет), залипание разрешения имени в
    getaddrinfo и часть состязательных сценариев TCP. Поэтому у позднего прохода есть второй рубеж
    на уровне контракта: конверт урезается по общему сетевому дедлайну перед каждой попыткой, а
    результат, пришедший после дедлайна, отбрасывается целиком. Настоящая надёжность по времени
    подтверждается только владельческой живой проверкой."""
    from urllib3.util import Timeout as _Timeout

    # urllib3 не принимает неположительный таймаут: истёкший конверт должен дать мгновенный отказ,
    # а не исключение конфигурации. Вызывающий и так проверяет дедлайн до попытки, это второй рубеж.
    total = max(LATE_MIN_TIMEOUT, float(remaining))
    return _Timeout(total=total, connect=total, read=total)


def _semantic_budget() -> float | None:
    """Сколько секунд отведено всей семантической фазе. None — бюджета нет (офлайн-режим).

    Семантическая проверка — ОБЯЗАТЕЛЬНЫЙ шлюз допуска: без неё ни один кандидат не может быть
    принят. Поэтому фаза получает гарантированный пол, а не только долю остатка. Пол ограничен
    долей остатка сверху: на коротком бюджете фаза не имеет права забрать всё оставшееся время."""
    from . import http as http_mod

    left = http_mod.time_left()
    if left is None:
        return None
    left = max(0.0, left)
    early = min(left * SEMANTIC_PHASE_MAX_SHARE,
                max(left * SEMANTIC_PHASE_SHARE, SEMANTIC_PHASE_FLOOR))
    # Общая семантическая доля не изменилась: 18 с как было. Изменилось её РАСПРЕДЕЛЕНИЕ — часть
    # уходит адресной поздней проверке (см. LATE_SEMANTIC_RESERVE), поэтому ранняя фаза ограничена
    # сверху своей долей.
    return min(early, EARLY_SEMANTIC_BUDGET)


def _call_batch(items: list[dict], domain: str, model_uri: str, timeout: float) -> list:
    """Один ограниченный по времени вызов нормализации. Возвращает разобранные строки ответа."""
    listing = "\n".join(f"{i}. {c['phrase']}" for i, c in enumerate(items))
    r = requests.post(COMPLETION_URL, timeout=timeout,
                      headers={"Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}"},
                      json={"modelUri": model_uri,
                            "completionOptions": {"temperature": TEMPERATURE, "maxTokens": 1500,
                                                  "reasoningOptions": {"mode": "DISABLED"}},
                            "messages": [{"role": "user", "text": CANON_PROMPT
                                          .replace("{domain}", domain or "технологии")
                                          .replace("{items}", listing)}]})
    r.raise_for_status()
    text = r.json()["result"]["alternatives"][0]["message"]["text"]
    m = re.search(r"\[.*\]", text, re.S)
    return json.loads(m.group(0)) if m else []


def _late_call_batch(items: list[dict], domain: str, model_uri: str, remaining: float) -> list:
    """Один поздний вызов, жёстко ограниченный остатком конверта.

    Отличается от `_call_batch` ровно одним: таймаут — объект с общим пределом, а не скаляр.
    Промпт, схема запроса, температура, maxTokens и разбор ответа те же."""
    return _call_batch(items, domain, model_uri, _bounded_timeout(remaining))


def _attempt_timeout(remaining: float | None) -> float:
    """Сколько времени можно отдать ОДНОМУ вызову.

    Вызов получает всё, кроме резерва восстановления, — то есть остаётся реалистичным для провайдера
    с фиксированной задержкой, но не забирает всю возможность фазы: после таймаута гарантированно
    остаётся бюджет на повтор меньшей порцией.

    Если резерв выкроить уже нельзя, вызов получает остаток целиком: на таком бюджете попытка всё
    равно одна, и урезать её означало бы не проверить ничего."""
    if remaining is None:
        return TIMEOUT
    with_reserve = remaining - SEMANTIC_RECOVERY_RESERVE
    if with_reserve >= SEMANTIC_MIN_CALL_BUDGET:
        return min(TIMEOUT, with_reserve)
    return min(TIMEOUT, remaining)


def canonicalize(cands: list[dict], domain: str = "", batch: int = LLM_BATCH,
                 report: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Семантическая нормализация короткого списка растущими порциями с частичным успехом.

    Порция начинается маленькой, удваивается после успеха и делится пополам после таймаута.
    Успешные оценки фиксируются сразу и отказом следующего вызова не отменяются. Непроверенные
    кандидаты остаются DETERMINISTIC_ONLY и подтверждены быть не могут — контракт безопасности
    не ослабляется."""
    from . import http as http_mod

    telemetry = report if report is not None else {}
    telemetry.update({"configured": False, "attempted": False, "outcome": "not_configured",
                      "failure_type": None, "candidate_count_requested": 0,
                      "candidate_count_verified": 0, "candidate_count_unverified": len(cands),
                      "batches_attempted": 0, "batches_succeeded": 0, "batch_sizes": [],
                      "model": None})
    model_uri, why = configured_model()
    if not model_uri or not cands:
        telemetry["failure_type"] = why
        if why:
            log.info("семантическая нормализация без модели: %s", why)
        return cands, []
    telemetry.update({"configured": True, "model": model_uri})

    picked = shortlist(cands)
    telemetry["candidate_count_requested"] = len(picked)
    budget = _semantic_budget()
    deadline = None if budget is None else http_mod.monotonic() + budget
    ceiling = max(1, batch)
    size = ceiling                      # реалистичная для провайдера первая порция, не микропорция
    pending, verified = list(picked), 0

    while pending and telemetry["batches_attempted"] < SEMANTIC_MAX_ATTEMPTS:
        remaining = None if deadline is None else deadline - http_mod.monotonic()
        if remaining is not None and remaining < SEMANTIC_MIN_CALL_BUDGET:
            telemetry["failure_type"] = telemetry["failure_type"] or "semantic_budget_exhausted"
            break
        chunk = pending[:size]
        telemetry["attempted"] = True
        telemetry["batches_attempted"] += 1
        telemetry["batch_sizes"].append(len(chunk))
        try:
            rows = _call_batch(chunk, domain, model_uri, _attempt_timeout(remaining))
        except Exception as e:  # noqa: BLE001 — отказ порции не отменяет уже проверенных
            failure = type(e).__name__
            telemetry["failure_type"] = failure
            log.warning("порция семантической нормализации не удалась (%d фраз): %s", len(chunk), e)
            if "Timeout" in failure and size > SEMANTIC_MIN_BATCH:
                size = max(SEMANTIC_MIN_BATCH, size // 2)   # те же фразы, но меньшей порцией
                continue
            # Порция минимального размера всё равно не уложилась (или отказала не по таймауту):
            # она пропускается, а остальные кандидаты своей попытки не лишаются. Бесконечности здесь
            # нет — счётчик попыток и нижняя граница бюджета закрывают цикл.
            pending = pending[len(chunk):]
            continue
        telemetry["batches_succeeded"] += 1
        by_index = dict(enumerate(chunk))
        for cand, a in _validate_rows(rows, by_index):
            cand["assessment"] = a.to_dict()
            cand["canonical_label"] = a.canonical_name
            cand["merge_key"] = a.merge_key or merge_key(cand["phrase"])
            cand["high_certainty"] = deterministic_high_certainty(a)
            cand["canonical_source"] = model_uri
            if a.rejected:
                cand["noise_reason"] = a.reject_reason
            verified += 1
        pending = pending[len(chunk):]
        size = min(ceiling, size * 2)                         # после сжатия возвращаемся к полной

    telemetry["candidate_count_verified"] = verified
    telemetry["candidate_count_unverified"] = len(cands) - verified
    telemetry["outcome"] = ("successful" if verified and verified >= telemetry["candidate_count_requested"]
                            else "partial" if verified else "failed")
    log.info("семантическая нормализация моделью %s: порций %d/%d %s, проверено %d из %d кандидатов",
             model_uri, telemetry["batches_succeeded"], telemetry["batches_attempted"],
             telemetry["batch_sizes"], verified, len(cands))
    keep = [c for c in cands if not c.get("noise_reason")]
    dropped = [c for c in cands if c.get("noise_reason")]
    return keep, dropped


# ---------- склейка вариантов одной технологии (REPAIR 3, REPAIR B, REPAIR D) ----------

# Минимальная похожесть токенов, при которой склейка по ключу от модели считается проверенной.
LLM_MERGE_SIMILARITY = float(os.getenv("WSIGNALS_LLM_MERGE_SIMILARITY", "0.34"))


def token_similarity(a: str, b: str) -> float:
    """Жаккар по нормализованным содержательным токенам."""
    x = {_stem(t) for t in _words(a) if t not in NON_DISTINGUISHING and len(t) > 1}
    y = {_stem(t) for t in _words(b) if t not in NON_DISTINGUISHING and len(t) > 1}
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def _llm_key(cand: dict) -> str | None:
    a = cand.get("assessment") or {}
    return cand.get("merge_key") if a.get("source") == "llm" and cand.get("merge_key") else None


def _sanity_ok(lead: dict, other: dict) -> bool:
    """Детерминированная проверка склейки, предложенной моделью.

    Модель может назвать один ключ двум разным технологиям, и такая склейка необратима. Поэтому ключ
    от модели принимается только когда он подтверждается детерминированно: либо канонические названия
    совпадают после нормализации, либо фразы достаточно похожи по токенам. Ложная склейка хуже, чем
    два отдельных кандидата."""
    lead_canon = (lead.get("canonical_label") or lead["phrase"])
    other_canon = (other.get("canonical_label") or other["phrase"])
    if merge_key(lead_canon) == merge_key(other_canon):
        return True
    if merge_key(lead["phrase"]) == merge_key(other["phrase"]):
        return True
    return max(token_similarity(lead_canon, other_canon),
               token_similarity(lead["phrase"], other["phrase"])) >= LLM_MERGE_SIMILARITY


def _absorb(lead: dict, other: dict) -> None:
    """Переносит свидетельства поглощённого варианта в представителя семьи."""
    for channel, count in (other.get("channels") or {}).items():
        lead.setdefault("channels", {})
        lead["channels"][channel] = lead["channels"].get(channel, 0) + count
    lead["recent_docs"] = lead.get("recent_docs", 0) + other.get("recent_docs", 0)
    lead["prior_docs"] = lead.get("prior_docs", 0) + other.get("prior_docs", 0)
    # P1-B. Перенесённое свидетельство помечается фразой, по которой оно было найдено. Дальше
    # материализация запишет это в документ, и подтверждение через более широкий поглощённый
    # вариант станет отличимым от подтверждения через выжившее название.
    seen = {(w.get("id") or w.get("doi")) for w in lead.get("examples", [])}
    for w in other.get("examples", []):
        if (w.get("id") or w.get("doi")) not in seen and len(lead.get("examples", [])) < 5:
            w.setdefault("attached_via", other["phrase"])
            lead.setdefault("examples", []).append(w)
    names = {r.get("full_name") for r in lead.get("repos", [])}
    for r in other.get("repos", []):
        if r.get("full_name") not in names and len(lead.get("repos", [])) < 4:
            r.setdefault("attached_via", other["phrase"])
            lead.setdefault("repos", []).append(r)
    lead.setdefault("merged_from", []).append(other["phrase"])
    other["noise_reason"] = f"вариант того же явления, склеен с «{lead['phrase']}»"
    other["merged_into"] = lead["phrase"]


def _late_remaining(local_deadline: float) -> float:
    """Меньший из двух сроков: местный конверт поздней проверки и общий сетевой дедлайн."""
    from . import http as http_mod

    remaining = local_deadline - http_mod.monotonic()
    left = http_mod.time_left()
    return remaining if left is None else min(remaining, left)


# R4. Поля оценки, которые поздний проход имеет право переписать. Всё остальное — личность и
# метаданные ранжирования, сложившиеся ДО склейки: канон, ключ склейки, исходная фраза,
# специфичность, флаги неоднозначности и признаки продукта/общей рубрики. Поздний ответ модели
# приходит после склейки и пересматривать их не может: карточка не должна публиковать новую
# позднюю личность.
LATE_ASSESSMENT_FIELDS = ("source", "is_technology", "domain_relevance", "reject_reason")


def _apply_late_assessment(cand: dict, assessment) -> None:
    """Обновляет у кандидата ТОЛЬКО поля, которые читает замороженный предикат подтверждения.

    Существующая оценка берётся за основу, а не заменяется целиком: замена стирала бы канон, ключ
    склейки, специфичность и флаги неоднозначности, сложившиеся в ранней фазе."""
    fresh = assessment.to_dict()
    merged = dict(cand.get("assessment") or {})
    for field_name in LATE_ASSESSMENT_FIELDS:
        merged[field_name] = fresh.get(field_name)
    # Кандидат без ранней оценки вообще: тогда брать за основу нечего, но личность всё равно
    # берётся из самого кандидата, а не из ответа модели.
    merged.setdefault("original_phrase", cand.get("original_phrase") or cand.get("phrase"))
    merged.setdefault("canonical_name", cand.get("canonical_label")
                      or cand.get("original_phrase") or cand.get("phrase"))
    merged.setdefault("merge_key", cand.get("merge_key"))
    cand["assessment"] = merged


def verify_late(targets: list[dict], domain: str = "", report: dict | None = None,
                budget_s: float | None = None) -> int:
    """Адресная поздняя семантическая проверка. Возвращает число подтверждённых кандидатов.

    Идёт ПОСЛЕ склейки и сбора свидетельств, поэтому личность трогать нельзя: канонические имена,
    ключи склейки, поглощённые варианты, происхождение привязки, документы и счётчики источников
    здесь НЕ пишутся. Записывается ровно одно поле — `assessment`, то есть тот минимум, который
    читает замороженный предикат подтверждения.

    Модель, промпт, схема ответа, `_validate_row` и MIN_DOMAIN_RELEVANCE — те же, что у ранней
    фазы. Второй модели нет. Фактических свидетельств из ответа модели не создаётся: ответ может
    только подтвердить или не подтвердить уже собранное.

    Fail-closed: таймаут, отказ соединения, неразобранный ответ, отсутствующая или чужая строка
    оставляют кандидата неподтверждённым. Ответ, пришедший после общего сетевого дедлайна, тоже
    не подтверждает ничего: его строки отбрасываются целиком.

    Гарантия по времени здесь контрактная, а не транспортная: конверт урезается по остатку общего
    дедлайна перед каждой попыткой, а запоздавший результат не влияет на решение. Абсолютной стены
    по времени HTTP-стек не даёт (см. `_bounded_timeout`), поэтому фактическая надёжность по
    времени требует владельческой живой проверки."""
    from . import http as http_mod

    telemetry = {"configured": False, "attempted": False, "target_count": len(targets),
                 "targets": [str(c.get("original_phrase") or c.get("phrase")) for c in targets],
                 "attempt_count": 0, "batch_sizes": [], "valid_rows": 0, "verified_count": 0,
                 "negative_or_uncertain_count": 0, "failure_type": None,
                 "budget_s": None, "wall_s": 0.0, "deadline_exhausted": False}
    if report is not None:
        report.update(telemetry)
    if not targets:
        return 0
    model_uri, why = configured_model()
    if not model_uri:
        telemetry["failure_type"] = why
        if report is not None:
            report.update(telemetry)
        return 0
    telemetry["configured"] = True

    # R2. Запрошенный конверт ВСЕГДА урезается до реального остатка общего дедлайна, в том числе
    # когда вызывающий передал budget_s явно. Иначе поздний проход обещал бы себе 8 с в момент,
    # когда до сетевого дедлайна осталось две: запрос ушёл бы за стену запроса.
    budget = LATE_SEMANTIC_RESERVE if budget_s is None else float(budget_s)
    left = http_mod.time_left()
    if left is not None:
        budget = min(budget, max(0.0, left))
    telemetry["budget_s"] = round(budget, 2)
    deadline = http_mod.monotonic() + budget
    started = http_mod.monotonic()

    chunk = list(targets[:LATE_MAX_TARGETS])
    verified = 0
    while chunk:
        # R2. Перед КАЖДОЙ попыткой пересчитываются ОБА срока — местный конверт и общий сетевой
        # дедлайн, — и берётся меньший. Иначе поздний проход живёт по своему таймеру и способен
        # уйти за общую стену запроса.
        remaining = _late_remaining(deadline)
        if remaining < SEMANTIC_MIN_CALL_BUDGET:
            telemetry["deadline_exhausted"] = True
            telemetry["failure_type"] = telemetry["failure_type"] or "late_budget_exhausted"
            break
        telemetry["attempted"] = True
        telemetry["attempt_count"] += 1
        telemetry["batch_sizes"].append(len(chunk))
        try:
            rows = _late_call_batch(chunk, domain, model_uri, remaining)
        except Exception as e:  # noqa: BLE001 — отказ поздней порции не отменяет ничего собранного
            telemetry["failure_type"] = type(e).__name__
            log.warning("поздняя семантическая проверка не удалась (%d фраз): %s", len(chunk), e)
            if "Timeout" in type(e).__name__ and len(chunk) > SEMANTIC_MIN_BATCH:
                chunk = chunk[:max(SEMANTIC_MIN_BATCH, len(chunk) // 2)]
                continue
            break
        # R2. Ответ, пришедший ПОСЛЕ любого из двух сроков, решения менять не имеет права: его
        # строки отбрасываются целиком. Транспорт не даёт абсолютной гарантии по времени (см.
        # докстроку модуля), поэтому запоздавший результат отсекается здесь, на уровне контракта.
        if _late_remaining(deadline) <= 0.0:
            telemetry["deadline_exhausted"] = True
            telemetry["failure_type"] = "late_result_after_deadline"
            log.warning("поздний ответ получен после дедлайна: строки отброшены (%d фраз)", len(chunk))
            break
        by_index = dict(enumerate(chunk))
        for cand, a in _validate_rows(rows, by_index):
            telemetry["valid_rows"] += 1
            _apply_late_assessment(cand, a)
            if semantically_verified(cand):
                verified += 1
            else:
                telemetry["negative_or_uncertain_count"] += 1
        break
    telemetry["verified_count"] = verified
    telemetry["wall_s"] = round(http_mod.monotonic() - started, 2)
    if report is not None:
        report.update(telemetry)
    return verified


def semantically_verified(cand: dict) -> bool:
    """Кандидат получил ПОДТВЕРЖДАЮЩУЮ оценку модели.

    Тот же предикат, что и у `query.verification_status`: модель, признала технологией, не отвергла
    и связала с направлением. Отвергнутая моделью фраза подтверждённой не считается, поэтому склейка
    не может превратить отказ модели в подтверждение (P1-D)."""
    a = cand.get("assessment") or {}
    return bool(a.get("source") == "llm" and a.get("is_technology") and not a.get("reject_reason")
                and float(a.get("domain_relevance", 0.0)) >= MIN_DOMAIN_RELEVANCE)


def collapse(cands: list[dict]) -> tuple[list[dict], list[dict]]:
    """Схлопывает варианты одного явления до дорогого обогащения и до ранжирования.

    Представителем семьи становится кандидат с наибольшим всплеском. Оригинальные фразы и происхождение
    свидетельств сохраняются: каналы, работы и репозитории объединяются, а поглощённые фразы попадают
    в `merged_from`. Склейка по ключу от модели проходит детерминированную проверку."""
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for c in cands:
        key = _llm_key(c) or merge_key(c["phrase"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(c)
    kept, merged_away = [], []
    for key in order:
        # P1-D. Представителем семьи становится ПОДТВЕРЖДЁННЫЙ моделью вариант, если он есть.
        #
        # Измеренный дефект: представитель выбирался только по всплеску, а короткий список для
        # семантической проверки строится по произведению всплеска и специфичности. Поэтому
        # подтверждённый кандидат мог быть поглощён неподтверждённым родственником с бо́льшим
        # всплеском, и семья теряла уже оплаченное подтверждение, становясь DETERMINISTIC_ONLY.
        # Внутри уже разрешённой семьи подтверждение доминирует над его отсутствием; порядок
        # внутри каждой из двух групп прежний, поэтому результат остаётся детерминированным
        # и не зависит от порядка входа.
        group = sorted(groups[key],
                       key=lambda c: (0 if semantically_verified(c) else 1,
                                      -c.get("burst", 0.0), len(c["phrase"])))
        lead, rest = group[0], group[1:]
        lead["merge_key"] = key
        lead.setdefault("merged_from", [])
        rejected_merges = []
        for other in rest:
            if _llm_key(other) and not _sanity_ok(lead, other):
                # Ключ от модели не подтверждён: кандидат остаётся отдельным.
                other["merge_key"] = merge_key(other["phrase"])
                other.setdefault("merge_notes", []).append(
                    f"склейка с «{lead['phrase']}» по ключу модели отклонена детерминированной проверкой")
                rejected_merges.append(other)
                continue
            _absorb(lead, other)
            merged_away.append(other)
        kept.append(lead)
        # Отклонённые склейки пересобираются как самостоятельные семьи.
        if rejected_merges:
            sub_kept, sub_merged = collapse(rejected_merges)
            kept.extend(sub_kept)
            merged_away.extend(sub_merged)
    return kept, merged_away


def duplicate_rate(phrases: list[str]) -> float:
    """Доля фраз, которые являются вариантами уже присутствующего в списке явления.

    Используется для измерения качества склейки до и после: 0.0 означает, что вариантов нет."""
    if not phrases:
        return 0.0
    keys = [merge_key(p) for p in phrases]
    return round(1 - len(set(keys)) / len(keys), 3)
