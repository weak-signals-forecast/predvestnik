"""Разбор произвольного запроса в структурированный план поиска (PHASE 4).

До ремонта запрос разбирался словарём из 14 направлений, а всё, что в словарь не попало, уходило
в OpenAlex по-русски. Теперь есть явный путь: одна **явно сконфигурированная** модель YandexGPT
раскладывает свободный русский запрос в схему. Автоматического выбора модели нет: модель закреплена
(`RUNTIME_YANDEX_MODEL`), может быть подтверждена переменной окружения, сверяется со списком разрешённых
и пишется в журнал целиком (`gpt://<folder>/<model>`).

Модель здесь только раскладывает запрос. Она не приносит фактов, не называет технологии и не влияет
на решение: всё, что она возвращает, это поисковые строки, по которым дальше ищут открытые источники.

Контракт ответа:
    {"domain": str, "english_queries": [str], "russian_queries": [str],
     "synonyms": [str], "exclusions": [str]}

Без ключей, при ошибке сети, при неразобранном JSON или при несоответствии схеме используется
детерминированный разбор (словарь направлений `query.parse_query`). Запрос из-за недоступности
модели не падает никогда.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

import requests

log = logging.getLogger("wsignals.decompose")

# Модель YandexGPT, закреплённая за прогоном: проверена живым вызовом REST (HTTP 200, версия 25.03.2025).
# Идентификатор уходит в modelUri как есть: суффикс версии (`/latest`, `/rc`) не дописывается никогда.
RUNTIME_YANDEX_MODEL = "yandexgpt-5-lite"
# Разрешён ровно один идентификатор: ни автоподбора, ни молчаливой подмены модели быть не может.
ALLOWED_YANDEX_MODELS = {RUNTIME_YANDEX_MODEL}
DEFAULT_MODEL = RUNTIME_YANDEX_MODEL
COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
TEMPERATURE = 0.1
MAX_TOKENS = 700
TIMEOUT = float(os.getenv("WSIGNALS_LLM_TIMEOUT", "20"))

PROMPT = """Ты помогаешь поисковой системе по научно-техническим источникам.
Разложи запрос пользователя в план поиска. Не придумывай технологий, названий компаний и фактов:
верни только поисковые формулировки.

ВАЖНО. Пользователь ищет зарождающиеся технологии ВНУТРИ предметной области. Слова, которыми он
описывает задачу поиска — «слабые сигналы», «зарождающиеся тренды», «перспективные решения», «emerging»,
«weak signals», «future», — это ОПИСАНИЕ ЗАДАЧИ, а не предмет поиска. В поисковые фразы их включать
нельзя: иначе поиск найдёт работы про обнаружение слабых сигналов вместо технологий области.
Ищи по предметной области.

Предметная область запроса: {domain}

Верни СТРОГО один JSON-объект без пояснений и без markdown:
{"domain": "краткое название направления по-русски",
 "english_queries": ["3-6 английских поисковых фраз по 1-4 слова"],
 "russian_queries": ["2-4 русские поисковые фразы"],
 "synonyms": ["0-6 синонимов и близких терминов, английских или русских"],
 "exclusions": ["0-6 слов, которые указывают на НЕ относящийся к запросу результат"]}

Исходный запрос пользователя (для контекста): {query}"""

MAX_ITEMS = 8
MAX_LEN = 80


def _without_intent(items: list[str]) -> list[str]:
    """Убирает из поисковых фраз пересказ намерения пользователя (REPAIR A)."""
    from . import intent as intent_mod

    out = []
    for item in items:
        if intent_mod.is_intent_restatement(item):
            continue
        cleaned = intent_mod.strip_intent_tokens(item).strip()
        if cleaned and not intent_mod.is_intent_restatement(cleaned):
            out.append(cleaned)
    return list(dict.fromkeys(out))


def _is_latin(text: str) -> bool:
    low = (text or "").lower()
    return bool(re.search(r"[a-z]", low)) and not re.search(r"[а-яё]", low)


@dataclass
class QueryPlan:
    """План поиска. `source` показывает, кем он построен, и уходит в журнал запроса."""
    domain: str
    english_queries: list[str] = field(default_factory=list)
    russian_queries: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    source: str = "deterministic"          # llm | deterministic
    model: str | None = None               # полный идентификатор модели, если работала модель
    fallback_reason: str | None = None
    intent: dict = field(default_factory=dict)   # разделение намерения и области (REPAIR A)

    def seeds(self, limit: int = 4) -> list[str]:
        """Поисковые фразы для OpenAlex и GitHub. Язык этих источников английский, поэтому кириллические
        остатки сюда не попадают: искать «способы охлаждения» в OpenAlex бессмысленно.

        Если английских фраз не оказалось совсем, отдаётся кириллица — иначе искать было бы нечего,
        — и это состояние видно в `untranslated`."""
        latin = [s.strip() for s in [*self.english_queries, *self.synonyms]
                 if s.strip() and _is_latin(s)]
        if latin:
            return list(dict.fromkeys(latin))[:limit]
        return list(dict.fromkeys(s.strip() for s in self.russian_queries if s.strip()))[:limit]

    @property
    def untranslated(self) -> bool:
        """Поисковых фраз на английском нет: запрос не удалось перевести ни словарём, ни моделью."""
        return not any(_is_latin(s) for s in [*self.english_queries, *self.synonyms])

    def to_dict(self) -> dict:
        return {"domain": self.domain, "english_queries": self.english_queries,
                "russian_queries": self.russian_queries, "synonyms": self.synonyms,
                "exclusions": self.exclusions, "source": self.source, "model": self.model,
                "fallback_reason": self.fallback_reason, "intent": dict(self.intent)}


def configured_model_name() -> str:
    """Имя модели рантайма — единственная точка её выбора для всего сервиса.

    Значение либо задано в окружении явно, либо равно закреплённому `RUNTIME_YANDEX_MODEL`.
    Перебора моделей, обращения к списку моделей облака и подстановки версии здесь нет."""
    return os.getenv("YANDEX_QUERY_MODEL") or os.getenv("YANDEX_MODEL") or DEFAULT_MODEL


def configured_model() -> tuple[str | None, str | None]:
    """Возвращает (model_uri, причина отказа). Выбор модели только явный, из переменных окружения."""
    if (os.getenv("LLM_PROVIDER", "").lower() or None) != "yandexgpt":
        return None, "LLM_PROVIDER не равен yandexgpt"
    key, folder = os.getenv("YANDEX_API_KEY"), os.getenv("YANDEX_FOLDER_ID")
    if not key or not folder:
        return None, "не заданы YANDEX_API_KEY или YANDEX_FOLDER_ID"
    model = configured_model_name()
    if model not in ALLOWED_YANDEX_MODELS:
        return None, f"модель {model} не входит в разрешённый ТЗ список YandexGPT"
    return f"gpt://{folder}/{model}", None


def _clean(items, limit: int = MAX_ITEMS) -> list[str]:
    out = []
    for x in items if isinstance(items, list) else []:
        if not isinstance(x, str):
            continue
        s = re.sub(r"\s+", " ", x).strip().strip('"').strip()
        if 1 < len(s) <= MAX_LEN:
            out.append(s)
    return list(dict.fromkeys(out))[:limit]


def validate(payload: dict) -> QueryPlan | None:
    """Строгая проверка схемы. Лишние поля отбрасываются, недостающее обязательное поле это отказ."""
    if not isinstance(payload, dict):
        return None
    domain = payload.get("domain")
    if not isinstance(domain, str) or not domain.strip():
        return None
    en = _clean(payload.get("english_queries"))
    if not en:
        return None                                    # без английских фраз искать нечем: это не валидный план
    return QueryPlan(domain=re.sub(r"\s+", " ", domain).strip()[:MAX_LEN], english_queries=en,
                     russian_queries=_clean(payload.get("russian_queries")),
                     synonyms=_clean(payload.get("synonyms")), exclusions=_clean(payload.get("exclusions")))


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _call_model(query: str, model_uri: str, domain: str = "") -> str | None:
    from . import http as http_mod

    timeout = http_mod.call_budget(TIMEOUT)
    if timeout is None:
        raise TimeoutError("бюджет запроса исчерпан: разбор запроса моделью не начинается")
    r = requests.post(COMPLETION_URL, timeout=timeout,
                      headers={"Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}"},
                      json={"modelUri": model_uri,
                            "completionOptions": {"temperature": TEMPERATURE, "maxTokens": MAX_TOKENS,
                                                  "reasoningOptions": {"mode": "DISABLED"}},
                            "messages": [{"role": "user", "text": PROMPT.replace("{domain}", domain or query)
                                          .replace("{query}", query)}]})
    r.raise_for_status()
    return r.json()["result"]["alternatives"][0]["message"]["text"]


def decompose(query: str, deterministic) -> QueryPlan:
    """Раскладывает запрос. `deterministic` это функция запасного разбора (`query.parse_query`).

    Намерение пользователя отделяется от предметной области ДО обращения к модели, и модель получает
    область явным полем (REPAIR A). Оба пути — модельный и детерминированный — обязаны соблюдать один
    и тот же контракт: искать по области, а не по словам о задаче.

    Никакая ошибка модели не поднимается наружу: план всегда возвращается."""
    from . import intent as intent_mod

    parsed_intent = intent_mod.split(query)
    model_uri, why = configured_model()
    if model_uri:
        log.info("разбор запроса моделью %s (temperature=%s), предметная область: %s",
                 model_uri, TEMPERATURE, parsed_intent.domain)
        try:
            payload = _parse_json(_call_model(query, model_uri, domain=parsed_intent.domain))
            plan = validate(payload) if payload is not None else None
            if plan is not None:
                plan.source, plan.model = "llm", model_uri
                plan.intent = parsed_intent.to_dict()
                plan.english_queries = _without_intent(plan.english_queries)
                plan.synonyms = _without_intent(plan.synonyms)
                if not plan.english_queries:
                    # Модель вернула только пересказ намерения: это не план поиска.
                    why = "модель вернула только пересказ намерения пользователя"
                    log.warning("%s: %s", model_uri, why)
                else:
                    return plan
            else:
                why = "ответ модели не прошёл проверку схемы"
                log.warning("%s: %s", model_uri, why)
        except Exception as e:  # noqa: BLE001 — недоступность модели не должна валить запрос
            why = f"модель недоступна: {type(e).__name__}"
            log.warning("%s: %s", model_uri, e)
    # Детерминированный разбор возвращает и английские фразы из словаря направлений, и непереведённый
    # русский остаток. Раскладываем их по своим полям, чтобы кириллица не уходила в англоязычные источники.
    parsed = list(deterministic(query))
    english = [s for s in parsed if _is_latin(s)]
    remainder = [s for s in parsed if not _is_latin(s) and s.strip() != query.strip()]
    if remainder:
        why = f"{why}; непереведённый остаток запроса: {', '.join(remainder)}"
    return QueryPlan(domain=(parsed_intent.domain or query.strip())[:MAX_LEN], english_queries=english,
                     russian_queries=list(dict.fromkeys([query.strip(), *remainder])),
                     source="deterministic", model=None, fallback_reason=why,
                     intent=parsed_intent.to_dict())
