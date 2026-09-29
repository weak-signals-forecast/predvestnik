"""Адаптер языковых моделей из разрешённого ТЗ списка. Модель используется только для двух задач:
перевода поискового запроса и русского резюме по уже найденным источникам. Решение, является ли технология
слабым сигналом, модель не принимает: его принимает классификатор на признаках из открытых источников.

Выбор провайдера явный и логируется. Переменные окружения:
  LLM_PROVIDER=yandexgpt   YANDEX_API_KEY, YANDEX_FOLDER_ID, YANDEX_MODEL (только закреплённая yandexgpt-5-lite)
  LLM_PROVIDER=gigachat    GIGACHAT_TOKEN (access token), GIGACHAT_MODEL (GigaChat-2-Pro)
  LLM_PROVIDER=openai      OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL (gpt-4.1 | gpt-5.6-luna | Qwen3.6-35B-A3B через совместимый сервер)
Без ключей: запрос переводится локальной моделью Helsinki-NLP/opus-mt-ru-en, резюме собирается по шаблону из данных.
"""
from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache

import requests

from .decompose import ALLOWED_YANDEX_MODELS, COMPLETION_URL, configured_model, configured_model_name

log = logging.getLogger("wsignals.llm")
# Модели НЕ-яндексовых провайдеров. К YandexGPT этот список не применяется никогда: объединённый
# список пропускал бы чужую модель (`gpt-4.1`, `GigaChat-2-Pro`, `Qwen…`) в эндпоинт Яндекса.
# Гейт, имя модели и готовый URI для YandexGPT приходят только из `decompose.configured_model()`.
ALLOWED_OTHER_PROVIDERS = {"GigaChat-2", "GigaChat-2-Pro", "GigaChat-2-Max",
                           "gpt-4.1", "gpt-5.6-luna", "Qwen3.6-35B-A3B", "Qwen3-235B-A22B"}
PROMPT = ("Ты технологический аналитик банка. По источникам ниже напиши на русском JSON с полями description "
          "(что это за технология, 2 предложения), advantage (потенциальное преимущество, 1 предложение), case "
          "(конкретный кейс-пример из источников с названием компании или работы). Используй только факты из источников, "
          "ничего не добавляй от себя.\nТехнология: {tech}\nИсточники:\n{src}")


def provider() -> str | None:
    p = os.getenv("LLM_PROVIDER", "").lower()
    return p if p in {"yandexgpt", "gigachat", "openai"} else None


def model_name() -> str | None:
    """Имя модели для журнала и происхождения карточки.

    Для YandexGPT возвращается только имя, прошедшее единственный гейт (`decompose`): отвергнутая
    конфигурация не должна выглядеть в карточке как рабочая модель."""
    p = provider()
    if p == "yandexgpt":
        name = configured_model_name()
        return name if name in ALLOWED_YANDEX_MODELS else None
    return {"gigachat": os.getenv("GIGACHAT_MODEL", "GigaChat-2-Pro"),
            "openai": os.getenv("OPENAI_MODEL", "gpt-4.1")}.get(p or "")


# REPAIR C. Обращения к модели идут мимо wsignals.http, поэтому общий дедлайн запроса на них
# не распространялся: финализация продолжала ходить в сеть уже за сетевым дедлайном и выводила
# ответ за объявленную стену. Теперь каждый вызов спрашивает остаток бюджета и не начинается,
# если не помещается. Выбор модели, закреплённый идентификатор и сборка URI не затронуты.
LLM_DEFAULT_TIMEOUT = float(os.getenv("WSIGNALS_LLM_CHAT_TIMEOUT", "60"))


def _budgeted_timeout() -> float | None:
    from . import http as http_mod

    return http_mod.call_budget(LLM_DEFAULT_TIMEOUT)


def _chat_yandexgpt(prompt: str) -> str | None:
    """Единственный путь к облаку Яндекса.

    Гейт, имя модели и готовый `modelUri` берутся из `decompose.configured_model()`. Здесь нет ни
    собственного списка моделей, ни сборки URI: отказ конфигурации означает отсутствие HTTP-вызова."""
    model_uri, why = configured_model()
    if not model_uri:
        log.warning("модель YandexGPT не сконфигурирована (%s), генерация отключена", why)
        return None
    timeout = _budgeted_timeout()
    if timeout is None:
        log.info("бюджет запроса исчерпан: обращение к модели не начинается")
        return None
    log.info("LLM: провайдер yandexgpt, модель %s", model_uri)
    try:
        r = requests.post(COMPLETION_URL, timeout=timeout,
                          headers={"Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}"},
                          json={"modelUri": model_uri,
                                "completionOptions": {"temperature": 0.2, "maxTokens": 600},
                                "messages": [{"role": "user", "text": prompt}]})
        return r.json()["result"]["alternatives"][0]["message"]["text"]
    except Exception as e:  # noqa: BLE001
        log.warning("LLM недоступна: %s", e)
        return None


def _chat(prompt: str) -> str | None:
    p = provider()
    if not p:
        return None
    if p == "yandexgpt":
        return _chat_yandexgpt(prompt)
    m = model_name()
    if m not in ALLOWED_OTHER_PROVIDERS:
        log.warning("модель %s не входит в разрешённый список ТЗ, генерация отключена", m)
        return None
    timeout = _budgeted_timeout()
    if timeout is None:
        log.info("бюджет запроса исчерпан: обращение к модели не начинается")
        return None
    log.info("LLM: провайдер %s, модель %s", p, m)
    try:
        if p == "gigachat":
            r = requests.post("https://gigachat.devices.sberbank.ru/api/v1/chat/completions", timeout=timeout,
                              headers={"Authorization": f"Bearer {os.environ['GIGACHAT_TOKEN']}"},
                              json={"model": m, "temperature": 0.2, "messages": [{"role": "user", "content": prompt}]})
            return r.json()["choices"][0]["message"]["content"]
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        r = requests.post(f"{base}/chat/completions", timeout=timeout, headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                          json={"model": m, "temperature": 0.2, "messages": [{"role": "user", "content": prompt}]})
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        log.warning("LLM недоступна: %s", e)
        return None


def summarize(tech: str, sources: list[dict]) -> dict | None:
    src = "\n".join(f"- {s['title']} ({s['source_name']}, {s.get('published')}): {s.get('snippet', '')[:400]}" for s in sources)
    out = _chat(PROMPT.format(tech=tech, src=src))
    if not out:
        return None
    m = re.search(r"\{.*\}", out, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


@lru_cache(maxsize=1)
def _marian():
    from transformers import MarianMTModel, MarianTokenizer
    name = "Helsinki-NLP/opus-mt-ru-en"
    return MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name).eval()


def translate_term(term: str) -> str | None:
    """Русское название технического термина. Только через разрешённую модель: универсальный машинный перевод
    на терминах ошибается («rag poisoning» -> «отравление тряпкой»), поэтому локальная модель здесь не используется."""
    out = _chat("Переведи название технологии на русский коротко, как в отраслевом отчёте. Сохрани принятые "
                "аббревиатуры латиницей (MCP, RAG, KYC, LLM). Верни только перевод, без пояснений.\n" + term)
    return out.strip().strip('"').split("\n")[0][:90] if out else None


def translate_query(text: str) -> str:
    out = _chat(f"Переведи поисковый запрос на английский, верни только перевод: {text}")
    if out:
        return out.strip().strip('"')
    try:
        import torch
        tok, m = _marian()
        with torch.no_grad():
            ids = m.generate(**tok([text], return_tensors="pt"), max_new_tokens=40)
        return tok.batch_decode(ids, skip_special_tokens=True)[0]
    except Exception as e:  # noqa: BLE001
        log.warning("локальный перевод недоступен: %s", e)
        return text
