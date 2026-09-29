"""Разбор произвольного запроса: модель YandexGPT, строгая схема, детерминированный запасной путь (PHASE 4, PHASE 8)."""
import sys
from pathlib import Path

import re

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import decompose  # noqa: E402
from wsignals.query import parse_query  # noqa: E402

VALID = {"domain": "кибербезопасность ИИ",
         "english_queries": ["ai agent security", "llm supply chain security"],
         "russian_queries": ["безопасность ИИ-агентов"],
         "synonyms": ["agent sandboxing"],
         "exclusions": ["marketing", "vacancy"]}


@pytest.fixture
def yandex_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "test-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folder1")
    monkeypatch.setenv("YANDEX_QUERY_MODEL", "yandexgpt-5-lite")


def reply(monkeypatch, text, calls=None):
    """Подменяет вызов облака. `calls` собирает фактические аргументы POST для проверки контракта."""
    class R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"alternatives": [{"message": {"text": text}}]}}

    def post(*a, **kw):
        if calls is not None:
            calls.append({"args": a, "kwargs": kw})
        return R()

    monkeypatch.setattr(requests, "post", post)


# ---------- явная конфигурация модели ----------

def test_model_is_explicit_and_logged(yandex_env, monkeypatch, caplog):
    import json as _json
    reply(monkeypatch, _json.dumps(VALID))
    with caplog.at_level("INFO", logger="wsignals.decompose"):
        plan = decompose.decompose("безопасность ИИ-агентов", parse_query)
    assert plan.source == "llm"
    assert plan.model == "gpt://folder1/yandexgpt-5-lite"        # полный идентификатор модели
    assert "gpt://folder1/yandexgpt-5-lite" in caplog.text
    assert "test-key" not in caplog.text                          # ключ в журнал не попадает


def test_no_automatic_model_selection(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")
    monkeypatch.setenv("YANDEX_QUERY_MODEL", "some-other-model/latest")
    uri, why = decompose.configured_model()
    assert uri is None and "не входит в разрешённый" in why


def test_temperature_is_low(yandex_env):
    assert decompose.TEMPERATURE <= 0.2


# ---------- рантайм закреплён за проверенной моделью Lite 5 ----------

def test_runtime_model_is_pinned_to_the_live_verified_identifier():
    """Ровно одна разрешённая модель — та, что проверена живым вызовом REST."""
    assert decompose.RUNTIME_YANDEX_MODEL == "yandexgpt-5-lite"
    assert decompose.ALLOWED_YANDEX_MODELS == {"yandexgpt-5-lite"}
    assert decompose.DEFAULT_MODEL == "yandexgpt-5-lite"


def test_explicit_lite5_is_accepted_and_uri_has_no_version_suffix(yandex_env):
    uri, why = decompose.configured_model()
    assert why is None
    assert uri == "gpt://folder1/yandexgpt-5-lite"      # идентификатор ровно такой, как в живой проверке
    assert "/latest" not in uri and "/rc" not in uri    # версия не дописывается неявно


def test_default_model_is_lite5_without_any_model_variable(monkeypatch):
    """Без YANDEX_QUERY_MODEL и YANDEX_MODEL рантайм всё равно пинится на проверенную модель, а не на /latest."""
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")
    monkeypatch.delenv("YANDEX_QUERY_MODEL", raising=False)
    monkeypatch.delenv("YANDEX_MODEL", raising=False)
    uri, why = decompose.configured_model()
    assert (uri, why) == ("gpt://f/yandexgpt-5-lite", None)


@pytest.mark.parametrize("model", ["yandexgpt/latest", "yandexgpt-lite/latest", "yandexgpt/rc",
                                   "yandexgpt-lite/rc", "yandexgpt-5-lite/latest", "gpt-4.1"])
def test_models_outside_the_pinned_contract_are_rejected(monkeypatch, model):
    """Устаревшие и посторонние идентификаторы отвергаются, а не подменяются молча."""
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")
    monkeypatch.setenv("YANDEX_QUERY_MODEL", model)
    uri, why = decompose.configured_model()
    assert uri is None and "не входит в разрешённый" in why


def test_both_yandex_paths_use_one_and_the_same_model(monkeypatch):
    """Разбор запроса и генерация карточки не должны звать разные модели одного облака."""
    from wsignals import llm
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")
    monkeypatch.delenv("YANDEX_QUERY_MODEL", raising=False)
    monkeypatch.delenv("YANDEX_MODEL", raising=False)
    assert llm.model_name() == decompose.configured_model_name() == decompose.RUNTIME_YANDEX_MODEL
    # Кросс-провайдерный список к Яндексу не применяется: гейт только `ALLOWED_YANDEX_MODELS`.
    assert decompose.RUNTIME_YANDEX_MODEL not in llm.ALLOWED_OTHER_PROVIDERS


# ---------- контракт вызова облака ----------

def test_endpoint_and_auth_header_are_unchanged(yandex_env, monkeypatch):
    """Эндпоинт и схема авторизации те же, что проверены живым вызовом; ключ уходит только в заголовок."""
    import json as _json
    calls = []
    reply(monkeypatch, _json.dumps(VALID), calls)
    plan = decompose.decompose("безопасность ИИ-агентов", parse_query)
    assert plan.source == "llm"
    assert decompose.COMPLETION_URL == "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    sent = calls[0]
    assert sent["args"][0] == decompose.COMPLETION_URL
    assert sent["kwargs"]["headers"] == {"Authorization": "Api-Key test-key"}
    assert sent["kwargs"]["json"]["modelUri"] == "gpt://folder1/yandexgpt-5-lite"
    assert _json.dumps(sent["kwargs"]["json"], ensure_ascii=False).find("test-key") == -1   # ключа нет в теле


# ---------- схема ----------

def test_schema_is_validated_strictly():
    assert decompose.validate(VALID) is not None
    assert decompose.validate({"domain": "x"}) is None                      # нет english_queries
    assert decompose.validate({"english_queries": ["a b"]}) is None         # нет domain
    assert decompose.validate({"domain": "", "english_queries": ["a"]}) is None
    assert decompose.validate("не объект") is None


def test_schema_drops_garbage_items_but_keeps_valid_ones():
    plan = decompose.validate({"domain": "финтех", "english_queries": ["agentic payments", 42, "", "x" * 500,
                                                                      "tokenized deposits"]})
    assert plan.english_queries == ["agentic payments", "tokenized deposits"]


# ---------- отказ модели никогда не валит запрос ----------

def test_invalid_json_falls_back_to_deterministic_parsing(yandex_env, monkeypatch):
    reply(monkeypatch, "конечно! вот ваш ответ: {не json,,,}")
    plan = decompose.decompose("слабые сигналы в кибербезопасности", parse_query)
    assert plan.source == "deterministic"
    assert plan.fallback_reason == "ответ модели не прошёл проверку схемы"
    assert "cybersecurity" in plan.english_queries


def test_schema_violating_json_falls_back(yandex_env, monkeypatch):
    reply(monkeypatch, '{"domain": "финтех"}')
    plan = decompose.decompose("перспективные решения в финтехе", parse_query)
    assert plan.source == "deterministic" and "fintech" in plan.english_queries


def test_llm_unavailable_falls_back_without_raising(yandex_env, monkeypatch):
    def boom(*a, **kw):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(requests, "post", boom)
    plan = decompose.decompose("слабые сигналы в кибербезопасности", parse_query)
    assert plan.source == "deterministic" and plan.fallback_reason.startswith("модель недоступна")
    assert plan.seeds()


def test_missing_credentials_fall_back(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    plan = decompose.decompose("слабые сигналы в кибербезопасности", parse_query)
    assert plan.source == "deterministic" and plan.model is None


# ---------- произвольный русский запрос ----------

@pytest.mark.parametrize("query", [
    "слабые сигналы в кибербезопасности",
    "перспективные решения в финтехе",
    "зарождающиеся тренды в квантовых вычислениях",
    "новые технологии переработки промышленных отходов",
    "что происходит с малыми модульными реакторами",
])
def test_arbitrary_russian_query_always_yields_a_plan(monkeypatch, query):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr("wsignals.llm.translate_query", lambda t: f"[{t}]")
    plan = decompose.decompose(query, parse_query)
    assert plan.domain and query in plan.russian_queries
    assert plan.seeds(), query          # без поисковых фраз запрос был бы бессмысленным


@pytest.mark.parametrize("query,expect_latin", [
    ("слабые сигналы в кибербезопасности", True),
    ("новые способы охлаждения дата-центров", True),
    ("что происходит с малыми модульными реакторами", False),
])
def test_cyrillic_remainder_never_reaches_english_sources(monkeypatch, query, expect_latin):
    """OpenAlex и GitHub англоязычные: кириллический остаток разбора туда уходить не должен."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr("wsignals.llm.translate_query", lambda t: t)      # переводчика нет
    plan = decompose.decompose(query, parse_query)
    assert plan.untranslated is not expect_latin
    if expect_latin:
        assert all(not re.search(r"[а-яё]", s.lower()) for s in plan.seeds()), plan.seeds()
    else:
        # Перевести нечем: запрос всё равно выполняется, но это состояние видно в плане и в журнале.
        assert plan.seeds() and plan.fallback_reason


def test_llm_plan_seeds_are_english_only(yandex_env, monkeypatch):
    import json as _json
    reply(monkeypatch, _json.dumps({**VALID, "synonyms": ["агентные платежи", "agent sandboxing"]}))
    plan = decompose.decompose("безопасность ИИ-агентов", parse_query)
    assert "агентные платежи" not in plan.seeds()
    assert "agent sandboxing" in plan.seeds()
