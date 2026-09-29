"""Гейт модели на реальной поверхности `wsignals.llm` (REPAIR V2).

Регрессия на P0 `YANDEX_ALLOWLIST_BYPASS_VIA_UNION_GATE`: объединённый список моделей всех провайдеров
пропускал чужую модель (`gpt-4.1`, `GigaChat-2-Pro`, `Qwen…`) в эндпоинт Яндекса, потому что URI там
собирался отдельно от гейта. При `LLM_PROVIDER=yandexgpt` в облако Яндекса может уходить единственный
идентификатор — `yandexgpt-5-lite`. Настоящих сетевых вызовов в тестах нет.
"""
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import decompose, llm  # noqa: E402

PINNED_URI = "gpt://folderX/yandexgpt-5-lite"

# Члены кросс-провайдерного списка: ни один не имеет права уйти в облако Яндекса.
OTHER_PROVIDER_MODELS = ["gpt-4.1", "gpt-5.6-luna", "GigaChat-2", "GigaChat-2-Pro", "GigaChat-2-Max",
                         "Qwen3.6-35B-A3B", "Qwen3-235B-A22B"]
LEGACY_YANDEX_ALIASES = ["yandexgpt/latest", "yandexgpt-lite/latest", "yandexgpt/rc", "yandexgpt-lite/rc"]
GARBAGE = ["yandexgpt-5-lite/latest", "gpt://folderX/yandexgpt-5-lite", "  ", "полная-ерунда", "../../etc/passwd"]


@pytest.fixture
def calls(monkeypatch):
    """Перехватывает POST целиком: и факт вызова, и его аргументы. Сеть недоступна по построению."""
    seen = []

    class R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"alternatives": [{"message": {"text": '{"description": "d", "advantage": "a", "case": "c"}'}}]}}

    def post(*a, **kw):
        seen.append({"url": a[0] if a else kw.get("url"), "kwargs": kw})
        return R()

    monkeypatch.setattr(requests, "post", post)
    # Локальный переводчик не должен подниматься: тест обязан быть офлайновым в любом окружении.
    monkeypatch.setattr(llm, "_marian", lambda: (_ for _ in ()).throw(RuntimeError("локальная модель отключена в тесте")))
    return seen


@pytest.fixture
def yandex_creds(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "stub-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folderX")
    monkeypatch.delenv("YANDEX_QUERY_MODEL", raising=False)
    monkeypatch.delenv("YANDEX_MODEL", raising=False)


# ---------- закреплённая модель вызывается ----------

def test_default_config_calls_exactly_the_pinned_model(yandex_creds, calls):
    assert llm.translate_query("новые способы охлаждения дата-центров")
    assert len(calls) == 1
    assert calls[0]["kwargs"]["json"]["modelUri"] == PINNED_URI


@pytest.mark.parametrize("var", ["YANDEX_QUERY_MODEL", "YANDEX_MODEL"])
def test_explicit_pinned_model_from_either_variable_calls_the_cloud(yandex_creds, calls, monkeypatch, var):
    monkeypatch.setenv(var, "yandexgpt-5-lite")
    assert llm.translate_query("тест")
    assert [c["kwargs"]["json"]["modelUri"] for c in calls] == [PINNED_URI]


# ---------- обхода гейта нет ни через одну переменную ----------

@pytest.mark.parametrize("var", ["YANDEX_QUERY_MODEL", "YANDEX_MODEL"])
@pytest.mark.parametrize("model", OTHER_PROVIDER_MODELS + LEGACY_YANDEX_ALIASES + GARBAGE)
def test_no_identifier_but_the_pinned_one_reaches_the_yandex_endpoint(yandex_creds, calls, monkeypatch, var, model):
    monkeypatch.setenv(var, model)
    llm.translate_query("тест")
    assert calls == [], f"{var}={model!r} ушло в облако Яндекса"


@pytest.mark.parametrize("var", ["YANDEX_QUERY_MODEL", "YANDEX_MODEL"])
def test_empty_variable_is_treated_as_unset_not_as_an_identifier(yandex_creds, calls, monkeypatch, var):
    """Пустое значение переменной — это «не задано», а не идентификатор: действует закреплённая модель.

    Обхода здесь нет: в облако всё равно уходит только `yandexgpt-5-lite`. Порядок
    `YANDEX_QUERY_MODEL > YANDEX_MODEL > DEFAULT_MODEL` сохранён намеренно."""
    monkeypatch.setenv(var, "")
    llm.translate_query("тест")
    assert [c["kwargs"]["json"]["modelUri"] for c in calls] == [PINNED_URI]


@pytest.mark.parametrize("model", OTHER_PROVIDER_MODELS)
def test_cross_provider_model_is_not_reported_as_a_working_model(yandex_creds, monkeypatch, model):
    """Отвергнутая конфигурация не должна попасть в происхождение карточки как рабочая модель."""
    monkeypatch.setenv("YANDEX_MODEL", model)
    assert llm.model_name() is None
    assert model not in decompose.ALLOWED_YANDEX_MODELS


# ---------- публичные точки входа гейт не обходят ----------

@pytest.mark.parametrize("entrypoint", [
    lambda: llm.translate_query("тест"),
    lambda: llm.translate_term("rag poisoning"),
    lambda: llm.summarize("agent identity", [{"title": "t", "source_name": "s", "published": "2026-01-01",
                                              "snippet": "x"}]),
])
@pytest.mark.parametrize("var", ["YANDEX_QUERY_MODEL", "YANDEX_MODEL"])
def test_public_entrypoints_cannot_bypass_the_gate(yandex_creds, calls, monkeypatch, entrypoint, var):
    monkeypatch.setenv(var, "gpt-4.1")
    entrypoint()
    assert calls == []


@pytest.mark.parametrize("entrypoint,expected_calls", [
    (lambda: llm.translate_query("тест"), 1),
    (lambda: llm.translate_term("rag poisoning"), 1),
    (lambda: llm.summarize("agent identity", [{"title": "t", "source_name": "s", "published": "2026-01-01",
                                               "snippet": "x"}]), 1),
])
def test_public_entrypoints_use_the_pinned_uri_when_configured(yandex_creds, calls, entrypoint, expected_calls):
    entrypoint()
    assert len(calls) == expected_calls
    assert {c["kwargs"]["json"]["modelUri"] for c in calls} == {PINNED_URI}


# ---------- эндпоинт, авторизация и отсутствие второго сборщика URI ----------

def test_endpoint_and_auth_are_unchanged(yandex_creds, calls):
    llm.translate_query("тест")
    sent = calls[0]
    assert sent["url"] == "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    assert sent["url"] == decompose.COMPLETION_URL
    assert sent["kwargs"]["headers"] == {"Authorization": "Api-Key stub-key"}
    assert "stub-key" not in str(sent["kwargs"]["json"])


def test_yandex_is_not_gated_by_the_cross_provider_list(yandex_creds):
    """Единственный список для Яндекса — `decompose.ALLOWED_YANDEX_MODELS`."""
    assert not (llm.ALLOWED_OTHER_PROVIDERS & decompose.ALLOWED_YANDEX_MODELS)
    assert all("yandexgpt" not in m for m in llm.ALLOWED_OTHER_PROVIDERS)
    assert llm.model_name() == decompose.configured_model_name() == decompose.RUNTIME_YANDEX_MODEL


def test_llm_module_builds_no_yandex_uri_of_its_own():
    """URI Яндекса собирается ровно в одном месте — `decompose.configured_model()`."""
    src = (ROOT / "wsignals" / "llm.py").read_text(encoding="utf-8")
    assert "gpt://" not in src


def test_missing_credentials_make_no_call(calls, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.delenv("YANDEX_API_KEY", raising=False)
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)
    monkeypatch.setenv("YANDEX_QUERY_MODEL", "yandexgpt-5-lite")
    llm.translate_query("тест")
    assert calls == [] and llm.summarize("x", []) is None


# ---------- поведение не-яндексовых провайдеров не изменилось ----------

def test_other_providers_keep_their_own_allowlist(calls, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gigachat")
    monkeypatch.setenv("GIGACHAT_TOKEN", "stub")
    monkeypatch.setenv("GIGACHAT_MODEL", "GigaChat-2-Pro")
    assert llm.model_name() == "GigaChat-2-Pro"
    llm._chat("тест")
    assert calls[0]["url"] == "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    calls.clear()
    monkeypatch.setenv("GIGACHAT_MODEL", "неразрешённая-модель")
    llm._chat("тест")
    assert calls == []
