"""Телеметрия запуска: разбор запроса и семантическая проверка должны быть честно описаны (REPAIR F)."""
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import provenance, relevance  # noqa: E402


def cand(phrase):
    return {"phrase": phrase, "original_phrase": phrase, "canonical_label": phrase, "burst": 1.0,
            "channels": {"литература": 3, "репозитории": 2}, "recent_docs": 3, "prior_docs": 0,
            "examples": [], "repos": []}


def yandex(monkeypatch, behaviour):
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")

    class R:
        def __init__(self, text):
            self._t = text

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"alternatives": [{"message": {"text": self._t}}]}}

    state = {"n": 0}

    def post(url, **kw):
        idx = state["n"]
        state["n"] += 1
        out = behaviour(idx)
        if isinstance(out, Exception):
            raise out
        return R(out)

    monkeypatch.setattr(requests, "post", post)


def ok_rows(n):
    return "[" + ", ".join(
        f'{{"i": {i}, "canonical_name": "alpha beta{i}", "is_technology": true, '
        f'"domain_relevance": 0.9, "is_product_or_brand": false, "is_generic_phrase": false, '
        f'"merge_key": null, "reject_reason": null}}' for i in range(n)) + "]"


REQUIRED_FIELDS = ("configured", "attempted", "outcome", "failure_type",
                   "candidate_count_requested", "candidate_count_verified",
                   "candidate_count_unverified", "model")


def test_report_has_every_required_field_when_not_configured(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    report = {}
    relevance.canonicalize([cand("alpha beta")], report=report)
    for f in REQUIRED_FIELDS:
        assert f in report, f
    assert report["configured"] is False and report["attempted"] is False
    assert report["outcome"] == "not_configured"


def test_configured_but_failed_is_not_reported_as_working(monkeypatch):
    """Главный дефект наблюдаемости: configured=true при нуле проверенных кандидатов."""
    yandex(monkeypatch, lambda i: requests.Timeout("down"))
    report = {}
    relevance.canonicalize([cand(f"alpha beta{i}") for i in range(4)], report=report)
    assert report["configured"] is True
    assert report["attempted"] is True
    assert report["outcome"] == "failed"
    assert report["failure_type"] == "Timeout"
    assert report["candidate_count_verified"] == 0
    assert report["candidate_count_unverified"] == 4


def test_partial_outcome_is_distinguishable_from_success(monkeypatch):
    """Частичный успех обязан оставаться отличимым от полного.

    Планировщик теперь восстанавливается после одиночного таймаута (повтор меньшей порцией), поэтому
    сценарий «упал ровно первый вызов» даёт полный успех. Частичность проверяется на провайдере,
    который деградирует НАВСЕГДА после первых удачных порций."""
    yandex(monkeypatch, lambda i: ok_rows(8) if i == 0 else requests.Timeout("down"))
    report = {}
    relevance.canonicalize([cand(f"alpha beta{i}") for i in range(16)], report=report)
    assert report["outcome"] == "partial"
    assert 0 < report["candidate_count_verified"] < report["candidate_count_requested"] + 1


def test_counts_are_consistent(monkeypatch):
    yandex(monkeypatch, lambda i: ok_rows(8))
    cands = [cand(f"alpha beta{i}") for i in range(10)]
    report = {}
    relevance.canonicalize(cands, report=report)
    assert report["candidate_count_verified"] + report["candidate_count_unverified"] == len(cands)
    assert report["candidate_count_requested"] <= relevance.SEMANTIC_SHORTLIST


def test_model_identifier_is_reported_without_the_key(monkeypatch):
    yandex(monkeypatch, lambda i: ok_rows(8))
    report = {}
    relevance.canonicalize([cand("alpha beta")], report=report)
    assert report["model"] == "gpt://f/yandexgpt-5-lite"
    assert "k" != report["model"] and "Api-Key" not in str(report)


def test_run_metadata_still_carries_the_pinned_model_and_no_secret(monkeypatch):
    import json

    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "SECRET-VALUE")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folderZ")
    meta = provenance.run_metadata("q")
    assert meta["semantic_normalizer"]["pinned_model"] == "yandexgpt-5-lite"
    assert "SECRET-VALUE" not in json.dumps(meta, ensure_ascii=False)
