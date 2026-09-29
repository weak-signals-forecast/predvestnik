"""Герметичная проверка API и хранилища: без сети и без обращений к внешним источникам.

    python scripts/smoke_api.py

Подменяется только конвейер запроса (`api.main.run_query`) — всё остальное настоящее: FastAPI-обработчики,
фоновый поток, SQLAlchemy, миграция добавленных колонок, сериализация карточки обратно клиенту.
С `DATABASE_URL` проверяется PostgreSQL, без неё — SQLite. Используется в Docker-смоуке.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE = {
    "title": "Agent identity attestation for autonomous payments", "url": "https://doi.org/10.1/abc",
    "source_name": "Nature", "source_type": "научная публикация", "published": "2026-03-01",
    "published_at": "2026-03-01", "language": "en", "trust": "высокий", "trust_reason": "научная публикация",
    "canonical_url": "https://doi.org/10.1/abc", "retrieved_at": "2026-09-20T10:00:00+00:00",
    "source_family": "science", "is_primary": True, "content_hash": "a" * 32, "adapter_status": "OK",
}


def card(name: str, decision: str, tier: str) -> dict:
    return {
        "technology": name, "technology_original": name, "original_phrase": name, "canonical_label": name,
        "merge_key": name, "merged_from": [], "tier": tier, "status": "слабый сигнал",
        "score": 0.21, "rank_score": 0.21, "emergence_score": 0.66, "maturity_risk": 0.05,
        "evidence_confidence": 0.95, "hype_risk": 0.02, "legacy_model_probability": 0.61,
        "legacy_model_reliable": False, "legacy_model_interval_90": None,
        "decision": decision, "decision_reasons": ["подтверждено конкретными документами: 2 (code, science)"],
        "decision_reason_codes": ["accepted" if decision == "ACCEPT" else "low_evidence_confidence"],
        "supporting_document_count": 2, "supporting_document_families": ["code", "science"],
        "verification": "SEMANTICALLY_VERIFIED", "high_certainty": False,
        "measured_aggregate_families": ["code", "media", "science"],
        "concrete_supporting_families": ["code", "science"],
        "evidence_quality": {"independent_supports": 2, "distinct_origins": 2, "claims": []},
        "diversity_rank": 1, "sibling_key": "identity",
        "description": "описание", "advantage": "преимущество", "case": None,
        "why_weak_signal": ["↑ доля препринтов: 40 %"], "takeoff": None, "historical_twin": None,
        "description_generated": False, "generator": None, "advantage_is_original_en": True,
        "evidence": {"burst": 2.0, "recent_docs": 3, "prior_docs": 0, "adapters": {}, "failed_adapters": [],
                     "no_result_adapters": [], "positive_families": ["science"],
                     "corroborating_families": ["science", "code"], "completeness": 1.0,
                     "retrieved_at": "2026-09-20T10:00:00+00:00", "from_cache": False},
        "features": {}, "sources": [SOURCE], "sources_note": None,
        "retrieved_at": "2026-09-20T10:00:00+00:00", "from_cache": False, "assessment": None,
        "canonical_source": None, "name_source": "тест",
    }


RUN_META = {"run_id": "0" * 32, "commit_sha": "deadbeef", "config_hash": "c" * 32, "query": "smoke",
            "as_of": "2026-09-20", "cache_state": {"available": True, "entries": 0},
            "provider_status": {"отказы адаптеров": {}},
            "semantic_normalizer": {"configured": False, "model": None, "reason": "не задан"}}

RESULT = {
    "query": "smoke", "seeds": ["agent identity"], "date": "2026-09-20", "model": "logreg",
    "run": RUN_META,
    "query_plan": {"source": "deterministic", "model": None},
    "timings": {"total_wall_s": 0.0},
    "stats": {"accepted_count": 1, "watchlist_count": 1, "rejected_count": 1, "секунд": 0.1},
    "signals": [card("agent identity", "ACCEPT", "подтверждённый слабый сигнал")],
    "watchlist": [card("machine credentials", "LOW_EVIDENCE", "требует дополнительных доказательств")],
    "excluded": [{"technology": "claude skills", "score": 0.0, "decision": "NOISE",
                  "code": "semantic_rejection", "reason": "название продукта"}],
}


def main() -> int:
    import api.main as app

    health = app.health()
    assert health["status"] in ("ok", "degraded"), health
    assert health["database"] == "ok", f"база недоступна: {health}"
    assert isinstance(health["query_budget_s"], (int, float)), health
    print(f"health: {health}")

    app.run_query = lambda *a, **kw: RESULT                       # подменяем только конвейер
    started = app.search(app.SearchIn(query="дымовая проверка API"))
    run_id = started["id"]
    for _ in range(100):
        run = app.get_search(run_id)
        if run["status"] != "running":
            break
        time.sleep(0.1)
    assert run["status"] == "done", run.get("error")

    assert len(run["signals"]) == 1, run["signals"]
    assert len(run["watchlist"]) == 1, run["watchlist"]
    # Записи без записанного решения ACCEPT подтверждёнными не показываются никогда.
    assert run.get("unverified_legacy") == [], run.get("unverified_legacy")
    signal = run["signals"][0]
    for key in ("emergence_score", "maturity_risk", "evidence_confidence", "hype_risk",
                "legacy_model_probability", "legacy_model_reliable", "rank_score",
                "decision", "decision_reason_codes", "supporting_document_count"):
        assert key in signal, f"в карточке нет поля {key}"
    assert signal["tier"] == "подтверждённый слабый сигнал"
    assert run["watchlist"][0]["tier"] == "требует дополнительных доказательств"

    src = signal["sources"][0]
    for key in ("canonical_url", "retrieved_at", "source_family", "is_primary", "content_hash",
                "adapter_status", "published_at"):
        assert key in src, f"в источнике нет поля происхождения {key}"
    assert src["canonical_url"] == SOURCE["canonical_url"] and src["is_primary"] is True

    assert run["excluded"][0]["code"] == "semantic_rejection", run["excluded"]
    meta = run.get("run") or {}
    for key in ("run_id", "commit_sha", "config_hash", "cache_state", "provider_status", "semantic_normalizer"):
        assert key in meta, f"метаданные запуска не дошли до клиента: нет {key}"
    assert meta["run_id"] == RUN_META["run_id"]
    assert app.list_searches(5), "список запусков пуст"
    print(f"search {run_id}: signals={len(run['signals'])} watchlist={len(run['watchlist'])} "
          f"excluded={len(run['excluded'])}")
    print("SMOKE_API: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
