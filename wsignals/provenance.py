"""Происхождение запуска: метаданные, по которым запуск можно воспроизвести и сравнить (R2-H).

Два запуска сервиса раньше нельзя было честно сравнить: в выдаче не было ни версии кода, ни состояния
кэша, ни того, работала ли семантическая нормализация. «Стало лучше» проверить было нечем.

Здесь собирается ровно то, что нужно для сравнения, и ничего секретного:

    run_id             уникальный идентификатор запуска
    commit_sha         версия кода
    config_hash        отпечаток поведенчески значимых настроек (без секретов)
    query              сам запрос
    as_of              дата, на которую собраны данные
    cache_state        сводка по дисковому кэшу
    provider_status    состояние адаптеров
    semantic_normalizer статус нормализатора и идентификатор модели, если он задан

Значения ключей и токенов здесь не появляются никогда: в отпечаток входит только ФАКТ наличия ключа,
а не сам ключ. Идентификатор модели читается из уже существующей конфигурации и не меняется.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Переменные окружения, наличие которых влияет на поведение. В отпечаток идёт только факт наличия.
SECRET_PRESENCE_KEYS = ("YANDEX_API_KEY", "YANDEX_FOLDER_ID", "GIGACHAT_TOKEN", "OPENAI_API_KEY",
                        "GITHUB_TOKEN", "OPENALEX_API_KEY")


def _pinned_model() -> str:
    """Закреплённый идентификатор модели из контракта. Никаких секретов и никакого автовыбора."""
    from . import decompose

    return decompose.RUNTIME_YANDEX_MODEL


def commit_sha() -> str:
    """Версия кода. Сначала переменная окружения (её выставляет образ), затем git, затем «unknown»."""
    env = os.getenv("WSIGNALS_COMMIT_SHA")
    if env:
        return env.strip()[:40]
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()[:40]
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def behavioral_config() -> dict:
    """Поведенчески значимые настройки. Секретов здесь нет: только факт наличия ключа."""
    from . import claim_relevance, decision, query, relevance
    from . import evidence_quality as eq
    from . import http as http_mod

    plan = query.budget_plan()
    cfg = {
        "decision": {
            "min_corroborating_families": decision.MIN_CORROBORATING_FAMILIES,
            "min_evidence_confidence": decision.MIN_EVIDENCE_CONFIDENCE,
            "maturity_veto": decision.MATURITY_VETO,
            "hype_veto": decision.HYPE_VETO,
            "min_emergence": decision.MIN_EMERGENCE,
            "unknown_maturity_risk": decision.UNKNOWN_MATURITY_RISK,
            "maturity_weights": dict(sorted(decision.MATURITY_WEIGHTS.items())),
        },
        "evidence": {
            "min_independent_supports": eq.MIN_INDEPENDENT_SUPPORTS,
            "min_distinct_origins": eq.MIN_DISTINCT_ORIGINS,
            "primary_claim": eq.PRIMARY_CLAIM,
            # Претензионно-относительная релевантность: третье условие подтверждения наряду с
            # разрядом доверия и применимостью семейства. Решает, какие документы вообще
            # засчитываются, поэтому поведенчески значима.
            "claim_relevance": {
                "matched_field": "title",
                # Контракт V3 — фразовый. Множества токенов источником истины больше не являются:
                # заголовок обязан содержать авторитетную фразу подряд и не внутри дефисного
                # сращения. Лесенка личности удалена как авторитет подтверждения.
                "match_kind": "phrase_in_title",
                # Безусловно авторитетно только ИСХОДНОЕ название. Канон модели — свободное
                # переименование: для показа и кластеризации годится, для истины доказательства —
                # только при детерминированной равнозначности исходному названию.
                "identity_variants": ["original_phrase", "phrase"],
                "canonical_label_authority": "deterministic_equivalence_only",
                "llm_rename_is_claim_evidence_authority": False,
                "normalization": ["casefold", "unicode", "punctuation", "singular_plural"],
                "hyphen_compound_boundary": "enforced",
                "merged_alias_authority": "deterministic_equivalence_only",
                "alias_equivalence_rule": "same_token_multiset",
                "acronym_alias_without_provenance": "rejected",
                "merged_attachment_must_be_authoritative": True,
                "unknown_counts_as_support": False,
                "tokenizer": "unicode_word",
                "cross_language_state": claim_relevance.CROSS_LANGUAGE_UNVERIFIED,
            },
        },
        "relevance": {
            "min_domain_relevance": relevance.MIN_DOMAIN_RELEVANCE,
            "min_specificity": relevance.MIN_SPECIFICITY,
            "llm_merge_similarity": relevance.LLM_MERGE_SIMILARITY,
        },
        # Ограничители семантической проверки: от них напрямую зависит, сколько кандидатов вообще
        # может получить подтверждение, поэтому они входят в отпечаток конфигурации (REPAIR B).
        "semantic_verification": {
            "batch": relevance.LLM_BATCH,
            "shortlist": relevance.SEMANTIC_SHORTLIST,
            "phase_share": relevance.SEMANTIC_PHASE_SHARE,
            "min_call_budget_s": relevance.SEMANTIC_MIN_CALL_BUDGET,
            # P1-C. Растущие порции: от этих чисел напрямую зависит, сколько кандидатов успевает
            # получить подтверждение, поэтому они входят в отпечаток конфигурации.
            "min_batch": relevance.SEMANTIC_MIN_BATCH,
            "max_attempts": relevance.SEMANTIC_MAX_ATTEMPTS,
            "phase_floor_s": relevance.SEMANTIC_PHASE_FLOOR,
            "phase_max_share": relevance.SEMANTIC_PHASE_MAX_SHARE,
            "recovery_reserve_s": relevance.SEMANTIC_RECOVERY_RESERVE,
            # R1. Разделение семантической доли поведенчески значимо не меньше остальных чисел:
            # от него зависит, сколько кандидатов успевает проверить ранняя фаза и сколько времени
            # остаётся адресной поздней. Ставить это только в `semantic_normalizer_status()` нельзя:
            # тот словарь в отпечаток конфигурации не входит, и смена ручки прошла бы незаметно.
            "early_budget_s": relevance.EARLY_SEMANTIC_BUDGET,
            "late_reserve_s": relevance.LATE_SEMANTIC_RESERVE,
            "late_max_targets": relevance.LATE_MAX_TARGETS,
            "pinned_model": _pinned_model(),
        },
        # P1-B. Порог достаточности доказательства хайпа: решает, когда отказ источника блокирует
        # принятие, поэтому тоже поведенчески значим.
        # Политика достаточности доказательства хайпа: решает, когда отказ источника блокирует
        # принятие. Замена счётчика документов на перечень измеряющих семейств поведенчески значима.
        "hype_evidence": {"primary_source": decision.HYPE_PRIMARY_SOURCE,
                          "signal_families": sorted(decision.HYPE_SIGNAL_FAMILIES),
                          "substitute": "document_hype_signal"},
        "budget": plan,
        "retrieval": {
            "live_retries": http_mod.LIVE.retries,
            "live_timeout": http_mod.LIVE.timeout,
            "min_request_budget": http_mod.MIN_REQUEST_BUDGET_S,
            "keep_wiki": query.LIVE_KEEP_WIKI,
            "keep_full": query.LIVE_KEEP_FULL,
            # Фактическая ширина третьего этапа: не меньше запрошенного ТОПа (см. query.stage3_width).
            "keep_full_rule": "max(keep_full, top)",
            "github_stats_on_live_path": query.LIVE_GITHUB_STATS,
            "enrich_top": query.ENRICH_TOP,
            "watchlist_emergence": query.WATCHLIST_EMERGENCE,
            "budget_share": dict(sorted(query.BUDGET_SHARE.items())),
        },
        # Закреплённый контрактом идентификатор модели — поведенчески значимая несекретная настройка.
        "credentials_present": {k: bool(os.getenv(k)) for k in SECRET_PRESENCE_KEYS},
    }
    return cfg


def config_hash(config: dict | None = None) -> str:
    """Устойчивый отпечаток настроек: одинаковые настройки — одинаковый хеш."""
    cfg = behavioral_config() if config is None else config
    raw = json.dumps(cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def semantic_normalizer_status() -> dict:  # noqa: D401
    """Статус нормализатора (R2-C).

    Разделены две разные вещи:

      `model`        — модель, которая РЕАЛЬНО использовалась. Без учётных данных это `null`,
                       и так и должно быть: писать сюда закреплённое имя означало бы утверждать,
                       что модель работала, когда она не работала;
      `pinned_model` — закреплённый контрактом идентификатор. Он известен всегда, не зависит от
                       учётных данных и секретом не является.

    Значения ключей не читаются и не логируются: `configured_model()` возвращает только URI без
    секретов, а закреплённое имя — константа контракта."""
    from . import decompose

    from . import relevance

    model_uri, why = decompose.configured_model()
    return {"configured": bool(model_uri), "model": model_uri,
            "pinned_model": decompose.RUNTIME_YANDEX_MODEL,
            "reason": None if model_uri else why,
            # Разделение семантической доли объявляется в отпечатке: аудит должен видеть, из чего
            # складываются 18 с и сколько зарезервировано под адресную позднюю проверку.
            "early_budget_s": relevance.EARLY_SEMANTIC_BUDGET,
            "late_reserve_s": relevance.late_reserve_seconds(),
            "late_reserve_configured_s": relevance.LATE_SEMANTIC_RESERVE,
            "late_max_targets": relevance.LATE_MAX_TARGETS}


def cache_state() -> dict:
    """Сводка по дисковому кэшу: сколько записей и насколько они свежие."""
    from . import http as http_mod

    directory = http_mod.CACHE_DIR
    try:
        files = list(directory.glob("*.json"))
    except OSError:
        return {"directory": str(directory), "available": False, "entries": 0}
    newest = max((f.stat().st_mtime for f in files), default=None)
    return {"directory": str(directory), "available": True, "entries": len(files),
            "newest_entry_utc": (datetime.fromtimestamp(newest, timezone.utc).isoformat(timespec="seconds")
                                 if newest else None)}


def run_metadata(query_text: str, *, plan=None, provider_status: dict | None = None,
                 as_of: str | None = None, run_id: str | None = None) -> dict:
    """Полные метаданные запуска. Секретные значения не попадают сюда ни при каких условиях."""
    return {
        "run_id": run_id or uuid.uuid4().hex,
        "commit_sha": commit_sha(),
        "config_hash": config_hash(),
        "query": query_text,
        "as_of": as_of or date.today().isoformat(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cache_state": cache_state(),
        "provider_status": dict(provider_status or {}),
        "semantic_normalizer": semantic_normalizer_status(),
        "query_plan_source": getattr(plan, "source", None) if plan is not None else None,
    }
