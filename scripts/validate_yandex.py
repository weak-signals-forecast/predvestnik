"""Проверка реального пути YandexGPT на трёх контрольных запросах (REPAIR 5).

    LLM_PROVIDER=yandexgpt YANDEX_API_KEY=... YANDEX_FOLDER_ID=... \
    YANDEX_QUERY_MODEL=yandexgpt-5-lite python scripts/validate_yandex.py

Печатает точный идентификатор модели (`gpt://<folder>/<model>`) и результат разбора каждого запроса.
Секреты не печатаются никогда: ключ в вывод не попадает, идентификатор папки печатается только в составе
modelUri, потому что без него идентификатор модели неполон — при необходимости скройте его в отчёте.

Код возврата 0 — все три запроса разобраны моделью и прошли проверку схемы.
Код возврата 2 — учётных данных нет: это BLOCKED, а не PASS.
Код возврата 1 — учётные данные есть, но путь не отработал.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERIES = [
    "перспективные технологии управления квантовыми сенсорами",
    "новые способы охлаждения дата-центров",
    "технологии защиты автономных ИИ-агентов",
]

REQUIRED_ENV = ["LLM_PROVIDER=yandexgpt", "YANDEX_API_KEY", "YANDEX_FOLDER_ID",
                "YANDEX_QUERY_MODEL (необязательно: модель закреплена в коде)"]


def main() -> int:
    from wsignals import decompose, relevance
    from wsignals.query import parse_query

    model_uri, why = decompose.configured_model()
    if not model_uri:
        print("BLOCKED_REAL_YANDEX_VALIDATION")
        print(f"причина: {why}")
        print("требуются переменные окружения: " + ", ".join(REQUIRED_ENV))
        print(f"закреплённая модель: {decompose.RUNTIME_YANDEX_MODEL}")
        print(f"разрешённые модели: {', '.join(sorted(decompose.ALLOWED_YANDEX_MODELS))}")
        return 2

    print(f"MODEL_URI: {model_uri}")
    print(f"temperature: {decompose.TEMPERATURE}, timeout: {decompose.TIMEOUT}s")
    ok = True
    for q in QUERIES:
        plan = decompose.decompose(q, parse_query)
        status = "LLM" if plan.source == "llm" else f"FALLBACK ({plan.fallback_reason})"
        print(f"\n--- {q}\n    путь: {status}")
        print("    " + json.dumps(plan.to_dict(), ensure_ascii=False)[:1200])
        if plan.source != "llm":
            ok = False
            continue
        assert plan.domain and plan.english_queries, "схема пуста"
        assert plan.model == model_uri, "в плане другой идентификатор модели"
        # Семантическая нормализация на кандидатах, построенных из самого плана: проверяем второй вызов.
        cands = [{"phrase": p, "original_phrase": p, "burst": 1.0,
                  "channels": {"литература": 2}, "recent_docs": 2, "prior_docs": 0, "examples": [], "repos": []}
                 for p in plan.english_queries[:6]]
        keep, dropped = relevance.canonicalize(cands, domain=plan.domain)
        source = (keep or dropped)[0].get("assessment", {}).get("source") if (keep or dropped) else "нет"
        print(f"    канонизация: оставлено {len(keep)}, отбраковано {len(dropped)}, путь оценки: {source}")
    print("\nREAL_YANDEXGPT_RUN: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
