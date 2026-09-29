"""REST API сервиса слабых сигналов.

    uvicorn api.main:app --reload --port 8000

POST /api/search          {"query": "слабые сигналы в кибербезопасности"} -> {"id": 1, "status": "running"}
GET  /api/search/{id}     статус, журнал выполнения, ТОП-15 с карточками и источниками, исключённые кандидаты
GET  /api/search          последние запросы
GET  /api/model           метрики модели этапа 1
GET  /api/health          модель, база, бюджет интерактивного запроса
"""
from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from wsignals import db
from wsignals.query import run as run_query

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Слабые сигналы: API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SearchIn(BaseModel):
    query: str = Field(min_length=2, max_length=300, examples=["слабые сигналы в кибербезопасности"])
    top: int = Field(15, ge=1, le=30)
    candidates: int = Field(160, ge=10, le=250)
    # Общий бюджет интерактивного запроса. По умолчанию берётся из WSIGNALS_QUERY_BUDGET (90 с).
    budget_s: float | None = Field(None, ge=10, le=600)


def _worker(run_id: int, body: SearchIn) -> None:
    try:
        result = run_query(body.query, body.top, body.candidates,
                           log=lambda line: db.append_log(run_id, line), budget_s=body.budget_s)
        db.save_result(run_id, result)
    except Exception as e:  # noqa: BLE001
        db.fail(run_id, f"{e}\n{traceback.format_exc()}")


@app.post("/api/search")
def search(body: SearchIn):
    run_id = db.new_run(body.query)
    threading.Thread(target=_worker, args=(run_id, body), daemon=True).start()
    return {"id": run_id, "status": "running"}


@app.get("/api/search/{run_id}")
def get_search(run_id: int):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(404, "запрос не найден")
    return run


@app.get("/api/search")
def list_searches(limit: int = 20):
    return db.list_runs(limit)


@app.get("/api/model")
def model_info():
    path = ROOT / "reports" / "model_metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@app.get("/api/health")
def health():
    """Готовность сервиса: модель на месте, база отвечает, бюджет интерактивного запроса известен."""
    from wsignals.query import TARGET_SECONDS, budget_plan

    try:
        db.list_runs(1)
        database = "ok"
    except Exception as e:  # noqa: BLE001 — health не должен падать
        database = f"error: {type(e).__name__}"
    plan = budget_plan()
    return {"status": "ok" if database == "ok" else "degraded",
            "model": (ROOT / "models" / "signal_model.joblib").exists(),
            "database": database,
            "query_budget_s": plan["hard_wall_budget_s"], "query_target_s": TARGET_SECONDS,
            **plan}
