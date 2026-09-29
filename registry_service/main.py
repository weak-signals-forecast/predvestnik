"""API базы слабых сигналов (контейнер registry).

    uvicorn registry_service.main:app --port 8010

POST /api/registry/search   {"query": "перспективные технологии в финтехе"} -> ТОП-15 карточек в формате ТЗ
GET  /api/registry/health   готовность: база, векторная модель, PostgreSQL, ключ YandexGPT
GET  /api/registry/model    общая картина моделей (SHAP) — что влияет на силу сигнала
GET  /                      витрина (статическая страница с примерами)

При старте база сигналов и связи «кто стоит за сигналом» загружаются в PostgreSQL (DATABASE_URL), каждый запрос и
его ответ записываются туда же (таблица registry_runs). Без DATABASE_URL — SQLite в каталоге кэша.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .service import CACHE_DIR, registry

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Слабые сигналы: база сигналов", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Base(DeclarativeBase):
    pass


class Signal(Base):
    __tablename__ = "registry_signals"
    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    term_en: Mapped[str] = mapped_column(Text)
    name_ru: Mapped[str] = mapped_column(Text)
    description_ru: Mapped[str] = mapped_column(Text)
    areas: Mapped[list] = mapped_column(JSON)
    first_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score: Mapped[float] = mapped_column(Float)                  # смесь моделей (ранг), порядок выдачи
    expert_p: Mapped[float | None] = mapped_column(Float, nullable=True)
    history_p: Mapped[float | None] = mapped_column(Float, nullable=True)
    card: Mapped[dict] = mapped_column(JSON)                     # источники, ряды, SHAP, кто стоит за сигналом


class Actor(Base):
    __tablename__ = "registry_actors"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(ForeignKey("registry_signals.key"))
    relation: Mapped[str] = mapped_column(String(20))           # research | startup
    name: Mapped[str] = mapped_column(Text)
    org_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    country: Mapped[str | None] = mapped_column(String(8), nullable=True)
    works: Mapped[int | None] = mapped_column(Integer, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)


class Run(Base):
    __tablename__ = "registry_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    created: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    took_s: Mapped[float] = mapped_column(Float)
    result: Mapped[dict] = mapped_column(JSON)


def _db_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    return url if url.startswith("postgresql") else f"sqlite:///{CACHE_DIR / 'registry.db'}"


_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = create_engine(_db_url(), pool_pre_ping=True)
    return _engine


def load_database() -> dict:
    """База сигналов и связи — в PostgreSQL (идемпотентно: таблицы реестра очищаются и заполняются заново)."""
    r = registry()
    Base.metadata.create_all(engine())
    n_act = 0
    with Session(engine()) as s:
        s.execute(delete(Actor))
        s.execute(delete(Signal))
        for i in r.sig.index:
            k = r.sig.key[i]
            c = r.cards.get(k, {})
            fy = r.sig.first_year[i]
            s.add(Signal(key=k, term_en=r.ph[i], name_ru=r.sig.name_ru[i], description_ru=r.sig.description_ru[i],
                         areas=list(r.sig.areas[i]), first_year=int(fy) if fy == fy else None,
                         score=float(r.sig.expert[i]), card=c,
                         expert_p=float(r.sig.expert_p[i]) if r.sig.expert_p[i] == r.sig.expert_p[i] else None,
                         history_p=float(r.sig.history_p[i]) if r.sig.history_p[i] == r.sig.history_p[i] else None))
        s.flush()
        for k, c in r.cards.items():
            a = c.get("actors") or {}
            for o in a.get("top_orgs") or []:
                s.add(Actor(key=k, relation="research", name=o["name"], org_type=o.get("type") or None,
                            country=o.get("country") or None, works=int(o.get("works") or 0)))
                n_act += 1
            for y in a.get("yc") or []:
                s.add(Actor(key=k, relation="startup", name=y.get("name") or "", url=y.get("url")))
                n_act += 1
        s.commit()
    return {"signals": len(r.sig), "actors": n_act}


_DB_STATE: dict = {}


@app.on_event("startup")
def startup() -> None:
    try:
        _DB_STATE.update(load_database(), status="ok", url=_db_url().split("@")[-1])
    except Exception as e:  # noqa: BLE001
        _DB_STATE.update(status=f"ошибка: {str(e)[:200]}")


class SearchIn(BaseModel):
    query: str = Field(min_length=2, max_length=300, examples=["перспективные технологии в финтехе"])
    use_llm: bool = Field(True, description="false — без YandexGPT: ТОП-15 по порядку поиска")


@app.post("/api/registry/search")
def search(body: SearchIn):
    res = registry().answer(body.query.strip(), use_llm=body.use_llm)
    try:
        with Session(engine()) as s:
            s.add(Run(query=body.query, created=datetime.now(timezone.utc), took_s=res["meta"]["took_s"], result=res))
            s.commit()
    except Exception:  # noqa: BLE001  запись в журнал не должна ронять ответ
        pass
    return res


@app.get("/api/registry/health")
def health():
    r = registry()
    from .service import MODEL
    return {"base_size": len(r.sig), "cards": len(r.cards), "embedding_model": MODEL, "database": _DB_STATE,
            "yandexgpt_key": r._llm_ready()}


@app.get("/api/registry/model")
def model():
    return registry().model_global


web = ROOT / "web"
if web.exists():
    app.mount("/", StaticFiles(directory=web, html=True), name="web")
