"""Хранение запросов, найденных сигналов, источников и исключений. PostgreSQL в Docker, SQLite для локальной разработки.

DATABASE_URL=postgresql+psycopg://wsignals:wsignals@db:5432/wsignals   (docker compose)
DATABASE_URL не задан -> sqlite:///data/wsignals.db
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, inspect,
                        select, text)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

ROOT = Path(__file__).resolve().parents[1]


def _url() -> str:
    url = os.getenv("DATABASE_URL", "")
    if url.startswith("postgresql"):
        return url
    return f"sqlite:///{ROOT / 'data' / 'wsignals.db'}"


class Base(DeclarativeBase):
    pass


class SearchRun(Base):
    __tablename__ = "search_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="running")     # running, done, error
    created: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    finished: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    seeds: Mapped[list | None] = mapped_column(JSON, nullable=True)
    stats: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Метаданные запуска для воспроизводимости (R2-H): run_id, commit_sha, config_hash, кэш, адаптеры.
    run_meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    log: Mapped[list] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    signals: Mapped[list["Signal"]] = relationship(back_populates="run", cascade="all, delete-orphan", order_by="Signal.rank")
    excluded: Mapped[list["Excluded"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Signal(Base):
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("search_runs.id"))
    rank: Mapped[int] = mapped_column(Integer)
    technology: Mapped[str] = mapped_column(Text)
    # Ранжирующий балл, а не уверенность модели: см. REPAIR 4 и decision.Decision.rank_score.
    score: Mapped[float] = mapped_column(Float)
    # Подтверждённый слабый сигнал или список наблюдения (REPAIR 7).
    tier: Mapped[str | None] = mapped_column(String(40), nullable=True)
    card: Mapped[dict] = mapped_column(JSON)           # описание, преимущество, кейс, предикторы, двойник, признаки
    run: Mapped[SearchRun] = relationship(back_populates="signals")
    sources: Mapped[list["Source"]] = relationship(back_populates="signal", cascade="all, delete-orphan")


class Source(Base):
    __tablename__ = "sources"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signal_id: Mapped[int] = mapped_column(ForeignKey("signals.id"))
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    source_name: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(40))
    published: Mapped[str | None] = mapped_column(String(20), nullable=True)
    language: Mapped[str] = mapped_column(String(8))
    trust: Mapped[str] = mapped_column(String(20))
    trust_reason: Mapped[str] = mapped_column(Text)
    # ---- минимальное происхождение (PHASE 7) ----
    canonical_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    source_family: Mapped[str | None] = mapped_column(String(30), nullable=True)
    is_primary: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    adapter_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    signal: Mapped[Signal] = relationship(back_populates="sources")


class Excluded(Base):
    __tablename__ = "excluded"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("search_runs.id"))
    technology: Mapped[str] = mapped_column(Text)
    score: Mapped[float] = mapped_column(Float)
    decision: Mapped[str | None] = mapped_column(String(20), nullable=True)   # LOW_EVIDENCE | MATURE | HYPE | NOISE
    code: Mapped[str | None] = mapped_column(String(40), nullable=True)       # машиночитаемая причина
    reason: Mapped[str] = mapped_column(Text)
    run: Mapped[SearchRun] = relationship(back_populates="excluded")


_engine = None


# Поля источника, которые едут из карточки в БД и обратно.
SOURCE_FIELDS = ("title", "url", "source_name", "source_type", "language", "trust", "trust_reason",
                 "canonical_url", "retrieved_at", "source_family", "content_hash", "adapter_status")
ADDED_EXCLUDED_COLUMNS = {"decision": "VARCHAR(20)", "code": "VARCHAR(40)"}
ADDED_SIGNAL_COLUMNS = {"tier": "VARCHAR(40)"}
ADDED_RUN_COLUMNS = {"run_meta": "JSON"}
ADDED_SOURCE_COLUMNS = {"canonical_url": "TEXT", "retrieved_at": "VARCHAR(40)", "source_family": "VARCHAR(30)",
                        "is_primary": "BOOLEAN", "content_hash": "VARCHAR(64)", "adapter_status": "VARCHAR(20)"}


def _add_missing_columns(eng) -> None:
    """Дополняет существующую таблицу источников новыми колонками происхождения.

    Миграций в проекте нет и вводить их в этой задаче не нужно: колонки добавляются только как NULL-able,
    ничего не переносится и не переписывается."""
    insp = inspect(eng)
    tables = set(insp.get_table_names())
    for table, columns in (("sources", ADDED_SOURCE_COLUMNS), ("excluded", ADDED_EXCLUDED_COLUMNS),
                           ("signals", ADDED_SIGNAL_COLUMNS), ("search_runs", ADDED_RUN_COLUMNS)):
        if table not in tables:
            continue
        have = {c["name"] for c in insp.get_columns(table)}
        missing = {k: v for k, v in columns.items() if k not in have}
        if not missing:
            continue
        with eng.begin() as conn:
            for name, ddl in missing.items():
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


def engine():
    global _engine
    if _engine is None:
        _engine = create_engine(_url(), pool_pre_ping=True)
        _add_missing_columns(_engine)
        Base.metadata.create_all(_engine)
    return _engine


WATCHLIST_TIER = "требует дополнительных доказательств"
CONFIRMED_TIER = "подтверждённый слабый сигнал"


def is_confirmed(card: dict) -> bool:
    """Подтверждённым считается только запись, у которой ЗАПИСАНО авторитетное решение ACCEPT.

    Раньше подтверждённым считалось всё, что не помечено как список наблюдения. Записи прежних
    версий без колонки tier (NULL после миграции) или без поля decision в карточке поэтому
    показывались как подтверждённые слабые сигналы, хотя состояния подтверждения у них нет."""
    return card.get("decision") == "ACCEPT" and card.get("tier") == CONFIRMED_TIER


def split_cards(cards: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """(подтверждённые, список наблюдения, записи без достаточного состояния подтверждения)."""
    confirmed = [c for c in cards if is_confirmed(c)]
    watchlist = [c for c in cards if not is_confirmed(c) and c.get("tier") == WATCHLIST_TIER]
    unverified = [c for c in cards if not is_confirmed(c) and c.get("tier") != WATCHLIST_TIER]
    return confirmed, watchlist, unverified


def _cards(run: SearchRun) -> list[dict]:
    return [{**sig.card, "id": sig.id, "rank": sig.rank, "tier": sig.tier, "sources": [
        {**{c: getattr(src, c) for c in (*SOURCE_FIELDS, "published", "is_primary")}, "published_at": src.published}
        for src in sig.sources]} for sig in run.signals]


def new_run(query: str) -> int:
    with Session(engine()) as s:
        run = SearchRun(query=query, log=[])
        s.add(run)
        s.commit()
        return run.id


def append_log(run_id: int, line: str) -> None:
    with Session(engine()) as s:
        run = s.get(SearchRun, run_id)
        run.log = [*run.log, line]
        s.commit()


def save_result(run_id: int, result: dict) -> None:
    with Session(engine()) as s:
        run = s.get(SearchRun, run_id)
        run.status, run.finished = "done", datetime.now(timezone.utc)
        run.seeds, run.stats = result["seeds"], result["stats"]
        run.run_meta = result.get("run")
        for i, card in enumerate([*result["signals"], *result.get("watchlist", [])], 1):
            sig = Signal(rank=i, technology=card["technology"], score=card["score"],
                         tier=card.get("tier"),
                         card={k: v for k, v in card.items() if k != "sources"})
            sig.sources = [Source(**{k: (src.get(k) if k in ("canonical_url", "retrieved_at", "source_family",
                                                             "content_hash", "adapter_status") else (src.get(k) or ""))
                                     for k in SOURCE_FIELDS},
                                  is_primary=src.get("is_primary"),
                                  published=src.get("published")) for src in card["sources"]]
            run.signals.append(sig)
        run.excluded = [Excluded(**{k: v for k, v in e.items() if k in ("technology", "score", "decision", "code", "reason")})
                        for e in result["excluded"]]
        s.commit()


def fail(run_id: int, error: str) -> None:
    with Session(engine()) as s:
        run = s.get(SearchRun, run_id)
        run.status, run.error, run.finished = "error", error[:2000], datetime.now(timezone.utc)
        s.commit()


def get_run(run_id: int) -> dict | None:
    with Session(engine()) as s:
        run = s.get(SearchRun, run_id)
        if not run:
            return None
        confirmed, watchlist, unverified = split_cards(_cards(run))
        return {
            "id": run.id, "query": run.query, "status": run.status, "created": run.created.isoformat(),
            "finished": run.finished.isoformat() if run.finished else None, "seeds": run.seeds, "stats": run.stats,
            "log": run.log, "error": run.error, "run": run.run_meta,
            "signals": confirmed, "watchlist": watchlist, "unverified_legacy": unverified,
            "excluded": [{"technology": e.technology, "score": e.score, "decision": e.decision,
                          "code": e.code, "reason": e.reason} for e in run.excluded],
        }


def list_runs(limit: int = 20) -> list[dict]:
    with Session(engine()) as s:
        rows = s.scalars(select(SearchRun).order_by(SearchRun.id.desc()).limit(limit)).all()
        return [{"id": r.id, "query": r.query, "status": r.status, "created": r.created.isoformat(), "stats": r.stats} for r in rows]
