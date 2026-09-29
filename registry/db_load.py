"""Загрузка базы слабых сигналов в PostgreSQL (таблицы registry_signals, registry_series) рядом с таблицами сервиса.

    DATABASE_URL=postgresql+psycopg://wsignals:wsignals@localhost:5432/wsignals python -m registry.db_load
    python -m registry.db_load              # без DATABASE_URL — SQLite data/registry.db (проверка локально)

Таблицы сервиса (search_runs, signals, sources, excluded) не трогаются. Загрузка идемпотентна: таблицы реестра
очищаются и заполняются заново из signal_base.parquet той версии реестра, на которую указывает GPB_REGISTRY.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .corpus import REGISTRY, SOURCES, YEARS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Base(DeclarativeBase):
    pass


class RegistrySignal(Base):
    """Одна запись базы: технология, прошедшая отбор (данные + одна проверка YandexGPT при сборке)."""
    __tablename__ = "registry_signals"
    key: Mapped[str] = mapped_column(String(200), primary_key=True)     # ключ сущности (отсортированные основы слов)
    phrase: Mapped[str] = mapped_column(Text)                           # английский термин из корпуса
    name_ru: Mapped[str] = mapped_column(Text)
    description_ru: Mapped[str] = mapped_column(Text)
    areas: Mapped[list] = mapped_column(JSON)                           # области ТЗ
    aliases: Mapped[list] = mapped_column(JSON)                         # склеенные варианты написания
    kind: Mapped[str | None] = mapped_column(String(40), nullable=True) # тип по проверке: метод, устройство, …
    first_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    expert: Mapped[float] = mapped_column(Float)                        # экспертный балл (порядок выдачи)
    org_sim: Mapped[float] = mapped_column(Float)
    biz_layers: Mapped[int] = mapped_column(Integer)                    # сколько бизнес-слоёв: HN, YC, пресса
    learned: Mapped[float] = mapped_column(Float)
    registry: Mapped[str] = mapped_column(Text)                         # версия реестра (каталог)
    loaded: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RegistrySeries(Base):
    """Число работ по годам и источникам — для графика в карточке и проверки роста."""
    __tablename__ = "registry_series"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(ForeignKey("registry_signals.key"))
    source: Mapped[str] = mapped_column(String(20))
    year: Mapped[int] = mapped_column(Integer)
    n: Mapped[int] = mapped_column(Integer)


class RegistryActor(Base):
    """Связь «технология → кто за ней стоит»: организация из авторов свежих работ или стартап YC (registry/actors.py).
    Граф в виде связей: узлы — технологии (registry_signals) и организации, рёбра — эта таблица."""
    __tablename__ = "registry_actors"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(ForeignKey("registry_signals.key"))
    relation: Mapped[str] = mapped_column(String(20))                   # research (работы) | startup (YC)
    name: Mapped[str] = mapped_column(Text)
    org_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    country: Mapped[str | None] = mapped_column(String(8), nullable=True)
    works: Mapped[int | None] = mapped_column(Integer, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)


def _url() -> str:
    url = os.getenv("DATABASE_URL", "")
    return url if url.startswith("postgresql") else f"sqlite:///{os.path.join(ROOT, 'data', 'registry.db')}"


def main() -> int:
    b = pd.read_parquet(REGISTRY / "signal_base.parquet")
    sci = pd.read_parquet(REGISTRY / "all_phrases.parquet", columns=["key", "eid"])
    eid = dict(zip(sci.key, sci.eid))
    df = np.load(REGISTRY / "counts.npz")["df"]
    eng = create_engine(_url(), pool_pre_ping=True)
    Base.metadata.create_all(eng)
    now = datetime.now(timezone.utc)
    to_list = lambda v: [str(x) for x in v] if v is not None and not isinstance(v, str) else []
    import json
    act = REGISTRY / "actors.parquet"
    actors = pd.read_parquet(act) if act.exists() else None
    with Session(eng) as s:
        s.execute(delete(RegistryActor))
        s.execute(delete(RegistrySeries))
        s.execute(delete(RegistrySignal))
        for r in b.itertuples():
            s.add(RegistrySignal(key=r.key, phrase=r.phrase.replace("_", " "), name_ru=r.name_ru,
                                 description_ru=r.description_ru, areas=to_list(r.areas), aliases=to_list(r.aliases),
                                 kind=r.kind, first_year=int(r.first_year) if pd.notna(r.first_year) else None,
                                 expert=float(r.expert), org_sim=float(r.org_sim), biz_layers=int(r.biz),
                                 learned=float(r.learned), registry=REGISTRY.name, loaded=now))
        s.flush()
        n_series = 0
        for r in b.itertuples():
            e = eid.get(r.key)
            if e is None:
                continue
            for si, src in enumerate(SOURCES):
                for yi, y in enumerate(YEARS):
                    n = int(df[int(e), si, yi])
                    if n:
                        s.add(RegistrySeries(key=r.key, source=src, year=int(y), n=n))
                        n_series += 1
        n_act = 0
        if actors is not None:
            base_keys = set(b.key)
            for a in actors.itertuples():
                if a.key not in base_keys:
                    continue
                for o in json.loads(a.top_orgs):
                    s.add(RegistryActor(key=a.key, relation="research", name=o["name"], org_type=o["type"] or None,
                                        country=o["country"] or None, works=int(o["works"])))
                    n_act += 1
                for y in json.loads(a.yc):
                    s.add(RegistryActor(key=a.key, relation="startup", name=y["name"], url=y.get("url")))
                    n_act += 1
        s.commit()
    print(f"{_url().split('@')[-1]}: registry_signals {len(b)}, registry_series {n_series}, registry_actors {n_act} "
          f"(реестр {REGISTRY.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
