"""Явное состояние доказательства: OK / NO_RESULTS / ERROR (PHASE 1).

Главная правка ремонта. До неё отказ источника превращался в нули признаков, а нули модель этапа 1
читает как «технология никому не известна», то есть **отсутствие данных повышало уверенность в слабом
сигнале** (аудит: P(weak | пустой след) = 0.915). Теперь наблюдение источника несёт свой статус:

  OK          источник ответил и что-то нашёл;
  NO_RESULTS  источник ответил, но по фразе ничего нет. Это валидный ноль, его можно использовать;
  ERROR       источник не ответил. Числом ноль НЕ становится и в расчёт признаков не входит.

Инварианты, закреплённые тестами:
  * ERROR никогда не увеличивает ни emergence_score, ни evidence_confidence;
  * если все источники в ERROR, принять кандидата нельзя ни при каких значениях модели;
  * NO_RESULTS и ERROR различимы в выдаче, в журнале адаптеров и в БД.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import http

OK, NO_RESULTS, ERROR = "OK", "NO_RESULTS", "ERROR"

# Семейство источника: независимость подтверждений считается по семействам, а не по адаптерам,
# иначе «две независимые проверки» выполняются двумя лентами одного агрегатора.
SOURCE_FAMILY = {
    "openalex": "science",
    "github": "code",
    "news": "media",
    "hn": "community",
    "wiki": "encyclopedia",
}
# Семейства, которые засчитываются как независимое подтверждение существования технологии.
CORROBORATING_FAMILIES = {"science", "code", "media"}
# Семейства класса D: ранний индикатор, сам по себе подтверждением не является (ТЗ: соцсети, сообщества).
LEAD_ONLY_FAMILIES = {"community"}
# Википедия отвечает на вопрос о зрелости, а не о существовании: в подтверждения не входит.
MATURITY_FAMILIES = {"encyclopedia"}

# По какому полю видно, что источник действительно что-то нашёл.
_POSITIVE = {
    "openalex": lambda v: (v.get("oa_total") or 0) > 0,
    # gh_corpus_repos — новые репозитории за 12 месяцев, найденные при сборе корпуса запроса.
    # Это реально полученное измерение, а не подстановка: см. query._corpus_code_observation.
    "github": lambda v: (v.get("gh_total") or 0) > 0 or (v.get("gh_corpus_repos") or 0) > 0,
    "news": lambda v: (v.get("news_1y") or 0) > 0,
    "hn": lambda v: (v.get("hn_total") or 0) > 0,
    "wiki": lambda v: bool(v.get("wiki_article")) or (v.get("wiki_mentions_log") or 0) > 0,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Observation:
    """Одно наблюдение одного источника по одной фразе."""
    source: str
    status: str                                   # OK | NO_RESULTS | ERROR
    values: dict = field(default_factory=dict)    # счётчики; при ERROR остаётся пустым
    error_type: str | None = None                 # timeout | rate_limit | server_error | network | http_error | deadline | parse
    error: str | None = None
    retrieved_at: str | None = None
    from_cache: bool = False
    stale: bool = False

    @property
    def family(self) -> str:
        return SOURCE_FAMILY.get(self.source, "other")

    @property
    def succeeded(self) -> bool:
        """Источник ответил. Ноль в ответе это тоже ответ."""
        return self.status in (OK, NO_RESULTS)

    def to_dict(self) -> dict:
        return {"source": self.source, "family": self.family, "status": self.status,
                "error_type": self.error_type, "error": self.error, "retrieved_at": self.retrieved_at,
                "from_cache": self.from_cache, "stale": self.stale}


def observe(source: str, fn, *args, **kwargs) -> Observation:
    """Вызывает адаптер и возвращает наблюдение с явным статусом.

    Любое исключение адаптера становится ERROR, а не нулями. Класс ошибки берётся из FetchError,
    если адаптер его поднял, иначе это «parse» или «network» по типу исключения."""
    meta_before = len(http.current_context().fetches) if http.current_context() else 0
    try:
        values = fn(*args, **kwargs) or {}
    except http.FetchError as e:
        return Observation(source=source, status=ERROR, error_type=e.error_type, error=str(e)[:200],
                           retrieved_at=_now())
    except Exception as e:  # noqa: BLE001 — отказоустойчивость парсеров по ТЗ
        kind = http.TIMEOUT if isinstance(e, TimeoutError) else http.PARSE
        return Observation(source=source, status=ERROR, error_type=kind, error=f"{type(e).__name__}: {e}"[:200],
                           retrieved_at=_now())
    ctx = http.current_context()
    fetches = ctx.fetches[meta_before:] if ctx else []
    from_cache = bool(fetches) and all(f.get("from_cache") for f in fetches)
    stale = any(f.get("stale") for f in fetches)
    retrieved = min((f["retrieved_at"] for f in fetches if f.get("retrieved_at")), default=_now())
    positive = _POSITIVE.get(source, lambda v: bool(v))(values)
    return Observation(source=source, status=OK if positive else NO_RESULTS, values=values,
                       retrieved_at=retrieved, from_cache=from_cache, stale=stale)


@dataclass
class EvidenceState:
    """Все наблюдения по одному кандидату плюс производные, которыми пользуется решение."""
    observations: dict[str, Observation] = field(default_factory=dict)

    def add(self, obs: Observation) -> None:
        prev = self.observations.get(obs.source)
        # Один источник может опрашиваться в несколько заходов (каскад): успех не затирается отказом,
        # а отказ не затирает успех — значения сливаются.
        if prev is None:
            self.observations[obs.source] = obs
            return
        if prev.status == ERROR and obs.succeeded:
            self.observations[obs.source] = obs
        elif prev.succeeded and obs.succeeded:
            merged = {**prev.values, **obs.values}
            status = OK if _POSITIVE.get(obs.source, lambda v: bool(v))(merged) else NO_RESULTS
            self.observations[obs.source] = Observation(
                source=obs.source, status=status, values=merged,
                retrieved_at=min(x for x in (prev.retrieved_at, obs.retrieved_at) if x),
                from_cache=prev.from_cache and obs.from_cache, stale=prev.stale or obs.stale)

    # ---------- срезы ----------

    def succeeded(self) -> dict[str, Observation]:
        return {k: o for k, o in self.observations.items() if o.succeeded}

    def failed(self) -> dict[str, Observation]:
        return {k: o for k, o in self.observations.items() if o.status == ERROR}

    def ok(self, source: str) -> bool:
        """Источник ответил (OK или NO_RESULTS). Только тогда его нулю можно верить."""
        o = self.observations.get(source)
        return bool(o and o.succeeded)

    def positive_families(self) -> set[str]:
        return {o.family for o in self.observations.values() if o.status == OK}

    def corroborating_families(self) -> set[str]:
        return self.positive_families() & CORROBORATING_FAMILIES

    def lead_only_families(self) -> set[str]:
        return self.positive_families() & LEAD_ONLY_FAMILIES

    def completeness(self) -> float:
        """Доля источников, которые ответили. Отказ снижает её и через неё снижает доверие к выводу."""
        if not self.observations:
            return 0.0
        return len(self.succeeded()) / len(self.observations)

    def all_failed(self) -> bool:
        return bool(self.observations) and not self.succeeded()

    def values(self) -> dict:
        """Счётчики только по ответившим источникам. Значений отказавшего источника здесь нет."""
        out: dict = {}
        for o in self.observations.values():
            if o.succeeded:
                out.update(o.values)
        return out

    def uses_stale_cache(self) -> bool:
        return any(o.stale for o in self.observations.values() if o.succeeded)

    def retrieved_at(self) -> str | None:
        stamps = [o.retrieved_at for o in self.observations.values() if o.succeeded and o.retrieved_at]
        return min(stamps) if stamps else None

    def adapter_status(self) -> dict[str, dict]:
        return {k: o.to_dict() for k, o in sorted(self.observations.items())}

    def to_dict(self) -> dict:
        return {
            "adapters": self.adapter_status(),
            "failed_adapters": sorted(self.failed()),
            "no_result_adapters": sorted(k for k, o in self.observations.items() if o.status == NO_RESULTS),
            "positive_families": sorted(self.positive_families()),
            "corroborating_families": sorted(self.corroborating_families()),
            "completeness": round(self.completeness(), 3),
            "retrieved_at": self.retrieved_at(),
            "from_cache": self.uses_stale_cache(),
        }
