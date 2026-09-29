"""HTTP-клиент с дисковым кэшем, повторами и ограничением частоты по хостам.

Кэш делает сбор воспроизводимым: признаки датасета фиксируются на дату снимка и не пересчитываются
при каждом запуске обучения. Ключ кэша: метод, URL и параметры. Секреты в ключ не попадают.

Два профиля повторов (PHASE 6):
  BATCH  офлайн-сбор датасета: длинные ожидания допустимы, важна полнота;
  LIVE   интерактивный запрос из интерфейса: жёсткий таймаут на запрос, мало повторов,
         общий дедлайн на весь запрос. Ни одно ожидание не выходит за дедлайн.

Каждый вызов `get` фиксирует происхождение ответа (`retrieved_at`, `from_cache`, `stale`,
`canonical_url`, `content_hash`) в контексте `fetch_context`, откуда его забирает слой доказательств.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.getenv("WSIGNALS_CACHE", ROOT / "data" / "cache"))
# Wikimedia требует User-Agent с контактом, иначе быстро отвечает 429
USER_AGENT = "wsignals/0.2 (https://github.com/weak-signals-forecast/predvestnik; research on emerging technologies)"

# Минимальный интервал между запросами к хосту, секунды. Подобран под публичные лимиты.
MIN_INTERVAL = {
    "api.openalex.org": 0.12,
    "api.github.com": 2.2,       # поиск: 30 запросов в минуту с токеном
    "hn.algolia.com": 0.25,
    "news.google.com": 1.0,
    "en.wikipedia.org": 0.5,
    "wikimedia.org": 0.5,
    "export.arxiv.org": 3.5,
}

_locks: dict[str, threading.Lock] = {}
_last: dict[str, float] = {}
_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT


@dataclass(frozen=True)
class RetryPolicy:
    """Сколько раз и как долго повторять запрос. Значения в секундах."""
    retries: int
    timeout: float
    backoff_base: float
    backoff_cap: float


# Офлайн-сбор: как было до ремонта, полнота важнее времени.
BATCH = RetryPolicy(retries=7, timeout=30.0, backoff_base=2.0, backoff_cap=90.0)
# Живой путь: худший случай на адаптер ≈ timeout*(retries+1) + паузы ≈ 25 с, и всё равно режется дедлайном.
LIVE = RetryPolicy(
    retries=int(os.getenv("WSIGNALS_LIVE_RETRIES", "2")),
    timeout=float(os.getenv("WSIGNALS_LIVE_TIMEOUT", "8")),
    backoff_base=float(os.getenv("WSIGNALS_LIVE_BACKOFF", "0.75")),
    backoff_cap=float(os.getenv("WSIGNALS_LIVE_BACKOFF_CAP", "3")),
)

# Меньше этого запускать новый сетевой запрос бессмысленно: он всё равно не успеет ответить,
# а попытка съест остаток бюджета финализации (REPAIR E).
MIN_REQUEST_BUDGET_S = float(os.getenv("WSIGNALS_MIN_REQUEST_BUDGET", "1.5"))

# Классы ошибок адаптера. Ноль в данных и отказ источника это разные вещи (PHASE 1).
TIMEOUT, RATE_LIMIT, SERVER_ERROR, NETWORK, HTTP_ERROR, DEADLINE, PARSE, AUTH_ERROR = (
    "timeout", "rate_limit", "server_error", "network", "http_error", "deadline", "parse", "auth_error")


class FetchError(RuntimeError):
    """Отказ источника. Несёт класс ошибки, чтобы выше по стеку его не спутали с нулём."""

    def __init__(self, message: str, error_type: str = HTTP_ERROR, status_code: int | None = None):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code


class DeadlineExceeded(FetchError):
    def __init__(self, message: str):
        super().__init__(message, DEADLINE)


# ---------- контекст живого запроса: дедлайн и журнал происхождения ----------

_ctx = threading.local()


@dataclass
class _Context:
    deadline: float | None            # time.monotonic(), после которого новые запросы не начинаются
    policy: RetryPolicy
    fetches: list[dict]
    lock: threading.Lock


def monotonic() -> float:
    """Единая точка времени. Вынесена отдельно, чтобы тесты подменяли часы и не зависели от реального
    хода времени: измерения бюджета не должны быть плавающими."""
    return time.monotonic()


def _context() -> _Context | None:
    return getattr(_ctx, "ctx", None)


@contextmanager
def fetch_context(budget_s: float | None = None, policy: RetryPolicy | None = None):
    """Открывает контекст живого запроса: общий дедлайн, профиль повторов и журнал происхождения.

    Контекст наследуется дочерними потоками только явно, через `adopt_context`: ThreadPoolExecutor
    создаёт свои потоки, у которых нет thread-local родителя."""
    ctx = _Context(deadline=(monotonic() + budget_s) if budget_s else None,
                   policy=policy or (LIVE if budget_s else BATCH), fetches=[], lock=threading.Lock())
    prev = getattr(_ctx, "ctx", None)
    _ctx.ctx = ctx
    try:
        yield ctx
    finally:
        _ctx.ctx = prev


@contextmanager
def phase(seconds: float):
    """Бюджет одного этапа: дедлайн временно ужимается, но никогда не растягивается.

    Нужен, чтобы ранние этапы не съедали весь бюджет и решающий этап всегда получал свою долю."""
    ctx = _context()
    if ctx is None or seconds is None:
        yield
        return
    previous = ctx.deadline
    limit = monotonic() + seconds
    ctx.deadline = limit if previous is None else min(previous, limit)
    try:
        yield
    finally:
        ctx.deadline = previous


@contextmanager
def adopt_context(ctx: "_Context | None"):
    """Переносит контекст родителя в рабочий поток пула."""
    prev = getattr(_ctx, "ctx", None)
    _ctx.ctx = ctx
    try:
        yield ctx
    finally:
        _ctx.ctx = prev


def current_context() -> "_Context | None":
    return _context()


def time_left() -> float | None:
    ctx = _context()
    if ctx is None or ctx.deadline is None:
        return None
    return ctx.deadline - monotonic()


def deadline_passed() -> bool:
    left = time_left()
    return left is not None and left <= 0


def call_budget(default_timeout: float, minimum: float | None = None) -> float | None:
    """Таймаут для сетевого вызова, который идёт МИМО этого модуля (обращения к языковой модели).

    Возвращает допустимый таймаут или None, если начинать вызов уже нельзя. Благодаря этому общий
    дедлайн распространяется и на вызовы модели: раньше они шли напрямую через requests со своим
    таймаутом в 60 с и продлевали ответ далеко за объявленную стену (REPAIR C)."""
    floor = MIN_REQUEST_BUDGET_S if minimum is None else minimum
    left = time_left()
    if left is None:
        return default_timeout
    if left < floor:
        return None
    return min(default_timeout, left)


def too_late_to_start(minimum: float | None = None) -> bool:
    """Осталось меньше, чем нужно на осмысленный запрос. Новую сетевую работу начинать нельзя."""
    left = time_left()
    return left is not None and left < (MIN_REQUEST_BUDGET_S if minimum is None else minimum)


def _record(meta: dict) -> None:
    ctx = _context()
    if ctx is not None:
        with ctx.lock:
            ctx.fetches.append(meta)


def taken_fetches() -> list[dict]:
    """Забирает и очищает журнал запросов текущего контекста."""
    ctx = _context()
    if ctx is None:
        return []
    with ctx.lock:
        out, ctx.fetches = list(ctx.fetches), []
    return out


# ---------- сам клиент ----------

def _key(url: str, params: dict | None) -> str:
    raw = json.dumps([url, sorted((params or {}).items())], ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()


def _throttle(host: str) -> None:
    lock = _locks.setdefault(host, threading.Lock())
    with lock:
        wait = MIN_INTERVAL.get(host, 0.3) - (monotonic() - _last.get(host, 0))
        if wait > 0:
            _sleep(wait)
        _last[host] = monotonic()


def _sleep(seconds: float) -> None:
    """Пауза, которая никогда не выходит за общий дедлайн запроса."""
    left = time_left()
    if left is not None:
        seconds = min(seconds, max(0.0, left))
    if seconds > 0:
        time.sleep(seconds)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def _policy(retries: int | None, timeout: float | None) -> RetryPolicy:
    ctx = _context()
    base = ctx.policy if ctx is not None else BATCH
    return RetryPolicy(retries=base.retries if retries is None else retries,
                       timeout=base.timeout if timeout is None else timeout,
                       backoff_base=base.backoff_base, backoff_cap=base.backoff_cap)


def _server_message(response) -> str:
    """Короткое сообщение сервера: без него типизированный отказ не говорит, ЧТО именно сломалось."""
    # Помощник работает на пути ОТКАЗА и сам падать не должен ни при каком ответе.
    try:
        payload = response.json()
        if isinstance(payload, dict):
            message = payload.get("message") or payload.get("error")
            if message:
                return str(message)[:200]
    except Exception:  # noqa: BLE001 — тело может быть не JSON или объект вовсе без .json()
        pass
    try:
        return (response.text or "")[:200]
    except Exception:  # noqa: BLE001
        return ""


def _cached(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def get(url: str, params: dict | None = None, headers: dict | None = None, *, use_cache: bool = True,
        max_age_days: float | None = None, timeout: float | None = None, retries: int | None = None) -> str:
    """Возвращает тело ответа. Кэширует только успешные ответы.

    Отказ источника поднимается как FetchError с классом ошибки. Ноль в теле ответа это валидное
    наблюдение и FetchError не вызывает: различать их обязан вызывающий слой."""
    pol = _policy(retries, timeout)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_key(url, params)}.json"
    item = _cached(path) if path.exists() else None
    if use_cache and item is not None:
        age = (time.time() - item["ts"]) / 86400
        if max_age_days is None or age <= max_age_days:
            _record({"url": url, "retrieved_at": _iso(item["ts"]), "from_cache": True, "stale": False,
                     "status": "ok", "content_hash": hashlib.sha256(item["body"].encode()).hexdigest()[:32]})
            return item["body"]
    host = urlparse(url).netloc
    last_error, last_type, last_code = None, NETWORK, None
    for attempt in range(pol.retries + 1):
        if too_late_to_start():
            last_error, last_type = "исчерпан общий бюджет запроса", DEADLINE
            break
        _throttle(host)
        left = time_left()
        # Таймаут запроса никогда не превышает осмысленный остаток бюджета. Пола в 0,5 с больше нет:
        # он позволял запросу пережить дедлайн и срывал гарантию по общему времени.
        if left is not None and left < MIN_REQUEST_BUDGET_S:
            last_error, last_type = "исчерпан общий бюджет запроса", DEADLINE
            break
        request_timeout = pol.timeout if left is None else min(pol.timeout, left)
        try:
            r = _session.get(url, params=params, headers=headers, timeout=request_timeout)
        except requests.Timeout as e:
            last_error, last_type = str(e), TIMEOUT
            _sleep(min(pol.backoff_cap, pol.backoff_base * 2 ** attempt))
            continue
        except requests.RequestException as e:
            last_error, last_type = str(e), NETWORK
            _sleep(min(pol.backoff_cap, pol.backoff_base * 2 ** attempt))
            continue
        rate_limited = r.status_code == 403 and "rate limit" in r.text.lower()
        if r.status_code == 429 or rate_limited:
            last_error, last_type, last_code = f"HTTP {r.status_code}", RATE_LIMIT, r.status_code
        elif r.status_code in (500, 502, 503, 504):
            last_error, last_type, last_code = f"HTTP {r.status_code}", SERVER_ERROR, r.status_code
        elif r.status_code in (401, 403):
            # Отдельный класс: просроченный или неверный токен нельзя путать с кривым запросом.
            raise FetchError(f"HTTP {r.status_code} for {url}: {_server_message(r)}", AUTH_ERROR, r.status_code)
        elif r.status_code >= 400:
            raise FetchError(f"HTTP {r.status_code} for {url}: {_server_message(r)}", HTTP_ERROR, r.status_code)
        else:
            ts = time.time()
            path.write_text(json.dumps({"ts": ts, "url": url, "params": params, "body": r.text},
                                       ensure_ascii=False), encoding="utf-8")
            _record({"url": r.url or url, "retrieved_at": _iso(ts), "from_cache": False, "stale": False,
                     "status": "ok", "content_hash": hashlib.sha256(r.text.encode()).hexdigest()[:32]})
            return r.text
        retry_after = r.headers.get("Retry-After")
        wait = float(retry_after) if retry_after and retry_after.isdigit() else min(pol.backoff_cap, 5 * 2 ** attempt)
        _sleep(wait)
    # Источник не ответил. Отдаём просроченный кэш, если он есть: это реально полученное когда-то
    # свидетельство, а не выдуманное. Возраст и признак «из кэша» уходят в карточку.
    if use_cache and item is not None:
        _record({"url": url, "retrieved_at": _iso(item["ts"]), "from_cache": True, "stale": True,
                 "status": "stale_cache", "error_type": last_type,
                 "content_hash": hashlib.sha256(item["body"].encode()).hexdigest()[:32]})
        return item["body"]
    _record({"url": url, "retrieved_at": _iso(time.time()), "from_cache": False, "stale": False,
             "status": "error", "error_type": last_type})
    if last_type == DEADLINE:
        raise DeadlineExceeded(f"{url}: {last_error}")
    raise FetchError(f"{url}: {last_error}", last_type, last_code)


def get_json(url: str, params: dict | None = None, headers: dict | None = None, **kw):
    body = get(url, params, headers, **kw)
    try:
        return json.loads(body)
    except ValueError as e:
        raise FetchError(f"{url}: некорректный JSON ({e})", PARSE) from e


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
