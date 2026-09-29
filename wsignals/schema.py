"""Единая схема документа из любого источника. Поля покрывают требования ТЗ к отображению источника
и минимальный контракт происхождения (PHASE 7).

Добавлено ремонтом: canonical_url, retrieved_at, is_primary, source_family, content_hash, adapter_status.
Реестра, печатей, промоушена и объектного хранилища здесь нет и не предполагается.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Семейство источника: по нему считается независимость подтверждений.
SOURCE_TYPE_FAMILY = {
    "научная публикация": "science", "препринт": "science", "патент": "patent", "реестр": "registry",
    "репозиторий": "code", "новости": "media", "пресс-релиз": "press", "сообщество": "community",
    "энциклопедия": "encyclopedia", "блог": "community", "соцсеть": "community", "агрегатор": "aggregator",
}
# Первичный источник: автор сам сообщает о работе. Агрегатор и лента таковым не являются.
PRIMARY_TYPES = {"научная публикация", "препринт", "патент", "реестр", "репозиторий"}
SECONDARY_TYPES = {"новости", "агрегатор", "энциклопедия"}
# Хосты-редиректы: ссылка не является канонической.
REDIRECT_HOSTS = {"news.google.com", "news.url.google.com"}
TRACKING = re.compile(r"^(utm_|fbclid|gclid|yclid|ref|ref_src|oc$)")


def canonicalize_url(url: str) -> tuple[str | None, bool]:
    """Канонический URL и признак «ссылка ведёт напрямую». Редирект агрегатора каноническим не считается."""
    if not url:
        return None, False
    if url.lower().startswith("10.") or url.lower().startswith("doi:"):
        return "https://doi.org/" + url.split(":", 1)[-1].lstrip("/"), True
    try:
        parts = urlsplit(url)
    except ValueError:
        return None, False
    if not parts.scheme or not parts.netloc:
        return None, False
    host = parts.netloc.lower().removeprefix("www.")
    if host in REDIRECT_HOSTS:
        return None, False
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                       if not TRACKING.match(k.lower())])
    return urlunsplit((parts.scheme.lower(), host, parts.path.rstrip("/") or "/", query, "")), True


@dataclass
class Document:
    title: str
    url: str
    source_name: str            # издание, база или площадка: «OpenAlex / Nature», «Хабр», «GitHub»
    source_type: str            # научная публикация, препринт, новости, пресс-релиз, репозиторий, сообщество, энциклопедия
    published: str | None       # ISO-дата публикации, если известна
    language: str               # ru, en, ...
    trust: str = ""             # высокий, средний, пониженный; проставляет trust.assess
    trust_reason: str = ""
    snippet: str = ""
    extra: dict = field(default_factory=dict)
    # ---- происхождение (PHASE 7) ----
    canonical_url: str | None = None      # прямая ссылка без редиректов и меток трекинга
    retrieved_at: str | None = None       # когда документ реально получен, ISO-8601 UTC
    source_family: str = ""               # science | code | media | press | community | encyclopedia | ...
    is_primary: bool | None = None        # первичный источник, если это выводимо из типа
    content_hash: str | None = None       # sha256 заголовка и фрагмента, для сверки дублей
    adapter_status: str = "OK"            # OK | NO_RESULTS | ERROR | STALE_CACHE
    # ---- претензионно-относительная релевантность (см. claim_relevance) ----
    # Документ назван про ту же технологию, что и кандидат. None означает «не проверялось» и
    # подтверждением НЕ считается: неизвестная релевантность это не положительная релевантность.
    claim_relevant: bool | None = None
    claim_match: str = ""                 # какой вариант названия кандидата совпал
    # P1-B. Происхождение ПРИВЯЗКИ: по какой именно фразе документ попал к кандидату. До склейки это
    # сам кандидат, после склейки — поглощённый вариант. Без этой записи нельзя отличить документ,
    # пришедший по выжившему названию, от пришедшего по более широкому псевдониму.
    attachment_phrase: str = ""
    attachment_normalized: str = ""
    attachment_source: str = ""           # own | merged
    claim_relevance_reason: str = ""      # человекочитаемое объяснение решения

    @property
    def published_at(self) -> str | None:
        """Синоним `published` в терминах контракта происхождения."""
        return self.published

    def with_provenance(self, retrieved_at: str | None = None, adapter_status: str | None = None) -> "Document":
        """Дозаполняет поля происхождения тем, что выводимо из самого документа."""
        canon, direct = canonicalize_url(self.url)
        self.canonical_url = canon
        self.source_family = self.source_family or SOURCE_TYPE_FAMILY.get(self.source_type, "other")
        if self.is_primary is None:
            self.is_primary = True if self.source_type in PRIMARY_TYPES and direct else (
                False if self.source_type in SECONDARY_TYPES or not direct else None)
        if retrieved_at:
            self.retrieved_at = retrieved_at
        if adapter_status:
            self.adapter_status = adapter_status
        if not self.content_hash:
            raw = f"{self.title}\n{self.canonical_url or self.url}\n{self.snippet[:500]}"
            self.content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["published_at"] = self.published
        return d
