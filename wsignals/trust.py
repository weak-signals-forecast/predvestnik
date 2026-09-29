"""Уровень доверия источнику и claim-относительное доверие (R2-B).

ТЗ: доверенные источники это госорганы, регуляторы, международные организации, университеты, научные центры,
компании-разработчики, отраслевые ассоциации, научные публикации, патентные базы, реестры, отраслевые медиа.
Соцсети, блоги, агрегаторы, пресс-релизы: только первичный индикатор с пониженным доверием.

R2-B. Раньше неизвестный источник молча получал «средний», то есть проходил как полноценное
подтверждение уровня A/B. Это была дыра, открытая наружу: любая лента без отраслевой специализации
годилась в качестве конкретного доказательства. Теперь неклассифицированный источник получает явный
разряд «неизвестно», и этот разряд подтверждением **не является** (fail-closed).

Доверие claim-относительное. Репозиторий на GitHub — первичное доказательство того, что технология
существует и реализована в коде, но не доказательство внедрения на рынке. Лента новостей может
подтвердить, что об announce говорили, но не что технология работает. Онтологии здесь нет: это
маленькая таблица «семейство источника → какие утверждения оно может подтверждать первично».
"""
from __future__ import annotations

import re

from .schema import Document

HIGH_DOMAINS = re.compile(
    r"(\.gov(\.\w+)?$|\.edu$|\.ac\.\w+$|europa\.eu$|oecd\.org$|bis\.org$|imf\.org$|worldbank\.org$|nist\.gov$|"
    r"iso\.org$|ietf\.org$|w3\.org$|cbr\.ru$|gov\.ru$|nature\.com$|science\.org$|ieee\.org$|acm\.org$|arxiv\.org$|"
    r"springer\.com$|sciencedirect\.com$|wipo\.int$|fips\.ru$|patents\.google\.com$)")
MEDIA_DOMAINS = re.compile(
    r"(reuters\.com|bloomberg\.com|ft\.com|wsj\.com|techcrunch\.com|theverge\.com|wired\.com|arstechnica\.com|"
    r"siliconangle\.com|venturebeat\.com|eetimes\.com|spectrum\.ieee\.org|datacenterdynamics\.com|"
    r"rbc\.ru|kommersant\.ru|vedomosti\.ru|cnews\.ru|tadviser\.ru|habr\.com|securitylab\.ru|comnews\.ru|forbes\.ru)")
PRESS_RELEASE = re.compile(r"(prnewswire|businesswire|globenewswire|einpresswire|accesswire|newswire|пресс-релиз)", re.I)
LOW_TYPES = {"сообщество", "блог", "соцсеть", "агрегатор"}

# Разряды доверия. UNKNOWN введён R2-B и намеренно НЕ входит в SUPPORTING_TIERS.
TIER_A, TIER_B, TIER_C, TIER_UNKNOWN = "высокий", "средний", "пониженный", "неизвестно"
SUPPORTING_TIERS = frozenset({TIER_A, TIER_B})
TIER_ORDER = {TIER_UNKNOWN: 0, TIER_C: 1, TIER_B: 2, TIER_A: 3}

# Утверждения, которые система делает о технологии. Список короткий и закрытый.
CLAIM_EXISTENCE = "существование"          # технология названа и описана
CLAIM_RESEARCH = "исследования"            # есть исследовательская база
CLAIM_IMPLEMENTATION = "реализация"        # есть работающий код или артефакт
CLAIM_ADOPTION = "внедрение"               # применяется на практике, есть рынок
CLAIM_REGULATION = "регулирование"         # признана регулятором или стандартом
CLAIMS = (CLAIM_EXISTENCE, CLAIM_RESEARCH, CLAIM_IMPLEMENTATION, CLAIM_ADOPTION, CLAIM_REGULATION)

# Какое семейство источника может быть ПЕРВИЧНЫМ доказательством какого утверждения.
# Пустое множество означает «только вторичный индикатор, самостоятельным доказательством не является».
PRIMARY_FOR: dict[str, frozenset[str]] = {
    "science": frozenset({CLAIM_EXISTENCE, CLAIM_RESEARCH}),
    "patent": frozenset({CLAIM_EXISTENCE, CLAIM_RESEARCH}),
    "registry": frozenset({CLAIM_EXISTENCE, CLAIM_REGULATION}),
    "code": frozenset({CLAIM_EXISTENCE, CLAIM_IMPLEMENTATION}),
    "media": frozenset({CLAIM_EXISTENCE}),
    "encyclopedia": frozenset({CLAIM_EXISTENCE}),
    "press": frozenset(),
    "community": frozenset(),
    "aggregator": frozenset(),
    "other": frozenset(),
}


def assess(doc: Document) -> Document:
    host = re.sub(r"^https?://(www\.)?", "", doc.url).split("/")[0].lower()
    t = doc.source_type
    if PRESS_RELEASE.search(host) or PRESS_RELEASE.search(doc.source_name) or t == "пресс-релиз":
        doc.trust, doc.trust_reason = TIER_C, "пресс-релиз: первичный индикатор, требует независимого подтверждения"
    elif t in {"научная публикация", "патент", "реестр"} or HIGH_DOMAINS.search(host):
        doc.trust, doc.trust_reason = TIER_A, "научная публикация, патентная база, регулятор или университет"
    elif t == "препринт":
        doc.trust, doc.trust_reason = TIER_B, "препринт: без рецензирования, но с проверяемыми авторами"
    elif MEDIA_DOMAINS.search(host) or MEDIA_DOMAINS.search(doc.source_name.lower()):
        doc.trust, doc.trust_reason = TIER_B, "профессиональное отраслевое или деловое СМИ"
    elif t == "репозиторий":
        doc.trust, doc.trust_reason = TIER_B, "код разработчика: подтверждает практическую реализацию"
    elif t in LOW_TYPES:
        doc.trust, doc.trust_reason = TIER_C, "сообщество или блог: первичный индикатор"
    else:
        # R2-B: fail-closed. Неизвестное издание больше не выдаётся за отраслевое СМИ.
        doc.trust, doc.trust_reason = TIER_UNKNOWN, ("издание не опознано: качество источника неизвестно, "
                                                     "подтверждением не считается")
    return doc


def is_supporting(doc) -> bool:
    """Годится ли документ в качестве конкретного подтверждения (разряд A или B)."""
    return getattr(doc, "trust", "") in SUPPORTING_TIERS


def supports_claim(doc, claim: str) -> bool:
    """Может ли документ быть ПЕРВИЧНЫМ доказательством этого утверждения.

    Разряд отвечает за качество источника, таблица `PRIMARY_FOR` — за то, о чём этот источник вправе
    свидетельствовать. GitHub первичен для «реализации», но не для «внедрения»."""
    if not is_supporting(doc):
        return False
    family = getattr(doc, "source_family", "") or "other"
    return claim in PRIMARY_FOR.get(family, frozenset())


def claims_supported(doc) -> list[str]:
    return [c for c in CLAIMS if supports_claim(doc, c)]
