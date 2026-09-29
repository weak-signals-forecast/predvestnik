"""Настоящий шлюз качества конкретного доказательства и трассируемость утверждений (R2-C, R2-D, R2-E).

Что было не так. Шлюз «есть документ уровня A/B» был почти тавтологией: `concrete_documents`
материализовала только научные работы и репозитории, а оба этих типа по построению получают A или B.
Отклонить такой шлюз не мог ничего. Одновременно карточка могла написать «независимые подтверждения:
медиа, наука», хотя показан был только научный документ: семейство бралось из агрегатных счётчиков,
а не из материализованных документов.

Что сделано.

1. Разряд источника перестал быть достаточным: документ должен быть ПЕРВИЧНЫМ для утверждения
   «технология существует» (`trust.supports_claim`). Неопознанное издание, пресс-релиз и сообщество
   не проходят (R2-B, R2-C).
2. Независимость считается по ключу независимости, а не по семейству: перепечатка одного анонса
   в десяти лентах — одно свидетельство (R2-E).
3. Агрегатные семейства и семейства конкретных документов разделены явно:
   `measured_aggregate_families` против `concrete_supporting_families`. Любое утверждение в выдаче
   несёт ссылки на документы, которыми оно подтверждено; семейство, у которого документов нет,
   описывается как «активность по счётчикам», а не как подтверждение (R2-D).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import claim_relevance, independence, trust

# ACCEPT требует минимум столько независимых конкретных подтверждений.
MIN_INDEPENDENT_SUPPORTS = 2
# И минимум столько различных происхождений или семейств среди них: два документа одного издателя
# независимым подтверждением не являются.
MIN_DISTINCT_ORIGINS = 2

# Утверждение, которое обязано опираться на конкретный документ, чтобы кандидат был принят.
PRIMARY_CLAIM = trust.CLAIM_EXISTENCE


@dataclass
class EvidenceClaim:
    """Одно утверждение выдачи и то, чем именно оно подтверждено."""
    claim: str
    families: list[str] = field(default_factory=list)
    documents: list[dict] = field(default_factory=list)   # {title, url, source_name, trust, family}
    aggregate_only: bool = False

    @property
    def materialized(self) -> bool:
        return bool(self.documents)

    def to_dict(self) -> dict:
        return {"claim": self.claim, "families": list(self.families), "documents": list(self.documents),
                "aggregate_only": self.aggregate_only,
                "wording": ("активность по счётчикам источников, конкретные документы не получены"
                            if self.aggregate_only else "подтверждено конкретными документами")}


def _ref(doc) -> dict:
    return {"title": (getattr(doc, "title", "") or "")[:200],
            "url": getattr(doc, "canonical_url", None) or getattr(doc, "url", ""),
            "source_name": getattr(doc, "source_name", ""),
            "trust": getattr(doc, "trust", ""),
            "family": getattr(doc, "source_family", "") or "other",
            "independence_key": independence.independence_key(doc),
            # Происхождение подтверждения: ЧЕМ документ отнесён к этому кандидату (или почему нет).
            "claim_relevant": getattr(doc, "claim_relevant", None),
            "claim_match": getattr(doc, "claim_match", ""),
            "claim_relevance_reason": getattr(doc, "claim_relevance_reason", ""),
            # Происхождение ПРИВЯЗКИ доходит до сериализации: по какой фразе документ попал к
            # кандидату и был ли он перенесён склейкой (правило D).
            "attachment_phrase": getattr(doc, "attachment_phrase", ""),
            "attachment_normalized": getattr(doc, "attachment_normalized", ""),
            "attachment_source": getattr(doc, "attachment_source", "")}


def counts_as_support(doc) -> bool:
    """Может ли документ попасть в подтверждение по признаку релевантности кандидату.

    P1-C. Раньше здесь стояло `is not False`, то есть НЕразмеченный документ (`None`) засчитывался
    как подтверждение, — при том что отпечаток конфигурации объявлял `unknown_counts_as_support:
    false`. Реализация и объявленный контракт противоречили друг другу, и любой путь, не проставивший
    признак, молча открывал подтверждение.

    Теперь читается строго: подтверждает только явное `True`. `False` — правило отработало и связи
    не нашло; `None` — релевантность не устанавливали, и это тоже не подтверждение."""
    return getattr(doc, "claim_relevant", None) is True


@dataclass
class EvidenceQuality:
    """Результат разбора конкретных документов кандидата."""
    documents: list = field(default_factory=list)                # все материализованные
    independent: list = field(default_factory=list)              # по одному на ключ независимости
    supporting: list = field(default_factory=list)               # первичные для PRIMARY_CLAIM
    unrated: list = field(default_factory=list)                  # разряд «неизвестно»
    irrelevant: list = field(default_factory=list)               # получены, но не про этого кандидата
    concrete_supporting_families: list[str] = field(default_factory=list)
    measured_aggregate_families: list[str] = field(default_factory=list)
    aggregate_only_families: list[str] = field(default_factory=list)
    independent_supports: int = 0
    distinct_origins: int = 0
    same_work_merges: int = 0                                    # версии одной научной работы (P1-A)
    syndication_rate: float = 0.0
    origin_concentration: float = 0.0
    claims: list[EvidenceClaim] = field(default_factory=list)

    @property
    def has_supporting_document(self) -> bool:
        return bool(self.supporting)

    @property
    def independently_corroborated(self) -> bool:
        return (self.independent_supports >= MIN_INDEPENDENT_SUPPORTS
                and max(len(self.concrete_supporting_families), self.distinct_origins) >= MIN_DISTINCT_ORIGINS)

    def to_dict(self) -> dict:
        return {
            "concrete_supporting_families": list(self.concrete_supporting_families),
            "measured_aggregate_families": list(self.measured_aggregate_families),
            "aggregate_only_families": list(self.aggregate_only_families),
            "supporting_document_count": len(self.supporting),
            "independent_supports": self.independent_supports,
            "distinct_origins": self.distinct_origins,
            "same_work_merges": self.same_work_merges,
            "unrated_document_count": len(self.unrated),
            # Документы получены и видны, но к этому кандидату не относятся: подтверждением не служат.
            "claim_irrelevant_document_count": len(self.irrelevant),
            "claim_irrelevant_documents": [_ref(d) for d in self.irrelevant[:5]],
            "materialized_document_count": len(self.documents),
            "syndication_rate": self.syndication_rate,
            "origin_concentration": self.origin_concentration,
            "claims": [c.to_dict() for c in self.claims],
        }


def assess_documents(documents, aggregate_families=None, candidate=None) -> EvidenceQuality:
    """Разбирает конкретные документы кандидата и строит трассируемые утверждения.

    `candidate` — личность кандидата ПОСЛЕ склейки. Когда она передана, релевантность каждого
    документа пересчитывается здесь заново: документ, пришедший с поглощённым широким вариантом,
    не наследует подтверждение, а проверяется против выжившего канонического названия."""
    docs = list(documents or [])
    if candidate is not None:
        claim_relevance.annotate(docs, candidate)
    aggregate = sorted(set(aggregate_families or []))
    independent = independence.independent_documents(docs)
    # Три независимых условия подтверждения: разряд доверия, применимость семейства к утверждению
    # и отношение документа К ЭТОМУ КАНДИДАТУ. Последнее добавлено ремонтом claim-relative.
    supporting = [d for d in independent
                  if trust.supports_claim(d, PRIMARY_CLAIM) and counts_as_support(d)]
    irrelevant = [d for d in docs if getattr(d, "claim_relevant", None) is not True]
    unrated = [d for d in docs if getattr(d, "trust", "") == trust.TIER_UNKNOWN]

    support_families = sorted({(getattr(d, "source_family", "") or "other") for d in supporting})
    origins = {independence.origin(d) for d in supporting}
    q = EvidenceQuality(
        documents=docs, independent=independent, supporting=supporting, unrated=unrated,
        irrelevant=irrelevant,
        concrete_supporting_families=support_families,
        measured_aggregate_families=aggregate,
        aggregate_only_families=[f for f in aggregate if f not in set(support_families)],
        independent_supports=len({independence.independence_key(d) for d in supporting}),
        distinct_origins=len(origins),
        syndication_rate=independence.syndication_rate(docs),
        origin_concentration=independence.origin_concentration(docs),
        same_work_merges=independence.same_work_merges(docs),
    )

    # Утверждения строятся только из материализованных документов; семейства без документов
    # выносятся в отдельное агрегатное утверждение с честной формулировкой (R2-D).
    # В утверждение попадает только то, что этого кандидата действительно касается: иначе карточка
    # снова писала бы «подтверждено наукой», показывая работу про другую технологию.
    by_claim: dict[str, list] = {}
    for d in independent:
        if not counts_as_support(d):
            continue
        for claim in trust.claims_supported(d):
            by_claim.setdefault(claim, []).append(d)
    for claim, items in by_claim.items():
        q.claims.append(EvidenceClaim(
            claim=claim,
            families=sorted({(getattr(d, "source_family", "") or "other") for d in items}),
            documents=[_ref(d) for d in items[:5]]))
    if q.aggregate_only_families:
        q.claims.append(EvidenceClaim(claim="активность в источниках",
                                      families=list(q.aggregate_only_families),
                                      documents=[], aggregate_only=True))
    return q


def gate(quality: EvidenceQuality) -> tuple[bool, str | None, str | None]:
    """Шлюз качества конкретного доказательства. Возвращает (пройден, код причины, текст причины)."""
    if not quality.documents:
        return False, "supporting_evidence_not_materialized", (
            "агрегатные счётчики обнадёживают, но ни одного конкретного документа получить не удалось")
    if not quality.supporting:
        # Отдельная, точная причина: документы приемлемого качества есть, но ни один из них не про
        # эту технологию. Смешивать её с «качество источника не подошло» нельзя — это разные отказы
        # и разные действия владельца.
        if quality.irrelevant and any(trust.supports_claim(d, PRIMARY_CLAIM) for d in quality.irrelevant):
            return False, "no_claim_relevant_evidence", (
                f"получено документов: {len(quality.documents)}, но ни один из них не назван про "
                "технологию кандидата: совпадение только по общим словам запроса. "
                "Упоминание темы подтверждением технологии не является")
        detail = []
        if quality.unrated:
            detail.append(f"неопознанных изданий: {len(quality.unrated)}")
        families = sorted({(getattr(d, "source_family", "") or "other") for d in quality.documents})
        detail.append("семейства полученных документов: " + ", ".join(families))
        return False, "no_acceptable_source_quality", (
            "ни один полученный документ не является первичным доказательством существования "
            "технологии приемлемого качества (" + "; ".join(detail) + ")")
    if not quality.independently_corroborated:
        return False, "no_independent_corroboration", (
            f"независимых подтверждений {quality.independent_supports} из необходимых "
            f"{MIN_INDEPENDENT_SUPPORTS}, различных происхождений {quality.distinct_origins}: "
            "свидетельства не независимы")
    return True, None, None
