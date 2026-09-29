"""Итоговое решение по кандидату: четыре разделённые оси вместо одной вероятности (PHASE 2–3).

Аудит: единственным основанием для включения была вероятность логистической регрессии, обученной так,
что «нет следа» читается как «слабый сигнал». Здесь решение разложено на четыре измерения, каждое
из которых считается по измеренным величинам и по явному правилу:

  EMERGENCE_SCORE     насколько технология именно сейчас появляется (рост, свежий код, свежие работы);
  MATURITY_RISK       насколько она на самом деле уже зрелая;
  EVIDENCE_CONFIDENCE насколько вообще есть на что опираться (независимые семейства, доверие, полнота опроса);
  HYPE_RISK           насколько след объясняется маркетингом, а не содержанием.

Вероятность модели этапа 1 остаётся, но как `legacy_model_probability`: предиктор и элемент объяснения,
не право вето и не право включения. Модель в этой задаче не переобучается.

Правила намеренно простые и монотонные: пороги названы константами, каждое сработавшее правило пишет
причину в `decision_reasons` и машиночитаемый код в `decision_reason_codes`. Под текущие метки они
не подбирались.

REPAIR 1. ACCEPT требует трёх вещей одновременно:
  1) достаточности доказательства на агрегатном уровне (независимые семейства, качество, полнота опроса);
  2) хотя бы одного конкретного подтверждающего документа уровня доверия A или B;
  3) хотя бы одного независимого подтверждения помимо него (второе семейство источников).
Если агрегатные счётчики обнадёживают, но конкретные документы до дедлайна получить не удалось —
это `LOW_EVIDENCE` с кодом `supporting_evidence_not_materialized`, а не принятый сигнал.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from . import evidence_quality as eq
from . import independence, trust
from .evidence import EvidenceState

# ---------- пороги решения (простые, прозрачные, не подобранные под разметку) ----------

MIN_CORROBORATING_FAMILIES = 2      # ТЗ: независимое подтверждение, а не одна лента
MIN_EVIDENCE_CONFIDENCE = 0.45
MATURITY_VETO = 0.60
HYPE_VETO = 0.60
MIN_EMERGENCE = 0.15
# Отказ источника не даёт ни нуля, ни единицы: зрелость просто неизвестна.
UNKNOWN_MATURITY_RISK = 0.50

LOW_EVIDENCE, MATURE, HYPE, NOISE, ACCEPT = "LOW_EVIDENCE", "MATURE", "HYPE", "NOISE", "ACCEPT"

# Уровни доверия по ТЗ. A и B годятся как подтверждающий документ, C — только первичный индикатор,
# «неизвестно» (R2-B) не годится вовсе.
TRUST_A, TRUST_B, TRUST_C = trust.TIER_A, trust.TIER_B, trust.TIER_C
TRUST_UNKNOWN = trust.TIER_UNKNOWN
SUPPORTING_TRUST = trust.SUPPORTING_TIERS
MIN_SUPPORTING_DOCUMENTS = 1
MIN_INDEPENDENT_FAMILIES = 2

# R2-A. Состояние семантической проверки кандидата.
SEMANTICALLY_VERIFIED = "SEMANTICALLY_VERIFIED"      # нормализация прошла и подтвердила технологию
DETERMINISTIC_ONLY = "DETERMINISTIC_ONLY"            # нормализации не было, лексика чистая
SEMANTIC_UNCERTAIN = "SEMANTIC_UNCERTAIN"            # нормализации не было и лексика неоднозначна
VERIFICATION_STATES = (SEMANTICALLY_VERIFIED, DETERMINISTIC_ONLY, SEMANTIC_UNCERTAIN)


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _saturate(x: float, scale: float) -> float:
    """Монотонный перевод счётчика в [0, 1]: log1p(x)/log1p(scale), выше scale насыщается."""
    return _clip(math.log1p(max(0.0, x)) / math.log1p(scale))


# ---------- EMERGENCE ----------

# Вес каждой компоненты фиксирован. Знаменатель всегда полный: отказавший источник даёт в числитель
# ноль и не выбывает из знаменателя, поэтому отказ может только понизить оценку, но не повысить её.
EMERGENCE_WEIGHTS = {"burst": 0.30, "science_recency": 0.15, "code_novelty": 0.25,
                     "community_news": 0.15, "preprint_share": 0.15}


def emergence(state: EvidenceState, burst: float = 0.0) -> tuple[float, dict]:
    v = state.values()
    parts = {k: 0.0 for k in EMERGENCE_WEIGHTS}
    if state.ok("openalex"):
        parts["burst"] = _clip(0.5 * _clip(max(burst, 0.0) / 6.0) + 0.5 * _clip(max(v.get("oa_growth", 0.0), 0.0) / 2.5))
        parts["science_recency"] = _clip(0.6 * _clip(v.get("oa_last_share", 0.0) / 0.5)
                                         + 0.4 * _clip(1.0 - v.get("oa_age", 0) / 10.0))
        parts["preprint_share"] = _clip(v.get("oa_preprint_share", 0.0) / 0.4)
    if state.ok("github"):
        parts["code_novelty"] = _clip(0.6 * _clip(v.get("gh_new_share", 0.0) / 0.6)
                                      + 0.4 * _saturate(v.get("gh_12m", 0), 200))
    community = _clip(max(v.get("hn_growth", 0.0), 0.0) / 2.0) if state.ok("hn") else 0.0
    fresh_news = _clip(v.get("news_30d_share", 0.0) / 0.25) if state.ok("news") else 0.0
    parts["community_news"] = max(community, fresh_news)
    score = sum(EMERGENCE_WEIGHTS[k] * parts[k] for k in EMERGENCE_WEIGHTS)
    return round(_clip(score), 3), {k: round(x, 3) for k, x in parts.items()}


# ---------- MATURITY (v2) ----------

# Прежняя модель была неработоспособна по построению: непрерывная ветка упиралась в 0.50, а средний
# разряд Википедии давал 0.55 — оба ниже порога вето 0.60. Вето могли выдать только два дискретных
# условия, всё остальное не влияло ни на что.
#
# Формула v2. Зрелость это взвешенное среднее измеримых признаков, каждый в [0, 1]:
#
#     volume        = sat(oa_total, 3000)                       объём научного следа
#     age           = clip(oa_age / 10)                          возраст литературы
#     breadth       = clip(oa_fields / 6)                        широта по научным областям
#     persistence   = sat(oa_prior, 300)                          активность до текущей волны
#     encyclopedia  = 0, либо 0.3 + 0.7 * clip(wiki_age / 8)      описана ли как общее знание
#     adoption      = sat(gh_total, 5000)                         масштаб реализации в коде
#
#     maturity_risk = max( Σ w_i · c_i / Σ w_i  по ДОСТУПНЫМ i ,  дискретные признаки )
#
# Свойства, которые требовались и которые проверяются тестами:
#   1. непрерывная ветка способна пересечь 0.60: при volume=age=breadth=persistence=1 и отсутствующей
#      статье она даёт 0.80 (см. таблицу отклика в tests/test_maturity_surface.py);
#   2. несколько умеренных признаков складываются в вето: 800 работ за 6 лет по 4 областям со статьёй
#      трёхлетней давности дают ≈0.71;
#   3. недоступный источник исключается из ЗНАМЕНАТЕЛЯ, а не считается нулём, поэтому отсутствие данных
#      не может снизить риск; сверх того при недоступной Википедии действует floor UNKNOWN_MATURITY_RISK.
#      Обратная защита: одна лишь неизвестность не даёт вето. Если по ПОЛНОМУ знаменателю (то есть считая
#      недостающие признаки нулём — нижняя граница) риск ниже порога, итог зажимается чуть ниже порога.
#      Монотонность сохраняется: обе величины монотонны, min(x, const) монотонна, а при пересечении
#      порога нижней границей зажим снимается и риск только растёт;
#   4. монотонность: все c_i не убывают по своим входам, веса положительны, знаменатель при фиксированном
#      наборе доступных источников постоянен, а max с дискретными признаками монотонность сохраняет;
#   5. большой и широкий след даёт высокий риск даже при небольшом возрасте: 5000 работ по 6 областям
#      в возрасте 4 лет дают ≈0.68;
#   6. темпа роста в формуле нет вообще — быстрый рост не может «стереть» зрелость.
#
# Пороги под примеры аудита не подбирались: веса заданы по смыслу признаков, порог вето не менялся.

MATURITY_WEIGHTS = {"volume": 0.28, "age": 0.20, "breadth": 0.16, "persistence": 0.16,
                    "encyclopedia": 0.20, "adoption": 0.10}
MATURITY_SCALES = {"volume": 3000.0, "age": 10.0, "breadth": 6.0, "persistence": 300.0,
                   "encyclopedia": 8.0, "adoption": 5000.0}
# Сама по себе статья в энциклопедии уже означает, что технология описана как общее знание.
ENCYCLOPEDIA_FLOOR = 0.30


def maturity_components(state: EvidenceState) -> dict[str, float | None]:
    """Компоненты зрелости. `None` означает «источник не ответил»: такой признак исключается из
    знаменателя, а не превращается в ноль."""
    v = state.values()
    science = state.ok("openalex")
    out: dict[str, float | None] = {
        "volume": _saturate(v.get("oa_total", 0), MATURITY_SCALES["volume"]) if science else None,
        "age": _clip(v.get("oa_age", 0) / MATURITY_SCALES["age"]) if science else None,
        "breadth": _clip(v.get("oa_fields", 0) / MATURITY_SCALES["breadth"]) if science else None,
        "persistence": _saturate(v.get("oa_prior", 0), MATURITY_SCALES["persistence"]) if science else None,
        "encyclopedia": None,
        "adoption": None,
    }
    if state.ok("wiki"):
        if v.get("wiki_article"):
            out["encyclopedia"] = _clip(ENCYCLOPEDIA_FLOOR + (1 - ENCYCLOPEDIA_FLOOR)
                                        * _clip(v.get("wiki_age", 0) / MATURITY_SCALES["encyclopedia"]))
        else:
            out["encyclopedia"] = 0.0
    # gh_total приходит только когда на живом пути включены счётчики поиска GitHub. Репозитории корпуса
    # для зрелости не годятся: это репозитории за последние 12 месяцев, признак появления, а не зрелости.
    if "gh_total" in v:
        out["adoption"] = _saturate(v.get("gh_total", 0), MATURITY_SCALES["adoption"])
    return out


def maturity(state: EvidenceState) -> tuple[float, list[str], bool]:
    v = state.values()
    reasons: list[str] = []
    unknown = not state.ok("wiki")
    components = maturity_components(state)

    available = {k: c for k, c in components.items() if c is not None}
    weight = sum(MATURITY_WEIGHTS[k] for k in available)
    numerator = sum(MATURITY_WEIGHTS[k] * c for k, c in available.items())
    risk = numerator / weight if weight else 0.0
    # Нижняя граница: недостающие признаки считаются нулём. Если даже она ниже порога, вето выдаёт
    # не измерение, а неизвестность — такого вето быть не должно.
    grounded = numerator / sum(MATURITY_WEIGHTS.values())
    if len(available) < len(components) and grounded < MATURITY_VETO:
        risk = min(risk, MATURITY_VETO - 0.01)
    if available:
        top = sorted(available.items(), key=lambda kv: -MATURITY_WEIGHTS[kv[0]] * kv[1])[:3]
        strong = [k for k, c in top if c >= 0.5]
        if risk >= 0.45 and strong:
            reasons.append("накопленные признаки зрелости: " + ", ".join(
                f"{MATURITY_NAMES_RU[k]} {available[k]:.2f}" for k in strong))

    # Дискретные признаки, каждый из которых сам по себе означает зрелость.
    if state.ok("wiki") and v.get("wiki_article") and v.get("wiki_age", 0) >= 5:
        risk = max(risk, 1.0)
        reasons.append(f"зрелая технология: отдельная статья в Википедии существует {int(v['wiki_age'])} лет")
    if state.ok("openalex") and v.get("oa_total", 0) >= 3000 and v.get("oa_age", 0) >= 8:
        risk = max(risk, 1.0)
        reasons.append(f"сформированное научное направление: {int(v['oa_total'])} публикаций, "
                       f"первые работы {int(v['oa_age'])} лет назад")

    if unknown:
        # Нельзя утверждать, что статьи нет: источник не ответил. Ставим «неизвестно», а не ноль.
        risk = max(risk, UNKNOWN_MATURITY_RISK)
        reasons.append("зрелость не проверена: Википедия не ответила")
    return round(_clip(risk), 3), reasons, unknown


MATURITY_NAMES_RU = {"volume": "объём литературы", "age": "возраст литературы",
                     "breadth": "широта научных областей", "persistence": "активность до текущей волны",
                     "encyclopedia": "описана в энциклопедии", "adoption": "масштаб реализации в коде"}


# ---------- HYPE ----------

# R2-F. Дешёвые признаки синдикации и промо. NLP-модель хайпа не обучается.
SYNDICATION_VETO = 0.50            # половина полученных текстов — перепечатки одного анонса
CONCENTRATION_MIN_DOCS = 3         # концентрация считается только когда документов достаточно
PR_TO_PRIMARY_IMBALANCE = 2.0      # промо-документов вдвое больше, чем первичных


# P1-B. Достаточность доказательства хайпа отделена от величины риска хайпа.
#
# Измеренный дефект: все ветки `hype` включались только при `state.ok(источник)`, поэтому ОТКАЗ
# провайдера давал `hype_risk = 0.0` — то есть выглядел как измеренное отсутствие хайпа. Один и тот
# же кандидат с новостями OK и 90 % пресс-релизов получал HYPE, а с новостями ERROR — ACCEPT. Отказ
# источника делал принятие ЛЕГЧЕ, что прямо противоречит контракту «отказ источника это не ноль».
#
# Новый контракт: ERROR означает «доказательство хайпа недоступно», а не «хайпа нет». Величина риска
# при этом не выдумывается: ни 1.0, ни автоматического HYPE здесь нет.
#
# Новостной канал — основной измеритель маркетингового шума. Если он не ответил, хайп можно считать
# измеренным только когда есть достаточная альтернатива: полученные конкретные документы, по которым
# работают проверки перепечаток, промо-перекоса и концентрации происхождения.
HYPE_PRIMARY_SOURCE = "news"
# Семейства документов, которые САМИ несут промо-сигнал: только на них и работают проверки
# перекоса в промо и перепечаток. Научная статья и репозиторий про маркетинговый шум не говорят
# ничего — ни за, ни против.
HYPE_SIGNAL_FAMILIES = frozenset({"press", "aggregator"})


def hype_evidence_available(state: EvidenceState, quality=None) -> bool:
    """Есть ли чем измерить хайп — ПО ИЗМЕРЕНИЯМ, а не по числу документов вообще.

    Прошлая версия считала достаточным любые три конкретных документа. Независимый аудит показал
    цену этой замены: три научных и кодовых документа «закрывали» медийное измерение, и один и тот
    же кандидат с новостями OK и 90 % пресс-релизов получал HYPE, а с новостями ERROR — ACCEPT.
    Отказ источника снова делал принятие легче.

    Достаточность теперь привязана к конкретным веткам расчёта хайпа:

      * новостной канал ответил (OK или NO_RESULTS) — медийное измерение выполнено;
      * есть документы промо-семейств или с неопознанным доверием — работают проверки перекоса
        в промо и концентрации происхождения;
      * наблюдается ненулевая перепечатка — работает проверка синдикации.

    Одного лишь КОЛИЧЕСТВА научных и кодовых документов для медийного измерения недостаточно
    ни при каком N."""
    if state.ok(HYPE_PRIMARY_SOURCE):
        return True
    # Заменителем служит только ФАКТИЧЕСКИ СРАБОТАВШИЙ документный признак хайпа. Просто наличие
    # одного пресс-релиза среди первичных работ медийного насыщения не измеряет: отсутствие сигнала
    # в ненемедийных документах не является доказательством отсутствия хайпа.
    return document_hype_signal(quality)[0] > 0.0


def document_hype_signal(quality=None) -> tuple[float, list[str]]:
    """Риск хайпа, который видно ПО САМИМ ПОЛУЧЕННЫМ ДОКУМЕНТАМ, без новостных счётчиков.

    Одна реализация на два применения: величина риска в `hype` и проверка того, есть ли вообще чем
    мерить медийное измерение, когда новостной канал отказал."""
    risk, reasons = 0.0, []
    if quality is None or not getattr(quality, "documents", None):
        return risk, reasons
    if quality.syndication_rate >= SYNDICATION_VETO:
        risk = max(risk, 0.8)
        reasons.append(f"перепечатки: {quality.syndication_rate:.0%} полученных текстов повторяют друг друга")
    promo = [d for d in quality.documents
             if (getattr(d, "source_family", "") in HYPE_SIGNAL_FAMILIES)
             or getattr(d, "trust", "") == TRUST_UNKNOWN]
    primary = quality.supporting
    # Перекос считается только когда есть знаменатель. Полное отсутствие первичных документов —
    # это недостаток доказательства, а не доказательство хайпа: такой случай разбирает шлюз
    # качества с точной причиной, и подменять его хайпом нельзя (оси остаются раздельными).
    if primary and len(promo) >= PR_TO_PRIMARY_IMBALANCE * len(primary):
        risk = max(risk, 0.7)
        reasons.append(f"перекос в промо: {len(promo)} промо-документов против {len(primary)} первичных")
    if (len(quality.documents) >= CONCENTRATION_MIN_DOCS and quality.origin_concentration >= 0.99
            and "science" not in quality.concrete_supporting_families):
        risk = max(risk, 0.7)
        reasons.append("все свидетельства из одного источника: концентрация на одной компании или площадке")
    return risk, reasons


def hype(state: EvidenceState, quality=None, burst: float = 0.0) -> tuple[float, list[str]]:
    v = state.values()
    reasons: list[str] = []
    risk = 0.0
    if state.ok("news"):
        press = v.get("news_press_share", 0.0)
        if press >= 0.5:
            risk = max(risk, 0.85)
            reasons.append(f"маркетинговый шум: пресс-релизы составляют {press:.0%} новостей")
        else:
            risk = max(risk, _clip(press / 0.5) * 0.5)
        science_thin = state.ok("openalex") and v.get("oa_total", 0) < 10
        code_thin = state.ok("github") and v.get("gh_total", 0) < 5
        if v.get("news_1y", 0) >= 30 and science_thin and code_thin:
            risk = max(risk, 0.75)
            reasons.append("внимание медиа без научного и инженерного следа")
    if state.ok("openalex"):
        if (v.get("oa_peak_ratio", 1.0) < 0.5 and (v.get("oa_peak_year") or 9999) <= 2023
                and v.get("oa_age", 0) >= 5 and v.get("oa_total", 0) >= 50):
            risk = max(risk, 0.8)
            reasons.append("угасшая волна: публикаций в прошлом году меньше половины от пикового года")
        # Всплеск без предыстории: резкий рост при отсутствии активности до волны и крошечном объёме.
        if burst >= 3.0 and v.get("oa_prior", 0) == 0 and v.get("oa_total", 0) < 15:
            risk = max(risk, 0.65)
            reasons.append("всплеск без предыстории: до текущей волны публикаций не было, а объём следа мал")

    # Признаки по конкретным документам (R2-F).
    doc_risk, doc_reasons = document_hype_signal(quality)
    risk = max(risk, doc_risk)
    reasons += doc_reasons
    return round(_clip(risk), 3), reasons


# ---------- EVIDENCE CONFIDENCE ----------

FAMILY_CONFIDENCE = {0: 0.0, 1: 0.35, 2: 0.65, 3: 0.85}
HIGH_TRUST_BONUS = 0.10
UNKNOWN_MATURITY_PENALTY = 0.15


def evidence_confidence(state: EvidenceState, maturity_unknown: bool) -> tuple[float, list[str]]:
    reasons: list[str] = []
    fam = state.corroborating_families()
    base = FAMILY_CONFIDENCE.get(len(fam), 0.85)
    v = state.values()
    # Научная публикация это высокое доверие по ТЗ. Без неё подтверждение слабее.
    if "science" in fam and v.get("oa_total", 0) >= 3:
        base += HIGH_TRUST_BONUS
    # Полнота опроса: каждый не ответивший источник снижает доверие к выводу. Отказ не «нейтрален».
    conf = base * state.completeness()
    if state.failed():
        reasons.append("источники не ответили: " + ", ".join(sorted(state.failed())))
    if maturity_unknown:
        conf -= UNKNOWN_MATURITY_PENALTY
    if state.uses_stale_cache():
        conf -= 0.05
        reasons.append("часть свидетельств взята из просроченного кэша")
    return round(_clip(conf), 3), reasons


# ---------- итог ----------

def supporting_documents(documents) -> list:
    """Конкретные документы, которые годятся в подтверждение: разряд A или B, первичность для
    утверждения о существовании технологии И отношение к самому кандидату. Пресс-релизы, сообщества
    и неопознанные издания подтверждением не являются (R2-B, R2-C); документ про другую технологию —
    тоже не является (claim-relative)."""
    return [d for d in (documents or [])
            if trust.supports_claim(d, eq.PRIMARY_CLAIM) and eq.counts_as_support(d)]


@dataclass
class Decision:
    decision: str
    decision_reasons: list[str] = field(default_factory=list)
    decision_reason_codes: list[str] = field(default_factory=list)
    supporting_document_count: int = 0
    supporting_document_families: list[str] = field(default_factory=list)
    # R2-A / R2-D: состояние семантической проверки и разделение агрегатных и конкретных семейств.
    verification: str = DETERMINISTIC_ONLY
    measured_aggregate_families: list[str] = field(default_factory=list)
    concrete_supporting_families: list[str] = field(default_factory=list)
    evidence_quality: dict = field(default_factory=dict)
    emergence_score: float = 0.0
    maturity_risk: float = 0.0
    evidence_confidence: float = 0.0
    hype_risk: float = 0.0
    # Честность показателей. `evidence_confidence` измеряет ДОСТУПНОСТЬ И КАЧЕСТВО ИСТОЧНИКОВ
    # (сколько адаптеров ответило, каких разрядов документы), а НЕ «насколько доказана эта
    # технология». Претензионная достаточность — отдельная величина: есть ли хоть один документ,
    # НАЗВАННЫЙ про этого кандидата. Смешивать их нельзя: иначе 0,95 читается как «95 % доказательств
    # подтверждают технологию», хотя ни один документ может быть не про неё.
    claim_evidence_sufficient: bool = False
    claim_relevant_document_count: int = 0
    # P1-B. False означает: хайп НЕ измерен (источник отказал), а не «хайпа нет».
    hype_measured: bool = True
    legacy_model_probability: float | None = None
    legacy_model_reliable: bool = False
    emergence_parts: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.decision == ACCEPT

    def rank_score(self) -> float:
        """Порядок выдачи: появление, взвешенное качеством доказательства, со штрафом за зрелость и хайп."""
        return round(self.emergence_score * self.evidence_confidence * (1 - self.maturity_risk) * (1 - self.hype_risk), 4)

    def to_dict(self) -> dict:
        return {"decision": self.decision, "decision_reasons": list(self.decision_reasons),
                "decision_reason_codes": list(self.decision_reason_codes),
                "supporting_document_count": self.supporting_document_count,
                "supporting_document_families": list(self.supporting_document_families),
                "verification": self.verification,
                "measured_aggregate_families": list(self.measured_aggregate_families),
                "concrete_supporting_families": list(self.concrete_supporting_families),
                "evidence_quality": dict(self.evidence_quality),
                "emergence_score": self.emergence_score, "maturity_risk": self.maturity_risk,
                # Имена намеренно разведены: доступность источников против претензионной
                # достаточности. Первое не является утверждением о доказанности технологии.
                "source_availability_confidence": self.evidence_confidence,
                "evidence_confidence": self.evidence_confidence,
                "claim_evidence_sufficient": self.claim_evidence_sufficient,
                "claim_relevant_document_count": self.claim_relevant_document_count,
                "hype_risk": self.hype_risk,
                "hype_measured": self.hype_measured,
                "legacy_model_probability": self.legacy_model_probability,
                "legacy_model_reliable": self.legacy_model_reliable,
                "rank_score": self.rank_score(), "emergence_parts": self.emergence_parts,
                "evidence": self.evidence}


def decide(state: EvidenceState, *, burst: float = 0.0, noise_reason: str | None = None,
           legacy_probability: float | None = None, documents=None,
           noise_code: str = "semantic_rejection",
           verification: str = DETERMINISTIC_ONLY, high_certainty: bool = False,
           candidate=None) -> Decision:
    """Единственная точка, где кандидат получает итоговый статус.

    `high_certainty` — морфологический признак названия технологии. Он сохраняется и показывается,
    но правом на окончательное подтверждение НЕ обладает (R2-B).

    `candidate` — личность кандидата после склейки. По ней пересчитывается претензионно-относительная
    релевантность документов: документ подтверждает технологию, только если назван про неё, а не
    просто найден по тем же словам запроса.

    Порядок правил фиксирован и монотонен:
    шум → агрегатная достаточность → зрелость → хайп → конкретные документы → появление → принять."""
    em, parts = emergence(state, burst)
    mat, mat_reasons, mat_unknown = maturity(state)
    quality = eq.assess_documents(documents, aggregate_families=state.corroborating_families(),
                                  candidate=candidate)
    hyp, hyp_reasons = hype(state, quality=quality, burst=burst)
    hype_measured = hype_evidence_available(state, quality)
    conf, conf_reasons = evidence_confidence(state, mat_unknown)
    # Вектор признаков модели этапа 1 полон только когда измерены все её источники, включая счётчики
    # поиска GitHub. На живом пути они по умолчанию не запрашиваются (лимит 30 запросов в минуту),
    # поэтому вероятность модели там честно помечается как непоказательная.
    legacy_reliable = (not state.failed() and bool(state.succeeded())
                       and "gh_total" in state.values() and state.ok("wiki"))
    d = Decision(decision=ACCEPT, emergence_score=em, maturity_risk=mat, evidence_confidence=conf,
                 claim_evidence_sufficient=bool(quality.supporting),
                 claim_relevant_document_count=len(quality.supporting),
                 hype_risk=hyp, hype_measured=hype_measured,
                 legacy_model_probability=(round(float(legacy_probability), 3)
                                                          if legacy_probability is not None else None),
                 legacy_model_reliable=legacy_reliable, emergence_parts=parts, evidence=state.to_dict())
    reasons: list[str] = []

    supporting = quality.supporting
    d.supporting_document_count = len(supporting)
    d.supporting_document_families = list(quality.concrete_supporting_families)
    d.verification = verification if verification in VERIFICATION_STATES else DETERMINISTIC_ONLY
    d.measured_aggregate_families = list(quality.measured_aggregate_families)
    d.concrete_supporting_families = list(quality.concrete_supporting_families)
    d.evidence_quality = quality.to_dict()

    if noise_reason:
        d.decision, d.decision_reasons, d.decision_reason_codes = NOISE, [noise_reason], [noise_code]
        return d

    # --- шлюз достаточности доказательства (PHASE 2) ---
    if state.all_failed():
        d.decision = LOW_EVIDENCE
        d.decision_reasons = ["все источники недоступны: принять кандидата нельзя"] + conf_reasons
        d.decision_reason_codes = ["all_adapters_failed"]
        return d
    fam = state.corroborating_families()
    if not fam and state.lead_only_families():
        d.decision = LOW_EVIDENCE
        d.decision_reasons = ["подтверждение только из сообществ и лент раннего отклика: "
                              "по ТЗ это первичный индикатор, не подтверждение"] + conf_reasons
        d.decision_reason_codes = ["lead_only_evidence"]
        return d
    if len(fam) < MIN_CORROBORATING_FAMILIES:
        d.decision = LOW_EVIDENCE
        d.decision_reasons = [f"недостаточно независимых подтверждений: {len(fam)} семейство источников "
                              f"из необходимых {MIN_CORROBORATING_FAMILIES}"] + conf_reasons
        d.decision_reason_codes = ["insufficient_corroborating_families"]
        return d
    if conf < MIN_EVIDENCE_CONFIDENCE:
        d.decision = LOW_EVIDENCE
        d.decision_reasons = [f"качество доказательства ниже порога: {conf:.2f} < {MIN_EVIDENCE_CONFIDENCE}"] + conf_reasons
        d.decision_reason_codes = ["low_evidence_confidence"]
        return d

    # --- вето зрелости, затем шлюз доказательства, затем хайп ---
    # Зрелость измеряется по агрегатным счётчикам и не зависит от того, удалось ли получить документы,
    # поэтому проверяется первой: иначе зрелые технологии попадали бы в список наблюдения.
    # Шлюз доказательства стоит перед хайпом, чтобы «промо вместо доказательств» получало точную
    # причину, а хайп оставался утверждением о кандидате, доказательства по которому есть.
    if mat >= MATURITY_VETO:
        d.decision = MATURE
        d.decision_reasons = mat_reasons or ["признаки зрелой технологии"]
        d.decision_reason_codes = ["maturity_veto"]
        return d
    # --- настоящий шлюз качества конкретного доказательства (R2-C, R2-E) ---
    passed, code, reason = eq.gate(quality)
    if not passed:
        d.decision = LOW_EVIDENCE
        d.decision_reasons = [reason] + conf_reasons
        d.decision_reason_codes = [code]
        return d

    if hyp >= HYPE_VETO:
        d.decision = HYPE
        d.decision_reasons = hyp_reasons or ["признаки маркетингового шума"]
        d.decision_reason_codes = ["hype_veto"]
        return d
    # --- семантическая проверка (R2-A, ужесточено R2-B) ---
    # Окончательное подтверждение даёт ТОЛЬКО успешная семантическая нормализация.
    #
    # Раньше его мог дать и узкий морфологический признак (`high_certainty`). Проверка на свежих
    # состязательных фразах показала, что это небезопасно: правило смотрит на позицию головного слова,
    # поэтому перевёрнутые n-граммы и обрывки со служебным словом получали подтверждение. Морфология
    # остаётся полезным сигналом — она показывается в карточке и участвует в порядке выдачи, — но
    # права на ACCEPT у неё больше нет.
    if d.verification != SEMANTICALLY_VERIFIED:
        d.decision = LOW_EVIDENCE
        explanation = (": лексика кандидата неоднозначна" if d.verification == SEMANTIC_UNCERTAIN
                       else ": нормализация не выполнялась")
        note = ("морфология указывает на название технологии, но подтверждением это не является"
                if high_certainty else
                "детерминированных оснований считать фразу названием технологии недостаточно")
        d.decision_reasons = [
            "семантическая проверка не выполнена" + explanation + ". " + note
            + ". Доказательства собраны и видны, но подтверждённым слабым сигналом кандидат не называется"
        ] + conf_reasons
        d.decision_reason_codes = ["semantic_verification_pending"]
        return d

    # --- достаточность доказательства хайпа (P1-B) ---
    # Проверка стоит ПОСЛЕ шлюза доказательства и семантики намеренно: отказ источника может только
    # ОТНЯТЬ принятие, но не меняет причину у кандидата, который и так был отклонён раньше. Иначе
    # падение провайдера переписывало бы причины отказа у всей выдачи.
    if not d.hype_measured:
        d.decision = LOW_EVIDENCE
        d.decision_reasons = [
            "хайп измерить нечем: источник маркетингового шума не ответил, а других полученных "
            "документов недостаточно для проверки перепечаток и промо-перекоса. Отказ источника это "
            "не подтверждение отсутствия хайпа, поэтому кандидат не принимается"
        ] + conf_reasons
        d.decision_reason_codes = ["hype_evidence_unavailable"]
        return d

    if em < MIN_EMERGENCE:
        d.decision = NOISE
        d.decision_reasons = [f"нет измеримого появления: emergence {em:.2f} < {MIN_EMERGENCE}"]
        d.decision_reason_codes = ["no_measurable_emergence"]
        return d

    d.decision_reason_codes = ["accepted"]
    # R2-D. Формулировки разделены: конкретные документы называются подтверждением, агрегатные
    # счётчики — активностью. Утверждение «подтверждено медиа» без медиа-документа невозможно.
    reasons.append(f"подтверждено конкретными документами: {len(supporting)} "
                   f"({', '.join(quality.concrete_supporting_families)}), "
                   f"независимых свидетельств {quality.independent_supports}")
    if quality.aggregate_only_families:
        reasons.append("активность по счётчикам без полученных документов: "
                       + ", ".join(quality.aggregate_only_families))
    reasons.append(f"семантическая проверка: {d.verification}")
    reasons.append(f"появление {em:.2f}, качество доказательства {conf:.2f}, зрелость {mat:.2f}, хайп {hyp:.2f}")
    # Оговорки видны и у принятых кандидатов: иначе отказ источника исчезает из выдачи.
    if mat_unknown:
        reasons.append("зрелость не проверена: Википедия не ответила")
    reasons += conf_reasons
    if not legacy_reliable:
        reasons.append("вероятность модели этапа 1 не показательна: её вектор признаков собран не полностью")
    d.decision_reasons = reasons
    return d
