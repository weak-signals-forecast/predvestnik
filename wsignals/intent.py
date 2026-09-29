"""Разделение намерения пользователя и предметной области запроса (REPAIR A, уточнено P1-B).

Живая проверка показала корневую ошибку продукта: слова, которыми пользователь описывает ЗАДАЧУ
(«слабые сигналы», «зарождающиеся тренды», «перспективные решения», «emerging», «future»), уходили
в поиск как предметные термины. Для запроса «слабые сигналы в кибербезопасности» система искала
документы, содержащие словосочетание «weak signal», и находила обработку сигналов, акустику и SNR
вместо технологий кибербезопасности.

Намерение и область — разные вещи:

    запрос    «слабые сигналы в кибербезопасности»
    намерение найти зарождающиеся технологии          (горизонт-сканирование)
    область   кибербезопасность                       (что искать в источниках)

P1-B. Первая версия ремонта вычитала слова намерения БЕЗУСЛОВНО и по одному. Независимый аудит
показал, чем это плохо: «methods for weak acoustic signal detection» превращалось в «methods
acoustic detection», а «weak signal propagation reporter» — в «propagation reporter». То есть там,
где слабый сигнал является настоящим предметом (радиоастрономия, акустика, ЭЭГ), продукт терял
именно определяющий признак области.

Поэтому маркеры намерения разделены на два класса:

  TASK_PATTERNS       «перспективные технологии», «emerging technologies», «зарождающиеся тренды» —
                      вершина словосочетания это общая рубрика (технологии, решения, тренды).
                      Названием технологии это не бывает никогда, вычитается всегда.

  AMBIGUOUS_PATTERNS  «слабые сигналы», «weak signals», «early warning» — вершина это содержательное
                      существительное. Это ЗАДАЧА только в инструктивной рамке («слабые сигналы В
                      кибербезопасности»), а «усилители слабых сигналов в радиоастрономии» и
                      «weak signal propagation» — обычный предмет, и трогать его нельзя.

Список ожидаемых технологий здесь отсутствует и появиться не может: правила морфологические.

ВАЖНО о границах ответственности (P1-B). Этот модуль отвечает за разбор ПОЛЬЗОВАТЕЛЬСКОГО ЗАПРОСА.
Он НЕ является словарём запрещённых слов для названий технологий-кандидатов: гигиена кандидатов
живёт в `relevance` и опирается на структуру фразы, а не на глобальный запрет токенов.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Маркеры задачи с ОБЩЕЙ рубрикой в вершине: «перспективные технологии», «зарождающиеся тренды».
# Такое словосочетание не бывает названием технологии, поэтому вычитается всегда.
TASK_PATTERNS = (
    r"зарождающ\w*\s+(?:тренд\w*|технологи\w*|направлени\w*|решени\w*)",
    r"перспективн\w*\s+(?:решени\w*|технологи\w*|направлени\w*|разработк\w*)",
    r"будущ\w*\s+(?:технологи\w*|решени\w*|направлени\w*)",
    r"нов\w*\s+(?:технологи\w*|решени\w*|способ\w*|метод\w*)",
    r"тренд\w*",
    r"горизонт\w*\s+сканировани\w*",
    r"что\s+происходит\s+с",
    r"emerging\s+(?:technolog\w*|trends?|solutions?|directions?)",
    r"promising\s+(?:technolog\w*|solutions?|approaches?|directions?)",
    r"future\s+(?:technolog\w*|solutions?|trends?|directions?)",
    r"horizon\s+scanning",
    r"technology\s+foresight",
)
# Маркеры с СОДЕРЖАТЕЛЬНОЙ вершиной: задачей они являются только в инструктивной рамке.
AMBIGUOUS_PATTERNS = (
    r"слаб\w*\s+сигнал\w*",
    r"ранн\w*\s+(?:сигнал\w*|признак\w*|стади\w*)",
    r"weak\s+signals?",
    r"early\s+(?:signals?|warnings?|indicators?)",
)
TASK_RE = re.compile("|".join(TASK_PATTERNS), re.I | re.U)
AMBIGUOUS_RE = re.compile("|".join(AMBIGUOUS_PATTERNS), re.I | re.U)
# Полный набор маркеров: используется только там, где нужно ЗАМЕТИТЬ намерение, а не вычесть его.
INTENT_RE = re.compile("|".join(TASK_PATTERNS + AMBIGUOUS_PATTERNS), re.I | re.U)

# Предлоги, вводящие предметную область: «слабые сигналы В кибербезопасности».
DOMAIN_PREPOSITIONS = {
    "в", "во", "на", "по", "для", "о", "об", "обо", "при", "среди", "внутри", "вокруг",
    "in", "of", "for", "within", "across", "on", "about", "around", "at", "regarding",
}
# Глаголы-просьбы: инструктивная рамка, а не предметное содержание («найди слабые сигналы в X»).
REQUEST_WORDS = {
    "найди", "найти", "ищи", "искать", "поищи", "покажи", "показать", "подбери", "подобрать",
    "дай", "выяви", "выявить", "собери", "собрать", "перечисли", "нужны", "нужно", "хочу",
    "find", "show", "search", "list", "give", "identify", "discover", "surface", "detect",
}

# Одиночные слова-намерения. ТОЛЬКО для чистки поисковых фраз, которые вернула модель
# (`decompose._without_intent`): там любое упоминание задачи — мусор в поиске. Для разбора
# пользовательского запроса и для гигиены кандидатов этот список не применяется (P1-B).
INTENT_WORDS = {
    "слабые", "слабых", "сигналы", "сигналов", "сигналах", "зарождающиеся", "зарождающихся",
    "перспективные", "перспективных", "будущие", "новые", "новых", "тренд", "тренды", "трендов",
    "ранние", "ранних", "emerging", "promising", "future", "trend", "trends", "weak", "signal",
    "signals", "early", "novel", "upcoming", "next",
}
# Служебные связки, которые не несут предметного содержания.
GLUE_WORDS = {
    "в", "во", "на", "по", "о", "об", "для", "и", "с", "со", "к", "ко", "у", "из", "за", "от", "до",
    "при", "про", "над", "под", "а", "но", "или", "как", "что", "это", "сфере", "области",
    "направлении", "направления", "решения", "решений", "технологии", "технологий",
    "in", "of", "for", "the", "a", "an", "on", "at", "to", "and", "or", "about", "within", "area",
    "field", "domain", "technologies", "technology", "solutions",
}


@dataclass
class QueryIntent:
    """Результат разделения: что искать (`domain`) и зачем (`intent_markers`)."""
    original: str
    domain: str
    intent_markers: list[str]
    domain_is_fallback: bool = False       # область вычленить не удалось, используем исходный запрос

    @property
    def horizon_scanning(self) -> bool:
        """Запрос сформулирован как поиск зарождающегося, а не как поиск конкретной вещи."""
        return bool(self.intent_markers)

    def to_dict(self) -> dict:
        return {"original": self.original, "domain": self.domain,
                "intent_markers": list(self.intent_markers),
                "domain_is_fallback": self.domain_is_fallback,
                "horizon_scanning": self.horizon_scanning}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\-]", " ", text, flags=re.U)).strip()


def _is_instruction_frame(text: str, match: re.Match) -> bool:
    """Маркер с содержательной вершиной описывает ЗАДАЧУ, а не предмет, только если:

      * перед ним нет предметных слов («усилители слабых сигналов» — предмет), и
      * после него идёт предлог, вводящий область («слабые сигналы В агротехе»),
        либо не идёт ничего («слабые сигналы» целиком).

    Иначе маркер — обычное определение при следующем существительном («weak signal propagation»)."""
    before = [w.lower() for w in _clean(text[:match.start()]).split()]
    if any(w not in GLUE_WORDS and w not in REQUEST_WORDS and w not in INTENT_WORDS for w in before):
        return False
    after = [w.lower() for w in _clean(text[match.end():]).split()]
    return not after or after[0] in DOMAIN_PREPOSITIONS


def task_marker_spans(text: str) -> list[tuple[int, int]]:
    """Участки запроса, описывающие ЗАДАЧУ поиска. Пересекающиеся участки объединяются."""
    spans = [m.span() for m in TASK_RE.finditer(text)]
    spans += [m.span() for m in AMBIGUOUS_RE.finditer(text) if _is_instruction_frame(text, m)]
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def split(query: str) -> QueryIntent:
    """Делит свободный запрос на предметную область и маркеры намерения.

    Вычитаются ТОЛЬКО участки, распознанные как описание задачи. Отдельные слова («weak», «signal»,
    «early») из области не удаляются: там, где слабый сигнал и есть предмет, он обязан сохраниться
    (P1-B).

    Область никогда не бывает пустой: если после вычитания намерения ничего не остаётся, областью
    считается исходный запрос, а факт отмечается флагом `domain_is_fallback`."""
    original = (query or "").strip()
    spans = task_marker_spans(original)
    markers = [original[s:e].strip() for s, e in spans if original[s:e].strip()]
    rest, cursor = [], 0
    for start, end in spans:
        rest.append(original[cursor:start])
        cursor = end
    rest.append(original[cursor:])
    words = [w for w in _clean(" ".join(rest)).split() if w]
    kept = [w for w in words if w.lower() not in GLUE_WORDS and w.lower() not in REQUEST_WORDS]
    if not kept:
        # Вся фраза оказалась намерением: область вычленить нечем, работаем по исходному запросу.
        return QueryIntent(original=original, domain=original, intent_markers=markers,
                           domain_is_fallback=True)
    return QueryIntent(original=original, domain=" ".join(kept), intent_markers=markers)


def strip_intent_tokens(phrase: str) -> str:
    """Убирает слова-намерения из ПОИСКОВОЙ ФРАЗЫ, которую вернула модель.

    Применяется только к плану поиска (`decompose`), где любое упоминание задачи — мусор в запросе
    к источникам. К названиям технологий-кандидатов это правило не применяется (P1-B).

    Пустую строку не возвращает: если после вычитания ничего не осталось, возвращается исходная
    фраза (решение о ней принимают другие правила)."""
    words = _clean(phrase).split()
    kept = [w for w in words if w.lower() not in INTENT_WORDS]
    return " ".join(kept) if kept else phrase


def is_intent_restatement(phrase: str) -> bool:
    """Поисковая фраза целиком состоит из слов о задаче поиска и служебной лексики: это пересказ
    намерения пользователя, а не предметный запрос. Используется при чистке плана поиска."""
    words = [w.lower() for w in _clean(phrase).split()]
    if not words:
        return False
    return all(w in INTENT_WORDS or w in GLUE_WORDS for w in words)
