"""Русские названия технологий для выдачи.

ТЗ требует, чтобы вся аналитическая выдача была на русском, при этом оригинальное название источника сохраняется.
Машинный перевод общего назначения для терминов непригоден: «rag poisoning» он переводит как «отравление тряпкой»,
«payment rails» как «платные рельсы». Поэтому русское название берётся, по убыванию надёжности:

  1. словарь терминов: 198 пар «английская фраза — русское название», написанных нашей командой.
     Эталонные 100 строк таблицы организаторов из словаря ИСКЛЮЧЕНЫ: таблица организаторов не
     является авторитетом отрисовки в рантайме (деконтаминация рантайма). Она остаётся только
     в обучении, бенчмарке и справочных инструментах;
  2. разрешённая языковая модель, если задан ключ (перевод помечается как генеративный);
  3. оригинальное название, если ничего не подошло.
"""
from __future__ import annotations

import functools
import re
from pathlib import Path

from . import llm

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dataset" / "technologies.csv"
STOP = {"the", "a", "an", "of", "for", "and", "in", "on", "to", "with", "ai", "llm", "llms"}
# Готовые названия для терминов, которые пословно не собираются
PHRASES = {
    "payment rails": "платёжные рельсы", "red team": "редтиминг", "red teaming": "редтиминг",
    "harness engineering": "инженерия исполняющей обвязки агентов", "on-policy distillation": "дистилляция on-policy",
    "ai pentesting": "тестирование на проникновение с ИИ", "prompt injection": "инъекции в промпт",
}
# Головное слово фразы: то, что переводится в именительном падеже и ставится первым
HEAD = {
    "security": "безопасность", "payments": "платежи", "payment": "платежи", "memory": "память", "compression": "сжатие",
    "distillation": "дистилляция", "poisoning": "отравление", "skills": "навыки", "models": "модели", "model": "модель",
    "agents": "агенты", "agent": "агент", "identity": "идентичность", "insurance": "страхование", "monitoring": "мониторинг",
    "detection": "детекция", "benchmark": "бенчмарк", "benchmarks": "бенчмарки", "protocol": "протокол", "chips": "чипы",
    "chip": "чип", "sensors": "сенсоры", "sensor": "сенсор", "orchestration": "оркестрация", "routing": "маршрутизация",
    "inference": "инференс", "cooling": "охлаждение", "welding": "сварка", "watermarking": "водяные знаки",
    "unlearning": "разобучение", "marketplace": "маркетплейс", "engineering": "инженерия", "guardrails": "ограничители",
    "scanners": "сканеры", "scanner": "сканер", "firewall": "межсетевой экран", "sandbox": "песочница",
    "attestation": "аттестация", "provenance": "происхождение", "governance": "управление", "compliance": "комплаенс",
    "hand": "кисть", "hands": "кисти", "actuators": "приводы", "batteries": "аккумуляторы", "tools": "инструменты",
    "sensing": "сенсорика", "computing": "вычисления", "learning": "обучение", "training": "обучение",
    "networks": "сети", "network": "сеть", "storage": "хранение", "search": "поиск", "reasoning": "рассуждение",
    "verification": "верификация", "authentication": "аутентификация", "encryption": "шифрование",
    "simulation": "моделирование", "optimization": "оптимизация", "automation": "автоматизация", "scoring": "скоринг",
}
# Аббревиатуры, которые могут быть головным словом фразы: «децентрализованный KYC»
ABBR = {"kyc": "KYC", "aml": "AML", "rag": "RAG", "mcp": "MCP", "llm": "LLM", "iot": "IoT", "api": "API",
        "cbdc": "CBDC", "tee": "TEE", "npu": "NPU", "vla": "VLA"}
# Определения: основа прилагательного, окончание подбирается по роду и числу головного слова
ADJ = {
    "agentic": "агентн", "machine": "машинн", "decentralized": "децентрализованн", "diffusion": "диффузионн",
    "local": "локальн", "autonomous": "автономн", "confidential": "конфиденциальн", "federated": "федеративн",
    "neuromorphic": "нейроморфн", "photonic": "фотонн", "quantum": "квантов", "orbital": "орбитальн",
    "tactile": "тактильн", "analog": "аналогов", "secure": "защищённ", "open-source": "открыт",
    "language": "языков", "industrial": "промышленн", "synthetic": "синтетическ", "hybrid": "гибридн",
}
# Род и число головных слов, чтобы определение согласовывалось: «квантовая сенсорика», «квантовые вычисления»
GENDER = {
    "безопасность": "f", "платежи": "p", "память": "f", "сжатие": "n", "дистилляция": "f", "отравление": "n",
    "навыки": "p", "модели": "p", "модель": "f", "агенты": "p", "агент": "m", "идентичность": "f", "страхование": "n",
    "мониторинг": "m", "детекция": "f", "бенчмарк": "m", "бенчмарки": "p", "протокол": "m", "чипы": "p", "чип": "m",
    "сенсоры": "p", "сенсор": "m", "оркестрация": "f", "маршрутизация": "f", "инференс": "m", "охлаждение": "n",
    "сварка": "f", "водяные знаки": "p", "разобучение": "n", "маркетплейс": "m", "инженерия": "f", "ограничители": "p",
    "сканеры": "p", "сканер": "m", "межсетевой экран": "m", "песочница": "f", "аттестация": "f", "происхождение": "n",
    "управление": "n", "комплаенс": "m", "кисть": "f", "кисти": "p", "приводы": "p", "аккумуляторы": "p",
    "инструменты": "p", "сенсорика": "f", "вычисления": "p", "обучение": "n", "сети": "p", "сеть": "f",
    "хранение": "n", "поиск": "m", "рассуждение": "n", "верификация": "f", "аутентификация": "f", "шифрование": "n",
    "моделирование": "n", "оптимизация": "f", "автоматизация": "f", "скоринг": "m",
}


def agree(stem: str, head: str) -> str:
    """Окончание прилагательного по роду и числу головного слова. После г, к, х, ж, ш, ч, щ пишется «и»."""
    g = GENDER.get(head, "m")
    soft = stem[-1] in "гкхжшчщ"
    return stem + {"m": "ий" if soft else "ый", "f": "ая", "n": "ое", "p": "ие" if soft else "ые"}[g]
# Зависимые слова: ставятся после головного, в родительном падеже или как аббревиатура
GEN = {
    "agent": "агентов", "agents": "агентов", "model": "моделей", "models": "моделей", "context": "контекста",
    "token": "токенов", "tokens": "токенов", "memory": "памяти", "data": "данных", "payment": "платежей",
    "code": "кода", "llm": "LLM", "rag": "RAG", "mcp": "MCP", "ai": "ИИ", "kyc": "KYC", "aml": "AML", "iot": "IoT",
    "edge": "на периферии", "cloud": "в облаке", "robot": "роботов", "robots": "роботов",
    "blockchain": "на блокчейне", "crypto": "в криптовалюте",
}


PHRASES_N: dict[str, str] = {}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _tokens(s: str) -> set[str]:
    return {t for t in _norm(s).split() if t not in STOP}


PHRASES_N.update({_norm(k): v for k, v in PHRASES.items()})


# Строки, пришедшие из таблицы организаторов «100 слабых технологических сигналов». Они
# опознаются по собственному полю организаторов `org_score` (у строк нашей команды оно пустое);
# тот же набор даёт и метка `weak_signal` — обе проверки согласованы и закреплены тестом.
ORGANIZER_COLUMNS = ("org_score", "org_stage", "org_sources")
ORGANIZER_LABEL = "weak_signal"


def _filled(value) -> bool:
    """Значение действительно заполнено. Пустая ячейка CSV приходит из pandas как NaN, и наивное
    `str(value)` дало бы строку «nan» — правдивую по Python и ложную по смыслу."""
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def _is_organizer_row(row: dict) -> bool:
    """Строка-эталон организаторов. Названием технологии в рантайме она быть не может."""
    if str(row.get("label") or "").strip() == ORGANIZER_LABEL:
        return True
    return any(_filled(row.get(col)) for col in ORGANIZER_COLUMNS)


@functools.lru_cache(maxsize=1)
def glossary() -> list[tuple[str, set[str], str]]:
    """Пары «английская фраза, её токены, русское название» из размеченного датасета.

    ЭТАЛОННЫЕ СТРОКИ ОРГАНИЗАТОРОВ ИСКЛЮЧЕНЫ (деконтаминация рантайма).

    Измерено: кандидат «mcp security», пришедший из источников, отрисовывался ДОСЛОВНЫМ названием
    строки S004 таблицы организаторов — «Сканеры безопасности MCP-серверов и защита от tool
    poisoning». Это делало таблицу организаторов авторитетом отрисовки продукта: система выдавала
    эталонное название там, где сама ничего подобного не устанавливала.

    В словаре остаются только 198 пар, написанных командой независимо от таблицы организаторов.
    Эталонные 100 строк остаются в датасете для обучения, бенчмарка и справочных инструментов, но
    авторитетом отрисовки в рантайме не являются. Никакого другого зашитого списка технологий
    взамен не вводится."""
    import pandas as pd

    if not DATASET.exists():
        return []
    df = pd.read_csv(DATASET)
    out = []
    for row in df.to_dict("records"):
        if _is_organizer_row(row):
            continue
        q, name = row.get("query"), row.get("name")
        if isinstance(q, str) and isinstance(name, str):
            out.append((_norm(q), _tokens(q), name.strip()))
    return out


def lookup(phrase: str) -> tuple[str, float, str] | None:
    """Ближайшее русское название из словаря: сам перевод, мера совпадения токенов и английская фраза строки."""
    q, qt = _norm(phrase), _tokens(phrase)
    if not qt:
        return None
    best, score, src = None, 0.0, ""
    for eng, tokens, ru in glossary():
        if not tokens:
            continue
        j = 1.0 if q == eng else len(qt & tokens) / len(qt | tokens)
        if j > score:
            best, score, src = ru, j, eng
    return (best, score, src) if best else None


def short(name: str, limit: int = 70) -> str:
    """Название из таблицы бывает длинным описанием. Берём смысловое начало до скобки или тире."""
    head = re.split(r"\s+[—–-]\s+|\s*\(|:\s", name)[0].strip()
    head = head if len(head) >= 12 else name
    return head if len(head) <= limit else head[: limit - 1].rstrip() + "…"


def compose(phrase: str) -> str | None:
    """Собрать русское название по частям: определения перед головным словом, зависимые после него.
    Неизвестные слова остаются латиницей и ставятся в конец: «дистилляция on-policy»."""
    toks = _norm(phrase).split()
    key = " ".join(toks)
    if key in PHRASES_N:
        return PHRASES_N[key]
    head_i = next((i for i in range(len(toks) - 1, -1, -1) if toks[i] in HEAD), None)
    abbr_head = head_i is None and toks and toks[-1] in ABBR
    if abbr_head:
        head_i = len(toks) - 1
    if head_i is None:
        return None
    head_word = ABBR[toks[head_i]] if abbr_head else HEAD[toks[head_i]]
    if head_i is None:
        return None
    adj, gen, rest = [], [], []
    for i, t in enumerate(toks):
        if i == head_i:
            continue
        if t in ADJ:
            adj.append(agree(ADJ[t], head_word))
        elif t in GEN:
            gen.append(GEN[t])
        elif t in HEAD and i < head_i:
            gen.append(HEAD[t])
        else:
            # незнакомое слово остаётся латиницей; название продукта пишем с заглавной: «навыки Claude»
            rest.append(t.capitalize() if t.isalpha() and len(t) >= 4 else t)
    return " ".join(adj + [head_word] + gen + rest).strip()


def technology_name(phrase: str) -> dict:
    """Русское название технологии, оригинал и способ получения. Оригинал сохраняется всегда."""
    hit = lookup(phrase)
    head = _norm(phrase).split()[-1] if _norm(phrase) else ""
    # Совпадение принимается, только если головное слово запроса есть и в английской фразе словаря:
    # иначе «agent memory» подтянет «защиту от отравления памяти агентов», а это другая технология.
    extra_heads = {t for t in (hit[2].split() if hit else []) if t in HEAD} - set(_norm(phrase).split()) if hit else set()
    if hit and hit[1] >= 0.6 and head in hit[2].split() and not extra_heads:
        return {"ru": short(hit[0]), "original": phrase, "source": "словарь терминов проекта",
                "match": round(hit[1], 2), "generated": False}
    built = compose(phrase)
    if built:
        return {"ru": built, "original": phrase, "source": "сборка по словарю терминов", "generated": False}
    gen = llm.translate_term(phrase)
    if gen:
        return {"ru": gen, "original": phrase, "source": f"генеративный перевод, модель {llm.model_name()}", "generated": True}
    return {"ru": phrase, "original": phrase, "source": "название оставлено на языке оригинала", "generated": False}
