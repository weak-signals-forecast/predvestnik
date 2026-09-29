"""Разбор текста на фразы-кандидаты в названия технологий.

Текст режется на куски по служебным словам и знакам препинания (как в RAKE), внутри куска берутся все
подпоследовательности из 2–4 слов. Так фраза не перескакивает через «of», «for», «the» и не склеивает
соседние мысли. Одиночные слова берутся, только если похожи на имя: аббревиатура или смесь букв и
цифр (CXL, NPU, x402). Дефис считается пробелом («quantum-inspired» = «quantum inspired»), кроме составных
терминов со служебным словом внутри: «processing-in-pixel» остаётся одним токеном processing_in_pixel.
"""
from __future__ import annotations

import re

# Границы кусков: служебные слова и академическая «вода». Слова вроде model, learning, network, data
# сюда намеренно не входят: они часть названий («foundation model», «federated learning»).
BOUNDARY = set("""
a an the of for and or nor but if then than so as at by in on to into onto from with within without via per
is are was were be been being am do does did done has have had having can could may might must shall should will would
this that these those it its they them their there here we our us you your he she his her i me my who whom whose which
what when where why how whether while also both each either neither every all any some such no not only own same other
about above below between among across after before during through throughout toward towards under over up down out off
more most less least very too much many few further again once just even still yet already however thus hence therefore
new novel proposed propose proposes present presents presented study studies paper work works approach approaches method
methods towards toward using use uses used based via improved improving improve improves enhanced enhancing enhance
efficient effective robust simple scalable comprehensive systematic review survey overview analysis analyses investigation
case evaluation evaluating evaluate results result finding findings role impact impacts effect effects application applications
applying applied challenges challenge opportunities opportunity perspective perspectives future current recent state art
toward framework frameworks design designing designed development developing developed implementation implementing
et al vs versus via i.e e.g etc one two three first second third
against like unlike beyond despite near around along upon amid whose
state_of_the_art
""".split())

# Модификаторы не могут начинать фразу, общие слова не могут её заканчивать. Внутри допустимо всё.
# model, system, tool, network на краях разрешены: «foundation model», «tool poisoning», «model context protocol».
# Общие фразы с ними отсеются позже, на рядах по годам (они давно и широко распространены).
START_BAD = set("""
large small big high low general generic various several different multiple single key main important significant
based driven enabled powered aware oriented real specific certain particular potential possible typical common standard
end_to_end
""".split())
END_BAD = set("""
based driven enabled powered aware oriented level scale time times data information task tasks problem problems
issue issues performance accuracy quality efficiency cost costs process processes way ways setting settings
""".split())

TOKEN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*(?:\.[0-9]+)?")
NAME_1GRAM = re.compile(r"^(?=.*[a-z])(?=.*\d)[a-z0-9]{2,12}$")   # x402, gpt4, 6g, int8 — смесь букв и цифр
SPLIT = re.compile(r"[.,;:!?()\[\]{}\"'“”‘’/\\|<>=+*&%#@~`^$—–]|\s-\s")


HYPHENATED = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")
YEAR = re.compile(r"^(19|20)\d\d$")


def _glue(m: re.Match) -> str:
    """Составной термин через дефис со служебным словом внутри остаётся одним токеном:
    processing-in-pixel, compute-in-memory, know-your-agent, human-in-the-loop. Остальные дефисы = пробел."""
    parts = m.group(0).split("-")
    if any(p in BOUNDARY for p in parts[1:-1]):
        return "_".join(parts)
    return " ".join(parts)


def normalize(text: str) -> str:
    t = (text or "").lower().replace("_", " ")
    return HYPHENATED.sub(_glue, t)


def chunks(text: str, acronyms: set[str] | None = None) -> list[list[str]]:
    """Куски текста без служебных слов, каждый — список токенов."""
    out: list[list[str]] = []
    for piece in SPLIT.split(normalize(text)):
        cur: list[str] = []
        for tok in TOKEN.findall(piece):
            # число остаётся частью фразы, только если идёт за словом («http 402», «iso 20022»);
            # отдельно стоящее число («на 100 gpu») и год режут фразу
            num_bad = tok.replace(".", "").isdigit() and (not cur or YEAR.match(tok) or cur[-1].isdigit())
            if tok in BOUNDARY or num_bad:
                if cur:
                    out.append(cur)
                cur = []
            else:
                cur.append(tok)
        if cur:
            out.append(cur)
    return out


def ngrams(chunk: list[str], n_min: int = 2, n_max: int = 4):
    L = len(chunk)
    for n in range(n_min, min(n_max, L) + 1):
        for i in range(L - n + 1):
            g = chunk[i:i + n]
            if g[0] in START_BAD or g[-1] in END_BAD or g[0].isdigit() or len(g[0]) < 2 or len(g[-1]) < 2:
                continue
            yield " ".join(g)


def acronyms_in(raw: str) -> set[str]:
    """Аббревиатуры из исходного регистра: CXL, NPU, MCP (2–6 заглавных, возможно с цифрами)."""
    return {m.lower() for m in re.findall(r"\b[A-Z][A-Z0-9]{1,5}\b", raw or "") if not m.isdigit()}


def candidates(raw: str) -> set[str]:
    """Все фразы-кандидаты документа (множество: считаем документную частоту)."""
    out: set[str] = set()
    for ch in chunks(raw):
        out.update(ngrams(ch))
        for t in ch:
            if NAME_1GRAM.match(t) or ("_" in t and t not in BOUNDARY):   # x402; processing_in_pixel
                out.add(t)
    for a in acronyms_in(raw):
        if a not in BOUNDARY and len(a) >= 3:
            out.add(a)
    return out


def stem(w: str) -> str:
    """Грубое приведение к единственному числу: caches → cache, policies → policy. Составные токены не трогаем."""
    if "_" in w:
        return w
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("sses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is", "ics")):
        return w[:-1]
    return w


def entity_key(phrase: str) -> str:
    """Ключ сущности: отсортированный набор основ слов. Склеивает число и порядок слов:
    «kv caches» = «kv cache», «watermarking llms» = «llm watermarking», «learning federated» = «federated learning»."""
    return " ".join(sorted({stem(w) for w in phrase.split()}))
