"""Независимость свидетельств: лёгкий ключ вместо графа связей (R2-E).

Две ссылки на одну и ту же перепечатку пресс-релиза — это одно свидетельство, а не два независимых
подтверждения. Раньше независимость считалась только по семейству источника, поэтому десять копий
одного анонса в десяти лентах выглядели как десять подтверждений внутри семейства «медиа».

Ключ независимости строится из того, что уже известно о документе, без дополнительных запросов:

  семейство источника          science / code / media / press / community / …
  происхождение                зарегистрированный домен; для GitHub — владелец репозитория;
                               для научной публикации — префикс DOI (издатель)
  синдикация                   нормализованный заголовок: одинаковый текст в разных лентах
                               считается одним свидетельством

Графовой базы здесь нет и не предполагается. Ключ — строка, сравнение — равенство.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

# Хостинги кода: независимым происхождением считается владелец репозитория, а не сам хостинг.
CODE_HOSTS = {"github.com", "gitlab.com", "bitbucket.org", "codeberg.org", "gitflic.ru", "gitverse.ru"}
# Многоуровневые публичные суффиксы, встречающиеся в источниках проекта.
MULTI_SUFFIX = ("co.uk", "com.br", "com.au", "co.jp", "com.cn", "org.uk", "ac.uk", "gov.uk", "com.tr")
STOP_TITLE_WORDS = {"the", "a", "an", "of", "for", "and", "in", "on", "to", "with", "new", "says", "report"}
# Семейства, где один и тот же текст расходится по многим изданиям. Там перепечатка — одно свидетельство
# независимо от издателя: именно так выглядит разосланный пресс-релиз.
SYNDICATION_FAMILIES = frozenset({"press", "media", "aggregator", "community"})


def registrable_domain(url: str) -> str | None:
    """Домен второго уровня без www и поддоменов: `ir.reuters.com` и `www.reuters.com` это один издатель."""
    try:
        host = (urlsplit(url).netloc or "").lower().split(":")[0]
    except ValueError:
        return None
    if not host:
        return None
    host = host.removeprefix("www.")
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    tail2 = ".".join(parts[-2:])
    if tail2 in MULTI_SUFFIX or ".".join(parts[-2:]) in MULTI_SUFFIX:
        return ".".join(parts[-3:])
    return tail2


def doi_registrant(url: str) -> str | None:
    """Префикс DOI — это издатель: `10.1145/...` это ACM. Разные статьи одного издателя независимыми
    свидетельствами по происхождению не считаются, хотя и остаются разными документами."""
    m = re.search(r"(?:doi\.org/|^doi:|\b)(10\.\d{4,9})/", url or "")
    return m.group(1) if m else None


def code_origin(url: str) -> str | None:
    """Владелец репозитория: десять репозиториев одной организации — одно происхождение."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    host = (parts.netloc or "").lower().removeprefix("www.")
    if host not in CODE_HOSTS:
        return None
    segments = [s for s in (parts.path or "").split("/") if s]
    return f"{host}:{segments[0].lower()}" if segments else host


# P1-A. ОДНА НАУЧНАЯ РАБОТА != ДВА НЕЗАВИСИМЫХ ПОДТВЕРЖДЕНИЯ.
#
# Независимый аудит воспроизвёл нарушение уже действующего контракта: препринт и журнальная версия
# ОДНОЙ работы имеют разные префиксы DOI, поэтому давали independent_supports = 2 и в одиночку
# закрывали шлюз корроборации. Публикационная стадия и издатель — это не независимая проверка.
#
# Личность работы определяется тем, что уже есть в документе, без единого дополнительного запроса
# и без обращения к модели: канонический адрес, полный DOI, набор содержательных слов заголовка,
# а также заголовочное ядро до подзаголовка (журнальная версия часто добавляет «: подзаголовок»).
# Порог намеренно консервативен — лучше ошибочно склеить две работы, чем ошибочно признать одну
# работу двумя независимыми свидетельствами. Простого пересечения тематических слов для склейки
# недостаточно: нужна и близкая длина заголовка, и почти полное вхождение короткого в длинный.
WORK_FAMILIES = frozenset({"science", "patent"})
WORK_TITLE_MIN_TOKENS = 4          # на коротких заголовках близость неразличима от совпадения темы
WORK_TITLE_CONTAINMENT = 0.85      # почти все слова короткого заголовка есть в длинном
WORK_TITLE_LENGTH_RATIO = 0.75     # заголовки сопоставимой длины: переименование, а не другая работа

# Обычные разделители подзаголовка в публикациях. Дефис сюда НЕ входит: он живёт внутри терминов
# («low-noise amplifier»), и резать по нему значило бы ломать настоящие названия.
SUBTITLE_SEPARATORS = ":\u2014\u2013"                     # двоеточие, длинное тире, среднее тире
_SUBTITLE_SPLIT = re.compile(f"[{SUBTITLE_SEPARATORS}]")


def full_doi(url: str) -> str | None:
    """Полный DOI, а не только префикс издателя: точное тождество работы."""
    m = re.search(r"(?:doi\.org/|^doi:|\b)(10\.\d{4,9}/[^\s?#]+)", url or "", re.I)
    return m.group(1).rstrip(".").lower() if m else None


def work_tokens(title: str) -> frozenset[str]:
    """Содержательные слова заголовка. Порядок и пунктуация не важны, служебные слова отброшены."""
    return frozenset(w for w in re.findall(r"[a-zа-яё0-9]+", (title or "").lower())
                     if w not in STOP_TITLE_WORDS)


def title_core(title: str) -> frozenset[str]:
    """Заголовочное ядро: слова ДО первого разделителя подзаголовка.

    Пустое множество, если подзаголовка нет или он пуст: тогда правило ядра просто не применяется."""
    parts = _SUBTITLE_SPLIT.split(title or "", maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        return frozenset()
    return work_tokens(parts[0])


def _is_subtitle_variant(a_title: str, b_title: str) -> bool:
    """Один заголовок ЦЕЛИКОМ является ядром другого: журнальная версия добавила подзаголовок.

        «Adaptive photonic control for quantum sensors»
        «Adaptive photonic control for quantum sensors: a scalable experimental architecture»

    Сравнивается ПОЛНЫЙ заголовок одной работы с ЯДРОМ другой, и никогда ядро с ядром. Разница
    принципиальная: две разные работы с общим родовым началом («Adaptive photonic control: quantum
    sensor calibration» и «Adaptive photonic control: integrated LiDAR arrays») имеют одинаковые
    ядра, но ни один из их полных заголовков ядром другого не является, поэтому они не сливаются.
    Совпадения по префиксу здесь нет: требуется точное равенство множеств слов.

    Ядро короче порога не годится: на двух-трёх родовых словах это уже совпадение темы, а не работы."""
    for full_title, other_title in ((a_title, b_title), (b_title, a_title)):
        core = title_core(other_title)
        if len(core) >= WORK_TITLE_MIN_TOKENS and core == work_tokens(full_title):
            return True
    return False


def same_scientific_work(a, b) -> bool:
    """Два документа — разные представления ОДНОЙ работы (препринт и журнал, зеркало, переиздание).

    Детерминированно, без сети и без модели. Ни белых списков журналов, ни перечней технологий."""
    url_a = getattr(a, "canonical_url", None) or getattr(a, "url", "") or ""
    url_b = getattr(b, "canonical_url", None) or getattr(b, "url", "") or ""
    if url_a and url_a == url_b:
        return True
    doi_a, doi_b = full_doi(url_a), full_doi(url_b)
    if doi_a and doi_a == doi_b:
        return True
    ta, tb = work_tokens(getattr(a, "title", "")), work_tokens(getattr(b, "title", ""))
    if not ta or not tb:
        return False
    if ta == tb:
        return True                                   # тот же заголовок с точностью до регистра и знаков
    if _is_subtitle_variant(getattr(a, "title", ""), getattr(b, "title", "")):
        return True                                   # журнальная версия добавила подзаголовок
    short, long = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(short) < WORK_TITLE_MIN_TOKENS or len(long) < WORK_TITLE_MIN_TOKENS:
        return False
    if len(short) / len(long) < WORK_TITLE_LENGTH_RATIO:
        return False
    return len(short & long) / len(short) >= WORK_TITLE_CONTAINMENT


def normalized_title(title: str) -> str:
    """Заголовок без служебных слов и пунктуации: ловит перепечатку одного текста в разных лентах."""
    words = [w for w in re.findall(r"[a-zа-яё0-9]+", (title or "").lower()) if w not in STOP_TITLE_WORDS]
    return " ".join(words[:10])


def origin(doc) -> str:
    """Происхождение документа: то, что делает свидетельство зависимым от другого."""
    url = getattr(doc, "canonical_url", None) or getattr(doc, "url", "") or ""
    family = getattr(doc, "source_family", "") or "other"
    code = code_origin(url)
    if code:
        return code
    if family in {"science", "patent"}:
        registrant = doi_registrant(url)
        if registrant:
            return f"doi:{registrant}"
    domain = registrable_domain(url)
    if domain:
        return domain
    name = re.sub(r"\s+", " ", (getattr(doc, "source_name", "") or "")).strip().lower()
    return name or "unknown-origin"


def independence_key(doc) -> str:
    """Ключ независимости. Два документа с одинаковым ключом считаются ОДНИМ свидетельством.

    В новостных семействах ключ строится ТОЛЬКО по тексту заголовка, без издателя: один анонс,
    разосланный в десять лент, — это одно свидетельство, а не десять подтверждений. В научных,
    патентных и кодовых семействах разные работы одного издателя или владельца остаются разными
    свидетельствами, поэтому в ключ входят и происхождение, и заголовок."""
    family = getattr(doc, "source_family", "") or "other"
    title = normalized_title(getattr(doc, "title", ""))
    if family in SYNDICATION_FAMILIES and title:
        return f"syndicated|{title}"
    return f"{family}|{origin(doc)}|{title}"


def syndication_key(doc) -> str:
    """Подпись перепечатки: один и тот же текст в разных изданиях. Издатель в ключ не входит."""
    return normalized_title(getattr(doc, "title", ""))


def independent_documents(documents) -> list:
    """Оставляет по одному документу на каждое независимое свидетельство, сохраняя порядок.

    Два прохода. Сначала совпадение ключа независимости: перепечатка, тот же издатель с тем же
    заголовком, тот же владелец репозитория. Затем — тождество научной работы (P1-A): препринт и
    журнальная версия одной работы схлопываются в одно свидетельство, хотя у них разные издатели,
    разные DOI и слегка разные заголовки. При схлопывании остаётся документ с лучшим разрядом
    доверия."""
    from .trust import TIER_ORDER

    def tier(d) -> int:
        return TIER_ORDER.get(getattr(d, "trust", ""), 0)

    best: dict[str, object] = {}
    order: list[str] = []
    for d in documents or []:
        key = independence_key(d)
        if key not in best:
            best[key] = d
            order.append(key)
        elif tier(d) > tier(best[key]):
            best[key] = d
    deduped = [best[k] for k in order]

    # Версии одной работы — это СВЯЗНЫЕ КОМПОНЕНТЫ отношения тождества, а не цепочка сравнений
    # с представителем. Препринт, версия с подзаголовком и версия с иным переименованием могут быть
    # связаны каждая через препринт, но не напрямую друг с другом. Любой однопроходный вариант даёт
    # тогда результат, зависящий от порядка документов; компоненты связности — нет.
    n = len(deduped)
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def family_of(d) -> str:
        return getattr(d, "source_family", "") or "other"

    for i in range(n):
        if family_of(deduped[i]) not in WORK_FAMILIES:
            continue
        for j in range(i + 1, n):
            if family_of(deduped[j]) != family_of(deduped[i]):
                continue
            if root(i) != root(j) and same_scientific_work(deduped[i], deduped[j]):
                parent[root(j)] = root(i)

    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(root(i), []).append(i)
    # Порядок — по первому появлению компоненты; представитель — версия с лучшим разрядом доверия,
    # при равенстве самая ранняя. Обе величины детерминированы, поэтому результат воспроизводим.
    def best(members: list[int]) -> int:
        return max(members, key=lambda i: (tier(deduped[i]), -i))

    return [deduped[best(members)]
            for _, members in sorted(components.items(), key=lambda kv: min(kv[1]))]


def same_work_merges(documents) -> int:
    """Сколько документов оказались другой версией уже учтённой научной работы. Диагностика P1-A."""
    docs = list(documents or [])
    by_key = {independence_key(d): d for d in docs}
    return max(0, len(by_key) - len(independent_documents(docs)))


def syndication_rate(documents) -> float:
    """Доля документов, являющихся перепечаткой уже встреченного текста. 0.0 — перепечаток нет."""
    docs = [d for d in (documents or []) if getattr(d, "title", "")]
    if len(docs) < 2:
        return 0.0
    keys = [syndication_key(d) for d in docs]
    return round(1 - len(set(keys)) / len(keys), 3)


def origin_concentration(documents) -> float:
    """Доля документов из самого частого происхождения: 1.0 — всё из одного места."""
    docs = list(documents or [])
    if not docs:
        return 0.0
    counts: dict[str, int] = {}
    for d in docs:
        key = origin(d)
        counts[key] = counts.get(key, 0) + 1
    return round(max(counts.values()) / len(docs), 3)
