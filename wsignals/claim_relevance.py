"""Связь документа с КАНДИДАТОМ: документ подтверждает технологию, а не совпадает по словам.

Контракт V3 — ФРАЗОВЫЙ. Предыдущие версии строили личность из МНОЖЕСТВА токенов, и независимый аудит
показал, что на множествах безопасного правила не получается:

  * отсев общей лексики сжимал «ai security» до «security», и работа о продовольственной
    безопасности закрывала кандидата;
  * возврат общей лексики (лесенка V2) чинил это, но порождал ложное подтверждение
    «общее слово × существительное»: «Food security forecasting with AI» содержит и «ai», и
    «security» — в разных местах заголовка — и снова засчитывалось;
  * поглощённый склейкой псевдоним наследовал авторитет без доказанной равнозначности, поэтому
    «mcp», «agent» и даже «customer relationship management» подтверждали чужие технологии.

Поэтому множества как источник истины удалены. Правило теперь такое.

  A. АВТОРИТЕТНЫЕ ВАРИАНТЫ. Безусловно авторитетно только ИСХОДНОЕ название кандидата — то, по
     которому он найден в источниках (`original_phrase`, `phrase`). Нормализация детерминированная
     и ограниченная: регистр, Юникод, пунктуация и пробелы, число (единственное/множественное).
     Ни лесенки, ни возврата общей лексики, ни пересечения множеств.

     Каноническое название (`canonical_label`) — это СВОБОДНОЕ переписывание моделью. Для показа,
     канонической подачи и кластеризации оно годится, для истины доказательства — только если
     детерминированно равнозначно исходному названию. Измеренная атака: исходное «quantum key
     distribution», канон «homomorphic encryption» — работа и репозиторий про гомоморфное шифрование
     давали «наука + код» и открывали ACCEPT другой технологии. Неравнозначный канон помечается
     NON_AUTHORITATIVE_FOR_CLAIM_EVIDENCE и остаётся в кандидате только для показа.

  B. РАВНОЗНАЧНОСТЬ ПСЕВДОНИМА. Попадание в `merged_from` равнозначности НЕ означает. Псевдоним
     становится авторитетным только по детерминированному правилу: то же мультимножество токенов
     при ином порядке («AML KYC» ↔ «KYC AML»), либо отличие только числом или пунктуацией.
     Сокращение («mcp» при «model context protocol») требует ЯВНОГО происхождения аббревиатуры,
     установленного ДО сопоставления; такого источника в системе нет, поэтому сокращение остаётся
     происхождением поиска и подтверждения не даёт. Одинаковое каноническое имя от модели, общий
     кластер склейки, непересекающиеся множества и подмножества равнозначностью не являются.

  C. ФРАЗОВОЕ СОВПАДЕНИЕ. Заголовок должен содержать авторитетную фразу ИМЕННО КАК ФРАЗУ —
     подряд и в том же порядке, — а не все её слова где-то в заголовке. Совпадение не может
     начинаться или заканчиваться внутри дефисного сращения: «Cutting-edge caching» не называет
     «edge caching». Аннотация по-прежнему не участвует.

  D. ПРОИСХОЖДЕНИЕ ПРИВЯЗКИ. Документ, привязанный через поглощённый вариант
     (`attachment_source == "merged"`), подтверждает только если сама фраза привязки авторитетна
     по правилу B. Иначе он остаётся видимым свидетельством поиска, но в подтверждение не идёт.

Точность предпочтена полноте: `LOW_EVIDENCE` — приемлемый исход, ложное конкретное подтверждение —
нет. Проверка НЕ удаляет документ: он остаётся в карточке и в происхождении, участвует в счётчиках
перепечаток и концентрации, но не засчитывается в ПОДТВЕРЖДЕНИЕ. Неизвестная релевантность
подтверждением не является (fail-closed).
"""
from __future__ import annotations

import re

# Служебные слова: собственного содержания не несут. Используются ТОЛЬКО чтобы отличить название от
# пустой фразы; в авторитетность варианта они больше не вмешиваются.
STOPWORDS = frozenset("""
a an the of for and or in on at to with without from by via as is are be being been
its their our your his her this that these those it they we you he she
using used use uses toward towards into onto over under between among across per
""".split())

# Состояния личности и сопоставления.
IDENTITY_NOT_DEFINED = "IDENTITY_NOT_DEFINED"
INSUFFICIENT_IDENTITY_SPECIFICITY = "INSUFFICIENT_IDENTITY_SPECIFICITY"
ALIAS_NOT_PROVEN_EQUIVALENT = "ALIAS_NOT_PROVEN_EQUIVALENT"
NON_AUTHORITATIVE_FOR_CLAIM_EVIDENCE = "NON_AUTHORITATIVE_FOR_CLAIM_EVIDENCE"
ATTACHMENT_NOT_AUTHORITATIVE = "ATTACHMENT_NOT_AUTHORITATIVE"
CROSS_LANGUAGE_UNVERIFIED = "CROSS_LANGUAGE_UNVERIFIED"

_WORD = re.compile(r"\w+", re.UNICODE)
_LATIN = re.compile(r"[a-z0-9]")
# Дефис и неразрывный дефис СРАЩИВАЮТ слова. Тире (–, —) — разделитель, а не сращение.
_HYPHENS = frozenset("-\u2011")


def _stem(token: str) -> str:
    """Минимальная детерминированная нормализация числа. Никакой морфологии сверх этого."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokens(text: str) -> list[str]:
    """Нормализованные токены строки. Юникод-безопасно: кириллица токенизируется наравне с латиницей."""
    return [_stem(m.group(0)) for m in _WORD.finditer((text or "").lower())]


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def token_sequence(text: str) -> tuple[list[str], list[bool]]:
    """Токены и признак «сращён дефисом с предыдущим».

    Признак нужен фразовому совпадению: «Cutting-edge caching» не называет «edge caching», потому
    что «edge» здесь — часть сращения «cutting-edge», а не начало самостоятельной фразы."""
    low = (text or "").lower()
    out, bound, previous_end = [], [], None
    for m in _WORD.finditer(low):
        gap = low[previous_end:m.start()] if previous_end is not None else ""
        out.append(_stem(m.group(0)))
        bound.append(bool(gap) and all(ch in _HYPHENS for ch in gap))
        previous_end = m.end()
    return out, bound


def phrase_occurs(phrase: list[str], title_tokens: list[str], title_bound: list[bool]) -> bool:
    """Встречается ли фраза в заголовке ИМЕННО КАК ФРАЗА: подряд, в том же порядке и не внутри
    дефисного сращения с любого края."""
    n = len(phrase)
    if not n or n > len(title_tokens):
        return False
    for i in range(len(title_tokens) - n + 1):
        if title_tokens[i:i + n] != phrase:
            continue
        if title_bound[i]:                                   # начало внутри сращения
            continue
        after = i + n
        if after < len(title_tokens) and title_bound[after]:  # конец внутри сращения
            continue
        return True
    return False


def identity_tokens(phrase: str) -> tuple[list[str], str | None]:
    """Токены авторитетного варианта названия и состояние, если вариант непригоден.

    Никакого отсева общей лексики: берётся название КАК ЕСТЬ, нормализованное по регистру, Юникоду,
    пунктуации и числу. Непригоден только вариант без слов или целиком из служебных слов."""
    toks = tokens(phrase)
    if not toks:
        return [], IDENTITY_NOT_DEFINED
    if all(t in _STOP_STEMS for t in toks):
        return [], INSUFFICIENT_IDENTITY_SPECIFICITY
    return toks, None


_STOP_STEMS = frozenset(_stem(w) for w in STOPWORDS)


def equivalent(a: str, b: str) -> bool:
    """Детерминированная равнозначность двух названий (правило B).

    Совпадает мультимножество нормализованных токенов — значит различие только в порядке, числе или
    пунктуации. Ничего сверх этого: ни сокращений без явного происхождения, ни подмножеств, ни
    «модель назвала их одинаково»."""
    ta, tb = tokens(a), tokens(b)
    return bool(ta) and sorted(ta) == sorted(tb)


def _originating_names(candidate) -> list[str]:
    """ИСХОДНЫЕ названия кандидата: то, по чему он был извлечён из корпуса.

    Это единственная безусловная опора подтверждения: фраза пришла из самих источников, а не из
    переписывания моделью."""
    if isinstance(candidate, str):
        return [candidate]
    if candidate is None:
        return []
    get = candidate.get if isinstance(candidate, dict) else (lambda k, d=None: getattr(candidate, k, d))
    return [get("original_phrase"), get("phrase")]


def _canonical_name(candidate):
    if isinstance(candidate, str) or candidate is None:
        return None
    get = candidate.get if isinstance(candidate, dict) else (lambda k, d=None: getattr(candidate, k, d))
    return get("canonical_label")


def _own_names(candidate) -> list[str]:
    """Все собственные названия: исходные плюс канон. Авторитетность канона решается отдельно."""
    canonical = _canonical_name(candidate)
    return _originating_names(candidate) + ([canonical] if canonical else [])


def _merged_names(candidate) -> list[str]:
    if isinstance(candidate, str) or candidate is None:
        return []
    get = candidate.get if isinstance(candidate, dict) else (lambda k, d=None: getattr(candidate, k, d))
    return list(get("merged_from") or [])


def identity_report(candidate) -> list[tuple[str, list[str], str | None]]:
    """Все варианты названия с состоянием каждого.

    Собственные названия авторитетны по умолчанию. Поглощённый псевдоним — только если он
    детерминированно равнозначен какому-нибудь собственному названию."""
    out, seen = [], set()

    def add(name, toks, state):
        key = (name.strip().lower(), tuple(toks), state)
        if key in seen:
            return
        seen.add(key)
        out.append((name.strip(), toks, state))

    # Исходные названия авторитетны безусловно: по ним кандидат и был найден в источниках.
    own_usable = []
    for name in _originating_names(candidate):
        if not isinstance(name, str) or not name.strip():
            continue
        toks, state = identity_tokens(name)
        if state is None:
            own_usable.append(name)
        add(name, toks, state)

    # Канон — это СВОБОДНОЕ переписывание моделью. Для показа и кластеризации он годится, для
    # ИСТИНЫ ДОКАЗАТЕЛЬСТВА — только если детерминированно равнозначен исходному названию.
    #
    # Измеренная атака: original_phrase «quantum key distribution», canonical_label «homomorphic
    # encryption». Работа и репозиторий про гомоморфное шифрование давали «наука + код» и открывали
    # ACCEPT совершенно другой технологии. Семантическое переименование подтверждением быть не может:
    # LOW_EVIDENCE предпочтительнее подтверждения через переименование.
    canonical = _canonical_name(candidate)
    if isinstance(canonical, str) and canonical.strip():
        toks, state = identity_tokens(canonical)
        if state is None and not any(equivalent(canonical, own) for own in own_usable):
            toks, state = [], NON_AUTHORITATIVE_FOR_CLAIM_EVIDENCE
        if state is None:
            own_usable.append(canonical)
        add(canonical, toks, state)
    for name in _merged_names(candidate):
        if not isinstance(name, str) or not name.strip():
            continue
        toks, state = identity_tokens(name)
        if state is None and not any(equivalent(name, own) for own in own_usable):
            toks, state = [], ALIAS_NOT_PROVEN_EQUIVALENT
        add(name, toks, state)
    return out


def identity_variants(candidate) -> list[tuple[str, list[str]]]:
    """Только авторитетные варианты."""
    return [(name, toks) for name, toks, state in identity_report(candidate) if state is None]


def authoritative_attachment(doc, candidate) -> bool:
    """Правило D: свидетельство, перенесённое склейкой, подтверждает только если сама фраза
    привязки авторитетна для выжившего кандидата."""
    if getattr(doc, "attachment_source", "") != "merged":
        return True
    phrase = getattr(doc, "attachment_phrase", "") or ""
    if not phrase.strip():
        return False
    authoritative = [n for n, _, st in identity_report(candidate) if st is None]
    return any(equivalent(phrase, name) for name in authoritative)


def document_identity_text(doc) -> str:
    """Текст, по которому судят, ЧЕМУ посвящён документ.

    Только заголовок. У репозитория заголовок собран как «имя: описание», поэтому описание входит
    сюда естественным образом. Аннотация намеренно не участвует: см. модульную докстроку."""
    return getattr(doc, "title", "") or ""


def _has_latin(items) -> bool:
    return any(_LATIN.search(t) for t in items)


def match(doc, candidate) -> tuple[bool, str, str]:
    """(релевантен, вариант названия, причина). Fail-closed: не установили — значит нет."""
    report = identity_report(candidate)
    variants = [(n, t) for n, t, st in report if st is None]
    if not report:
        return False, "", "личность кандидата не задана: релевантность документа не проверяема"
    if not variants:
        states = sorted({st for _, _, st in report if st})
        return False, "", (f"{states[0] if states else IDENTITY_NOT_DEFINED}: "
                           "у кандидата нет авторитетного варианта названия")

    # Прямое называние проверяется ПЕРВЫМ, до правила происхождения привязки.
    #
    # Измерено на владельческих живых артефактах: из 114 отклонённых документов 110 отклонены
    # правилом привязки и лишь 4 — самим фразовым правилом; у 28 из этих 110 заголовок НЕЗАВИСИМО
    # называет выжившего кандидата и проходит фразовый контракт как есть. Это ложноотрицательные:
    # документ про технологию кандидата отбрасывался только потому, что в список кандидата он попал
    # через поглощённый вариант другой ширины.
    #
    # Собственный заголовок документа — свидетельство более прямое, чем маршрут, которым документ
    # был перенесён склейкой. Поэтому: если заголовок сам проходит СУЩЕСТВУЮЩЕЕ фразовое правило
    # против авторитетного варианта кандидата, происхождение привязки его не вето́рует.
    #
    # Обратное неверно и здесь не вводится: привязка сама по себе релевантности НЕ даёт. Ни одна
    # ветка ниже не признаёт документ подтверждающим без прямого называния, а при отсутствии
    # прямого совпадения правило привязки действует ровно как раньше. Пороги, токенизация,
    # равнозначность и состав авторитетных вариантов не тронуты.
    title = document_identity_text(doc)
    title_tokens, title_bound = token_sequence(title) if title.strip() else ([], [])

    for name, toks in variants:
        if phrase_occurs(toks, title_tokens, title_bound):
            return True, name, f"заголовок документа называет «{name}»"
        # Слитное написание того же названия: «vibecoding» это «vibe coding» без пробела.
        # Только для многословных: у однословного правило выродилось бы в поиск подстроки.
        if len(toks) >= 2 and "".join(toks) in title_tokens:
            return True, name, f"заголовок документа называет «{name}» слитным написанием"

    # Прямого называния нет — защита происхождения привязки действует без изменений.
    if not authoritative_attachment(doc, candidate):
        return False, "", (f"{ATTACHMENT_NOT_AUTHORITATIVE}: документ привязан через поглощённый "
                           f"вариант «{getattr(doc, 'attachment_phrase', '')}», который не является "
                           "доказанно равнозначным выжившему названию")

    if not title.strip():
        return False, "", "у документа нет заголовка: отнести его к кандидату нечем"
    if not title_tokens:
        return False, "", "заголовок документа не содержит слов: отнести его к кандидату нечем"

    present = set(title_tokens)
    if _has_latin([t for _, toks in variants for t in toks]) and not _has_latin(present):
        return False, "", (f"{CROSS_LANGUAGE_UNVERIFIED}: заголовок документа на другом языке, "
                           "а детерминированного псевдонима кандидата на этом языке нет; "
                           "подтверждением не считается")
    if _has_latin(present) and not _has_latin([t for _, toks in variants for t in toks]):
        return False, "", (f"{CROSS_LANGUAGE_UNVERIFIED}: название кандидата на другом языке, "
                           "а детерминированного псевдонима на языке документа нет; "
                           "подтверждением не считается")
    return False, "", ("заголовок документа не называет технологию кандидата: "
                       f"фразы {', '.join(repr(n) for n, _ in variants[:3])} в нём нет")


def annotate(documents, candidate) -> list:
    """Проставляет каждому документу признак претензионной релевантности и ЧЕМ она установлена.

    Вызывается ПОСЛЕ склейки, по выжившей канонической личности: документ, пришедший вместе с
    поглощённым вариантом, заново проверяется против того кандидата, которому теперь приписан."""
    for doc in documents or []:
        ok, name, reason = match(doc, candidate)
        try:
            doc.claim_relevant = ok
            doc.claim_match = name
            doc.claim_relevance_reason = reason
        except AttributeError:                       # не наш тип документа: молча не размечаем
            continue
    return list(documents or [])


def is_relevant(doc) -> bool:
    """Строгое прочтение признака. None («не размечен») подтверждением НЕ является."""
    return getattr(doc, "claim_relevant", None) is True
