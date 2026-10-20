"""Открытая модель на GPU вместо API YandexGPT: промпты из набора для сервиса → файл → ответы обратно в набор.

    REGISTRY_BUNDLE=artifacts/registry_v5 python -m registry.llm_offline prompts --out prompts.jsonl
    REGISTRY_BUNDLE=artifacts/registry_v5 python -m registry.llm_offline import --answers answers.jsonl

Зачем. Мини-набор (registry.mini_bundle) собран без ключа: название — английская фраза, описания нет, LLM-оси
не посчитаны. Те же два вопроса, что задают registry.llm_assess (смысловая проверка: технология ли, русское
название и описание) и registry.llm_axes (родитель, механизм с дословной цитатой, тип, абстракт), задаются
открытой моделью на GPU (gpu/llm_local.py). Промпты и проверка ответов — те же функции
модулей, поэтому ответ не по схеме или без дословного повтора фразы отбрасывается так же, как ответ YandexGPT.

prompts: по LLM_BATCH_ASSESS записей в промпт проверки, по LLM_BATCH_AXES — в промпт осей; заголовки источников
из карточек набора. import: русские названия и описания — в signals.parquet и cards.json.gz (карточка перестаёт
быть unverified только при принятом ответе), векторы названий emb.npy пересчитываются, оси — в
cache/llm_axes.jsonl, флаги «не технология» — в cache/llm_assess.jsonl; поле model в записях кэша = имя модели.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import llm_assess, llm_axes

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.getenv("REGISTRY_BUNDLE", ROOT / "artifacts" / "registry"))
B_ASSESS = int(os.getenv("LLM_BATCH_ASSESS", "5"))
B_AXES = int(os.getenv("LLM_BATCH_AXES", "3"))
MODEL = os.getenv("LLM_OFFLINE_MODEL", "Qwen2.5-7B-Instruct")


def _load(bundle: Path | None = None):
    bundle = bundle or BUNDLE
    sig = pd.read_parquet(bundle / "signals.parquet")
    with gzip.open(bundle / "cards.json.gz", "rt", encoding="utf-8") as fh:
        cards = json.load(fh)
    return sig, cards


def _titles(card: dict, n: int) -> list[str]:
    return [(d.get("title") or "").strip()[:160] for d in (card.get("sources") or []) if d.get("title")][:n]


def _axes_items(sig: pd.DataFrame, cards: dict) -> list[dict]:
    out = []
    for r in sig.itertuples():
        c = cards.get(r.key) or {}
        src = [{"title": (d.get("title") or "")[:180], "date": str(d.get("published") or "")[:10],
                "type": d.get("source_type") or d.get("source_name") or ""} for d in (c.get("sources") or []) if d.get("title")][:6]
        out.append({"key": r.key, "phrase": r.phrase.replace("_", " "), "name_ru": r.name_ru,
                    "description_ru": r.description_ru or "", "meas": getattr(r, "meas", "") or "", "sources": src})
    return out


# Промпт проверки писался под ИИ: «технология» — метод, устройство, архитектура или протокол. Для физики, материалов,
# энергетики и биотеха так выпадают материалы и эффекты («альтермагнитные сверхпроводники» → «не технология»,
# тестовый прогон 20.10). Здесь определение расширено; остальной промпт и проверка ответа — как в llm_assess.
TECH_OLD = "is_technology = true только для конкретного метода, устройства, архитектуры или протокола."
TECH_NEW = ("is_technology = true для конкретного метода, устройства, архитектуры, протокола, материала или класса "
            "материалов, физического или биологического эффекта, на котором строится технология.")


def assess_prompt() -> str:
    assert TECH_OLD in llm_assess.PROMPT
    return llm_assess.PROMPT.replace(TECH_OLD, TECH_NEW)


def prompts(out: Path, only_missing: bool = False) -> None:
    sig, cards = _load()
    if only_missing:                     # только записи без ответа в кэше набора (новые после пересборки)
        done = set()
        p = BUNDLE / "cache" / "llm_assess.jsonl"
        if p.exists():
            done = {json.loads(l)["phrase"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
        sig = sig[~sig.phrase.isin(done)].reset_index(drop=True)
    pre = "d-" if only_missing else ""     # другие id: ответы обоих прогонов можно склеить в один файл
    rows = []
    keys = list(sig.key)
    for s in range(0, len(sig), B_ASSESS):
        part = sig.iloc[s:s + B_ASSESS]
        listing = "\n".join(f"{i}. {ph}" + "".join(f"\n   — {t}" for t in _titles(cards.get(k) or {}, llm_assess.TITLES))
                            for i, (k, ph) in enumerate(zip(part.key, part.phrase), 1))
        rows.append({"id": f"{pre}as:{s}", "keys": list(part.key), "prompt": assess_prompt().replace("{items}", listing),
                     "max_new": 180 * len(part)})
    items = _axes_items(sig, cards)
    for s in range(0, len(items), B_AXES):
        part = items[s:s + B_AXES]
        rows.append({"id": f"{pre}ax:{s}", "keys": [it["key"] for it in part], "prompt": llm_axes.prompt_for(part),
                     "max_new": 450 * len(part)})
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_as = sum(1 for r in rows if "as:" in r["id"])
    print(f"записей {len(keys)}; промптов {len(rows)} (проверка {n_as}, оси {len(rows) - n_as}) -> {out}")


_CJK = re.compile(r"[^\W\d_A-Za-zА-Яа-яЁёΑ-Ωα-ω]")   # буква не латиницы, не кириллицы и не греческого (β-Ga2O3)
_GLUED = re.compile(r"[а-яё][A-Za-z]|[A-Za-z][а-яё]", re.I)


# служебные слова испанского, португальского, французского, немецкого, индонезийского: новые подобласти OpenAlex
# приносят неанглийские работы, их фразы — не названия технологий («sistem penerimaan murid baru»)
_FOREIGN = set("""en la el los las del para con por una que mediante tiempos desaf vac investigaci educaci
do da das dos em com ao na comportamento gest une des les du et pour avec dans sur strat und mit von zur eine
dan yang pada dalam di untuk terhadap pengaruh sebagai berbasis menggunakan dengan ke dari meningkatkan sistem
penerimaan murid baru magang hijau solusi energi terbarukan efisiensi operasional ekstrak""".split())
_FOREIGN_STRONG = {"berbasis", "untuk", "pengaruh", "terhadap", "meningkatkan", "sebagai", "dengan", "mediante", "para"}


def foreign_phrase(phrase: str) -> bool:
    w = phrase.replace("_", " ").split()
    hits = [x for x in w if x in _FOREIGN]
    return len(hits) >= 2 or any(x in _FOREIGN_STRONG for x in w)


# обрывок заголовка со сказуемым: «doping unlocks», «topology governs», «regulation stabilizes» — в химических и
# материаловедческих заголовках часто пишут «X stabilizes Y». Существительные на -s («triggers», «transforms»,
# «controls», «drives» в смысле накопителей редки в наборе) в список не входят, кроме drives.
_VERB = re.compile(r"^(stabiliz|regulat|enabl|boost|unlock|reveal|driv|govern|promot|improv|enhanc|achiev|induc|suppress|"
                   r"mediat|facilitat|accelerat|dictat|determin|modulat|activat|inhibit|empower|redefin|outperform|"
                   r"decoupl|steer)(es|s)$")


def verb_fragment(phrase: str) -> bool:
    return any(_VERB.match(x) for x in phrase.replace("_", " ").split())


def prune(bundle: Path | None = None) -> None:
    """Убрать из набора неанглийские фразы и обрывки со сказуемым (signals, emb, cards, axes — согласованно)."""
    global BUNDLE
    if bundle:
        BUNDLE = bundle
    sig, cards = _load()
    drop = {k for k, p in zip(sig.key, sig.phrase) if foreign_phrase(p) or verb_fragment(p)}
    keep = [i for i, k in enumerate(sig.key) if k not in drop]
    E = np.load(BUNDLE / "emb.npy")
    sig.iloc[keep].reset_index(drop=True).to_parquet(BUNDLE / "signals.parquet")
    np.save(BUNDLE / "emb.npy", E[keep])
    with gzip.open(BUNDLE / "cards.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({k: c for k, c in cards.items() if k not in drop}, fh, ensure_ascii=False)
    ax = pd.read_parquet(BUNDLE / "axes.parquet")
    ax[~ax.key.isin(drop)].reset_index(drop=True).to_parquet(BUNDLE / "axes.parquet")
    print(f"убрано {len(drop)}: осталось {len(keep)}")


def clean_name(s: str) -> str:
    """Русское название без иероглифов и склеек латиницы с кириллицей («КаузальныйStudent»); иначе пусто."""
    s = (s or "").strip()
    if not s or _CJK.search(s) or _GLUED.search(s) or not re.search(r"[а-яё]", s, re.I):
        return ""
    return s


def clean_text(s: str) -> str:
    s = (s or "").strip()
    return "" if _CJK.search(s) else s


def _json_array(raw: str):
    m = re.search(r"\[.*\]", raw or "", re.S)
    try:
        return json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []


def import_answers(answers: Path, prompts_file: Path, reembed: bool, ref: Path | None = None) -> None:
    """ref — набор, по которому строились промпты (после пересборки базы часть записей ушла, у части изменилось
    написание фразы): ответы проверяются по фразам и источникам ref, в набор идут только записи, которые в нём есть."""
    sig, cards = _load()
    pr = {r["id"]: r for r in (json.loads(l) for l in prompts_file.read_text(encoding="utf-8").splitlines() if l.strip())}
    ph = dict(zip(sig.key, sig.phrase))
    items = {it["key"]: it for it in _axes_items(sig, cards)}
    ph_ref, items_ref = dict(ph), dict(items)
    if ref:
        rs, rc = _load(ref)
        ph_ref.update(dict(zip(rs.key, rs.phrase)))
        items_ref.update({it["key"]: it for it in _axes_items(rs, rc)})
    assess, axes, bad = {}, {}, {"as": 0, "ax": 0}
    for l in answers.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        a = json.loads(l)
        p = pr.get(a["id"])
        if not p:
            continue
        keys = p["keys"]
        if any(k not in ph_ref for k in keys):  # промпт от неизвестной версии набора
            continue
        if "as:" in a["id"]:
            rows = _json_array(a["raw"])
            ok = llm_assess._validate(rows, len(keys), [ph_ref[k] for k in keys])
            # ответ «не технология» без русского названия _validate отбрасывает (название обязательно), но сам вердикт
            # нужен для отсева: принимаем его с пустым названием, если фраза повторена дословно и флаги — булевы
            for x in rows if isinstance(rows, list) else []:
                j = x.get("i") if isinstance(x, dict) else None
                if not isinstance(j, int) or isinstance(j, bool) or not 1 <= j <= len(keys) or (j - 1) in ok:
                    continue
                flags = [x.get("is_technology"), x.get("is_product_or_brand"), x.get("is_generic_phrase")]
                if (all(isinstance(f, bool) for f in flags) and not (flags[0] and not flags[1] and not flags[2])
                        and llm_assess._norm(x.get("phrase")) == llm_assess._norm(ph_ref[keys[j - 1]])):
                    ok[j - 1] = {"is_technology": flags[0], "is_product_or_brand": flags[1], "is_generic_phrase": flags[2],
                                 "name_ru": "", "description_ru": "", "areas": [], "keep": False,
                                 "reject_reason": str(x.get("reject_reason") or "не конкретная технология")[:200]}
            bad["as"] += len(keys) - len(ok)
            for i, v in ok.items():
                if keys[i] in ph:
                    assess[keys[i]] = {"phrase": ph[keys[i]], **v, "model": MODEL}
        else:
            got = llm_axes.parse(a["raw"], [items_ref[k] for k in keys])
            bad["ax"] += len(keys) - len(got)
            for x in got:
                if x["key"] in ph:
                    axes[x["key"]] = {**x, "phrase": ph[x["key"]], "model": MODEL}

    named = 0
    for i, k in enumerate(sig.key):
        v = assess.get(k)
        if not v:
            continue
        name, desc = clean_name(v["name_ru"]), clean_text(v["description_ru"])
        if desc:
            sig.at[i, "description_ru"] = desc
        if name:                                     # битое название (иероглифы, склейка) — остаётся английский термин
            sig.at[i, "name_ru"] = name
            named += 1
        c = cards.get(k)
        if c is not None:
            if name:
                c["name_ru"] = name
            if desc:
                c["description"] = desc
            c["unverified"] = not v["keep"]
            if not v["keep"]:
                c["reject_reason"] = v["reject_reason"]
    # удаляется запись, только когда оба ответа явно говорят «не технология»: проверка — не технология, тип по осям —
    # рубрика или задача; рубрика по осям — сама по себе. Пустой тип — это не ответ, а его отсутствие (на тесте так
    # уходили настоящие темы физики). Одиночный вердикт «не технология» — отметка G: сервис снижает балл
    # (rank.GENERAL_PENALTY), запись опускается, но не пропадает.
    junk_kind = {"рубрика", "задача"}
    drop = {k for k, v in assess.items() if not v["keep"] and (axes.get(k) or {}).get("kind", "") in junk_kind}
    drop |= {k for k, x in axes.items() if x.get("kind") == "рубрика"}
    drop |= {k for k, p in zip(sig.key, sig.phrase) if foreign_phrase(p) or verb_fragment(p)}
    general = {k for k, v in assess.items() if not v["keep"]} - drop
    keep_idx = [i for i, k in enumerate(sig.key) if k not in drop]
    ax_p = BUNDLE / "axes.parquet"
    if drop:
        E_old = np.load(BUNDLE / "emb.npy")
        sig = sig.iloc[keep_idx].reset_index(drop=True)
        np.save(BUNDLE / "emb.npy", E_old[keep_idx])
        cards = {k: c for k, c in cards.items() if k not in drop}
    if ax_p.exists():
        ap_ = pd.read_parquet(ax_p)
        ap_ = ap_[~ap_.key.isin(drop)].reset_index(drop=True)
        ap_["verdict"] = np.where(ap_.key.isin(general), "G", None)
        ap_.to_parquet(ax_p)
    print(f"общая формулировка (G, балл снижен): {len(general)}")
    for x in axes.values():                          # стандарт по ответу модели — в блок «исключено» (rank.is_standard)
        if x.get("kind") == "стандарт или протокол":
            sig.loc[sig.key == x["key"], "kind"] = "протокол"
    print(f"убрано из набора: {len(drop)} (не технология или рубрика)")
    sig.to_parquet(BUNDLE / "signals.parquet")
    with gzip.open(BUNDLE / "cards.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(cards, fh, ensure_ascii=False)
    cache = BUNDLE / "cache"
    cache.mkdir(exist_ok=True)
    with (cache / "llm_assess.jsonl").open("w", encoding="utf-8") as fh:
        for v in assess.values():
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")
    with (cache / "llm_axes.jsonl").open("w", encoding="utf-8") as fh:
        for v in axes.values():
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")
    n = len(sig)
    print(f"записей {n}: названия и описания {named} ({named / n:.0%}), не технология {sum(not v['keep'] for v in assess.values())}; "
          f"оси {len(axes)} ({len(axes) / n:.0%}), с проверенной цитатой {sum(x['quote_ok'] for x in axes.values())}; "
          f"отброшено ответов: проверка {bad['as']}, оси {bad['ax']}")
    if reembed:
        reembed_bundle(sig, cards)


def reembed_bundle(sig: pd.DataFrame | None = None, cards: dict | None = None) -> None:
    """Векторы записей: русское название (или английская фраза) и заголовки источников (mini_bundle.item_text)."""
    from .mini_bundle import embed_items
    if sig is None:
        sig, cards = _load()
    names = [f"{n}. {d}" if isinstance(d, str) and d else n for n, d in zip(sig.name_ru, sig.description_ru)]
    E = embed_items(names, [cards.get(k) or {} for k in sig.key])
    np.save(BUNDLE / "emb.npy", E)
    print(f"векторы записей пересчитаны: {E.shape}")


NAMES_PROMPT = """Переведи на русский названия технологий из научных работ. Название — 2–6 слов, как его написал бы
русскоязычный специалист; устоявшиеся аббревиатуры и имена собственные оставь латиницей. Опирайся на заголовок работы.
Верни СТРОГО JSON-массив: [{"i": <номер>, "term": "<английский термин дословно>", "name_ru": "<русское название>"}]

{items}"""
NAMES_BATCH = 10


def names_prompts(out: Path) -> None:
    """Перевод названий записей, у которых нет русского названия (другой, более крупной моделью)."""
    sig, cards = _load()
    todo = sig[~sig.name_ru.str.contains("[а-яА-ЯёЁ]", regex=True)].reset_index(drop=True)
    rows = []
    for s in range(0, len(todo), NAMES_BATCH):
        part = todo.iloc[s:s + NAMES_BATCH]
        items = "\n".join(f"{j}. {ph.replace('_', ' ')}" + "".join(f"\n   — {t}" for t in _titles(cards.get(k) or {}, 1))
                          for j, (k, ph) in enumerate(zip(part.key, part.phrase), 1))
        rows.append({"id": f"n:{s}", "keys": list(part.key), "prompt": NAMES_PROMPT.replace("{items}", items),
                     "max_new": 40 * len(part)})
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"записей без русского названия {len(todo)}; промптов {len(rows)} -> {out}")


def names_import(answers: Path, prompts_file: Path) -> None:
    sig, cards = _load()
    pr = {r["id"]: r for r in (json.loads(l) for l in prompts_file.read_text(encoding="utf-8").splitlines() if l.strip())}
    ph = dict(zip(sig.key, sig.phrase))
    pos = {k: i for i, k in enumerate(sig.key)}
    got = {}
    for l in answers.read_text(encoding="utf-8").splitlines():
        a = json.loads(l) if l.strip() else {}
        p = pr.get(a.get("id"))
        if not p:
            continue
        for x in _json_array(a["raw"]) or []:
            j = x.get("i") if isinstance(x, dict) else None
            if not isinstance(j, int) or isinstance(j, bool) or not 1 <= j <= len(p["keys"]):
                continue
            k = p["keys"][j - 1]
            if k in ph and llm_assess._norm(x.get("term")) == llm_assess._norm(ph[k].replace("_", " ")):
                name = clean_name(str(x.get("name_ru") or ""))
                if name:
                    got[k] = name[:80]
    for k, name in got.items():
        sig.at[pos[k], "name_ru"] = name
        if k in cards:
            cards[k]["name_ru"] = name
    sig.to_parquet(BUNDLE / "signals.parquet")
    with gzip.open(BUNDLE / "cards.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(cards, fh, ensure_ascii=False)
    idx = [pos[k] for k in got]
    if idx:
        from .mini_bundle import embed_items
        E = np.load(BUNDLE / "emb.npy")
        names = [f"{sig.name_ru[i]}. {sig.description_ru[i]}" if isinstance(sig.description_ru[i], str) and sig.description_ru[i]
                 else sig.name_ru[i] for i in idx]
        E[idx] = embed_items(names, [cards.get(sig.key[i]) or {} for i in idx])
        np.save(BUNDLE / "emb.npy", E)
    print(f"русских названий добавлено {len(got)}; векторы пересчитаны у {len(idx)} записей")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prompts", "import", "reembed", "names", "names-import", "prune"])
    ap.add_argument("--out", default="prompts.jsonl")
    ap.add_argument("--answers", default="answers.jsonl")
    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--no-reembed", action="store_true")
    ap.add_argument("--ref", default="", help="import: набор, по которому строились промпты")
    ap.add_argument("--only-missing", action="store_true", help="prompts: только записи без ответа в кэше набора")
    a = ap.parse_args()
    if a.cmd == "prompts":
        prompts(Path(a.out), a.only_missing)
    elif a.cmd == "reembed":
        reembed_bundle()
    elif a.cmd == "prune":
        prune()
    elif a.cmd == "names":
        names_prompts(Path(a.out))
    elif a.cmd == "names-import":
        names_import(Path(a.answers), Path(a.prompts))
    else:
        import_answers(Path(a.answers), Path(a.prompts), not a.no_reembed, Path(a.ref) if a.ref else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
