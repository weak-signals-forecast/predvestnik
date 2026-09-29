"""Этап 2 на очищенном пуле: запрос → одобренные моделью технологии → ТОП-15. На запросе модель не вызывается.

    python -m registry.stage2_pool       # -> <registry>/stage2_pool_top15.csv и сводка

Пул: верхние POOL кандидатов реестра по обученному баллу (после фильтра общих оборотов), из них — только
одобренные точечной проверкой YandexGPT (llm_assess.jsonl: конкретная технология, не продукт, не общая рубрика).
Перед отбором: убираются технологии с общими русскими названиями (RU_GENERIC: кроме «ИИ», «системы»,
«передовой» и т. п. ничего нет) и дубли по русскому названию (остаётся лучшая по баллу).
Запрос очищается от слов-намерений функцией команды wsignals.intent.split («перспективные технологии в финтехе»
→ «финтехе»: иначе «перспективные» притягивает «передовые системы ИИ»).
Отбор по теме: эмбеддинги USER-bge-m3 темы запроса против «русское название — описание» из проверки
(русский запрос к русскому тексту; расширение запроса моделью не нужно). Берутся TOPIC ближайших.
ТОП-15 четырьмя способами (для сравнения вклада шагов):
  ru_rel      — только близость к запросу;
  ru_learned  — обученный балл сигнальности среди TOPIC ближайших;
  ru_blend    — произведение процентилей близости и балла;
  ru_tag      — обученный балл среди ближайших с меткой нужной области от модели;
  ru_mix      — обученный балл среди TOPIC лучших по смеси «смысл запроса + доля работ в рубриках области»
                (технология должна либо публиковаться в рубриках области, либо иметь метку области от модели).
Разметка выдачи — колонка weak_signal (правило шага 9), метки переносятся по (область, фраза) из прошлых замеров.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from .corpus import REGISTRY
from .semantic import LABELS, _key
from .stage2 import AREAS, GENERIC_END, GENERIC_START, area_of_signals

REG = REGISTRY
POOL = 20000
PER_AREA = 3000          # топ внутри рубрик каждой области (registry/pool.py)
TOPIC = 300
MODEL = "deepvk/USER-bge-m3"
# Общие слова русских названий: если кроме них в названии ничего нет, это не технология
# («Программное обеспечение с поддержкой ИИ», «Передовые системы ИИ», «Генеративный ИИ»). Сравнение по началу слова.
RU_GENERIC = ("ии", "искусствен", "интеллект", "систем", "инструмент", "программн", "обеспечен", "разработк",
              "платформ", "технолог", "решени", "передов", "генеративн", "интеграц", "помощ", "применен",
              "поддержк", "основе", "использован", "с", "на", "в", "для", "и", "нов", "современ", "интеллектуальн",
              "умн", "цифров", "продвинут", "перспективн", "ai")


def ru_specific(name: str) -> bool:
    """В русском названии есть хотя бы одно содержательное слово (слова режутся и по дефису: «ИИ-агентов»)."""
    import re
    def generic(t: str) -> bool:
        return t in RU_GENERIC or any(t.startswith(g) for g in RU_GENERIC if len(g) > 2)
    return any(not generic(t) for t in re.findall(r"[a-zа-яё0-9]+", name.lower().replace("ё", "е")))


def ru_norm(name: str) -> str:
    import re
    return " ".join(sorted(re.findall(r"[a-zа-яё0-9]+", name.lower().replace("ё", "е"))))


def main() -> int:
    from sentence_transformers import SentenceTransformer
    import torch
    from . import pool as pool_mod
    pool = pool_mod.select(REG, POOL, PER_AREA).copy()
    pool["ph"] = pool.phrase.str.replace("_", " ")
    llm = {}
    for l in (REG / "llm_assess.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            x = json.loads(l)
            llm[x["phrase"]] = x
    pool = pool[pool.ph.map(lambda p: p in llm and llm[p]["keep"])].reset_index(drop=True)
    pool["name_ru"] = pool.ph.map(lambda p: llm[p]["name_ru"])
    pool["description_ru"] = pool.ph.map(lambda p: llm[p]["description_ru"])
    pool["areas"] = pool.ph.map(lambda p: llm[p]["areas"])
    shares = pool_mod.area_shares(REG, pool.reset_index(drop=True))
    for area in pool_mod.AREA_GROUPS_POOL:
        pool["share:" + area] = shares[area].values
    n0 = len(pool)
    spec = pool.name_ru.map(ru_specific)
    pool = pool[spec]
    n1 = len(pool)
    # дубли по русскому названию (регистр, «ё», порядок слов не важны): остаётся запись с лучшим баллом
    pool = pool.assign(rkey=pool.name_ru.map(ru_norm)).sort_values("learned", ascending=False)
    pool = pool.drop_duplicates("rkey").reset_index(drop=True)
    print(f"одобренных моделью в пуле: {n0:,}; общих русских названий убрано: {n0 - n1:,}; "
          f"дублей по названию: {n1 - len(pool):,}; осталось {len(pool):,}".replace(",", " "), flush=True)

    m = SentenceTransformer(MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")
    m.max_seq_length = 64
    texts = (pool.name_ru + " — " + pool.description_ru).tolist()
    E = m.encode(texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    lab = pd.read_csv(LABELS)
    lab = lab[lab.same >= 0.5]
    key_sig = {}
    for _, x in lab.iterrows():
        key_sig.setdefault(_key(x.candidate), []).append(int(x.id))
    sig_area = area_of_signals()

    out, summary = [], []
    for area, (ru, _en) in AREAS.items():
        from wsignals import intent
        topic = intent.split(ru).domain            # тема без слов-намерений: «перспективные технологии в финтехе» → «финтехе»
        q = m.encode([topic], normalize_embeddings=True, convert_to_numpy=True)[0]
        allp = pool.assign(relevance=E @ q)
        # смесь: смысл запроса + доля работ в рубриках области (из корпуса, от формулировки не зависит)
        allp["mix"] = 0.5 * allp.relevance.rank(pct=True) + 0.5 * allp["share:" + area].rank(pct=True)
        mix = allp[(allp["share:" + area] >= pool_mod.MIN_AREA_SHARE) | allp.areas.map(lambda a: area in a)]
        mix = mix.nlargest(TOPIC, "mix").copy()
        t = allp.nlargest(TOPIC, "relevance").copy()
        t["ru_rel"] = t.relevance
        t["ru_learned"] = t.learned
        t["ru_blend"] = t.learned.rank(pct=True) * t.relevance.rank(pct=True)
        t["pool_pct"] = t.learned.rank(pct=True)           # сила сигнала внутри темы — для карточки
        area_sigs = {i for i, a in sig_area.items() if a == area}
        mix["pool_pct"] = mix.learned.rank(pct=True)
        for method in ("ru_rel", "ru_learned", "ru_blend", "ru_tag", "ru_mix"):
            src = (t[t.areas.map(lambda a: area in a)] if method == "ru_tag" else mix if method == "ru_mix" else t)
            top = src.nlargest(15, "learned" if method in ("ru_tag", "ru_mix") else method).copy()
            hits = {s for k in top.key for s in key_sig.get(k, []) if s in area_sigs}
            summary.append({"area": area, "method": method, "org_signals": len(hits), "n": len(top)})
            top["area"], top["method"], top["pos"] = area, method, range(1, len(top) + 1)
            top["org_signal"] = [";".join(str(s) for s in key_sig.get(k, []) if s in area_sigs) for k in top.key]
            out.append(top[["area", "method", "pos", "phrase", "name_ru", "description_ru", "relevance", "learned",
                            "pool_pct", "org_signal"]])
    res = pd.concat(out, ignore_index=True)
    labels = {}
    for f in ("stage2_top15.csv", "stage2_pool_top15.csv"):
        p = REG / f
        if p.exists():
            prev = pd.read_csv(p)
            if "weak_signal" in prev:
                for _, x in prev.dropna(subset=["weak_signal"]).iterrows():
                    labels[(x.area, x.phrase)] = x.weak_signal
    res["weak_signal"] = [labels.get((a, p), np.nan) for a, p in zip(res.area, res.phrase)]
    res.to_csv(REG / "stage2_pool_top15.csv", index=False)
    s = pd.DataFrame(summary)
    print("\nСигналы организаторов своей области в ТОП-15:")
    print(s.pivot_table(index="area", columns="method", values="org_signals").to_string())
    print("\nвсего:", s.groupby("method").org_signals.sum().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
