"""Замер этапа 2: запрос по области → пул темы из реестра → ТОП-15. Протокол зафиксирован до замера.

    python -m registry.stage2            # -> <registry>/stage2_top15.csv и сводка

Запросы: шесть областей ТЗ по-русски (как написал бы член жюри) плюс прямой английский перевод названия
области. Расширение запроса вручную не делается: автор знает список организаторов, это была бы подсказка;
в продукте расширение делает YandexGPT, так что этот замер скорее занижает качество.

Пул темы: POOL кандидатов реестра с лучшей суммой двух процентилей — близости по смыслу (USER-bge-m3, русский
запрос → английский реестр; максимум по русскому запросу и английскому названию) и доли свежих работ сущности
в рубриках области (AREA_GROUPS: классификация arXiv и подобласти OpenAlex). Первая версия замера брала только
близость по смыслу — пул состоял из фраз со словом «fintech» и т. п., технологии области туда не попадали. ТОП-15 из пула четырьмя способами:
  relevance — только близость к запросу;
  hand      — ручной балл (лучший из научного и бизнес-слоя);
  learned   — обученный балл (registry/ranker.py, обучен на dev-половине);
  blend     — обученный балл × близость (произведение процентилей внутри пула).
Оценка: (1) сигналы организаторов этой области в ТОП-15 — по разметке пар («точно» / «ядро»), отдельно
test-половина, которую модель ранжирования не видела; (2) точность в первых 15 — разметка выдачи в колонке
weak_signal файла stage2_top15.csv по правилу: конкретная технология на ранней стадии (не продукт и не
версия, не общее слово или зонтичный термин, не зрелая массовая технология).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .corpus import REGISTRY
from .semantic import LABELS, _key, queries as signal_queries

REG = REGISTRY
POOL = 500
MODEL = "deepvk/USER-bge-m3"
# Рубрики областей по официальной классификации arXiv и подобластям OpenAlex (зафиксировано до замера).
# Технология относится к области, если заметная доля её свежих работ выходит в этих рубриках — как бы она
# ни называлась («tokenized money market fund» не похоже на слово «fintech», но публикуется в q-fin).
AREA_GROUPS = {
    "Edge": ("ax:cs.NI", "ax:cs.AR", "ax:cs.ET", "oa:1705", "oa:1708"),
    "Защита ИИ": ("ax:cs.CR",),
    "Индустриальный ИИ": ("ax:eess.SY", "oa:2207", "oa:2209"),
    "Инфраструктура ИИ": ("ax:cs.DC", "ax:cs.AR", "ax:cs.PF", "ax:cs.ET", "oa:1708"),
    "Роботы": ("ax:cs.RO", "oa:2207"),
    "Финтех": ("q-fin", "econ", "oa:2003"),
}
MIN_AREA_DOCS = 5
# Общие обороты — не названия технологий. Список модификаторов взят у команды (wsignals/query.py,
# GENERIC_MODIFIERS), только многословные и «пустые» окончания: одиночное «ai» не трогаем («ai red teaming»).
GENERIC_START = ("ai driven", "ai powered", "ai based", "ai enabled", "ai native", "ai optimized", "next generation",
                 "intelligent", "smart", "modern", "future", "effective", "robust", "novel", "innovative", "advanced",
                 "enhanced", "enhancing", "emerging", "traditional", "democratizing", "delivering", "securing", "deployed")
GENERIC_END = ("landscape", "insights", "trends", "solutions", "strategies", "practices", "challenges", "risks",
               "techniques", "approach", "overview", "expertise", "demand", "decisions")
AREAS = {
    "Edge": ("перспективные технологии в Edge AI и периферийных вычислениях", "edge AI"),
    "Защита ИИ": ("перспективные технологии в защите ИИ", "AI security"),
    "Индустриальный ИИ": ("перспективные технологии в индустриальном ИИ", "industrial AI"),
    "Инфраструктура ИИ": ("перспективные технологии в инфраструктуре ИИ", "AI infrastructure"),
    "Роботы": ("перспективные технологии в робототехнике", "robotics"),
    "Финтех": ("перспективные технологии в финтехе", "fintech"),
}


def embed_candidates(c: pd.DataFrame, m) -> np.ndarray:
    path = REG / "emb_stage2_user-bge-m3.npy"
    if path.exists() and np.load(path, mmap_mode="r").shape[0] == len(c):
        return np.load(path)
    E = m.encode([p.replace("_", " ") for p in c.phrase], batch_size=256, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
    np.save(path, E)
    return E


def area_of_signals() -> dict[int, str]:
    import openpyxl
    from .semantic import XLSX
    rows = list(openpyxl.load_workbook(XLSX, read_only=True).worksheets[0].iter_rows(values_only=True))
    hi = next(i for i, r in enumerate(rows) if r and "Источники" in [str(c) for c in r])
    h = [str(x) for x in rows[hi]]
    return {int(r[h.index("№")]): str(r[h.index("Область")]) for r in rows[hi + 1:] if r[h.index("№")]}


def main() -> int:
    from sentence_transformers import SentenceTransformer
    import torch
    r = pd.read_parquet(REG / "ranked.parquet")
    c = r[r.candidate].reset_index(drop=True)
    m = SentenceTransformer(MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")
    m.max_seq_length = 32
    E = embed_candidates(c, m).astype(np.float32)
    c["hand"] = c.hand_rank_score
    ph = c.phrase.str.replace("_", " ")
    generic = ph.str.startswith(GENERIC_START) | ph.str.endswith(GENERIC_END)
    print(f"общих оборотов среди кандидатов: {int(generic.sum()):,} из {len(c):,} — в пул темы не идут".replace(",", " "))
    c = c[~generic].reset_index(drop=True)
    E = E[~generic.values]
    # доля свежих работ сущности в рубриках области (по научному слою; у чисто бизнесовых сущностей — нет)
    g = np.load(REG / "groups.npz")
    groups = [str(x) for x in g["groups"]]
    grp = g["grp"]
    sci = pd.read_parquet(REG / "all_phrases.parquet", columns=["key", "eid"])
    eid = c.key.map(dict(zip(sci.key, sci.eid)))
    has = eid.notna().values
    tot = np.zeros(len(c))
    tot[has] = grp[:, eid[has].astype(int).values].sum(axis=0)
    area_share = {}
    for area, prefixes in AREA_GROUPS.items():
        rows = [i for i, name in enumerate(groups) if any(name == p or name.startswith("ax:" + p + ".") or
                                                         name.startswith(p + ".") for p in prefixes)]
        a = np.zeros(len(c))
        a[has] = grp[np.ix_(rows, eid[has].astype(int).values)].sum(axis=0) if rows else 0
        area_share[area] = np.where(tot >= MIN_AREA_DOCS, a / np.maximum(tot, 1), 0.0)

    lab = pd.read_csv(LABELS)
    lab = lab[lab.same >= 0.5]
    key_sig = {}
    for _, x in lab.iterrows():
        key_sig.setdefault(_key(x.candidate), []).append((int(x.id), float(x.same)))
    sig_area = area_of_signals()

    out, summary, diag = [], [], []
    for area, (ru, en) in AREAS.items():
        Q = m.encode([ru, en], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        rel = (E @ Q.T).max(axis=1)
        # пул темы: близость по смыслу и доля работ в рубриках области, поровну (процентили среди кандидатов)
        cc = c.assign(semantic=rel, area_share=area_share[area])
        cc["relevance"] = 0.5 * cc.semantic.rank(pct=True) + 0.5 * cc.area_share.rank(pct=True)
        pool = cc.nlargest(POOL, "relevance").copy()
        pool["blend"] = pool.learned.rank(pct=True) * pool.relevance.rank(pct=True)
        area_sigs = {i for i, a in sig_area.items() if a == area}
        # диагностика: сколько сигналов области вообще в пуле и на каком месте в пуле по обученному баллу
        pool["lrank"] = pool.learned.rank(ascending=False, method="min")
        pool["pool_pct"] = pool.learned.rank(pct=True)          # сила сигнала внутри темы — для карточки
        in_pool = {}
        for k, lr in zip(pool.key, pool.lrank):
            for sid, _same in key_sig.get(k, []):
                if sid in area_sigs:
                    in_pool[sid] = min(in_pool.get(sid, 1e9), lr)
        diag.append({"area": area, "signals_in_pool": len(in_pool),
                     "median_rank_in_pool": float(np.median(list(in_pool.values()))) if in_pool else None,
                     "in_pool_top50": sum(v <= 50 for v in in_pool.values())})
        for method in ("relevance", "hand", "learned", "blend"):
            top = pool.nlargest(15, method).copy()
            hits = {}
            for k in top.key:
                for sid, same in key_sig.get(k, []):
                    if sid in area_sigs:
                        hits[sid] = max(hits.get(sid, 0), same)
            test = {s: v for s, v in hits.items() if s % 2 == 0}
            summary.append({"area": area, "method": method, "signals_in_top15": len(hits),
                            "exact": sum(v >= 1 for v in hits.values()), "test_signals": len(test),
                            "area_signals": len(area_sigs)})
            top["area"], top["method"], top["pos"] = area, method, range(1, len(top) + 1)
            top["org_signal"] = [";".join(str(s) for s, _ in key_sig.get(k, []) if s in area_sigs) for k in top.key]
            out.append(top[["area", "method", "pos", "phrase", "relevance", "learned", "hand", "pool_pct", "org_signal"]])
    res = pd.concat(out, ignore_index=True)
    path = REG / "stage2_top15.csv"
    if path.exists():   # разметку выдачи не теряем
        prev = pd.read_csv(path)
        if "weak_signal" in prev:
            res = res.merge(prev[["area", "phrase", "weak_signal"]].drop_duplicates(["area", "phrase"]),
                            on=["area", "phrase"], how="left")
    if "weak_signal" not in res:
        res["weak_signal"] = np.nan
    res.to_csv(path, index=False)
    s = pd.DataFrame(summary)
    print("Сигналы организаторов своей области в ТОП-15 (из 16–17 на область):")
    print(s.pivot_table(index="area", columns="method", values="signals_in_top15").to_string())
    print("\nСигналы организаторов в пуле темы (из 500) и их место в пуле по обученному баллу:")
    print(pd.DataFrame(diag).to_string(index=False))
    print("\nВсего по шести областям:", s.groupby("method")[["signals_in_top15", "exact", "test_signals"]].sum().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
