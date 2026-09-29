"""Продуктовый путь поиска по реестру: запрос → 40 кандидатов (до проверки моделью) и ТОП-15 без проверки.

    python -m registry.search "перспективные технологии в финтехе" ["ещё запрос" ...]
    python -m registry.search --eval        # 6 областей ТЗ + 6 новых + 6 свежих запросов -> search_top40.csv

Шаги (модель на этом этапе не вызывается):
  1. тема запроса без слов-намерений (`wsignals.intent.split`);
  2. эмбеддинг темы (USER-bge-m3) против «русское название — описание» одобренных технологий пула (llm_assess);
  3. область ТЗ: кандидат — ближайшее к теме название области; принимается, если запрос почти называет её
     (сходство ≥ AREA_NAMED) или её подтверждают VOTE_NN ближайших технологий пула по тегам областей (вес ≥ VOTE_MIN).
     Иначе «квантовые вычисления» попадали в «Роботы», а «медицина» — в Edge. Пороги выбраны по правильности
     области на 22 запросах (18 оценочных + 4 вне областей), не по качеству выдачи. Прежний способ (по примерам
     организаторов) ошибался в 3 из 12;
  4. mode="mix" (по умолчанию, лучшая размеченная конфигурация ru_mix, 29/90): если область найдена —
     смесь 50 % смысла запроса + 50 % доли работ в рубриках области (из корпуса, от формулировки не зависит),
     только технологии с долей ≥ MIN_AREA_SHARE или тегом области; TOPIC лучших → по обученному баллу;
     mode="examples" — прежний вариант: близость к примерам организаторов (dev-половина) и балл внутри страты объёма
     (разметка: 18/90 после проверки Pro — хуже, оставлен для сравнения);
  5. отсев зрелого: если тема набрала ≥ 3 работ (arXiv + OpenAlex) в каком-то году ≤ MATURE_YEAR, это не ранняя
     стадия. Порог выбран по разметке запросов по областям (верных теряется 1 из 29, неверных уходит 27 из 55);
     на новых и свежих запросах, не участвовавших в подборе: верных 1 из 27, неверных 66 из 172;
  6. склейка дублей по смыслу (сходство «название — описание» ≥ DUP_SIM с уже взятым) → 40 кандидатов.
Ограничение: запрос вне шести областей ищется только по смыслу темы, порядок — по обученному баллу.
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd

from .corpus import REGISTRY
from .stage2_pool import ru_norm, ru_specific

REG = REGISTRY
MODEL = "deepvk/USER-bge-m3"
TOPIC = 300
SEM_TOP = 5             # столько ближайших по смыслу проходят мимо фильтра области
TOPIC_BASE = 80          # база в 7 раз меньше пула: по теме берём меньше, иначе балл перебивает тему
KEEP = 40
DUP_SIM = 0.88
AREA_NAMED = 0.75
VOTE_NN = 100
VOTE_MIN = 0.12
MATURE_YEAR = 2019
AREA_NAMES = {"Edge": "Edge AI: ИИ на устройствах и периферийные вычисления",
              "Защита ИИ": "Защита ИИ: безопасность систем искусственного интеллекта",
              "Индустриальный ИИ": "Индустриальный ИИ: ИИ в промышленности и производстве",
              "Инфраструктура ИИ": "Инфраструктура ИИ: чипы, память, сети и дата-центры для ИИ",
              "Роботы": "Роботы и робототехника",
              "Финтех": "Финтех: платежи, банки, кредитование, страхование и оценка рисков, криптоактивы и финансовые рынки"}
# новые запросы: сформулированы иначе; выдача по ним один раз размечена (18 % после Pro) — уже не «слепые»
NEW_QUERIES = ["что нового в безопасности больших языковых моделей",
               "какие технологии ИИ появляются в банках и платежах",
               "роботы для складов и заводов: что на подходе",
               "новое железо для обучения и запуска нейросетей",
               "ИИ прямо на смартфонах и датчиках",
               "технологии для умного производства и промышленного оборудования"]
# третья серия: записана до того, как по ней была получена хоть одна выдача; по ней — итоговый замер
FRESH_QUERIES = ["как защищают нейросети от атак и утечек данных",
                 "новые подходы к антифроду и кредитному скорингу",
                 "гуманоидные и мобильные роботы: ранние технологии",
                 "чипы, память и охлаждение для дата-центров ИИ",
                 "локальный запуск моделей на устройствах без облака",
                 "ИИ для предиктивного обслуживания и контроля качества на заводе"]


def first_year(keys: pd.Series) -> pd.Series:
    """Первый год, когда у темы ≥ 3 работ в arXiv + OpenAlex (NaN, если такого года нет)."""
    from .corpus import SOURCES, YEARS
    sci = pd.read_parquet(REG / "all_phrases.parquet", columns=["key", "eid"])
    eid = keys.map(dict(zip(sci.key, sci.eid)))
    df = np.load(REG / "counts.npz")["df"]
    t = df[:, SOURCES.index("arxiv")] + df[:, SOURCES.index("openalex")]
    first = np.where((t >= 3).any(axis=1), np.array(YEARS)[(t >= 3).argmax(axis=1)], np.nan)
    return eid.map(lambda e: first[int(e)] if pd.notna(e) else np.nan)


def acceleration(keys: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Ускорение темы: доля работ arXiv + OpenAlex в 2024–2026 к доле 2020–2022, и число работ за 2024–2026."""
    import json as _json
    from .corpus import SOURCES, YEARS
    sci = pd.read_parquet(REG / "all_phrases.parquet", columns=["key", "eid"])
    eid = keys.map(dict(zip(sci.key, sci.eid)))
    df = np.load(REG / "counts.npz")["df"]
    tot = _json.loads((REG / "totals.json").read_text(encoding="utf-8"))
    src = [SOURCES.index("arxiv"), SOURCES.index("openalex")]
    yi = {y: i for i, y in enumerate(YEARS)}
    rec, base = [yi[y] for y in (2024, 2025, 2026)], [yi[y] for y in (2020, 2021, 2022)]
    n_rec = sum(tot.get(f"{s}|{y}", 0) for s in ("arxiv", "openalex") for y in (2024, 2025, 2026))
    n_base = sum(tot.get(f"{s}|{y}", 0) for s in ("arxiv", "openalex") for y in (2020, 2021, 2022))
    t = df[:, src].sum(axis=1)
    d_rec, d_base = t[:, rec].sum(axis=1), t[:, base].sum(axis=1)
    acc = (d_rec / max(n_rec, 1) + 1e-12) / (d_base / max(n_base, 1) + 1e-12)
    f = lambda arr: eid.map(lambda e: float(arr[int(e)]) if pd.notna(e) else np.nan)
    return f(acc), f(d_rec)


# калька «zero-shot» у YandexGPT Lite («в ноль выстрелов», «нулевое-выстреловое») — правильно: без обучения на примерах
_ZERO_SHOT = re.compile(r"(с\s+)?(нулев\w*[\s-]+(выстрел|стрел)\w*|в\s+ноль\s+выстрел\w*|нулевым\s+выстрелом)", re.I)


def fix_ru(name: str) -> str:
    if not isinstance(name, str):
        return name
    m = _ZERO_SHOT.match(name.strip())
    if m:                                             # «Нулевое-выстреловое прогнозирование» → «Прогнозирование без …»
        rest = name.strip()[m.end():].strip()
        out = f"{rest} без обучения на примерах"
    else:
        out = _ZERO_SHOT.sub("без обучения на примерах", name)
    return out[:1].upper() + out[1:] if out != name else name


class Searcher:
    def __init__(self, mode: str = "mix", base: bool = False):
        """base=True — кандидаты только из базы сигналов (signal_base.parquet), порядок — экспертный балл."""
        self.mode = mode
        self.order_col = "expert" if base else "learned"
        from sentence_transformers import SentenceTransformer
        import torch
        from . import pool as pool_mod
        from .semantic import queries as sig_queries
        from .stage2 import area_of_signals
        self.m = SentenceTransformer(MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")
        self.m.max_seq_length = 64
        llm = {}
        for l in (REG / "llm_assess.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                x = json.loads(l)
                llm[x["phrase"]] = x
        p = pool_mod.select(REG, 20000, 3000).copy()
        p["ph"] = p.phrase.str.replace("_", " ")
        p = p[p.ph.map(lambda s: s in llm and llm[s]["keep"])]
        p["name_ru"] = p.ph.map(lambda s: fix_ru(llm[s]["name_ru"]))
        p["description_ru"] = p.ph.map(lambda s: llm[s]["description_ru"])
        p["areas"] = p.ph.map(lambda s: llm[s]["areas"])
        p = p[p.name_ru.map(ru_specific)]
        p = p.assign(rkey=p.name_ru.map(ru_norm)).sort_values("learned", ascending=False).drop_duplicates("rkey")
        p = p.reset_index(drop=True)
        p["first_year"] = first_year(p.key)
        if base:
            b = pd.read_parquet(REG / "signal_base.parquet",
                                columns=["key", "expert", "org_sim", "biz", "aliases", "kind", "areas", "tier"])
            # своя принадлежность к области (словарь предмета + рубрики, signal_base.our_areas) — для фильтра на
            # запросе; метка Lite (areas) остаётся для голосования при определении области запроса
            p = p.merge(b.rename(columns={"areas": "areas_our"}), on="key", how="inner")
            full = pd.read_parquet(REG / "signal_base.parquet")      # подтверждённые составные и рыночные сигналы
            extra = full[full.key.str.startswith(("combo:", "mkt:"))]  # (registry/passports.py) — нет в пуле реестра
            if len(extra):
                extra = extra.assign(areas_our=extra.areas)
                p = pd.concat([p, extra[[c for c in p.columns if c in extra.columns]]], ignore_index=True)
            p = p.reset_index(drop=True)
        sh = pool_mod.area_shares(REG, p)
        for a in pool_mod.AREA_GROUPS_POOL:
            p["share:" + a] = sh[a].values
        # балл внутри страты объёма: крупные темы соревнуются с крупными, малые — с малыми (перекос к популярному)
        vol = p.sci_log_volume.fillna(0) + p.biz_log_hn.fillna(0)
        p["stratum"] = pd.qcut(vol.rank(method="first"), 10, labels=False)
        p["learned_strat"] = p.groupby("stratum").learned.rank(pct=True)
        self.pool = p
        cache = REG / "emb_search_pool.npz"
        texts = (self.pool.name_ru + " — " + self.pool.description_ru).tolist()
        if cache.exists() and list(np.load(cache, allow_pickle=True)["keys"]) == self.pool.key.tolist():
            self.E = np.load(cache, allow_pickle=True)["E"]
        else:
            self.E = self.m.encode(texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False)
            np.savez(cache, E=self.E, keys=np.array(self.pool.key.tolist(), dtype=object))
        # примеры областей — только dev-половина сигналов организаторов
        sig_area = area_of_signals()
        self.seeds = {}
        for q in sig_queries():
            if q["id"] % 2 == 1 and q["id"] in sig_area:
                self.seeds.setdefault(sig_area[q["id"]], []).append(f"{q['ru']} ({q['en']})")
        self.S = {a: self.m.encode(v, normalize_embeddings=True, convert_to_numpy=True) for a, v in self.seeds.items()}
        cent = {a: self.m.encode([AREA_NAMES[a]] + self.seeds[a], normalize_embeddings=True).mean(axis=0)
                for a in self.seeds}
        self.C = {a: c / np.linalg.norm(c) for a, c in cent.items()}
        self.N = {a: self.m.encode([AREA_NAMES[a]], normalize_embeddings=True)[0] for a in self.seeds}
        self.area_n = {a: max(1, int(self.pool.areas.map(lambda l, a=a: a in l).sum())) for a in self.N}

    def area_of(self, qv: np.ndarray) -> tuple[str | None, float]:
        # область = ближайшее название области, если с ним согласны теги ближайших технологий пула
        sims = {a: float(self.N[a] @ qv) for a in self.N}
        a = max(sims, key=sims.get)
        sim = self.E @ qv
        vote = dict.fromkeys(self.N, 0.0)
        for i in np.argsort(-sim)[:VOTE_NN]:
            for t in self.pool.areas.iat[i]:
                if t in vote:
                    vote[t] += sim[i] / self.area_n[t] ** 0.5      # поправка на размер области в пуле
        if self.order_col == "expert":                  # поиск по базе: голоса крупнее, пороги подобраны на базе
            ok = sims[a] >= AREA_NAMED or vote[a] >= 0.35 or (sims[a] >= 0.42 and vote[a] >= 0.20)
        else:
            ok = sims[a] >= AREA_NAMED or vote[a] >= VOTE_MIN
        return (a if ok else None), sims[a]

    def search(self, query: str) -> tuple[pd.DataFrame, str | None]:
        from wsignals import intent
        topic = intent.split(query).domain or query
        qv = self.m.encode([topic], normalize_embeddings=True, convert_to_numpy=True)[0]
        area, _ = self.area_of(qv)
        rel = self.E @ qv
        if getattr(self, "hyde", None):                 # HyDE: + ближайшее из гипотетических описаний (registry/hyde.py)
            H = self.m.encode(self.hyde, normalize_embeddings=True, convert_to_numpy=True)
            w = getattr(self, "hyde_w", 0.5)
            rel = (1 - w) * rel + w * (self.E @ H.T).max(axis=1)
        t = self.pool.assign(rel_query=rel)
        if self.mode == "mix":
            if area:
                from .pool import MIN_AREA_SHARE
                t["rel_example"] = t["share:" + area]
                sw = getattr(self, "share_w", 0.5)
                qr = t.rel_query.rank(pct=True)
                t["relevance"] = (1 - sw) * qr + sw * t["share:" + area].rank(pct=True)
                # самые близкие по смыслу не отсекаются областью: область запроса могла определиться неточно,
                # а у сигнала на стыке областей метка другая
                sem = t.index.isin(t.rel_query.nlargest(getattr(self, "sem_top", SEM_TOP)).index)
                t.loc[sem, "relevance"] = np.maximum(t.relevance[sem], qr[sem])
                if "areas_our" in t:     # база: своя принадлежность уже учитывает рубрики, но для Edge и промышленности
                    # требует предметного слова — доля рубрик сама по себе пускала сети дата-центров в Edge
                    t = t[t.areas_our.map(lambda l: l is not None and area in list(l)) | sem]
                else:
                    t = t[(t["share:" + area] >= MIN_AREA_SHARE) | t.areas.map(lambda l: area in l) | sem]
            else:
                t["rel_example"] = np.nan
                t["relevance"] = t.rel_query
            if self.order_col == "expert":
                near = t.nlargest(TOPIC_BASE, "relevance")
                ew = getattr(self, "expert_w", 0.5)
                near = near.assign(order=(1 - ew) * near.relevance.rank(pct=True) + ew * near.expert
                                   - (near.tier.fillna(2) - 1 if "tier" in near else 0))   # ярус 1 разметки — первым
            else:
                near = t.nlargest(TOPIC, "relevance")
                near = near.assign(order=near[self.order_col])
            near = near.sort_values("order", ascending=False)
        elif area:
            ex = (self.E @ self.S[area].T)                     # сходство с каждым примером области
            t["rel_example"] = np.sort(ex, axis=1)[:, -3:].mean(axis=1)   # среднее трёх ближайших примеров
            t["relevance"] = 0.5 * t.rel_query.rank(pct=True) + 0.5 * t.rel_example.rank(pct=True)
        else:
            t["rel_example"] = np.nan
            t["relevance"] = t.rel_query
        if self.mode != "mix":
            near = t.nlargest(TOPIC, "relevance")
            # порядок: близость к теме (60 %) и балл сигнальности внутри страты объёма (40 %)
            near = near.assign(order=0.6 * near.relevance.rank(pct=True) + 0.4 * near.learned_strat)
            near = near.sort_values("order", ascending=False)
        near = near[~(near.first_year <= MATURE_YEAR)]           # зрелое — не слабый сигнал
        idx = near.index.to_numpy()
        picked: list[int] = []
        for i in idx:                                       # склейка дублей по смыслу
            if picked and float((self.E[picked] @ self.E[i]).max()) >= DUP_SIM:
                continue
            picked.append(i)
            if len(picked) >= KEEP:
                break
        n_main = len(picked)
        extra_n = getattr(self, "extra", 0)
        if extra_n and self.mode == "mix" and self.order_col == "expert":
            # «ещё по теме» за пределами 40 кандидатов: следующие по тому же порядку, моделью не проверяются
            far = t.nlargest(TOPIC_BASE * 3, "relevance")
            far = far[~(far.first_year <= MATURE_YEAR)]
            ew = getattr(self, "expert_w", 0.5)
            far = far.assign(order=(1 - ew) * far.relevance.rank(pct=True) + ew * far.expert
                             - (far.tier.fillna(2) - 1 if "tier" in far else 0)).sort_values("order", ascending=False)
            ext: list[int] = []
            for i in far.index:
                if i in picked or float((self.E[picked + ext] @ self.E[i]).max()) >= DUP_SIM:
                    continue
                ext.append(i)
                if len(ext) >= extra_n:
                    break
            picked = picked + ext
        out = t.loc[picked].copy()
        out["extra"] = [False] * n_main + [True] * (len(out) - n_main)
        out["rank"] = range(1, len(out) + 1)
        out["pool_pct"] = out.learned.rank(pct=True)
        return out, area


def main() -> int:
    from .stage2 import AREAS
    args = sys.argv[1:]
    base = "--base" in args
    use_hyde = "--hyde" in args
    args = [a for a in args if a not in ("--base", "--hyde")]
    s = Searcher(base=base)
    s.extra = 60 if base else 0                    # «ещё по теме» для страницы: 60 следующих без проверки моделью
    hyde = json.loads((REG / "hyde.json").read_text(encoding="utf-8")) if use_hyde and (REG / "hyde.json").exists() else {}
    if args == ["--eval"]:
        qs = ([("области", q) for q, _en in AREAS.values()] + [("новые", q) for q in NEW_QUERIES]
              + [("свежие", q) for q in FRESH_QUERIES])
    else:
        qs = [("", q) for q in args]
    rows = []
    for tag, q in qs:
        s.hyde = hyde.get(q)
        out, area = s.search(q)
        print(f"\n{q}  [область: {area or '—'}]")
        for r in out.head(15).itertuples():
            print(f"  {r.rank:>2}. {r.name_ru} ({r.phrase.replace('_', ' ')})")
        out = out.assign(query=q, query_set=tag, area_detected=area)
        rows.append(out[["query", "query_set", "area_detected", "rank", "key", "phrase", "name_ru", "description_ru",
                         "learned", "pool_pct", "relevance", "rel_query", "rel_example", "extra"]])
    if args == ["--eval"]:
        out = REG / ("search_top40_base.csv" if base else "search_top40.csv")
        pd.concat(rows).to_csv(out, index=False)
        print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
