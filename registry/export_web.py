"""Выгрузка выдачи этапа 2 для веб-страницы: web/data/demo.json. Тот же контракт будет отдавать API реестра.

    python -m registry.export_web [--method blend]

Карточка:
  key, name (оригинал), name_ru (null, пока нет YandexGPT), strength (0–1: процентиль обученного балла среди
  кандидатов реестра), facts (человеческие фразы из измерений), ladder (ступени зрелости), series (доли по годам
  по слоям), explain (вклад признаков в обученный балл), description/advantage/case (null до YandexGPT),
  sources (документы-подтверждения из корпуса: название, ссылка, дата, тип, язык, доверие).

Источники ищутся в корпусе для технологий выдачи: документ подходит, если среди его фраз есть вариант этой
сущности. Отбор: наука — самые цитируемые работы OpenAlex и свежие препринты arXiv, HN — самые обсуждаемые
истории, пресса — самые свежие статьи, YC — компании. Доверие — по логике wsignals/trust.py: рецензируемая
статья — высокий, препринт и отраслевые СМИ — средний, сообщество и пресс-релизы — пониженный.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import math
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import CORPUS, REGISTRY, SOURCES, YEARS, docs, files
from .text import candidates, entity_key

REG = REGISTRY
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "data" / "demo.json"
R, B = (2024, 2025, 2026), (2019, 2020, 2021, 2022)
PER_LAYER = 3
PRESS_RELEASE = re.compile(r"prnewswire|businesswire|globenewswire|accessnewswire|newswire", re.I)
LABELS = {
    "sci_log_volume": "Объём научных работ", "sci_growth": "Рост доли в науке", "sci_age": "Возраст темы в науке",
    "sci_log_rate": "Доля среди всех научных работ", "sci_spread": "Охват разных научных областей",
    "sci_sources": "Число научных источников", "sci_oa_share": "Доля журнальных статей (против препринтов)",
    "biz_log_hn": "Обсуждения на Hacker News", "biz_hn_growth": "Рост обсуждений на Hacker News",
    "biz_log_yc": "Стартапы Y Combinator", "biz_log_press": "Публикации в прессе",
    "biz_log_launches": "Запуски продуктов на Hacker News", "layers": "Ступеней лестницы зрелости",
    "n_words": "Длина названия (конкретность)", "name_like": "Цифры и версии в названии (признак продукта)",
    "specificity": "Общие слова в названии", "wiki_present": "Есть статья в Википедии",
    "wiki_age": "Возраст статьи в Википедии", "biz_log_raised": "Деньги стартапов (SEC Form D)",
    "biz_funded": "Стартапов с раундом (SEC Form D)", "first_year": "Свежесть темы (год первого появления)",
    "theory_share": "Доля теоретических работ", "academic_share": "Доля работ по теории связи и управления",
}
# регистр аббревиатур в названиях, пока нет русских названий от YandexGPT
ACRONYMS = {w: w.upper() for w in """ai ml llm llms mcp rag rl iot kv cxl npu gpu tpu vla vlm vlms rwa kyc aml api
edge-ai 5g 6g ev evs bess tee zk pqc rf lidar hbm dram sram nvme soc mcu cpu""".split()} | {"llms": "LLMs", "vlms": "VLMs", "evs": "EVs"}
EXAMPLE_MATCH = {"Edge": ["edge", "периферийн", "граничн"], "Защита ИИ": ["защит", "безопасн"],
                 "Индустриальный ИИ": ["индустриал", "промышлен", "производств", "завод"],
                 "Инфраструктура ИИ": ["инфраструктур", "цод", "дата-центр", "чип"],
                 "Роботы": ["робот"], "Финтех": ["финтех", "финанс", "банк", "платеж", "платёж"]}


def times(x: float) -> str:
    return f"{x:.1f}".replace(".", ",").replace(",0", "") if x < 10 else f"{x:.0f}"


def plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(int(n))
    return one if n % 10 == 1 and n % 100 != 11 else few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many


# ---------- документы-подтверждения ----------

_P2E: dict[str, int] = {}
_WANT: set[int] = set()
_BIZ: dict[str, str] = {}


def _init(want_eids: list[int], biz_keys: list[str]) -> None:
    global _P2E, _WANT, _BIZ
    phrases = [l for l in (REG / "vocab.txt").read_text(encoding="utf-8").split("\n") if l]
    p2e = np.load(REG / "phrase2entity.npy")
    _WANT = set(want_eids)
    _P2E = {ph: int(e) for ph, e in zip(phrases, p2e) if int(e) in _WANT}
    _BIZ = {k: k for k in biz_keys}


def _science_file(args):
    source, path = args
    hits = []
    for d in docs(source, path):
        if d["year"] not in (2023, 2024, 2025, 2026):
            continue
        es = {e for ph in candidates(d["title"] + " . " + d["text"]) if (e := _P2E.get(ph)) is not None}
        if es:
            hits.append((sorted(es), source, d))
    return hits


def science_evidence(want: dict[int, str]) -> dict[str, list[dict]]:
    """Ключ сущности -> документы науки. Цитируемость OpenAlex берём из сырой записи (docs её не отдаёт)."""
    todo = [(s, p) for s, p in files() if s in ("openalex", "arxiv") and
            (s != "openalex" or int(re.search(r"year=(\d+)", p).group(1)) >= 2023)]
    found = collections.defaultdict(list)
    with ProcessPoolExecutor(6, initializer=_init, initargs=(list(want), [])) as ex:
        for hits in ex.map(_science_file, todo, chunksize=4):
            for es, source, d in hits:
                for e in es:
                    found[want[e]].append((source, d))
    return found


def openalex_meta(ids: set[str]) -> dict[str, dict]:
    meta = {}
    for f in glob.glob(str(CORPUS / "openalex/subfield=*/year=202[3-6]/*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for l in fh:
                rid = l[8:l.find('"', 8)] if l.startswith('{"id": "') else None   # номер работы — в начале строки
                if rid not in ids:
                    continue
                r = json.loads(l)
                if r.get("id") in ids:
                    src = r.get("source") or {}
                    meta[r["id"]] = {"cited": int(r.get("cited_by_count") or 0), "date": r.get("publication_date"),
                                     "lang": r.get("language") or "en", "type": r.get("type"),
                                     "venue": (src.get("name") if isinstance(src, dict) else None) or "OpenAlex",
                                     "url": r.get("doi") or r.get("landing_url") or f"https://openalex.org/{r['id']}"}
    return meta


def business_evidence(keys: set[str]) -> dict[str, list[dict]]:
    """HN, YC, пресса — по ключу сущности (варианты фраз бизнес-слоя склеиваются тем же ключом)."""
    out = collections.defaultdict(list)
    for f in sorted(glob.glob(str(CORPUS / "hn/202[3-6]-*.jsonl.gz"))):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for l in fh:
                r = json.loads(l)
                if (r.get("score") or 0) < 5 or not r.get("title"):
                    continue
                for k in {entity_key(ph) for ph in candidates(r["title"])} & keys:
                    out[k].append({"layer": "hn", "title": r["title"], "score": r["score"],
                                   "url": r.get("url") or f"https://news.ycombinator.com/item?id={r['id']}",
                                   "published": r["time"][:10], "source_name": "Hacker News",
                                   "source_type": "сообщество", "language": "en", "trust": "пониженный"})
    for f in glob.glob(str(CORPUS / "yc/*.json")):
        for r in json.load(open(f, encoding="utf-8")):
            text = " . ".join(str(r.get(x) or "") for x in ("one_liner", "long_description"))
            for k in {entity_key(ph) for ph in candidates(text)} & keys:
                out[k].append({"layer": "yc", "title": f"{r.get('name')}: {r.get('one_liner') or ''}".strip(": "),
                               "url": r.get("url") or r.get("website"), "published": r.get("batch"),
                               "source_name": "Y Combinator", "source_type": "компания-разработчик",
                               "language": "en", "trust": "средний", "score": 0})
    for f in glob.glob(str(CORPUS / "sitemaps/*/*.jsonl.gz")):
        dom = Path(f).parent.name
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for l in fh:
                    r = json.loads(l)
                    if r.get("date_source") not in ("url", "news") or r["date"][:4] < "2023":
                        continue
                    slug = r["url"].rstrip("/").rsplit("/", 1)[-1].replace("-", " ")
                    title = r.get("title") or slug
                    for k in {entity_key(ph) for ph in candidates(title)} & keys:
                        pr = bool(PRESS_RELEASE.search(dom))
                        out[k].append({"layer": "press", "title": title[:160], "url": r["url"], "published": r["date"][:10],
                                       "source_name": dom, "source_type": "пресс-релиз" if pr else "новости",
                                       "language": "en", "trust": "пониженный" if pr else "средний", "score": 0})
        except (EOFError, OSError):
            continue
    return out


# ---------- карточки ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="blend")
    ap.add_argument("--top-file", default="stage2_top15.csv", help="stage2_top15.csv или stage2_pool_top15.csv")
    ap.add_argument("--out", default=str(OUT), help="куда писать JSON (по умолчанию — данные страницы)")
    a = ap.parse_args()
    out = Path(a.out)
    import joblib
    t0 = time.time()
    top = pd.read_csv(REG / a.top_file)
    if "method" in top:
        top = top[top.method == a.method]
    else:                                   # выдача по базе (search_top40_base / selected_top15): по запросам
        if "pos" not in top:
            top = top[top["rank"] <= 15].assign(pos=lambda d: d["rank"])
        top = top.assign(area=top.area_detected.fillna("—"))
    ranked = pd.read_parquet(REG / "ranked.parquet")
    cand = ranked[ranked.candidate]
    pct = dict(zip(cand.key, cand.learned.rank(pct=True)))
    sci = pd.read_parquet(REG / "all_phrases.parquet")
    biz = pd.read_parquet(REG / "all_business.parquet")
    phrase2key = dict(zip(ranked.phrase, ranked.key))
    wp, fp = REG / "wiki.parquet", REG / "funding.parquet"
    wiki = dict(zip(*pd.read_parquet(wp)[["key", "wiki_title"]].values.T)) if wp.exists() else None
    fund = pd.read_parquet(fp).set_index("key") if fp.exists() else None
    top["key"] = (top.phrase.map(phrase2key).fillna(top["key"]) if "key" in top else top.phrase.map(phrase2key))
    keys = set(top.key.dropna())
    sci_by_key = sci.set_index("key")
    biz_by_key = biz.set_index("key")

    # ряды по годам
    counts = np.load(REG / "counts.npz")["df"]
    totals = json.loads((REG / "totals.json").read_text(encoding="utf-8"))
    N = np.array([[totals.get(f"{s}|{y}", 0) for y in YEARS] for s in SOURCES], dtype=float)
    bs = np.load(REG / "business_series.npz")
    bkey = {k: i for i, k in enumerate(bs["keys"])}
    bsrc = list(bs["sources"])

    # документы-подтверждения
    # источники ищутся по всему корпусу — дорого, поэтому кэшируются по набору технологий выдачи
    import pickle
    cache = REG / "web_evidence.pkl"
    cached = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    if cached.get("keys") == keys:
        sev, meta, bev = cached["sev"], cached["meta"], cached["bev"]
    else:
        want = {int(sci_by_key.loc[k, "eid"]): k for k in keys if k in sci_by_key.index}
        sev = science_evidence(want)
        oa_ids = {d["id"] for k in sev for s, d in sev[k] if s == "openalex"}
        meta = openalex_meta(oa_ids)
        bev = business_evidence(keys)
        cache.write_bytes(pickle.dumps({"keys": keys, "sev": dict(sev), "meta": meta, "bev": dict(bev)}))
    print(f"источники собраны за {time.time() - t0:.0f} с", flush=True)

    mdl = joblib.load(REG / "ranker.joblib")
    pipe, feats = mdl["model"], mdl["features"]
    imp, sc, lr = pipe[0], pipe[1], pipe[2]

    shap_tab = scored_tab = shap_feats = None
    if (REG / "expert_shap.parquet").exists():          # экспертная модель и её SHAP (registry/expert_model.py)
        shap_tab = pd.read_parquet(REG / "expert_shap.parquet").drop_duplicates("key").set_index("key")
        shap_feats = [c for c in shap_tab.columns if c != "base_value"]
        scored_tab = pd.read_parquet(REG / "expert_scored.parquet").drop_duplicates("key").set_index("key")
    from .passports import load as load_passports, passport_view
    passports = load_passports()                        # паспорта сигналов (registry/passports.py)
    card_text = {}                                      # преимущество и кейс (registry/card_text.py), кэш по фразе
    if (REG / "card_text.jsonl").exists():
        for l in (REG / "card_text.jsonl").read_text(encoding="utf-8").splitlines():
            x = json.loads(l)
            card_text[x["phrase"]] = x
    twins = {}                                          # исторический двойник (registry/twins.py)
    if (REG / "twins.parquet").exists():
        for x in pd.read_parquet(REG / "twins.parquet").itertuples():
            twins[x.key] = {"term": x.twin, "takeoff": bool(x.twin_takeoff), "growth": float(x.twin_growth),
                            "docs_then": int(x.twin_docs_then), "docs_later": int(x.twin_docs_later),
                            "yc_after": int(x.twin_yc_after), "press_after": int(x.twin_press_after)}
    actors = {}
    if (REG / "actors.parquet").exists():               # кто стоит за сигналом (registry/actors.py)
        for x in pd.read_parquet(REG / "actors.parquet").itertuples():
            actors[x.key] = {"orgs": int(x.orgs), "companies": int(x.companies), "universities": int(x.universities),
                             "countries": int(x.countries), "cn_share": float(x.cn_share),
                             "top_orgs": json.loads(x.top_orgs), "yc": json.loads(x.yc),
                             "funded": int(x.funded), "raised": float(x.raised)}
    hist_tab = hist_feats = None
    from .history_model import LABELS as HIST_LABELS
    if (REG / "history_shap.parquet").exists():         # модель взлёта на истории (registry/history_model.py)
        hist_tab = pd.read_parquet(REG / "history_shap.parquet").drop_duplicates("key").set_index("key")
        hist_feats = list(hist_tab.columns)
    results, examples = {}, []
    from .stage2 import AREAS
    plan = ([(area, q) for area, (q, _en) in AREAS.items()] if "query" not in top
            else [(g.area.iloc[0], q) for q, g in top.groupby("query", sort=False)])
    for area, query in plan:
        rows = (top[top.area == area] if "query" not in top else top[top["query"] == query]).sort_values("pos")
        # сила сигнала: процентиль обученного балла внутри пула темы (500 близких к запросу технологий)
        # сила сигнала — тот же итоговый балл (смесь двух моделей), по которому упорядочена выдача
        in_pool = (rows.expert if "expert" in rows and rows.expert.notna().all()
                   else rows.pool_pct if "pool_pct" in rows else rows.learned.rank(pct=True))
        cards = []
        for _, r in rows.iterrows():
            k = r.key
            if isinstance(k, str) and k.startswith(("combo:", "mkt:")):   # составной или рыночный сигнал с паспортом
                from .passports import passport_card
                pc = passport_card(k, float(in_pool.loc[r.name]), passports)
                if isinstance(r.get("headline"), str) and r.headline.strip():
                    pc["name_ru"] = r.headline.strip()
                ct = card_text.get(str(pc.get("name", "")).lower())      # преимущество и кейс — тот же кэш по фразе
                if ct:
                    pc["advantage"] = {"text": ct["advantage"], "generated": True}
                    pc["case"] = {"text": ct["case"], "generated": True}
                pc["unverified"] = r.get("unverified") is True
                cards.append(pc)
                continue
            s = sci_by_key.loc[k] if k in sci_by_key.index else None
            b = biz_by_key.loc[k] if k in biz_by_key.index else None
            facts = []
            if s is not None and s.volume_recent > 0:
                ax_years = np.array(YEARS)[counts[int(s.eid), SOURCES.index("arxiv")] >= 2]
                if len(ax_years) and ax_years.min() >= 2022:
                    facts.append(f"Термин в работах arXiv — с {int(ax_years.min())} года")
                if not math.isnan(s.growth) and s.growth >= 1:
                    facts.append(f"Доля научных работ выросла в {times(2 ** s.growth)} раза к 2019–2022")
                facts.append(f"{int(s.volume_recent)} {plural(s.volume_recent, 'запись', 'записи', 'записей')} в научных источниках "
                             f"за 2024–2026 (arXiv и OpenAlex, возможны пересечения)")
            if b is not None:
                if b.yc_recent > 0:
                    facts.append(f"{int(b.yc_recent)} {plural(b.yc_recent, 'стартап', 'стартапа', 'стартапов')} Y Combinator в 2024–2026")
                if b.hn_launches > 0:
                    facts.append(f"{int(b.hn_launches)} {plural(b.hn_launches, 'запуск продукта', 'запуска продуктов', 'запусков продуктов')} на Hacker News")
                elif b.hn_recent > 0:
                    facts.append(f"{int(b.hn_recent)} {plural(b.hn_recent, 'обсуждение', 'обсуждения', 'обсуждений')} на Hacker News")
                if b.press_recent > 0:
                    facts.append(f"{int(b.press_recent)} {plural(b.press_recent, 'публикация', 'публикации', 'публикаций')} в прессе")
            if fund is not None and k in fund.index and fund.loc[k, "funded_recent"] > 0:
                fr = fund.loc[k]
                facts.append(f"{int(fr.funded_recent)} {plural(fr.funded_recent, 'стартап', 'стартапа', 'стартапов')} YC по теме "
                             f"привлекли ${fr.raised_recent / 1e6:,.0f} млн в 2023–2026 (SEC Form D)".replace(",", " "))
            if wiki is not None and k not in wiki:
                facts.append("Отдельной статьи в Википедии нет")
            if s is not None and s.max_rate_recent > 0:
                pct_txt = "меньше 0,01" if s.max_rate_recent < 1e-4 else f"{s.max_rate_recent * 100:.2f}".replace(".", ",")
                facts.append(f"Доля в свежих научных работах: {pct_txt} %")
            ladder = [
                {"layer": "Наука", "present": bool(s is not None and s.volume_recent > 0),
                 "value": f"{int(s.volume_recent)} записей" if s is not None and s.volume_recent > 0 else "нет"},
                {"layer": "Hacker News", "present": bool(b is not None and b.hn_recent > 0),
                 "value": f"{int(b.hn_recent)} историй" if b is not None and b.hn_recent > 0 else "нет"},
                {"layer": "Стартапы YC", "present": bool(b is not None and b.yc_recent > 0),
                 "value": f"{int(b.yc_recent)} компаний" if b is not None and b.yc_recent > 0 else "нет"},
                {"layer": "Пресса", "present": bool(b is not None and b.press_recent > 0),
                 "value": f"{int(b.press_recent)} статей" if b is not None and b.press_recent > 0 else "нет"},
                ({"layer": "Энциклопедия", "present": k in wiki,
                  "value": ("статья: " + wiki[k].replace("_", " ")) if k in wiki else "статьи нет"}
                 if wiki is not None else {"layer": "Энциклопедия", "present": None, "value": "слой ещё не подключён"}),
            ]
            lines = []
            if s is not None:
                e = int(s.eid)
                sh = (counts[e, 0] + counts[e, 1]) / np.maximum(N[0] + N[1], 1)
                lines.append({"label": "наука (доля работ)", "color": "science", "values": [round(float(v) * 1e4, 3) for v in sh]})
            if k in bkey:
                i = bkey[k]
                hn = bs["df"][i, bsrc.index("hn")] / np.maximum(bs["totals"][bsrc.index("hn")], 1)
                lines.append({"label": "Hacker News (доля историй)", "color": "hn", "values": [round(float(v) * 1e4, 3) for v in hn]})
                lines.append({"label": "стартапы YC (штук)", "color": "yc", "values": [int(v) for v in bs["df"][i, bsrc.index("yc")]]})
            if shap_tab is not None and k in shap_tab.index:
                # разложение балла экспертной модели (градиентный бустинг) по признакам: SHAP, вклад в логит
                sv = shap_tab.loc[k, shap_feats].astype(float).values
                xv = scored_tab.loc[k, shap_feats] if k in scored_tab.index else None
                order = np.argsort(-np.abs(sv))[:7]
                explain = [{"label": LABELS.get(shap_feats[j], shap_feats[j]), "value": round(float(sv[j]), 3),
                            "detail": (f"значение признака: {float(xv.iloc[j]):.2f}" if xv is not None
                                       and pd.notna(xv.iloc[j]) else "нет данных")} for j in order]
                explain_history = None
                if hist_tab is not None and k in hist_tab.index:      # что говорит история (2021 → 2024–2026)
                    hv = hist_tab.loc[k, hist_feats].astype(float).values
                    ho = np.argsort(-np.abs(hv))[:5]
                    explain_history = [{"label": HIST_LABELS.get(hist_feats[j], hist_feats[j]),
                                        "value": round(float(hv[j]), 3)} for j in ho]
            else:
                explain_history = None
                # разложение балла: вклад признака = вес × стандартизованное значение
                x = ranked.loc[ranked.key == k, feats]
                xs = sc.transform(imp.transform(x))[0][:len(feats)]
                contrib = lr.coef_[0][:len(feats)] * xs
                order = np.argsort(-np.abs(contrib))[:7]
                explain = [{"label": LABELS.get(feats[j], feats[j]), "value": round(float(contrib[j]), 3),
                            "detail": f"значение признака: {float(x.iloc[0, j]):.2f}"} for j in order]
            # источники: наука (цитируемость OpenAlex, свежесть arXiv), затем HN, YC, пресса
            src = []
            sc_docs = sev.get(k, [])
            oa_docs = sorted((d for s_, d in sc_docs if s_ == "openalex" and d["id"] in meta),
                             key=lambda d: -meta[d["id"]]["cited"])[:PER_LAYER]
            for d in oa_docs:
                mm = meta[d["id"]]
                pre = mm["type"] == "preprint"
                src.append({"title": d["title"], "url": mm["url"], "published": mm["date"], "source_name": mm["venue"],
                            "source_type": "препринт" if pre else "научная публикация", "language": mm["lang"],
                            "trust": "средний" if pre else "высокий"})
            ax_docs = sorted((d for s_, d in sc_docs if s_ == "arxiv"), key=lambda d: -d["year"])[:PER_LAYER]
            for d in ax_docs:
                aid = d["id"].split(":", 1)[1]
                src.append({"title": d["title"], "url": f"https://arxiv.org/abs/{aid}", "published": str(d["year"]),
                            "source_name": "arXiv", "source_type": "препринт", "language": "en", "trust": "средний"})
            bdocs = bev.get(k, [])
            for layer, key_fn in (("hn", lambda d: -d["score"]), ("yc", lambda d: 0), ("press", lambda d: d["published"] or "")):
                ds = [d for d in bdocs if d["layer"] == layer]
                ds = sorted(ds, key=key_fn, reverse=(layer == "press"))[:PER_LAYER]
                src += [{kk: v for kk, v in d.items() if kk not in ("layer", "score")} for d in ds]
            nm = " ".join(ACRONYMS.get(w, w) for w in r.phrase.replace("_", "-").split())
            seen_titles, uniq = set(), []   # одна работа бывает в двух местах (SSRN и arXiv) — склеиваем по заголовку
            for d in src:
                tk = re.sub(r"[^a-z0-9]+", " ", (d.get("title") or "").lower()).strip()
                if tk and tk not in seen_titles:
                    seen_titles.add(tk)
                    uniq.append(d)
            src = uniq
            has_ru = "name_ru" in r and isinstance(r.name_ru, str)
            if "headline" in r and isinstance(r.headline, str) and r.headline.strip():
                r = r.copy()
                r["name_ru"] = r.headline.strip()      # формулировка сигнала в стиле экспертов (select)
            cards.append({"key": k, "unverified": r.get("unverified") is True, "name": nm[:1].upper() + nm[1:],
                          "name_ru": (r.name_ru[:1].upper() + r.name_ru[1:]) if has_ru else None,
                          "strength": round(float(in_pool.loc[r.name]), 3), "strength_registry": round(float(pct.get(k, 0)), 3), "facts": facts, "ladder": ladder,
                          "series": {"years": list(YEARS), "lines": lines}, "explain": explain,
                          "explain_history": explain_history, "actors": actors.get(k), "twin": twins.get(k),
                          "passport": passport_view(passports.get("base:" + k)),
                          "verification": ({f: (r.get(f) if pd.notna(r.get(f)) else None)
                                            for f in ("sources_confirm", "quote_ok", "early", "mass", "quote", "url")}
                                           if "verified" in r and r.get("verified") is True else None),
                          "description": ({"text": r.description_ru, "generated": True}
                                          if has_ru and isinstance(r.description_ru, str) else None),
                          "why": ({"text": r.why, "generated": True} if "why" in r and isinstance(r.why, str) else None),
                          "advantage": ({"text": card_text[nm.lower()]["advantage"], "generated": True}
                                        if nm.lower() in card_text else None),
                          "case": ({"text": card_text[nm.lower()]["case"], "generated": True}
                                   if nm.lower() in card_text else None), "sources": src})
        results[query] = {"query": query, "area": area, "pool_size": 80 if ("base" in a.top_file or "selected" in a.top_file) else 300 if "pool" in a.top_file else 500, "method": a.method, "cards": cards}
        area_queries = {q for q, _en in AREAS.values()}
        examples.append({"area": area, "query": query, "match": EXAMPLE_MATCH.get(area, []),
                         "label": area if query in area_queries else query})
    model_global = None
    if (REG / "expert_shap_global.json").exists():
        g = json.loads((REG / "expert_shap_global.json").read_text(encoding="utf-8"))
        model_global = [{"label": LABELS.get(x["feature"], x["feature"]), "mean_abs": round(x["mean_abs"], 3),
                         "direction": round(x["direction"], 2)} for x in g[:12]]
    model_global_history = None
    if (REG / "history_shap_global.json").exists():
        g = json.loads((REG / "history_shap_global.json").read_text(encoding="utf-8"))
        model_global_history = [{"label": x["label"], "mean_abs": round(x["mean_abs"], 3),
                                 "direction": round(x["direction"], 2)} for x in g[:10]]
    data = {"registry_size": int(len(cand)), "corpus_snapshot": time.strftime("%Y-%m-%d"), "model_global": model_global,
            "model_global_history": model_global_history,
            "sources": ["OpenAlex", "arXiv", "HF Daily Papers", "Hacker News", "Y Combinator", "пресса"],
            "examples": examples, "results": results}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out} ({out.stat().st_size // 1024} КБ), {time.time() - t0:.0f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
