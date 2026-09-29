"""Объединение трёх слоёв детекторов в один ранжированный список сигналов с карточками.

    python src/signals.py [--top 40]

Слои: темы (topics.csv, weak_signal_score), рождения кластеров (windows.csv), подтемы-сироты
(orphan_births.csv), всплески терминов (terms.csv). Единица объединения: множество документов.
Кандидаты от разных детекторов, чьи множества документов пересекаются (коэффициент перекрытия ≥ 0.5),
склеиваются в одну карточку. Чем больше независимых детекторов указали на сигнал, тем выше он в списке.

Карточки собираются из данных по шаблонам, без LLM. Выход: data/signals/signals.md, signals.json, signals.csv
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster import OUT as CL, WINDOWS, YEARS, PARTIAL_YEAR_FACTOR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("SIGNALS_DIR", ROOT / "data" / "signals"))
DOMAIN_TITLES = {"hydrogen": "Водород и пиролиз метана", "ccus_methane": "CCUS и метановые утечки",
                 "lng_gaschem": "СПГ и газохимия", "pipelines": "Трубопроводы"}
INDUSTRY = ["sinopec", "petrochina", "cnpc", "china national petroleum", "china national offshore", "cnooc", "aramco",
            "shell", "bp ", "totalenergies", "total s.a", "exxon", "chevron", "equinor", "eni ", "gazprom", "rosneft",
            "lukoil", "novatek", "baker hughes", "schlumberger", "slb", "halliburton", "linde", "air liquide",
            "air products", "siemens", "general electric", "mitsubishi", "petrobras", "adnoc", "qatarenergy", "petronas",
            "woodside", "santos", "repsol", "omv", "conocophillips", "occidental", "wintershall", "uniper", "engie",
            "snam", "enagás", "national grid", "kinder morgan", "tc energy", "enbridge", "honeywell", "johnson matthey",
            "topsoe", "haldor", "toyota", "hyundai", "kawasaki", "chiyoda", "jgc", "technip", "saipem", "wood plc",
            "climeworks", "carbon engineering", "occidental", "ghgsat", "kayrros", "bridger", "tanaka", "sumitomo"]
SPARK = "▁▂▃▄▅▆▇█"


def sparkline(counts) -> str:
    m = max(counts) or 1
    return "".join(SPARK[min(7, int(c / m * 7.999))] for c in counts)


def overlap(a: set, b: set) -> float:
    return len(a & b) / max(1, min(len(a), len(b)))


def year_counts(years: pd.Series) -> list[float]:
    c = years.value_counts().reindex(YEARS, fill_value=0).to_numpy(dtype=float)
    c[-1] *= PARTIAL_YEAR_FACTOR
    return c.tolist()


# ---------- кандидаты от каждого слоя ----------

def candidates(df: pd.DataFrame, top_terms: int):
    """Каждый кандидат: слой, имя, множество id документов, нормированный ранг 0..1 (1 = лучший), пояснение."""
    cands = []
    by_topic = df.groupby("topic")["id"].agg(set)

    topics = pd.read_csv(CL / "topics.csv")
    topics = topics[topics["n"] <= 400].reset_index(drop=True)  # больше 400 это уже сильный тренд
    for i, t in topics.iterrows():
        cands.append(dict(layer="тема", name=t["name"], docs=by_topic.get(t["topic"], set()),
                          rank=1 - i / len(topics), score=t["weak_signal_score"],
                          why=f"Тема растёт: лог-рост {t['growth']:+.2f} (2023-25 к 2019-21), у {t['precedent_gap']:.0%} "
                              f"соседей свежих статей нет более ранних прецедентов, рождение темы {t['birth_year']}."))

    win = pd.read_csv(CL / "windows.csv")
    born = win[win["born"] & (win["n"] >= 15) & win["window"].isin([f"{a}-{b}" for a, b in WINDOWS[-3:]])]
    born = born.sort_values("precedent_sim").reset_index(drop=True)
    for i, w in born.iterrows():
        cands.append(dict(layer="рождение кластера", name=w["name"], docs=set(str(w["members"]).split()),
                          rank=1 - i / max(1, len(born)), score=-w["precedent_sim"],
                          why=f"В окне {w['window']} кластер из {w['n']} статей появился без родителя: сходство с лучшим "
                              f"кандидатом {w['parent_sim']:.2f} не выше, чем с соседней темой того же окна {w['sibling_sim']:.2f}."))

    orph = pd.read_csv(CL / "orphan_births.csv")
    orph = orph[orph["window"].isin([f"{a}-{b}" for a, b in WINDOWS[-3:]]) & (orph["n"] >= 10)]
    orph = orph.sort_values("precedent_sim").reset_index(drop=True)
    for i, o in orph.iterrows():
        cands.append(dict(layer="подтема без предшественников", name=o["name"], docs=set(str(o["members"]).split()),
                          rank=1 - i / max(1, len(orph)), score=-o["precedent_sim"],
                          why=f"В окне {o['window']} группа из {o['n']} статей, у которых сходство с любой более ранней "
                              f"работой {o['precedent_sim']:.2f} попадает в нижние 20% окна."))

    terms = pd.read_csv(CL / "terms.csv").head(top_terms).reset_index(drop=True)
    text = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).str.lower()
    cv = CountVectorizer(ngram_range=(1, 3), vocabulary=list(terms["term"]), binary=True,
                         token_pattern=r"(?u)\b[a-z][a-z0-9\-]{2,}\b")
    M = cv.fit_transform(text).tocsc()
    ids = df["id"].to_numpy()
    for i, t in terms.iterrows():
        rows = M[:, i].nonzero()[0]
        cands.append(dict(layer="всплеск термина", name=t["term"], docs=set(ids[rows]),
                          rank=1 - i / len(terms), score=t["score"],
                          why=f"Термин «{t['term']}»: {t['prior_docs']} документов в 2017-2019, {t['recent_docs']} в 2023-2025, "
                              f"сконцентрирован в немногих темах (концентрация {1 - t['topic_entropy']:.2f})."))
    return cands


# ---------- склейка ----------

def merge(cands: list[dict], threshold: float = 0.5) -> list[dict]:
    cands = sorted(cands, key=lambda c: -c["rank"])
    cards = []
    for c in cands:
        if len(c["docs"]) < 8:
            continue
        for card in cards:
            if overlap(card["core"], c["docs"]) >= threshold:
                card["docs"] |= c["docs"]
                card["evidence"].append(c)
                break
        else:
            cards.append(dict(core=set(c["docs"]), docs=set(c["docs"]), evidence=[c]))
    for card in cards:
        layers = {e["layer"] for e in card["evidence"]}
        # Ранг внутри слоя плюс премия за каждый независимый слой сверх первого.
        best_by_layer = {}
        for e in card["evidence"]:
            best_by_layer[e["layer"]] = max(best_by_layer.get(e["layer"], 0), e["rank"])
        card["score"] = sum(best_by_layer.values()) + 0.5 * (len(layers) - 1)
        card["layers"] = sorted(layers)
        # Имя: самый короткий термин, если есть всплеск термина, иначе имя лучшего свидетельства.
        terms = sorted((e for e in card["evidence"] if e["layer"] == "всплеск термина"), key=lambda e: -e["rank"])
        if terms:
            name = terms[0]["name"]
            longer = [e["name"] for e in terms[1:] if len(e["name"]) > len(name) + 3]
            if (len(name) <= 5 or "-" in name and len(name) <= 8) and longer:
                name = f"{name} ({longer[0]})"
            card["name"] = name
        else:
            card["name"] = card["evidence"][0]["name"]
    return sorted(cards, key=lambda c: -c["score"])


# ---------- карточки ----------

def build_card(rank: int, card: dict, df: pd.DataFrame) -> dict:
    sub = df[df["id"].isin(card["docs"])]
    counts = year_counts(sub["year"])
    yi = {y: i for i, y in enumerate(YEARS)}
    recent = sum(counts[yi[2023]:yi[2025] + 1]) / 3
    prior = sum(counts[yi[2019]:yi[2021] + 1]) / 3
    growth = math.log((recent + 1) / (prior + 1))
    birth = next((y for y, n in zip(YEARS, counts) if n >= 3), YEARS[0])
    doms = Counter(d for ds in sub["domains"] for d in ds)
    orgs = Counter(o for os in sub["orgs"] for o in os)
    countries = Counter(c for cs in sub["countries"] for c in cs)
    rec_mask = sub["year"] >= 2023
    c_rec = Counter(c for cs in sub[rec_mask]["countries"] for c in cs)
    c_pri = Counter(c for cs in sub[~rec_mask]["countries"] for c in cs)
    cn_rec = c_rec["CN"] / max(1, sum(c_rec.values()))
    cn_pri = c_pri["CN"] / max(1, sum(c_pri.values()))
    industry = sorted({o for o in orgs if any(k in o.lower() for k in INDUSTRY)})
    cross = float(sub["cross_domain"].mean()) if "cross_domain" in sub else 0.0
    # Примеры: документы, на которые указало больше всего свидетельств, из ядра сигнала.
    votes = Counter(i for e in card["evidence"] for i in e["docs"])
    core = sub[sub["id"].isin(card["core"])].assign(votes=lambda d: d["id"].map(votes))
    examples = core.sort_values(["votes", "cited_by", "year"], ascending=[False, False, False]).head(3)

    why = [e["why"] for e in card["evidence"]]
    signals = []
    if len(card["layers"]) >= 2:
        signals.append(f"Независимо подтверждено {len(card['layers'])} детекторами: {', '.join(card['layers'])}.")
    if cn_rec - cn_pri >= 0.15:
        signals.append(f"Доля китайских аффилиаций выросла с {cn_pri:.0%} до {cn_rec:.0%}.")
    if industry:
        signals.append(f"Уже подхвачено индустрией: {', '.join(industry[:5])}.")
    else:
        signals.append("Пока только академические группы, индустриальных заявителей нет: ранняя стадия.")
    if cross >= 0.3:
        signals.append(f"Мост между направлениями: {cross:.0%} ближайших соседей из других областей.")
    if counts[yi[2025]] > 2 * max(1, counts[yi[2023]]):
        signals.append("Годовой темп с 2023 вырос более чем вдвое.")

    return dict(
        rank=rank, name=card["name"], score=round(card["score"], 2), layers=card["layers"],
        domain=DOMAIN_TITLES.get(doms.most_common(1)[0][0], "") if doms else "",
        n_docs=len(sub), birth_year=birth, growth=round(growth, 2),
        counts=[int(round(c)) for c in counts], sparkline=sparkline(counts),
        top_orgs=[o for o, _ in orgs.most_common(5)],
        top_countries=[f"{k} {v}" for k, v in countries.most_common(5)],
        china_share_recent=round(cn_rec, 2), industry=industry[:8],
        detectors=why, signals=signals,
        examples=[dict(title=r["title"], year=int(r["year"]), url=r["url"],
                       venue=r["venue"] if isinstance(r.get("venue"), str) else None) for _, r in examples.iterrows()],
        aliases=sorted({", ".join(e["name"].split(", ")[:4]) for e in card["evidence"]} - {card["name"]}),
        doc_ids=sorted(card["docs"]), core_ids=sorted(card["core"]),
    )


def render_md(cards: list[dict]) -> str:
    L = ["# Слабые сигналы: объединённый список\n",
         f"Годы {YEARS[0]}-{YEARS[-1]}, последний год досчитан до годового темпа. "
         "Сигналы от четырёх детекторов склеены по пересечению множеств документов; "
         "чем больше независимых детекторов, тем выше место.\n"]
    for c in cards:
        L.append(f"\n---\n\n## {c['rank']}. {c['name']}\n")
        L.append(f"**{c['domain']}** · {c['n_docs']} документов · с {c['birth_year']} · рост {c['growth']:+.2f} · "
                 f"score {c['score']} · слои: {', '.join(c['layers'])}\n")
        L.append(f"Динамика `{c['sparkline']}` {' '.join(map(str, c['counts']))}\n")
        if c["aliases"]:
            L.append(f"Также известно как: {'; '.join(a[:60] for a in c['aliases'][:4])}\n")
        L.append("**Почему это сигнал**")
        for s in c["signals"] + c["detectors"]:
            L.append(f"- {s}")
        L.append(f"\n**Кто:** {'; '.join(c['top_orgs'][:4])}  \n**Страны:** {', '.join(c['top_countries'])}")
        if c["industry"]:
            L.append(f"**Индустрия:** {', '.join(c['industry'])}")
        L.append("\n**Примеры**")
        for e in c["examples"]:
            L.append(f"- [{e['title']}]({e['url']}) ({e['year']}, {e['venue'] or 'без площадки'})")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--top-terms", type=int, default=150)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CL / "doc_topics.parquet")
    cands = candidates(df, args.top_terms)
    print("кандидатов по слоям:", dict(Counter(c["layer"] for c in cands)))
    cards = merge(cands)
    print(f"карточек после склейки {len(cards)}, из них с ≥2 слоями {sum(len(c['layers']) >= 2 for c in cards)}")
    built = [build_card(i + 1, c, df) for i, c in enumerate(cards[:args.top])]

    (OUT / "signals.md").write_text(render_md(built), encoding="utf-8")
    (OUT / "signals.json").write_text(json.dumps(built, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in b.items() if not isinstance(v, (list, dict))} | {"layers": ", ".join(b["layers"])}
                  for b in built]).to_csv(OUT / "signals.csv", index=False)
    print(f"результаты в {OUT.relative_to(ROOT)}")
    for b in built[:15]:
        print(f"  {b['rank']:2d}. {b['name'][:45]:45s} {b['domain'][:22]:22s} n={b['n_docs']:4d} слои={len(b['layers'])} {b['sparkline']}")


if __name__ == "__main__":
    main()
