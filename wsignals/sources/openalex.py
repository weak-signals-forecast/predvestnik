"""OpenAlex: научные публикации и препринты (включая arXiv). Бесплатно, без ключа, ~10 запросов в секунду."""
from __future__ import annotations

import math
import os
from datetime import date

from ..http import get_json
from ..schema import Document

API = "https://api.openalex.org/works"
YEARS = list(range(2010, 2027))


def _base(query: str) -> dict:
    p = {"filter": f'title_and_abstract.search:"{query}",publication_year:{YEARS[0]}-{YEARS[-1]}'}
    if os.getenv("OPENALEX_MAILTO"):
        p["mailto"] = os.environ["OPENALEX_MAILTO"]
    if os.getenv("OPENALEX_API_KEY"):
        p["api_key"] = os.environ["OPENALEX_API_KEY"]
    return p


def _groups(query: str, by: str) -> dict[str, int]:
    d = get_json(API, {**_base(query), "group_by": by})
    return {str(g["key_display_name"] or g["key"]): int(g["count"]) for g in d.get("group_by", [])}


def stats(query: str, light: bool = False) -> dict:
    """light=True: только годовой ряд, один запрос. Используется на первой ступени каскада открытого запроса."""
    years = _groups(query, "publication_year")
    counts = [years.get(str(y), 0) for y in YEARS]
    total = sum(counts)
    yi = {y: i for i, y in enumerate(YEARS)}
    # 2026 неполный: досчитываем до годового темпа по доле прошедших месяцев на дату снимка
    today = date.today()
    partial = 12 / max(1.0, today.month - 1 + today.day / 30) if today.year == 2026 else 1.0
    c = counts[:]
    c[yi[2026]] = counts[yi[2026]] * partial
    recent = (c[yi[2024]] + c[yi[2025]] + c[yi[2026]]) / 3
    prior = (c[yi[2019]] + c[yi[2020]] + c[yi[2021]]) / 3
    first = next((y for y, n in zip(YEARS, counts) if n >= 3), None)
    done = counts[: yi[2025] + 1]                          # пик только по завершённым годам: неполный 2026 не в счёт
    peak = max(done) if done else 0
    peak_year = YEARS[done.index(peak)] if peak else None
    types = _groups(query, "type") if total and not light else {}
    inst = _groups(query, "authorships.institutions.type") if total and not light else {}
    fields = _groups(query, "primary_topic.field.id") if total and not light else {}
    inst_total = sum(inst.values()) or 1
    return {
        "oa_total": total,
        "oa_recent": recent,
        "oa_prior": prior,
        "oa_growth": math.log((recent + 1) / (prior + 1)),
        "oa_age": (2026 - first) if first else 0,
        "oa_peak_ratio": (c[yi[2025]] + 1) / (peak + 1),   # < 1: тема прошла пик (признак угасшего хайпа)
        "oa_peak_year": peak_year,
        "oa_last_share": c[yi[2025]] / total if total else 0.0,
        "oa_preprint_share": types.get("preprint", 0) / total if total else 0.0,
        "oa_company_share": inst.get("company", 0) / inst_total,
        "oa_fields": len(fields),
        "oa_years": counts,
    }


def search(query: str, limit: int = 25, from_year: int = 2023) -> list[Document]:
    p = _base(query)
    p["filter"] = f'title_and_abstract.search:"{query}",publication_year:{from_year}-2026'
    p.update({"per-page": min(limit, 50), "sort": "relevance_score:desc",
              "select": "id,doi,title,publication_date,type,language,primary_location,cited_by_count,abstract_inverted_index"})
    docs = []
    for w in get_json(API, p).get("results", []):
        loc = w.get("primary_location") or {}
        venue = (loc.get("source") or {}).get("display_name") or "OpenAlex"
        inv = w.get("abstract_inverted_index") or {}
        words = sorted((i, t) for t, idx in inv.items() for i in idx)
        docs.append(Document(
            title=w.get("title") or "", url=w.get("doi") or w["id"], source_name=venue,
            source_type="препринт" if w.get("type") == "preprint" else "научная публикация",
            published=w.get("publication_date"), language=w.get("language") or "en",
            snippet=" ".join(t for _, t in words)[:600], extra={"cited_by": w.get("cited_by_count", 0)}))
    return docs
