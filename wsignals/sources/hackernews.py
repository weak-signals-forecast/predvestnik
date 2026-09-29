"""Hacker News (Algolia API): ранний отклик разработческого сообщества. Доверие пониженное, только как индикатор."""
from __future__ import annotations

import time

from ..http import get_json
from ..schema import Document

API = "https://hn.algolia.com/api/v1/search"


def _count(query: str, since: int | None = None, until: int | None = None) -> int:
    filters = []
    if since:
        filters.append(f"created_at_i>{since}")
    if until:
        filters.append(f"created_at_i<{until}")
    p = {"query": f'"{query}"', "tags": "story", "hitsPerPage": 0}
    if filters:
        p["numericFilters"] = ",".join(filters)
    return int(get_json(API, p, max_age_days=7).get("nbHits", 0))


def stats(query: str) -> dict:
    now = int(time.time())
    y = 365 * 86400
    last = _count(query, since=now - y)
    prior = _count(query, since=now - 3 * y, until=now - y)
    total = _count(query)
    return {"hn_12m": last, "hn_prior24m": prior, "hn_total": total}


def search(query: str, limit: int = 10) -> list[Document]:
    d = get_json(API, {"query": f'"{query}"', "tags": "story", "hitsPerPage": limit}, max_age_days=1)
    return [Document(title=h.get("title") or "", url=h.get("url") or f"https://news.ycombinator.com/item?id={h['objectID']}",
                     source_name="Hacker News", source_type="сообщество", published=(h.get("created_at") or "")[:10] or None,
                     language="en", extra={"points": h.get("points", 0)}) for h in d.get("hits", [])]
