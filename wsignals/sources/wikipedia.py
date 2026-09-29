"""Википедия: признак зрелости. Отдельная статья, её возраст и просмотры означают, что технология уже в мейнстриме."""
from __future__ import annotations

import math
import re
from datetime import date, timedelta

from ..http import get_json

API = "https://en.wikipedia.org/w/api.php"
VIEWS = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{t}/monthly/{a}/{b}"
STOP = {"a", "an", "the", "of", "for", "and", "in", "on", "to", "with", "ai", "llm"}


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP}


def stats(query: str) -> dict:
    d = get_json(API, {"action": "query", "list": "search", "srsearch": f'"{query}"', "srlimit": 5,
                       "srinfo": "totalhits", "format": "json"}, max_age_days=30)
    hits = d.get("query", {}).get("search", [])
    mentions = int(d.get("query", {}).get("searchinfo", {}).get("totalhits", 0))
    q = _tokens(query)
    best, overlap = None, 0.0
    for h in hits:
        t = _tokens(h["title"])
        ov = len(q & t) / max(1, len(q | t))
        if ov > overlap:
            best, overlap = h["title"], ov
    out = {"wiki_mentions_log": math.log1p(mentions), "wiki_article": 0, "wiki_age": 0.0, "wiki_views_log": 0.0}
    if best and overlap >= 0.5:
        rev = get_json(API, {"action": "query", "prop": "revisions", "titles": best, "rvlimit": 1, "rvdir": "newer",
                             "rvprop": "timestamp", "format": "json", "redirects": 1}, max_age_days=30)
        page = next(iter(rev["query"]["pages"].values()))
        ts = (page.get("revisions") or [{}])[0].get("timestamp")
        title = page.get("title", best).replace(" ", "_")
        end = date(2026, 9, 1)
        try:
            v = get_json(VIEWS.format(t=title, a=(end - timedelta(days=365)).strftime("%Y%m01"), b=end.strftime("%Y%m01")),
                         max_age_days=30)
            views = sum(i.get("views", 0) for i in v.get("items", []))
        except Exception:
            views = 0
        out.update({"wiki_article": 1, "wiki_age": (2026 - int(ts[:4])) if ts else 0.0, "wiki_views_log": math.log1p(views),
                    "wiki_title": page.get("title", best)})
    return out
