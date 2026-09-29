"""Новости через RSS Google News: русскоязычные и зарубежные издания с датой и названием источника."""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from datetime import timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote

from ..http import get, now_utc
from ..schema import Document

RSS = "https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={gl}:{lang}"
PRESS = re.compile(r"(PR Newswire|Business Wire|GlobeNewswire|EIN Presswire|ACCESSWIRE|Newswire|PRWeb|Пресс-релиз)", re.I)


def _items(query: str, when: str | None, lang: str, exact: bool = True) -> list[dict]:
    q = (f'"{query}"' if exact else query) + (f" when:{when}" if when else "")
    hl, gl = ("ru", "RU") if lang == "ru" else ("en-US", "US")
    url = RSS.format(q=quote(q), hl=hl, gl=gl, lang=lang)
    root = ET.fromstring(get(url, max_age_days=7))
    out = []
    for it in root.findall(".//item"):
        try:
            dt = parsedate_to_datetime(it.findtext("pubDate"))
        except (TypeError, ValueError):
            dt = None
        out.append({"title": html.unescape(it.findtext("title") or ""), "link": it.findtext("link") or "",
                    "source": it.findtext("source") or "", "date": dt})
    return out


SATURATED = 100          # RSS отдаёт не больше 100 элементов: столько означает насыщение выдачи


def stats(query: str) -> dict:
    """Счётчики новостей. Свежие за 30 дней считаются по датам той же годовой ленты: это один запрос
    вместо двух, а лента троттлится по секунде на запрос и на живом пути стоит дороже всего.

    Отдельный запрос за 30 дней делается только когда годовая лента насыщена (100 элементов) и
    подсчёт по ней занизил бы свежие новости."""
    year = _items(query, "1y", "en")
    sources = {i["source"] for i in year}
    press = sum(bool(PRESS.search(i["source"])) for i in year)
    if len(year) >= SATURATED:
        month = len(_items(query, "30d", "en"))
    else:
        edge = now_utc() - timedelta(days=30)
        month = sum(1 for i in year if i["date"] and i["date"] >= edge)
    return {
        "news_1y": len(year),
        "news_30d": month,
        "news_sources": len(sources),
        "news_press_share": press / len(year) if year else 0.0,
    }


def search(query: str, lang: str = "ru", when: str = "1y", limit: int = 30, exact: bool = True) -> list[Document]:
    docs = []
    for i in _items(query, when, lang, exact)[:limit]:
        title = re.sub(r"\s+-\s+[^-]+$", "", i["title"])  # Google News дописывает « - Издание»
        docs.append(Document(
            title=title, url=i["link"], source_name=i["source"],
            source_type="пресс-релиз" if PRESS.search(i["source"]) else "новости",
            published=i["date"].date().isoformat() if i["date"] else None, language=lang))
    return docs


def days_since(d) -> float | None:
    return (now_utc() - d).days if d else None
