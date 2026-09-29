"""Ингест статей и препринтов из OpenAlex по направлениям из domains.yaml.

Пример:
    python src/ingest_openalex.py --domain hydrogen --from 2015-01-01 --limit 500
    python src/ingest_openalex.py --all --from 2015-01-01

Результат: data/openalex/<domain>.jsonl, одна запись на документ в единой схеме.
"""
import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import openalex_params  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOMAINS_FILE = ROOT / "src" / "domains.yaml"
OUT_DIR = ROOT / "data" / "openalex"
API = "https://api.openalex.org/works"

SELECT = ",".join([
    "id", "doi", "title", "publication_date", "publication_year", "type",
    "language", "cited_by_count", "authorships", "primary_topic", "concepts",
    "abstract_inverted_index", "primary_location",
])

session = requests.Session()
session.headers["User-Agent"] = "weak-signals-hackathon/0.1"


def load_domains() -> dict:
    with open(DOMAINS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def rebuild_abstract(inv: dict | None) -> str:
    """OpenAlex хранит аннотацию как инвертированный индекс, собираем обратно."""
    if not inv:
        return ""
    positions = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def normalize(work: dict, domain: str, query: str) -> dict:
    """Единая схема документа, общая для всех источников."""
    authors, orgs, countries = [], set(), set()
    for a in work.get("authorships") or []:
        name = (a.get("author") or {}).get("display_name")
        if name:
            authors.append(name)
        for inst in a.get("institutions") or []:
            if inst.get("display_name"):
                orgs.add(inst["display_name"])
            if inst.get("country_code"):
                countries.add(inst["country_code"])
    topic = work.get("primary_topic") or {}
    concepts = [c["display_name"] for c in (work.get("concepts") or []) if c.get("score", 0) > 0.3]
    loc = work.get("primary_location") or {}
    venue = (loc.get("source") or {}).get("display_name")
    return {
        "id": work["id"].rsplit("/", 1)[-1],
        "source": "openalex",
        "doc_type": work.get("type"),
        "date": work.get("publication_date"),
        "year": work.get("publication_year"),
        "title": work.get("title") or "",
        "abstract": rebuild_abstract(work.get("abstract_inverted_index")),
        "authors": authors,
        "orgs": sorted(orgs),
        "countries": sorted(countries),
        "venue": venue,
        "language": work.get("language"),
        "cited_by": work.get("cited_by_count", 0),
        "topic": topic.get("display_name"),
        "subfield": (topic.get("subfield") or {}).get("display_name"),
        "codes": concepts,
        "doi": work.get("doi"),
        "url": work.get("id"),
        "domain": domain,
        "matched_queries": [query],
    }


def is_relevant(doc: dict, spec: dict) -> bool:
    """Фильтр релевантности из domains.yaml. С аннотацией проверяются все группы all_of по заголовку
    и аннотации; без аннотации только первая группа по заголовку и названию темы OpenAlex.
    Любое совпадение с none_of выбрасывает документ."""
    rules = spec.get("relevance")
    if not rules:
        return True
    if doc.get("abstract"):
        text = f"{doc['title']} {doc['abstract']}".lower()
        groups = rules.get("all_of", [])
    else:
        text = f"{doc['title']} {doc.get('topic') or ''} {doc.get('subfield') or ''}".lower()
        groups = rules.get("all_of", [])[:1]
    if any(re.search(x, text) for x in rules.get("none_of", [])):
        return False
    return all(re.search(x, text) for x in groups)


def fetch_query(query: str, date_from: str, date_to: str, limit: int | None):
    """Курсорная пагинация по одному поисковому запросу."""
    filters = [
        f"title_and_abstract.search:{query}",
        f"from_publication_date:{date_from}",
        f"to_publication_date:{date_to}",
        "type:article|preprint|review",
    ]
    params = {
        "filter": ",".join(filters),
        "select": SELECT,
        "per-page": 200,
        "cursor": "*",
        **openalex_params(),
    }
    fetched = 0
    while True:
        for attempt in range(5):
            r = session.get(API, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            break
        else:
            raise RuntimeError(f"OpenAlex недоступен для запроса: {query}")
        payload = r.json()
        for w in payload["results"]:
            yield w
            fetched += 1
            if limit and fetched >= limit:
                return
        cursor = payload["meta"].get("next_cursor")
        if not cursor:
            return
        params["cursor"] = cursor
        time.sleep(0.12)  # держимся ниже 10 запросов в секунду


def ingest_domain(domain: str, spec: dict, date_from: str, date_to: str, limit: int | None) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{domain}.jsonl"
    docs: dict[str, dict] = {}
    n_dropped = 0
    for q in spec["queries"]:
        n_new = 0
        for w in fetch_query(q, date_from, date_to, limit):
            d = normalize(w, domain, q)
            if d["id"] in docs:
                docs[d["id"]]["matched_queries"].append(q)
                continue
            if not is_relevant(d, spec):
                n_dropped += 1
                continue
            docs[d["id"]] = d
            n_new += 1
        print(f"  [{domain}] {q!r}: +{n_new} новых, всего {len(docs)}")
    print(f"  [{domain}] фильтр релевантности отсеял {n_dropped}")
    with open(out, "w", encoding="utf-8") as f:
        for d in docs.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[{domain}] сохранено {len(docs)} документов в {out.relative_to(ROOT)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", help="ключ направления из domains.yaml")
    ap.add_argument("--all", action="store_true", help="все направления")
    ap.add_argument("--from", dest="date_from", default="2015-01-01")
    ap.add_argument("--to", dest="date_to", default=date.today().isoformat())
    ap.add_argument("--limit", type=int, help="максимум документов на один запрос (для отладки)")
    args = ap.parse_args()

    domains = load_domains()
    if args.all:
        targets = list(domains)
    elif args.domain:
        if args.domain not in domains:
            sys.exit(f"Нет направления {args.domain!r}. Доступны: {', '.join(domains)}")
        targets = [args.domain]
    else:
        ap.error("укажите --domain или --all")

    for name in targets:
        print(f"== {domains[name]['title']} ==")
        ingest_domain(name, domains[name], args.date_from, args.date_to, args.limit)


if __name__ == "__main__":
    main()
