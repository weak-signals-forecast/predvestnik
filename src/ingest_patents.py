"""Ингест патентов из Google Patents Public Data (BigQuery) по CPC-кодам из domains.yaml.

Пример:
    python src/ingest_patents.py --domain hydrogen --dry-run      # только оценка объёма запроса
    python src/ingest_patents.py --domain hydrogen --from 2015-01-01
    python src/ingest_patents.py --all --from 2015-01-01

Нужны GCP_PROJECT и GOOGLE_APPLICATION_CREDENTIALS в .env.
Результат: data/patents/<domain>.jsonl, одна запись на патентное семейство в единой схеме.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import yaml
from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BQ_MAX_BYTES_BILLED, GCP_PROJECT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOMAINS_FILE = ROOT / "src" / "domains.yaml"
OUT_DIR = ROOT / "data" / "patents"

# Одна запись на семейство: берём самую раннюю публикацию семейства,
# чтобы не считать одно изобретение несколько раз (заявка, патент, страны).
SQL = """
WITH hits AS (
  SELECT
    p.family_id,
    p.publication_number,
    p.country_code,
    p.kind_code,
    p.publication_date,
    p.filing_date,
    p.priority_date,
    (SELECT t.text FROM UNNEST(p.title_localized) t WHERE t.language = 'en' LIMIT 1) AS title,
    (SELECT a.text FROM UNNEST(p.abstract_localized) a WHERE a.language = 'en' LIMIT 1) AS abstract,
    ARRAY(SELECT DISTINCT i.name FROM UNNEST(p.inventor_harmonized) i) AS inventors,
    ARRAY(SELECT DISTINCT a.name FROM UNNEST(p.assignee_harmonized) a) AS assignees,
    ARRAY(SELECT DISTINCT a.country_code FROM UNNEST(p.assignee_harmonized) a WHERE a.country_code != '') AS assignee_countries,
    ARRAY(SELECT c.code FROM UNNEST(p.cpc) c) AS cpc_codes,
    ARRAY(SELECT c.code FROM UNNEST(p.cpc) c WHERE c.inventive) AS cpc_inventive,
    ROW_NUMBER() OVER (PARTITION BY p.family_id ORDER BY p.publication_date, p.publication_number) AS rn
  FROM `patents-public-data.patents.publications` p
  WHERE p.publication_date BETWEEN @date_from AND @date_to
    AND EXISTS (
      SELECT 1 FROM UNNEST(p.cpc) c
      WHERE {cpc_condition}
    )
)
SELECT * EXCEPT(rn) FROM hits WHERE rn = 1 AND title IS NOT NULL
"""


def load_domains() -> dict:
    with open(DOMAINS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cpc_condition(codes: list[str]) -> str:
    """CPC в таблице хранится как 'C01B3/24'. Префикс с '/' в конце ловит всю группу."""
    return " OR ".join(f"STARTS_WITH(c.code, '{code}')" for code in codes)


def int_date(s: str) -> int:
    return int(s.replace("-", ""))


def fmt_date(n: int | None) -> str | None:
    if not n:
        return None
    s = str(n)
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def normalize(row, domain: str) -> dict:
    # Приоритетная дата это самый ранний момент, когда изобретение зафиксировано.
    # Для слабых сигналов именно она, а не дата публикации, отражает время появления.
    return {
        "id": row.publication_number,
        "source": "google_patents",
        "doc_type": "patent",
        "date": fmt_date(row.priority_date) or fmt_date(row.filing_date) or fmt_date(row.publication_date),
        "year": int(str(row.priority_date or row.filing_date or row.publication_date)[:4]),
        "publication_date": fmt_date(row.publication_date),
        "title": row.title or "",
        "abstract": row.abstract or "",
        "authors": list(row.inventors),
        "orgs": list(row.assignees),
        "countries": list(row.assignee_countries),
        "venue": row.country_code,          # патентное ведомство публикации
        "kind": row.kind_code,
        "family_id": row.family_id,
        "codes": list(row.cpc_codes),
        "codes_inventive": list(row.cpc_inventive),
        "url": f"https://patents.google.com/patent/{row.publication_number}",
        "domain": domain,
    }


def build_job(client, spec, date_from, date_to, dry_run):
    sql = SQL.format(cpc_condition=cpc_condition(spec["cpc"]))
    cfg = bigquery.QueryJobConfig(
        dry_run=dry_run,
        use_query_cache=True,
        maximum_bytes_billed=BQ_MAX_BYTES_BILLED,
        query_parameters=[
            bigquery.ScalarQueryParameter("date_from", "INT64", int_date(date_from)),
            bigquery.ScalarQueryParameter("date_to", "INT64", int_date(date_to)),
        ],
    )
    return client.query(sql, job_config=cfg)


def ingest_domain(client, domain, spec, date_from, date_to, dry_run) -> None:
    job = build_job(client, spec, date_from, date_to, dry_run=True)
    gb = job.total_bytes_processed / 1e9
    print(f"  [{domain}] запрос обработает ~{gb:.1f} ГБ (лимит {BQ_MAX_BYTES_BILLED/1e9:.0f} ГБ)")
    if dry_run:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{domain}.jsonl"
    job = build_job(client, spec, date_from, date_to, dry_run=False)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in job.result():
            f.write(json.dumps(normalize(row, domain), ensure_ascii=False) + "\n")
            n += 1
    billed = (job.total_bytes_billed or 0) / 1e9
    print(f"[{domain}] сохранено {n} семейств в {out.relative_to(ROOT)}, оплачено {billed:.1f} ГБ")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--from", dest="date_from", default="2015-01-01")
    ap.add_argument("--to", dest="date_to", default=date.today().isoformat())
    ap.add_argument("--dry-run", action="store_true", help="только оценить объём, ничего не скачивать")
    args = ap.parse_args()

    if not GCP_PROJECT:
        sys.exit("В .env не задан GCP_PROJECT. См. инструкцию по настройке BigQuery.")
    client = bigquery.Client(project=GCP_PROJECT)

    domains = load_domains()
    targets = list(domains) if args.all else [args.domain] if args.domain else ap.error("укажите --domain или --all")
    for name in targets:
        print(f"== {domains[name]['title']} ==")
        ingest_domain(client, name, domains[name], args.date_from, args.date_to, args.dry_run)


if __name__ == "__main__":
    main()
