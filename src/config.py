from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/signals")


def openalex_params() -> dict:
    """Параметры, которые добавляются к каждому запросу в OpenAlex."""
    params = {}
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO
    return params

GCP_PROJECT = os.getenv("GCP_PROJECT")
BQ_MAX_BYTES_BILLED = int(os.getenv("BQ_MAX_BYTES_BILLED", "50000000000"))
