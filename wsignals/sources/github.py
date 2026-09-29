"""GitHub: практическая реализация технологии в коде. Токен берётся из GITHUB_TOKEN или `gh auth token`."""
from __future__ import annotations

import logging
import math
import os
import subprocess
from datetime import timedelta
from functools import lru_cache

from ..http import AUTH_ERROR, FetchError, get_json, now_utc
from ..schema import Document

API = "https://api.github.com/search/repositories"
log = logging.getLogger("wsignals.github")


@lru_cache(maxsize=1)
def _headers() -> dict:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        try:
            token = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            token = ""
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def fetch(params: dict, **kw):
    """Запрос к поиску GitHub с одной попыткой без токена, если токен отвергнут.

    REPAIR D. Живая проверка показала 100 % отказов адаптера с типом `http_error`, по которому нельзя
    отличить просроченный токен от кривого запроса. Теперь такой отказ типизирован как `auth_error`,
    несёт сообщение сервера, и на него делается ровно одна попытка без заголовка авторизации:
    неисправный токен переводит адаптер в режим меньшего лимита, а не в полный отказ. Токен здесь
    не читается и не печатается."""
    headers = _headers()
    try:
        return get_json(API, params, headers, **kw)
    except FetchError as e:
        if e.error_type != AUTH_ERROR or "Authorization" not in headers:
            raise
        log.warning("GitHub отверг авторизацию (%s): повтор без токена, лимит будет ниже", e)
        anonymous = {k: v for k, v in headers.items() if k != "Authorization"}
        return get_json(API, params, anonymous, **kw)


def stats(query: str) -> dict:
    q = f'"{query}" in:name,description,readme'
    d = fetch({"q": q, "sort": "stars", "per_page": 5}, max_age_days=14)
    since = (now_utc() - timedelta(days=365)).date().isoformat()
    new = fetch({"q": f"{q} created:>{since}", "per_page": 1}, max_age_days=14)
    total = int(d.get("total_count", 0))
    stars = max((r.get("stargazers_count", 0) for r in d.get("items", [])), default=0)
    return {"gh_total": total, "gh_12m": int(new.get("total_count", 0)),
            "gh_new_share": int(new.get("total_count", 0)) / total if total else 0.0,
            "gh_max_stars_log": math.log1p(stars)}


def search(query: str, limit: int = 5) -> list[Document]:
    d = fetch({"q": f'"{query}" in:name,description', "sort": "stars", "per_page": limit}, max_age_days=1)
    return [Document(title=f"{r['full_name']}: {r.get('description') or ''}"[:200], url=r["html_url"], source_name="GitHub",
                     source_type="репозиторий", published=(r.get("created_at") or "")[:10] or None, language="en",
                     extra={"stars": r.get("stargazers_count", 0)}) for r in d.get("items", [])]
