"""Чтение офлайн-корпуса (~/gpb_corpus) для построения реестра.

Источники реестра: OpenAlex по подобластям, arXiv, Hugging Face Daily Papers.

Фильтр качества OpenAlex (по признакам, без списков слов):
  * записи, чей источник arXiv, пропускаются: arXiv скачан целиком отдельно, иначе одна работа считалась
    бы дважды и «подтверждение двумя источниками» было бы фиктивным;
  * самиздат пропускается: ни у одного автора нет организации и нет ни одной ссылки на литературу.
    Это в основном поток Zenodo, OSF и записи без источника: методички, каталоги, «книги»-пересказы.
    В 2026 году это ~42 % записей OpenAlex, в 2024 — ~16 %; без фильтра рост свежих лет раздут мусором. Точечная выгрузка
openalex_targeted сюда НЕ входит: она построена по запросам к 100 сигналам организаторов, и её
использование при построении реестра было бы утечкой разметки в оценку.
"""
from __future__ import annotations

import ast
import glob
import gzip
import json
import os
import re
from pathlib import Path

CORPUS = Path(os.environ.get("GPB_CORPUS", Path.home() / "gpb_corpus"))
# папка сборки реестра; отдельная версия — для сравнения «до/после» без перезаписи
REGISTRY = Path(os.environ.get("GPB_REGISTRY", CORPUS / "registry"))
# год среза для «машины времени»: реестр строится так, будто сейчас конец этого года (по умолчанию — 2026)
CUTOFF = int(os.environ.get("REGISTRY_CUTOFF", "2026"))
SOURCES = ("openalex", "arxiv", "hf")
YEARS = tuple(range(2012, 2027))


def files(corpus: Path = CORPUS) -> list[tuple[str, str]]:
    """(источник, путь) всех частей корпуса."""
    out = [("openalex", f) for f in sorted(glob.glob(str(corpus / "openalex/subfield=*/year=*/*.jsonl.gz")))]
    out += [("arxiv", f) for f in sorted(glob.glob(str(corpus / "arxiv/set=*/*.jsonl.gz")))]
    out += [("hf", f) for f in sorted(glob.glob(str(corpus / "hf/daily_papers/*.jsonl.gz")))]
    return out


def _skip_openalex(r: dict) -> bool:
    src = r.get("source") or {}
    name = (src.get("name") or "") if isinstance(src, dict) else str(src)
    if name.startswith("arXiv"):
        return True
    au = r.get("authorships") or []
    return not any(a.get("institutions") for a in au if isinstance(a, dict)) and not (r.get("references") or [])


def _lines(path: str):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            yield from fh
    except (EOFError, OSError):
        return


ARXIV_SETS = ("cs", "stat", "eess", "q-fin", "econ", "quant-ph")   # порядок = приоритет владельца


def _arxiv_cats(cats) -> list[str]:
    try:
        v = ast.literal_eval(cats) if isinstance(cats, str) else cats
        return [c for c in (v or []) if c]
    except (ValueError, SyntaxError):
        return re.findall(r"[a-z\-]+(?:\.[A-Za-z\-]+)?", cats or "")


def _arxiv_owner(cats: list[str]) -> str | None:
    """Раздел выгрузки, в котором статью считаем: раздел основной категории, иначе первый из наших по приоритету."""
    arch = [c.split(".")[0] for c in cats]
    if arch and arch[0] in ARXIV_SETS:
        return arch[0]
    return next((s for s in ARXIV_SETS if s in arch), None)


def docs(source: str, path: str):
    """Документы файла: dict(year, group, title, text). group — подобласть OpenAlex или основная категория arXiv."""
    if source == "openalex":
        sub = re.search(r"subfield=(\d+)", path).group(1)
        for l in _lines(path):
            r = json.loads(l)
            if _skip_openalex(r):
                continue
            kw = " ; ".join(k for k in r.get("keywords") or [] if k)
            yield {"year": r.get("year"), "group": f"oa:{sub}", "title": r.get("title") or "",
                   "text": " . ".join([r.get("title") or "", r.get("abstract") or "", kw]), "id": r.get("id"),
                   "doi": r.get("doi")}
    elif source == "arxiv":
        set_name = re.search(r"set=([^/]+)", path).group(1).split(":")[-1]
        seen: set[str] = set()   # в выгрузке OAI бывают повторы записи при обновлении версии
        for l in _lines(path):
            r = json.loads(l)
            cats = _arxiv_cats(r.get("categories"))
            if r["id"] in seen or _arxiv_owner(cats) != set_name:
                continue   # статья с перекрёстной категорией учитывается один раз, в разделе-владельце
            seen.add(r["id"])
            y = int((r.get("created") or "0")[:4] or 0)
            yield {"year": y, "group": "ax:" + (cats[0] if cats else ""), "title": r.get("title") or "",
                   "text": (r.get("title") or "") + " . " + (r.get("abstract") or ""), "id": "arXiv:" + r["id"],
                   "doi": r.get("doi")}
    elif source == "hf":
        for l in _lines(path):
            r = json.loads(l)
            kw = " ; ".join(r.get("ai_keywords") or [])
            yield {"year": int(r["date"][:4]), "group": "hf", "title": (r.get("title") or "") + " ; " + kw,
                   "text": " . ".join([r.get("title") or "", r.get("summary") or "", kw]),
                   "id": "arXiv:" + str(r.get("arxiv_id")), "doi": None}
