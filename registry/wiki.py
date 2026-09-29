"""Слой Википедии: есть ли у технологии статья или перенаправление в английской Википедии.

    python -m registry.wiki      # -> <registry>/wiki.parquet (key, wiki_title)

Энциклопедия — поздняя ступень лестницы зрелости: статья появляется, когда тема уже сложилась. У команды возраст
статьи в Википедии — один из сильнейших признаков зрелости. Здесь пока только наличие (статья или
перенаправление); возраст по номеру страницы — следующий шаг.

Данные: список всех названий основного пространства (статьи и перенаправления),
dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz (~109 МБ, CC BY-SA), в
<corpus>/wikipedia/enwiki-all-titles-ns0.gz. Название приводится к тому же ключу сущности, что и фразы реестра
(регистр, дефисы, число и порядок слов не важны); уточнение в скобках отрезается: «Tokenization (data security)»
→ «tokenization».
"""
from __future__ import annotations

import gzip
import re
import sys
import time

import pandas as pd

from .corpus import CORPUS, REGISTRY
from .text import entity_key

REG = REGISTRY
DUMP = CORPUS / "wikipedia" / "enwiki-all-titles-ns0.gz"
PAREN = re.compile(r"\s*\([^)]*\)\s*$")
TOK = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


def norm_title(t: str) -> str:
    t = PAREN.sub("", t.replace("_", " ")).lower().replace("-", " ")
    return " ".join(TOK.findall(t))


def main() -> int:
    t0 = time.time()
    keys = set(pd.read_parquet(REG / "all_phrases.parquet", columns=["key"]).key)
    bp = REG / "all_business.parquet"
    if bp.exists():
        keys |= set(pd.read_parquet(bp, columns=["key"]).key)
    found: dict[str, str] = {}
    n = 0
    with gzip.open(DUMP, "rt", encoding="utf-8", errors="replace") as fh:
        next(fh, None)   # заголовок page_title
        for line in fh:
            n += 1
            title = line.rstrip("\n")
            nt = norm_title(title)
            if not nt or nt.count(" ") > 4:
                continue
            k = entity_key(nt)
            if k in keys and k not in found:
                found[k] = title
    out = pd.DataFrame({"key": list(found), "wiki_title": list(found.values())})
    out.to_parquet(REG / "wiki.parquet", index=False)
    print(f"названий Википедии: {n:,}; совпало с сущностями реестра: {len(out):,} из {len(keys):,}; {time.time() - t0:.0f} с".replace(",", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
