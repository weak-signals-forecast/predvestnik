"""Вариант В, лексическая часть: бизнес-слой реестра (стартапы, запуски продуктов, пресса) без языковой модели.

    python -m registry.business     # -> <corpus>/registry/business.parquet

Источники и что из них берём:
  hn     — истории Hacker News (open-index/hacker-news): заголовок, год; рейтинг ≥ HN_MIN_SCORE (отсекает
           основную массу спама). Префиксы «Show HN:», «Launch HN:» срезаются, запуск продукта помечается.
  yc     — компании Y Combinator: однострочник, описание, теги; год — год набора.
  press  — карты сайтов прессы, только записи с надёжной датой (из адреса статьи или новостной карты);
           без котировок, страниц монет, научных сайтов и вакансий. Текст — заголовок или адрес статьи.

Словарь: фразы свежих лет (2023–2026) с документной частотой ≥ MIN_DF хотя бы в одном источнике. Для каждой
фразы — ряды по годам и источникам, доли на число документов года (как в научном слое). Кандидат:
  hn_recent + yc_recent + press_recent >= MIN_VOLUME,
  рост доли в HN (R к B) >= MIN_GROWTH или впервые встречается с NEW_SINCE, или есть свежие компании YC,
  не массовая (доля свежих заголовков HN <= MAX_RATE), не общая, не обрывок.
К каждой фразе присоединяется научный ряд из all_phrases.parquet: так видна лестница «наука → бизнес».
Пороги зафиксированы до сравнения со списком организаторов.
"""
from __future__ import annotations

import collections
import glob
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import CORPUS, REGISTRY
from .text import candidates, entity_key

REG = REGISTRY
YEARS = tuple(range(2012, 2027))
R, B = (2024, 2025, 2026), (2019, 2020, 2021, 2022)
VOCAB_YEARS = (2023, 2024, 2025, 2026)
SRC = ("hn", "yc", "press")
HN_MIN_SCORE = 2
MIN_DF = 3
MIN_VOLUME = 5
MIN_GROWTH = 1.0
NEW_SINCE = 2022
MAX_RATE = 1e-3
FRAGMENT_SHARE = 0.8
NOT_PRESS = re.compile(r"frontiersin|elib\.uni|pubs\.rsc|science\.org|elsevierpure|ijiemjournal|is\.mpg|sei\.cmu|"
                       r"ibecbarcelona|coinpaprika|finance\.yahoo|jobswithstartups|app\.fundz|trysignalbase")
YC_BATCH = re.compile(r"\(?\byc\s*[wsfx]\d{2}\b\)?", re.I)   # «(YC S24)» в заголовках запусков — служебная разметка
HN_PREFIX = re.compile(r"^\s*(show|launch|ask|tell)\s+hn\s*[:\-–—]\s*", re.I)
GENERIC = set("""ai app apps tool tools platform startup startups company companies open source new free best way ways
guide how why what use using build building built day year years world people thing things time""".split())


def stream():
    """(источник, год, текст, признак запуска)."""
    for f in sorted(glob.glob(str(CORPUS / "hn/*.jsonl.gz"))):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for l in fh:
                r = json.loads(l)
                if (r.get("score") or 0) < HN_MIN_SCORE or not r.get("title"):
                    continue
                t = r["title"]
                launch = bool(HN_PREFIX.match(t))
                yield "hn", int(r["time"][:4]), YC_BATCH.sub(" ", HN_PREFIX.sub("", t)), launch
    for f in glob.glob(str(CORPUS / "yc/*.json")):
        for r in json.load(open(f, encoding="utf-8")):
            m = re.search(r"(20\d\d)", r.get("batch") or "")
            if not m:
                continue
            yield "yc", int(m.group(1)), " . ".join(str(r.get(k) or "") for k in ("one_liner", "long_description", "tags")), True
    for f in glob.glob(str(CORPUS / "sitemaps/*/*.jsonl.gz")):
        if NOT_PRESS.search(Path(f).parent.name):
            continue
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for l in fh:
                    r = json.loads(l)
                    if r.get("date_source") not in ("url", "news"):
                        continue
                    slug = r["url"].rstrip("/").rsplit("/", 1)[-1].replace("-", " ").replace("_", " ")
                    slug = re.sub(r"\.(html?|php|aspx?)$", "", slug)
                    yield "press", int(r["date"][:4]), r.get("title") or slug, False
        except (EOFError, OSError):
            continue


def build() -> pd.DataFrame:
    t0 = time.time()
    si, yi = {s: i for i, s in enumerate(SRC)}, {y: i for i, y in enumerate(YEARS)}
    # проход 1: словарь по свежим годам (заголовки короткие, память позволяет считать напрямую)
    vc = {s: collections.Counter() for s in SRC}
    for src, y, text, _ in stream():
        if y in VOCAB_YEARS:
            vc[src].update(candidates(text))
    # сущности: варианты фразы склеиваются по ключу (как в научном слое), название — самый частый вариант
    keep = {ph for s in SRC for ph, c in vc[s].items() if c >= (2 if s == "yc" else MIN_DF)}
    freq = {ph: sum(vc[s][ph] for s in SRC) for ph in keep}
    best: dict[str, tuple[int, str]] = {}
    for ph in keep:
        k = entity_key(ph)
        if k not in best or freq[ph] > best[k][0]:
            best[k] = (freq[ph], ph)
    keys = sorted(best)
    kid = {k: i for i, k in enumerate(keys)}
    vocab = [best[k][1] for k in keys]
    vid = {ph: kid[entity_key(ph)] for ph in keep}
    del vc
    print(f"словарь бизнес-слоя: {len(vocab):,} фраз, {time.time() - t0:.0f} с".replace(",", " "), flush=True)
    # проход 2: ряды
    df = np.zeros((len(vocab), len(SRC), len(YEARS)), dtype=np.int32)
    launches = np.zeros(len(vocab), dtype=np.int32)
    N = np.zeros((len(SRC), len(YEARS)), dtype=np.float64)
    for src, y, text, launch in stream():
        if y not in yi:
            continue
        N[si[src], yi[y]] += 1
        ids = list({i for ph in candidates(text) if (i := vid.get(ph)) is not None})
        if ids:
            df[ids, si[src], yi[y]] += 1
            if launch and src == "hn" and y in R:
                launches[ids] += 1
    print(f"ряды: {time.time() - t0:.0f} с; документов по источникам: " +
          ", ".join(f"{s} {int(N[si[s]].sum()):,}" for s in SRC).replace(",", " "), flush=True)

    np.savez_compressed(REG / "business_series.npz", df=df, keys=np.array(keys), sources=np.array(SRC),
                        years=np.array(YEARS), totals=N)   # ряды по годам — для графиков в карточке
    iR, iB = [yi[y] for y in R], [yi[y] for y in B]
    dR, dB = df[:, :, iR].sum(axis=2).astype(float), df[:, :, iB].sum(axis=2).astype(float)
    nR, nB = N[:, iR].sum(axis=1), N[:, iB].sum(axis=1)
    h = si["hn"]
    growth = np.log2(((dR[:, h] + 1) / nR[h]) / ((dB[:, h] + 1) / nB[h]))
    growth = np.where(dR[:, h] + dB[:, h] >= 3, growth, np.nan)
    tot = df.sum(axis=1)
    has2 = tot >= 2
    first_year = np.where(has2.any(axis=1), np.array(YEARS)[has2.argmax(axis=1)], 0)
    t = pd.DataFrame({"phrase": vocab, "key": keys, "hn_recent": dR[:, h].astype(int), "hn_base": dB[:, h].astype(int),
                      "yc_recent": dR[:, si["yc"]].astype(int), "press_recent": dR[:, si["press"]].astype(int),
                      "hn_launches": launches, "growth": growth, "first_year": first_year,
                      "hn_rate_recent": dR[:, h] / nR[h]})
    t["volume_recent"] = t.hn_recent + t.yc_recent + t.press_recent
    t["generic"] = t.phrase.map(lambda ph: all(w in GENERIC for w in ph.split()))
    vol = dict(zip(t.phrase, t.volume_recent))
    best = {}
    for ph, v in vol.items():
        w = ph.split()
        for n in range(1, len(w)):
            for i in range(len(w) - n + 1):
                sub = " ".join(w[i:i + n])
                if sub in vol and v > best.get(sub, 0):
                    best[sub] = v
    t["super_share"] = [best.get(ph, 0) / v if v else 0 for ph, v in zip(t.phrase, t.volume_recent)]
    t["candidate"] = ((t.volume_recent >= MIN_VOLUME)
                      & ((t.growth >= MIN_GROWTH) | (t.first_year >= NEW_SINCE) | (t.yc_recent >= 2))
                      & (t.hn_rate_recent <= MAX_RATE) & ~t.generic & (t.super_share < FRAGMENT_SHARE))
    # научная ступень лестницы
    sci = pd.read_parquet(REG / "all_phrases.parquet", columns=["key", "volume_recent", "growth", "first_year"])
    sci = sci.rename(columns={"volume_recent": "sci_recent", "growth": "sci_growth", "first_year": "sci_first_year"})
    t = t.merge(sci, on="key", how="left")   # по ключу сущности, а не по написанию
    t["layers"] = ((t.sci_recent.fillna(0) > 0).astype(int) + (t.hn_recent > 0) + (t.yc_recent > 0) + (t.press_recent > 0))
    g = t.growth.fillna(0).clip(0, 6)
    t["score"] = (g + 1.0 * (t.first_year >= NEW_SINCE) + 0.7 * np.log1p(t.yc_recent) + 0.3 * np.log1p(t.hn_launches)
                  + 0.5 * (t.layers - 1).clip(0, 3) - 0.5 * np.log10((t.volume_recent / 200).clip(lower=1)))
    return t


def main() -> int:
    t = build()
    t.to_parquet(REG / "all_business.parquet", index=False)   # признаки всех сущностей — для обучения ранжирования
    reg = t[t.candidate].sort_values("score", ascending=False).reset_index(drop=True)
    reg.to_parquet(REG / "business.parquet", index=False)
    print(f"в бизнес-реестре: {len(reg):,}".replace(",", " "))
    print(reg.head(50)[["phrase", "hn_recent", "yc_recent", "press_recent", "hn_launches", "growth", "first_year",
                        "sci_recent", "layers"]].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
