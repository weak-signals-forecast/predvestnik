"""Оценка реестра на 100 сигналах организаторов. Единственное место, где читается их список.

    python -m registry.evaluate [--queries data/labels/positives_queries.csv]

Правило совпадения зафиксировано до первого замера. Запись реестра p покрывает технологию t, если после
приведения слов к основе (срезаем окончание -s):
  * все слова p входят в запрос t, и p покрывает не меньше 60 % значимых слов t (минимум 2 слова), или
  * все значимые слова t входят в p (реестр нашёл более длинную формулировку).
Одиночное слово (x402, CXL) засчитывается как частичное совпадение и в основную цифру не идёт.

Сочетание А2 «a × b» покрывает t, если каждый из терминов a и b делит с t хотя бы одно значимое слово,
а их объединение проходит то же правило, что и одиночная фраза.

Разбиение: нечётные номера — dev (на них разрешено смотреть при доработке), чётные — test (итоговая цифра).
Цифры: полнота в первых K записях реестра для K = 500, 1000, 2000, 5000 и во всём реестре; отдельно —
есть ли фраза в словаре вообще (чтобы отличать «не нашли» от «нашли, но отбросили фильтром»).
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import pandas as pd

from .corpus import CORPUS, REGISTRY

REG = REGISTRY
ROOT = Path(__file__).resolve().parents[1]
STOP = set("a an the of for and or in on to with via based using".split())
KS = (500, 1000, 2000, 5000)


def stems(text: str) -> list[str]:
    ws = re.findall(r"[a-z0-9]+", text.lower().replace("-", " "))
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in ws]


def content(text: str) -> set[str]:
    return {w for w in stems(text) if w not in STOP}


def covers(p: set[str], q: set[str]) -> bool:
    if len(p) < 2:
        return False
    if p <= q and len(p) >= max(2, math.ceil(0.6 * len(q))):
        return True
    return q <= p


def evaluate(queries: Path) -> None:
    reg = pd.read_parquet(REG / "registry.parquet")
    allp = pd.read_parquet(REG / "all_phrases.parquet")
    reg_sets = [content(p) for p in reg.phrase]
    cpath = REG / "combos.parquet"
    combos = pd.read_parquet(cpath) if cpath.exists() else pd.DataFrame(columns=["a", "b", "phrase"])
    combo_sets = [(content(a), content(b)) for a, b in zip(combos.a, combos.b)]
    bpath = REG / "business.parquet"
    biz = pd.read_parquet(bpath) if bpath.exists() else pd.DataFrame(columns=["phrase"])
    biz_sets = [content(p) for p in biz.phrase]
    all_sets = list(zip(allp.phrase, (content(p) for p in allp.phrase)))
    rows = []
    for q in csv.DictReader(open(queries, encoding="utf-8")):
        qs = content(q["query"])
        rank = next((i + 1 for i, ps in enumerate(reg_sets) if covers(ps, qs)), None)
        partial = next((i + 1 for i, ph in enumerate(reg.phrase) if " " not in ph and ph in qs), None)
        in_vocab = [ph for ph, ps in all_sets if covers(ps, qs)]
        brank = next((i + 1 for i, ps in enumerate(biz_sets) if covers(ps, qs)), None)
        crank = next((i + 1 for i, (ca, cb) in enumerate(combo_sets)
                      if ca & qs and cb & qs and covers(ca | cb, qs)), None)
        rows.append({"id": int(q["id"]), "split": "dev" if int(q["id"]) % 2 else "test", "query": q["query"],
                     "rank": rank, "match": reg.phrase[rank - 1] if rank else "", "partial_rank": partial,
                     "in_vocab": len(in_vocab) > 0, "vocab_examples": "; ".join(in_vocab[:3]),
                     "biz_rank": brank, "biz_match": biz.phrase.iloc[brank - 1] if brank else "",
                     "combo_rank": crank, "combo_match": combos.phrase.iloc[crank - 1] if crank else ""})
    r = pd.DataFrame(rows)
    out = REG / "evaluation.csv"
    r.to_csv(out, index=False)
    print(f"реестр: {len(reg):,} записей\n".replace(",", " "))
    head = "| | " + " | ".join(f"топ-{k}" for k in KS) + " | весь реестр | есть в словаре |"
    print(head)
    print("|---" * (len(KS) + 3) + "|")
    for sp in ("dev", "test", "все"):
        s = r if sp == "все" else r[r.split == sp]
        cells = [f"{(s['rank'] <= k).sum()}/{len(s)}" for k in KS]
        print(f"| {sp} | " + " | ".join(cells) + f" | {s['rank'].notna().sum()}/{len(s)} | {s.in_vocab.sum()}/{len(s)} |")
    print(f"\nА2, новые сочетания: {len(combos):,} записей".replace(",", " "))
    print("| | " + " | ".join(f"топ-{k}" for k in KS) + " | все сочетания | А или А2 |")
    print("|---" * (len(KS) + 3) + "|")
    for sp in ("dev", "test", "все"):
        s = r if sp == "все" else r[r.split == sp]
        cells = [f"{(s['combo_rank'] <= k).sum()}/{len(s)}" for k in KS]
        either = (s["rank"].notna() | s["combo_rank"].notna()).sum()
        print(f"| {sp} | " + " | ".join(cells) + f" | {s['combo_rank'].notna().sum()}/{len(s)} | {either}/{len(s)} |")
    print(f"\nВ, бизнес-слой (HN, YC, пресса): {len(biz):,} записей".replace(",", " "))
    print("| | " + " | ".join(f"топ-{k}" for k in KS) + " | весь бизнес-слой | А, А2 или В |")
    print("|---" * (len(KS) + 3) + "|")
    for sp in ("dev", "test", "все"):
        s = r if sp == "все" else r[r.split == sp]
        cells = [f"{(s['biz_rank'] <= k).sum()}/{len(s)}" for k in KS]
        anyr = (s["rank"].notna() | s["combo_rank"].notna() | s["biz_rank"].notna()).sum()
        print(f"| {sp} | " + " | ".join(cells) + f" | {s['biz_rank'].notna().sum()}/{len(s)} | {anyr}/{len(s)} |")
    print(f"\nподробно: {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(ROOT / "data/labels/positives_queries.csv"))
    evaluate(Path(ap.parse_args().queries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
