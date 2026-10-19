"""Открытая модель на GPU вместо API YandexGPT: промпты из набора для сервиса → файл → ответы обратно в набор.

    REGISTRY_BUNDLE=artifacts/registry_v5 python -m registry.llm_offline prompts --out prompts.jsonl
    REGISTRY_BUNDLE=artifacts/registry_v5 python -m registry.llm_offline import --answers answers.jsonl

Зачем. Мини-набор (registry.mini_bundle) собран без ключа: название — английская фраза, описания нет, LLM-оси
не посчитаны. Те же два вопроса, что задают registry.llm_assess (смысловая проверка: технология ли, русское
название и описание) и registry.llm_axes (родитель, механизм с дословной цитатой, тип, абстракт), задаются
открытой моделью на кластере (for_cesga/server/llm_local.py). Промпты и проверка ответов — те же функции
модулей, поэтому ответ не по схеме или без дословного повтора фразы отбрасывается так же, как ответ YandexGPT.

prompts: по LLM_BATCH_ASSESS записей в промпт проверки, по LLM_BATCH_AXES — в промпт осей; заголовки источников
из карточек набора. import: русские названия и описания — в signals.parquet и cards.json.gz (карточка перестаёт
быть unverified только при принятом ответе), векторы названий emb.npy пересчитываются, оси — в
cache/llm_axes.jsonl, флаги «не технология» — в cache/llm_assess.jsonl; поле model в записях кэша = имя модели.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import llm_assess, llm_axes

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.getenv("REGISTRY_BUNDLE", ROOT / "artifacts" / "registry"))
B_ASSESS = int(os.getenv("LLM_BATCH_ASSESS", "5"))
B_AXES = int(os.getenv("LLM_BATCH_AXES", "3"))
MODEL = os.getenv("LLM_OFFLINE_MODEL", "Qwen2.5-7B-Instruct")


def _load():
    sig = pd.read_parquet(BUNDLE / "signals.parquet")
    with gzip.open(BUNDLE / "cards.json.gz", "rt", encoding="utf-8") as fh:
        cards = json.load(fh)
    return sig, cards


def _titles(card: dict, n: int) -> list[str]:
    return [(d.get("title") or "").strip()[:160] for d in (card.get("sources") or []) if d.get("title")][:n]


def _axes_items(sig: pd.DataFrame, cards: dict) -> list[dict]:
    out = []
    for r in sig.itertuples():
        c = cards.get(r.key) or {}
        src = [{"title": (d.get("title") or "")[:180], "date": str(d.get("published") or "")[:10],
                "type": d.get("source_type") or d.get("source_name") or ""} for d in (c.get("sources") or []) if d.get("title")][:6]
        out.append({"key": r.key, "phrase": r.phrase.replace("_", " "), "name_ru": r.name_ru,
                    "description_ru": r.description_ru or "", "meas": getattr(r, "meas", "") or "", "sources": src})
    return out


def prompts(out: Path) -> None:
    sig, cards = _load()
    rows = []
    keys = list(sig.key)
    for s in range(0, len(sig), B_ASSESS):
        part = sig.iloc[s:s + B_ASSESS]
        listing = "\n".join(f"{i}. {ph}" + "".join(f"\n   — {t}" for t in _titles(cards.get(k) or {}, llm_assess.TITLES))
                            for i, (k, ph) in enumerate(zip(part.key, part.phrase), 1))
        rows.append({"id": f"as:{s}", "keys": list(part.key), "prompt": llm_assess.PROMPT.replace("{items}", listing),
                     "max_new": 180 * len(part)})
    items = _axes_items(sig, cards)
    for s in range(0, len(items), B_AXES):
        part = items[s:s + B_AXES]
        rows.append({"id": f"ax:{s}", "keys": [it["key"] for it in part], "prompt": llm_axes.prompt_for(part),
                     "max_new": 450 * len(part)})
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_as = sum(1 for r in rows if r["id"].startswith("as:"))
    print(f"записей {len(keys)}; промптов {len(rows)} (проверка {n_as}, оси {len(rows) - n_as}) -> {out}")


def _json_array(raw: str):
    m = re.search(r"\[.*\]", raw or "", re.S)
    try:
        return json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []


def import_answers(answers: Path, prompts_file: Path, reembed: bool) -> None:
    sig, cards = _load()
    pr = {r["id"]: r for r in (json.loads(l) for l in prompts_file.read_text(encoding="utf-8").splitlines() if l.strip())}
    ph = dict(zip(sig.key, sig.phrase))
    items = {it["key"]: it for it in _axes_items(sig, cards)}
    assess, axes, bad = {}, {}, {"as": 0, "ax": 0}
    for l in answers.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        a = json.loads(l)
        p = pr.get(a["id"])
        if not p:
            continue
        keys = p["keys"]
        if any(k not in ph for k in keys):      # промпт от другой версии набора
            continue
        if a["id"].startswith("as:"):
            ok = llm_assess._validate(_json_array(a["raw"]), len(keys), [ph[k] for k in keys])
            bad["as"] += len(keys) - len(ok)
            for i, v in ok.items():
                assess[keys[i]] = {"phrase": ph[keys[i]], **v, "model": MODEL}
        else:
            got = llm_axes.parse(a["raw"], [items[k] for k in keys])
            bad["ax"] += len(keys) - len(got)
            for x in got:
                axes[x["key"]] = {**x, "model": MODEL}

    named = 0
    for i, k in enumerate(sig.key):
        v = assess.get(k)
        if not v:
            continue
        sig.at[i, "name_ru"], sig.at[i, "description_ru"] = v["name_ru"], v["description_ru"]
        c = cards.get(k)
        if c is not None:
            c["name_ru"], c["description"] = v["name_ru"], v["description_ru"]
            c["unverified"] = not v["keep"]
            if not v["keep"]:
                c["reject_reason"] = v["reject_reason"]
        named += 1
    sig.to_parquet(BUNDLE / "signals.parquet")
    with gzip.open(BUNDLE / "cards.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(cards, fh, ensure_ascii=False)
    cache = BUNDLE / "cache"
    cache.mkdir(exist_ok=True)
    with (cache / "llm_assess.jsonl").open("w", encoding="utf-8") as fh:
        for v in assess.values():
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")
    with (cache / "llm_axes.jsonl").open("w", encoding="utf-8") as fh:
        for v in axes.values():
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")
    n = len(sig)
    print(f"записей {n}: названия и описания {named} ({named / n:.0%}), не технология {sum(not v['keep'] for v in assess.values())}; "
          f"оси {len(axes)} ({len(axes) / n:.0%}), с проверенной цитатой {sum(x['quote_ok'] for x in axes.values())}; "
          f"отброшено ответов: проверка {bad['as']}, оси {bad['ax']}")
    if reembed:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("deepvk/USER-bge-m3", device="cpu")
        m.max_seq_length = 64
        E = m.encode(list(sig.name_ru), batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        np.save(BUNDLE / "emb.npy", E.astype(np.float16))
        print(f"векторы названий пересчитаны: {E.shape}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prompts", "import"])
    ap.add_argument("--out", default="prompts.jsonl")
    ap.add_argument("--answers", default="answers.jsonl")
    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--no-reembed", action="store_true")
    a = ap.parse_args()
    if a.cmd == "prompts":
        prompts(Path(a.out))
    else:
        import_answers(Path(a.answers), Path(a.prompts), not a.no_reembed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
