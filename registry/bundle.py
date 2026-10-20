"""Набор данных для сервиса базы сигналов (контейнер registry): всё, что нужно для ответа на запрос, без корпуса.

    python -m registry.bundle            # -> artifacts/registry/

Состав:
  signals.parquet  — база сигналов: названия, описания, области, доли рубрик областей, баллы моделей, измерения;
  emb.npy          — векторы «название — описание» (USER-bge-m3, float16), в порядке строк signals.parquet;
  cards.json.gz    — карточка каждой технологии: источники (название, ссылка, дата, тип, язык, доверие), ряды по
                     годам, лестница, разложение балла (SHAP двух моделей), кто стоит за сигналом, преимущество/кейс;
  examples.json    — образцы слабых сигналов по областям для промпта выбора — из нашей разметки (не материалы
                     организаторов);
  model_global.json — общая картина SHAP двух моделей для витрины;
  cache/           — уже полученные ответы YandexGPT (выбор, HyDE, тексты карточек), чтобы не спрашивать повторно.
"""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import REGISTRY as REG

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "registry"


def main() -> int:
    from .adjudicate import measurements
    from .search import Searcher
    OUT.mkdir(parents=True, exist_ok=True)
    s = Searcher(base=True)
    p = s.pool.copy()
    line = measurements(REG)["line"]
    p["meas"] = p.key.map(line)
    b = pd.read_parquet(REG / "signal_base.parquet", columns=["key", "expert_p", "history_p"])
    p = p.merge(b, on="key", how="left")
    keep = ["key", "phrase", "name_ru", "description_ru", "areas", "areas_our", "aliases", "kind", "first_year", "expert", "expert_p",
            "history_p", "learned", "biz", "org_sim", "meas", "tier"] + [c for c in p.columns if c.startswith("share:")]
    p = p[[c for c in keep if c in p.columns]].reset_index(drop=True)
    p["areas"] = p.areas.map(lambda v: [str(x) for x in v] if v is not None and not isinstance(v, str) else [])
    p["areas_our"] = p.areas_our.map(lambda v: [str(x) for x in v] if v is not None and not isinstance(v, str) else [])
    p["aliases"] = p.aliases.map(lambda v: [str(x) for x in v] if v is not None and not isinstance(v, str) else [])
    p.to_parquet(OUT / "signals.parquet")
    np.save(OUT / "emb.npy", s.E.astype(np.float16))
    print(f"база: {len(p)} сигналов, векторы {s.E.shape}", flush=True)

    # карточки всех сигналов базы — тем же кодом, что страница (источники из корпуса, ряды, SHAP, организации)
    top = p[["key", "phrase", "name_ru", "description_ru", "learned"]].assign(
        query="__bundle__", area_detected=None, pos=range(1, len(p) + 1), pool_pct=p.expert.rank(pct=True))
    top.to_csv(REG / "bundle_top.csv", index=False)
    raw = OUT / "cards_raw.json"
    subprocess.run([sys.executable, "-W", "ignore", "-m", "registry.export_web", "--method", "bundle",
                    "--top-file", "bundle_top.csv", "--out", str(raw)], check=True)
    d = json.loads(raw.read_text(encoding="utf-8"))
    cards = {c["key"]: c for r in d["results"].values() for c in r["cards"]}
    with gzip.open(OUT / "cards.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(cards, fh, ensure_ascii=False)
    (OUT / "model_global.json").write_text(json.dumps(
        {"expert": d.get("model_global"), "history": d.get("model_global_history"), "sources": d.get("sources"),
         "corpus_snapshot": d.get("corpus_snapshot")}, ensure_ascii=False), encoding="utf-8")
    raw.unlink()
    print(f"карточек: {len(cards)}", flush=True)

    # образцы для промпта выбора: размеченные сигналы по областям (теги областей из проверки пула)
    lab = pd.read_csv(REG / "expert_labels.csv").query("y == 1")
    ex = p[p.key.isin(set(lab.key))]
    examples: dict[str, list[str]] = {}
    for r in ex.itertuples():
        for a in (r.areas or ["другое"]):
            examples.setdefault(a, []).append(f"{r.name_ru} ({r.phrase.replace('_', ' ')})")
    (OUT / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")

    (OUT / "cache").mkdir(exist_ok=True)
    for f in ("select_cache.jsonl", "hyde.json", "card_text.jsonl"):
        if (REG / f).exists():
            shutil.copy(REG / f, OUT / "cache" / f)
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"-> {OUT}: {size / 1e6:.1f} МБ; образцы по областям: " +
          ", ".join(f"{a} {len(v)}" for a, v in examples.items()))
    return 0


if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] != "axes"):
    sys.exit(main())


def axes_bundle() -> int:
    """Оси слабости и новые баллы для записей базы -> artifacts/registry/axes.parquet (читает сервис при REGISTRY_RANK_V2=1).

        python -m registry.bundle axes

    Берёт axes.parquet (registry/axes.py), expert_scored_axes.parquet и history_scored_axes.parquet (модели с осями),
    вердикт экспертной проверки базы (data/labels/base_review.csv). Записи базы без осей получают NaN: сервис
    тогда использует прежний балл."""
    from .axes import AXES
    base = pd.read_parquet(OUT / "signals.parquet", columns=["key"])
    ax = pd.read_parquet(REG / "axes.parquet").drop_duplicates("key")[["key", "parent", "docs_now"] + AXES]
    out = base.merge(ax, on="key", how="left")
    for name, col in (("expert_scored_axes.parquet", "expert_p"), ("history_scored_axes.parquet", "history_p")):
        p = REG / name
        if p.exists():
            s = pd.read_parquet(p, columns=["key", col]).drop_duplicates("key").rename(columns={col: col + "_axes"})
            out = out.merge(s, on="key", how="left")
    review = ROOT / "data" / "labels" / "base_review.csv"
    if review.exists():
        rv = pd.read_csv(review)[["key", "verdict"]].drop_duplicates("key")
        out = out.merge(rv, on="key", how="left")
    out.to_parquet(OUT / "axes.parquet")
    print(f"оси базы: {len(out)} записей, без осей {int(out.stage_accel.isna().sum())}, "
          f"с новым баллом {int(out.get('expert_p_axes', pd.Series(dtype=float)).notna().sum())} -> {OUT / 'axes.parquet'}")
    return 0


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "axes":
    sys.exit(axes_bundle())
