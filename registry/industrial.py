"""Промышленные кандидаты, которых YandexGPT Lite ещё не проверяла, — список ключей для llm_assess --keys-file.

    python -m registry.industrial          # -> industrial_keys.txt

Пул для проверки Lite когда-то отбирался по общему баллу, и промышленных технологий в нём мало: в базе их было 20.
Здесь берутся технологии реестра, которые относятся к промышленности (предметное слово в термине — ПЛК, SCADA,
сварка, станки, контроль качества, предиктивное обслуживание, …, или проекты ЕС 2022–2026 при доле промышленных
рубрик), не зрелые, не обрывки и не рубрики «ИИ + слово», не теория, и ещё не проверенные. Порядок — смесь двух
моделей (экспертной и исторической), в список — верхние LIMIT.
"""
from __future__ import annotations

import json
import re
import sys

import pandas as pd

from .corpus import REGISTRY as REG

LIMIT = 1500


def main() -> int:
    from .search import MATURE_YEAR, first_year
    from .signal_base import AREA_LEXICON, is_fragment, is_umbrella, rubric_share, theory_share, THEORY_MAX
    from .pool import AREA_GROUPS_POOL
    r = pd.read_parquet(REG / "ranked.parquet", columns=["key", "phrase", "candidate"])
    r = r[r.candidate].drop_duplicates("key").reset_index(drop=True)
    have = {json.loads(l)["phrase"] for l in (REG / "llm_assess.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    r = r[~r.phrase.str.replace("_", " ").isin(have)]
    r = r[~r.phrase.map(is_fragment) & ~r.phrase.map(is_umbrella)]
    lex = re.compile(AREA_LEXICON["Индустриальный ИИ"])
    r["lex"] = r.phrase.str.replace("_", " ").map(lambda p: bool(lex.search(p)))
    cord = pd.read_parquet(REG / "cordis.parquet", columns=["key", "cordis_recent"])
    r = r.merge(cord, on="key", how="left")
    r["ind_share"] = rubric_share(r.key, AREA_GROUPS_POOL["Индустриальный ИИ"])
    r = r[r.lex | ((r.cordis_recent.fillna(0) >= 3) & (r.ind_share >= 0.3))]
    r["first_year"] = first_year(r.key)
    r = r[~(r.first_year <= MATURE_YEAR)]
    r["theory"] = theory_share(r.key)
    r = r[~(r.theory >= THEORY_MAX)]
    s = r.merge(pd.read_parquet(REG / "expert_scored.parquet", columns=["key", "expert_p"]).drop_duplicates("key"),
                on="key", how="left").merge(pd.read_parquet(REG / "history_scored.parquet", columns=["key", "history_p"])
                                            .drop_duplicates("key"), on="key", how="left")
    s["blend"] = 0.3 * s.history_p.rank(pct=True) + 0.7 * s.expert_p.rank(pct=True)
    s = s.nlargest(LIMIT, "blend")
    (REG / "industrial_keys.txt").write_text("\n".join(s.key) + "\n", encoding="utf-8")
    print(f"промышленных кандидатов без проверки Lite: {len(r)}; в список: {len(s)} "
          f"(по предметному слову {int(s.lex.sum())}, по проектам ЕС и рубрикам {int((~s.lex).sum())})")
    print("примеры:", ", ".join(s.phrase.str.replace("_", " ").head(40)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
