"""Стабилизация ранжирования: несколько прогонов кластеризации с разными зёрнами UMAP,
склейка карточек между прогонами по пересечению документов, ранг как медиана и диапазон.

    python src/stabilize.py --seeds 5            # прогнать 5 зёрен и агрегировать
    python src/stabilize.py --aggregate-only     # только агрегировать готовые прогоны

Каждый прогон пишет в data/runs/seed_<k>/{clusters,signals}. Итог: data/signals/stable.md, stable.csv, stable.json.
Сигнал считается устойчивым, если он появился в большинстве прогонов; ранг сортируется по медиане,
при равенстве по числу прогонов. Сигналы, найденные в одном прогоне из пяти, уходят в конец.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "runs"
OUT = ROOT / "data" / "signals"
PY = sys.executable
SEEDS = [42, 7, 123, 2024, 31337]


def run_seed(seed: int) -> Path:
    base = RUNS / f"seed_{seed}"
    env = {**os.environ, "CLUSTERS_DIR": str(base / "clusters"), "SIGNALS_DIR": str(base / "signals")}
    log = open(base.with_suffix(".log"), "w")
    (base / "clusters").mkdir(parents=True, exist_ok=True)
    for cmd in ([PY, "src/cluster.py", "--seed", str(seed)], [PY, "src/terms.py"], [PY, "src/signals.py"]):
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"seed {seed} готов", flush=True)
    return base


def overlap(a: set, b: set) -> float:
    return len(a & b) / max(1, min(len(a), len(b)))


def aggregate(run_dirs: list[Path], top: int) -> list[dict]:
    n_runs = len(run_dirs)
    cards = []
    for k, d in enumerate(run_dirs):
        for c in json.loads((d / "signals" / "signals.json").read_text(encoding="utf-8")):
            c["run"] = k
            c["core"] = set(c["core_ids"])
            c["docs"] = set(c["doc_ids"])
            cards.append(c)
    cards.sort(key=lambda c: (c["rank"], -c["score"]))
    groups = []
    for c in cards:
        for g in groups:
            if c["run"] in g["runs"]:
                continue  # из одного прогона в группу попадает одна карточка
            if overlap(g["core"], c["core"]) >= 0.5 or overlap(g["core"], c["docs"]) >= 0.7:
                g["members"].append(c)
                g["runs"].add(c["run"])
                g["core"] |= c["core"]
                break
        else:
            groups.append({"members": [c], "runs": {c["run"]}, "core": set(c["core"])})

    out = []
    for g in groups:
        ranks = [m["rank"] for m in g["members"]]
        absent = n_runs - len(ranks)
        ranks_full = ranks + [top + 1] * absent  # отсутствие в прогоне = хуже последнего места
        names = Counter(m["name"] for m in g["members"])
        best = max(g["members"], key=lambda m: len(m["layers"]) * 100 - m["rank"])
        out.append({
            "name": names.most_common(1)[0][0], "domain": best["domain"],
            "runs_seen": len(ranks), "n_runs": n_runs,
            "rank_median": median(ranks_full), "rank_min": min(ranks), "rank_max": max(ranks),
            "score_mean": round(sum(m["score"] for m in g["members"]) / len(ranks), 2),
            "layers_max": max(len(m["layers"]) for m in g["members"]),
            "layers": sorted({l for m in g["members"] for l in m["layers"]}),
            "n_docs": int(median(m["n_docs"] for m in g["members"])),
            "birth_year": best["birth_year"], "growth": best["growth"], "sparkline": best["sparkline"],
            "counts": best["counts"], "signals": best["signals"], "detectors": best["detectors"],
            "top_orgs": best["top_orgs"], "top_countries": best["top_countries"], "industry": best["industry"],
            "examples": best["examples"], "aliases": sorted(set(names) - {names.most_common(1)[0][0]})[:4],
        })
    out.sort(key=lambda r: (r["rank_median"], -r["runs_seen"], -r["score_mean"]))
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


def render(rows: list[dict], top: int) -> str:
    n_runs = rows[0]["n_runs"] if rows else 0
    L = [f"# Устойчивые слабые сигналы: {n_runs} прогонов с разными зёрнами\n",
         "Ранг это медиана по прогонам, отсутствие в прогоне считается хуже последнего места. "
         "Диапазон показывает, как сильно сигнал плавал. Устойчивость: в скольких прогонах сигнал появился.\n",
         "| # | Сигнал | Направление | Устойчивость | Ранг: медиана (мин–макс) | Слои | Документов |",
         "|---|---|---|---|---|---|---|"]
    for r in rows[:top]:
        L.append(f"| {r['rank']} | {r['name'][:50]} | {r['domain']} | {r['runs_seen']}/{r['n_runs']} | "
                 f"{r['rank_median']:.0f} ({r['rank_min']}–{r['rank_max']}) | {r['layers_max']} | {r['n_docs']} |")
    L.append("\n")
    for r in rows[:top]:
        L.append(f"\n---\n\n## {r['rank']}. {r['name']}\n")
        L.append(f"**{r['domain']}** · устойчивость {r['runs_seen']}/{r['n_runs']} · ранг {r['rank_median']:.0f} "
                 f"({r['rank_min']}–{r['rank_max']}) · {r['n_docs']} документов · с {r['birth_year']} · рост {r['growth']:+.2f} · "
                 f"слои: {', '.join(r['layers'])}\n")
        L.append(f"Динамика `{r['sparkline']}` {' '.join(map(str, r['counts']))}\n")
        if r["aliases"]:
            L.append(f"Также: {'; '.join(a[:60] for a in r['aliases'])}\n")
        L.append("**Почему это сигнал**")
        for s in r["signals"] + r["detectors"]:
            L.append(f"- {s}")
        L.append(f"\n**Кто:** {'; '.join(r['top_orgs'][:4])}  \n**Страны:** {', '.join(r['top_countries'])}")
        if r["industry"]:
            L.append(f"**Индустрия:** {', '.join(r['industry'])}")
        L.append("\n**Примеры**")
        for e in r["examples"]:
            L.append(f"- [{e['title']}]({e['url']}) ({e['year']}, {e['venue'] or 'без площадки'})")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    RUNS.mkdir(parents=True, exist_ok=True)
    dirs = [RUNS / f"seed_{s}" for s in seeds] if args.aggregate_only else [run_seed(s) for s in seeds]
    dirs = [d for d in dirs if (d / "signals" / "signals.json").exists()]
    rows = aggregate(dirs, args.top)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stable.md").write_text(render(rows, args.top), encoding="utf-8")
    (OUT / "stable.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} | {"layers": ", ".join(r["layers"])}
                  for r in rows]).to_csv(OUT / "stable.csv", index=False)
    stable = sum(r["runs_seen"] >= math_ceil(len(dirs) / 2) for r in rows[:args.top])
    print(f"прогонов {len(dirs)}, групп {len(rows)}, в топ-{args.top} устойчивых (≥ половины прогонов): {stable}")
    for r in rows[:20]:
        print(f"  {r['rank']:2d}. {r['name'][:44]:44s} {r['runs_seen']}/{r['n_runs']}  ранг {r['rank_median']:>4.0f} ({r['rank_min']}-{r['rank_max']})  слои={r['layers_max']}")


def math_ceil(x: float) -> int:
    return int(-(-x // 1))


if __name__ == "__main__":
    main()
