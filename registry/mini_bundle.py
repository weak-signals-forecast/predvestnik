"""Мини-набор данных для сервиса из реестра без слоёв рынка и без проверки моделью (запасной путь и срез по области).

    GPB_CORPUS=~/gpb_corpus GPB_REGISTRY=~/gpb_corpus/registry_condmat REGISTRY_MIN_SOURCES=1 \
    python -m registry.mini_bundle --out artifacts/registry_condmat --top 800 --area "Материалы и физика"

Когда нужен. Полный набор (registry/bundle.py) требует бизнес-слоёв, проверки YandexGPT и паспортов. Здесь же из
реестра, обработанного `registry.apply_models`, собирается минимально достаточный набор для сервиса: записи с баллами
моделей и осями, векторы названий, карточки с источниками из корпуса, рядами по годам, фактами и SHAP экспертной
модели. Русских названий и описаний без ключа нет: название — английская фраза, описание пустое, карточка помечена
`unverified`. При появлении ключа `registry.llm_axes` и `registry.llm_assess` дополняют тот же набор.

Отбор: кандидаты реестра, 2–4 слова, не обрывок и не рубрика (правила signal_base), не общие слова, первый год
позже MATURE_YEAR, объём ≥ MIN_VOLUME, не стандарт по голове названия; верхние --top по смеси моделей.
Сервис: REGISTRY_BUNDLE=<out> REGISTRY_RANK_V2=1 uvicorn registry_service.main:app --port 8012
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .axes import AXES, LABELS as AXES_LABELS
from .corpus import CORPUS, CUTOFF, REGISTRY, SOURCES, YEARS, docs, files
from .text import candidates

ROOT = Path(__file__).resolve().parents[1]
MATURE_YEAR = 2019
MIN_VOLUME = 8
PER_PHRASE_SOURCES = 6
AREAS6 = ("Edge", "Защита ИИ", "Индустриальный ИИ", "Инфраструктура ИИ", "Роботы", "Финтех")

_WANT: dict[str, str] = {}


def _init(want: dict[str, str]) -> None:
    global _WANT
    _WANT = want


def _scan(args):
    """Один файл корпуса: документы, в тексте которых есть фразы набора -> (key, запись источника)."""
    source, path = args
    out = []
    for d in docs(source, path):
        if d["year"] not in YEARS or not d["title"]:
            continue
        hits = {k for ph in candidates(d["text"]) if (k := _WANT.get(ph)) is not None}
        if not hits:
            continue
        if source == "arxiv":
            aid = str(d["id"]).replace("arXiv:", "")
            rec = {"title": d["title"], "url": f"https://arxiv.org/abs/{aid}", "published": str(d["year"]),
                   "source_name": "arXiv", "source_type": "препринт", "language": "en", "trust": "средний",
                   "year": d["year"]}
        else:
            rec = {"title": d["title"], "url": d.get("doi") or "", "published": str(d["year"]),
                   "source_name": "OpenAlex", "source_type": "научная публикация", "language": "en",
                   "trust": "высокий", "year": d["year"]}
        for k in hits:
            out.append((k, rec))
    return out


def sources_for(keys_by_phrase: dict[str, str], workers: int = 6) -> dict[str, list[dict]]:
    """Для каждой фразы — до PER_PHRASE_SOURCES документов корпуса, свежие годы первыми."""
    acc: dict[str, list[dict]] = {k: [] for k in set(keys_by_phrase.values())}
    todo = [(s, p) for s, p in files() if s in ("arxiv", "openalex")]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(keys_by_phrase,)) as ex:
        for hits in ex.map(_scan, todo, chunksize=2):
            for k, rec in hits:
                acc[k].append(rec)
    for k, lst in acc.items():
        lst.sort(key=lambda r: -r["year"])
        acc[k] = [{kk: v for kk, v in r.items() if kk != "year"} for r in lst[:PER_PHRASE_SOURCES]]
    return acc


DOMAINS = (   # область по группе (раздел arXiv, подобласть OpenAlex); порядок важен — первое совпадение
    ("Биотех и медицина", r"^ax:(q-bio|physics\.(bio|med)-ph)"),
    ("Финтех и экономика", r"^(ax:(q-fin|econ)|oa:(1404|2003))"),
    ("Инженерия и электроника", r"^(oa:(2207|2208|2209)|ax:(eess\.(SP|SY)|cs\.(AR|ET|SY)|physics\.(app-ph|optics|ins-det|acc-ph)))"),
    ("Материалы и физика", r"^(ax:(cond-mat|quant-ph|physics|hep|astro|nucl|gr-qc|atom-ph|nlin|chao-dyn|math-ph)|oa:3107)"),
    ("ИИ и ПО", r"."),
)


def domain_of(reg: Path, s: pd.DataFrame) -> pd.Series:
    """Область фразы: та, в чьих группах у её сущности больше всего документов за свежие годы (groups.npz)."""
    import re
    g = np.load(reg / "groups.npz")
    groups = [str(x) for x in g["groups"]]
    dom_idx = [next(i for i, (_, rx) in enumerate(DOMAINS) if re.search(rx, gr)) for gr in groups]
    ap_ = pd.read_parquet(reg / "all_phrases.parquet", columns=["key", "eid"]).drop_duplicates("key")
    eid = s.key.map(dict(zip(ap_.key, ap_.eid))).fillna(-1).astype(int).values
    grp = g["grp"]
    agg = np.zeros((len(DOMAINS), grp.shape[1]), np.int64)
    for gi, di in enumerate(dom_idx):
        agg[di] += grp[gi]
    best = agg.argmax(axis=0)
    names = np.array([d for d, _ in DOMAINS], dtype=object)
    return pd.Series(np.where(eid >= 0, names[best[np.maximum(eid, 0)]], "ИИ и ПО"), index=s.index)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "registry_mini"))
    ap.add_argument("--top", type=int, default=800)
    ap.add_argument("--area", default="другое", help="тег области для всех записей набора")
    ap.add_argument("--models", default="registry_v3")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--quota", default="", help="квоты по областям вместо --top: «ИИ и ПО=1500,Материалы и физика=1000»")
    a = ap.parse_args()
    from .signal_base import is_fragment, is_umbrella
    from registry_service.rank import STANDARD_HEAD
    reg, out = REGISTRY, Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(exist_ok=True)

    s = pd.read_parquet(reg / "scored.parquet")
    gen = pd.read_parquet(reg / "all_phrases.parquet", columns=["key", "generic"]).drop_duplicates("key")
    s = s.merge(gen, on="key", how="left").reset_index(drop=True)
    ph = s.phrase.str.replace("_", " ")
    # голова фразы — существительное: без проверки моделью иначе наверх выходят обрывки «symmetry engineered»,
    # «electrically programmable», «physics grounded». Хвост на -ed/-ing/-able/-ive/-ous/-ic/-ly — не название технологии.
    import re
    adj_head = re.compile(r"\w+(ed|ing|able|ible|ive|ous|ic|al|ly|ized|ised)$")
    noun_ok = {"optics", "photonics", "electronics", "spintronics", "ceramics", "plastics", "dynamics", "mechanics",
               "physics", "thermal", "material", "materials", "metal", "metals", "crystal", "crystals", "signal", "signals"}
    bad_head = ph.map(lambda p: (w := p.split()[-1]) not in noun_ok and bool(adj_head.search(w)))
    ok = ((s.n_words >= 2) & (s.n_words <= 4) & ~ph.map(is_fragment) & ~ph.map(is_umbrella) & ~bad_head
          & (s.first_year > MATURE_YEAR) & (np.expm1(s.sci_log_volume) >= MIN_VOLUME)
          & ~ph.map(lambda p: bool(STANDARD_HEAD.search(p))))
    ok &= ~s.generic.fillna(False).astype(bool)
    if a.quota:
        # квоты по областям: модели обучены на ИИ, при общем отборе физика и биотех вытесняются целиком
        s["domain"] = domain_of(reg, s)
        quota = dict(q.split("=") for q in a.quota.split(","))
        b = pd.concat([s[ok & (s.domain == d)].nlargest(int(n), "blend") for d, n in quota.items()])
        b = b.sort_values("blend", ascending=False).reset_index(drop=True)
        print("по областям: " + ", ".join(f"{d} {int((b.domain == d).sum())}" for d in quota), flush=True)
    else:
        b = s[ok].nlargest(a.top, "blend").reset_index(drop=True)
        b["domain"] = a.area
    b["ph"] = b.phrase.str.replace("_", " ")
    print(f"кандидатов {len(s):,}, после фильтров {int(ok.sum()):,}, в набор {len(b)}".replace(",", " "), flush=True)

    # ряды по годам и факты
    df = np.load(reg / "counts.npz")["df"]
    tot = json.loads((reg / "totals.json").read_text(encoding="utf-8"))
    ap_ = pd.read_parquet(reg / "all_phrases.parquet", columns=["key", "eid"]).drop_duplicates("key")
    eid = b.key.map(dict(zip(ap_.key, ap_.eid))).astype(int).values
    oa, ax = SOURCES.index("openalex"), SOURCES.index("arxiv")
    yi = {y: i for i, y in enumerate(YEARS)}
    years = [y for y in YEARS if y <= CUTOFF]
    docs_y = np.stack([df[eid, oa, yi[y]] + df[eid, ax, yi[y]] for y in years], axis=1)
    tot_y = np.array([tot.get(f"openalex|{y}", 0) + tot.get(f"arxiv|{y}", 0) for y in years])
    share_y = docs_y / np.maximum(tot_y, 1) * 100
    recent = docs_y[:, [years.index(y) for y in range(CUTOFF - 2, CUTOFF + 1)]].sum(axis=1)
    base_ = docs_y[:, [years.index(y) for y in range(CUTOFF - 7, CUTOFF - 3)]].sum(axis=1)
    growth = (share_y[:, [years.index(y) for y in range(CUTOFF - 2, CUTOFF + 1)]].mean(axis=1) + 1e-6) / \
             (share_y[:, [years.index(y) for y in range(CUTOFF - 7, CUTOFF - 3)]].mean(axis=1) + 1e-6)

    # SHAP экспертной модели — вклад признаков в балл (как в полном наборе)
    import joblib
    import shap
    from .export_web import LABELS
    em = joblib.load(CORPUS / a.models / "expert_model_axes.joblib")
    feats = em["features"]
    X = b[feats].astype(float)
    sv = shap.TreeExplainer(em["model"]).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    labels = {**LABELS, **AXES_LABELS}

    # источники из корпуса
    src = sources_for(dict(zip(b.ph, b.key)), a.workers)

    cards: dict[str, dict] = {}
    for i, r in b.iterrows():
        order = np.argsort(-np.abs(sv[i]))[:7]
        explain = [{"label": labels.get(feats[j], feats[j]), "value": round(float(sv[i, j]), 3),
                    "detail": f"значение признака: {X.iat[i, j]:.2f}"} for j in order if np.isfinite(X.iat[i, j])]
        facts = [f"Термин в работах — с {int(r.first_year)} года",
                 f"Доля научных работ выросла в {growth[i]:.1f} раза к {CUTOFF - 7}–{CUTOFF - 4}".replace(".", ","),
                 f"{int(recent[i])} записей в научных источниках за {CUTOFF - 2}–{CUTOFF}"]
        if isinstance(r.parent, str) and r.parent:
            facts.append(f"Родительская технология по названию: {r.parent}")
        cards[r.key] = {
            "key": r.key, "unverified": True, "name": r.ph, "name_ru": r.ph[:1].upper() + r.ph[1:],
            "strength": round(float(r.blend), 3), "strength_registry": round(float(r.blend), 3),
            "facts": facts,
            "ladder": [{"layer": "Наука", "present": True, "value": f"{int(recent[i])} записей"},
                       {"layer": "Hacker News", "present": False, "value": "не измерялось"},
                       {"layer": "Стартапы YC", "present": False, "value": "не измерялось"},
                       {"layer": "Пресса", "present": False, "value": "не измерялось"},
                       {"layer": "Энциклопедия", "present": False, "value": "не измерялось"}],
            "series": {"years": years, "lines": [{"label": "наука (доля работ, %)", "color": "science",
                                                  "values": [round(float(v), 4) for v in share_y[i]]}]},
            "explain": explain, "explain_history": None, "actors": None, "twin": None, "passport": None,
            "description": None, "advantage": None, "case": None, "why": None, "verification": None,
            "sources": src.get(r.key, []), "kind": None,
        }

    # signals.parquet в контракте сервиса
    sig = pd.DataFrame({
        "key": b.key, "phrase": b.phrase, "name_ru": [cards[k]["name_ru"] for k in b.key], "description_ru": "",
        "areas": [[d] for d in b.domain], "areas_our": [[]] * len(b), "aliases": [[]] * len(b), "kind": None,
        "first_year": b.first_year.astype(float), "expert": b.blend.astype(float), "expert_p": b.expert_p.astype(float),
        "history_p": b.history_p.astype(float), "learned": b.learned.astype(float), "biz": 0.0, "org_sim": 0.0,
        "meas": [f"работ по годам — " + ", ".join(f"{y}: {int(n)}" for y, n in zip(years[-6:], docs_y[i, -6:]))
                 + f"; первый год: {int(b.first_year[i])}" for i in range(len(b))],
        "tier": 1,
    })
    for ar in AREAS6:
        sig["share:" + ar] = 0.0
    sig.to_parquet(out / "signals.parquet")

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("deepvk/USER-bge-m3", device="cpu")
    m.max_seq_length = 64
    E = m.encode([cards[k]["name_ru"] for k in b.key], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    np.save(out / "emb.npy", E.astype(np.float16))
    with gzip.open(out / "cards.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(cards, fh, ensure_ascii=False)

    axcols = ["key", "parent", "docs_now"] + AXES
    axes_df = b[[c for c in axcols if c in b.columns]].copy()
    axes_df["expert_p_axes"], axes_df["history_p_axes"], axes_df["verdict"] = b.expert_p, b.history_p, None
    axes_df.to_parquet(out / "axes.parquet")
    (out / "examples.json").write_text("{}", encoding="utf-8")
    v3 = ROOT / "artifacts" / "registry"
    for name in ("model_global.json", "calibration.json"):
        if (v3 / name).exists():
            (out / name).write_text((v3 / name).read_text(encoding="utf-8"), encoding="utf-8")
    n_src = sum(1 for k in b.key if src.get(k))
    print(f"-> {out}: записей {len(sig)}, с источниками {n_src}, векторы {E.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
