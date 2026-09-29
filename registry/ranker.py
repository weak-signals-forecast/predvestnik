"""Шаг 8. Обученный балл сигнальности вместо ручной формулы (этап 1 ТЗ и ранжирование этапа 2).

    python -m registry.ranker            # обучение на dev-половине, оценка мест test-половины, метрики этапа 1

Таблица сущностей: научные признаки (all_phrases.parquet) и бизнес-признаки (all_business.parquet),
соединённые по ключу сущности. Признаки — только измерения корпуса, без названий.

Классы при обучении:
  1  слабые сигналы организаторов: сущности, размеченные как «точно» или «ядро» в semantic_pairs.csv;
     для оценки ранжирования берётся ТОЛЬКО dev-половина (нечётные номера);
  0  отрицательные примеры команды (data/labels/negatives.yaml): зрелые технологии и угасший хайп;
  0  фон: случайная выборка кандидатов реестра (PU-обучение: среди них могут быть и настоящие сигналы,
     но в массе это шум, общие слова и продукты — ровно то, что нужно опустить в выдаче).
Модель — логистическая регрессия на стандартизованных признаках: её веса можно показать и объяснить.

Оценка:
  ранжирование — места test-сигналов (чётные номера, не участвовали в обучении) среди всех кандидатов реестра,
                 против ручного балла;
  этап 1 — кросс-валидация 5×: сигналы организаторов (все, что нашлись в реестре) против отрицательных
           примеров; accuracy, precision, recall, F1.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import yaml

from .corpus import REGISTRY
from .semantic import LABELS, _key
from .text import entity_key

REG = REGISTRY
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BACKGROUND_N = 3000
SEED = 7

FEATURES = ["sci_log_volume", "sci_growth", "sci_age", "sci_log_rate", "sci_spread", "sci_sources", "sci_oa_share",
            "biz_log_hn", "biz_hn_growth", "biz_log_yc", "biz_log_press", "biz_log_launches",
            "layers", "n_words", "name_like", "specificity", "wiki_present", "biz_log_raised", "biz_funded"]


def table() -> pd.DataFrame:
    sci = pd.read_parquet(REG / "all_phrases.parquet")
    biz = pd.read_parquet(REG / "all_business.parquet")
    s = pd.DataFrame({"key": sci.key, "sci_phrase": sci.phrase, "sci_candidate": sci.candidate, "sci_score": sci.score,
                      "sci_log_volume": np.log1p(sci.volume_recent), "sci_growth": sci.growth,
                      "sci_age": np.where(sci.first_year > 0, 2026 - sci.first_year, np.nan),
                      "sci_log_rate": np.log10(sci.max_rate_recent + 1e-7), "sci_spread": sci.spread,
                      "sci_sources": sci.sources_recent,
                      "sci_oa_share": sci.oa_recent / (sci.oa_recent + sci.ax_recent).replace(0, np.nan)})
    b = pd.DataFrame({"key": biz.key, "biz_phrase": biz.phrase, "biz_candidate": biz.candidate, "biz_score": biz.score,
                      "biz_log_hn": np.log1p(biz.hn_recent), "biz_hn_growth": biz.growth, "biz_log_yc": np.log1p(biz.yc_recent),
                      "biz_log_press": np.log1p(biz.press_recent), "biz_log_launches": np.log1p(biz.hn_launches)})
    t = s.merge(b, on="key", how="outer")
    t["phrase"] = t.sci_phrase.fillna(t.biz_phrase)
    # энциклопедия (поздняя ступень лестницы) и деньги (Form D через компании YC), если слои собраны
    wp, fp = REG / "wiki.parquet", REG / "funding.parquet"
    wiki = set(pd.read_parquet(wp).key) if wp.exists() else set()
    t["wiki_present"] = t.key.isin(wiki).astype(int)
    if fp.exists():
        f = pd.read_parquet(fp)
        t = t.merge(f[["key", "raised_recent", "funded_recent"]], on="key", how="left")
    else:
        t["raised_recent"], t["funded_recent"] = 0.0, 0
    t["biz_log_raised"] = np.log1p(t.raised_recent.fillna(0) / 1e6)
    t["biz_funded"] = t.funded_recent.fillna(0)
    t["candidate"] = t.sci_candidate.fillna(False).astype(bool) | t.biz_candidate.fillna(False).astype(bool)
    t["layers"] = (t.sci_log_volume.fillna(0) > 0).astype(int) + (t.biz_log_hn.fillna(0) > 0) + \
                  (t.biz_log_yc.fillna(0) > 0) + (t.biz_log_press.fillna(0) > 0)
    t["n_words"] = t.phrase.str.count(" ") + 1
    t["name_like"] = t.phrase.str.contains(r"\d").astype(int)
    # специфичность: насколько редко самое редкое слово фразы встречается в словаре научного слоя.
    # «financial insights» — оба слова повсюду; «tokenized money market fund» — «tokenized» редкое.
    import collections
    wc = collections.Counter(w for ph in (REG / "vocab.txt").read_text(encoding="utf-8").split("\n") if ph
                             for w in set(ph.split()))
    t["specificity"] = t.phrase.map(lambda ph: np.log10(min((wc.get(w, 1) for w in ph.split()), default=1)))
    # ручной балл для сравнения: лучший из двух слоёв после приведения к единой шкале (ранги)
    t["hand_rank_score"] = np.fmax(t.sci_score.rank(pct=True), t.biz_score.rank(pct=True))
    return t


def labels(t: pd.DataFrame) -> tuple[dict[str, int], set[str]]:
    """Ключ сущности -> id сигнала (по разметке «ядро» и выше); ключи отрицательных примеров."""
    lab = pd.read_csv(LABELS)
    lab = lab[(lab.same >= 0.5) & ~lab.candidate.str.contains(" × ")]
    pos = {}
    for _, r in lab.iterrows():
        pos.setdefault(_key(r.candidate), int(r.id))
    neg = set()
    for area in yaml.safe_load(open(ROOT / "data/labels/negatives.yaml", encoding="utf-8")).values():
        for kind in area.values():
            for en, _ru in kind:
                neg.add(entity_key(" ".join(en.lower().replace("-", " ").split())))
    keys = set(t.key)
    return {k: v for k, v in pos.items() if k in keys}, neg & keys


def fit(X: pd.DataFrame, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                      LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))
    return m.fit(X, y)


def main() -> int:
    t = table()
    pos, neg = labels(t)
    rng = np.random.default_rng(SEED)
    cand = t[t.candidate]
    dev_pos = {k for k, i in pos.items() if i % 2 == 1}
    test_pos = {k: i for k, i in pos.items() if i % 2 == 0}
    bg = cand[~cand.key.isin(dev_pos | neg)].sample(BACKGROUND_N, random_state=SEED)
    train = pd.concat([t[t.key.isin(dev_pos)].assign(y=1), t[t.key.isin(neg)].assign(y=0), bg.assign(y=0)])
    m = fit(train[FEATURES], train.y.values)
    t["learned"] = m.predict_proba(t[FEATURES])[:, 1]

    print(f"сущностей {len(t):,}, кандидатов реестра {len(cand):,}; сигналов в реестре: dev {len(dev_pos)}, test {len(test_pos)}; "
          f"отрицательных примеров найдено {len(neg)}".replace(",", " "))
    c = t[t.candidate | t.key.isin(test_pos)].copy()   # test-сигналы ранжируем среди кандидатов реестра
    for col, name in (("hand_rank_score", "ручной балл"), ("learned", "обученный балл")):
        c["r"] = c[col].rank(ascending=False, method="min")
        best = c[c.key.isin(test_pos)].assign(id=lambda d: d.key.map(test_pos)).groupby("id").r.min()
        print(f"  {name:15}: test-сигналов {len(best)}, место среди {len(c):,} — медиана {int(best.median()):,}, "
              f"в первых 100: {int((best <= 100).sum())}, 1000: {int((best <= 1000).sum())}, 5000: {int((best <= 5000).sum())}".replace(",", " "))

    coef = m[-1].coef_[0][:len(FEATURES)]
    print("\nВеса признаков (стандартизованные):")
    for f, w in sorted(zip(FEATURES, coef), key=lambda x: -abs(x[1])):
        print(f"  {f:18} {w:+.2f}")

    # этап 1: кросс-валидация, сигналы организаторов против отрицательных примеров команды
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedGroupKFold
    # по одной сущности на сигнал (самая «весомая» по научному объёму), варианты одного сигнала не делятся
    # между обучением и проверкой; отрицательные — каждый своя группа
    P = t[t.key.isin(set(pos))].assign(y=1, grp=lambda d: d.key.map(pos))
    P = P.sort_values("sci_log_volume", ascending=False).drop_duplicates("grp")
    N = t[t.key.isin(neg)].assign(y=0)
    N["grp"] = -np.arange(1, len(N) + 1)
    d = pd.concat([P, N]).reset_index(drop=True)
    pred = np.zeros(len(d))
    for rep in range(5):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=rep)
        for tr, te in cv.split(d[FEATURES], d.y.values, d.grp.values):
            mm = clone(m).fit(d[FEATURES].iloc[tr], d.y.values[tr])
            pred[te] += mm.predict_proba(d[FEATURES].iloc[te])[:, 1] / 5
    yhat = pred >= 0.5
    tp, fp = int((yhat & (d.y == 1)).sum()), int((yhat & (d.y == 0)).sum())
    fn = int((~yhat & (d.y == 1)).sum())
    acc = float((yhat == (d.y == 1)).mean())
    pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    print(f"\nЭтап 1 (CV 5×5 по группам-сигналам, {int(d.y.sum())} сигналов против {int((d.y == 0).sum())} отрицательных): "
          f"accuracy {acc:.0%}, precision {pr:.0%}, recall {rc:.0%}, F1 {2 * pr * rc / max(1e-9, pr + rc):.0%}")
    t[["key", "phrase", "candidate", "learned", "hand_rank_score", *FEATURES]].to_parquet(REG / "ranked.parquet", index=False)
    import joblib
    joblib.dump({"model": m, "features": FEATURES}, REG / "ranker.joblib")   # для разложения балла в карточке
    return 0


if __name__ == "__main__":
    sys.exit(main())
