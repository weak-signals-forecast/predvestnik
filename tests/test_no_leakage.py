"""Тест на утечку из будущего: признаки термина на момент заморозки F не должны зависеть
от документов, опубликованных после F.

Строим синтетический корпус 2015–2026, считаем признаки при заморозке 2019, затем заменяем
ВСЕ документы после 2019 случайным текстом, темами и организациями и считаем снова.
Для каждого термина-кандидата все признаки обязаны совпасть до последнего знака.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecast import FEATURES, context, features_at, unit_terms  # noqa: E402

FREEZE = 2019
VOCAB = [f"word{i}" for i in range(60)]
SIGNALS = ["turquoise hydrogen", "molten metal", "digital twin", "swing adsorption", "fiber optic", "leak detection"]
TOPICS = [f"Topic {i}" for i in range(8)]
ORGS = [f"University {i}" for i in range(15)] + ["Sinopec (China)", "Shell (Netherlands)"]


def make_corpus(seed=0, n=900):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        year = 2015 + i % 12
        words = rng.sample(VOCAB, 12)
        # сигнальные термины появляются с разной интенсивностью по годам, часть уже до заморозки
        sig = rng.choice(SIGNALS) if rng.random() < 0.6 else None
        title = " ".join(words[:4]) + (f" {sig}" if sig and rng.random() < 0.7 else "")
        abstract = " ".join(words[4:]) + (f" {sig} study" if sig else "")
        rows.append(dict(
            id=f"W{i}", year=year, title=title, abstract=abstract, topic=rng.choice(TOPICS),
            orgs=rng.sample(ORGS, rng.randint(1, 3)), countries=rng.sample(["CN", "US", "DE", "RU"], rng.randint(1, 2)),
            venue=f"Journal {rng.randint(1, 6)}", domains=[rng.choice(["hydrogen", "ccus_methane", "pipelines"])],
            doc_type=rng.choice(["article", "article", "preprint"]), language="en",
        ))
    df = pd.DataFrame(rows)
    df["text"] = (df["title"] + ". " + df["abstract"]).str.lower()
    df["title_l"] = df["title"].str.lower()
    return df


def scramble_future(df, seed=1):
    """Всё, что после заморозки, заменяем на несвязанный мусор: другой словарь, темы, организации."""
    rng = random.Random(seed)
    df = df.copy()
    fut = df["year"] > FREEZE
    junk_vocab = [f"zzz{i}" for i in range(80)]
    for idx in df.index[fut]:
        words = rng.sample(junk_vocab, 12)
        df.at[idx, "title"] = " ".join(words[:4])
        df.at[idx, "abstract"] = " ".join(words[4:])
        df.at[idx, "topic"] = "Junk topic"
        df.at[idx, "orgs"] = ["Junk org"]
        df.at[idx, "countries"] = ["XX"]
        df.at[idx, "venue"] = "Junk venue"
        df.at[idx, "doc_type"] = "preprint"
    df["text"] = (df["title"] + ". " + df["abstract"]).str.lower()
    df["title_l"] = df["title"].str.lower()
    return df


def features(df):
    ctx = context(df)
    U = unit_terms(df, ctx)
    X = features_at(U, ctx, FREEZE)
    return X.set_index("term")[FEATURES].sort_index()


@pytest.fixture(scope="module")
def pair():
    df = make_corpus()
    return features(df), features(scramble_future(df))


def test_candidate_population_exists(pair):
    real, _ = pair
    assert len(real) >= 3, "синтетический корпус должен давать кандидатов, иначе тест пустой"


def test_features_at_freeze_do_not_depend_on_future(pair):
    real, scrambled = pair
    common = real.index.intersection(scrambled.index)
    assert len(common) == len(real), f"после замены будущего пропали кандидаты: {set(real.index) - set(common)}"
    diff = (real.loc[common] - scrambled.loc[common]).abs().max()
    bad = diff[diff > 1e-9]
    assert bad.empty, f"признаки зависят от будущего: {bad.to_dict()}"


def test_future_edits_do_change_labels_but_not_features(pair):
    """Контроль: будущее действительно изменилось (иначе тест выше тривиален)."""
    df = make_corpus()
    scr = scramble_future(df)
    assert (df.loc[df.year > FREEZE, "text"] != scr.loc[scr.year > FREEZE, "text"]).all()
    assert (df.loc[df.year <= FREEZE, "text"] == scr.loc[scr.year <= FREEZE, "text"]).all()
