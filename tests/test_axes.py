"""Оси слабости (registry/axes.py): правила на синтетических рядах, без корпуса и сети."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from registry import axes
from registry.corpus import SOURCES, YEARS


def test_parent_found_for_tail_and_without_first_word():
    a = pd.DataFrame({"phrase": ["quantization", "mixed precision quantization", "adaptive mixed precision quantization",
                                 "brand new thing"],
                      "first_year": [2012, 2018, 2024, 2025]})
    docs_now = np.array([5000.0, 400.0, 20.0, 30.0])
    p = axes.parent_axis(a, docs_now, cutoff=2026)
    assert p.parent[2] == "quantization"            # самый объёмный из зрелых родителей
    assert p.parent[1] == "quantization"
    assert pd.isna(p.parent[3]) and np.isnan(p.incremental_lex[3])
    assert p.incremental_lex[2] > p.incremental_lex[1] > 0


def test_parent_must_be_mature():
    a = pd.DataFrame({"phrase": ["young parent", "child of young parent"], "first_year": [2024, 2025]})
    p = axes.parent_axis(a, np.array([5000.0, 10.0]), cutoff=2026)
    assert pd.isna(p.parent[1])                      # родитель моложе cutoff−PARENT_AGE — не зрелый


def test_stage_axis_accel_and_peak():
    share = np.array([[0, 0, 1, 2, 4, 8],      # ускоряется ровно: log-приросты равны → accel 0
                      [0, 1, 4, 5, 5.5, 5.8],  # замедляется → accel < 0
                      [1, 4, 8, 6, 3, 1]],     # прошла пик → peak_ratio < 0,5
                     dtype=float) * 1e-4
    s = axes.stage_axis(share, docs_all=np.array([100, 100, 100.0]), docs_now=np.array([90, 50, 10.0]))
    assert abs(s.stage_accel[0]) < 1e-3
    assert s.stage_accel[1] < 0
    assert s.stage_peak_ratio[2] < 0.5
    assert s.stage_recent_share[0] == pytest.approx(0.9)


def test_layer_axis_lags(tmp_path):
    years = np.array(YEARS)
    df = np.zeros((2, 3, len(years)), dtype=np.int32)
    yi = {y: i for i, y in enumerate(years)}
    df[0, 0, yi[2023]] = 3            # hn в 2023
    df[0, 2, yi[2025]] = 1            # пресса в 2025
    np.savez(tmp_path / "business_series.npz", df=df, keys=np.array(["a", "b"]), sources=np.array(["hn", "yc", "press"]),
             years=years, totals=np.ones((3, len(years))))
    out = axes.layer_axis(pd.Series(["a", "b"]), np.array([2021.0, 2022.0]), tmp_path, cutoff=2026)
    assert out.layer_count[0] == 3 and out.layer_only_science[0] == 0
    assert out.layer_lag_hn[0] == 2 and out.layer_lag_market[0] == 4
    assert out.layer_count[1] == 1 and out.layer_only_science[1] == 1
    # срез 2022: hn 2023 и пресса 2025 ещё не видны
    past = axes.layer_axis(pd.Series(["a"]), np.array([2021.0]), tmp_path, cutoff=2022)
    assert past.layer_only_science[0] == 1


@pytest.mark.parametrize("phrase,expected", [
    ("ultra ethernet", True), ("ucie chiplet interconnect", True), ("agent protocol", True), ("http 2.0", True),
    ("rhombohedral multilayer graphene", False), ("agentic payments", False), ("tool poisoning", False)])
def test_standard_rule(phrase, expected):
    assert bool(axes.STANDARD.search(phrase)) is expected


def test_compute_end_to_end(tmp_path):
    """Мини-реестр: три фразы, два источника, ряды по годам — compute() даёт все оси."""
    phrases = ["language models", "medical language models", "wild new thing"]
    a = pd.DataFrame({"phrase": phrases, "key": phrases, "eid": [0, 1, 2], "first_year": [2012, 2024, 2025]})
    a.to_parquet(tmp_path / "all_phrases.parquet")
    df = np.zeros((3, len(SOURCES), len(YEARS)), dtype=np.int32)
    yi = {y: i for i, y in enumerate(YEARS)}
    for y in YEARS:
        df[0, 0, yi[y]] = 300
    df[1, 0, yi[2024]], df[1, 0, yi[2025]], df[1, 0, yi[2026]] = 2, 6, 20
    df[2, 1, yi[2025]], df[2, 1, yi[2026]] = 3, 9
    np.savez(tmp_path / "counts.npz", df=df, sources=np.array(SOURCES), years=np.array(YEARS))
    (tmp_path / "totals.json").write_text(json.dumps({f"{s}|{y}": 10000 for s in SOURCES for y in YEARS}))
    out = axes.compute(tmp_path, cutoff=2026)
    assert list(out.key) == phrases
    assert out.parent[1] == "language models" and out.incremental_lex[1] > 3
    assert pd.isna(out.parent[2])
    assert out.stage_recent_share[1] == 1.0
    assert set(axes.AXES) <= set(out.columns)
