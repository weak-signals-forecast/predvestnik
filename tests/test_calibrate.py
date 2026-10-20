"""Калибровка балла (registry/calibrate.py) и уверенность в карточке (registry_service/rank.confidence)."""
from __future__ import annotations

import numpy as np

from registry import calibrate
from registry_service import rank


def test_fit_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    oof = rng.uniform(0, 1, 600)
    y = (rng.uniform(0, 1, 600) < oof ** 2).astype(int)        # вероятность растёт с баллом
    t = calibrate.fit(oof, y)
    assert t["x"][0] == 0.0 and t["x"][-1] == 1.0
    assert all(b >= a for a, b in zip(t["y"], t["y"][1:]))       # изотоническая: не убывает
    assert 0.0 <= min(t["y"]) and max(t["y"]) <= 1.0
    assert calibrate.apply(t, 0.9) > calibrate.apply(t, 0.2)


def test_apply_clips_and_handles_nan():
    t = {"x": [0.0, 0.5, 1.0], "y": [0.1, 0.4, 0.8]}
    assert calibrate.apply(t, -1) == 0.1 and calibrate.apply(t, 2) == 0.8
    assert abs(calibrate.apply(t, 0.25) - 0.25) < 1e-9
    assert np.isnan(calibrate.apply(t, float("nan")))


def test_confidence_rounds_to_five_percent():
    t = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
    assert rank.confidence({"expert_p_axes": 0.63}, t) == 0.65
    assert rank.confidence({"expert_p_axes": 0.63}, None) is None
    assert rank.confidence({"expert": 0.63}, t) is None
