"""Калибровка балла в вероятность: «уверенность модели > 75 %» из ТЗ должна быть вероятностью, а не процентилем.

    GPB_REGISTRY=~/gpb_corpus/registry_v3 python -m registry.calibrate      # -> artifacts/registry/calibration.json

Изотоническая регрессия на предсказаниях экспертной модели вне обучения (expert_oof_axes.csv: oof → y). Монотонная
кусочно-линейная таблица хранится в JSON; сервис применяет её к expert_p_axes и показывает «уверенность» отдельно
от балла ранжирования (балл — порядок среди записей базы, уверенность — доля сигналов среди записей с таким баллом).
Ограничение, честно: разметка — 1 026 записей, 127 сигналов; доверительный коридор широкий, поэтому в карточке
уверенность округляется до 5 %.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "registry" / "calibration.json"


def fit(oof: np.ndarray, y: np.ndarray, bins: int = 20) -> dict:
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, y)
    xs = np.linspace(0, 1, bins + 1)
    ys = iso.predict(xs)
    return {"x": [round(float(v), 4) for v in xs], "y": [round(float(v), 4) for v in ys],
            "n": int(len(y)), "positives": int(y.sum()), "method": "isotonic on out-of-fold expert_p_axes"}


def apply(table: dict, p: float) -> float:
    """Кусочно-линейная интерполяция таблицы калибровки; вне [0, 1] — обрезка."""
    if p != p:
        return float("nan")
    return float(np.interp(min(max(p, 0.0), 1.0), table["x"], table["y"]))


def main() -> int:
    src = REGISTRY / "expert_oof_axes.csv"
    if not src.exists():
        src = REGISTRY / "expert_oof.csv"
    d = pd.read_csv(src)
    t = fit(d.oof.values.astype(float), d.y.values.astype(int))
    OUT.write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")
    cal = np.array([apply(t, v) for v in d.oof])
    print(f"калибровка на {t['n']} записях ({t['positives']} сигналов) -> {OUT}")
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        m = (cal >= lo) & (cal < hi)
        if m.sum():
            print(f"  уверенность {lo:.2f}–{min(hi, 1):.2f}: записей {int(m.sum())}, доля сигналов по факту {d.y[m].mean():.2f}")
    print(f"  записей с уверенностью > 0,75: {int((cal > 0.75).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
