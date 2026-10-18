"""Порядок и фильтры выдачи, версия 2 (флаг REGISTRY_RANK_V2=1). Чистые функции без модели векторов — проверяются тестами.

Что меняется против версии 1 (service.py, порядок (эшелон, −балл)):
  исключения   стандарт/протокол (правило по названию из registry/axes.py или тип «протокол» от модели),
               зрелое (первый год ≤ MATURE_YEAR), зрелое по экспертной проверке базы (вердикт «M») —
               уходят из ТОП-15 в блок «исключено» с причиной (требование ТЗ: показывать логику исключения);
  слабость     0,7·экспертная + 0,3·историческая модели с осями (artifacts/registry/axes.parquet), минус штраф
               за инкрементальность (зрелый родитель на порядок объёмнее); без осей — прежний балл базы;
  порядок      близость к запросу × слабость, без приоритета эшелона: место в ТОП-15 зависит от запроса;
  группировка  дубли по смыслу не выбрасываются, а прикрепляются к первому как подсигналы («направление»).
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

MATURE_YEAR = 2019
INC_MAX, INC_PENALTY = 3.0, 0.15
GENERAL_PENALTY = 0.35         # вердикт «G» экспертной проверки базы: общая формулировка, не конкретная технология
W_EXPERT, W_HISTORY = 0.7, 0.3
_STD = (r"\b(v?\d+\.\d+|version \d|rfc ?\d+|ieee ?\d|iso ?\d|iec ?\d|3gpp|etsi|w3c|oasis|ietf)\b|"
        r"\b(protocol|protocols|standard|standards|specification|spec|interconnect express|ethernet|wi-?fi \d|"
        r"bluetooth|usb|hdmi|pcie|pci express|nvlink|cxl|ucie|uefi|onnx|openapi|a2a|mcp)\b")
STANDARD = re.compile(_STD, re.I)                        # имя стандарта где угодно в названии
STANDARD_HEAD = re.compile(r"(?:" + _STD + r")\s*$", re.I)   # имя стандарта в голове (конце) названия


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def is_standard(row: dict) -> bool:
    """Стандарт — когда название и есть имя стандарта: совпадение правила в ГОЛОВЕ фразы («a2a protocol»,
    «ultra ethernet»), либо тип «протокол» от модели, подтверждённый правилом где угодно в названии. Тема вокруг
    стандарта («mcp security», «cxl shared memory») стандартом не считается: голова — «security», «memory»."""
    phrase = str(row.get("phrase") or "").replace("_", " ").strip()
    return bool(STANDARD_HEAD.search(phrase)) or (row.get("kind") == "протокол" and bool(STANDARD.search(phrase)))


def exclusion(row: dict) -> tuple[str, str] | None:
    """(код, причина по-русски) или None. row — поля записи базы и её осей."""
    if is_standard(row):
        return "standard", "отраслевой стандарт или протокол"
    fy = _f(row.get("first_year"))
    if fy == fy and fy <= MATURE_YEAR:
        return "mature", f"зрелая технология: в науке с {int(fy)} года"
    if row.get("verdict") == "M":
        return "mature_review", "зрелая технология по экспертной проверке базы"
    return None


def incremental(row: dict) -> bool:
    v = _f(row.get("incremental_lex"))
    return v == v and v >= INC_MAX


def weakness(row: dict) -> float:
    """Балл слабости 0–1: смесь моделей с осями минус штраф за улучшение известного; без осей — прежний балл."""
    e, h = _f(row.get("expert_p_axes")), _f(row.get("history_p_axes"))
    if e == e and h == h:
        w = W_EXPERT * e + W_HISTORY * h
    elif e == e:
        w = e
    else:
        w = _f(row.get("expert"))
        if w != w:
            return 0.0
    if incremental(row) or row.get("kind") == "составной":     # составной «зрелый метод × объект» — улучшение известного
        w -= INC_PENALTY
    if row.get("verdict") == "G":
        w -= GENERAL_PENALTY
    return float(min(max(w, 0.0), 1.0))


def parent_note(row: dict) -> str | None:
    """Строка для карточки: «Родитель: quantization (объём 5 000 работ). Кандидат на порядок меньше…»."""
    p = row.get("parent")
    if not p or (isinstance(p, float) and math.isnan(p)):
        return None
    pd_, inc = _f(row.get("parent_docs_log")), _f(row.get("incremental_lex"))
    docs = int(round(math.expm1(pd_))) if pd_ == pd_ else None
    size = f" ({docs:,} работ за три года)".replace(",", " ") if docs else ""
    if inc == inc and inc >= INC_MAX:
        return (f"Родительская технология: {p}{size}. Кандидат в {2 ** inc:.0f} раз меньше родителя — "
                f"возможно улучшение известного, балл снижен.")
    if row.get("kind") == "составной":
        return f"Составной сигнал: известный метод ({p}) приходит к новому объекту; балл снижен как улучшение известного."
    return f"Родительская технология: {p}{size}. Кандидат отличим от родителя по объёму и динамике."


def order(cand: pd.DataFrame, rel: pd.Series, axes: pd.DataFrame | None) -> pd.Series:
    """Порядок кандидатов: процентиль близости к запросу × слабость. Возвращает Series order по индексу cand."""
    rows = cand.to_dict("index")
    if axes is not None:
        ax = axes.set_index("key").to_dict("index")
        rows = {i: {**r, **ax.get(r.get("key"), {})} for i, r in rows.items()}
    w = pd.Series({i: weakness(r) for i, r in rows.items()})
    rp = rel.rank(pct=True)
    return (rp * w).rename("order")


def group(indices: list[int], is_dup) -> list[tuple[int, list[int]]]:
    """Группировка: кандидат, похожий на уже взятый (is_dup(i, j) == True), становится подсигналом j."""
    out: list[tuple[int, list[int]]] = []
    for i in indices:
        for j, subs in out:
            if is_dup(i, j):
                subs.append(i)
                break
        else:
            out.append((i, []))
    return out


def axes_view(row: dict) -> dict:
    """Числа осей для карточки (витрина рисует полоски из этих значений 0–1)."""
    inc = _f(row.get("incremental_lex"))
    acc = _f(row.get("stage_accel"))
    only = _f(row.get("layer_only_science"))
    nov = _f(row.get("doc_novelty_mean"))
    out = {
        "incremental": None if inc != inc else float(min(max(inc / 6.0, 0.0), 1.0)),
        "stage_early": None if acc != acc else float(1 / (1 + math.exp(-acc))),
        "layer_early": None if only != only else float(only),
        "novelty": None if nov != nov else float(min(max(nov / 0.5, 0.0), 1.0)),
        "weakness": weakness(row),
    }
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}
