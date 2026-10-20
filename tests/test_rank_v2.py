"""Порядок и фильтры версии 2 (registry_service/rank.py): чистые правила, без модели векторов и сети."""
from __future__ import annotations

import math

import pandas as pd

from registry_service import rank


def test_exclusion_standard_by_rule_and_by_kind():
    assert rank.exclusion({"phrase": "ultra ethernet", "first_year": 2024}) == ("standard", "отраслевой стандарт или протокол")
    assert rank.exclusion({"phrase": "a2a protocol", "first_year": 2025})[0] == "standard"
    assert rank.exclusion({"phrase": "universal chiplet interconnect express", "first_year": 2025})[0] == "standard"
    # тема вокруг стандарта — не стандарт: голова фразы не имя стандарта
    assert rank.exclusion({"phrase": "mcp security", "first_year": 2025}) is None
    assert rank.exclusion({"phrase": "cxl shared memory", "first_year": 2025}) is None
    # тип «протокол» от модели — только с подтверждением правилом в названии
    assert rank.exclusion({"phrase": "model context protocol servers", "kind": "протокол", "first_year": 2025})[0] == "standard"
    assert rank.exclusion({"phrase": "verifiable ai", "kind": "протокол", "first_year": 2025}) is None


def test_exclusion_mature_by_year_and_by_review():
    assert rank.exclusion({"phrase": "federated learning", "first_year": 2016})[0] == "mature"
    assert rank.exclusion({"phrase": "old thing", "first_year": 2024, "verdict": "M"})[0] == "mature_review"
    assert rank.exclusion({"phrase": "agentic payments", "first_year": 2026, "verdict": "keep"}) is None
    assert rank.exclusion({"phrase": "no year at all"}) is None


def test_weakness_blend_penalty_and_fallback():
    base = {"expert_p_axes": 0.8, "history_p_axes": 0.5}
    assert rank.weakness(base) == 0.8 * 0.7 + 0.5 * 0.3
    assert rank.weakness({**base, "incremental_lex": 4.0}) == rank.weakness(base) - rank.INC_PENALTY
    assert rank.weakness({"expert": 0.63}) == 0.63                 # без осей — прежний балл базы
    assert rank.weakness({"expert": 0.63, "verdict": "G"}) == round(0.63 - rank.GENERAL_PENALTY, 10) or abs(rank.weakness({"expert": 0.63, "verdict": "G"}) - 0.43) < 1e-9
    assert abs(rank.weakness({"expert": 0.9, "kind": "составной"}) - (0.9 - rank.INC_PENALTY)) < 1e-9
    assert rank.weakness({"expert_p_axes": float("nan"), "expert": float("nan")}) == 0.0
    assert 0.0 <= rank.weakness({"expert_p_axes": 0.05, "history_p_axes": 0.0, "incremental_lex": 9}) <= 1.0


def test_parent_note_texts():
    assert rank.parent_note({"parent": None}) is None
    assert rank.parent_note({"parent": float("nan")}) is None
    inc = rank.parent_note({"parent": "quantization", "parent_docs_log": math.log1p(5000), "incremental_lex": 4.0})
    assert inc.startswith("Родительская технология: quantization (5 000 работ") and "улучшение известного" in inc
    ok = rank.parent_note({"parent": "graphene", "parent_docs_log": math.log1p(300), "incremental_lex": 1.0})
    assert "отличим от родителя" in ok


def test_order_depends_on_query_relevance_and_weakness():
    cand = pd.DataFrame({"key": ["a", "b", "c"], "expert": [0.9, 0.5, 0.5], "tier": [2, 1, 1]}, index=[10, 11, 12])
    rel = pd.Series([0.1, 0.9, 0.5], index=cand.index)
    axes = pd.DataFrame({"key": ["a", "b", "c"], "expert_p_axes": [0.9, 0.9, 0.9], "history_p_axes": [0.9, 0.9, 0.9],
                         "incremental_lex": [float("nan"), float("nan"), 5.0]})
    o = rank.order(cand, rel, axes)
    assert o[11] > o[12] > o[10]            # близкий к запросу выше; инкрементальный ниже при равной близости
    o1 = rank.order(cand, rel, None)        # без осей — прежний балл, порядок всё равно от запроса
    assert o1[11] > o1[10]


def test_group_attaches_duplicates_as_subsignals():
    dup = {(2, 1), (4, 3)}
    out = rank.group([1, 2, 3, 4, 5], lambda i, j: (i, j) in dup)
    assert out == [(1, [2]), (3, [4]), (5, [])]


def test_axes_view_ranges():
    v = rank.axes_view({"incremental_lex": 3.0, "stage_accel": 0.0, "layer_only_science": 1.0, "doc_novelty_mean": 0.25,
                        "expert_p_axes": 0.6, "history_p_axes": 0.4})
    assert v["incremental"] == 0.5 and v["stage_early"] == 0.5 and v["layer_early"] == 1.0 and v["novelty"] == 0.5
    assert v["weakness"] == round(0.6 * 0.7 + 0.4 * 0.3 - rank.INC_PENALTY, 3)
    empty = rank.axes_view({"expert": 0.3})
    assert empty["incremental"] is None and empty["weakness"] == 0.3


def test_flag_off_by_default(monkeypatch):
    """Без REGISTRY_RANK_V2 сервис остаётся на прежнем порядке (демо-путь не меняется)."""
    import importlib
    import sys
    monkeypatch.delenv("REGISTRY_RANK_V2", raising=False)
    sys.modules.pop("registry_service.service", None)
    svc = importlib.import_module("registry_service.service")
    assert svc.RANK_V2 is False and svc.DROP_MATURE is False
