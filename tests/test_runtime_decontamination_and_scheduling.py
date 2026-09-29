"""Деконтаминация рантайма от эталона организаторов и порядок работ по бюджету.

Сеть не используется, часы фальшивые. Ни одного обращения к живому провайдеру.
"""
import collections
import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import query, relevance, ru  # noqa: E402
from wsignals.sources import github, hackernews, news, openalex, wikipedia  # noqa: E402

DATASET = ROOT / "data" / "dataset" / "technologies.csv"


def _rows():
    with DATASET.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# Таблица организаторов в публичный репозиторий не входит: в датасете только 198 строк команды.
# Проверки, которым нужны сами строки организаторов, выполняются там, где таблица есть локально.
needs_organizer_rows = pytest.mark.skipif(
    not any(r["label"] == ru.ORGANIZER_LABEL for r in _rows()),
    reason="таблицы организаторов нет в репозитории")


# ============ A. эталон организаторов не является авторитетом отрисовки ============

@needs_organizer_rows
def test_organizer_rows_are_identifiable_by_two_independent_markers():
    rows = _rows()
    by_label = {r["tech_id"] for r in rows if r["label"] == ru.ORGANIZER_LABEL}
    by_columns = {r["tech_id"] for r in rows
                  if any(str(r.get(c) or "").strip() for c in ru.ORGANIZER_COLUMNS)}
    assert by_label == by_columns and len(by_label) == 100


def test_organizer_names_are_absent_from_the_runtime_glossary():
    organizer = {r["name"].strip() for r in _rows() if r["label"] == ru.ORGANIZER_LABEL}
    rendered = {name for _, _, name in ru.glossary()}
    assert rendered.isdisjoint(organizer)
    assert len(ru.glossary()) == len(_rows()) - len(organizer)


@needs_organizer_rows
def test_mcp_security_cannot_become_the_organizer_reference_name():
    """Измеренная утечка: кандидат из источников отрисовывался ДОСЛОВНЫМ названием строки S004."""
    s004 = next(r for r in _rows() if r["tech_id"] == "S004")
    rendered = ru.technology_name("mcp security")
    assert rendered["ru"] != s004["name"].strip()
    assert rendered["original"] == "mcp security"
    assert rendered["source"] != "словарь терминов проекта" or rendered["ru"] not in {
        r["name"].strip() for r in _rows() if r["label"] == ru.ORGANIZER_LABEL}


@needs_organizer_rows
@pytest.mark.parametrize("phrase", ["mcp security", "agent identity", "model compression",
                                    "rag poisoning", "prompt injection"])
def test_no_upstream_candidate_renders_as_an_organizer_reference_name(phrase):
    organizer = {r["name"].strip() for r in _rows() if r["label"] == ru.ORGANIZER_LABEL}
    assert ru.technology_name(phrase)["ru"] not in organizer


def test_runtime_modules_do_not_read_the_organizer_workbook():
    """Таблица организаторов читается только инструментом построения датасета.

    `wsignals/dataset.py` — офлайновый инструмент: он не импортируется ни одним модулем пути
    запроса, что здесь и проверяется вместе с отсутствием других ссылок."""
    import ast

    importers = []
    for path in (ROOT / "wsignals").rglob("*.py"):
        if path.name == "dataset.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(a.name.endswith("dataset") for a in node.names):
                importers.append(path.name)
            if isinstance(node, ast.ImportFrom) and any(a.name == "dataset" for a in node.names):
                importers.append(path.name)
    assert importers == [], importers
    offenders = [p.name for p in (ROOT / "wsignals").rglob("*.py")
                 if p.name != "dataset.py" and "вводные данные" in p.read_text(encoding="utf-8")]
    assert offenders == [], offenders


def test_no_hardcoded_technology_list_replaced_the_glossary():
    """Эталон убран, но подменять его другим зашитым перечнем технологий нельзя.

    Проверяется ИСПОЛНЯЕМЫЙ код: в пояснениях строка S004 упоминается как разобранный пример
    утечки, и это нормально."""
    import ast

    tree = ast.parse((ROOT / "wsignals" / "ru.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    code = ast.unparse(tree)
    organizer_names = {r["name"].strip() for r in _rows() if r["label"] == ru.ORGANIZER_LABEL}
    for name in organizer_names:
        assert name not in code, name
    for leaked in ("S001", "S002", "S004", "tech_id"):
        assert leaked not in code, leaked


# ============ B. стартовый размер семантической порции ============

def test_semantic_batch_starts_at_the_measured_working_size():
    assert relevance.LLM_BATCH == 4
    assert relevance.SEMANTIC_MIN_BATCH <= relevance.LLM_BATCH


# ============ C/D/E. порядок работ и пропуск терминальных ============

class Clock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def advance(self, d):
        self.t += d

    def sleep(self, d):
        self.t += d


@pytest.fixture
def scheduler(monkeypatch):
    """Подменяет все адаптеры и часы; записывает порядок обращений."""
    from wsignals import http

    clock, calls = Clock(), []
    monkeypatch.setattr(http, "monotonic", clock.monotonic)
    monkeypatch.setattr(http.time, "sleep", clock.sleep)

    def stub(name, values, cost):
        def fn(phrase, *a, **kw):
            clock.advance(cost)
            calls.append((name, phrase, round(clock.t - 1000.0, 2)))
            return dict(values)
        return fn

    monkeypatch.setattr(openalex, "stats", stub("openalex", {
        "oa_total": 40, "oa_growth": 1.8, "oa_age": 2, "oa_last_share": 0.45, "oa_recent": 20,
        "oa_preprint_share": 0.3, "oa_peak_ratio": 1.2, "oa_fields": 2, "oa_prior": 3}, 0.62))
    monkeypatch.setattr(wikipedia, "stats", stub("wiki", {"wiki_article": 0, "wiki_age": 0}, 0.55))
    monkeypatch.setattr(news, "stats", stub("news", {
        "news_1y": 8, "news_30d": 3, "news_30d_share": 0.37, "news_press_share": 0.1}, 0.9))
    monkeypatch.setattr(hackernews, "stats", stub("hn", {
        "hn_total": 6, "hn_12m": 5, "hn_growth": 1.1}, 0.7))
    monkeypatch.setattr(github, "stats", stub("github", {
        "gh_total": 30, "gh_12m": 25, "gh_new_share": 0.8}, 0.8))
    return clock, calls


def _cand(i, noise=False):
    c = {"phrase": f"alpha beta{i} gamma", "original_phrase": f"alpha beta{i} gamma",
         "canonical_label": f"alpha beta{i} gamma", "burst": float(1000 - i),
         "channels": {"литература": 3}, "recent_docs": 3, "prior_docs": 0,
         "examples": [], "repos": [], "assessment": {"specificity": 1.0}}
    if noise:
        c["noise_reason"] = "пересказ задачи поиска, а не название технологии"
    return c


def _run(calls_clock, live=40, terminal=60, workset=30, network=82.0):
    from wsignals import http

    cands = [_cand(i) for i in range(live)] + [_cand(1000 + i, noise=True) for i in range(terminal)]
    timings = {}
    with http.fetch_context(budget_s=network):
        rows = query.cascade_collect(cands, workers=8, log=lambda *a: None, keep_wiki=24,
                                     keep_full=14, budget={"stage1": network * 0.18,
                                                           "stage2": network * 0.16},
                                     corpus_adapters={}, with_github_stats=False,
                                     timings=timings, admission_workset=workset)
    return rows, timings


def test_terminal_candidates_get_no_expensive_request(scheduler):
    clock, calls = scheduler
    rows, timings = _run(scheduler)
    asked = {p for name, p, _ in calls if name == "openalex"}
    terminal_phrases = {r["cand"]["phrase"] for r in rows if r["cand"].get("noise_reason")}
    assert timings["stage1_skipped_terminal"] == 60
    assert asked.isdisjoint(terminal_phrases), "по терминальному кандидату запрос не делается"


def test_skipped_candidates_stay_visible_and_counted(scheduler):
    rows, timings = _run(scheduler)
    skipped = [r for r in rows if r.get("not_evaluated")]
    assert len(skipped) == 60
    assert all(r["decision"] == query.NOISE for r in skipped), "решение сохраняется"
    assert all(r["reasons"] for r in skipped), "причина сохраняется для аудита"


def test_a_skipped_candidate_is_not_reported_as_a_provider_failure(scheduler):
    """F: «запрос не начинали» не то же самое, что «провайдер отказал»."""
    rows, _ = _run(scheduler)
    for r in rows:
        if r.get("not_evaluated"):
            # Наблюдения источника, которого не спрашивали, нет вовсе: ни нуля, ни отказа.
            assert "oa_total" not in r["state"].values(), "ноль не выдумывается"
            assert "openalex" not in r["state"].failed(), "и отказом это не считается"
            assert not r["state"].ok("openalex")


def test_admission_critical_news_runs_before_optional_enrichment(scheduler):
    clock, calls = scheduler
    _run(scheduler)
    order = [n for n, _, _ in calls]
    assert "news" in order and "hn" in order
    assert order.index("news") < order.index("hn"), order[:5]


def test_admission_critical_measurement_reaches_the_whole_workset(scheduler):
    """Измеренный дефект: новости не успевали начаться ни для одного кандидата."""
    clock, calls = scheduler
    rows, timings = _run(scheduler)
    stage3 = [r for r in rows if r.get("stage") == 3]
    measured = {p for n, p, _ in calls if n == "news"}
    assert stage3, "ступень 3 обязана состояться"
    assert all(r["cand"]["phrase"] in measured for r in stage3)
    assert timings["stage3_admission_critical"] == len(stage3)


def test_expensive_late_work_is_bounded_by_the_output_size(scheduler):
    rows, timings = _run(scheduler, workset=6)
    assert timings["stage3_workset"] <= 6
    assert timings["stage3_admission_critical"] <= 6


def test_budget_truncation_is_reported_only_when_candidates_were_truncated(scheduler):
    rows, timings = _run(scheduler)
    truncated = [r for r in rows if r.get("reason_code") == "budget_exhausted"]
    assert timings["stage3_admission_critical"] > 0
    assert (not truncated) == (timings["stage3_admission_critical"] == len(
        [r for r in rows if r.get("stage") == 3]))


def test_no_deadline_was_increased():
    plan = query.budget_plan()
    assert plan["hard_wall_budget_s"] == 90.0
    assert plan["network_deadline_s"] == 82.0
