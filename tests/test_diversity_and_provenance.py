"""Разнообразие ТОПа без ложной склейки и происхождение запуска (R2-G, R2-H)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import provenance, query, relevance  # noqa: E402


def cand(phrase, burst=1.0):
    return {"cand": {"phrase": phrase, "original_phrase": phrase, "canonical_label": phrase, "burst": burst},
            "score": burst}


# ---------- R2-G: разнообразие ----------

def test_diversity_moves_siblings_down_without_dropping_them():
    rows = [cand("rag security", 1.00), cand("llm security", 0.98), cand("mcp security", 0.96),
            cand("agentic payments", 0.90), cand("token compression", 0.85)]
    ordered = query.diversify(rows, lambda r: r["score"])
    phrases = [r["cand"]["phrase"] for r in ordered]
    assert len(phrases) == len(rows), "кандидаты не выбрасываются"
    assert set(phrases) == {r["cand"]["phrase"] for r in rows}, "кандидаты не выдумываются"
    # Первые три места больше не заняты одним головным словом.
    heads = [query.sibling_key(r["cand"]) for r in ordered[:3]]
    assert len(set(heads)) == 3, phrases


def test_original_score_is_preserved_and_diversity_rank_is_separate():
    rows = [cand("rag security", 1.0), cand("llm security", 0.9), cand("quantum sensing", 0.5)]
    ordered = query.diversify(rows, lambda r: r["score"])
    assert [r["score"] for r in rows] == [1.0, 0.9, 0.5], "исходные баллы не меняются"
    assert [r["diversity_rank"] for r in ordered] == [1, 2, 3]
    assert all("diversity_adjusted_score" in r for r in ordered)


def test_the_strongest_candidate_still_leads():
    rows = [cand("rag security", 1.0), cand("llm security", 0.99), cand("quantum sensing", 0.2)]
    ordered = query.diversify(rows, lambda r: r["score"])
    assert ordered[0]["cand"]["phrase"] == "rag security"


def test_diversity_never_merges_distinct_technologies():
    """Разнообразие — это только порядок. Ключи склейки родственников остаются разными."""
    rows = [cand("rag security", 1.0), cand("llm security", 0.9), cand("mcp security", 0.8)]
    query.diversify(rows, lambda r: r["score"])
    keys = {relevance.merge_key(r["cand"]["phrase"]) for r in rows}
    assert len(keys) == 3


def test_sibling_concentration_measures_before_and_after():
    siblings = [{"phrase": p, "canonical_label": p} for p in
                ["rag security", "llm security", "mcp security", "agentic payments"]]
    mixed = [{"phrase": p, "canonical_label": p} for p in
             ["rag security", "agentic payments", "token compression", "quantum sensing"]]
    assert query.sibling_concentration(siblings) > query.sibling_concentration(mixed)
    assert query.sibling_concentration(mixed) == 0.0
    assert query.sibling_concentration([]) == 0.0


def test_diversity_is_stable_for_a_single_candidate():
    rows = [cand("rag poisoning", 0.4)]
    ordered = query.diversify(rows, lambda r: r["score"])
    assert len(ordered) == 1 and ordered[0]["diversity_rank"] == 1


# ---------- R2-H: происхождение запуска ----------

def test_config_hash_is_deterministic_for_the_same_config():
    assert provenance.config_hash() == provenance.config_hash()


@pytest.mark.parametrize("path,value", [
    (("decision", "maturity_veto"), 0.61),
    (("decision", "hype_veto"), 0.55),
    (("evidence", "min_independent_supports"), 3),
    (("relevance", "min_specificity"), 0.9),
    (("retrieval", "keep_full"), 99),
    (("budget", "hard_wall_budget_s"), 120),
])
def test_changing_a_behavioural_setting_changes_the_hash(path, value):
    base = provenance.behavioral_config()
    before = provenance.config_hash(base)
    changed = json.loads(json.dumps(base))
    node = changed
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    assert provenance.config_hash(changed) != before, path


def test_credential_presence_is_recorded_as_a_flag_not_a_value(monkeypatch):
    monkeypatch.setenv("YANDEX_API_KEY", "super-secret-value")
    cfg = provenance.behavioral_config()
    blob = json.dumps(cfg, ensure_ascii=False)
    assert "super-secret-value" not in blob
    assert cfg["credentials_present"]["YANDEX_API_KEY"] is True


def test_credential_presence_changes_the_hash(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    without = provenance.config_hash()
    monkeypatch.setenv("OPENALEX_API_KEY", "k")
    assert provenance.config_hash() != without


def test_run_metadata_has_every_required_field():
    meta = provenance.run_metadata("слабые сигналы в кибербезопасности")
    for key in ("run_id", "commit_sha", "config_hash", "query", "as_of", "cache_state",
                "provider_status", "semantic_normalizer"):
        assert key in meta, key
    assert meta["query"] == "слабые сигналы в кибербезопасности"
    assert len(meta["run_id"]) >= 16 and len(meta["config_hash"]) >= 16


def test_run_ids_are_unique_per_run():
    assert provenance.run_metadata("q")["run_id"] != provenance.run_metadata("q")["run_id"]


def test_no_secret_values_appear_in_run_metadata(monkeypatch):
    secrets = {"YANDEX_API_KEY": "yk-secret", "GITHUB_TOKEN": "gho-secret",
               "OPENAI_API_KEY": "sk-secret", "GIGACHAT_TOKEN": "gc-secret"}
    for k, v in secrets.items():
        monkeypatch.setenv(k, v)
    blob = json.dumps(provenance.run_metadata("q"), ensure_ascii=False)
    for v in secrets.values():
        assert v not in blob, v


def test_semantic_normalizer_status_is_read_only_metadata(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    status = provenance.semantic_normalizer_status()
    assert status["configured"] is False and status["model"] is None and status["reason"]


def test_commit_sha_prefers_the_environment(monkeypatch):
    monkeypatch.setenv("WSIGNALS_COMMIT_SHA", "a" * 40)
    assert provenance.commit_sha() == "a" * 40


def test_cache_state_reports_entries(monkeypatch, tmp_path):
    from wsignals import http as http_mod

    monkeypatch.setattr(http_mod, "CACHE_DIR", tmp_path)
    assert provenance.cache_state()["entries"] == 0
    (tmp_path / "x.json").write_text("{}", encoding="utf-8")
    state = provenance.cache_state()
    assert state["entries"] == 1 and state["available"] is True and state["newest_entry_utc"]


def test_run_metadata_is_json_serializable_for_api_and_db():
    """Метаданные должны переживать сериализацию в JSON: их пишет и API, и колонка БД.
    Полный путь через FastAPI и SQLAlchemy проверяет tests/test_api_smoke.py."""
    meta = provenance.run_metadata("q")
    assert json.loads(json.dumps(meta, ensure_ascii=False))["run_id"] == meta["run_id"]
