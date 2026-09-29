"""Один авторитетный терминальный статус, безопасная деградация и провенанс модели (R2-A, R2-B, R2-C).

Пороги в этом файле не задаются и не подбираются: все они читаются из кода.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import decompose, provenance, query, relevance  # noqa: E402
from wsignals.decision import (ACCEPT, DETERMINISTIC_ONLY, HYPE, LOW_EVIDENCE, MATURE,  # noqa: E402
                               MIN_EMERGENCE, NOISE, SEMANTIC_UNCERTAIN, SEMANTICALLY_VERIFIED,
                               VERIFICATION_STATES, decide)
from wsignals.evidence import NO_RESULTS, OK, EvidenceState, Observation  # noqa: E402
from wsignals.schema import Document  # noqa: E402
from wsignals.trust import assess as assess_trust  # noqa: E402


def doc(title, url, source_type, name=""):
    _d = assess_trust(Document(title=title, url=url, source_name=name, source_type=source_type,
                                 published="2026-03-01", language="en")).with_provenance()
    # P1-C. Признак претензионной релевантности читается строго: подтверждает только явное
    # True. Эта фикстура ПО ПОСТРОЕНИЮ представляет документ про рассматриваемого кандидата,
    # поэтому помечается явно. Неразмеченные документы проверяются отдельно тестом,
    # доказывающим, что они не открывают ACCEPT.
    _d.claim_relevant = True
    return _d


PAPER = doc("Study", "https://doi.org/10.1145/1", "научная публикация", "ACM")
REPO = doc("acme/x", "https://github.com/acme/x", "репозиторий", "GitHub")
DOCS = [PAPER, REPO]


def state(growth=1.8, new_share=0.8, last_share=0.45, wiki=None, press=0.1,
          preprint=0.3, age=2, gh_12m=25, hn=True, news_1y=8, news_share=0.37):
    s = EvidenceState()
    s.add(Observation("openalex", OK, {"oa_total": 40, "oa_growth": growth, "oa_age": age,
                                       "oa_last_share": last_share, "oa_preprint_share": preprint,
                                       "oa_peak_ratio": 1.2, "oa_fields": 2, "oa_prior": 3}))
    s.add(Observation("github", OK, {"gh_total": 30, "gh_12m": gh_12m, "gh_new_share": new_share}))
    s.add(Observation("news", OK, {"news_1y": news_1y, "news_30d": 3, "news_30d_share": news_share,
                                   "news_press_share": press}))
    s.add(Observation("hn", OK if hn else NO_RESULTS,
                      {"hn_total": 6, "hn_12m": 5, "hn_growth": 1.1} if hn
                      else {"hn_total": 0, "hn_12m": 0, "hn_growth": 0.0}))
    s.add(Observation("wiki", OK if wiki else NO_RESULTS, wiki or {"wiki_article": 0, "wiki_age": 0}))
    return s


# Профили подобраны ИЗМЕРЕНИЕМ так, чтобы emergence_score попал ровно в заданные точки на границах
# полосы. Пороги при этом не менялись: меняются только входные наблюдения.
BOUNDARY = {
    0.149: dict(growth=0.0, new_share=0.0, last_share=0.0, preprint=0.20, age=2,
                gh_12m=2, hn=False, news_1y=4, news_share=0.0),
    0.150: dict(growth=0.0, new_share=0.0, last_share=0.0, preprint=0.15, age=2,
                gh_12m=2, hn=False, news_1y=4, news_share=0.0),
    0.151: dict(growth=0.0, new_share=0.0, last_share=0.0, preprint=0.20, age=2,
                gh_12m=2, hn=False, news_1y=4, news_share=0.0),
    0.299: dict(growth=0.0, new_share=0.2, last_share=0.1, preprint=0.30, age=2,
                gh_12m=2, hn=False, news_1y=4, news_share=0.0),
    0.300: dict(growth=0.0, new_share=0.3, last_share=0.2, preprint=0.30, age=2,
                gh_12m=2, hn=False, news_1y=4, news_share=0.0),
}
BOUNDARY_BURST = {0.149: 0.2, 0.150: 1.0, 0.151: 0.3, 0.299: 2.0, 0.300: 0.3}


# ---------- R2-A: единственный авторитетный терминальный статус ----------

def test_terminal_state_is_defined_for_every_decision():
    from wsignals.decision import ACCEPT as A
    assert set(query.TERMINAL_STATE) == {A, LOW_EVIDENCE, MATURE, HYPE, NOISE}
    for decision, (tier, status) in query.TERMINAL_STATE.items():
        assert tier and status, decision


def test_only_accept_maps_to_weak_signal():
    assert query.terminal_state(ACCEPT) == ("подтверждённый слабый сигнал", "слабый сигнал")
    for decision in (LOW_EVIDENCE, MATURE, HYPE, NOISE):
        assert query.terminal_state(decision)[1] != query.WEAK_SIGNAL_STATUS, decision


def test_low_evidence_maps_to_watchlist():
    tier, status = query.terminal_state(LOW_EVIDENCE)
    assert "дополнительных доказательств" in tier and status == query.WATCHLIST_STATUS


@pytest.mark.parametrize("decision", [MATURE, HYPE, NOISE])
def test_vetoed_decisions_map_to_excluded(decision):
    assert query.terminal_state(decision)[1] == "исключён"


def test_no_second_classifier_hides_in_status():
    """`status` обязан быть чистой функцией решения: одинаковое решение — одинаковый статус."""
    seen = {}
    for decision in query.TERMINAL_STATE:
        seen.setdefault(decision, query.terminal_state(decision))
        assert query.terminal_state(decision) == seen[decision]
    assert len({s for _, s in query.TERMINAL_STATE.values()}) == 3   # сигнал / наблюдение / исключён


def test_status_no_longer_depends_on_the_emergence_threshold():
    """Порог WATCHLIST_EMERGENCE остаётся для подачи, но терминальную классификацию не определяет."""
    source = (ROOT / "wsignals" / "query.py").read_text(encoding="utf-8")
    card = source[source.index("def build_card("):source.index("def run(")]
    assert "WATCHLIST_EMERGENCE" not in card


# ---------- R2-A: границы полосы появления ----------

@pytest.mark.parametrize("target", sorted(BOUNDARY))
def test_emergence_boundary_produces_the_expected_authoritative_decision(target):
    """Пять точек на границах полосы. Пороги читаются из кода и не подстраиваются."""
    d = decide(state(**BOUNDARY[target]), burst=BOUNDARY_BURST[target], documents=DOCS,
               verification=SEMANTICALLY_VERIFIED)
    assert d.emergence_score == target, (target, d.emergence_score)
    expected = ACCEPT if target >= MIN_EMERGENCE else NOISE
    assert d.decision == expected, (target, d.decision)


@pytest.mark.parametrize("target", sorted(BOUNDARY))
def test_no_contradictory_terminal_combination_at_any_boundary(target):
    """Тройка decision/tier/status согласована на каждой границе и при любом состоянии проверки."""
    for verification in VERIFICATION_STATES:
        d = decide(state(**BOUNDARY[target]), burst=BOUNDARY_BURST[target], documents=DOCS,
                   verification=verification)
        tier, status = query.terminal_state(d.decision)
        if d.decision == ACCEPT:
            assert (tier, status) == ("подтверждённый слабый сигнал", "слабый сигнал")
        else:
            assert status != query.WEAK_SIGNAL_STATUS, (target, verification, d.decision)
        if status == query.WEAK_SIGNAL_STATUS:
            assert d.decision == ACCEPT and d.verification == SEMANTICALLY_VERIFIED


def test_emergence_just_below_the_minimum_is_not_accepted():
    d = decide(state(**BOUNDARY[0.149]), burst=BOUNDARY_BURST[0.149], documents=DOCS,
               verification=SEMANTICALLY_VERIFIED)
    assert d.emergence_score < MIN_EMERGENCE and d.decision != ACCEPT
    assert query.terminal_state(d.decision)[1] == "исключён"


def test_the_old_contradiction_band_is_now_coherent():
    """Полоса [MIN_EMERGENCE, WATCHLIST_EMERGENCE) раньше давала ACCEPT + «под наблюдением»."""
    found = decide(state(**BOUNDARY[0.151]), burst=BOUNDARY_BURST[0.151], documents=DOCS,
                   verification=SEMANTICALLY_VERIFIED)
    assert found.decision == ACCEPT
    assert MIN_EMERGENCE <= found.emergence_score < query.WATCHLIST_EMERGENCE
    tier, status = query.terminal_state(found.decision)
    assert (tier, status) == ("подтверждённый слабый сигнал", "слабый сигнал")


# ---------- R2-B: детерминированная морфология не подтверждает ----------

def assess(phrase):
    return relevance.assess(phrase, {"security", "ai"}, set(),
                            channels={"литература": 3, "репозитории": 2}, lit_docs=3)


REVERSED_NGRAMS = ["attacks poisoning", "detection intrusion", "tuning instruction",
                   "encryption homomorphic", "calibration sensor"]
FUNCTION_WORD_FRAGMENTS = ["of tokenization", "and compression", "based attestation",
                           "for orchestration", "with quantization"]
GRAMMATICAL_GARBAGE = ["poisoning the", "the detection of", "compression and", "attestation based on"]
VALID_NARROW = ["rag poisoning", "grasp synthesis", "deposit tokenization", "wafer bonding"]
VALID_BROAD = ["federated learning", "quantum computing", "homomorphic encryption", "model compression"]


@pytest.mark.parametrize("phrase", REVERSED_NGRAMS + FUNCTION_WORD_FRAGMENTS + GRAMMATICAL_GARBAGE)
def test_malformed_phrases_cannot_be_confirmed_without_semantics(phrase):
    a = assess(phrase)
    d = decide(state(), burst=3.0, documents=DOCS, verification=DETERMINISTIC_ONLY,
               high_certainty=relevance.deterministic_high_certainty(a))
    assert d.decision == LOW_EVIDENCE, (phrase, d.decision)
    assert d.decision_reason_codes == ["semantic_verification_pending"]


@pytest.mark.parametrize("phrase", VALID_NARROW + VALID_BROAD)
def test_valid_technologies_also_wait_for_semantics_rather_than_being_guessed(phrase):
    """Симметрия: правильные названия тоже не подтверждаются морфологией — решает семантика."""
    a = assess(phrase)
    d = decide(state(), burst=3.0, documents=DOCS, verification=DETERMINISTIC_ONLY,
               high_certainty=relevance.deterministic_high_certainty(a))
    assert d.decision == LOW_EVIDENCE


@pytest.mark.parametrize("phrase", VALID_NARROW + VALID_BROAD)
def test_valid_technologies_are_confirmed_once_semantics_succeed(phrase):
    d = decide(state(), burst=3.0, documents=DOCS, verification=SEMANTICALLY_VERIFIED)
    assert d.decision == ACCEPT, phrase


def test_no_phrase_specific_lists_were_added():
    """Ни одна из состязательных фраз не должна появиться в исполняемом коде."""
    import ast

    for module in ("relevance.py", "decision.py", "query.py"):
        tree = ast.parse((ROOT / "wsignals" / module).read_text(encoding="utf-8"))
        if ast.get_docstring(tree):
            tree.body = tree.body[1:]
        code = ast.unparse(tree)
        for phrase in REVERSED_NGRAMS + FUNCTION_WORD_FRAGMENTS + GRAMMATICAL_GARBAGE + VALID_NARROW:
            assert phrase not in code, (module, phrase)


def test_deterministic_heuristic_is_preserved_and_still_reported():
    """Эвристика не удалена: она вычисляется и попадает в объяснение, просто не подтверждает."""
    a = assess("rag poisoning")
    assert relevance.deterministic_high_certainty(a) is True
    d = decide(state(), burst=3.0, documents=DOCS, verification=DETERMINISTIC_ONLY, high_certainty=True)
    assert "морфология указывает" in " ".join(d.decision_reasons)


# ---------- R2-B: отказ нормализатора — рабочее состояние ----------

def test_llm_outage_keeps_evidence_and_vetoes_working():
    d = decide(state(), burst=3.0, documents=DOCS, verification=DETERMINISTIC_ONLY)
    assert d.decision == LOW_EVIDENCE
    assert d.supporting_document_count == 2                      # доказательства материализованы
    assert d.concrete_supporting_families == ["code", "science"]
    assert d.evidence_quality["independent_supports"] == 2


def test_vetoes_still_fire_during_an_outage():
    mature = decide(state(wiki={"wiki_article": 1, "wiki_age": 11}), burst=3.0, documents=DOCS,
                    verification=DETERMINISTIC_ONLY)
    assert mature.decision == MATURE
    hype = decide(state(press=0.9), burst=3.0, documents=DOCS, verification=DETERMINISTIC_ONLY)
    assert hype.decision == HYPE
    noise = decide(state(), burst=3.0, documents=DOCS, verification=DETERMINISTIC_ONLY,
                   noise_reason="название продукта")
    assert noise.decision == NOISE


def test_uncertain_and_deterministic_states_are_distinguishable_in_the_output():
    a = decide(state(), burst=3.0, documents=DOCS, verification=DETERMINISTIC_ONLY)
    b = decide(state(), burst=3.0, documents=DOCS, verification=SEMANTIC_UNCERTAIN)
    assert a.verification != b.verification
    assert "нормализация не выполнялась" in " ".join(a.decision_reasons)
    assert "лексика кандидата неоднозначна" in " ".join(b.decision_reasons)


# ---------- R2-C: закреплённая модель против фактически использованной ----------

def test_pinned_model_is_reported_even_without_credentials(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    status = provenance.semantic_normalizer_status()
    assert status["configured"] is False
    assert status["model"] is None                                # честно: модель не работала
    assert status["pinned_model"] == decompose.RUNTIME_YANDEX_MODEL == "yandexgpt-5-lite"
    assert status["reason"]


def test_actual_model_appears_only_when_configured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "not-a-real-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folderQ")
    status = provenance.semantic_normalizer_status()
    assert status["configured"] is True
    assert status["model"] == "gpt://folderQ/yandexgpt-5-lite"
    assert status["pinned_model"] == "yandexgpt-5-lite"


def test_pinned_model_is_part_of_the_behavioural_config():
    cfg = provenance.behavioral_config()
    assert cfg["semantic_verification"]["pinned_model"] == "yandexgpt-5-lite"


def test_no_secret_value_reaches_metadata_or_config_hash(monkeypatch):
    import json

    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "SECRET-DO-NOT-LEAK")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folderQ")
    meta = json.dumps(provenance.run_metadata("q"), ensure_ascii=False)
    cfg = json.dumps(provenance.behavioral_config(), ensure_ascii=False)
    assert "SECRET-DO-NOT-LEAK" not in meta and "SECRET-DO-NOT-LEAK" not in cfg
    assert provenance.config_hash()               # хеш считается и без утечки значения


def test_run_metadata_carries_the_pinned_model():
    meta = provenance.run_metadata("слабые сигналы в кибербезопасности")
    assert meta["semantic_normalizer"]["pinned_model"] == "yandexgpt-5-lite"
