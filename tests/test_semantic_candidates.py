"""Семантическая нормализация кандидатов и склейка вариантов (REPAIR 2, REPAIR 3).

Правила проверяются как общие, а не как список запрещённых фраз: для каждого правила есть и
отбракованный пример, и контрольный пример той же формы, который проходить обязан.
"""
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import relevance  # noqa: E402

SEEDS = {"cybersecurity", "security", "ai", "fintech", "financial", "payments"}
RICH = {"channels": {"литература": 3, "репозитории": 2}, "lit_docs": 3}


def a(phrase, seeds=SEEDS, exclusions=frozenset()):
    return relevance.assess(phrase, set(seeds), set(exclusions), channels=RICH["channels"], lit_docs=RICH["lit_docs"])


def cand(phrase, burst=1.0, **kw):
    c = {"phrase": phrase, "original_phrase": phrase, "burst": burst,
         "channels": {"литература": 3, "репозитории": 2}, "recent_docs": 3, "prior_docs": 1,
         "examples": [], "repos": []}
    c.update(kw)
    return c


# ---------- схема ----------

def test_assessment_has_the_full_contract():
    out = a("agent identity").to_dict()
    for key in ("original_phrase", "canonical_name", "is_technology", "domain_relevance",
                "is_product_or_brand", "is_generic_phrase", "merge_key", "reject_reason"):
        assert key in out, key
    assert isinstance(out["is_technology"], bool)
    assert 0.0 <= out["domain_relevance"] <= 1.0


# ---------- правило: головное слово рынка или обзора — вето ----------

@pytest.mark.parametrize("phrase", ["robotics market", "payments industry", "quantum landscape",
                                    "security trends", "battery ecosystem"])
def test_discourse_head_is_never_a_technology(phrase):
    """Рынок, отрасль, тренд и обзор не бывают технологией ни при каком определении."""
    r = a(phrase)
    assert r.rejected and r.is_generic_phrase and "рынку или обзору" in r.reject_reason


# ---------- правило: общее техническое головное слово — ШТРАФ, а не вето (REPAIR A) ----------

@pytest.mark.parametrize("phrase", ["key management", "data integration", "spectral analysis",
                                    "protocol development", "memory management", "sensor integration",
                                    "thermal analysis", "firmware development", "identity management",
                                    "payload integration"])
def test_generic_technical_head_with_a_specific_modifier_survives(phrase):
    """Ложные отказы аудита. Общее головное слово не может быть смертельным вето, когда его определяет
    осмысленное техническое понятие."""
    r = a(phrase)
    assert not r.rejected, (phrase, r.reject_reason)
    assert r.is_technology and r.ambiguous          # выживает, но помечен как неоднозначный
    assert r.specificity < 1.0                      # и с пониженной специфичностью


@pytest.mark.parametrize("phrase", ["llm tools", "financial ai", "ai security", "advanced solutions"])
def test_generic_head_without_any_specific_modifier_is_rejected(phrase):
    """Когда содержательного слова нет вообще, отказ остаётся: это действительно общая фраза."""
    r = a(phrase)
    assert r.rejected and r.is_generic_phrase


@pytest.mark.parametrize("phrase", ["agent identity", "stablecoin payments", "quantum sensing",
                                    "confidential computing", "solid state battery"])
def test_concrete_head_survives(phrase):
    r = a(phrase)
    assert not r.rejected and not r.ambiguous and r.specificity == 1.0, phrase


def test_model_class_head_is_a_penalty_not_a_veto():
    """«industrial ai» это рубрика, но «industrial» — содержательное слово, поэтому фраза доходит
    до семантической нормализации со штрафом, а не отвергается лексикой."""
    r = a("industrial ai")
    assert not r.rejected and r.ambiguous and r.specificity < 1.0


def test_model_class_inside_the_phrase_is_fine():
    assert not a("llm watermarking").rejected


# ---------- правило: герундий в начале ----------

@pytest.mark.parametrize("phrase", ["deploying ai agents", "securing payment rails", "evaluating quantum devices"])
def test_leading_gerund_is_an_action_not_a_technology(phrase):
    r = a(phrase)
    assert r.rejected and "действия" in r.reject_reason


@pytest.mark.parametrize("phrase", ["federated learning", "continual pretraining", "quantum computing",
                                    "edge caching", "rag poisoning"])
def test_gerund_nouns_are_not_mistaken_for_actions(phrase):
    assert not a(phrase).rejected, phrase


# ---------- правило: прилагательное в конце ----------

@pytest.mark.parametrize("phrase", ["ai-driven educational", "agent based", "quantum sensitive",
                                    "network defensive"])
def test_trailing_adjective_is_not_a_noun_phrase(phrase):
    r = a(phrase)
    assert r.rejected and "прилагательным" in r.reject_reason


@pytest.mark.parametrize("phrase", ["optical signal", "battery material", "payment terminal", "consensus protocol"])
def test_nouns_that_look_adjectival_are_kept(phrase):
    assert not a(phrase).rejected, phrase


# ---------- правило: продукты и служебная лексика ----------

@pytest.mark.parametrize("phrase", ["claude skills", "openai operator", "nvidia inference"])
def test_unambiguous_brands_are_rejected(phrase):
    """Коины без обычного технического значения — сильная улика продукта."""
    r = a(phrase)
    assert r.rejected and r.is_product_or_brand and not r.is_technology


@pytest.mark.parametrize("phrase,domain", [
    ("meta learning", "ml"), ("swift transactions", "финтех"), ("visa tokenization", "финтех"),
    ("apple silicon", "аппаратура"), ("oracle database", "данные"), ("spark streaming", "данные"),
    ("rust toolchain", "разработка"), ("go runtime", "разработка"),
])
def test_ambiguous_brand_tokens_do_not_veto_ordinary_technical_meaning(phrase, domain):
    """Коллизии брендов. Слово может быть брендом и одновременно обычным техническим словом;
    совпадение по одному токену не повод отвергать фразу."""
    r = a(phrase)
    assert not r.rejected, (phrase, domain, r.reject_reason)
    assert r.ambiguous and "может быть названием продукта" in " ".join(r.ambiguity_flags)


def test_ambiguous_brand_without_any_other_technical_word_is_a_product():
    """Контекст решает: бренд плюс только общая лексика — это продукт."""
    r = a("google solutions")
    assert r.rejected and r.is_product_or_brand


@pytest.mark.parametrize("phrase", ["awesome agents", "starter template", "agent cookbook"])
def test_repo_furniture_is_rejected(phrase):
    r = a(phrase)
    assert r.rejected and "оформление репозитория" in r.reject_reason


@pytest.mark.parametrize("phrase", ["mcp servers", "inference server", "telemetry dashboard"])
def test_soft_plumbing_is_a_penalty_not_a_veto(phrase):
    """Обвязка бывает частью настоящего названия, поэтому это флаг и штраф, а не отказ."""
    r = a(phrase)
    assert not r.rejected and r.ambiguous, (phrase, r.reject_reason)


# ---------- отношение к направлению ----------

def test_domain_relevance_drops_for_isolated_off_domain_phrases():
    near = relevance.domain_relevance("payment rails", SEEDS, {"литература": 3, "репозитории": 1}, 3)
    far = relevance.domain_relevance("soil moisture", SEEDS, {"новости": 1}, 0)
    assert near > far and far < relevance.MIN_DOMAIN_RELEVANCE


def test_weak_specificity_and_weak_relevance_together_reject():
    """Отказ по слабости требует, чтобы слабыми были ОБА признака сразу."""
    r = relevance.assess("cloud api management", SEEDS, set(), channels={"новости": 1}, lit_docs=0)
    assert r.rejected and "технической специфичности" in r.reject_reason


def test_strong_specificity_survives_weak_relevance():
    """Специфичная фраза из одного канала выживает: судить о ней должны доказательства, не лексика."""
    r = relevance.assess("cryogenic interconnect", SEEDS, set(), channels={"новости": 1}, lit_docs=0)
    assert not r.rejected, r.reject_reason


def test_prefilter_is_high_recall_across_domains():
    """Общее правило, а не список исключений: по десяти направлениям законные названия проходят."""
    corpus = ["perovskite tandem", "solid oxide electrolyser", "cryogenic interconnect",
              "millimeter wave backhaul", "digital twin", "zero knowledge rollup",
              "homomorphic inference", "neuromorphic accelerator", "bioprinted scaffold",
              "hydrogen electrolyser", "catalyst screening", "wafer bonding",
              "battery management", "grid integration", "failure analysis", "protocol development",
              "supply chain traceability", "satellite laser link", "quantum repeater",
              "photonic interconnect"]
    rejected = [c for c in corpus if a(c).rejected]
    assert rejected == [], rejected


def test_plan_exclusions_are_applied():
    assert relevance.assess("neural pipeline", SEEDS, {"neural"}, channels=RICH["channels"], lit_docs=3).rejected


# ---------- правила общие, а не список плохих примеров ----------

def test_no_bad_example_is_hardcoded():
    """Плохие примеры из аудита могут быть упомянуты в документации модуля как мотивация,
    но ни один из них не должен встречаться в исполняемом коде: правила обязаны быть общими."""
    import ast

    tree = ast.parse((ROOT / "wsignals" / "relevance.py").read_text(encoding="utf-8"))
    if ast.get_docstring(tree):
        tree.body = tree.body[1:]
    code = ast.unparse(tree)
    for leaked in ("llm tools", "financial ai", "deploying ai agents", "ai-driven educational", "cloud dependency"):
        assert leaked not in code, leaked


# ---------- REPAIR 3: склейка вариантов ----------

def test_plural_and_singular_forms_collapse():
    kept, merged = relevance.collapse([cand("mcp server", burst=2.0), cand("mcp servers", burst=1.0)])
    assert len(kept) == 1 and len(merged) == 1
    assert kept[0]["phrase"] == "mcp server"
    assert kept[0]["merged_from"] == ["mcp servers"]


def test_word_order_variants_collapse():
    kept, merged = relevance.collapse([cand("rag poisoning", burst=3.0), cand("poisoning of rag", burst=1.0)])
    assert len(kept) == 1 and merged[0]["merged_into"] == "rag poisoning"


def test_generic_modifier_variants_collapse():
    kept, _ = relevance.collapse([cand("agent identity", burst=3.0), cand("ai agent identity", burst=1.0)])
    assert len(kept) == 1


def test_different_technologies_do_not_collapse():
    kept, merged = relevance.collapse([cand("rag poisoning"), cand("agent identity"), cand("quantum sensing")])
    assert len(kept) == 3 and merged == []


def test_collapse_preserves_evidence_lineage():
    lead = cand("mcp server", burst=3.0, examples=[{"id": "W1"}], repos=[{"full_name": "a/b"}])
    other = cand("mcp servers", burst=1.0, examples=[{"id": "W2"}], repos=[{"full_name": "c/d"}])
    kept, merged = relevance.collapse([lead, other])
    survivor = kept[0]
    assert {w["id"] for w in survivor["examples"]} == {"W1", "W2"}
    assert {r["full_name"] for r in survivor["repos"]} == {"a/b", "c/d"}
    assert survivor["recent_docs"] == 6                      # свидетельства складываются, а не теряются
    assert merged[0]["noise_reason"].startswith("вариант того же явления")


def test_product_and_technology_do_not_collapse_into_one():
    """«claude skills» это продукт, «agent skills» — технология: разные ключи, склейки быть не должно."""
    kept, merged = relevance.collapse([cand("agent skills", burst=3.0), cand("claude skills", burst=1.0)])
    assert len(kept) == 2 and merged == []


def test_llm_supplied_merge_key_collapses_semantic_aliases(monkeypatch):
    """Семантические синонимы без общих слов склеиваются только через merge_key от модели."""
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")
    monkeypatch.setenv("YANDEX_QUERY_MODEL", "yandexgpt-5-lite")

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            rows = [{"i": 0, "canonical_name": "agent identity", "is_technology": True, "domain_relevance": 0.9,
                     "is_product_or_brand": False, "is_generic_phrase": False, "merge_key": "agent-identity",
                     "reject_reason": None},
                    {"i": 1, "canonical_name": "agent identity", "is_technology": True, "domain_relevance": 0.9,
                     "is_product_or_brand": False, "is_generic_phrase": False, "merge_key": "agent-identity",
                     "reject_reason": None}]
            return {"result": {"alternatives": [{"message": {"text": str(rows).replace("'", '"')
                                                             .replace("True", "true").replace("False", "false")
                                                             .replace("None", "null")}}]}}

    monkeypatch.setattr(requests, "post", lambda *a, **kw: R())
    cands = [cand("agent identity", burst=3.0), cand("machine credentials", burst=1.0)]
    keep, dropped = relevance.canonicalize(cands, domain="кибербезопасность")
    kept, merged = relevance.collapse(keep)
    assert len(kept) == 1 and merged[0]["merged_into"] == "agent identity"


def test_llm_rejection_marks_the_candidate_without_dropping_the_original_phrase(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"alternatives": [{"message": {"text":
                    '[{"i": 0, "canonical_name": "x", "is_technology": false, "domain_relevance": 0.2,'
                    ' "is_product_or_brand": false, "is_generic_phrase": true, "merge_key": null,'
                    ' "reject_reason": null}]'}}]}}

    monkeypatch.setattr(requests, "post", lambda *a, **kw: R())
    keep, dropped = relevance.canonicalize([cand("some phrase")])
    assert keep == [] and dropped[0]["original_phrase"] == "some phrase"
    assert dropped[0]["noise_reason"]


@pytest.mark.parametrize("bad", [
    '[{"i": 0}]',
    '[{"i": 0, "canonical_name": "x", "is_technology": "yes", "domain_relevance": 0.5,'
    ' "is_product_or_brand": false, "is_generic_phrase": false, "merge_key": null, "reject_reason": null}]',
    '[{"i": 0, "canonical_name": "x", "is_technology": true, "domain_relevance": 5,'
    ' "is_product_or_brand": false, "is_generic_phrase": false, "merge_key": null, "reject_reason": null}]',
    "не json вовсе",
])
def test_invalid_llm_rows_are_ignored_and_the_deterministic_result_stands(monkeypatch, bad):
    monkeypatch.setenv("LLM_PROVIDER", "yandexgpt")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"alternatives": [{"message": {"text": bad}}]}}

    monkeypatch.setattr(requests, "post", lambda *a, **kw: R())
    keep, dropped = relevance.canonicalize([cand("agent identity")])
    assert keep and keep[0]["phrase"] == "agent identity" and dropped == []
