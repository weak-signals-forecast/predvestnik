"""Тесты правил, которые не требуют данных и внешних сервисов.

    .venv/bin/python -m pytest -q tests
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest_openalex import is_relevant, rebuild_abstract  # noqa: E402
from forecast import dedupe, is_generic  # noqa: E402

DOMAINS = yaml.safe_load((ROOT / "src" / "domains.yaml").read_text(encoding="utf-8"))


def doc(title, abstract="", topic=None, subfield=None):
    return {"title": title, "abstract": abstract, "topic": topic, "subfield": subfield}


# ---------- фильтр релевантности на ингесте ----------

def test_pipelines_keeps_gas_pipeline_corrosion():
    d = doc("Pipeline corrosion under CO2 conditions", "Internal corrosion of natural gas transmission pipelines is studied.")
    assert is_relevant(d, DOMAINS["pipelines"])


def test_pipelines_drops_water_and_medical():
    assert not is_relevant(doc("Leak detection in water distribution pipelines", "Urban water supply network."), DOMAINS["pipelines"])
    assert not is_relevant(doc("Digital twin of aortic aneurysm", "Blood flow in the artery is simulated as a pipeline."), DOMAINS["pipelines"])


def test_no_abstract_uses_title_and_topic_only():
    d = doc("A review on pipeline corrosion, in-line inspection and corrosion growth models", "", topic="Corrosion Behavior and Inhibition")
    assert is_relevant(d, DOMAINS["pipelines"])


def test_lng_drops_plant_extracts_but_keeps_gas_chemistry():
    assert not is_relevant(doc("Antioxidant activity of methanol extract of leaves", "Phytochemical screening of a plant extract."), DOMAINS["lng_gaschem"])
    assert is_relevant(doc("Methanol synthesis from natural gas via syngas", "Techno-economic study of a methanol production plant."), DOMAINS["lng_gaschem"])


def test_hydrogen_drops_biomass_pyrolysis_without_hydrogen():
    assert not is_relevant(doc("Pyrolysis of food waste for biochar", "Slow pyrolysis of biomass residues."), DOMAINS["hydrogen"])
    assert is_relevant(doc("Turquoise hydrogen by methane pyrolysis", "Hydrogen production without CO2 emissions."), DOMAINS["hydrogen"])


# ---------- сборка аннотации OpenAlex ----------

def test_rebuild_abstract_from_inverted_index():
    inv = {"hydrogen": [1], "Turquoise": [0], "pyrolysis": [3], "by": [2]}
    assert rebuild_abstract(inv) == "Turquoise hydrogen by pyrolysis"
    assert rebuild_abstract(None) == ""


# ---------- общие слова и склейка фраз в прогнозе ----------

def test_generic_single_adjectives_are_dropped():
    for w in ("interpretable", "recurrent", "engineered", "predicting", "scalable", "circularity"):
        assert is_generic(w), w


def test_technical_terms_survive():
    for t in ("digital twin framework", "amine-grafted", "api x52", "sustainable aviation fuel", "tropomi", "uav"):
        assert not is_generic(t), t


def test_generic_phrases_without_a_noun_are_dropped():
    assert is_generic("strategic deployment")
    assert is_generic("spatially explicit")
    assert not is_generic("natural gas networks")


def test_dedupe_merges_nested_and_overlapping_phrases():
    df = pd.DataFrame({
        "term": ["dac", "capture dac", "air capture dac", "digital twin", "air carbon capture", "direct air carbon"],
        "recent_docs": [64, 64, 62, 44, 51, 51],
        "p": [0.9, 0.85, 0.8, 0.7, 0.6, 0.55],
    })
    kept = list(dedupe(df, "p")["term"])
    assert kept == ["dac", "digital twin", "air carbon capture"]
