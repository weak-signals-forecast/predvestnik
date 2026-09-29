"""Прямое называние против происхождения привязки (ремонт ложноотрицательных).

Измеренный дефект. На владельческих живых артефактах 21e4676 из 114 отклонённых документов 110
отклонены правилом происхождения привязки и лишь 4 — самим фразовым контрактом. У 28 из этих 110
заголовок НЕЗАВИСИМО называет выжившего кандидата и проходит существующее фразовое правило без
каких-либо послаблений. Причина: `match()` спрашивал `authoritative_attachment()` РАНЬШЕ, чем
смотрел на заголовок, а `equivalent()` требует совпадения мультимножества токенов — поэтому
документ, попавший в список кандидата через поглощённый вариант другой ширины, отбрасывался
независимо от того, о чём он сам.

Что здесь проверяется:

  A — прямое называние восстанавливает эти документы;
  B — привязка по-прежнему fail-closed там, где прямого называния НЕТ;
  C — порядок: прямое совпадение самодостаточно, привязка не вето́рует его и сама подтверждения
      не создаёт.

Ни один порог, ни токенизация, ни равнозначность, ни состав авторитетных вариантов здесь не
меняются: восстановление идёт ТОЛЬКО через уже существующее фразовое правило.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import claim_relevance as CR, trust  # noqa: E402
from wsignals.schema import Document  # noqa: E402


def doc(title, url="https://x.test/a", source_type="научная публикация", name="J"):
    return trust.assess(Document(title=title, url=url, source_name=name, source_type=source_type,
                                 published="2026-01-01", language="en")).with_provenance()


def attached(title, via, source="merged", **kw):
    d = doc(title, **kw)
    d.attachment_phrase, d.attachment_source = via, source
    return d


def survivor(phrase, absorbed):
    return {"phrase": phrase, "canonical_label": phrase, "original_phrase": phrase,
            "merged_from": list(absorbed)}


# ---------------------------------------------------------------------------
# A. Восстановление ложноотрицательных (примеры из аудированных живых артефактов)
# ---------------------------------------------------------------------------

RECOVERABLE = [
    ("ai-driven penetration", "AI-Driven Penetration Testing for Enterprise Networks", "penetration testing"),
    ("autonomous cybersecurity", "Autonomous cybersecurity operations at cloud scale", "powered cybersecurity"),
    ("kyc aml", "KYC/AML screening with graph analytics in retail banking", "kyc aml compliance"),
]


@pytest.mark.parametrize("phrase,title,via", RECOVERABLE)
def test_A_direct_title_match_survives_a_non_equivalent_attachment(phrase, title, via):
    cand = survivor(phrase, [via])
    ok, name, reason = CR.match(attached(title, via), cand)
    assert ok is True, reason
    assert name == phrase
    assert CR.ATTACHMENT_NOT_AUTHORITATIVE not in reason


@pytest.mark.parametrize("phrase,title,via", RECOVERABLE)
def test_A_recovery_is_exactly_the_existing_phrase_rule(phrase, title, via):
    """Восстановление не вводит нового правила: тот же вердикт, что у собственной привязки."""
    cand = survivor(phrase, [via])
    merged = CR.match(attached(title, via, "merged"), cand)
    own = CR.match(attached(title, via, "own"), cand)
    assert merged[0] == own[0] is True
    assert merged[1] == own[1]


def test_A_attachment_phrase_itself_is_not_what_establishes_relevance():
    """Релевантность даёт заголовок, а не фраза привязки: убираем называние — и документа нет."""
    cand = survivor("autonomous cybersecurity", ["powered cybersecurity"])
    named = CR.match(attached("Autonomous cybersecurity operations at cloud scale",
                              "powered cybersecurity"), cand)
    unnamed = CR.match(attached("Powered cybersecurity market outlook 2026",
                                "powered cybersecurity"), cand)
    assert named[0] is True
    assert unnamed[0] is False and CR.ATTACHMENT_NOT_AUTHORITATIVE in unnamed[2]


# ---------------------------------------------------------------------------
# B. Привязка остаётся fail-closed
# ---------------------------------------------------------------------------

def test_B_v3_broad_merge_lattice_coupler_stays_rejected():
    """Классический контрпример V3: широкий поглощённый вариант не подтверждает узкого выжившего."""
    cand = survivor("lattice coupler", ["coupler"])
    ok, _, reason = CR.match(attached("A compact coupler for photonic integrated circuits", "coupler"), cand)
    assert ok is False and CR.ATTACHMENT_NOT_AUTHORITATIVE in reason


B_COUNTEREXAMPLES = [
    ("ai security", "Artificial intelligence literacy review for primary school teachers", "ai"),
    ("soil sensing", "Soil moisture retrieval from passive microwave observations", "sensing"),
    ("agent framework", "todo-list: a tiny command line task tracker written in Go", "agent"),
    ("edge caching", "Cutting-edge caching strategies for PostgreSQL", "caching"),
]


@pytest.mark.parametrize("phrase,title,via", B_COUNTEREXAMPLES)
def test_B_known_false_support_probes_stay_rejected(phrase, title, via):
    ok, _, reason = CR.match(attached(title, via), survivor(phrase, [via]))
    assert ok is False
    assert CR.ATTACHMENT_NOT_AUTHORITATIVE in reason or "не называет технологию кандидата" in reason


def test_B_canonical_rename_attack_still_has_no_evidence_authority():
    """Свободное переименование моделью авторитетным вариантом не становится и привязку не чинит."""
    renamed = {"phrase": "гомоморфное шифрование", "original_phrase": "гомоморфное шифрование",
               "canonical_label": "homomorphic encryption", "name_source": "llm",
               "merged_from": ["homomorphic encryption"]}
    d = attached("Homomorphic encryption for privacy preserving analytics", "homomorphic encryption")
    assert CR.authoritative_attachment(d, renamed) is False
    ok, name, _ = CR.match(d, renamed)
    assert name != "homomorphic encryption", "переименование не может быть основанием подтверждения"


def test_B_synthetic_adversarial_absorbed_phrase_without_direct_naming():
    """A поглощает B; документ привязан через B и A напрямую НЕ называет."""
    cand = survivor("federated analytics", ["analytics"])
    ok, _, reason = CR.match(attached("Analytics dashboards for retail inventory planning", "analytics"), cand)
    assert ok is False and CR.ATTACHMENT_NOT_AUTHORITATIVE in reason


def test_B_untitled_document_attached_through_a_broad_variant_stays_rejected():
    cand = survivor("federated analytics", ["analytics"])
    ok, _, reason = CR.match(attached("", "analytics"), cand)
    assert ok is False and CR.ATTACHMENT_NOT_AUTHORITATIVE in reason


# ---------------------------------------------------------------------------
# C. Доказательство порядка
# ---------------------------------------------------------------------------

def test_C_direct_match_is_sufficient_on_its_own():
    cand = survivor("ai-driven penetration", ["penetration testing"])
    for source, via in (("own", "ai-driven penetration"), ("merged", "penetration testing")):
        ok, _, _ = CR.match(attached("AI-Driven Penetration Testing for Enterprise Networks", via, source), cand)
        assert ok is True, f"{source}/{via}"


def test_C_attachment_mismatch_cannot_veto_a_valid_direct_match():
    cand = survivor("autonomous cybersecurity", ["powered cybersecurity"])
    title = "Autonomous cybersecurity operations at cloud scale"
    for via in ("powered cybersecurity", "cybersecurity", "", "совершенно другая фраза"):
        ok, _, reason = CR.match(attached(title, via), cand)
        assert ok is True, f"привязка «{via}» не должна вето́ровать прямое называние: {reason}"


def test_C_attachment_cannot_manufacture_a_match_where_direct_evidence_fails():
    """Даже РАВНОЗНАЧНАЯ привязка не делает подтверждающим документ, который кандидата не называет."""
    cand = survivor("ai agents", ["ai agent"])
    ok, _, reason = CR.match(attached("Weather nowcasting with convolutional networks", "ai agent"), cand)
    assert ok is False
    assert "не называет технологию кандидата" in reason
    # И это не отказ по привязке: привязка здесь как раз авторитетна.
    assert CR.authoritative_attachment(attached("x", "ai agent"), cand) is True


def test_C_attachment_gate_still_runs_when_there_is_no_direct_match():
    """Порядок причин сохранён: без прямого совпадения первым сообщается именно отказ по привязке."""
    cand = survivor("federated analytics", ["analytics"])
    bad = CR.match(attached("Analytics dashboards for retail inventory planning", "analytics"), cand)
    good = CR.match(attached("Analytics dashboards for retail inventory planning",
                             "federated analytics", "own"), cand)
    assert CR.ATTACHMENT_NOT_AUTHORITATIVE in bad[2]
    assert CR.ATTACHMENT_NOT_AUTHORITATIVE not in good[2] and good[0] is False


def test_C_authoritative_attachment_helper_is_unchanged_in_meaning():
    """Сама функция правила привязки не тронута: её вердикты прежние."""
    cand = survivor("ai agents", ["ai agent"])
    assert CR.authoritative_attachment(attached("t", "ai agent", "merged"), cand) is True
    assert CR.authoritative_attachment(attached("t", "agent", "merged"), cand) is False
    assert CR.authoritative_attachment(attached("t", "", "merged"), cand) is False
    assert CR.authoritative_attachment(attached("t", "что угодно", "own"), cand) is True
