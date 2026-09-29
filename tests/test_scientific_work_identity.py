"""Одна научная работа — одно свидетельство (P1-A).

Контракт был и раньше: ОКОНЧАТЕЛЬНОЕ ПРИНЯТИЕ ТРЕБУЕТ НЕЗАВИСИМОГО ПОДТВЕРЖДЕНИЯ. Нарушало его то,
что разные представления одной работы (препринт и журнальная версия) имеют разные DOI, разных
издателей и слегка разные заголовки, и потому считались двумя независимыми свидетельствами.

Здесь проверяются СВОЙСТВА правила, а не список ожидаемых работ. Белых списков журналов,
перечней технологий и данных организатора в правиле нет и быть не может.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import independence  # noqa: E402
from wsignals.evidence_quality import MIN_INDEPENDENT_SUPPORTS, assess_documents, gate  # noqa: E402
from wsignals.schema import Document  # noqa: E402
from wsignals.trust import assess  # noqa: E402

TITLE = "Adaptive lattice calibration for cryogenic photonic interconnects"
SUBTITLED = TITLE + ": a scalable experimental architecture"
EM_DASH = TITLE + " \u2014 a scalable experimental architecture"
EN_DASH = TITLE + " \u2013 a scalable experimental architecture"
RETITLED = "Adaptive lattice calibration for cryogenic photonic interconnects at scale"
OTHER = "Thermal drift compensation in superconducting waveguide couplers"


def doc(title, url, source_type="научная публикация", name="Venue"):
    _d = assess(Document(title=title, url=url, source_name=name, source_type=source_type,
                           published="2026-02-01", language="en")).with_provenance()
    # P1-C. Признак претензионной релевантности читается строго: подтверждает только явное
    # True. Эта фикстура ПО ПОСТРОЕНИЮ представляет документ про рассматриваемого кандидата,
    # поэтому помечается явно. Неразмеченные документы проверяются отдельно тестом,
    # доказывающим, что они не открывают ACCEPT.
    _d.claim_relevant = True
    return _d


def supports(docs):
    return assess_documents(docs, aggregate_families=["science"]).independent_supports


# ---------- A–F: обязательные случаи задачи ----------

def test_a_same_canonical_url_twice_is_one_work():
    d = doc(TITLE, "https://doi.org/10.1038/s41586-026-00001-1")
    assert supports([d, d]) == 1


def test_b_exact_same_normalized_title_preprint_and_journal_is_one_work():
    pre = doc(TITLE, "https://doi.org/10.48550/arXiv.2602.00001", "препринт", "arXiv")
    jrn = doc(TITLE.upper(), "https://doi.org/10.1038/s41586-026-00001-1")
    assert supports([pre, jrn]) == 1


def test_c_realistic_minor_retitle_of_the_same_work_is_one_work():
    pre = doc(TITLE, "https://doi.org/10.48550/arXiv.2602.00001", "препринт", "arXiv")
    jrn = doc(RETITLED, "https://doi.org/10.1016/j.optica.2026.02.003")
    assert supports([pre, jrn]) == 1


def test_d_same_topic_but_genuinely_distinct_titles_stay_independent():
    a = doc(TITLE, "https://doi.org/10.1038/s41586-026-00001-1")
    b = doc(OTHER, "https://doi.org/10.1016/j.optica.2026.02.003")
    assert supports([a, b]) == 2


def test_e_same_publisher_genuinely_distinct_papers_stay_independent():
    a = doc(TITLE, "https://doi.org/10.1145/3716628")
    b = doc(OTHER, "https://doi.org/10.1145/3716999")
    assert supports([a, b]) == 2


def test_f_different_publishers_same_underlying_work_is_one_work():
    a = doc(TITLE, "https://doi.org/10.1145/3716628", name="ACM")
    b = doc(TITLE, "https://doi.org/10.1109/TQE.2026.123456", name="IEEE")
    assert supports([a, b]) == 1


# ---------- решающее свойство ----------

def test_one_underlying_work_alone_cannot_satisfy_the_corroboration_gate():
    """Сколько бы версий одной работы ни нашлось, шлюз независимости они не закрывают."""
    versions = [
        doc(TITLE, "https://doi.org/10.48550/arXiv.2602.00001", "препринт", "arXiv"),
        doc(RETITLED, "https://doi.org/10.1038/s41586-026-00001-1", name="Nature"),
        doc(TITLE, "https://doi.org/10.2139/ssrn.4600001", "препринт", "SSRN"),
        doc(TITLE, "https://doi.org/10.1109/TQE.2026.123456", name="IEEE"),
    ]
    q = assess_documents(versions, aggregate_families=["science"])
    assert q.independent_supports == 1 < MIN_INDEPENDENT_SUPPORTS
    assert q.same_work_merges == 3
    passed, code, _ = gate(q)
    assert passed is False and code == "no_independent_corroboration"


def test_two_distinct_works_still_close_the_gate():
    q = assess_documents([doc(TITLE, "https://doi.org/10.1038/a"),
                          doc(OTHER, "https://doi.org/10.1016/b")], aggregate_families=["science"])
    assert q.independent_supports >= MIN_INDEPENDENT_SUPPORTS
    assert gate(q)[0] is True


# ---------- свойства правила ----------

def test_registrant_alone_does_not_imply_independence():
    a = doc(TITLE, "https://doi.org/10.1038/a")
    b = doc(TITLE, "https://doi.org/10.9999/b")
    assert independence.origin(a) != independence.origin(b)
    assert independence.same_scientific_work(a, b)


def test_publication_stage_alone_does_not_imply_independence():
    pre = doc(TITLE, "https://arxiv.test/abs/2602.00001", "препринт", "arXiv")
    jrn = doc(TITLE, "https://doi.org/10.1038/a")
    assert independence.same_scientific_work(pre, jrn)


def test_shared_generic_topic_words_do_not_collapse_distinct_works():
    """Общая тема — не повод объявить две работы одной."""
    a = doc("Machine learning for photonic device design", "https://doi.org/10.1038/a")
    b = doc("Machine learning for photonic device fabrication yield prediction",
            "https://doi.org/10.1016/b")
    assert not independence.same_scientific_work(a, b)
    assert supports([a, b]) == 2


def test_short_titles_are_never_collapsed_by_similarity_alone():
    a = doc("Photonic interconnects", "https://doi.org/10.1038/a")
    b = doc("Photonic interconnects for datacentres", "https://doi.org/10.1016/b")
    assert not independence.same_scientific_work(a, b)


def test_rule_is_deterministic_and_order_independent():
    a = doc(TITLE, "https://doi.org/10.48550/arXiv.2602.00001", "препринт", "arXiv")
    b = doc(RETITLED, "https://doi.org/10.1038/a")
    c = doc(OTHER, "https://doi.org/10.1016/b")
    counts = {supports(order) for order in ([a, b, c], [c, b, a], [b, c, a], [c, a, b])}
    assert counts == {2}


def test_news_syndication_behaviour_is_untouched():
    a = doc("Acme ships lattice calibration", "https://www.prnewswire.com/a", "пресс-релиз", "PRN")
    b = doc("Acme ships lattice calibration", "https://finance.yahoo.test/b", "новости", "Yahoo")
    assert independence.independence_key(a) == independence.independence_key(b)


def test_code_repositories_are_not_merged_by_title_similarity():
    """Правило работы применяется только к научным семействам."""
    a = doc("someorg/lattice-calibration: toolkit", "https://github.com/someorg/lattice-calibration",
            "репозиторий", "GitHub")
    b = doc("otherorg/lattice-calibration: toolkit", "https://github.com/otherorg/lattice-calibration",
            "репозиторий", "GitHub")
    assert len(independence.independent_documents([a, b])) == 2


@pytest.mark.parametrize("module", ["independence.py", "evidence_quality.py"])
def test_no_journal_whitelist_or_expected_technology_in_the_rule(module):
    """Проверяется ИСПОЛНЯЕМЫЙ код: в пояснениях издатели упоминаются как примеры, и это нормально."""
    import ast

    source = (ROOT / "wsignals" / module).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    code = ast.unparse(tree).lower()
    for leaked in ("nature", "elsevier", "springer", "ieee", "acm", "whitelist", "scopus",
                   "impact_factor", "rag poisoning", "post-quantum", "stablecoin"):
        assert leaked not in code, (module, leaked)


# ---------- E–G: версия с подзаголовком (микроремонт) ----------
#
# Журнальная версия часто добавляет к заголовку препринта подзаголовок. Длина заголовка при этом
# меняется слишком сильно для правила «сопоставимой длины», но работа остаётся той же самой.

@pytest.mark.parametrize("separator,long_title", [
    ("двоеточие", SUBTITLED),
    ("длинное тире", EM_DASH),
    ("среднее тире", EN_DASH),
])
def test_e_f_g_subtitle_version_of_the_same_work_is_one_work(separator, long_title):
    pre = doc(TITLE, "https://doi.org/10.48550/arXiv.2602.00001", "препринт", "arXiv")
    jrn = doc(long_title, "https://doi.org/10.1038/s41586-026-00001-1")
    assert independence.same_scientific_work(pre, jrn), separator
    assert supports([pre, jrn]) == 1, separator


def test_h_shared_generic_prefix_with_different_subtitles_stays_independent():
    """Одинаковое НАЧАЛО до двоеточия — не одна работа: сравнивается полный заголовок с ядром."""
    a = doc("Adaptive photonic control: quantum sensor calibration", "https://doi.org/10.1038/a")
    b = doc("Adaptive photonic control: integrated lidar arrays", "https://doi.org/10.1016/b")
    assert not independence.same_scientific_work(a, b)
    assert supports([a, b]) == 2


def test_subtitle_rule_never_compares_core_with_core():
    """Оба ядра совпадают, но ни один полный заголовок ядром другого не является."""
    a, b = ("Surface codes: fault tolerant quantum computing",
            "Surface codes: distributed quantum networks")
    assert independence.title_core(a) == independence.title_core(b)
    assert not independence.same_scientific_work(doc(a, "https://doi.org/10.1038/a"),
                                                 doc(b, "https://doi.org/10.1016/b"))


def test_subtitle_core_below_the_token_floor_never_merges():
    """Родовое ядро из двух-трёх слов — это совпадение темы, а не работы."""
    a = doc("Deep learning", "https://doi.org/10.1038/a")
    b = doc("Deep learning: a survey of architectures", "https://doi.org/10.1016/b")
    assert not independence.same_scientific_work(a, b)


def test_hyphen_is_not_a_subtitle_separator():
    """Дефис живёт внутри терминов и резать по нему нельзя."""
    assert independence.title_core("Low-noise amplifier design for radio astronomy") == frozenset()
    a = doc("Low-noise amplifier design", "https://doi.org/10.1038/a")
    b = doc("Low-noise amplifier design for cryogenic radio astronomy receivers",
            "https://doi.org/10.1016/b")
    assert not independence.same_scientific_work(a, b)


def test_year_range_en_dash_does_not_create_a_spurious_core():
    """Средним тире записывают и диапазоны лет: ядро из двух слов ниже порога и не сливает."""
    a = doc("Photonic trends", "https://doi.org/10.1038/a")
    b = doc("Photonic trends 2019\u20132021 in datacentre interconnects", "https://doi.org/10.1016/b")
    assert not independence.same_scientific_work(a, b)


# ---------- решающее сквозное свойство микроремонта ----------

def test_one_work_with_subtitle_retitle_still_cannot_close_the_gate():
    """Препринт + журнальная версия с подзаголовком + разные издатели = ОДНО свидетельство."""
    versions = [
        doc(TITLE, "https://doi.org/10.48550/arXiv.2602.00001", "препринт", "arXiv"),
        doc(SUBTITLED, "https://doi.org/10.1038/s41586-026-00001-1", name="Nature"),
        doc(EM_DASH, "https://doi.org/10.1109/TQE.2026.123456", name="IEEE"),
        doc(RETITLED, "https://doi.org/10.1016/j.optica.2026.02.003", name="Optica"),
    ]
    q = assess_documents(versions, aggregate_families=["science"])
    assert q.independent_supports == 1 < MIN_INDEPENDENT_SUPPORTS
    assert q.same_work_merges == 3
    passed, code, _ = gate(q)
    assert passed is False and code == "no_independent_corroboration"


def test_subtitle_rule_is_order_independent():
    a = doc(TITLE, "https://doi.org/10.48550/arXiv.2602.00001", "препринт", "arXiv")
    b = doc(SUBTITLED, "https://doi.org/10.1038/a")
    c = doc(OTHER, "https://doi.org/10.1016/b")
    assert {supports(order) for order in ([a, b, c], [c, b, a], [b, c, a], [c, a, b])} == {2}


def test_work_identity_is_transitive_across_every_document_order():
    """Регрессия: тождество работы не должно зависеть от того, какой документ пришёл первым.

    Препринт, версия с подзаголовком и версия с иным переименованием связаны через препринт.
    Сравнение только с представителем группы теряло эту связь при смене представителя."""
    import itertools

    versions = [
        doc(TITLE, "https://doi.org/10.48550/arXiv.2602.00001", "препринт", "arXiv"),
        doc(SUBTITLED, "https://doi.org/10.1038/s41586-026-00001-1", name="Nature"),
        doc(EM_DASH, "https://doi.org/10.1109/TQE.2026.123456", name="IEEE"),
        doc(RETITLED, "https://doi.org/10.1016/j.optica.2026.02.003", name="Optica"),
    ]
    assert {supports(list(order)) for order in itertools.permutations(versions)} == {1}


def test_transitivity_does_not_leak_between_genuinely_distinct_works():
    """Контроль: транзитивность не должна склеивать разные работы через общего соседа."""
    import itertools

    docs = [
        doc(TITLE, "https://doi.org/10.48550/arXiv.2602.00001", "препринт", "arXiv"),
        doc(SUBTITLED, "https://doi.org/10.1038/a", name="Nature"),
        doc(OTHER, "https://doi.org/10.1016/b", name="Optica"),
    ]
    assert {supports(list(order)) for order in itertools.permutations(docs)} == {2}
