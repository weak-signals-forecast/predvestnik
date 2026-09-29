"""Внутренний состязательный бенчмарк качества выдачи (R2-I).

    python scripts/benchmark_quality.py            # сводка
    python scripts/benchmark_quality.py --json out.json

Это ИНЖЕНЕРНЫЙ бенчмарк, а НЕ эталон организаторов. Метки здесь расставлены нами и только там, где
ответ объективен: бренд — это бренд, «рынок» — это рубрика, бессмысленная именная группа — не
технология. Спорные случаи помечены `expert_review` и в точность не засчитываются: их размечает
эксперт (Маргарита), автоматической метки у них нет.

Бенчмарк герметичен: сети нет, все наблюдения источников и документы задаются фикстурами. Проверяется
именно решающий слой — доверие, шлюз качества, независимость, семантическая проверка, хайп, разнообразие.

Покрыты шесть областей организаторов (Edge, Защита ИИ, Индустриальный ИИ, Инфраструктура ИИ, Роботы,
Финтех) и состязательные классы: не по теме, бренды и продукты, широкие зрелые категории, пресс-релизы
и хайп, родственные семьи и дубли, узкие зарождающиеся технологии, частичный отказ провайдеров,
недоступная языковая модель.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsignals import query as query_mod          # noqa: E402
from wsignals import relevance, trust            # noqa: E402
from wsignals.decision import ACCEPT, HYPE, LOW_EVIDENCE, MATURE, NOISE, decide  # noqa: E402
from wsignals.evidence import ERROR, NO_RESULTS, OK, EvidenceState, Observation  # noqa: E402
from wsignals.schema import Document             # noqa: E402

# ---------- профили источников ----------

EMERGING_SCIENCE = {"oa_total": 45, "oa_recent": 30, "oa_prior": 4, "oa_growth": 1.9, "oa_age": 2,
                    "oa_fields": 2, "oa_last_share": 0.45, "oa_preprint_share": 0.35, "oa_peak_ratio": 1.3}
MATURE_SCIENCE = {"oa_total": 6000, "oa_recent": 900, "oa_prior": 800, "oa_growth": 0.1, "oa_age": 12,
                  "oa_fields": 7, "oa_last_share": 0.08, "oa_preprint_share": 0.05, "oa_peak_ratio": 0.9}
THIN_SCIENCE = {"oa_total": 2, "oa_recent": 2, "oa_prior": 0, "oa_growth": 0.4, "oa_age": 1,
                "oa_fields": 1, "oa_last_share": 0.9, "oa_preprint_share": 0.5, "oa_peak_ratio": 1.1}
FRESH_CODE = {"gh_total": 28, "gh_12m": 22, "gh_new_share": 0.78, "gh_max_stars_log": 4.2}
NO_CODE = {"gh_total": 0, "gh_12m": 0, "gh_new_share": 0.0, "gh_max_stars_log": 0.0}
CLEAN_NEWS = {"news_1y": 7, "news_30d": 3, "news_sources": 5, "news_press_share": 0.1, "news_30d_share": 0.43}
PR_NEWS = {"news_1y": 40, "news_30d": 22, "news_sources": 3, "news_press_share": 0.85, "news_30d_share": 0.55}
HN = {"hn_total": 5, "hn_12m": 4, "hn_prior24m": 1, "hn_growth": 1.0}
NO_WIKI = {"wiki_article": 0, "wiki_age": 0.0, "wiki_mentions_log": 0.6, "wiki_views_log": 0.0}
OLD_WIKI = {"wiki_article": 1, "wiki_age": 11.0, "wiki_mentions_log": 8.0, "wiki_views_log": 12.0}


def doc(title, url, source_type, name=""):
    d = trust.assess(Document(title=title, url=url, source_name=name, source_type=source_type,
                              published="2026-03-01", language="en")).with_provenance()
    # P1-C. Признак претензионной релевантности читается строго: подтверждает только явное True.
    # Этот бенчмарк проверяет ИНВАРИАНТЫ СЛОЯ РЕШЕНИЯ — разряды доверия, семейства, вето, — и его
    # синтетические документы по построению представляют свидетельство про рассматриваемого
    # кандидата. Само правило претензионной релевантности проверяется отдельным модулем тестов.
    d.claim_relevant = True
    return d


def paper(n=1, publisher="ACM", prefix="10.1145"):
    return doc(f"Study {n} of the method", f"https://doi.org/{prefix}/{n}", "научная публикация", publisher)


def repo(owner="acme", n=1):
    return doc(f"{owner}/project-{n}", f"https://github.com/{owner}/project-{n}", "репозиторий", "GitHub")


def press(n=1):
    return doc("Acme announces the platform", f"https://www.prnewswire.com/{n}", "пресс-релиз", "PR Newswire")


def unknown_outlet(n=1):
    return doc(f"Coverage {n}", f"https://unlisted-outlet-{n}.test/a", "новости", f"Outlet {n}")


def community(n=1):
    return doc(f"Discussion {n}", f"https://news.ycombinator.com/item?id={n}", "сообщество", "Hacker News")


def trade_media(n=1):
    return doc(f"Industry note {n}", f"https://habr.com/ru/articles/{n}", "новости", "Хабр")


def state(science=EMERGING_SCIENCE, code=FRESH_CODE, news=CLEAN_NEWS, hn=HN, wiki=NO_WIKI,
          failed=()) -> EvidenceState:
    s = EvidenceState()
    for name, values in (("openalex", science), ("github", code), ("news", news), ("hn", hn), ("wiki", wiki)):
        if name in failed:
            s.add(Observation(name, ERROR, error_type="timeout"))
            continue
        if values is None:
            continue
        positive = any(v for v in values.values())
        s.add(Observation(name, OK if positive else NO_RESULTS, values))
    return s


# ---------- случаи бенчмарка ----------

AREAS = ("Edge", "Защита ИИ", "Индустриальный ИИ", "Инфраструктура ИИ", "Роботы", "Финтех")

# label: True — обязан быть подтверждён, False — обязан НЕ быть подтверждён, None — на суд эксперта.
@dataclass
class Case:
    phrase: str
    area: str
    klass: str
    label: bool | None
    documents: list = field(default_factory=list)
    evidence: EvidenceState = field(default_factory=state)
    seeds: frozenset = frozenset()
    burst: float = 3.0
    note: str = ""


def emerging(phrase, area, seeds, docs=None, label=None):
    return Case(phrase, area, "узкая зарождающаяся", label,
                documents=docs if docs is not None else [paper(1), repo("acme")],
                evidence=state(), seeds=frozenset(seeds))


def cases() -> list[Case]:
    out: list[Case] = []

    # 1. Узкие зарождающиеся технологии по шести областям организаторов.
    #    Метка True только там, где фраза объективно является названием технологии.
    narrow = {
        "Edge": ["on-device quantization", "edge caching"],
        "Защита ИИ": ["rag poisoning", "prompt isolation"],
        "Индустриальный ИИ": ["weld seam inspection", "vibration diagnostics"],
        "Инфраструктура ИИ": ["token compression", "inference scheduling"],
        "Роботы": ["tactile sensing", "grasp synthesis"],
        "Финтех": ["deposit tokenization", "settlement netting"],
    }
    for area, phrases in narrow.items():
        for ph in phrases:
            out.append(emerging(ph, area, {area.lower()}, label=True))

    # 2. Не по теме: лексически чистые именные группы без технологического содержания.
    for ph in ("short drama", "local first", "morning routine", "office layout"):
        out.append(Case(ph, "—", "не по теме", False, documents=[paper(2), repo("beta")],
                        evidence=state(), note="чистая лексика, но не технология"))

    # 3. Бренды и продукты.
    for ph in ("claude skills", "openai operator", "chatgpt plugins", "google solutions"):
        out.append(Case(ph, "—", "бренд или продукт", False, documents=[paper(3), repo("gamma")],
                        evidence=state()))

    # 4. Широкие зрелые категории.
    for ph, area in (("machine learning", "Инфраструктура ИИ"), ("public key infrastructure", "Защита ИИ"),
                     ("industrial robotics", "Роботы"), ("mobile payments", "Финтех")):
        out.append(Case(ph, area, "широкая зрелая", False, documents=[paper(4), repo("delta")],
                        evidence=state(science=MATURE_SCIENCE, wiki=OLD_WIKI)))

    # 5. Пресс-релизы и хайп.
    out.append(Case("platform tokenization", "Финтех", "пресс-релизы и хайп", False,
                    documents=[paper(5), repo("eps")] + [press(i) for i in range(1, 6)],
                    evidence=state(news=PR_NEWS)))
    out.append(Case("synergy orchestration", "Инфраструктура ИИ", "пресс-релизы и хайп", False,
                    documents=[press(6), press(7), unknown_outlet(1)],
                    evidence=state(science=THIN_SCIENCE, code=NO_CODE, news=PR_NEWS)))
    out.append(Case("quantum blockchain", "Финтех", "пресс-релизы и хайп", False,
                    documents=[unknown_outlet(2), unknown_outlet(3), community(1)],
                    evidence=state(science=THIN_SCIENCE, code=NO_CODE, news=PR_NEWS)))

    # 6. Родственные семьи и дубли: одно головное слово, разные технологии.
    for ph in ("rag security", "llm security", "mcp security", "agent security"):
        out.append(Case(ph, "Защита ИИ", "родственная семья", None,
                        documents=[paper(6), repo("zeta")], evidence=state(),
                        note="родственники не склеиваются, но ТОП ими не заполняется"))

    # 7. Частичный отказ провайдеров.
    out.append(Case("federated inference", "Инфраструктура ИИ", "частичный отказ провайдеров", None,
                    documents=[paper(7), repo("eta")], evidence=state(failed=("news", "hn"))))
    out.append(Case("secure enclave attestation", "Защита ИИ", "частичный отказ провайдеров", None,
                    documents=[paper(8), repo("theta")], evidence=state(failed=("wiki",))))

    # 8. Доказательства только класса D или неизвестного качества.
    out.append(Case("payment fabric", "Финтех", "только D или неизвестное качество", False,
                    documents=[community(2), community(3)], evidence=state()))
    out.append(Case("robot fleet telemetry", "Роботы", "только D или неизвестное качество", False,
                    documents=[unknown_outlet(4), unknown_outlet(5)], evidence=state()))
    out.append(Case("edge model rollout", "Edge", "только D или неизвестное качество", False,
                    documents=[press(8), community(4), unknown_outlet(6)], evidence=state()))

    # 9. Зависимые свидетельства: перепечатка и один владелец.
    out.append(Case("industrial anomaly detection", "Индустриальный ИИ", "зависимые свидетельства", False,
                    documents=[press(9), doc("Acme announces the platform", "https://mirror.test/a",
                                             "новости", "Mirror")],
                    evidence=state()))
    out.append(Case("grasp planning", "Роботы", "зависимые свидетельства", False,
                    documents=[repo("omega", 1), repo("omega", 2)], evidence=state()))
    out.append(Case("sensor fusion calibration", "Роботы", "зависимые свидетельства", False,
                    documents=[paper(10), paper(11)], evidence=state(),
                    note="две работы одного издателя: независимости нет"))

    # 10. Независимое подтверждение разных типов.
    out.append(Case("photonic interconnect", "Инфраструктура ИИ", "независимое подтверждение", True,
                    documents=[paper(12, "Elsevier", "10.1016"), repo("iota"), trade_media(1)],
                    evidence=state()))
    out.append(Case("post-quantum migration", "Защита ИИ", "независимое подтверждение", True,
                    documents=[doc("Profile", "https://www.iso.org/standard/9", "реестр", "ISO"), repo("kappa")],
                    evidence=state()))
    return out


# ---------- прогон ----------

# Три режима нормализатора. Модель здесь не вызывается: её поведение ЗАДАНО явно, иначе бенчмарк
# мерил бы не систему, а качество конкретной модели.
#
#   unavailable  нормализатора нет: работает детерминированный путь. Это состояние аудита.
#   competent    нормализатор верно отвечает is_technology=false на фразы без технологического
#                содержания (не по теме, бренды) и подтверждает остальные. Такое поведение ожидается
#                от нормализатора; это ВЕРХНЯЯ ГРАНИЦА, а не измерение живой модели.
#   naive        нормализатор подтверждает всё, что прошло лексику. Нижняя граница: показывает,
#                насколько выдача зависит от качества нормализатора.
MODE_UNAVAILABLE, MODE_COMPETENT, MODE_NAIVE = "llm_unavailable", "llm_available_competent", "llm_available_naive"
# Классы, на которых компетентный нормализатор обязан сказать «это не технология».
NON_TECHNOLOGY_CLASSES = {"не по теме", "бренд или продукт"}


def run_case(c: Case, *, mode: str) -> dict:
    a = relevance.assess(c.phrase, set(c.seeds), set(),
                         channels={"литература": 3, "репозитории": 2}, lit_docs=3)
    high_certainty = relevance.deterministic_high_certainty(a)
    if mode == MODE_UNAVAILABLE:
        verification = query_mod.verification_status(a.to_dict())
    elif mode == MODE_COMPETENT:
        verification = ("SEMANTIC_UNCERTAIN" if (a.rejected or c.klass in NON_TECHNOLOGY_CLASSES)
                        else "SEMANTICALLY_VERIFIED")
    else:
        verification = "SEMANTIC_UNCERTAIN" if a.rejected else "SEMANTICALLY_VERIFIED"
    v = decide(c.evidence, burst=c.burst, noise_reason=a.reject_reason, documents=c.documents,
               verification=verification, high_certainty=high_certainty)
    return {"phrase": c.phrase, "area": c.area, "class": c.klass, "label": c.label,
            "decision": v.decision, "codes": v.decision_reason_codes,
            "verification": v.verification, "high_certainty": high_certainty,
            "supporting": v.supporting_document_count,
            "concrete_families": v.concrete_supporting_families,
            "aggregate_only": v.evidence_quality.get("aggregate_only_families", []),
            "claims": v.evidence_quality.get("claims", []),
            "note": c.note}


def metrics(rows: list[dict], all_cases: list[Case]) -> dict:
    confirmed = [r for r in rows if r["decision"] == ACCEPT]
    watchlist = [r for r in rows if r["decision"] == LOW_EVIDENCE]
    labelled = [r for r in confirmed if r["label"] is not None]
    true_pos = [r for r in labelled if r["label"] is True]
    noise_confirmed = [r for r in confirmed if r["label"] is False]

    # Подтверждён только по классу D или по неизвестному качеству — таких быть не должно.
    d_only = [r for r in confirmed if not r["concrete_families"]]
    unknown_trust = [r for r in confirmed if r["supporting"] == 0]
    # Утверждение без документа: семейство названо подтверждением, а документа нет.
    claim_without_doc = [r for r in confirmed
                         for cl in r["claims"]
                         if not cl["aggregate_only"] and not cl["documents"]]
    # Полезность списка наблюдения: доля кандидатов с реальным доказательством на руках.
    useful_watch = [r for r in watchlist if r["supporting"] >= 1]

    phrases_confirmed = [r["phrase"] for r in confirmed]
    return {
        "cases_total": len(rows),
        "confirmed_count": len(confirmed),
        "watchlist_count": len(watchlist),
        "rejected_count": len(rows) - len(confirmed) - len(watchlist),
        "confirmed_precision_by_expert_review": (round(len(true_pos) / len(labelled), 3) if labelled else None),
        "confirmed_labelled": len(labelled),
        "confirmed_pending_expert_review": len(confirmed) - len(labelled),
        "watchlist_usefulness": (round(len(useful_watch) / len(watchlist), 3) if watchlist else None),
        "obvious_noise_confirmed_count": len(noise_confirmed),
        "obvious_noise_confirmed": [r["phrase"] for r in noise_confirmed],
        "D_only_accept_count": len(d_only),
        "unknown_trust_accept_count": len(unknown_trust),
        "claim_without_document_count": len(claim_without_doc),
        "duplicate_rate": relevance.duplicate_rate(phrases_confirmed),
        "sibling_concentration": query_mod.sibling_concentration(
            [{"phrase": p, "canonical_label": p} for p in phrases_confirmed]),
        "expert_review_queue": sorted({r["phrase"] for r in rows if r["label"] is None
                                       and r["decision"] in (ACCEPT, LOW_EVIDENCE)}),
        "by_class": {},
    }


def by_class(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for r in rows:
        bucket = out.setdefault(r["class"], {"total": 0, ACCEPT: 0, LOW_EVIDENCE: 0, MATURE: 0, HYPE: 0, NOISE: 0})
        bucket["total"] += 1
        bucket[r["decision"]] = bucket.get(r["decision"], 0) + 1
    return out


def benchmark(mode: str = MODE_UNAVAILABLE) -> dict:
    cs = cases()
    rows = [run_case(c, mode=mode) for c in cs]
    m = metrics(rows, cs)
    m["by_class"] = by_class(rows)
    m["mode"] = mode
    m["areas_covered"] = sorted({c.area for c in cs if c.area != "—"})
    m["rows"] = rows
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    report = {m: benchmark(m) for m in (MODE_UNAVAILABLE, MODE_COMPETENT, MODE_NAIVE)}
    for mode, m in report.items():
        print(f"\n===== {mode} =====")
        if mode == MODE_COMPETENT:
            print("  (верхняя граница: поведение нормализатора задано, живая модель не измерялась)")
        if mode == MODE_NAIVE:
            print("  (нижняя граница: нормализатор подтверждает всё, что прошло лексику)")
        for key in ("cases_total", "confirmed_count", "watchlist_count", "rejected_count",
                    "confirmed_precision_by_expert_review", "confirmed_labelled",
                    "confirmed_pending_expert_review", "watchlist_usefulness",
                    "obvious_noise_confirmed_count", "D_only_accept_count",
                    "unknown_trust_accept_count", "claim_without_document_count",
                    "duplicate_rate", "sibling_concentration"):
            print(f"  {key:42s} {m[key]}")
        if m["obvious_noise_confirmed"]:
            print(f"  ! подтверждён очевидный шум: {m['obvious_noise_confirmed']}")
        print(f"  области организаторов: {', '.join(m['areas_covered'])}")
        print("  по классам:")
        for klass, b in sorted(m["by_class"].items()):
            parts = " ".join(f"{k}={v}" for k, v in b.items() if k != "total" and v)
            print(f"    {klass:34s} всего={b['total']:2d}  {parts}")
        print(f"  на разметку эксперту ({len(m['expert_review_queue'])}): "
              f"{', '.join(m['expert_review_queue'][:8])}")
    if a.json:
        Path(a.json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
