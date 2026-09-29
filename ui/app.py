"""Веб-интерфейс сервиса слабых сигналов (Streamlit). Вся выдача на русском.

    streamlit run ui/app.py            # API по умолчанию http://localhost:8000, переменная API_URL
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
TRUST_ICON = {"высокий": "🟢 высокий", "средний": "🟡 средний", "пониженный": "🟠 пониженный"}

st.set_page_config(page_title="Слабые сигналы", page_icon="📡", layout="wide")
st.markdown("""<style>
div[data-testid="stMetricValue"] {font-size: 2rem;}
.small {color:#5a6b7a; font-size:0.9rem;}
.pill {display:inline-block; padding:2px 8px; border-radius:10px; background:#eef3f9; margin:2px 4px 2px 0; font-size:0.85rem;}
</style>""", unsafe_allow_html=True)


def api(method: str, path: str, **kw):
    r = requests.request(method, f"{API}{path}", timeout=30, **kw)
    r.raise_for_status()
    return r.json()


DECISION_LABEL = {"ACCEPT": "🟢 принят", "LOW_EVIDENCE": "⚪ мало доказательств", "MATURE": "🔵 зрелая",
                  "HYPE": "🟠 хайп", "NOISE": "⚫ шум"}
VERIFICATION_LABEL = {"SEMANTICALLY_VERIFIED": "проверен семантически",
                      "DETERMINISTIC_ONLY": "только детерминированная проверка",
                      "SEMANTIC_UNCERTAIN": "семантика не определена"}
ADAPTER_LABEL = {"OK": "🟢 ответил", "NO_RESULTS": "⚪ пусто", "ERROR": "🔴 отказ"}


def axes(card: dict) -> str:
    return (f"появление {card.get('emergence_score', 0):.2f} · доказательство {card.get('evidence_confidence', 0):.2f}"
            f" · зрелость {card.get('maturity_risk', 0):.2f} · хайп {card.get('hype_risk', 0):.2f}")


def legacy_line(card: dict) -> str:
    """Формулировка о базовом предикторе. Это НЕ уверенность системы и не основание для включения."""
    p = card.get("legacy_model_probability")
    if p is None:
        return "экспериментальный базовый предиктор: не считался"
    iv = card.get("legacy_model_interval_90")
    tail = " · показателен" if card.get("legacy_model_reliable") else " · вектор признаков собран не полностью"
    return ("экспериментальный базовый предиктор (этап 1, в решении не участвует): "
            f"{p:.0%}" + (f" (90 %: {iv[0]:.0%}–{iv[1]:.0%})" if iv else "") + tail)


st.title("📡 Слабые сигналы")
st.caption("Поиск зарождающихся научно-технологических трендов по открытым источникам")

with st.sidebar:
    st.subheader("Базовый предиктор этапа 1")
    try:
        m = api("GET", "/api/model")
        best = m.get("best")
        if best:
            # Метрики относятся к экспериментальной модели этапа 1 на собственной разметке команды
            # (100 сигналов организаторов и 198 отрицательных примеров, составленных командой).
            # В решении о ТОП-15 эта модель не участвует, поэтому качеством выдачи они не являются.
            st.write(f"**{best}**")
            st.caption(f"F1 {m[best]['f1']:.0%} на собственной разметке команды (кросс-валидация). "
                       "Модель в решении о ТОП-15 не участвует: это не точность выдачи.")
    except Exception:
        st.write("API недоступен")
    st.subheader("Последние запросы")
    try:
        for r in api("GET", "/api/search"):
            if st.button(f"#{r['id']} {r['query'][:40]}", key=f"run{r['id']}"):
                st.session_state["run_id"] = r["id"]
    except Exception:
        pass

col_q, col_b = st.columns([5, 1])
query = col_q.text_input("Направление", placeholder="например: слабые сигналы в кибербезопасности", label_visibility="collapsed")
if col_b.button("Поиск", type="primary", use_container_width=True) and query.strip():
    st.session_state["run_id"] = api("POST", "/api/search", json={"query": query})["id"]
    st.session_state.pop("insight", None)

run_id = st.session_state.get("run_id")
if not run_id:
    st.info("Введите технологическое направление в свободной форме. Система соберёт свежие публикации, выделит "
            "кандидатов, проверит каждого по пяти открытым источникам и отделит слабые сигналы от зрелых технологий и хайпа.")
    st.stop()

run = api("GET", f"/api/search/{run_id}")
if run["status"] == "running":
    with st.status(f"Запрос «{run['query']}» выполняется…", expanded=True):
        for line in run["log"][-8:]:
            st.write(line)
    time.sleep(3)
    st.rerun()
if run["status"] == "error":
    st.error("Запрос завершился ошибкой")
    st.code(run["error"])
    st.stop()

stats = run["stats"] or {}
st.subheader(f"Запрос: {run['query']}")
st.caption("Поисковые фразы: " + ", ".join(run["seeds"] or []))
c1, c2, c3, c4, c5, c6 = st.columns(6)
processed = sum(stats.get(k, 0) for k in ("работ за 2025–2026", "работ за 2019–2022", "новых репозиториев", "заголовков новостей"))
c1.metric("Источников", f"{processed:,}".replace(",", " "), help="Обработано документов: публикации, репозитории, новости")
c2.metric("Кандидатов", stats.get("проверено кандидатов", 0), help="Технологий-кандидатов проверено каскадом")
c3.metric("Подтверждено", stats.get("accepted_count", 0),
          help="Кандидаты со статусом ACCEPT: достаточное доказательство, конкретный документ уровня A/B "
               "и независимое подтверждение")
c4.metric("Наблюдение", stats.get("watchlist_count", 0),
          help="Прошли вето зрелости, хайпа и шума, но доказательства пока не хватает. "
               "Подтверждёнными слабыми сигналами не считаются")
c5.metric("Доступность источников ≥ 75 %", stats.get("с доступностью источников ≥ 75 %", 0),
          help="Подтверждённые кандидаты, у которых ответили источники и доля доступности от 0,75. "
               "Это не доказанность технологии и не уверенность модели")
c6.metric("Отклонено", stats.get("rejected_count", stats.get("исключено", 0)),
          help="LOW_EVIDENCE, MATURE, HYPE, NOISE")

plan = stats.get("план запроса") or {}
if plan:
    st.caption(f"Разбор запроса: {plan.get('source')}"
               + (f", модель {plan.get('model')}" if plan.get("model") else f" ({plan.get('fallback_reason')})")
               + (f" · направление: {plan.get('domain')}" if plan.get("domain") else ""))
took, budget = stats.get("секунд"), stats.get("бюджет, с")
line = []
if took is not None and budget:
    line.append(f"время {took} с из бюджета {budget:.0f} с" + (" · бюджет исчерпан" if stats.get("бюджет исчерпан") else ""))
by = {k: stats.get(k, 0) for k in ("отклонено LOW_EVIDENCE", "отклонено MATURE", "отклонено HYPE", "отклонено NOISE")}
line.append("отклонено: " + ", ".join(f"{k.split()[-1]} {v}" for k, v in by.items()))
fails = stats.get("отказы адаптеров") or {}
line.append("отказы адаптеров: " + (", ".join(f"{k} ×{v}" for k, v in fails.items()) if fails else "нет"))
line.append(f"источников на принятого кандидата: {stats.get('источников на принятого кандидата', 0)}")
st.caption(" · ".join(line))

completeness = stats.get("search_completeness")
if completeness:
    shown = {"COMPLETE": st.info, "INCOMPLETE_SEARCH": st.warning, "PROVIDER_ERROR": st.error}
    if completeness.get("state") != "COMPLETE" or not run["signals"]:
        shown.get(completeness.get("state"), st.warning)(completeness.get("message", ""))
else:
    st.caption("Статус полноты поиска не записан: запуск прежней версии сервиса.")

signals = run["signals"]
watchlist = run.get("watchlist") or []
legacy = run.get("unverified_legacy") or []
insight = st.session_state.get("insight")

def signal_row(s, accent: str):
    with st.container(border=True):
        a, b, c = st.columns([5, 2, 1.4])
        a.markdown(f"**{s['rank']}. {s['technology']}** · {accent}")
        if s.get("technology_original") and s["technology_original"] != s["technology"]:
            a.caption(f"в источниках: {s['technology_original']}")
        if s.get("merged_from"):
            a.caption("варианты той же технологии: " + ", ".join(s["merged_from"]))
        a.caption(axes(s))
        a.markdown("".join(f"<span class='pill'>{p}</span>" for p in (s.get("why_weak_signal") or [])[:3]), unsafe_allow_html=True)
        b.metric("Появление", f"{s.get('emergence_score', 0):.0%}", help="Насколько технология появляется именно сейчас")
        b.caption(f"доказательство {s.get('evidence_confidence', 0):.0%} · "
                  f"документов A/B: {s.get('supporting_document_count', 0)}")
        b.caption(VERIFICATION_LABEL.get(s.get("verification"), s.get("verification", ""))
                  + (" · детерминированный признак технологии" if s.get("high_certainty") else ""))
        b.caption(f"ранжирующий балл {s.get('rank_score', 0):.3f}")
        b.caption(legacy_line(s))
        if c.button("Инсайт →", key=f"ins{s['id']}", use_container_width=True):
            st.session_state["insight"] = s["id"]
            st.rerun()


if insight is None:
    st.markdown("### Подтверждённые слабые сигналы")
    if not signals:
        if not completeness:
            st.info("Подтверждённых сигналов в этой записи нет. Полнота поиска для неё не записана, поэтому "
                    "отличить «ничего не найдено» от отказа источника нельзя.")
    for s in signals:
        signal_row(s, "🟢 подтверждён")
    if watchlist:
        st.markdown("### Список наблюдения: требуют дополнительных доказательств")
        st.caption("Эти кандидаты прошли вето зрелости, хайпа и шума, но доказательства пока недостаточно. "
                   "Подтверждёнными слабыми сигналами они не являются.")
        for s in watchlist:
            signal_row(s, "🟡 требует доказательств")
    if legacy:
        with st.expander(f"Записи без сохранённого статуса подтверждения ({len(legacy)})"):
            st.caption("Сохранены прежней версией сервиса без авторитетного решения. "
                       "Подтверждёнными слабыми сигналами не являются.")
            for s in legacy:
                signal_row(s, "⚪ статус не подтверждён")
    with st.expander(f"Исключённые кандидаты ({len(run['excluded'])}) и причины"):
        ex = pd.DataFrame(run["excluded"])
        if not ex.empty and "decision" in ex:
            ex["decision"] = ex["decision"].map(DECISION_LABEL).fillna(ex["decision"])
        st.dataframe(ex.rename(columns={"technology": "Кандидат", "score": "Ранжирующий балл",
                                        "decision": "Решение", "code": "Код причины",
                                        "reason": "Причина исключения"}),
                     use_container_width=True, hide_index=True)
else:
    s = next(x for x in [*signals, *watchlist, *legacy] if x["id"] == insight)
    if st.button("← К списку"):
        st.session_state.pop("insight")
        st.rerun()
    st.markdown(f"## {s['technology']}")
    if s.get("technology_original") and s["technology_original"] != s["technology"]:
        st.caption(f"Оригинальное название: {s['technology_original']} · {s.get('name_source', '')}")
    st.caption(f"Статус: {s.get('tier', '')}")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Появление", f"{s.get('emergence_score', 0):.0%}", help="Насколько технология появляется именно сейчас")
    d2.metric("Доказательство", f"{s.get('evidence_confidence', 0):.0%}",
              help="Сколько независимых семейств источников подтвердили и все ли адаптеры ответили")
    d3.metric("Зрелость", f"{s.get('maturity_risk', 0):.0%}", help="Риск того, что технология уже зрелая")
    d4.metric("Хайп", f"{s.get('hype_risk', 0):.0%}", help="Риск того, что след объясняется маркетингом")
    st.caption(f"Решение: {DECISION_LABEL.get(s.get('decision'), s.get('decision', ''))} "
               f"({', '.join(s.get('decision_reason_codes', []))}) · " + "; ".join(s.get("decision_reasons", [])))
    st.caption(f"Ранжирующий балл {s.get('rank_score', 0):.3f} — порядок выдачи, не вероятность и не уверенность.")
    st.caption(legacy_line(s))
    st.caption(f"Конкретных подтверждающих документов уровня A/B: {s.get('supporting_document_count', 0)} "
               f"({', '.join(s.get('concrete_supporting_families', [])) or 'нет'})")
    st.caption("Семантическая проверка: "
               + VERIFICATION_LABEL.get(s.get("verification"), s.get("verification", ""))
               + (" · подтверждено детерминированным признаком технологии" if s.get("high_certainty") else ""))
    aggregate_only = (s.get("evidence_quality") or {}).get("aggregate_only_families") or []
    if aggregate_only:
        st.caption("Активность по счётчикам без полученных документов (подтверждением не считается): "
                   + ", ".join(aggregate_only))
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Описание технологии")
        st.write(s["description"])
        langs = ", ".join(s.get("summary_source_languages") or [])
        generated_note = (f"Сгенерировано моделью {s.get('generator')} по найденным источникам"
                          + (f" (язык источников: {langs})" if langs else "") + ". Это пересказ, а не цитата.")
        if s.get("description_generated"):
            st.caption(generated_note)
        else:
            st.caption("Собрано по шаблону из измеренных счётчиков источников.")
        st.markdown("#### Потенциальное преимущество")
        st.write(s["advantage"] or "не найдено в источниках")
        if s.get("advantage_source") == "abstract_quote":
            st.caption("Дословный фрагмент аннотации научной работы на языке оригинала")
        elif s.get("advantage_source") == "generated":
            st.caption(generated_note)
        st.markdown("#### Кейс-пример")
        case = s.get("case")
        if isinstance(case, dict):
            st.markdown(f"[{case['title']}]({case['url']}), {case['source_name']}, {case.get('published') or 'дата не указана'}")
        elif case:
            st.write(case)
            if s.get("case_generated"):
                st.caption(generated_note)
    with right:
        st.markdown("#### Почему это слабый сигнал")
        for p in s["why_weak_signal"]:
            st.write(p)
        # Прогноз взлёта и двойник показываются только по полному вектору признаков
        # (forward_looking_supported). Записи прежних версий такого признака не имеют.
        tk = (s.get("takeoff") or {}) if s.get("forward_looking_supported") else {}
        if tk:
            st.markdown(f"#### Прогноз взлёта к {tk['horizon']}")
            pct = tk.get("percentile")
            st.write(f"**{tk['probability']:.0%}**. Сейчас около {tk['baseline_per_year']} научных работ в год."
                     + (f" Растёт быстрее, чем {pct:.0%} из 298 технологий нашего датасета." if pct is not None else ""))
            st.caption(f"Прогноз записан в реестр с датой. Взлёт засчитывается, если {tk['criterion']}.")
        tw = s.get("historical_twin") if s.get("forward_looking_supported") else None
        if tw:
            st.markdown("#### Исторический двойник")
            st.write(f"Сейчас технология похожа на «{tw['name']}» в {tw['year']} году. Эта технология {tw['fate']}.")
        ev = s["evidence"]
        st.markdown("#### Всплеск в литературе")
        st.write(f"{ev['recent_docs']} работ за 2025–2026 против {ev['prior_docs']} за 2019–2022")
        st.markdown("#### Состояние источников")
        adapters = (ev.get("adapters") or {})
        st.dataframe(pd.DataFrame([{"Источник": k, "Семейство": v.get("family"),
                                    "Статус": ADAPTER_LABEL.get(v.get("status"), v.get("status")),
                                    "Ошибка": v.get("error_type") or "", "Получено": v.get("retrieved_at") or "",
                                    "Из кэша": "да" if v.get("stale") else ""} for k, v in adapters.items()]),
                     use_container_width=True, hide_index=True)
        st.caption(f"Независимые подтверждения: {', '.join(ev.get('corroborating_families') or []) or 'нет'} · "
                   f"опрошено источников: {ev.get('completeness', 0):.0%}")
    claims = (s.get("evidence_quality") or {}).get("claims") or []
    if claims:
        st.markdown("#### Что именно подтверждает каждое утверждение")
        rows = []
        for c in claims:
            rows.append({"Утверждение": c["claim"], "Семейства": ", ".join(c["families"]),
                         "Основание": c["wording"],
                         "Документы": " · ".join(d["source_name"] or d["url"] for d in c["documents"]) or "—"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Источники")
    if s.get("sources_note"):
        st.warning(s["sources_note"])
    src = pd.DataFrame(s["sources"])
    if not src.empty:
        src["trust"] = src["trust"].map(TRUST_ICON).fillna(src["trust"])
        for col in ("canonical_url", "retrieved_at", "source_family", "is_primary", "adapter_status"):
            if col not in src:
                src[col] = None
        src["is_primary"] = src["is_primary"].map({True: "первичный", False: "вторичный"}).fillna("неизвестно")
        st.dataframe(src[["title", "source_name", "published", "source_type", "source_family", "is_primary",
                          "language", "trust", "retrieved_at", "url"]].rename(columns={
            "title": "Наименование", "source_name": "Издание", "published": "Дата", "source_type": "Тип",
            "source_family": "Семейство", "is_primary": "Первичность", "language": "Язык",
            "trust": "Доверие", "retrieved_at": "Получено", "url": "Ссылка"}),
            use_container_width=True, hide_index=True, column_config={"Ссылка": st.column_config.LinkColumn()})
        if s.get("from_cache"):
            st.caption("Часть свидетельств восстановлена из кэша прошлых запросов: дата получения указана в таблице.")
    with st.expander("Все признаки модели"):
        st.json(s["features"])
