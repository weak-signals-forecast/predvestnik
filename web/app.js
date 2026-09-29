// Слабые сигналы: страница выдачи. Данные — data/demo.json (выгрузка реестра), позже — API реестра.
// Контракт карточки описан в registry/export_web.py.
"use strict";

const DATA_URL = "data/demo.json";
const API_URL = null; // когда появится эндпоинт реестра: "/api/registry/search"

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => Number(n).toLocaleString("ru-RU");

let DATA = null;

async function load() {
  const r = await fetch(DATA_URL, { cache: "no-store" });
  if (!r.ok) throw new Error(`не удалось загрузить данные (${r.status})`);
  DATA = await r.json();
  renderExamples();
  $("#foot").innerHTML =
    `Реестр: ${fmt(DATA.registry_size)} технологий · срез корпуса: ${esc(DATA.corpus_snapshot)} · ` +
    `источники: ${esc(DATA.sources.join(", "))}`;
  renderGlobal();
  renderTimeMachine();
  const params = new URLSearchParams(location.search), q = params.get("q");
  if (q) {
    search(q);
    // ?open=N — сразу раскрыть N-ю карточку (ссылка для демонстрации), &more=1 — и её «Подробнее»
    const open = parseInt(params.get("open") || "0", 10);
    if (open > 0) {
      const tryOpen = (k) => {
        const rows = document.querySelectorAll("#results .row");
        if (rows.length < open) { if (k < 40) setTimeout(() => tryOpen(k + 1), 150); return; }
        const r = rows[open - 1];
        r.querySelector(".row-head").click();
        if (params.get("more") === "1") r.querySelectorAll("details.more-all").forEach((d) => { d.open = true; });
        r.scrollIntoView({ block: "start" });
      };
      tryOpen(0);
    }
  }
}

// общая картина модели: средний |вклад| признака (SHAP) и направление влияния
function renderGlobal() {
  const g = DATA.model_global;
  if (!g || !g.length) return;
  $("#model-global").innerHTML =
    globalBlock("Что сильнее всего влияет на силу сигнала — экспертная модель (70 % балла)", g) +
    (DATA.model_global_history && DATA.model_global_history.length
      ? globalBlock("Что предсказывало взлёт в истории — модель 2021 → 2024–2026 (30 % балла)", DATA.model_global_history)
      : "") +
    `<p class="muted" style="font-size:13px;margin:8px 0 0">Длина полосы — средний вклад признака в балл по всем технологиям (SHAP). ` +
    `Зелёный ↑ — чем больше значение, тем выше балл; красный ↓ — тем ниже.</p>`;
}

// машина времени: прогноз на конец 2021 года (модель не видела эти технологии при обучении) и исход к 2024–2026
async function renderTimeMachine() {
  let t;
  try {
    const r = await fetch("data/time_machine.json", { cache: "no-store" });
    if (!r.ok) return;
    t = await r.json();
  } catch { return; }
  const pct = (v) => (v < 0.1 ? `${(v * 100).toFixed(1).replace(".", ",")} %` : `${Math.round(v * 100)} %`);
  const num1 = (v) => String(v).replace(".", ",");
  const market = (m) => [m.yc ? `стартапов YC: ${m.yc}` : "", m.press ? `пресса: ${m.press}` : "",
    m.hn ? `Hacker News: ${m.hn}` : ""].filter(Boolean).join(", ") || "в рынок пока не вышла";
  $("#time-machine").innerHTML =
    `<p>Мы заморозили данные на конец ${t.cutoff} года и спросили модель, какие технологии — слабые сигналы. ` +
    `Прогноз для каждой технологии сделан моделью, которая её не видела при обучении. Исход — из будущего: ${esc(t.rule)}.</p>` +
    `<p><b>Из ${t.top} технологий, которые модель поставила бы наверх, к ${esc(t.future)} взлетели ${pct(t.top_rate)} — ` +
    `против ${pct(t.base_rate)} среди всех ${fmt(t.pool)} технологий ${t.cutoff} года: в ${num1(t.lift)} раза чаще ` +
    `(95 % интервал ${num1(t.lift_ci[0])}–${num1(t.lift_ci[1])}).</b> Промахи показаны тоже.</p>` +
    `<ul class="src">${t.items.map((x) => `<li>${x.takeoff ? "✅" : "▫️"} <b>${esc(x.term)}</b>` +
      `<div class="meta">работ в arXiv: ${fmt(x.docs_then)} в ${t.cutoff - 2}–${t.cutoff} → ${fmt(x.docs_later)} в ${esc(t.future)}; ` +
      `доля выросла в ${num1(x.growth)} раза; после ${t.cutoff}: ${esc(market(x.market_after))}</div></li>`).join("")}</ul>`;
}

function globalBlock(title, g) {
  const max = Math.max(...g.map((x) => x.mean_abs));
  return `<h3>${esc(title)}</h3>` +
    `<div class="contrib">${g.map((x) => {
      const w = Math.round(x.mean_abs / max * 50);
      const cls = x.direction >= 0 ? "p" : "n";
      const hint = x.direction >= 0 ? "чем больше, тем выше балл" : "чем больше, тем ниже балл";
      return `<span title="${hint}">${esc(x.label)} ${x.direction >= 0 ? "↑" : "↓"}</span>` +
        `<span class="cbar"><span class="${cls}" style="width:${w}%"></span></span>`;
    }).join("")}</div>`;
}

function renderExamples() {
  const box = $("#examples");
  box.innerHTML = "";
  const main = DATA.examples.filter((ex) => (ex.label || ex.area) === ex.area);
  const more = DATA.examples.filter((ex) => (ex.label || ex.area) !== ex.area);
  const chip = (ex) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.textContent = ex.label || ex.area;
    b.title = ex.query;
    b.addEventListener("click", () => { $("#q").value = ex.query; search(ex.query); });
    return b;
  };
  main.forEach((ex) => box.appendChild(chip(ex)));
  if (more.length) {
    const t = document.createElement("button");
    t.type = "button"; t.className = "chip chip-more"; t.textContent = `ещё примеры (${more.length})`;
    const wrap = document.createElement("div");
    wrap.className = "more-chips"; wrap.hidden = true;
    more.forEach((ex) => wrap.appendChild(chip(ex)));
    t.addEventListener("click", () => { wrap.hidden = !wrap.hidden; t.textContent = wrap.hidden ? `ещё примеры (${more.length})` : "свернуть"; });
    box.appendChild(t);
    box.appendChild(wrap);
  }
}

// живой поиск: витрина, открытая из контейнера registry, отправляет любой запрос в API
async function liveSearch(q) {
  const status = $("#status"), list = $("#results");
  status.innerHTML = `<div class="note">Ищем по базе слабых сигналов: поиск и выбор YandexGPT занимают до минуты…</div>`;
  let d;
  try {
    const r = await fetch("api/registry/search", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q }) });
    if (!r.ok) throw new Error(String(r.status));
    d = await r.json();
  } catch {
    localSearch(q);
    return;
  }
  const cards = [...d.signals, ...(d.more || [])].map((s) => ({ unverified: !!s.unverified,
    key: s.term_en, name_ru: s.name, name: s.term_en, facts: s.facts || [], ladder: s.ladder || [], series: s.series,
    explain: (s.shap || {}).expert || [], explain_history: (s.shap || {}).history, actors: s.actors, twin: s.twin,
    strength: s.score, why: s.why_signal ? { text: s.why_signal, generated: true } : null,
    description: s.description ? { text: s.description, generated: true } : null,
    advantage: s.advantage ? { text: s.advantage, generated: true } : null,
    case: s.case ? { text: s.case, generated: true } : null,
    passport: s.passport, verification: s.verification, kind: s.kind, explain_rule: s.explain_rule,
    sources: (s.sources || []).map((x) => ({ title: x.title, url: x.url, source_name: x.name, published: x.date,
      source_type: x.type, language: x.language, trust: x.trust })),
  }));
  status.innerHTML = `<p><b>ТОП-${Math.min(TOP, cards.length)}</b> по запросу «${esc(d.query)}»` +
    (d.area ? ` (область: ${esc(d.area)})` : "") + ` — найдены в базе из ${fmt(d.meta.base_size)} слабых сигналов, ` +
    `отобраны ${d.mode.startsWith("выбор") ? "YandexGPT" : "по силе сигнала"} и упорядочены по силе сигнала` +
    (cards.length > TOP ? `; ниже — ещё ${cards.length - TOP} по теме` : "") + `.</p>`;
  renderList(list, cards);
}

// без API (статический стенд): подбираем карточки из выгрузки по словам запроса, без языковой модели
const STOP_WORDS = (("перспективн технолог нов что как какие какой каких для при без или это его они она оно " +
  "на подход появляют появля ранние ранн прямо сейчас будущ тренд тренды сигнал сигналы слаб слабые " +
  "the and for with new technology technologies").split(" "));
const norm = (s) => String(s ?? "").toLowerCase().replace(/ё/g, "е");
const words = (s) => (norm(s).match(/[a-zа-я0-9]+/g) || []).filter((w) => w.length >= 3);
const stem = (w) => (w.length > 6 ? w.slice(0, w.length > 8 ? 6 : 5) : w);
const STOP = new Set(STOP_WORDS.map(stem));
// длинные основы — по префиксу («технологии» → «технол» не равно «техно»), короткие — только точно («при» ≠ «приложение»)
const isStop = (w) => STOP.has(w) || STOP_WORDS.some((s) => s.length >= 5 && w.startsWith(s.slice(0, 5)));

let INDEX = null;
function buildIndex() {
  const byKey = new Map();
  for (const r of Object.values(DATA.results)) {
    for (const c of r.cards) {
      const e = byKey.get(c.key) || { card: c, areas: new Set() };
      if (e.card.unverified && !c.unverified) e.card = c; // проверенная версия карточки важнее
      e.areas.add(r.area);
      byKey.set(c.key, e);
    }
  }
  INDEX = [...byKey.values()].map((e) => {
    const c = e.card, txt = (f) => (c[f] && c[f].text) || "";
    return { card: c, areas: e.areas, title: words(`${c.name} ${c.name_ru}`).map(stem),
      body: words(`${txt("description")} ${txt("why")} ${txt("advantage")} ${txt("case")}`).map(stem) };
  });
}

function localSearch(q) {
  if (!INDEX) buildIndex();
  const status = $("#status"), list = $("#results");
  const terms = [...new Set(words(q).map(stem).filter((w) => !isStop(w)))];
  const areas = new Set(DATA.examples.filter((ex) => ex.match.some((m) => norm(q).includes(m))).map((ex) => ex.area));
  const has = (arr, t) => arr.some((w) => w.startsWith(t) || t.startsWith(w) && w.length >= 4);
  list.innerHTML = "";
  const scored = INDEX.map((e) => {
    let s = 0, hit = 0;
    for (const t of terms) {
      const w = has(e.title, t) ? 3 : has(e.body, t) ? 1 : 0;
      s += w; hit += w > 0;
    }
    if ([...e.areas].some((a) => areas.has(a))) s += 2;
    return { e, s, hit };
  }).filter((x) => x.s > 0)
    .sort((a, b) => b.hit - a.hit || b.s * (0.6 + 0.4 * b.e.card.strength) - a.s * (0.6 + 0.4 * a.e.card.strength));
  const cards = scored.slice(0, 40).map((x) => x.e.card);
  if (!cards.length) {
    status.innerHTML = `<div class="note">По запросу «${esc(q)}» в демонстрационной выборке ничего не нашлось. ` +
      `На этом стенде нет живого поиска по всей базе — попробуйте другие слова или один из примеров выше.</div>`;
    return;
  }
  const partial = terms.length > 1 && scored[0].hit < terms.length;
  status.innerHTML = `<p><b>ТОП-${Math.min(TOP, cards.length)}</b> по запросу «${esc(q)}» — подобраны по словам запроса ` +
    `из ${fmt(INDEX.length)} сигналов демонстрационной выборки и упорядочены по совпадению и силе сигнала` +
    (cards.length > TOP ? `; ниже — ещё ${cards.length - TOP} по теме` : "") + `.</p>` +
    (partial ? `<div class="note">Точного совпадения со всеми словами запроса в выборке нет — показаны ближайшие ` +
      `по части слов.</div>` : "") +
    `<div class="note">Это стенд без живого поиска: YandexGPT здесь запрос не разбирает, а полная база ` +
    `(${fmt(DATA.registry_size)} технологий) доступна в развёрнутом сервисе.</div>`;
  renderList(list, cards);
}

// ТОП-15 — сразу; остальные сигналы по теме — под кнопкой, с продолжением нумерации
const TOP = 15;
function renderList(list, cards) {
  list.innerHTML = "";
  const top = cards.slice(0, TOP), rest = cards.slice(TOP);
  top.forEach((c, i) => list.appendChild(row(c, i + 1)));
  const checked = rest.filter((c) => !c.unverified), unchecked = rest.filter((c) => c.unverified);
  if (!rest.length) return;
  const sep = (text) => { const d = document.createElement("div"); d.className = "more-sep"; d.textContent = text; return d; };
  const btn = document.createElement("button");
  btn.className = "more-btn";
  btn.textContent = `Ещё ${rest.length} сигналов по теме`;
  btn.addEventListener("click", () => {
    btn.remove();
    let n = TOP;
    if (checked.length) {
      list.appendChild(sep(`Ещё по теме — за пределами ТОП-${TOP}, тоже проверены YandexGPT`));
      checked.forEach((c) => list.appendChild(row(c, ++n)));
    }
    if (unchecked.length) {
      list.appendChild(sep("Кандидаты по теме из базы — следующие по порядку, YandexGPT их не проверяла"));
      unchecked.forEach((c) => list.appendChild(row(c, ++n)));
    }
  });
  list.appendChild(btn);
}

function findResult(q) {
  const norm = (s) => s.toLowerCase().replace(/ё/g, "е").trim();
  const exact = DATA.results[q] || Object.values(DATA.results).find((r) => norm(r.query) === norm(q));
  if (exact) return { res: exact, fallback: false };
  // демо без API: подбираем пример по области, упомянутой в запросе
  const hit = DATA.examples.find((ex) => ex.match.some((m) => norm(q).includes(m)));
  return hit ? { res: DATA.results[hit.query], fallback: true } : { res: null, fallback: false };
}

function search(q) {
  document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c.title === q)));
  history.replaceState(null, "", `?q=${encodeURIComponent(q)}`);
  const { res, fallback } = findResult(q);
  const status = $("#status"), list = $("#results");
  list.innerHTML = "";
  if (!res || fallback) {
    liveSearch(q);
    return;
  }
  status.innerHTML =
    `<p><b>ТОП-${Math.min(TOP, res.cards.length)}</b> по запросу «${esc(res.query)}» — отобраны из ${fmt(res.pool_size)} близких по теме ` +
    `и упорядочены по силе сигнала` + (res.cards.length > TOP ? `; ниже — ещё ${res.cards.length - TOP} по теме` : "") + `.</p>`;
  renderList(list, res.cards);
}

function row(c, n) {
  const el = $("#row-tpl").content.firstElementChild.cloneNode(true);
  const head = $(".row-head", el), card = $(".card", el);
  $(".rank", el).textContent = n;
  if (c.unverified) el.classList.add("unverified");
  $(".name", el).textContent = c.name_ru || c.name;
  $(".orig", el).textContent = "";                // английский термин уже в скобках в названии — не повторяем
  const change = whatIsIt(c) || signalEvent(c) || (c.facts || [])[0] || "";
  $(".why", el).textContent = change.length > 150 ? change.slice(0, 147).replace(/\s+\S*$/, "") + "…" : change;
  const wf = whereFound(c);
  $(".stage", el).textContent = wf ? `найден: ${wf}` : "";
  const pct = Math.round(c.strength * 100);
  $(".fill", el).style.width = `${pct}%`;
  $(".val", el).textContent = pct;
  $(".val", el).title = "Балл ранжирования: место среди записей базы, не вероятность";
  $(".meter", el).setAttribute("aria-label", `Балл ранжирования ${pct} из 100`);
  $(".meter", el).title = "Балл ранжирования (0–100): место технологии среди записей базы по смеси двух моделей. " +
    "Это порядок, а не вероятность: 99 — выше 99 % записей базы, а не «99 % уверенности»";
  head.addEventListener("click", () => {
    const open = head.getAttribute("aria-expanded") === "true";
    head.setAttribute("aria-expanded", String(!open));
    if (!open && !card.dataset.ready) { card.innerHTML = cardHtml(c); card.dataset.ready = "1"; }
    card.hidden = open;
  });
  return el;
}

// «Что это» не показываем, если описание почти повторяет название
function sameAsName(c) {
  const d = (c.description && c.description.text) || "";
  const n = (s) => s.toLowerCase().replace(/ё/g, "е").replace(/[^a-zа-я0-9 ]/g, " ").replace(/\s+/g, " ").trim();
  const name = n((c.name_ru || "").replace(/\(.*?\)/g, ""));
  return !d || n(d) === name || n(d).length < name.length + 25;
}

// карточка построена вокруг вопроса пользователя «что появилось и почему стоит обратить внимание»:
// технология (заголовок) → сигнал (датированное событие) → прогноз (предположение); ниже — доказательства и подробности
const LAYER_SHORT = { "Наука": "наука", "Hacker News": "сообщество", "Стартапы YC": "стартапы", "Пресса": "пресса",
  "Проекты ЕС": "проекты ЕС", "Энциклопедия": "энциклопедия" };
function whereFound(c) {
  const on = (c.ladder || []).filter((s) => s.present === true).map((s) => LAYER_SHORT[s.layer] || s.layer.toLowerCase());
  return on.join(" · ");
}
// событие и цитата паспорта — только если в них есть все предметные слова термина (для «address poisoning» — и
// address, и poisoning): иначе работа о соседней теме («poisoning в федеративном обучении») выдавалась бы за сигнал
const FILLER = new Set(["ai", "llm", "llms", "for", "of", "the", "and", "based", "model", "models", "system", "systems", "a", "in", "on"]);
function aboutTerm(c, text) {
  if (c.kind === "составной") return true;          // у составных — отфильтровано при сборке (метод или объект)
  const stem = (w) => w.replace(/(ies|es|s|ing|ed)$/, "");
  const t = (text || "").toLowerCase();
  const words = String(c.name || "").toLowerCase().replace(/\(.*?\)/g, " ").split(/[^a-z0-9]+/)
    .filter((w) => w.length > 2 && !FILLER.has(w)).map(stem);
  return words.length > 0 && words.every((w) => t.includes(w));
}
function passportOn(c) {
  const p = c.passport || {};
  return p.confirmed && aboutTerm(c, `${p.quote || ""} ${p.event || ""}`);
}
function signalEvent(c) {
  // событие — только подтверждённое цитатой и о самом сигнале (у составных — отфильтровано при сборке, это c.case)
  const p = c.passport || {};
  if (c.kind === "составной") return c.case && c.case.text ? c.case.text : "";
  return passportOn(c) && p.event ? p.event : "";
}
function whatIsIt(c) {
  // что за сигнал — описание самой технологии; «что нового» из паспорта описывает конкретную работу-источник
  // и показывается в «Почему сейчас» рядом с ней (иначе работа о соседней теме выдавалась бы за суть сигнала)
  const p = c.passport || {};
  return ((c.description && c.description.text) || p.what_new || "").trim();
}
function participants(c) {
  const p = c.passport || {}, a = c.actors;
  const bits = [];
  if (p.who && !/^(исследовател|исследовани|проекты|учёные|ученые|авторы|разработчики)/i.test(p.who.trim()) && !/препринт|публикац/i.test(p.who))
    bits.push(`<p>${esc(p.who)}</p>`);
  if (a) {
    // в главном блоке — только компании, университеты, лаборатории и госорганизации с ≥ 2 работами по теме
    const top = (a.top_orgs || []).filter((o) => o.works >= 2 && ORG_OK.includes(o.type)).slice(0, 4);
    if (top.length) bits.push(`<ul class="orgs">${top.map((o) => `<li><b>${esc(o.name)}</b>` +
      `<span>${esc(ORG_TYPE[o.type] || o.type || "")}${o.country ? ", " + esc(o.country) : ""} · работ: ${fmt(o.works)}</span></li>`).join("")}</ul>`);
    if ((a.yc || []).length) bits.push(`<p class="small">Стартапы Y Combinator: ${a.yc.slice(0, 4).map((y) =>
      y.url ? `<a href="${esc(y.url)}" target="_blank" rel="noopener">${esc(y.name)}</a>` : esc(y.name)).join(", ")}</p>`);
  }
  if (!bits.length) return "";
  return bits.join("") + `<p class="note-small">Аффилиация авторов работ по теме — это участие в исследованиях, а не доказательство внедрения.</p>`;
}
function evidence(c) {
  const out = [], seen = new Set();
  const add = (title, url, what, date) => {
    const k = (title || "").toLowerCase().slice(0, 60);
    if (!title || seen.has(k)) return;
    seen.add(k);
    out.push(`<li><div>${url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(title)}</a>` : esc(title)}` +
      `${date ? ` <span class="small">· ${esc(date)}</span>` : ""}</div><div class="what">${esc(what)}</div></li>`);
  };
  const p = c.passport || {}, v = c.verification || {};
  const byTitle = (q) => (c.sources || []).find((s) => q && (s.title || "").toLowerCase().includes(q.toLowerCase().slice(0, 40)));
  if (passportOn(c) && p.quote) {
    const s = byTitle(p.quote);
    add(p.quote, p.url || (s && s.url), "Подтверждает событие из блока «Почему сейчас»: цитата найдена в источнике дословно", s && s.published);
  }
  if (v.quote && v.quote_ok && aboutTerm(c, v.quote)) {
    const s = byTitle(v.quote);
    add(v.quote, v.url || (s && s.url), "Подтверждает, что работы именно об этой технологии существуют (проверено на запросе)", s && s.published);
  }
  for (const s of c.sources || []) {
    if (out.length >= 3) break;
    add(s.title, s.url, `Публикация по теме · ${s.source_type || s.source_name || "источник"} · доверие: ${s.trust}`, s.published);
  }
  return out.length ? `<ol class="evid">${out.join("")}</ol>` : `<p class="placeholder">Документов-подтверждений в корпусе не нашлось.</p>`;
}
function cardHtml(c) {
  const p = c.passport || {};
  const what = whatIsIt(c), ev = signalEvent(c), where = whereFound(c);
  // почему сейчас: событие, датированное подтверждение и два факта из данных (рост, число работ)
  const dated = (() => {
    if (!(passportOn(c) && p.quote)) return "";
    const src = (c.sources || []).find((x) => (x.title || "").toLowerCase().includes(p.quote.toLowerCase().slice(0, 40)));
    return src && src.published ? `<p class="small"><b>${esc(src.published)}:</b> «${esc(p.quote)}»</p>` : "";
  })();
  const growthFacts = (c.facts || []).filter((f) => /вырос|работ|запис|проект|стартап|Hacker News|прессе/i.test(f)).slice(0, 2);
  const newTxt = passportOn(c) && p.what_new && p.what_new !== whatIsIt(c) ? `<p><b>Что появилось:</b> ${esc(p.what_new)}</p>` : "";
  const nowTxt = newTxt + (ev ? `<p>${esc(ev)}</p>` : "") + dated +
    (growthFacts.length ? `<ul class="facts small">${growthFacts.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>` : "");
  // прогноз — формулировка «почему это сигнал» от YandexGPT; у составных она совпадает с «почему рано» — не повторяем
  const forecast = c.why && c.why.text && c.why.text !== p.why_early ? c.why.text : "";
  const srcAll = c.sources || [];
  const caseTxt = c.case && c.case.text && c.case.text !== ev ? c.case.text : "";
  return `
    <section class="wide sig">
      <div class="kicker">Технология · ${esc(c.name_ru || c.name)}</div>
      <h3 class="blk">Что за сигнал</h3>
      <p class="big">${esc(what || "Описание появится после подключения YandexGPT.")}</p>
      <div class="tri">
        <div class="panel"><h4>Почему сейчас</h4>${nowTxt}</div>
        <div class="panel"><h4>Почему ранняя стадия</h4>
          ${p.why_early ? `<p>${esc(p.why_early)}</p>` : ""}
          ${where ? `<p class="small"><b>Где найден:</b> ${esc(where)}</p>` : ""}
          ${p.doubt ? `<p class="small"><b>Чего пока не знаем:</b> ${esc(p.doubt)}</p>` : ""}</div>
        <div class="panel"><h4>Зачем бизнесу</h4>
          ${c.advantage && c.advantage.text ? `<p><b>Какую проблему решает.</b> ${esc(c.advantage.text)}</p>` : ""}
          ${forecast ? `<p class="small"><b>Прогноз — предположение:</b> ${esc(forecast)}</p>` : ""}
          <p class="note-small">Предполагаемая польза, а не доказанный результат.</p></div>
      </div>
      ${participants(c) ? `<h3 class="blk">Кто уже пробует — участники исследований и разработок</h3>${participants(c)}` : ""}
      <h3 class="blk">Доказательства</h3>
      ${evidence(c)}
      <p class="gen-note">Источники подтверждают событие. Модель оценивает его значимость. Прогноз остаётся предположением. Формулировки — YandexGPT по источникам этой карточки.</p>
    </section>
    <details class="wide more-all"><summary>Подробнее: данные, графики, оценка модели, похожая динамика в прошлом</summary>
      <div class="more-grid">
        <section>
          <h3>Факты из данных</h3>
          <ul class="facts">${(c.facts || []).map((f) => `<li>${esc(f)}</li>`).join("")}</ul>
          ${p.next_fact ? `<p class="small"><b>Какой факт усилит сигнал:</b> ${esc(p.next_fact)}</p>` : ""}
          ${caseTxt ? `<p class="small"><b>Что исследуют:</b> ${esc(caseTxt)}</p>` : ""}
        </section>
        <section>
          <h3>Где найден сигнал</h3>
          ${ladderView(c.ladder)}
        </section>
        <section class="wide">
          <h3>Динамика по годам</h3>
          ${spark(c.series)}
        </section>
        <section class="wide">
          <h3>Почему такой балл ранжирования</h3>
          ${explainSentence(c)}
          ${c.explain && c.explain.length ? waterfall(c.explain) : (c.explain_rule ? `<p>${esc(c.explain_rule)}</p>` : "")}
          <p class="muted" style="font-size:13px;margin:8px 0 0">Как читать: от средней технологии каждый признак сдвигает балл — зелёный вверх, красный вниз; последняя строка — итог. Вклады посчитаны методом SHAP для модели, обученной на ≈1 000 размеченных технологиях; наведите на признак, чтобы увидеть его значение.</p>
          ${c.explain_history ? `<details class="more"><summary>Что говорит история</summary>
            <p class="muted" style="font-size:13px;margin:6px 0">Вторая модель: обучена на том, какие технологии 2021 года взлетели к 2024–2026.</p>
            ${waterfall(c.explain_history)}</details>` : ""}
        </section>
        ${c.actors ? `<section class="wide"><h3>Участники исследований и разработок — все</h3>${actorsBlock(c.actors)}</section>` : ""}
        ${c.twin ? `<section class="wide"><h3>Похожая динамика в прошлом</h3>${twinBlock(c.twin)}
          <p class="note-small">Сходство кривых не доказывает сходство будущего.</p></section>` : ""}
        <section class="wide">
          <h3>Все источники</h3>
          ${sources(srcAll)}
        </section>
      </div>
    </details>`;
}

// проверка по источникам на запросе: YandexGPT видела заголовки источников, цитата проверена программно
function verificationBlock(v) {
  if (!v) return "";
  const ok = v.sources_confirm && v.quote_ok;
  // что именно подтверждено: одна работа из источников карточки упоминает эту технологию (цитата сверена дословно).
  // Стадию всего направления одна работа не доказывает — поэтому оценку стадии здесь не показываем.
  return `<p class="verif ${ok ? "ok" : "warn"}"><b>${ok ? "✓ Подтверждено источником" : "Источники подтверждают частично"}:</b> ` +
    `работа из источников карточки говорит именно об этой технологии` +
    (v.quote ? ` — <span class="q">«${esc(v.quote)}»</span>` : "") +
    (v.url ? ` <a href="${esc(v.url)}" target="_blank" rel="noopener">источник</a>` : "") +
    `. <span class="gen">Это подтверждает существование работ, а не стадию всего направления.</span></p>`;
}

// паспорт сигнала — одна кнопка «Почему это ещё слабый сигнал?»: что появилось → чем подтверждено → почему ещё
// рано → что вызывает сомнение → какой следующий факт усилит сигнал. Подтверждение — дословная цитата источника.
function passportBlock(p) {
  if (!p) return "";
  const step = (n, title, text, extra) => text ? `<li><span class="pp-n">${n}</span><div><b>${esc(title)}</b>` +
    `<p>${esc(text)}</p>${extra || ""}</div></li>` : "";
  const proof = p.quote ? `<blockquote>«${esc(p.quote)}»${p.url ? ` — <a href="${esc(p.url)}" target="_blank" rel="noopener">источник</a>` : ""}</blockquote>` : "";
  const status = p.confirmed
    ? `<p class="pp-ok">✓ Источники подтверждают событие: цитата найдена дословно в заголовке источника.</p>`
    : `<p class="pp-warn">Источники не подтвердили конкретное событие — сигнал понижен в балле. Ниже — что удалось найти.</p>`;
  return `<details class="passport"><summary>Почему это ещё слабый сигнал?</summary>${status}<ol class="pp">` +
    step(1, "Что появилось", p.what_new) +
    step(2, "Чем подтверждено", [p.event, p.who].filter(Boolean).join(" · "), proof) +
    step(3, "Почему ещё рано", p.why_early) +
    step(4, "Что вызывает сомнение", p.doubt) +
    step(5, "Какой следующий факт усилит сигнал", p.next_fact) + `</ol>` +
    `<p class="gen-note">Паспорт заполнен YandexGPT только по источникам карточки; цитата проверена программно.</p></details>`;
}

// объяснение словами: те же вклады SHAP, что на полосах, без участия языковой модели
function explainSentence(c) {
  if (!c.explain || !c.explain.length) return "";
  const up = c.explain.filter((x) => x.value > 0).slice(0, 3).map((x) => x.label);
  const down = c.explain.filter((x) => x.value < 0).slice(0, 2).map((x) => x.label);
  return `<p style="margin:0 0 8px"><b>Балл ранжирования ${Math.round(c.strength * 100)} из 100</b> ` +
    `<span class="gen">(место среди записей базы, не вероятность)</span>. ` +
    (up.length ? `Больше всего балл повышают: ${esc(up.join(", ").toLowerCase().replace(/hacker news/g, "Hacker News").replace(/y combinator/g, "Y Combinator").replace(/sec form d/g, "SEC Form D").replace(/википедии/g, "Википедии"))}. ` : "") +
    (down.length ? `Понижают: ${esc(down.join(", ").toLowerCase().replace(/hacker news/g, "Hacker News").replace(/y combinator/g, "Y Combinator").replace(/sec form d/g, "SEC Form D").replace(/википедии/g, "Википедии"))}.` : "") + `</p>`;
}

// исторический двойник: похожая тема 2021 года с похожей траекторией — и что с ней стало
function twinBlock(t) {
  const g = String(t.growth).replace(".", ",");
  const market = [t.yc_after ? `стартапов YC: ${t.yc_after}` : "", t.press_after ? `в прессе: ${t.press_after}` : ""]
    .filter(Boolean).join(", ");
  return `<p style="margin:10px 0 0"><b>Исторический двойник.</b> В конце 2021 года так же выглядела тема ` +
    `«${esc(t.term)}». К 2024–2026 её доля в работах arXiv ${t.growth >= 1 ? `выросла в ${g} раза` : `снизилась (×${g})`} ` +
    `(${fmt(t.docs_then)} → ${fmt(t.docs_later)} работ)${market ? `, после 2021: ${esc(market)}` : ""} — ` +
    `${t.takeoff ? "это взлёт по правилу машины времени." : "взлётом это не стало."}</p>`;
}

// участники — только организации, которые разрабатывают или исследуют технологии; медицинские, некоммерческие и
// прочие совпадения по аффилиации авторов (вроде федерации гинекологов в агентских платежах) не показываем
const ORG_OK = ["company", "education", "facility", "government"];
const ORG_TYPE = { company: "компания", education: "университет", facility: "лаборатория", government: "госорганизация",
  healthcare: "медицина", nonprofit: "некоммерческая", archive: "архив", funder: "фонд", other: "другое" };

// организации из авторов свежих работ (OpenAlex 2023–2026), стартапы YC и раунды SEC Form D
function actorsBlock(a) {
  const money = a.raised > 0 ? `, привлечено $${(a.raised / 1e6).toFixed(1)} млн` : "";
  const head = `<p><b>${fmt(a.orgs)}</b> организаций в свежих работах (из них компаний: ${fmt(a.companies)}, ` +
    `университетов: ${fmt(a.universities)}), стран: ${fmt(a.countries)}` +
    (a.cn_share >= 0.3 ? `, доля Китая ${Math.round(a.cn_share * 100)} %` : "") +
    `. Стартапов YC: ${fmt(a.yc.length)}` + (a.funded ? `, с раундом по SEC Form D: ${fmt(a.funded)}${money}` : "") + `.</p>`;
  const orgItem = (o) => `<li>${esc(o.name)}<div class="meta">${esc(ORG_TYPE[o.type] || o.type || "")} · ` +
    `${esc(o.country || "страна неизвестна")} · работ: ${fmt(o.works)}</div></li>`;
  // организация с одной работой — случайное совпадение термина (вроде федерации гинекологов в агентских платежах):
  // показываем только тех, у кого по теме не меньше двух работ
  const top = a.top_orgs.filter((o) => o.works >= 2 && ORG_OK.includes(o.type));
  const orgs = top.length ? `<ul class="src">${top.slice(0, 3).map(orgItem).join("")}</ul>` +
    (top.length > 3 ? `<details class="more"><summary>Ещё организации</summary><ul class="src">` +
      `${top.slice(3).map(orgItem).join("")}</ul></details>` : "") +
    `<p class="gen-note">Организации — по авторам работ OpenAlex 2023–2026, где встречается термин; сопоставление автоматическое. ` +
    `Показаны компании, университеты, лаборатории и госорганизации, у которых не меньше двух таких работ.</p>` : "";
  const yc = a.yc.length ? `<p style="margin:10px 0 4px"><b>Стартапы Y Combinator</b></p><ul class="src">${a.yc.map((y) =>
    `<li>${y.url ? `<a href="${esc(y.url)}" target="_blank" rel="noopener">${esc(y.name)}</a>` : esc(y.name)}</li>`).join("")}</ul>` : "";
  return head + orgs + yc;
}

// склонение: 1 работа, 2 работы, 5 работ
function plural(n, one, few, many) {
  const a = Math.abs(n) % 100, b = a % 10;
  return a > 10 && a < 20 ? many : b === 1 ? one : b >= 2 && b <= 4 ? few : many;
}
const NOUNS = { "работ": ["работа", "работы", "работ"], "записей": ["запись", "записи", "записей"], "историй": ["история", "истории", "историй"],
  "компаний": ["компания", "компании", "компаний"], "статей": ["статья", "статьи", "статей"] };
function stepValue(s) {
  const m = /^(\d[\d\s]*)\s+(\S+)/.exec(s.value || "");
  if (!m) return { n: 0, text: s.value || "" };
  const n = parseInt(m[1].replace(/\s/g, ""), 10), forms = NOUNS[m[2]];
  return { n, text: forms ? `${fmt(n)} ${plural(n, ...forms)}` : s.value };
}

// лестница зрелости: ступени растут от науки к энциклопедии; закрашено — слой уже есть (насыщенность — по числу
// упоминаний), пунктир — ещё нет; метка — докуда тема дошла
function ladderView(ladder) {
  if (!ladder || !ladder.length) return "";
  const W = 360, H = 182, base = 132, gap = 6, n = ladder.length;
  const w = (W - gap * (n - 1)) / n;
  const vals = ladder.map(stepValue);
  const maxLog = Math.max(1, ...vals.map((v) => Math.log10(v.n + 1)));
  let reach = -1;
  ladder.forEach((s, i) => { if (s.present === true) reach = i; });
  const steps = ladder.map((s, i) => {
    const x = i * (w + gap), h = 26 + i * 20, y = base - h;
    const on = s.present === true, na = s.present !== true && s.present !== false;
    const op = on ? 0.35 + 0.65 * Math.log10(vals[i].n + 1) / maxLog : 0;
    const rect = on
      ? `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="var(--accent)" fill-opacity="${op.toFixed(2)}"/>`
      : `<rect x="${x + 1}" y="${y + 1}" width="${w - 2}" height="${h - 2}" rx="6" fill="none" stroke="var(--muted)" stroke-dasharray="${na ? "2 3" : "5 4"}" opacity=".6"/>`;
    const val = `<text x="${x + w / 2}" y="${base - 8}" font-size="11" text-anchor="middle" fill="${on ? "#fff" : "var(--muted)"}">${esc(on ? vals[i].text : (s.value || "нет"))}</text>`;
    const LINES = { "Hacker News": ["Hacker", "News"], "Стартапы YC": ["Стартапы", "YC"], "Энциклопедия": ["Энцикло-", "педия"] };
    const parts = LINES[s.layer] || [s.layer];
    const lab = `<text x="${x + w / 2}" y="${base + 15}" font-size="11" font-weight="600" text-anchor="middle" fill="var(--text)">` +
      parts.map((p, j) => `<tspan x="${x + w / 2}" dy="${j ? 12 : 0}">${esc(p)}</tspan>`).join("") + `</text>`;
    const here = i === reach ? `<text x="${x + w / 2}" y="${y - 8}" font-size="11" text-anchor="middle" fill="var(--accent)" font-weight="700">▼ дальше всего</text>` : "";
    return rect + val + lab + here;
  }).join("");
  const zones = `<text x="0" y="${H - 2}" font-size="10.5" fill="var(--muted)">наука</text>` +
    `<text x="${W / 2}" y="${H - 2}" font-size="10.5" text-anchor="middle" fill="var(--muted)">рынок</text>` +
    `<text x="${W}" y="${H - 2}" font-size="10.5" text-anchor="end" fill="var(--muted)">энциклопедия</text>`;
  const filled = ladder.filter((s) => s.present === true).length;
  const last = ladder[n - 1];
  // только то, что показывают наши данные: в каких слоях корпуса найден сигнал (без выводов о зрелости)
  const where = ladder.filter((s) => s.present === true).map((s) => s.layer.toLowerCase()).join(", ");
  const verdict = `Сигнал найден в ${filled} из ${n} слоёв корпуса${where ? `: ${where}` : ""}. ` +
    `Отсутствие в слое значит «не нашли в нашем снимке корпуса», а не «технология незрелая».`;
  return `<svg class="ladder-svg" viewBox="-4 0 ${W + 8} ${H}" role="img" aria-label="Где найден сигнал">${steps}${zones}</svg>` +
    `<p class="muted" style="font-size:13px;margin:6px 0 0">${esc(verdict)}</p>`;
}

function step(s) {
  const cls = s.present === true ? "on" : s.present === false ? "off" : "na";
  return `<div class="step ${cls}"><b>${esc(s.layer)}</b>${esc(s.value)}</div>`;
}

function textBlock(title, t) {
  if (t && t.text) {
    const tag = t.generated ? `<span class="gen">сформулировано моделью по источникам карточки</span>` : "";
    return `<p><b>${esc(title)}.</b> ${esc(t.text)}${tag}</p>`;
  }
  return `<p><b>${esc(title)}.</b> <span class="placeholder">появится после подключения YandexGPT: формулировка по источникам карточки</span></p>`;
}

// водопад SHAP: от средней технологии каждый признак сдвигает балл вверх (зелёный) или вниз (красный);
// видно, как из вкладов складывается итог. Числа — вклад в логит модели (единицы SHAP).
function waterfall(items) {
  if (!items || !items.length) return "";
  const num = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2).replace(".", ",");
  let acc = 0;
  const steps = items.map((x) => { const s = acc; acc += x.value; return { ...x, s, e: acc }; });
  const lo = Math.min(0, ...steps.map((t) => Math.min(t.s, t.e)));
  const hi = Math.max(0, ...steps.map((t) => Math.max(t.s, t.e)));
  const X = (v) => ((v - lo) / ((hi - lo) || 1)) * 100;
  const zero = X(0);
  const row = (label, left, width, cls, val, title) =>
    `<div class="wf-row"><span class="wf-label" title="${esc(title || "")}">${esc(label)}</span>` +
    `<span class="wf-track"><i class="wf-zero" style="left:${zero}%"></i>` +
    `<span class="wf-bar ${cls}" style="left:${left}%;width:${Math.max(width, 0.8)}%"></span></span>` +
    `<span class="wf-val ${cls}">${val}</span></div>`;
  const body = steps.map((t) => row(t.label, X(Math.min(t.s, t.e)), Math.abs(X(t.e) - X(t.s)),
    t.value >= 0 ? "p" : "n", num(t.value), t.detail)).join("");
  const total = row("Итог относительно средней технологии", Math.min(zero, X(acc)), Math.abs(X(acc) - zero),
    acc >= 0 ? "p total" : "n total", num(acc), "");
  return `<div class="wf"><div class="wf-row wf-head"><span class="wf-label">Средняя технология</span>` +
    `<span class="wf-track"><i class="wf-zero" style="left:${zero}%"></i></span><span class="wf-val">0</span></div>${body}${total}</div>`;
}

function contrib(items) {
  const max = Math.max(0.01, ...items.map((x) => Math.abs(x.value)));
  return `<div class="contrib">${items.map((x) => {
    const w = Math.round(Math.abs(x.value) / max * 50);
    const bar = x.value >= 0 ? `<span class="p" style="width:${w}%"></span>` : `<span class="n" style="width:${w}%"></span>`;
    return `<span title="${esc(x.detail || "")}">${esc(x.label)}</span><span class="cbar">${bar}</span>`;
  }).join("")}</div>`;
}

function sources(list) {
  if (!list || !list.length) return `<p class="placeholder">Документов-подтверждений в корпусе не нашлось.</p>`;
  const tcls = { "высокий": "hi", "средний": "mid", "пониженный": "lo" };
  // список, а не таблица: одинаково читается на компьютере и на телефоне
  return `<ul class="src">${list.map((s) => `<li>` +
    `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title)}</a>` +
    `<div class="meta">${esc(s.source_name)} · ${esc(s.published || "дата неизвестна")} · ${esc(s.source_type)} · ${esc(s.language)} · ` +
    `<span class="trust ${tcls[s.trust] || ""}" title="Уровень доверия источнику">доверие: ${esc(s.trust)}</span></div></li>`).join("")}</ul>`;
}

function spark(s) {
  // доли по годам, каждый ряд нормирован на свой максимум: сравниваем форму роста, а не абсолютные величины
  const W = 600, H = 110, P = 22;
  // пустые годы в начале не показываем: график начинается за год до первого ненулевого значения,
  // но не короче MIN_YEARS лет — иначе у свежих тем 10 лет пустой линии занимают весь график
  const MIN_YEARS = 5;
  const first = Math.min(...s.lines.map((l) => l.values.findIndex((v) => v > 0)).filter((i) => i >= 0));
  const start = Number.isFinite(first) ? Math.max(0, Math.min(first - 1, s.years.length - MIN_YEARS)) : 0;
  const years = s.years.slice(start);
  const x = (i) => P + (W - 2 * P) * i / (years.length - 1);
  const lines = s.lines.map((l) => ({ ...l, values: l.values.slice(start) })).filter((l) => l.values.some((v) => v > 0));
  const paths = lines.map((l) => {
    const m = Math.max(...l.values) || 1;
    const d = l.values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${(H - P - (H - 2 * P) * v / m).toFixed(1)}`).join("");
    return `<path d="${d}" fill="none" stroke="var(--${l.color})" stroke-width="2.2" stroke-linejoin="round"/>`;
  }).join("");
  void paths;                                          // вместо одного графика в своих масштабах — малые графики
  // по одному на источник: у каждого своя ось с числами и единицами (наука — работ на 10 тыс. всех работ года,
  // Hacker News — историй на 10 тыс., YC — штук), годы общие
  const unit = (l) => /наук/i.test(l.label) ? "работ на 10 тыс. всех научных работ года"
    : /hacker/i.test(l.label) ? "историй на 10 тыс. всех историй года" : "стартапов в год";
  const fmtv = (v) => (v >= 10 ? Math.round(v) : v >= 1 ? v.toFixed(1) : v.toFixed(2)).toString().replace(".", ",");
  const h = 64, pad = 8, left = 46;
  const xs = (i) => left + (W - left - P) * i / (years.length - 1);
  const step = years.length > 8 ? 2 : 1;               // на коротком отрезке подписываем каждый год
  const minis = lines.map((l, k) => {
    const m = Math.max(...l.values) || 1;
    const y = (v) => pad + (h - 2 * pad) * (1 - v / m);
    const d = l.values.map((v, i) => `${i ? "L" : "M"}${xs(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
    const area = `${d}L${xs(l.values.length - 1).toFixed(1)},${y(0)}L${xs(0).toFixed(1)},${y(0)}Z`;
    const last = k === lines.length - 1;
    const H2 = h + (last ? 16 : 0);
    const ticks = last ? years.map((yy, i) => (i % step === 0 || i === years.length - 1)
      ? `<text x="${xs(i)}" y="${h + 12}" font-size="11" text-anchor="middle" fill="var(--muted)">${yy}</text>` : "").join("") : "";
    return `<div class="mini"><div class="mini-title"><i style="background:var(--${l.color})"></i>${esc(l.label.replace(/\s*\(.*\)/, ""))}` +
      ` <span class="muted">— ${esc(unit(l))}</span></div>` +
      `<svg viewBox="0 0 ${W} ${H2}" role="img" aria-label="${esc(l.label)} по годам">` +
      `<line x1="${left}" x2="${W - P}" y1="${y(0)}" y2="${y(0)}" stroke="var(--line)"/>` +
      `<line x1="${left}" x2="${W - P}" y1="${y(m)}" y2="${y(m)}" stroke="var(--line)" stroke-dasharray="3 3"/>` +
      `<text x="${left - 6}" y="${y(m) + 4}" font-size="11" text-anchor="end" fill="var(--muted)">${fmtv(m)}</text>` +
      `<text x="${left - 6}" y="${y(0) + 4}" font-size="11" text-anchor="end" fill="var(--muted)">0</text>` +
      `<path d="${area}" fill="var(--${l.color})" opacity="0.12"/>` +
      `<path d="${d}" fill="none" stroke="var(--${l.color})" stroke-width="2.2" stroke-linejoin="round"/>${ticks}</svg></div>`;
  }).join("");
  return `<div class="minis">${minis}</div>`;
}

$("#search").addEventListener("submit", (e) => { e.preventDefault(); const q = $("#q").value.trim(); if (q) search(q); });
load().catch((e) => { $("#status").innerHTML = `<div class="note">Ошибка: ${esc(e.message)}</div>`; });
