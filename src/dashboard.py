"""Самодостаточный HTML-дашборд по data/signals/stable.json.

    python src/dashboard.py            # -> data/signals/dashboard.html

Данные вшиты в страницу, сервер не нужен. Открывается двойным щелчком, работает в светлой и тёмной теме.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "signals" / "stable.json"
FORECAST = ROOT / "data" / "forecast" / "forecast.json"
OUT = ROOT / "data" / "signals" / "dashboard.html"
DROP = {"doc_ids", "core_ids"}

TEMPLATE = r"""<meta charset="utf-8">
<title>Слабые сигналы газовой отрасли</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Golos+Text:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root{
  color-scheme:light;
  --bg:#f3f6f8; --surface:#ffffff; --surface-2:#e9eef2; --line:#d5dde4; --line-soft:#e6ecf0;
  --ink:#14202b; --ink-2:#4f6272; --ink-3:#7f8f9c; --accent:#1c5cab; --accent-ink:#ffffff; --accent-soft:#dfe9f7;
  --d1:#2a78d6; --d2:#eb6834; --d3:#1baf7a; --d4:#c98500;
  --good:#1d7a3f; --shadow:0 1px 2px rgba(20,32,43,.06),0 4px 14px rgba(20,32,43,.05);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --bg:#0f1519; --surface:#182027; --surface-2:#1f2a33; --line:#2c3944; --line-soft:#243039;
    --ink:#eef3f7; --ink-2:#a9b7c2; --ink-3:#7d8c98; --accent:#5598e7; --accent-ink:#0f1519; --accent-soft:#1f3350;
    --d1:#3987e5; --d2:#d95926; --d3:#199e70; --d4:#eda100;
    --good:#5ccb84; --shadow:0 1px 2px rgba(0,0,0,.4),0 6px 18px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --bg:#0f1519; --surface:#182027; --surface-2:#1f2a33; --line:#2c3944; --line-soft:#243039;
  --ink:#eef3f7; --ink-2:#a9b7c2; --ink-3:#7d8c98; --accent:#5598e7; --accent-ink:#0f1519; --accent-soft:#1f3350;
  --d1:#3987e5; --d2:#d95926; --d3:#199e70; --d4:#eda100;
  --good:#5ccb84; --shadow:0 1px 2px rgba(0,0,0,.4),0 6px 18px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Golos Text",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-size:14px;line-height:1.45}
.mono{font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
.num{font-variant-numeric:tabular-nums}
a{color:var(--accent)}
.wrap{max-width:1440px;margin:0 auto;padding:28px 28px 64px}
header{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:16px 32px;margin-bottom:22px}
h1{font-size:26px;font-weight:700;letter-spacing:-.01em;margin:0 0 4px;text-wrap:balance}
.sub{color:var(--ink-2);max-width:68ch;margin:0}
.eyebrow{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
.meta{color:var(--ink-3);font-size:12.5px;text-align:right}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:18px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px 12px;box-shadow:var(--shadow)}
.tile .v{font-size:28px;font-weight:600;letter-spacing:-.02em;line-height:1.1;margin:6px 0 2px}
.tile .h{color:var(--ink-2);font-size:12.5px}

.filters{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin-bottom:18px}
.filters label{color:var(--ink-2);font-size:12.5px;display:flex;align-items:center;gap:6px}
.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{border:1px solid var(--line);background:var(--surface);color:var(--ink-2);border-radius:999px;padding:4px 10px 4px 8px;font:inherit;font-size:12.5px;cursor:pointer;display:inline-flex;align-items:center;gap:6px}
.chip .sw{width:10px;height:10px;border-radius:2px;background:var(--c)}
.chip[aria-pressed="true"]{border-color:var(--c);color:var(--ink);background:color-mix(in srgb,var(--c) 12%,var(--surface))}
.chip:focus-visible,select:focus-visible,input:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
select,input[type=search]{font:inherit;font-size:13px;color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:5px 8px}
input[type=search]{min-width:200px}
.spacer{flex:1}
.count{color:var(--ink-3);font-size:12.5px}

.grid{display:grid;grid-template-columns:minmax(380px,5fr) minmax(420px,7fr);gap:18px;align-items:start}
@media (max-width:980px){.grid{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow)}
.panel-h{display:flex;justify-content:space-between;align-items:baseline;gap:12px;padding:14px 16px 6px}
.panel-h h2{font-size:15px;font-weight:600;margin:0}
.panel-h .hint{color:var(--ink-3);font-size:12px}
.chart-wrap{overflow-x:auto;padding:4px 8px 12px}
svg text{font-family:"Golos Text",system-ui,sans-serif}
.rowlbl{fill:var(--ink);font-size:12px;cursor:pointer}
.rowlbl:hover{fill:var(--accent)}
.axis{fill:var(--ink-3);font-size:11px}
.gridl{stroke:var(--line-soft);stroke-width:1}
.rng{stroke-width:2;stroke-linecap:round;opacity:.55}
.med{stroke:var(--surface);stroke-width:2}
.rowhit{fill:transparent;cursor:pointer}
.rowhit:hover + .rowmark .rng{opacity:.9}
.tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);padding:8px 10px;border-radius:6px;font-size:12px;line-height:1.4;max-width:280px;box-shadow:var(--shadow);z-index:10;display:none}
.tip b{font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;padding:0 16px 12px;color:var(--ink-2);font-size:12px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{width:10px;height:10px;border-radius:2px;background:var(--c);display:inline-block}

.list{display:flex;flex-direction:column}
.card{border-top:1px solid var(--line-soft)}
.card:first-child{border-top:0}
.card-h{display:grid;grid-template-columns:34px minmax(0,1fr) 128px 74px 60px;gap:12px;align-items:center;padding:10px 16px;cursor:pointer;width:100%;background:none;border:0;color:inherit;font:inherit;text-align:left}
.card-h:hover{background:var(--surface-2)}
.card-h .rk{color:var(--ink-3);font-size:13px}
.card-h .nm{font-weight:600;font-size:14px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.card-h .dm{display:inline-flex;align-items:center;gap:6px;color:var(--ink-2);font-size:12px;margin-top:2px}
.card-h .dm i{width:8px;height:8px;border-radius:2px;background:var(--c);display:inline-block}
.stab{font-size:12px;color:var(--ink-2);text-align:right}
.stab b{color:var(--ink);font-weight:600}
.stab .bar{display:flex;gap:2px;justify-content:flex-end;margin-top:3px}
.stab .bar i{width:9px;height:6px;border-radius:1px;background:var(--line)}
.stab .bar i.on{background:var(--good)}
.gr{font-size:12.5px;text-align:right;color:var(--ink-2)}
.gr b{color:var(--ink);font-weight:600}
.card-b{display:none;padding:2px 16px 16px 62px;color:var(--ink);font-size:13.5px}
.card.open .card-b{display:block}
.card.open .card-h{background:var(--surface-2)}
.card-b h4{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);margin:14px 0 6px}
.card-b ul{margin:0;padding-left:18px}
.card-b li{margin:3px 0}
.card-b .who{color:var(--ink-2)}
.card-b .ind{display:inline-block;background:var(--accent-soft);color:var(--ink);border-radius:4px;padding:2px 7px;font-size:12px;margin:2px 4px 2px 0}
.years{display:flex;gap:0;margin-top:8px;align-items:flex-end}
.years .y{flex:1;text-align:center;font-size:10.5px;color:var(--ink-3)}
.years .y b{display:block;color:var(--ink);font-weight:500;font-size:11.5px}
.empty{padding:28px 16px;color:var(--ink-3);text-align:center}
.fc{margin-top:18px}
.fc .panel-h{flex-wrap:wrap}
.fc-meta{display:flex;flex-wrap:wrap;gap:8px 18px;padding:0 16px 10px;color:var(--ink-2);font-size:12.5px}
.fc-meta b{color:var(--ink);font-weight:600}
.fc-table{width:100%;border-collapse:collapse;font-size:13px}
.fc-table th{text-align:left;font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);padding:8px 10px;border-bottom:1px solid var(--line)}
.fc-table td{padding:8px 10px;border-top:1px solid var(--line-soft);vertical-align:middle}
.fc-table tr:hover td{background:var(--surface-2)}
.fc-table .t{font-weight:600}
.fc-table .r{text-align:right}
.pbar{display:inline-block;width:70px;height:7px;border-radius:4px;background:var(--line);vertical-align:middle;margin-right:6px;overflow:hidden}
.pbar i{display:block;height:100%;background:var(--accent)}
.fate{display:inline-block;border-radius:4px;padding:1px 7px;font-size:12px}
.fate.up{background:color-mix(in srgb,var(--good) 16%,var(--surface));color:var(--good)}
.fate.hold{background:var(--surface-2);color:var(--ink-2)}
.fate.down{background:color-mix(in srgb,#c8432b 14%,var(--surface));color:#b13a25}
:root[data-theme="dark"] .fate.down{color:#f08c78}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .fate.down{color:#f08c78}}
.fc-wrap{overflow-x:auto}
@media (max-width:640px){.card-h{grid-template-columns:28px minmax(0,1fr) 74px}.card-h .sp,.card-h .gr{display:none}.card-b{padding-left:16px}.wrap{padding:16px}}
@media (prefers-reduced-motion:no-preference){.card-h{transition:background .12s}}
</style>

<div class="wrap">
<header>
  <div>
    <div class="eyebrow">Газпром · научно-технологический форсайт</div>
    <h1>Слабые сигналы газовой отрасли</h1>
    <p class="sub">Зарождающиеся темы в статьях и препринтах по четырём направлениям. Ранг это медиана по нескольким прогонам кластеризации, диапазон показывает, насколько сигнал плавал между прогонами.</p>
  </div>
  <div class="meta" id="meta"></div>
</header>

<div class="tiles" id="tiles"></div>

<div class="filters" role="group" aria-label="Фильтры">
  <div class="chips" id="domChips"></div>
  <label>Устойчивость <select id="fStab"><option value="0">любая</option><option value="3" selected>≥ 3 из 5 прогонов</option><option value="5">все 5</option></select></label>
  <label>Детекторов <select id="fLayers"><option value="1">≥ 1</option><option value="2" selected>≥ 2</option><option value="3">≥ 3</option></select></label>
  <label>Сортировка <select id="fSort"><option value="rank">по рангу</option><option value="growth">по росту</option><option value="n_docs">по объёму</option><option value="birth">по году рождения</option></select></label>
  <input type="search" id="fQ" placeholder="Поиск по названию и синонимам" aria-label="Поиск">
  <span class="spacer"></span>
  <span class="count" id="count"></span>
</div>

<div class="grid">
  <section class="panel">
    <div class="panel-h"><h2>Устойчивость ранга</h2><span class="hint">точка это медиана, отрезок это разброс между прогонами</span></div>
    <div class="legend" id="legend"></div>
    <div class="chart-wrap"><svg id="chart" role="img" aria-label="Ранг сигналов: медиана и разброс по прогонам"></svg></div>
  </section>
  <section class="panel">
    <div class="panel-h"><h2>Сигналы</h2><span class="hint">щёлкните строку, чтобы раскрыть карточку</span></div>
    <div class="list" id="list"></div>
  </section>
</div>

<section class="panel fc" id="fcPanel">
  <div class="panel-h"><h2>Прогноз: что взлетит к <span id="fcH"></span></h2><span class="hint">модель обучена на истории 2017–2021, признаки считаются только по прошлому</span></div>
  <div class="fc-meta" id="fcMeta"></div>
  <div class="fc-wrap"><table class="fc-table" id="fcTable"></table></div>
</section>
</div>
<div class="tip" id="tip"></div>

<script id="data" type="application/json">__DATA__</script>
<script id="fcdata" type="application/json">__FORECAST__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const YEARS = Array.from({length: DATA[0].counts.length}, (_, i) => 2015 + i);
const DOMAINS = [...new Set(DATA.map(d => d.domain))];
const DOMCOLOR = {}; DOMAINS.forEach((d, i) => DOMCOLOR[d] = `var(--d${i + 1})`);
const N_RUNS = DATA[0].n_runs;
const state = {domains: new Set(DOMAINS), stab: 3, layers: 2, sort: 'rank', q: ''};
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

// ---- сводка ----
{
  const top40 = DATA.filter(d => d.rank <= 40);
  const stable = top40.filter(d => d.runs_seen >= Math.ceil(N_RUNS / 2)).length;
  const multi = DATA.filter(d => d.layers_max >= 2).length;
  const allRuns = DATA.filter(d => d.runs_seen === N_RUNS).length;
  const tiles = [
    ['Сигналов всего', DATA.length, 'групп после склейки прогонов'],
    ['Устойчивых в топ-40', `${stable} из 40`, `появились не менее чем в ${Math.ceil(N_RUNS / 2)} прогонах`],
    ['Во всех прогонах', allRuns, `сигналов найдены во всех ${N_RUNS} прогонах`],
    ['Подтверждено ≥ 2 детекторами', multi, 'независимые слои согласны'],
  ];
  $('#tiles').innerHTML = tiles.map(([h, v, s]) => `<div class="tile"><div class="h">${h}</div><div class="v num">${v}</div><div class="h">${s}</div></div>`).join('');
  $('#meta').innerHTML = `${N_RUNS} прогонов · годы ${YEARS[0]}–${YEARS.at(-1)}<br>последний год досчитан до годового темпа`;
  $('#domChips').innerHTML = DOMAINS.map(d => `<button class="chip" aria-pressed="true" data-d="${esc(d)}" style="--c:${DOMCOLOR[d]}"><span class="sw"></span>${esc(d)}</button>`).join('');
  $('#legend').innerHTML = DOMAINS.map(d => `<span style="--c:${DOMCOLOR[d]}"><i></i>${esc(d)}</span>`).join('');
}

// ---- фильтрация ----
function filtered() {
  const q = state.q.trim().toLowerCase();
  let rows = DATA.filter(d => state.domains.has(d.domain) && d.runs_seen >= state.stab && d.layers_max >= state.layers
    && (!q || d.name.toLowerCase().includes(q) || d.aliases.join(' ').toLowerCase().includes(q)));
  const key = {rank: d => d.rank, growth: d => -d.growth, n_docs: d => -d.n_docs, birth: d => -d.birth_year}[state.sort];
  return rows.sort((a, b) => key(a) - key(b));
}

// ---- график устойчивости ----
function sparkline(counts, color) {
  const w = 120, h = 28, pad = 2, m = Math.max(...counts, 1);
  const pts = counts.map((c, i) => [pad + i * (w - 2 * pad) / (counts.length - 1), h - pad - (c / m) * (h - 2 * pad)]);
  const line = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ');
  const area = line + ` L${pts.at(-1)[0].toFixed(1)},${h - pad} L${pts[0][0].toFixed(1)},${h - pad} Z`;
  const e = pts.at(-1);
  return `<svg class="sp" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true"><path d="${area}" fill="${color}" opacity=".16"/><path d="${line}" fill="none" stroke="${color}" stroke-width="1.6"/><circle cx="${e[0]}" cy="${e[1]}" r="2.6" fill="${color}"/></svg>`;
}

function drawChart(rows) {
  const svg = $('#chart');
  const top = rows.slice(0, 40);
  const rowH = 22, left = 210, right = 22, topPad = 22, w = 560, plotW = w - left - right;
  const h = topPad + top.length * rowH + 10;
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`); svg.setAttribute('width', w); svg.setAttribute('height', h);
  const maxR = 41, x = r => left + (r - 1) / (maxR - 1) * plotW;
  let s = '';
  for (const r of [1, 10, 20, 30, 40]) s += `<line class="gridl" x1="${x(r)}" x2="${x(r)}" y1="${topPad - 6}" y2="${h - 8}"/><text class="axis" x="${x(r)}" y="12" text-anchor="middle">${r}</text>`;
  s += `<text class="axis" x="${left - 16}" y="12" text-anchor="end" style="font-weight:600">ранг →</text>`;
  top.forEach((d, i) => {
    const y = topPad + i * rowH + rowH / 2, c = DOMCOLOR[d.domain];
    const name = d.name.length > 30 ? d.name.slice(0, 29) + '…' : d.name;
    s += `<rect class="rowhit" x="0" y="${y - rowH / 2}" width="${w}" height="${rowH}" data-i="${d.rank}"/>`;
    s += `<g class="rowmark"><text class="rowlbl" x="${left - 10}" y="${y + 4}" text-anchor="end" data-i="${d.rank}">${esc(name)}</text>`;
    s += `<line class="rng" x1="${x(d.rank_min)}" x2="${x(Math.min(d.rank_max, maxR))}" y1="${y}" y2="${y}" stroke="${c}"/>`;
    if (d.runs_seen < N_RUNS) s += `<line x1="${x(Math.min(d.rank_max, maxR))}" x2="${x(maxR)}" y1="${y}" y2="${y}" stroke="${c}" stroke-width="2" stroke-dasharray="2 3" opacity=".45"/>`;
    s += `<circle class="med" cx="${x(Math.min(d.rank_median, maxR))}" cy="${y}" r="4.5" fill="${c}"/></g>`;
  });
  svg.innerHTML = s;
  const tip = $('#tip');
  svg.querySelectorAll('.rowhit').forEach(el => {
    const d = DATA.find(r => r.rank == el.dataset.i);
    el.addEventListener('mousemove', e => {
      tip.style.display = 'block'; tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 14) + 'px';
      tip.innerHTML = `<b>${esc(d.name)}</b><br>медиана ${d.rank_median}, разброс ${d.rank_min}–${d.rank_max}<br>в ${d.runs_seen} из ${d.n_runs} прогонов${d.runs_seen < d.n_runs ? '<br><span style="opacity:.75">пунктир: не найден в части прогонов</span>' : ''}`;
    });
    el.addEventListener('mouseleave', () => tip.style.display = 'none');
    el.addEventListener('click', () => openCard(d.rank));
  });
  svg.querySelectorAll('.rowlbl').forEach(el => el.addEventListener('click', () => openCard(el.dataset.i)));
}

// ---- карточки ----
function card(d) {
  const c = DOMCOLOR[d.domain];
  const bars = Array.from({length: d.n_runs}, (_, i) => `<i class="${i < d.runs_seen ? 'on' : ''}"></i>`).join('');
  const why = [...d.signals, ...d.detectors].map(t => `<li>${esc(t)}</li>`).join('');
  const ex = d.examples.map(e => `<li><a href="${esc(e.url)}" target="_blank" rel="noopener">${esc(e.title)}</a> <span class="who">(${e.year}, ${esc(e.venue || 'без площадки')})</span></li>`).join('');
  const years = d.counts.map((v, i) => `<div class="y"><b class="num">${v}</b>${String(YEARS[i]).slice(2)}</div>`).join('');
  return `<article class="card" id="c${d.rank}" style="--c:${c}">
    <button class="card-h" aria-expanded="false">
      <span class="rk num">${d.rank}</span>
      <span><span class="nm">${esc(d.name)}</span><br><span class="dm"><i></i>${esc(d.domain)}${d.aliases.length ? ` · ${esc(d.aliases[0].slice(0, 40))}` : ''}</span></span>
      ${sparkline(d.counts, c)}
      <span class="stab"><b class="num">${d.rank_median}</b> <span class="num">(${d.rank_min}–${d.rank_max})</span><span class="bar" title="в ${d.runs_seen} из ${d.n_runs} прогонов">${bars}</span></span>
      <span class="gr"><b class="num">${d.growth > 0 ? '+' : ''}${d.growth.toFixed(2)}</b><br><span class="num">${d.n_docs} док.</span></span>
    </button>
    <div class="card-b">
      <div class="years">${years}</div>
      <h4>Почему это сигнал</h4><ul>${why}</ul>
      <h4>Кто</h4><div class="who">${esc(d.top_orgs.slice(0, 4).join('; '))}<br>Страны: ${esc(d.top_countries.join(', '))}</div>
      ${d.industry.length ? `<h4>Индустрия</h4><div>${d.industry.map(i => `<span class="ind">${esc(i)}</span>`).join('')}</div>` : ''}
      ${d.aliases.length ? `<h4>Синонимы от других детекторов</h4><div class="who">${esc(d.aliases.join(' · '))}</div>` : ''}
      <h4>Примеры</h4><ul>${ex}</ul>
    </div></article>`;
}
function openCard(rank) {
  const el = document.getElementById('c' + rank);
  if (!el) return;
  el.classList.add('open'); el.querySelector('.card-h').setAttribute('aria-expanded', 'true');
  el.scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'center'});
}
function render() {
  const rows = filtered();
  $('#count').textContent = `${rows.length} из ${DATA.length}`;
  $('#list').innerHTML = rows.length ? rows.map(card).join('') : '<div class="empty">Ничего не подходит под фильтры. Ослабьте устойчивость или число детекторов.</div>';
  $('#list').querySelectorAll('.card-h').forEach(b => b.addEventListener('click', () => {
    const a = b.parentElement, open = a.classList.toggle('open'); b.setAttribute('aria-expanded', open);
  }));
  drawChart(rows);
}
$('#domChips').addEventListener('click', e => {
  const b = e.target.closest('.chip'); if (!b) return;
  const d = b.dataset.d, on = b.getAttribute('aria-pressed') === 'true';
  if (on && state.domains.size === 1) return;
  on ? state.domains.delete(d) : state.domains.add(d);
  b.setAttribute('aria-pressed', !on); render();
});
$('#fStab').addEventListener('change', e => { state.stab = +e.target.value; render(); });
$('#fLayers').addEventListener('change', e => { state.layers = +e.target.value; render(); });
$('#fSort').addEventListener('change', e => { state.sort = e.target.value; render(); });
$('#fQ').addEventListener('input', e => { state.q = e.target.value; render(); });
render();

// ---- прогноз ----
(function () {
  const FC = JSON.parse(document.getElementById('fcdata').textContent);
  const rows = (FC.forecast || []).filter(r => r.kind === 'термины');
  if (!rows.length) { $('#fcPanel').hidden = true; return; }
  $('#fcH').textContent = `${FC.horizon[0]}–${FC.horizon[1]}`;
  const v = (FC.validation || []).filter(r => r.kind === 'термины');
  const all = v.find(r => r.freeze === 'все');
  const meta = [`зафиксировано <b>${FC.date}</b>`];
  if (all) meta.push(`проверка на невиданных 2020–2021: точность топ-30 <b>${Math.round(all.ens * 100)}%</b> против базы ${Math.round(all.base * 100)}% (×${(all.ens / all.base).toFixed(1)}), топ-10 <b>${Math.round(all.ens10 * 100)}%</b> (×${(all.ens10 / all.base).toFixed(1)})`);
  meta.push('критерий успеха: частота термина в окне прогноза минимум вдвое выше, чем в 2024–2026');
  $('#fcMeta').innerHTML = meta.map(m => `<span>${m}</span>`).join('');
  const fate = r => r.twin_takeoff ? `<span class="fate up">вырос ×${r.twin_ratio.toFixed(1)}</span>`
    : r.twin_ratio >= 0.8 ? `<span class="fate hold">удержался ×${r.twin_ratio.toFixed(1)}</span>`
    : `<span class="fate down">угас ×${r.twin_ratio.toFixed(1)}</span>`;
  $('#fcTable').innerHTML = `<thead><tr><th>#</th><th>Термин</th><th>p(взлёт)</th><th>Динамика 2015–26</th><th class="r">Докум. 2024–26</th><th>Исторический двойник</th><th>Судьба двойника</th></tr></thead><tbody>` +
    rows.slice(0, 25).map(r => `<tr><td class="num">${r.rank}</td><td class="t">${esc(r.term)}</td>
      <td><span class="pbar"><i style="width:${Math.round(r.p * 100)}%"></i></span><span class="num">${r.p.toFixed(2)}</span></td>
      <td>${sparkline(r.counts, 'var(--accent)')}</td><td class="r num">${r.recent_docs}</td>
      <td>${esc(r.twin)} <span style="color:var(--ink-3)">(${r.twin_freeze})</span></td><td>${fate(r)}</td></tr>`).join('') + '</tbody>';
})();
</script>
"""


def main():
    # pandas пишет NaN как есть, а это не JSON: браузер падает на парсинге. Превращаем в null.
    rows = json.loads(SRC.read_text(encoding="utf-8"), parse_constant=lambda _: None)
    slim = [{k: v for k, v in r.items() if k not in DROP} for r in rows]
    payload = json.dumps(slim, ensure_ascii=False).replace("</", "<\\/")
    fc = json.loads(FORECAST.read_text(encoding="utf-8"), parse_constant=lambda _: None) if FORECAST.exists() \
        else {"date": "", "horizon": [2029, 2031], "validation": [], "forecast": []}
    fpayload = json.dumps(fc, ensure_ascii=False).replace("</", "<\\/")
    OUT.write_text(TEMPLATE.replace("__DATA__", payload).replace("__FORECAST__", fpayload), encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size // 1024} KB, сигналов {len(slim)}")


if __name__ == "__main__":
    main()
