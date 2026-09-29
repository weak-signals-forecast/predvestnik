"""База слабых сигналов: всё, что не зависит от запроса, проверяется один раз при сборке; на запросе модель только
выбирает из базы.

    python -m registry.signal_base candidates          # шаг 1, без модели -> base_candidates.parquet
    python -m registry.signal_base check --max-calls N # шаг 2, YandexGPT Pro по кандидатам (кэш base_check.jsonl)
    python -m registry.signal_base build               # шаг 3 -> signal_base.parquet (только прошедшие)
    python -m registry.signal_base eval                # покрытие размеченных строк на каждом шаге

Шаг 1 (данные, без модели): технологии, одобренные YandexGPT Lite (`llm_assess.jsonl`), без общих русских названий
→ не зрелые (первый год с ≥ 3 работами позже MATURE_YEAR) → без обрывков фраз и рубрик «ИИ + общее слово»
(правила по словам, см. FRAG_*, UMB_*) → склейка вариантов одной технологии (сходство «название —
описание» ≥ ALIAS_SIM; остаётся вариант с лучшим баллом, остальные — в `aliases`) → лучшие по обученному баллу внутри
каждой области ТЗ (PER_AREA) и среди остальных (OTHER).
Шаг 2: Pro относит каждого кандидата к одному типу (KINDS) — выбор из списка устойчивее, чем «да/нет»; измерения
из корпуса в промпте; ответ принимается только с дословным повтором фразы.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import REGISTRY, SOURCES, YEARS
from .search import MATURE_YEAR, Searcher

REG = REGISTRY
CAND = REG / "base_candidates.parquet"
CHECK = REG / "base_check.jsonl"
BASE = REG / "signal_base.parquet"
REVIEW = Path(__file__).resolve().parents[1] / "data" / "labels" / "base_review.csv"
DROP_VERDICTS = ("F", "D", "N")         # обрывок, дубль, не технология
ALIAS_SIM = 0.88
PER_AREA = 600
OTHER = 2000
RENEWAL_ACC, RENEWAL_MIN = 1.5, 10           # «обновление»: старый термин, который снова ускоряется
RENEWAL_PER_AREA, RENEWAL_OTHER = 0, 0            # выключено: +600 записей шума, 0 новых сигналов организаторов
# обрывки: фраза начинается хвостом предыдущего термина («time …» из «real time», «assisted …») или кончается
# определением без главного слова («… agentic», «… robotic», «… assisted»). На разметке: неверных 32 из 276, верных 3.
FRAG_START = {"time", "world", "real", "resource", "train", "scale", "native", "order", "assisted", "driven", "aware",
              "based", "enabled", "powered", "rich", "free", "shot", "end", "loop", "source", "grade", "level", "wide",
              "like", "term", "contact", "grained", "specific", "agnostic", "efficient", "invariant", "unit", "gold",
              "backed", "frontier", "class"}
FRAG_END = {"assisted", "agentic", "robotic", "financial", "real", "rich", "driven", "aware", "based", "free",
            "massive", "enabled", "powered", "native", "secure", "open", "deep", "co", "enables", "wrong", "llm",
            "robot", "mobile", "self", "class", "augmented", "specific", "backed", "guided", "integrated", "llama",
            # разговорные хвосты из заголовков Hacker News: «agents actually», «agents don», «agent writes»
            "actually", "don", "doesn", "writes", "write", "just", "really", "still", "now", "need", "works", "make",
            "makes", "get", "gets", "want", "think", "can", "will", "should", "going", "built", "made"}
# рубрики «ИИ + общее слово»: все слова — из ИИ-голов и общих слов. На разметке: неверных 35, верных 1.
UMB_HEADS = {"ai", "agentic", "llm", "llms", "agent", "agents", "genai", "generative", "intelligent"}
UMB_GENERIC = {"systems", "system", "platform", "platforms", "pipeline", "pipelines", "advisors", "assets", "concierge",
               "discovery", "os", "runtime", "infrastructure", "governed", "model", "models", "workloads", "clusters",
               "applications", "solutions", "tools", "services", "era", "adoption", "technology", "technologies",
               "compliant", "combining", "strategic", "native", "production", "enterprise", "personal", "research",
               "inference", "forecast", "engine", "stack", "ecosystem", "economy", "future", "development",
               "integration", "capabilities", "innovation", "transformation", "governance", "risk", "assessment",
               "assistants", "assistant", "open", "source", "scale", "secure"}


# после расширения базы (кандидаты без проверки Pro): обрывки с глагольной формой в начале («building agentic ai»,
# «linking device», «efficiently training large») и с висящим словом в конце («… large», «… multi», «hardware aligned»)
FRAG_START |= {"efficiently", "linking", "context", "latency", "leveraging", "integrating", "enabling", "using",
               "building", "towards", "toward", "improving", "enhancing", "accelerating", "scaling", "making",
               "multi", "hardware"}
FRAG_END |= {"large", "multi", "aligned", "extension", "access", "consensus", "transformer_based"}
UMB_GENERIC |= {"latency", "powered", "building", "architecture", "workflows", "workflow", "protocols", "context",
                "aware", "non", "deterministic", "extension", "driven", "access", "coding", "multi"}

# обрыв на прилагательном («safeguarding multimodal», «long horizon autonomous», «device generative») и заголовочные
# глаголы в начале («unlocking high», «parsing millions»): в базе 2 052 таких было больше сотни
FRAG_END |= {"multimodal", "embodied", "autonomous", "generative", "conversational", "scientific", "mathematical",
             "cryptographic", "visual", "spatial", "geometric", "kinematic", "dexterous", "personalized", "predictive",
             "generated", "controlled", "conditioned", "informed", "neutral", "early", "en", "efficiently", "exposed",
             "social", "cyber", "mental", "romantic", "detailed", "heuristic", "offensive", "institutional", "systemic",
             "regulated", "explicit", "diverse", "generalized", "consistent", "foundational", "minimal", "verifiable",
             "unified", "generalizable", "convolutional", "interactive", "hidden", "fractional", "high", "millions"}
FRAG_START |= {"unlocking", "parsing", "accelerate", "playing", "facing", "turning", "powering", "modal", "rethinking",
               "measuring", "strengthen", "accurate", "utilizing", "understanding", "connecting", "bridging", "running",
               "teaming", "sensing", "inferring", "preventing", "cost", "bandwidth", "tailed"}
FRAG_END |= {"world", "bypass", "scoped"}
UMB_GENERIC |= {"infra", "malicious", "offensive", "vulnerability", "vulnerabilities"}

# устойчивые термины, которые начинаются со слов из FRAG_START, но обрывками не являются
FRAG_OK_START = ("time series", "zero shot", "end to end", "real world asset", "scale out", "order book",
                 "source code", "loop closure", "cost aware", "cost effective", "cost efficient")


def is_fragment(phrase: str) -> bool:
    p = phrase.replace("_", " ")
    if p.startswith(FRAG_OK_START):
        return False
    w = p.split()
    if w[0] == "multi" and len(w) >= 3:                 # «multi agent …», «multi modal …» — обычные термины
        return w[-1] in FRAG_END
    if w[0] == "hardware" and len(w) >= 3:
        return w[-1] in FRAG_END
    return w[0] in FRAG_START or w[-1] in FRAG_END


def is_umbrella(phrase: str) -> bool:
    w = phrase.replace("_", " ").split()
    return 2 <= len(w) <= 4 and any(x in UMB_HEADS for x in w) and all(x in UMB_HEADS or x in UMB_GENERIC for x in w)


AREAS = ("Edge", "Защита ИИ", "Индустриальный ИИ", "Инфраструктура ИИ", "Роботы", "Финтех")
BLEND_KEEP_AREA = 0.6

# своя принадлежность к областям ТЗ вместо метки YandexGPT Lite: Lite ставила «Индустриальный ИИ» на любой
# корпоративный ИИ. Область подтверждается предметным словом (англ. термин, русское название, описание) или долей
# работ в рубриках области; для промышленности метка Lite без подтверждения не считается.
AREA_LEXICON = {
    # короткие слова — только целиком (\b…\b), основы — с начала слова
    "Индустриальный ИИ": r"\b(plc|plcs|scada|mes|cnc|iiot|ot|p&id|hmi)\b|\b(weld|machining|manufactur|factor(y|ies)|"
                         r"industrial|industry 4|predictive maintenance|condition monitoring|fault diagnos|"
                         r"quality inspection|visual inspection|process control|production line|assembly line|"
                         r"additive manufacturing|3d printing|procurement|semiconductor fab|power plant|oil and gas|"
                         r"mining (industry|operations|equipment|site)|mine site|steel|chemical plant|machine tool)|"
                         r"производств|завод|промышлен|станк|сварк|\bплк\b|цех|техобслуж|технологическ[а-я]* процесс",
    "Инфраструктура ИИ": r"\b(hbm|cxl|ucie|euv|dram|sram|gpu|gpus|tpu|fet|nvlink|infiniband|kv)\b|\b(chip|chiplet|"
                         r"memory|interconnect|optic|photonic|switch|data ?cent|datacent|cooling|accelerator|wafer|"
                         r"packaging|lithograph|inference serving|llm serving|cluster|ethernet|transistor|"
                         r"compute.in.memory|in.memory comput|neuromorphic|superconduct)|"
                         r"\bчип|памят|дата-центр|центр(ов|ы)? обработки данных|охлажд|ускорител|интерконнект|фотон|транзистор",
    "Edge": r"\b(iot|mcu|npu|npus|slm|slms|tinyml)\b|\b(on.device|edge (ai|llm|inference|comput|device)|smartphone|"
            r"microcontroller|embedded|wearable|in.sensor|quantiz|local (llm|large language|ai)|small language model|"
            r"mobile (llm|ai|inference|device))|смартфон|на устройств|датчик|периферийн|встраиваем|носим",
    "Роботы": r"\b(uav|uavs|vla)\b|\b(robot|humanoid|manipulat|locomotion|gripper|teleoperat|embodied|quadruped|"
              r"drone|autonomous vehicle|actuator|dexterous)|робот|гуманоид|манипул|беспилот",
    "Защита ИИ": r"\b(jailbreak|prompt injection|adversarial|red team|watermark|backdoor|poison|privacy|secur|"
                 r"attack|guardrail|alignment|unlearning|deepfake|provenance|zero.knowledge|confidential|"
                 r"safety|auditing|supply chain)|безопасн|атак|защит|водян|утечк|приватн",
    # «токен» и «credit» без финансового контекста — это токены и credit assignment языковых моделей, не финтех
    "Финтех": r"\b(aml|kyc|defi|cbdc|rwa)\b|\b(payment|bank|fintech|credit (scor|risk|card|default|union|decision)|"
              r"creditworth|loan|lending|insur|fraud|commerce|custody|rug pull|address poisoning|stablecoin|tokeniz(ed|ation) (asset|money|fund|deposit|securit|"
              r"bond|real)|(asset|rwa|real world asset) tokeniz|blockchain|crypto|trading|market maker|wallet|deposit|"
              r"financ|(digital|real world|crypto|financial) assets?|asset (management|manager|pricing))|"
              r"\b(платеж|банк|кредитн|кредитован|страхов|мошеннич|финанс|крипт|стейблкоин|"
              r"токенизац[а-я]* (актив|ден|фонд|депозит|ценн)|бирж)",
}


def our_areas(row, quota: bool = False) -> list[str]:
    """Своя принадлежность к областям ТЗ. quota=True — прежнее мягкое правило для финтеха (метка Lite или рубрика),
    только для квоты при сборке: строгое правило переносило записи в «другое» и сдвигало процентный порог там."""
    import re as _re
    text = " ".join(str(row.get(f) or "") for f in ("phrase", "name_ru", "description_ru")).replace("_", " ").lower()
    llm = set(row.get("areas") if row.get("areas") is not None else [])
    out = []
    name = " ".join(str(row.get(f) or "") for f in ("phrase", "name_ru")).replace("_", " ").lower()
    for a in AREAS:
        # финансы в описании часто лишь пример применения («в финансах, медицине…») — для финтеха только название
        lex = bool(_re.search(AREA_LEXICON[a], name if a == "Финтех" and not quota else text))
        rub = float(row.get("share:" + a) or 0) >= 0.30
        if a == "Индустриальный ИИ":
            ok = (lex or (rub and a in llm)) and not _re.search(r"medical|patient|clinical|медицин|пациент", text)
        elif a == "Финтех" and quota:
            ok = lex or rub or (a in llm)
        elif a in ("Edge", "Финтех"):                   # метка Lite без предметного слова — не Edge и не финтех
            ok = lex or (rub and a in llm)              # (Lite ставила «Финтех» на токены и credit assignment LLM)
        else:
            ok = lex or rub or (a in llm)
        if ok:
            out.append(a)
    return out


def main_area(areas) -> str:
    """Главная область для квоты: первая из областей ТЗ (в порядке узкие → широкие), иначе «другое»."""
    order = ("Инфраструктура ИИ", "Edge", "Индустриальный ИИ", "Роботы", "Финтех", "Защита ИИ")
    s = set(areas if areas is not None else [])
    return next((a for a in order if a in s), "другое")


def candidates() -> pd.DataFrame:
    s = Searcher()
    p = s.pool.copy()
    p["E_row"] = np.arange(len(p))
    # «обновление»: термин старый (первое появление до MATURE_YEAR), но снова ускоряется — доля работ 2024–2026
    # не меньше RENEWAL_ACC от доли 2020–2022 и не меньше RENEWAL_MIN работ. Эксперты заказчика считают такие
    # технологии слабыми сигналами (нейроморфные чипы в edge, роевые роботы, ловкие кисти): без этого правило
    # зрелости отсекало бы старые, но снова ускоряющиеся термины.
    from .search import acceleration
    acc, n_rec = acceleration(p.key)
    p["acceleration"], p["docs_recent"] = acc.values, n_rec.values
    old = p.first_year <= MATURE_YEAR
    p["renewal"] = old & (p.acceleration >= RENEWAL_ACC) & (p.docs_recent >= RENEWAL_MIN)
    p = p[~old | p.renewal]
    n_mature = len(p)
    p = p[~p.phrase.map(is_fragment) & ~p.phrase.map(is_umbrella)]
    # отбор кандидатов — по смеси двух моделей (экспертной и исторической), если они посчитаны; иначе — по обученному
    # баллу. По старому баллу выпадали сигналы вроде орбитальных дата-центров и AI SOC.
    em, hm = REG / "expert_scored.parquet", REG / "history_scored.parquet"
    order = "learned"
    if em.exists() and hm.exists():
        p = p.merge(pd.read_parquet(em, columns=["key", "expert_p"]).drop_duplicates("key"), on="key", how="left")
        p = p.merge(pd.read_parquet(hm, columns=["key", "history_p"]).drop_duplicates("key"), on="key", how="left")
        p["cand_blend"] = (BLEND_W_HISTORY * p.history_p.rank(pct=True)
                           + (1 - BLEND_W_HISTORY) * p.expert_p.rank(pct=True))
        order = "cand_blend"
    p = p.sort_values(order, ascending=False)
    # склейка вариантов: жадно по убыванию балла
    E = s.E
    keep, alias = [], {}
    kept_rows: list[int] = []
    for r in p.itertuples():
        if kept_rows:
            sims = E[kept_rows] @ E[r.E_row]
            j = int(sims.argmax())
            if sims[j] >= ALIAS_SIM:
                alias.setdefault(keep[j], []).append(r.phrase)
                continue
        keep.append(r.Index)
        kept_rows.append(r.E_row)
    p = p.loc[keep].copy()
    p["aliases"] = [alias.get(i, []) for i in p.index]
    parts = []
    fresh, ren = p[~p.renewal], p[p.renewal]           # у «обновлений» свои места — не вытесняют свежие сигналы
    for a in AREAS:
        m = fresh.areas.map(lambda l: a in l) | (fresh["share:" + a] >= 0.30)
        parts.append(fresh[m].nlargest(PER_AREA, order))
        mr = ren.areas.map(lambda l: a in l) | (ren["share:" + a] >= 0.30)
        parts.append(ren[mr].nlargest(RENEWAL_PER_AREA, order))
    parts.append(fresh.nlargest(OTHER, order))
    parts.append(ren.nlargest(RENEWAL_OTHER, order))
    c = pd.concat(parts).drop_duplicates("key")
    c = c.drop(columns=["E_row"])
    c.to_parquet(CAND)
    print(f"пул {len(s.pool)} → не зрелые {n_mature} → без обрывков и рубрик «ИИ + слово» {len(p) + sum(map(len, alias.values()))} → после склейки "
          f"вариантов {len(p)} → кандидатов в базу {len(c)}")
    return c


MODEL = "yandexgpt/rc"
BATCH = 10
KINDS_OK = ("метод", "архитектура", "устройство", "материал", "протокол", "класс решений")
KINDS_BAD = ("задача", "рубрика", "продукт", "обрывок", "не технология")
KINDS_HARD_BAD = ("продукт", "обрывок", "не технология")
PROMPT = """Ты составляешь базу технологических слабых сигналов для банка. Ниже кандидаты, найденные в корпусе научных
работ, Hacker News, стартапов Y Combinator и прессы, с измерениями из корпуса. Для КАЖДОГО кандидата выбери ОДИН тип:
  метод — алгоритм или подход с устойчивым названием («indirect prompt injection», «whole body control»);
  архитектура — устройство системы или модели («edge cloud llm inference»);
  устройство — аппаратное решение, чип, датчик, робот определённого класса;
  материал — вещество или структура с новыми свойствами;
  протокол — стандарт, интерфейс, протокол («universal chiplet interconnect express»);
  класс решений — новый класс продуктов или сервисов многих компаний («agentic payments», «robot foundation models»);
  задача — цель, проблема или показатель, а не способ («оформление кредита», «производительность в производстве»);
  рубрика — широкая область «ИИ + отрасль/инфраструктура» («безопасность LLM», «кластеры ИИ», «промышленный edge»);
  продукт — продукт или бренд одной компании («Llama Guard», «Ryzen AI»);
  обрывок — не законченное название, кусок фразы («hardware co», «engineering enables»);
  не технология — всё прочее.
Будь строгим: если название можно понять только как тему или задачу — это рубрика или задача.
Также оцени стадию: early — работ немного и их число растёт в последние 2–3 года; mature — много работ давно или
массовое применение; fading — пик прошёл. Опирайся на измерения.
Верни СТРОГО JSON-массив, по объекту на номер:
[{"i": <номер>, "phrase": "<английская фраза дословно>", "kind": "<тип>", "stage": "early|mature|fading",
  "reason": "<до 12 слов по-русски>"}]

Кандидаты:
{items}"""


def check(max_calls: int) -> None:
    from .adjudicate import _norm, measurements
    from .llm_assess import load_env
    import requests
    load_env()
    uri = f"gpt://{os.environ['YANDEX_FOLDER_ID']}/{MODEL}"
    c = pd.read_parquet(CAND)
    done = {json.loads(l)["phrase"] for l in CHECK.read_text(encoding="utf-8").splitlines()} if CHECK.exists() else set()
    lab = set(_labels().phrase)
    c = c.assign(ph=c.phrase.str.replace("_", " "), labeled=c.phrase.isin(lab))
    c = c[~c.ph.isin(done)].sort_values(["labeled", "learned"], ascending=[False, False])   # размеченные — первыми
    line = measurements(REG)["line"]
    calls = tokens = 0
    for i in range(0, len(c), BATCH):
        if calls >= max_calls:
            break
        items = c.iloc[i:i + BATCH]
        listing = "\n".join(f"{j}. {r.ph} — {r.name_ru}: {r.description_ru}\n   {line(r.key)}"
                             for j, r in enumerate(items.itertuples(), 1))
        for attempt in range(3):
            calls += 1
            try:
                r = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion", timeout=120,
                                  headers={"Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}"},
                                  json={"modelUri": uri, "completionOptions": {"temperature": 0.1, "maxTokens": 3000},
                                        "messages": [{"role": "user", "text": PROMPT.replace("{items}", listing)}]})
                r.raise_for_status()
                res = r.json()["result"]
                tokens += int((res.get("usage") or {}).get("totalTokens", 0))
                m = re.search(r"\[.*\]", res["alternatives"][0]["message"]["text"], re.S)
                got = 0
                with open(CHECK, "a", encoding="utf-8") as fh:
                    for x in (json.loads(m.group(0)) if m else []):
                        k = x.get("i") if isinstance(x, dict) else None
                        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= len(items):
                            continue
                        ph = items.iloc[k - 1].ph
                        if _norm(x.get("phrase")) != _norm(ph) or x.get("kind") not in KINDS_OK + KINDS_BAD:
                            continue
                        fh.write(json.dumps({"phrase": ph, "kind": x["kind"], "stage": x.get("stage"),
                                             "reason": str(x.get("reason") or "")[:120]}, ensure_ascii=False) + "\n")
                        got += 1
                break
            except Exception as e:
                print("  повтор:", str(e)[:100], flush=True)
                time.sleep(5 * (attempt + 1))
        print(f"  вызовов {calls}, токенов {tokens:,}, пачка {i // BATCH + 1}: принято {got}/{len(items)}".replace(",", " "),
              flush=True)
    print(f"итого вызовов {calls}, токенов {tokens:,} -> {CHECK}".replace(",", " "))


STRICT = REG / "base_strict.jsonl"
SAMPLE = REG / "base_sample_labels.csv"          # ручная разметка 119 случайных записей базы — только для оценки
STRICT_OK = ("технология",)
STRICT_KINDS = ("технология", "теория", "задача", "общий термин", "обрывок", "продукт", "зрелое")
STRICT_PROMPT = """Ты ведёшь технологический радар банка: список конкретных НОВЫХ технологий, за которыми стоит следить.
Кандидаты уже прошли первичный отбор, но примерно половина из них в радар не годится. Для КАЖДОГО выбери ОДНО:
  технология — конкретная новая технология с устойчивым названием, которую можно внедрять, покупать или
    разрабатывать: метод ИИ, архитектура модели, класс устройств, протокол, класс продуктов многих компаний.
    Примеры: «speculative decoding», «mixture of experts», «reconfigurable intelligent surface»,
    «post-quantum cryptography», «humanoid robots», «stablecoin payments», «digital twin of network».
  теория — результат или метод статистики, эконометрики, математики, теоретической физики или финансовой математики,
    нужный учёным, а не рынку. Примеры: «copula models», «difference in differences», «stochastic volatility»,
    «regret bounds», «capacity of gaussian channel».
  задача — прикладная задача или применение, а не способ её решения. Примеры: «прогноз урожая», «распознавание
    номеров», «оптимизация склада», «оценка кредитного риска».
  общий термин — широкое понятие без конкретики. Примеры: «ai systems», «data pipelines», «model training»,
    «feature extraction», «multi scale features».
  обрывок — незаконченное название, кусок фразы или словосочетание, которое так не пишут.
  продукт — продукт, бренд, библиотека, бенчмарк или датасет одной организации.
  зрелое — давно устоявшееся и массово применяемое.
Будь строгим: «технология» — только если аналитик уверенно впишет это в радар как отдельную строку.
Верни СТРОГО JSON-массив, по объекту на номер:
[{"i": <номер>, "phrase": "<английская фраза дословно>", "kind": "<одно из семи>", "reason": "<до 10 слов по-русски>"}]

Кандидаты:
{items}"""


def _strict_file(model: str):
    return STRICT if model == "pro" else REG / f"base_strict_{model}.jsonl"


def strict(max_calls: int, only_labeled: bool, model: str = "pro") -> None:
    """Вторая, строгая проверка по записям базы: model="pro" (yandexgpt/rc) или "lite" (yandexgpt-5-lite).
    Кэш у каждой модели свой: base_strict.jsonl / base_strict_lite.jsonl."""
    from .adjudicate import _norm, measurements
    from .llm_assess import load_env
    import requests
    from wsignals.decompose import RUNTIME_YANDEX_MODEL
    load_env()
    uri = f"gpt://{os.environ['YANDEX_FOLDER_ID']}/{MODEL if model == 'pro' else RUNTIME_YANDEX_MODEL}"
    out = _strict_file(model)
    b = pd.read_parquet(BASE)
    b["ph"] = b.phrase.str.replace("_", " ")
    done = {json.loads(l)["phrase"] for l in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    lab = set(pd.read_csv(SAMPLE).phrase) | set(_labels().phrase.str.replace("_", " "))
    b["labeled"] = b.ph.isin(lab)
    if only_labeled:
        b = b[b.labeled]
    b = b[~b.ph.isin(done)].sort_values(["labeled", "learned"], ascending=[False, False])
    line = measurements(REG)["line"]
    calls = tokens = 0
    for i in range(0, len(b), BATCH):
        if calls >= max_calls:
            break
        items = b.iloc[i:i + BATCH]
        listing = "\n".join(f"{j}. {r.ph} — {r.name_ru}: {r.description_ru}\n   {line(r.key)}"
                             for j, r in enumerate(items.itertuples(), 1))
        got = 0
        for attempt in range(3):
            calls += 1
            try:
                r = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion", timeout=120,
                                  headers={"Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}"},
                                  json={"modelUri": uri, "completionOptions": {"temperature": 0.1, "maxTokens": 3000},
                                        "messages": [{"role": "user", "text": STRICT_PROMPT.replace("{items}", listing)}]})
                r.raise_for_status()
                res = r.json()["result"]
                tokens += int((res.get("usage") or {}).get("totalTokens", 0))
                m = re.search(r"\[.*\]", res["alternatives"][0]["message"]["text"], re.S)
                rows = json.loads(m.group(0)) if m else []
                with open(out, "a", encoding="utf-8") as fh:
                    for x in rows:
                        k = x.get("i") if isinstance(x, dict) else None
                        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= len(items):
                            continue
                        ph = items.iloc[k - 1].ph
                        if _norm(x.get("phrase")) != _norm(ph) or x.get("kind") not in STRICT_KINDS:
                            continue
                        fh.write(json.dumps({"phrase": ph, "kind": x["kind"], "reason": str(x.get("reason") or "")[:120]},
                                            ensure_ascii=False) + "\n")
                        got += 1
                break
            except Exception as e:
                print("  повтор:", str(e)[:100], flush=True)
                time.sleep(5 * (attempt + 1))
        print(f"  вызовов {calls}, токенов {tokens:,}, пачка {i // BATCH + 1}: принято {got}/{len(items)}".replace(",", " "),
              flush=True)
    print(f"итого вызовов {calls}, токенов {tokens:,} -> {out}".replace(",", " "))


def strict_eval() -> None:
    for model in ("pro", "lite"):
        if _strict_file(model).exists():
            print(f"\n=== {model} ===")
            _strict_eval(_strict_file(model))


def _strict_eval(path) -> None:
    st = {json.loads(l)["phrase"]: json.loads(l)["kind"] for l in path.read_text(encoding="utf-8").splitlines()}
    t = pd.read_csv(SAMPLE).drop_duplicates("phrase")
    t = t[t.phrase.isin(st)]
    t["kind"] = t.phrase.map(st)
    ok = t.kind.isin(STRICT_OK)
    print(f"выборка базы ({len(t)}): технологий до проверки {t.y.mean():.0%}; после — оставлено {int(ok.sum())}, "
          f"из них технологий {t[ok].y.mean():.0%}; потеряно технологий {int((~ok & (t.y == 1)).sum())} из {int(t.y.sum())}")
    print(pd.crosstab(t.kind, t.y).to_string())
    lab = _labels()
    lab["phrase"] = lab.phrase.str.replace("_", " ")
    lab = lab[lab.phrase.isin(st) & (lab.y == 1)]
    lost = [p for p in lab.phrase if st[p] not in STRICT_OK]
    print(f"известные сигналы в базе: {len(lab)}, отклонено строгой проверкой {len(lost)}: {lost}")


THEORY_MAX = 0.5
# академия связи и управления: метаповерхности, MIMO, модельное прогнозирующее управление, диагностика, дефекты.
# Балл сигнальности их не отсекает (в arXiv они быстро растут); отличает рубрика и отсутствие в бизнес-слоях.
# Убираем, если ≥ ACADEMIC_MAX работ в этих рубриках и темы нет ни в HN, ни в YC, ни в прессе.
# На базе v3: убрано 280, из них 113 из 162 записей «мусорных семейств»; из 45 известных сигналов задет 1.
ACADEMIC_RUBRICS = ("ax:eess.SP", "ax:cs.IT", "ax:eess.SY", "ax:math.OC", "ax:eess.IV", "oa:2207", "oa:1705")
ACADEMIC_MAX = 0.4
# вторая ступень: с электротехникой OpenAlex (2208) — там же метаповерхности и MIMO, но и чипы; поэтому порог выше
# и железо (устройство, протокол, материал по проверке Pro) не трогаем: UCIe, CFET, EUV с высокой апертурой остаются.
ACADEMIC_RUBRICS_EE = ACADEMIC_RUBRICS + ("oa:2208",)
ACADEMIC_EE_MAX = 0.6
HARDWARE_KINDS = ("устройство", "протокол", "материал")
# академия количественных финансов и эконометрики: GARCH, опционы Хестона, портфели, difference-in-differences.
# Растут в arXiv q-fin, но это давние задачи без выхода в рынок. Убираем, если ≥ FIN_ACADEMIC_MAX работ в этих
# рубриках и темы нет ни в HN, ни в YC, ни в прессе. На базе 1 972: убрано 76; tokenised money, RWA, stablecoin,
# AI trading (есть бизнес-слои) остаются.
FIN_ACADEMIC_RUBRICS = ("ax:q-fin.ST", "ax:q-fin.PR", "ax:q-fin.MF", "ax:q-fin.RM", "ax:q-fin.PM", "ax:q-fin.CP",
                        "ax:q-fin.TR", "ax:q-fin.GN", "ax:econ.EM", "ax:econ.GN", "ax:econ.TH", "oa:2003", "oa:2002")
FIN_ACADEMIC_MAX = 0.4
# итоговый балл — смесь двух моделей по рангам среди кандидатов базы:
#   70 % экспертная модель (registry/expert_model.py: градиентный бустинг на ≈1 000 размеченных технологиях),
#   30 % модель взлёта на истории (registry/history_model.py: признаки на 2021, метка — взлёт к 2024–2026).
# На экспертной разметке (вне обучения): ROC AUC 0,80 против 0,785 у экспертной и 0,735 у исторической.
# В базе остаются верхние BLEND_KEEP кандидатов: по предсказаниям вне обучения сохраняется 72 % размеченных
# сигналов, доля сигналов среди оставшихся размеченных 32 % (было 15 %).
BLEND_W_HISTORY = 0.3
BLEND_KEEP = 0.5
# явное семейство академии 6G, которое проверка Pro называла «устройством»: без бизнес-слоёв — убрать
ACADEMIC_FAMILY = r"intelligent surface|reflecting surface|metasurface|\bris\b|\bmimo\b|cell free"


def rubric_share(keys: pd.Series, rubrics) -> np.ndarray:
    g = np.load(REG / "groups.npz")
    names = [str(x) for x in g["groups"]]
    grp = g["grp"]
    rows = [names.index(r) for r in rubrics if r in names]
    sci = pd.read_parquet(REG / "all_phrases.parquet", columns=["key", "eid"])
    e = keys.map(dict(zip(sci.key, sci.eid)))
    out = np.full(len(keys), np.nan)
    m = e.notna().values
    idx = e[m].astype(int).values
    tot = grp[:, idx].sum(axis=0)
    out[m] = np.where(tot >= 5, grp[rows][:, idx].sum(axis=0) / np.maximum(tot, 1), np.nan)
    return out


def theory_share(keys: pd.Series) -> np.ndarray:
    """Доля работ сущности в теоретических рубриках arXiv (математика, статистика кроме stat.ML, эконометрика,
    финансовая математика, физика высоких энергий и астрофизика). NaN, если работ по рубрикам меньше 5.
    Порог THEORY_MAX: на базе убирает 69 записей (копулы, diff-in-diff, модель Хестона), из известных сигналов — 1."""
    g = np.load(REG / "groups.npz")
    names = [str(x) for x in g["groups"]]
    grp = g["grp"]
    th = [i for i, n in enumerate(names)
          if (n.startswith("ax:math") and n != "ax:math.OC") or n.startswith(("ax:econ", "ax:hep", "ax:astro", "ax:nucl"))
          or n in ("ax:stat.ME", "ax:stat.ST", "ax:stat.AP", "ax:stat.CO", "ax:stat.OT", "ax:math-ph", "ax:funct-an",
                   "ax:alg-geom", "ax:gr-qc", "ax:q-fin.MF", "ax:q-fin.PR", "ax:q-fin.ST", "ax:q-fin.PM",
                   "ax:q-fin.RM", "ax:q-fin.CP")]
    sci = pd.read_parquet(REG / "all_phrases.parquet", columns=["key", "eid"])
    e = keys.map(dict(zip(sci.key, sci.eid)))
    out = np.full(len(keys), np.nan)
    m = e.notna().values
    idx = e[m].astype(int).values
    tot = grp[:, idx].sum(axis=0)
    out[m] = np.where(tot >= 5, grp[th][:, idx].sum(axis=0) / np.maximum(tot, 1), np.nan)
    return out


def expert_score(b: pd.DataFrame) -> pd.DataFrame:
    """Экспертный балл: средний ранг четырёх признаков того, что эксперты заказчика называют слабым сигналом.
      org_sim  — близость к 100 сигналам организаторов (среднее трёх ближайших): образец того, что эксперты признают;
      biz      — в скольких бизнес-слоях есть тема (Hacker News, YC, пресса): переход из науки в рынок;
      learned  — обученный балл сигнальности (рост, темп, разнообразие источников);
      first_year — свежесть.
    Проверка на отложенной половине организаторов (близость считалась только к нечётным): AUC 0,83 против 0,71 у
    одного обученного балла. В продукте близость считается ко всем 100 — это описание цели заказчиком."""
    from sentence_transformers import SentenceTransformer
    import torch
    from .search import MODEL as EMB
    from .semantic import queries
    m = SentenceTransformer(EMB, device="mps" if torch.backends.mps.is_available() else "cpu")
    m.max_seq_length = 64
    E = m.encode((b.name_ru + " — " + b.description_ru).tolist(), batch_size=128, normalize_embeddings=True)
    S = m.encode([f"{q['ru']} ({q['en']})" for q in queries()], normalize_embeddings=True)
    b = b.copy()
    b["org_sim"] = np.sort(E @ S.T, axis=1)[:, -3:].mean(axis=1)
    b["biz"] = sum((b[c].fillna(0) > 0).astype(int) for c in ("biz_log_hn", "biz_log_yc", "biz_log_press"))
    r = lambda c: b[c].fillna(-1).rank(pct=True)
    b["expert"] = (r("org_sim") + r("biz") + r("learned") + r("first_year")) / 4
    return b


def build() -> pd.DataFrame:
    """База: кандидаты, которых Pro отнесла к технологиям (KINDS_OK). Стадию от Pro храним, но не отсеиваем по ней:
    зрелость уже отсечена данными (MATURE_YEAR), а Pro называла «mature» и ранние технологии (co-packaged optics)."""
    c = pd.read_parquet(CAND)
    c["areas_llm"] = c.areas                            # метка YandexGPT Lite — для истории
    c["areas"] = [our_areas(r) for r in c.to_dict("records")]
    c["areas_quota"] = [our_areas({**r, "areas": r["areas_llm"]}, quota=True) for r in c.to_dict("records")]
    ch = {}
    for l in CHECK.read_text(encoding="utf-8").splitlines():
        x = json.loads(l)
        ch[x["phrase"]] = x
    ph = c.phrase.str.replace("_", " ")
    c["kind"] = ph.map(lambda p: ch.get(p, {}).get("kind"))
    c["stage_llm"] = ph.map(lambda p: ch.get(p, {}).get("stage"))
    c["check_reason"] = ph.map(lambda p: ch.get(p, {}).get("reason"))
    n_checked = int(c.kind.notna().sum())
    c["theory_share"] = theory_share(c.key)
    c = c[~c.phrase.map(is_fragment)]
    # проверка Pro отсеивает только продукты, обрывки и «не технологию»; «задачу» и «рубрику» не отсеивает — там она
    # ошибалась (миграция на постквантовую криптографию, emergent misalignment); это решает экспертная модель.
    # Кандидаты без проверки Pro (добавлены отбором по смеси моделей) проходят.
    b = c[~c.kind.isin(KINDS_HARD_BAD) & ~c.phrase.map(is_fragment) & ~c.phrase.map(is_umbrella)
          & ~(c.theory_share >= THEORY_MAX)].copy()
    b = expert_score(b)
    b["academic_share"] = rubric_share(b.key, ACADEMIC_RUBRICS)
    n0 = len(b)
    b["academic_share_ee"] = rubric_share(b.key, ACADEMIC_RUBRICS_EE)
    b = b[~((b.academic_share >= ACADEMIC_MAX) & (b.biz == 0))]
    b = b[~((b.academic_share_ee >= ACADEMIC_EE_MAX) & (b.biz == 0) & ~b.kind.isin(HARDWARE_KINDS))]
    b = b[~(b.phrase.str.replace("_", " ").str.contains(ACADEMIC_FAMILY) & (b.biz == 0))].copy()
    b["fin_academic_share"] = rubric_share(b.key, FIN_ACADEMIC_RUBRICS)
    b = b[~((b.fin_academic_share >= FIN_ACADEMIC_MAX) & (b.biz == 0))].copy()
    em, hm = REG / "expert_scored.parquet", REG / "history_scored.parquet"
    if em.exists() and hm.exists():                   # смесь двух моделей: фильтр и порядок выдачи
        s = c[["key"]].merge(pd.read_parquet(em, columns=["key", "expert_p"]).drop_duplicates("key"), on="key",
                             how="left").merge(pd.read_parquet(hm, columns=["key", "history_p"])
                                               .drop_duplicates("key"), on="key", how="left")
        s["blend"] = (BLEND_W_HISTORY * s.history_p.rank(pct=True)
                      + (1 - BLEND_W_HISTORY) * s.expert_p.rank(pct=True))
        # квота внутри области: балл сравнивается с кандидатами своей области, а не со всеми (иначе B2B-темы Edge,
        # «железа» и промышленности без Hacker News и YC проигрывали ИИ-темам и в базу доходило 28–37 % их кандидатов)
        ren = c.renewal if "renewal" in c else pd.Series(False, index=c.index)
        s = s.merge(c[["key", "areas"]].assign(bucket=c.areas_quota.map(lambda l: main_area(l)) + ren.map(
            lambda r: " · обновление" if r else "")), on="key", how="left")
        s["blend_in_area"] = s.groupby("bucket").blend.rank(pct=True)
        b = b.drop(columns=["expert_p", "history_p", "blend", "blend_in_area", "bucket"], errors="ignore").merge(
            s.drop(columns=["areas"]), on="key", how="left")
        b["expert_rank"] = b["expert"]
        b["expert"] = b.blend
        n1 = len(b)
        keep = b.bucket.map(lambda a: BLEND_KEEP if a.startswith("другое") else BLEND_KEEP_AREA)
        b = b[b.blend_in_area >= 1 - keep].copy()
        print(f"смесь моделей (квота внутри области: {BLEND_KEEP_AREA:.0%} в областях ТЗ, {BLEND_KEEP:.0%} в прочих): "
              f"убрано {n1 - len(b)}; по областям: " + ", ".join(f"{a} {n}" for a, n in b.bucket.value_counts().items()))
    print(f"академия связи и управления без выхода в бизнес-слои: убрано {n0 - len(b)}")
    # паспорта (registry/passports.py): источники сигнала базы не подтвердили событие — балл понижается;
    # подтверждённые составные и рыночные сигналы добавляются в базу отдельными записями
    from . import passports as P
    ps = P.load()
    if ps:
        pid = "base:" + b.key
        verdict = pid.map(lambda i: (ps.get(i) or {}).get("confirmed"))
        b["passport_confirmed"] = verdict
        b.loc[verdict.eq(False), "expert"] = b.loc[verdict.eq(False), "expert"] * P.UNCONFIRMED_FACTOR
        print(f"паспорта сигналов базы: подтверждено {int(verdict.eq(True).sum())}, не подтверждено "
              f"{int(verdict.eq(False).sum())} (балл ×{P.UNCONFIRMED_FACTOR}), без паспорта {int(verdict.isna().sum())}")
    if P.RECORDS.exists():
        extra = pd.read_parquet(P.RECORDS)
        if len(extra):
            b = pd.concat([b, extra.drop(columns=["docs", "growth_rule", "volume"], errors="ignore")], ignore_index=True)
            print(f"добавлено подтверждённых составных и рыночных сигналов: {len(extra)}")
    # экспертная разметка базы (data/labels/base_review.csv): каждую запись просмотрели глазами по критерию заказчика
    # «эксперт признал бы это слабым сигналом». Ярусы: 1 — признан слабым сигналом (идёт в выдачу первым);
    # 2 — общая формулировка или зрелое (добирает выдачу, если записей первого яруса по теме не хватает);
    # обрывки, дубли (лучший вариант остаётся) и «не технология» — вне базы. Записи без разметки (новые) — ярус 2.
    b["tier"] = 2
    if REVIEW.exists():
        rv = pd.read_csv(REVIEW)
        drop = set(rv.loc[rv.verdict.isin(DROP_VERDICTS), "key"])
        n_before = len(b)
        b = b[~b.key.isin(drop)].reset_index(drop=True)
        b.loc[b.key.isin(set(rv.loc[rv.verdict == "keep", "key"])), "tier"] = 1
        print(f"экспертная разметка базы: убрано {n_before - len(b)} (" + ", ".join(
            f"{v} {n}" for v, n in rv[rv.key.isin(drop)].reason.value_counts().items()) + f"); ярус 1 — "
              f"{int((b.tier == 1).sum())}, ярус 2 — {int((b.tier == 2).sum())}")
    b.to_parquet(BASE)
    print(f"кандидатов {len(c)}, проверено Pro {n_checked}, в базе {len(b)}")
    print(b.kind.value_counts().to_string())
    for a in AREAS:
        print(f"  {a:18} {int((b.areas.map(lambda l: a in l) | (b['share:' + a] >= 0.30)).sum())}")
    return b


def _labels() -> pd.DataFrame:
    rows = []
    for f in ("stage2_pool_top15.csv", "adjudicated_top15.csv", "adjudicated_top15_examples.csv",
              "search_top15_labels.csv"):
        t = pd.read_csv(REG / f)
        if f.startswith("stage2"):
            t = t[t.method == "ru_mix"]
        rows.append(t[["phrase", "weak_signal"]].dropna())
    t = pd.concat(rows)
    # сигнал — если хоть раз размечен верным; «не сигнал» — если ни разу (нули-дубли в списке не считаем сигналом)
    return t.groupby("phrase").weak_signal.max().rename("y").reset_index()


def evaluate() -> None:
    lab = _labels()
    pos, neg = set(lab[lab.y == 1].phrase), set(lab[lab.y == 0].phrase)
    s_pool = Searcher().pool
    steps = {"пул": s_pool, "не зрелые": s_pool[~(s_pool.first_year <= MATURE_YEAR)]}
    c = pd.read_parquet(CAND)
    in_c = set(c.phrase) | {a for l in c.aliases for a in l}
    for name, d in steps.items():
        ph = set(d.phrase)
        print(f"{name:12} сигналов {len(pos & ph):>3}/{len(pos)}  не-сигналов {len(neg & ph):>3}/{len(neg)}")
    print(f"{'кандидаты':12} сигналов {len(pos & in_c):>3}/{len(pos)}  не-сигналов {len(neg & in_c):>3}/{len(neg)}"
          f"  (с учётом склеенных вариантов)")
    if CHECK.exists():
        ch = {}
        for l in CHECK.read_text(encoding="utf-8").splitlines():
            x = json.loads(l)
            ch[x["phrase"]] = x
        fit = set(pd.read_csv(REG / "stage2_pool_top15.csv").query("method == 'ru_mix'").phrase)
        lab["phrase"] = lab.phrase.str.replace("_", " ")
        t = lab[lab.phrase.isin(ch)].copy()
        t["kind"] = t.phrase.map(lambda p: ch[p]["kind"])
        t["stage"] = t.phrase.map(lambda p: ch[p]["stage"])
        t["ok_kind"] = t.kind.isin(KINDS_OK)
        t["ok"] = t.ok_kind & t.stage.eq("early")
        for name, part in (("все проверенные", t), ("без примеров промпта", t[~t.phrase.isin(fit)])):
            for col in ("ok_kind", "ok"):
                keep_pos = int((part[col] & (part.y == 1)).sum()); keep_neg = int((part[col] & (part.y == 0)).sum())
                print(f"Pro, {name:22} {'тип' if col == 'ok_kind' else 'тип+early':9}: сигналов оставлено "
                      f"{keep_pos}/{int((part.y == 1).sum())}, не-сигналов {keep_neg}/{int((part.y == 0).sum())}")
        print(pd.crosstab(t.kind, t.y).to_string())
    if BASE.exists():
        b = pd.read_parquet(BASE)
        in_b = set(b.phrase) | {a for l in b.aliases for a in l}
        print(f"{'база':12} сигналов {len(pos & in_b):>3}/{len(pos)}  не-сигналов {len(neg & in_b):>3}/{len(neg)}")
        print("потерянные сигналы:", sorted(pos & in_c - in_b))


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "candidates":
        candidates()
    elif cmd == "check":
        n = int(sys.argv[sys.argv.index("--max-calls") + 1]) if "--max-calls" in sys.argv else 25
        check(n)
    elif cmd == "strict":
        n = int(sys.argv[sys.argv.index("--max-calls") + 1]) if "--max-calls" in sys.argv else 20
        model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "pro"
        strict(n, "--only-labeled" in sys.argv, model)
    elif cmd == "strict-eval":
        strict_eval()
    elif cmd == "build":
        build()
    elif cmd == "eval":
        evaluate()
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
