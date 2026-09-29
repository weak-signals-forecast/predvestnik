"""Составные сигналы (все области ТЗ): «метод ИИ × промышленный объект» (LLM для ПЛК, VLM для контроля качества, …).

    python -m registry.composites          # -> composites.parquet

Промышленный слабый сигнал обычно составной: новый метод ИИ приходит к старому промышленному объекту. Одной фразой
из заголовка он редко выражен, поэтому пары ищутся напрямую: в заголовке и аннотации работы (OpenAlex, arXiv) или
в названии и целях проекта ЕС (CORDIS) одновременно встречаются метод из словаря METHODS и объект из OBJECTS.
По каждой паре — число документов по годам, рост доли 2024–2026 к 2020–2022, первый год, проекты ЕС, до 8 свежих
документов (для карточки и для описания YandexGPT Lite). В базу идут растущие свежие пары (правила ниже).
"""
from __future__ import annotations

import collections
import glob
import gzip
import json
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .corpus import CORPUS, REGISTRY as REG, docs, files

METHODS = {  # канонический метод: (регулярное выражение, русское название)
    "llm": (r"\b(llms?|large language models?|gpt-?4|chatgpt)\b", "LLM"),
    "vlm": (r"\b(vlms?|vision[- ]language models?|multimodal (large )?language models?)\b", "визуально-языковые модели"),
    "foundation model": (r"\bfoundation models?\b", "фундаментальные модели"),
    "generative ai": (r"\bgenerative (ai|artificial intelligence|models?)\b|\bdiffusion models?\b", "генеративный ИИ"),
    "ai agent": (r"\b(ai agents?|llm agents?|agentic|multi-agent llm)\b", "ИИ-агенты"),
    "reinforcement learning": (r"\b(deep )?reinforcement learning\b", "обучение с подкреплением"),
    "digital twin": (r"\bdigital twins?\b", "цифровые двойники"),
    "graph neural network": (r"\bgraph neural networks?\b|\bgnns?\b", "графовые нейросети"),
    "physics-informed ml": (r"\bphysics[- ]informed (neural networks?|machine learning|ml)\b|\bpinns?\b", "физически информированное МО"),
    "federated learning": (r"\bfederated learning\b", "федеративное обучение"),
    "self-supervised learning": (r"\bself[- ]supervised\b", "самообучение без разметки"),
    "edge ai": (r"\b(edge ai|tinyml|on-device (ai|inference|learning))\b", "ИИ на устройстве"),
}
OBJECTS = {
    "plc": (r"\b(plcs?|programmable logic controllers?|iec 61131)\b", "ПЛК"),
    "scada": (r"\bscada\b", "SCADA"),
    "mes": (r"\b(manufacturing execution systems?)\b", "MES"),
    "cnc machining": (r"\b(cnc|machining|machine tools?)\b", "станков и механообработки"),
    "welding": (r"\bwelding\b", "сварки"),
    "additive manufacturing": (r"\b(additive manufacturing|3d printing|laser powder bed)\b", "аддитивного производства"),
    "visual inspection": (r"\b(visual inspection|quality inspection|defect detection|surface defects?)\b", "контроля качества"),
    "predictive maintenance": (r"\b(predictive maintenance|condition monitoring|remaining useful life|fault diagnosis)\b",
                               "предиктивного обслуживания"),
    "process control": (r"\b(process control|industrial control systems?|control loops?)\b", "управления процессами"),
    "p&id": (r"\b(p&ids?|piping and instrumentation)\b", "технологических схем (P&ID)"),
    "production scheduling": (r"\b(production scheduling|job shop|shop floor)\b", "планирования производства"),
    "supply chain": (r"\b(supply chains?|procurement)\b", "цепочек поставок и закупок"),
    "semiconductor manufacturing": (r"\b(semiconductor manufacturing|wafer (fab|inspection|manufacturing)|lithography)\b",
                                    "полупроводникового производства"),
    "industrial robots": (r"\b(industrial robots?|robotic assembly|robot(ic)? manufacturing|cobots?)\b",
                          "промышленных роботов"),
    "industrial time series": (r"\b(industrial (time series|sensor data)|multivariate sensor)\b",
                               "промышленных временных рядов"),
    "power grid": (r"\b(power grids?|smart grids?|power systems?|substations?)\b", "энергосистем"),
    "oil and gas": (r"\b(oil and gas|oil & gas|drilling|reservoir|pipelines?)\b", "нефтегаза"),
    "chemical process": (r"\b(chemical (process|plant|engineering)|process engineering)\b", "химических процессов"),
    "cad": (r"\b(cad|computer-aided design|generative design)\b", "проектирования (CAD)"),
    "warehouse": (r"\b(warehouses?|intralogistics)\b", "складов"),
    # Edge
    "smartphone": (r"\b(smartphones?|mobile phones?|android|iphone)\b", "смартфонов"),
    "microcontroller": (r"\b(microcontrollers?|mcus?|tinyml)\b", "микроконтроллеров"),
    "wearable": (r"\b(wearables?|smartwatch|smart glasses|hearing aids?)\b", "носимых устройств"),
    "vehicle": (r"\b(vehicles?|automotive|in-car|cars)\b", "автомобилей"),
    "pc npu": (r"\b(npus?|ai pcs?|laptops?)\b", "ПК и NPU"),
    # Финтех
    "payments": (r"\b(payments?|checkout|remittances?)\b", "платежей"),
    "kyc aml": (r"\b(kyc|aml|anti-money laundering|know your customer)\b", "KYC и AML"),
    "credit scoring": (r"\b(credit scoring|credit risk|lending|loans?)\b", "кредитования и скоринга"),
    "fraud": (r"\b(fraud|scams?)\b", "антифрода"),
    "insurance": (r"\b(insurance|underwriting|claims)\b", "страхования"),
    "banking": (r"\b(banks?|banking|core banking)\b", "банков"),
    "stablecoin": (r"\b(stablecoins?|tokenized|tokenization|cbdc|deposit tokens?)\b", "токенизации и стейблкоинов"),
    # Защита ИИ
    "mcp tools": (r"\b(mcp|model context protocol|tool use|tool calling)\b", "MCP и инструментов агентов"),
    "agent memory": (r"\b(agent memory|memory poisoning|long-term memory)\b", "памяти агентов"),
    "browser": (r"\b(browsers?|web agents?|computer use)\b", "браузерных агентов"),
    "identity": (r"\b(identity|authentication|authorization|access control|permissions)\b", "идентичности и прав доступа"),
    "model supply chain": (r"\b(model supply chain|model signing|sbom|provenance)\b", "цепочки поставок моделей"),
    # Роботы
    "humanoid": (r"\b(humanoids?)\b", "гуманоидов"),
    "surgery": (r"\b(surgical|surgery)\b", "хирургии"),
    "agriculture": (r"\b(agricultur\w*|farming|harvest\w*)\b", "сельского хозяйства"),
    "teleoperation": (r"\b(teleoperation|teleoperated)\b", "телеуправления"),
}
AREA_OF_OBJECT = {**{o: "Индустриальный ИИ" for o in ("plc", "scada", "mes", "cnc machining", "welding",
                     "additive manufacturing", "visual inspection", "predictive maintenance", "process control", "p&id",
                     "production scheduling", "supply chain", "semiconductor manufacturing", "industrial robots",
                     "industrial time series", "power grid", "oil and gas", "chemical process", "cad", "warehouse")},
                  **{o: "Edge" for o in ("smartphone", "microcontroller", "wearable", "vehicle", "pc npu")},
                  **{o: "Финтех" for o in ("payments", "kyc aml", "credit scoring", "fraud", "insurance", "banking",
                                            "stablecoin")},
                  **{o: "Защита ИИ" for o in ("mcp tools", "agent memory", "browser", "identity", "model supply chain")},
                  **{o: "Роботы" for o in ("humanoid", "surgery", "agriculture", "teleoperation")}}
_M = {k: re.compile(v[0], re.I) for k, v in METHODS.items()}
_O = {k: re.compile(v[0], re.I) for k, v in OBJECTS.items()}
YEARS = list(range(2016, 2027))
RECENT, BASE = (2024, 2025, 2026), (2020, 2021, 2022)


def _pairs(text: str) -> list[tuple[str, str]]:
    ms = [m for m, rx in _M.items() if rx.search(text)]
    if not ms:
        return []
    os_ = [o for o, rx in _O.items() if rx.search(text)]
    return [(m, o) for m in ms for o in os_]


def _file(args):
    source, path = args
    cnt, total, ex = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    for d in docs(source, path):
        y = d.get("year")
        if y not in YEARS:
            continue
        total[y] += 1
        for pr in _pairs((d.get("title") or "") + " . " + (d.get("text") or "")):
            cnt[(pr, y)] += 1
            if y >= 2024 and len(ex[pr]) < 3:
                ex[pr].append({"title": (d.get("title") or "")[:200], "year": y, "url": d.get("url"),
                               "source": source})
    return cnt, total, ex


def _cordis(path):
    cnt, ex = collections.Counter(), collections.defaultdict(list)
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if not n.startswith("project-"):
                continue
            try:
                d = json.loads(z.read(n))
            except Exception:
                continue
            y = str(d.get("startDate") or "")[:4]
            if not y.isdigit() or int(y) not in YEARS:
                continue
            for pr in _pairs(" . ".join(str(d.get(f) or "") for f in ("title", "objective", "keywords"))):
                cnt[(pr, int(y))] += 1
                if int(y) >= 2022 and len(ex[pr]) < 3:
                    ex[pr].append({"title": (d.get("title") or "")[:200], "year": int(y), "source": "cordis",
                                   "url": f"https://cordis.europa.eu/project/id/{d.get('id')}"})
    return cnt, ex


def main() -> int:
    todo = [(s, p) for s, p in files() if s in ("openalex", "arxiv") and
            (s != "openalex" or int(re.search(r"year=(\d+)", p).group(1)) >= YEARS[0])]
    cnt, total, ex = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    with ProcessPoolExecutor(6) as pool:
        for c, t, e in pool.map(_file, todo, chunksize=2):
            cnt.update(c)
            total.update(t)
            for k, v in e.items():
                ex[k] += v[: max(0, 6 - len(ex[k]))]
        cord = list(pool.map(_cordis, [CORPUS / "cordis" / "cordis-h2020projects-json.zip",
                                       CORPUS / "cordis" / "cordis-HORIZONprojects-json.zip"]))
    ccnt = collections.Counter()
    for c, e in cord:
        ccnt.update(c)
        for k, v in e.items():
            ex[k] += v[: max(0, 8 - len(ex[k]))]
    rows = []
    for m in METHODS:
        for o in OBJECTS:
            pr = (m, o)
            series = {y: cnt[(pr, y)] for y in YEARS}
            rec, bas = sum(series[y] for y in RECENT), sum(series[y] for y in BASE)
            if rec == 0:
                continue
            share = lambda ys: sum(series[y] for y in ys) / max(1, sum(total[y] for y in ys))
            growth = (share(RECENT) + 1e-9) / (share(BASE) + 1e-9)
            first = next((y for y in YEARS if series[y] >= 3), None)
            cser = {y: ccnt[(pr, y)] for y in YEARS}
            rows.append({"method": m, "object": o, "docs_recent": rec, "docs_base": bas, "growth": round(growth, 2),
                         "first_year": first, "cordis_recent": sum(cser[y] for y in range(2022, 2027)),
                         "series": json.dumps({str(y): series[y] for y in YEARS}),
                         "cordis_series": json.dumps({str(y): cser[y] for y in YEARS}),
                         "examples": json.dumps(ex.get(pr, [])[:8], ensure_ascii=False),
                         "name_ru": f"{METHODS[m][1][:1].upper() + METHODS[m][1][1:]} для {OBJECTS[o][1]}",
                         "phrase": f"{m} for {o}", "area": AREA_OF_OBJECT.get(o)})
    t = pd.DataFrame(rows)
    # в базу: свежая растущая пара — первое появление с 2019 года, рост доли ≥ 2, ≥ 10 работ за 2024–2026
    t["signal"] = (t.first_year.fillna(9999) >= 2019) & (t.growth >= 2) & (t.docs_recent >= 10)
    t = t.sort_values(["signal", "growth"], ascending=[False, False])
    t.to_parquet(REG / "composites.parquet")
    s = t[t.signal]
    print(f"пар метод × объект с работами 2024–2026: {len(t)}; растущих свежих (в базу): {len(s)}")
    for r in s.head(30).itertuples():
        print(f"  {r.name_ru:55} {r.docs_base:>4} → {r.docs_recent:>5}  ×{r.growth:5.1f}  с {r.first_year}  ЕС {r.cordis_recent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
