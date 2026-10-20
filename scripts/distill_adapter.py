"""Дистилляция кросс-энкодера в «адаптер запроса»: онлайн остаётся векторный поиск, но порядок — как у кросс-энкодера.

    REGISTRY_BUNDLE=artifacts/registry_v5d python scripts/distill_adapter.py inputs --out-dir distill/
    (кластер) for_cesga/server/teacher_scores.py: distill/pairs.jsonl + docs.jsonl -> teacher.npz
    REGISTRY_BUNDLE=artifacts/registry_v5d python scripts/distill_adapter.py train --dir distill/ --teacher teacher.npz

Учитель — BAAI/bge-reranker-v2-m3 на полном тексте записи (название, термин, описание, три заголовка), на A100.
Ученик — матрица W (1024×1024, W = I + ΔW): вектор запроса USER-bge-m3 заменяется на нормированный W·q, векторы записей
и модель не меняются; в сервисе это одно умножение (registry_service: query_adapter.npy в наборе).
Запросы обучения: русские запросы аналитика (scripts/make_ru_queries.py) и отложенные заголовки источников; запросы
модели-судьи и 18 злых запросов в обучение не входят. Деление на обучение и проверку — по записи-цели (запись
проверки не встречается в обучении). Метрики на проверке до и после: Recall@10 известной записи, nDCG@10 и Spearman
относительно порядка учителя.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from registry.mini_bundle import EMB_TITLES  # noqa: E402

BUNDLE = Path(os.getenv("REGISTRY_BUNDLE", "artifacts/registry"))
ROOT = Path(__file__).resolve().parents[2]
K = 50
TEST_SHARE = 0.2


def _doc_text(sig, cards, j) -> str:                       # как registry_service.Registry._doc_text
    c = cards.get(sig.key[j]) or {}
    desc = sig.description_ru[j] if isinstance(sig.description_ru[j], str) else ""
    ts = [(d.get("title") or "") for d in (c.get("sources") or [])][:3]
    return ". ".join(x for x in [sig.name_ru[j], sig.phrase[j].replace("_", " "), desc] + ts if x)


def _encoder():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("deepvk/USER-bge-m3", device="cpu")
    m.max_seq_length = 64
    return m


def inputs(a) -> None:
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    sig = pd.read_parquet(BUNDLE / "signals.parquet")
    with gzip.open(BUNDLE / "cards.json.gz", "rt", encoding="utf-8") as fh:
        cards = json.load(fh)
    pos = {k: i for i, k in enumerate(sig.key)}
    judge = {json.loads(l)["query"] for l in open(ROOT / "проверки" / "judge_queries.jsonl", encoding="utf-8")}
    qs = []
    for l in open(ROOT / "проверки" / "queries_ru.jsonl", encoding="utf-8"):      # русские запросы аналитика
        x = json.loads(l)
        if x["key"] in pos and x["query"] not in judge:
            qs.append({"query": x["query"], "key": x["key"], "lang": "ru"})
    for i, k in enumerate(sig.key):                                             # отложенные заголовки
        ts = [(d.get("title") or "").strip() for d in (cards.get(k) or {}).get("sources") or [] if d.get("title")]
        if len(ts) > EMB_TITLES + 1:
            qs.append({"query": ts[EMB_TITLES + 1][:200], "key": k, "lang": "en"})
    rng = np.random.default_rng(0)
    keys = sorted({q["key"] for q in qs})
    test_keys = set(rng.choice(keys, int(len(keys) * TEST_SHARE), replace=False).tolist())
    m = _encoder()
    Q = m.encode([q["query"] for q in qs], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    E = np.load(BUNDLE / "emb.npy").astype(np.float32)
    top = np.argsort(-(Q @ E.T), axis=1)[:, :K]
    with open(out / "pairs.jsonl", "w", encoding="utf-8") as fh:
        for j, q in enumerate(qs):
            fh.write(json.dumps({"qid": j, "query": q["query"], "lang": q["lang"], "target": pos[q["key"]],
                                 "test": q["key"] in test_keys, "cands": top[j].tolist()}, ensure_ascii=False) + "\n")
    with open(out / "docs.jsonl", "w", encoding="utf-8") as fh:
        for j in range(len(sig)):
            fh.write(json.dumps({"i": j, "text": _doc_text(sig, cards, j)}, ensure_ascii=False) + "\n")
    np.save(out / "queries_emb.npy", Q.astype(np.float32))
    n_test = sum(q["key"] in test_keys for q in qs)
    print(f"запросов {len(qs)} (ru {sum(q['lang'] == 'ru' for q in qs)}, en {sum(q['lang'] == 'en' for q in qs)}), "
          f"проверка {n_test}; пар {len(qs) * K}; записей {len(sig)} -> {out}")


def _ndcg(order_scores_teacher, k=10):
    t = order_scores_teacher
    ideal = np.sort(t)[::-1][:k]
    gain = lambda v: (2 ** v - 1) / np.log2(np.arange(2, len(v) + 2))
    return float(gain(t[:k]).sum() / max(gain(ideal).sum(), 1e-9))


def train(a) -> None:
    import torch
    d = Path(a.dir)
    pairs = [json.loads(l) for l in open(d / "pairs.jsonl", encoding="utf-8")]
    Q = torch.tensor(np.load(d / "queries_emb.npy"))
    E = torch.tensor(np.load(BUNDLE / "emb.npy").astype(np.float32))
    z = np.load(a.teacher)
    T = torch.tensor(z["scores"], dtype=torch.float32)          # [n_queries, K] в порядке cands
    C = torch.tensor([p["cands"] for p in pairs])
    test = torch.tensor([p["test"] for p in pairs])
    tgt = torch.tensor([p["target"] for p in pairs])
    lang = np.array([p["lang"] for p in pairs])
    dim = Q.shape[1]
    tau_s, tau_t = 0.05, 1.0

    def student(idx, W):
        q = torch.nn.functional.normalize(Q[idx] @ W.T, dim=1)
        return (q.unsqueeze(1) * E[C[idx]]).sum(-1)             # [b, K]

    def fit(rows, rank, l2, epochs):
        """W = I + A·B (ранг rank): мало параметров — меньше переобучения."""
        torch.manual_seed(0)
        A = (torch.randn(dim, rank) * 1e-3).requires_grad_(); B = (torch.randn(rank, dim) * 1e-3).requires_grad_()
        opt = torch.optim.Adam([A, B], lr=a.lr)
        for ep in range(epochs):
            perm = rows[torch.randperm(len(rows))]
            for s_ in range(0, len(perm), 256):
                b = perm[s_:s_ + 256]
                W = torch.eye(dim) + A @ B
                logp = torch.log_softmax(student(b, W) / tau_s, dim=1)
                pt = torch.softmax(T[b] / tau_t, dim=1)
                loss = torch.nn.functional.kl_div(logp, pt, reduction="batchmean") + l2 * ((A @ B) ** 2).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        return (torch.eye(dim) + A @ B).detach()

    def spear(W, rows):
        from scipy.stats import spearmanr
        with torch.no_grad():
            st = student(rows, W).numpy()
        tt = T[rows].numpy()
        return float(np.nanmean([spearmanr(st[i], tt[i])[0] for i in range(len(rows))]))

    # выбор ранга и регуляризации — на валидации внутри обучающих запросов (по записям-целям), не на проверке
    tr_all = (~test).nonzero().view(-1)
    tk = tgt[tr_all].numpy()
    rng = np.random.default_rng(1)
    vk = set(rng.choice(np.unique(tk), int(len(np.unique(tk)) * 0.2), replace=False).tolist())
    is_val = torch.tensor([k in vk for k in tk])
    tr, va = tr_all[~is_val], tr_all[is_val]
    base = spear(torch.eye(dim), va)
    best = (base, None)
    for rank in (8, 32):
        for l2 in (1e-2, 1e-1, 1.0):
            for epochs in (5, 15):
                W = fit(tr, rank, l2, epochs)
                v = spear(W, va)
                print(f"  ранг {rank}, l2 {l2}, эпох {epochs}: валидация {v:.3f} (без адаптера {base:.3f})", flush=True)
                if v > best[0]:
                    best = (v, (rank, l2, epochs))
    print(f"выбрано: {best[1]} (валидация {best[0]:.3f})", flush=True)
    if best[1] is None:
        W = torch.eye(dim)
    else:
        W = fit(tr_all, *best[1])                               # итог — на всех обучающих, проверка не тронута

    def evaluate(W, mask, name):
        idx = mask.nonzero().view(-1)
        with torch.no_grad():
            q = torch.nn.functional.normalize(Q[idx] @ W.T, dim=1)
            S = q @ E.T                                         # полный поиск по всем записям
            ranks = (S > S[torch.arange(len(idx)), tgt[idx]].unsqueeze(1)).sum(1) + 1
            st = student(idx, W).numpy()
        from scipy.stats import spearmanr
        tt = T[idx].numpy()
        pt = np.exp(tt - tt.max(1, keepdims=True)); pt /= pt.sum(1, keepdims=True)      # важность по учителю, ≥ 0
        nd = np.mean([_ndcg(pt[i][np.argsort(-st[i])] * 10) for i in range(len(idx))])
        sp = np.nanmean([spearmanr(st[i], tt[i])[0] for i in range(len(idx))])
        r = ranks.numpy()
        return {"set": name, "n": int(len(idx)), "recall@10": round(float((r <= 10).mean()), 3),
                "mrr": round(float((1 / r).mean()), 3), "ndcg@10_vs_teacher": round(float(nd), 3),
                "spearman_vs_teacher": round(float(sp), 3)}

    res = {}
    for name, mask in [("test", test), ("test_ru", test & torch.tensor(lang == "ru")), ("test_en", test & torch.tensor(lang == "en")),
                       ("train", ~test)]:
        res[name] = {"before": evaluate(torch.eye(dim), mask, name), "after": evaluate(W, mask, name)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    Path(a.report).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(d / "query_adapter.npy", W.numpy().astype(np.float32))
    print(f"адаптер -> {d / 'query_adapter.npy'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inputs", "train"])
    ap.add_argument("--out-dir", default="distill"); ap.add_argument("--dir", default="distill")
    ap.add_argument("--teacher", default="distill/teacher.npz")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-3); ap.add_argument("--report", default="distill_report.json")
    a = ap.parse_args()
    {"inputs": inputs, "train": train}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
