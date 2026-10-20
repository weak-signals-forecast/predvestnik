#!/usr/bin/env python3
"""Document vectors (bge-base) for the whole corpus with plain torch + transformers.
No sentence-transformers, no pandas, no registry needed: runs inside the stock
huggingface/transformers-pytorch-gpu container (one .sif file, no python env on STORE).

    GPB_CORPUS=/data/gpb_corpus \
    DOC_EMB=/data/doc_emb \
    python embed_docs_hf.py --batch 512

Output: one .npz per corpus folder (openalex subfield/year, arxiv set, hf) with
ids, year, group, emb (float16, L2-normalised). Entity lists (ents, ent_off) are
filled later by attach_ents.py when the registry vocabulary exists (if GPB_REGISTRY
with vocab.txt is available and --repo points to the predvestnik checkout, they are
computed here directly). Re-submitting continues: folders with an existing .npz are skipped.
Env: HF_HOME (model cache), HF_HUB_OFFLINE=1 on compute nodes.
"""
from __future__ import annotations

import argparse
import ast
import glob
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

CORPUS = Path(os.environ.get("GPB_CORPUS", Path.home() / "gpb_corpus"))
OUT = Path(os.environ.get("DOC_EMB", CORPUS / "registry_v5" / "doc_emb"))
YEARS = tuple(range(2012, 2027))
ARXIV_SETS = ("cs", "stat", "eess", "q-fin", "econ", "quant-ph", "cond-mat", "physics", "q-bio")


# ---------- corpus reading (copy of registry/corpus.py, pure python) ----------
def folders(corpus: Path) -> list[tuple[str, str, list[str]]]:
    """(source, folder, files) - one output file per folder keeps the file count small."""
    out = []
    for d in sorted(glob.glob(str(corpus / "openalex/subfield=*/year=*"))):
        fs = sorted(glob.glob(d + "/*.jsonl.gz"))
        if fs:
            out.append(("openalex", d, fs))
    for d in sorted(glob.glob(str(corpus / "arxiv/set=*"))):
        fs = sorted(glob.glob(d + "/*.jsonl.gz"))
        if fs:
            out.append(("arxiv", d, fs))
    d = str(corpus / "hf/daily_papers")
    fs = sorted(glob.glob(d + "/*.jsonl.gz"))
    if fs:
        out.append(("hf", d, fs))
    return out


def _skip_openalex(r: dict) -> bool:
    src = r.get("source") or (r.get("primary_location") or {}).get("source") or {}
    name = (src.get("display_name") or src.get("name") or "") if isinstance(src, dict) else str(src)
    if name.startswith("arXiv"):
        return True
    au = r.get("authorships") or []
    return not any(a.get("institutions") for a in au if isinstance(a, dict)) and not (r.get("references") or r.get("referenced_works") or [])


def _kw(r: dict) -> str:
    """keywords as strings or as OpenAlex dicts ({display_name|keyword: ...})."""
    out = []
    for k in r.get("keywords") or []:
        if isinstance(k, dict):
            k = k.get("display_name") or k.get("keyword") or ""
        if k:
            out.append(str(k))
    return " ; ".join(out)


def _abstract(r: dict) -> str:
    a = r.get("abstract")
    if a:
        return a
    inv = r.get("abstract_inverted_index")
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def _lines(path: str):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            yield from fh
    except (EOFError, OSError):
        return


def _arxiv_cats(cats) -> list[str]:
    try:
        v = ast.literal_eval(cats) if isinstance(cats, str) else cats
        return [c for c in (v or []) if c]
    except (ValueError, SyntaxError):
        return re.findall(r"[a-z\-]+(?:\.[A-Za-z\-]+)?", cats or "")


def _arxiv_owner(cats: list[str]) -> str | None:
    arch = [c.split(".")[0] for c in cats]
    if arch and arch[0] in ARXIV_SETS:
        return arch[0]
    return next((s for s in ARXIV_SETS if s in arch), None)


def docs(source: str, path: str, seen: set[str]):
    if source == "openalex":
        sub = re.search(r"subfield=(\d+)", path).group(1)
        for l in _lines(path):
            r = json.loads(l)
            if _skip_openalex(r):
                continue
            title = r.get("title") or r.get("display_name") or ""
            yield {"year": r.get("year") or r.get("publication_year"), "group": f"oa:{sub}", "title": title,
                   "text": " . ".join([title, _abstract(r), _kw(r)]), "id": r.get("id")}
    elif source == "arxiv":
        set_name = re.search(r"set=([^/]+)", path).group(1).split(":")[-1].split("_")[-1]
        for l in _lines(path):
            r = json.loads(l)
            cats = _arxiv_cats(r.get("categories"))
            if r["id"] in seen or _arxiv_owner(cats) != set_name:
                continue
            seen.add(r["id"])
            y = int((r.get("created") or "0")[:4] or 0)
            yield {"year": y, "group": "ax:" + (cats[0] if cats else ""), "title": r.get("title") or "",
                   "text": (r.get("title") or "") + " . " + (r.get("abstract") or ""), "id": "arXiv:" + r["id"]}
    elif source == "hf":
        for l in _lines(path):
            r = json.loads(l)
            kw = " ; ".join(r.get("ai_keywords") or [])
            yield {"year": int(r["date"][:4]), "group": "hf", "title": (r.get("title") or "") + " ; " + kw,
                   "text": " . ".join([r.get("title") or "", r.get("summary") or "", kw]),
                   "id": "arXiv:" + str(r.get("arxiv_id"))}


def name_for(source: str, folder: str) -> str:
    rel = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(Path(folder).relative_to(CORPUS)))
    return f"{source}__{rel}.npz"


# ---------- optional entity lists (needs registry vocab + repo) ----------
def entity_fn(repo: str):
    reg = os.environ.get("GPB_REGISTRY", "")
    if not repo or not reg or not (Path(reg) / "vocab.txt").exists():
        return None
    sys.path.insert(0, repo)
    from registry.text import candidates   # noqa: E402
    phrases = [l for l in (Path(reg) / "vocab.txt").read_text(encoding="utf-8").split("\n") if l]
    p2e = np.load(Path(reg) / "phrase2entity.npy")
    vocab = {ph: int(e) for ph, e in zip(phrases, p2e)}
    return lambda text: sorted({e for ph in candidates(text) if (e := vocab.get(ph)) is not None})


# ---------- model ----------
class Encoder:
    def __init__(self, name: str, max_seq: int, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        if device == "cuda":
            self.model.half()
        self.max_seq, self.device = max_seq, device

    def encode(self, texts: list[str], batch: int) -> np.ndarray:
        torch = self.torch
        out = []
        # sort by length: fewer padding tokens, ~2x faster on GPU
        order = np.argsort([len(t) for t in texts])[::-1]
        with torch.inference_mode():
            for i in range(0, len(texts), batch):
                chunk = [texts[j] for j in order[i:i + batch]]
                enc = self.tok(chunk, padding=True, truncation=True, max_length=self.max_seq, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                h = self.model(**enc).last_hidden_state[:, 0]          # CLS pooling (bge)
                h = torch.nn.functional.normalize(h.float(), dim=-1)
                out.append(h.cpu().numpy().astype(np.float16))
        emb = np.concatenate(out) if out else np.zeros((0, 768), np.float16)
        res = np.empty_like(emb)
        res[order] = emb
        return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-words", type=int, default=220)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--sources", default="openalex,arxiv,hf")
    ap.add_argument("--repo", default=os.environ.get("PV_REPO", ""), help="predvestnik checkout (for entity lists)")
    ap.add_argument("--folder-glob", default="", help="only folders whose path contains this substring (test)")
    ap.add_argument("--max-docs", type=int, default=0, help="at most N documents per folder (test)")
    ap.add_argument("--stride", type=int, default=1, help="keep every N-th document (test only)")
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    enc = Encoder(a.model, a.max_seq, dev)
    ents_of = entity_fn(a.repo)
    OUT.mkdir(parents=True, exist_ok=True)
    srcs = a.sources.split(",")
    todo = [(s, d, fs) for s, d, fs in folders(CORPUS) if s in srcs and not (OUT / name_for(s, d)).exists()
            and (not a.folder_glob or a.folder_glob in d)]
    allf = folders(CORPUS)
    print(f"device {dev}, model {a.model}, corpus {CORPUS}, out {OUT}, entities {'yes' if ents_of else 'later (attach_ents.py)'}", flush=True)
    for s_ in ("openalex", "arxiv", "hf"):
        n_f = sum(1 for s2, _, _ in allf if s2 == s_); n_p = sum(len(fs) for s2, _, fs in allf if s2 == s_)
        print(f"  {s_}: {n_f} folders, {n_p} files", flush=True)
    print(f"  folders to do now: {len(todo)}", flush=True)
    t0, n_total = time.time(), 0
    for k, (source, folder, fs) in enumerate(todo, 1):
        seen: set[str] = set()
        rows = []
        for f in fs:
            rows += [d for d in docs(source, f, seen) if d["year"] in YEARS and d["title"]]
            if a.max_docs and len(rows) >= a.max_docs:
                rows = rows[: a.max_docs]
                break
        if a.stride > 1:
            rows = rows[:: a.stride]
        texts = [" ".join(d["text"].split()[: a.max_words]) for d in rows]
        emb = enc.encode(texts, a.batch) if rows else np.zeros((0, 768), np.float16)
        ents = [ents_of(d["text"]) for d in rows] if ents_of else [[] for _ in rows]
        off = np.zeros(len(rows) + 1, dtype=np.int64)
        off[1:] = np.cumsum([len(e) for e in ents])
        tmp = OUT / (name_for(source, folder) + ".tmp.npz")
        np.savez_compressed(tmp, ids=np.array([str(d["id"]) for d in rows]),
                            year=np.array([d["year"] for d in rows], np.int16),
                            group=np.array([d["group"] for d in rows]), emb=emb,
                            ents=np.array([e for es in ents for e in es], np.int32), ent_off=off,
                            has_ents=np.array(bool(ents_of)))
        os.replace(tmp, OUT / name_for(source, folder))
        n_total += len(rows)
        el = time.time() - t0
        print(f"  {k}/{len(todo)} {Path(folder).relative_to(CORPUS)}: {len(rows):,} docs; total {n_total:,}, "
              f"{n_total / max(el, 1):.0f} docs/s, {el / 60:.0f} min", flush=True)
    print("done ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
