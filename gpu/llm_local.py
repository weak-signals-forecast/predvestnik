#!/usr/bin/env python3
"""Open LLM on one GPU instead of the YandexGPT API: run a file of ready prompts, save raw answers.

    HF_HOME=... python3 llm_local.py --model Qwen/Qwen2.5-7B-Instruct --inp prompts.jsonl --out answers.jsonl

Input: one JSON per line {"id": str, "prompt": str, "max_new": int}. Prompts are built on the VPS by the
project code (registry.llm_assess / registry.llm_axes), answers are parsed there by the same code, so this
script knows nothing about the project. Output: one JSON per line {"id", "raw", "sec"}; re-running skips ids
already in the output (resumable). Greedy decoding, batches of similar length, left padding.
torch + transformers only (works with transformers 4.41 in the container).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test)")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    out = Path(a.out)
    done = set()
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(l)["id"])
            except (ValueError, KeyError):
                pass
    todo = [r for r in rows if r["id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"prompts {len(rows)}, done {len(done)}, to run {len(todo)}", flush=True)
    if not todo:
        return 0

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32, attn_implementation="sdpa").to(dev)
    model.eval()
    print(f"model {a.model} on {dev}", flush=True)

    texts = {r["id"]: tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                              add_generation_prompt=True) for r in todo}
    lens = {k: len(tok(v).input_ids) for k, v in texts.items()}
    todo.sort(key=lambda r: lens[r["id"]])
    t0, n_tok, n_done = time.time(), 0, 0
    with out.open("a", encoding="utf-8") as fh:
        for s in range(0, len(todo), a.batch):
            part = todo[s:s + a.batch]
            t = time.time()
            enc = tok([texts[r["id"]] for r in part], return_tensors="pt", padding=True).to(dev)
            max_new = max(int(r.get("max_new", 1024)) for r in part)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            gen = gen[:, enc.input_ids.shape[1]:]
            dt = time.time() - t
            for r, g in zip(part, gen):
                raw = tok.decode(g, skip_special_tokens=True)
                n_tok += int((g != tok.pad_token_id).sum())
                fh.write(json.dumps({"id": r["id"], "raw": raw, "sec": round(dt / len(part), 2)}, ensure_ascii=False) + "\n")
            fh.flush()
            n_done += len(part)
            el = time.time() - t0
            print(f"  {n_done}/{len(todo)}  {n_tok / el:.0f} tok/s  {el / 60:.1f} min  "
                  f"eta {el / n_done * (len(todo) - n_done) / 60:.0f} min", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
