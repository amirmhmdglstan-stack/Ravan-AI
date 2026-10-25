"""Ravan evaluation suite (v2).

Runs a fixed, version-controlled probe set against one or more checkpoints and
records quantitative + qualitative results:
  - raw generated token ids (so "what happened" is auditable),
  - decoded text BEFORE any cleanup and AFTER chat post-processing,
  - stop reason, empty/whitespace-only detection, repetition metrics,
  - instruction-stream NLL normalized per byte (comparable across tokenizers),
  - tokenizer round-trip health.

Usage:
  python scripts/eval_suite.py --checkpoint model/sft_v2/best --tag sft_v2
  python scripts/eval_suite.py --checkpoint model/sft --tag original-v1
  python scripts/eval_suite.py --compare reports/eval_original-v1.json reports/eval_sft_v2.json
"""
import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "data" / "eval" / "probes.jsonl"
REPORTS = ROOT / "reports"
VALID_INSTR = ROOT / "data" / "processed" / "instructions_valid.jsonl"


def load_probes():
    return json.loads(PROBES.read_text(encoding="utf-8"))["probes"]


def format_prompt(messages, style="v2"):
    """Build the generation prompt for a message list in the given chat style.

    v1 (original experiment): '<|role|>\\n{content}\\n' per message, no per-turn
    terminator; matches scripts/evaluate.py of the first experiment exactly.
    v2 (fixed pipeline): every message closed with '<|end|>'.
    The trailing '<|assistant|>\\n' opens the answer slot in both.
    """
    if style == "v1":
        return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages) + "<|assistant|>\n"
    if style == "v2":
        return "".join(f"<|{m['role']}|>\n{m['content']}\n<|end|>\n" for m in messages) + "<|assistant|>\n"
    raise ValueError(f"unknown format style: {style}")


def repetition_metrics(text):
    toks = re.findall(r"\S+|\n", text)
    if not toks:
        return {"distinct1": 0.0, "max_token_repeat": 0, "empty": True}
    c = Counter(toks)
    distinct1 = len(c) / len(toks)
    best = cur = 1
    for a, b in zip(toks, toks[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    return {"distinct1": round(distinct1, 3), "max_token_repeat": best, "empty": False}


def generate(model, tok, prompt, max_new_tokens=64, greedy=True,
             temperature=0.8, top_p=0.9, repetition_penalty=1.1):
    enc = tok(prompt, return_tensors="pt")
    ctx = int(getattr(model.config, "n_positions", 2048))
    budget = ctx - int(enc["input_ids"].shape[1])
    if budget < 1:
        return {"gen_ids": [], "kept_ids": [], "raw_text": "", "answer": "",
                "answer_stripped": "", "stop_reason": "prompt_exceeds_context",
                "context_capped": True}
    max_new_tokens = min(max_new_tokens, budget)
    stop_ids = set()
    for sp in ("<|end|>", "<|user|>"):
        i = tok.convert_tokens_to_ids(sp)
        if i is not None and i != tok.unk_token_id:
            stop_ids.add(int(i))
    gen = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        temperature=temperature if not greedy else None,
        top_p=top_p if not greedy else None,
        repetition_penalty=repetition_penalty if not greedy else 1.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    new_ids = gen[0][enc["input_ids"].shape[1]:].tolist()
    # manual stop trimming: model.generate stops on eos only; also cut at <|user|>
    stop_reason = "max_new_tokens"
    out_ids = []
    for i, t in enumerate(new_ids):
        if t == tok.eos_token_id:
            stop_reason = "eos(<|end|>)"
            break
        if t in stop_ids and t != tok.eos_token_id:
            stop_reason = f"stop_token({tok.convert_ids_to_tokens([t])[0]})"
            break
        out_ids.append(t)
    raw = tok.decode(new_ids, skip_special_tokens=False)
    answer = tok.decode(out_ids, skip_special_tokens=True)
    return {
        "gen_ids": new_ids,
        "kept_ids": out_ids,
        "raw_text": raw,
        "answer": answer,
        "answer_stripped": answer.strip(),
        "stop_reason": stop_reason,
    }


def instr_nll_per_byte(model, tok):
    """NLL over the held-out instruction stream; normalized per BYTE of text.

    Per-byte normalization makes results comparable across tokenizers.
    Returns dict with token count, total nll, chars, nll/byte, ppl-per-token.
    """
    valid = [json.loads(s) for s in VALID_INSTR.read_text(encoding="utf-8").splitlines() if s.strip()]
    chunks = []
    for item in valid:
        text = "".join(f"<|{m['role']}|>\n{m['content']}\n<|end|>\n" for m in item["messages"])
        chunks.append(text)
    full = "\n".join(chunks)
    ids = tok.encode(full, add_special_tokens=False)
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    ctx = getattr(model.config, "n_positions", 512)
    with torch.no_grad():
        for st in range(0, len(ids) - 1, ctx - 1):
            arr = torch.tensor([ids[st:st + ctx]], dtype=torch.long)
            if arr.shape[1] < 2:
                continue
            logits = model(input_ids=arr[:, :-1]).logits
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                  arr[:, 1:].reshape(-1), reduction="sum")
            total += float(nll)
            count += arr.shape[1] - 1
    if was_training:
        model.train()
    n_bytes = len(full.encode("utf-8"))
    return {
        "tokens": count,
        "nll_total": round(total, 3),
        "text_bytes": n_bytes,
        "nll_per_byte": round(total / n_bytes, 4),
        "bits_per_byte": round(total / n_bytes / math.log(2), 4),
        "ppl_per_token": round(math.exp(min(30, total / max(1, count))), 2),
        "note": "ppl_per_token is only comparable within the SAME tokenizer; "
                "nll_per_byte/bits_per_byte are comparable across tokenizers.",
    }


def run_checkpoint(ckpt, tag, max_new_tokens, n_samples=0, style="v2"):
    tok = PreTrainedTokenizerFast.from_pretrained(str(ckpt))
    model = GPT2LMHeadModel.from_pretrained(str(ckpt)).eval()
    rows = []
    for p in load_probes():
        if n_samples and p["id"] > n_samples:
            break
        prompt = format_prompt(p["messages"], style)
        t0 = time.time()
        res = generate(model, tok, prompt, max_new_tokens=max_new_tokens, greedy=True)
        lat = round(time.time() - t0, 2)
        rep = repetition_metrics(res["answer"])
        res.update(
            id=p["id"], category=p["category"], prompt=p["messages"][-1]["content"],
            formatted_prompt=prompt, chat_format=style,
            latency_s=lat, **rep,
            whitespace_only=(res["answer"].strip() == ""),
        )
        rows.append(res)
    n_empty = sum(1 for r in rows if r["whitespace_only"])
    n_repeat = sum(1 for r in rows if r["max_token_repeat"] >= 5)
    n_ok = sum(1 for r in rows if not r["whitespace_only"] and r["max_token_repeat"] < 5)
    # sample a couple of greedy continuations too (not chat-formatted)
    cont_rows = []
    for text in ["در فارسی نیم‌فاصله", "The capital of France is"]:
        res = generate(model, tok, text, max_new_tokens=32, greedy=True)
        res.update(kind="continuation", prompt=text, **repetition_metrics(res["answer"]))
        cont_rows.append(res)
    health = {
        "roundtrip": all(
            tok.decode(tok.encode(s)) == s for s in ["سلام", "Hello!", "می‌روم", "«؟»"]
        ),
        "vocab": len(tok),
    }
    result = {
        "tag": tag,
        "checkpoint": str(ckpt),
        "decode": {"mode": "greedy", "max_new_tokens": max_new_tokens, "chat_format": style},
        "summary": {
            "probes": len(rows),
            "nonempty": len(rows) - n_empty,
            "empty_or_whitespace": n_empty,
            "repetitive": n_repeat,
            "looks_ok": n_ok,
            "empty_rate": round(n_empty / max(1, len(rows)), 3),
        },
        "instruction_stream": instr_nll_per_byte(model, tok),
        "tokenizer_health": health,
        "samples": rows,
        "continuations": cont_rows,
    }
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"eval_{tag}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tag": tag, **result["summary"], "nll_per_byte":
                      result["instruction_stream"]["nll_per_byte"]}, ensure_ascii=False))
    print("wrote", out)
    return result


def compare(path_a, path_b):
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))
    lines = []
    lines.append(f"| metric | {a['tag']} | {b['tag']} |")
    lines.append("|---|---|---|")
    for k in ("nonempty", "empty_or_whitespace", "repetitive", "looks_ok", "empty_rate"):
        lines.append(f"| {k} | {a['summary'][k]} | {b['summary'][k]} |")
    lines.append(f"| instruction nll/byte | {a['instruction_stream']['nll_per_byte']} "
                 f"| {b['instruction_stream']['nll_per_byte']} |")
    lines.append(f"| instruction bits/byte | {a['instruction_stream']['bits_per_byte']} "
                 f"| {b['instruction_stream']['bits_per_byte']} |")
    print("\n".join(lines))
    # per-probe answers side by side
    amap = {r["id"]: r for r in a["samples"]}
    bmap = {r["id"]: r for r in b["samples"]}
    print()
    for pid in sorted(amap):
        ra, rb = amap[pid], bmap.get(pid, {})
        print(f"--- probe {pid} [{ra['category']}]")
        print(f"  {a['tag']}: {ra['answer_stripped'][:220]!r}")
        print(f"  {b['tag']}: {rb.get('answer_stripped', '')[:220]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--format", dest="style", default="v2", choices=["v1", "v2"],
                    help="chat prompt format of the checkpoint (v1 = original experiment)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="only first N probes")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
        return
    if not a.checkpoint:
        ap.error("--checkpoint required (or --compare)")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    run_checkpoint(a.checkpoint, a.tag or Path(a.checkpoint).name, a.max_new_tokens, a.limit,
                   a.style)


if __name__ == "__main__":
    main()
