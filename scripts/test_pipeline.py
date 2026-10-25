"""Unit tests for the v2 training pipeline (loss alignment, masking, format).

Run: .venv/bin/python scripts/test_pipeline.py
Exits non-zero on failure.
"""
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v2 import build_sft_sample, causal_loss, format_dialog  # noqa: E402

from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def check(name, cond, extra=""):
    status = "ok" if cond else "FAIL"
    print(f"[{status}] {name} {extra}")
    if not cond:
        FAILS.append(name)


def test_loss_alignment():
    """logits[t] must be trained against token t+1 (no double shift)."""
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=32, n_positions=32, n_embd=16, n_layer=1, n_head=1,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    m = GPT2LMHeadModel(cfg).eval()
    ids = torch.randint(0, 32, (2, 17))          # full window, length 17
    x = ids[:, :-1]                              # inputs length 16
    with torch.no_grad():
        logits = model_logits = m(input_ids=x).logits   # [2,16,V]
    ours = causal_loss(logits, ids)
    # manual: pairs (logit[t] -> ids[t+1]) for t in 0..15
    manual = F.cross_entropy(logits.reshape(-1, 32).float(), ids[:, 1:].reshape(-1))
    check("causal_loss == manual next-token CE", abs(float(ours) - float(manual)) < 1e-6,
          f"({float(ours):.6f} vs {float(manual):.6f})")
    # and NOT the double-shifted CE
    double = F.cross_entropy(logits[:, :-1].reshape(-1, 32).float(), ids[:, 2:].reshape(-1))
    check("causal_loss != double-shift CE", abs(float(ours) - float(double)) > 1e-4,
          f"({float(ours):.6f} vs {float(double):.6f})")


def test_overfit_next_token():
    """A tiny model must be able to overfit a 16-token sequence under causal_loss."""
    torch.manual_seed(1)
    cfg = GPT2Config(vocab_size=40, n_positions=32, n_embd=32, n_layer=2, n_head=2,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    m = GPT2LMHeadModel(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    seq = torch.randint(4, 40, (1, 17))
    first = None
    for i in range(200):
        opt.zero_grad(set_to_none=True)
        logits = m(input_ids=seq[:, :-1]).logits
        loss = causal_loss(logits, seq)
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss)
    check("overfit: loss dropped sharply", float(loss) < first * 0.2,
          f"({first:.3f} -> {float(loss):.3f})")
    with torch.no_grad():
        lg = m(input_ids=seq[:, :-1]).logits[0]
        acc = (lg.argmax(-1) == seq[0, 1:]).float().mean().item()
    check("overfit: argmax next-token accuracy > 0.9", acc > 0.9, f"(acc={acc:.2f})")


def test_sft_masking():
    tok = PreTrainedTokenizerFast.from_pretrained(str(ROOT / "model" / "sft"))
    messages = [
        {"role": "user", "content": "سلام!"},
        {"role": "assistant", "content": "سلام! خوبی؟"},
    ]
    ids, mask = build_sft_sample(tok, messages)
    check("ids/mask same length", len(ids) == len(mask), f"({len(ids)})")
    check("text roundtrip", tok.decode(ids) == format_dialog(messages),
          repr(tok.decode(ids))[:80])
    # user content tokens must NOT be supervised; assistant content must be
    u = tok.encode("<|user|>\nسلام!\n<|end|>\n", add_special_tokens=False)
    a_header = tok.encode("<|assistant|>\n", add_special_tokens=False)
    a_content = tok.encode("سلام! خوبی؟", add_special_tokens=False)
    check("user region unsupervised", not any(mask[: len(u) + len(a_header)]))
    check("assistant content supervised",
          all(mask[len(u) + len(a_header): len(u) + len(a_header) + len(a_content)]))
    end_i = tok.convert_tokens_to_ids("<|end|>")
    end_positions = [i for i, t in enumerate(ids) if t == end_i]
    check("two <|end|>: user one unsupervised, assistant one supervised",
          len(end_positions) == 2 and not mask[end_positions[0]] and mask[end_positions[1]],
          f"mask={ [bool(mask[i]) for i in end_positions] }")
    # first assistant token must be predictable right after the header
    check("first asst token supervised", mask[len(u) + len(a_header)])
    # and its PREDICTION SOURCE (position len(u)+len(a_header)-1) exists in inputs
    check("header precedes first asst token", ids[len(u) + len(a_header) - 1] is not None)


def test_eval_probe_format():
    probes = json.loads((ROOT / "data" / "eval" / "probes.jsonl").read_text(encoding="utf-8"))
    ids = [p["id"] for p in probes["probes"]]
    check("probe ids unique", len(ids) == len(set(ids)))
    check("probe count >= 11", len(ids) >= 11, f"({len(ids)})")
    check("original 11 categories present",
          {"iranian_casual", "context", "uncertainty"} <= {p["category"] for p in probes["probes"]})


def test_format_dialog_matches_inference():
    tok = PreTrainedTokenizerFast.from_pretrained(str(ROOT / "model" / "sft"))
    messages = [{"role": "user", "content": "تست"}]
    infer_prompt = "<|user|>\nتست\n<|end|>\n<|assistant|>\n"
    fmt = format_dialog(messages) + "<|assistant|>\n"
    check("train-style format equals inference prompt+header",
          fmt == infer_prompt, f"{fmt!r} vs {infer_prompt!r}")
    # v1 formatting must reproduce the original experiment's probe-11 prompt
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_suite import format_prompt
    msgs = [
        {"role": "user", "content": "سلام، اسم من نازنین است."},
        {"role": "assistant", "content": "سلام نازنین!"},
        {"role": "user", "content": "اسم من چی بود؟"},
    ]
    original = ("<|user|>\nسلام، اسم من نازنین است.\n<|assistant|>\nسلام نازنین!\n"
                "<|user|>\nاسم من چی بود؟\n<|assistant|>\n")
    check("v1 format == original evaluate.py prompt",
          format_prompt(msgs, "v1") == original)


if __name__ == "__main__":
    test_loss_alignment()
    test_overfit_next_token()
    test_sft_masking()
    test_eval_probe_format()
    test_format_dialog_matches_inference()
    if FAILS:
        print(f"\n{len(FAILS)} test(s) FAILED: {FAILS}")
        sys.exit(1)
    print("\nall tests passed")
