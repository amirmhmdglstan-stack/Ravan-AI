"""Interactive / scripted generation with the fixed stopping rules (v2).

Differences from scripts/generate.py:
- records raw token ids and stop reason (never silently strips to ''),
- stops on <|end|> and <|user|>,
- default greedy for reproducibility; --sample for temperature sampling,
- respects the checkpoint's trained context length.
"""
import argparse
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser()
p.add_argument("--prompt", default="سلام، خوبی؟")
p.add_argument("--multi-turn", default=None,
               help="raw multi-turn text already containing <|user|>/<|assistant|> markers; "
                    "overrides --prompt")
p.add_argument("--max-new-tokens", type=int, default=80)
p.add_argument("--checkpoint", default=str(ROOT / "model" / "sft_v2" / "best"))
p.add_argument("--sample", action="store_true")
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top-p", type=float, default=0.9)
p.add_argument("--repetition-penalty", type=float, default=1.15)
a = p.parse_args()

torch.set_num_threads(max(1, torch.get_num_threads()))
d = Path(a.checkpoint)
m = GPT2LMHeadModel.from_pretrained(str(d)).eval()
t = PreTrainedTokenizerFast.from_pretrained(str(d))

if a.multi_turn is not None:
    base = a.multi_turn if a.multi_turn.endswith("\n") else a.multi_turn + "\n"
    if not base.endswith("<|end|>\n"):
        base += "<|end|>\n"
    prompt = base + "<|assistant|>\n"
else:
    prompt = f"<|user|>\n{a.prompt}\n<|end|>\n<|assistant|>\n"

ids = t(prompt, return_tensors="pt").input_ids
stop = set()
for s in ("<|end|>", "<|user|>"):
    i = t.convert_tokens_to_ids(s)
    if i is not None and i != t.unk_token_id:
        stop.add(int(i))

with torch.no_grad():
    out = m.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=min(a.max_new_tokens, m.config.n_positions - ids.shape[1]),
        do_sample=a.sample,
        temperature=a.temperature if a.sample else None,
        top_p=a.top_p if a.sample else None,
        repetition_penalty=a.repetition_penalty if a.sample else 1.0,
        eos_token_id=t.eos_token_id,
        pad_token_id=t.pad_token_id if t.pad_token_id is not None else t.eos_token_id,
    )

gen_ids = out[0][ids.shape[1]:].tolist()
kept, reason = [], "max_new_tokens"
for tok_id in gen_ids:
    if tok_id == t.eos_token_id:
        reason = "eos(<|end|>)"
        break
    if tok_id in stop and tok_id != t.eos_token_id:
        reason = f"stop_token({t.convert_ids_to_tokens([tok_id])[0]})"
        break
    kept.append(tok_id)

answer = t.decode(kept, skip_special_tokens=True)
print(f"prompt   : {prompt!r}")
print(f"gen_ids  : {gen_ids}")
print(f"stop     : {reason}")
print(f"raw_text : {t.decode(gen_ids, skip_special_tokens=False)!r}")
print(f"answer   : {answer.strip()!r}")
if not answer.strip():
    print("WARNING: empty answer (all whitespace or immediate stop)")
