# Ravan v2 production run — phases 4–9 (living report)

Status: **in progress** — pretraining running; sections marked `[FILL]` are
completed after each stage. v1 history: `PHASE1_DIAGNOSIS.md`, `FINAL_REPORT.md`.

## Why v2 (recap)

v1 failed for three documented root causes (see `PHASE1_DIAGNOSIS.md`): double
label-shift (held-out ppl ≈164–188), tokenizer never fed corpus text (3.2/3.9
tokens per word), greedy loop ending in `<|end|>` + `.strip()` → empty answers
(95.2% empty rate, 21/21 repetitive).

## Phase 4 — tokenizer (done)

- Candidates trained on an 81 MB stratified corpus sample: **8k** and **16k** BPE
  (min_freq 3, ByteLevel `use_regex`, NFC, full byte initial alphabet).
- Held-out efficiency (`reports/tokenizer_comparison.json`):

| vocab | bytes/tok probes | bytes/tok fa | bytes/tok en | tokens/word |
|---|---|---|---|---|
| 8192 | 4.67 | 5.37 | 3.16 | 1.786 |
| 16384 | 5.31 | 5.89 | 3.65 | 1.571 |

- Architecture benchmark (384d/6L/6H, seq 512, batch 4, fp32, 2 threads):

| vocab | params | train tok/s | bf16 tok/s |
|---|---|---|---|
| 8192 | 13.99 M | **1607** | — |
| 16384 | 17.14 M | 1350 | 486 |

- **Decision: 8k** — best effective text throughput (~900 words/s vs 859),
  smallest model for the token budget, 1.8× better than v1's tokenization.
  Special ids: `[UNK]`=0, bos/eos=`endoftext`=1, `<|user|>`=2, `<|assistant|>`=3,
  `<|end|>`=4 (asserted in tests).

## Phase 5 — pretraining (running)

- Corpus: **535,470,153 chars / 1,711,732 docs** (`corpus_stats.json`):
  fawiki-2018 166,965 docs/401.9 M chars · mizan-fa 865,836/57.9 M ·
  mizan-en 654,742/53.9 M · ganjoor-pd 24,150/21.9 M · starter 39/9,031.
  1% content-hash holdout (20,218 docs).
- Stream: `train.tok` 40,000,016 tokens (379,434 seed-1234-shuffled docs),
  `valid.tok` 256,627 tokens (204 docs).
- Model: GPT-2 architecture from random init — 13.99 M params, 384 embed /
  6 layers / 6 heads, ctx 512, dropout 0, AdamW lr 6e-4 (warmup 600, cosine to
  10%), batch 4 × 512, fp32, seed 4242. Ceiling 13,000 steps (≈26.6 M tokens),
  deterministic full-stream validation every 500 steps, `last.pt` every 750.
- Curve `[FILL at completion]`.

## Phase 6 — SFT (pending pretrain)

- Data: **121 train / 25 valid dialogs** — 28 kept from v1 train + 110 of 120
  agent-authored dialogs; v1's 8 valid pairs kept untouched for cross-version
  comparability. Zero user turns match the 21 eval probes (exact + fuzzy ≥0.70,
  all-turn check; rescue/force-drop lists; `sft_v2_stats.json`, test-enforced).
- Supervision mask: assistant content + `\n` + `<|end|>` only; user text,
  headers and user-side `<|end|>` unsupervised (asserted).
- Dry-run from an early pretrain checkpoint validated the full path
  (init → masked train → deterministic val → best-save → eval_suite → generate).

## Phase 7 — GGUF (pending)

- `scripts/export_gguf_v2.py`: computes *our* vocab's `chkhsh` exactly like
  llama.cpp's `get_vocab_base_pre`, patches `conversion/base.py` at runtime to
  map it to the `gpt-2` pre-tokenizer (identical ByteLevel+regex behavior),
  converts to F16, quantizes Q4_0, restores the converter. Outputs:
  `model/ravan-v2-f16.gguf`, `model/ravan-v2-Q4_0.gguf`.
- llama.cpp built natively: llama-cli, llama-server, llama-quantize,
  llama-gguf-split. `[native test results FILL]`

## Phase 8 — app (code updated)

- `app/index.html` now emits v2 format (per-message `<|end|>`), n_predict 120,
  repeat_penalty 1.15, stop `['<|end|>','<|user|>']`, memories folded into the
  first user turn. `windows/run-chat-v2.bat` launches llama-server on
  `ravan-v2-Q4_0.gguf` with ctx 512.

## Phase 9 — evaluation & release `[FILL]`

- `[final eval numbers: empty_rate / repetitive / looks_ok / nll_per_byte vs
  v1 baselines 0.952 / 21-21 / — / 1.9178 and exp-v1fix 0.048 / 0 / 20 / 1.5977]`
- `[native llama.cpp transcript]`
- `[README rewrite, release notes, sizes]`

## Licensing (binding)

See `data/DATA_MANIFEST.md`: Persian Wikipedia CC BY-SA 4.0 (attribution +
ShareAlike), MIZAN CC BY 4.0 (Kashefi 2018, github.com/omidkashefi/Mizan),
Ganjoor classical verse public domain (AI summaries excluded), all dialogs and
starter text original to this repository. Model outputs are statistical
patterns, not verifiable facts.
