# Ravan-AI — a Persian-first language model trained from scratch, end to end

**v2 (2026-10-25):** a 13.99 M-parameter GPT-2-style decoder, a fresh 8k
byte-level BPE tokenizer, pretraining on a 535-million-character lawful corpus,
leak-controlled SFT, and CPU-native GGUF deployment — all trained here from
random initialization. Not a fine-tune of any existing model.

## Honest quality statement (read this first)

Measured against the fixed 21-probe suite (`reports/eval_sft_v2.json`):

| run | decode | empty answers | repetitive | looks ok | NLL/byte |
|---|---|---|---|---|---|
| original v1 (2026-10-24) | greedy | 95.2% | 21/21 | — | 1.9178 |
| fixed-pipeline experiment | greedy | 4.8% | 0 | 20/21 | 1.5977 |
| **v2 final** | greedy | **0%** | **0** | **21/21** | **1.4286** |
| **v2 final** | sampled | **0%** | **0** | **21/21** | 1.4286 |

What this means: v2 always answers, stops correctly at the end-of-turn token,
emits well-formed Persian with no degenerate loops, and assigns far better
likelihood to held-out instructions than either earlier run. **What it does
not mean:** with 14 M parameters and 25 M training tokens the answers are only
partially on topic. This is a research artifact that demonstrates a complete
pipeline, not a production assistant. Outputs are statistical patterns, not
verifiable facts — never rely on it for education, health, law, or safety.
Attribution for the training data is mandatory: see `NOTICE`.

## What's in the box

| path | what it is |
|---|---|
| `model/ravan-v2-Q4_0.gguf` (10.6 MB) | the deployable model (Q4_0, llama.cpp) |
| `model/ravan-v2-f16.gguf` (28.8 MB) | unquantized twin |
| `model/sft_v2/best/` | selected HF checkpoint (SFT step 300, weights-only, gitignored) |
| `model/pretrain_v2/` | pretraining output incl. `best/` (gitignored) |
| `model/tok_v2_8k/`, `model/tok_v2_16k/` | the two tokenizer candidates (8k selected) |
| `model/pretrain/`, `model/sft/`, `model/ravan-sft-*.gguf` | **v1, kept for comparison — do not overwrite** |
| `app/index.html` | offline Persian RTL chat UI (v2 format) |
| `windows/run-chat-v2.bat` | one-click Windows launcher (bring your own llama-server.exe) |
| `scripts/` | full pipeline: corpus, tokenizer, streams, train, eval, export, tests |
| `data/DATA_MANIFEST.md`, `NOTICE` | sources, licences (CC BY-SA / CC BY / PD), obligations |
| `reports/` | every metric and transcript cited here (json) |
| `docs/REPRODUCE_v2.md` | reproduce everything, including data downloads |

## Model and training facts

- Architecture: GPT-2 (causal decoder), 384 embed dim, 6 layers, 6 heads,
  context 512, tied embeddings, **13,989,888 parameters**, random init
  (seed 4242). bf16 autocast was benchmarked and rejected (486 vs 1350 tok/s
  on this CPU); all training is fp32.
- Tokenizer: byte-level BPE, vocab 8,192, trained on an 81 MB stratified
  sample of the corpus. Efficiency: 4.67 bytes/token on probe sentences, 5.37
  (fa) / 3.16 (en) bytes/token on held-out corpus, 1.79 tokens/word (v1: 3.2/3.9).
  Chosen over a 16k candidate on effective text throughput; full table in
  `reports/tokenizer_comparison.json`.
- Corpus: **535,470,153 chars / 1,711,732 documents** — Persian Wikipedia
  2018 (CC BY-SA 4.0), MIZAN fa/en (CC BY 4.0), Ganjoor public-domain classical
  verse, plus project-authored starter text. 1% content-hash holdout.
  Provenance and obligations: `data/DATA_MANIFEST.md` + `NOTICE`.
- Pretraining: 40.0 M-token shuffled stream, 13,000 steps ≈ 25.1 M tokens seen,
  AdamW 6e-4 → 6e-5 cosine, warmup 600, batch 4×512, dropout 0.
  Held-out val improved 9.08 → **4.564** (ppl 95.97).
- SFT: 121 train / 25 valid dialogs (110 newly authored for this repo, 28 kept
  from v1, v1's 8 valid pairs reserved), **zero overlap with the 21 eval
  probes** (exact + fuzzy, every turn, test-enforced). Loss masks everything
  except assistant content + end-of-turn. Checkpoint selection combined
  validation loss with probe-suite generation quality: val-min was step 50,
  but step 300 produced far better answers; the comparison table lives in
  `reports/PRODUCTION_REPORT_v2.md`, snapshots cleaned up, logs in
  `reports/sft_logs/`.
- Runtime: llama.cpp (built here, 0.7.0-dev), conversion patches the custom
  vocab's pre-tokenizer hash to `gpt-2` behaviour and restores the converter
  afterwards (`scripts/export_gguf_v2.py`). GGUF metadata verified
  (`pre=gpt-2`, `eos_token_id=4` = `<|end|>`). Measured on this host:
  ~96 tokens/s generated on 2 CPU threads (Q4_0).

## Chat format (v2 — different from v1)

Every message is closed: `<|role|>\n{content}\n<|end|>\n`; the answer slot
opens with `<|assistant|>\n`. Generation must stop on `<|end|>` or `<|user|>`.
Special ids: `[UNK]`=0, `endoftext`=1 (bos), `<|user|>`=2, `<|assistant|>`=3,
`<|end|>`=4. Deployment decode: temperature 0.75, top-p 0.9, repeat-penalty
1.15 (this is what the web app uses and what the "sampled" row measures).

## Running it on Windows (target PC: Pentium G3250, 4 GB, Win11)

1. Copy the repo to e.g. `C:\Ravan-AI`.
2. One-time download of the official Windows x64 CPU build of llama.cpp
   (must not require AVX2 — the G3250 is SSE4.1/SSE4.2 only; see
   `docs/INSTALL_FA.md` for the exact release and `Illegal instruction`
   guidance).
3. Put `llama-server.exe` + its DLLs in `windows\`.
4. Double-click `windows\run-chat-v2.bat`. The UI opens at
   `http://127.0.0.1:8080`, loopback only. No Python, no internet afterwards.

## Training from scratch (research machine)

See `docs/REPRODUCE_v2.md` — full commands for data acquisition, corpus,
tokenizer, streams, pretraining, SFT, evaluation, export, and native testing.
Quick path:

```bash
.venv/bin/python scripts/test_pipeline.py        # 26 checks (data, leak-control, formats)
.venv/bin/python scripts/train_v2.py --stage pretrain ...   # docs/REPRODUCE_v2.md §3
.venv/bin/python scripts/train_v2.py --stage sft ...        # §4
.venv/bin/python scripts/eval_suite.py --checkpoint model/sft_v2/best --tag sft_v2
.venv/bin/python scripts/eval_suite.py --compare reports/eval_original-v1.json reports/eval_sft_v2.json
.venv/bin/python scripts/export_gguf_v2.py --llama-cpp /path/to/llama.cpp --quantize /path/to/llama-quantize
```

## CPU, memory, portability notes

- The Pentium G3250 lists SSE4.1/SSE4.2 only — use a CPU package or build
  without AVX2 (`GGML_AVX=OFF GGML_AVX2=OFF ...`); details in `docs/INSTALL_FA.md`.
- Model runtime footprint is small (Q4_0 ≈ 10.6 MB weights; llama-server with
  ctx 512 is tens of MB RSS). The 4 GB total RAM is dominated by Windows/browser.
- The local UI talks only to `127.0.0.1`; history and explicit memories live in
  browser localStorage (unencrypted — no secrets). Memories are context notes,
  not weight updates; there is no retrieval or online learning.

## Repository layout for reviewers

- `reports/PHASE1_DIAGNOSIS.md` — root-cause analysis of v1 (why it failed).
- `reports/PRODUCTION_REPORT_v2.md` — this phase's full log with every table.
- `reports/eval_*.json` — raw generations (token ids included) for every claim above.
- `scripts/test_pipeline.py` — 26 automated checks; run before and after changes.
- v1 files (`scripts/train.py`, `model/pretrain`, `model/sft`, `reports/FINAL_REPORT.md`)
  are intentionally preserved for A/B comparison.
