# Ravan v2 production run — phases 4–9 (final)

Date: 2026-10-25. Host: 2-thread CPU, 3.9 GiB RAM, no CUDA. All numbers below
are recorded artifacts in `reports/` (json) unless noted.

## Why v2 (recap)

v1 failed for three documented root causes (see `PHASE1_DIAGNOSIS.md`): double
label-shift (held-out ppl ≈164–188), tokenizer never fed corpus text (3.2/3.9
tokens per word), greedy loop ending in `<|end|>` + `.strip()` → empty answers
(95.2% empty rate, 21/21 repetitive).

## Phase 4 — tokenizer (done)

Candidates trained on an 81 MB stratified corpus sample (min_freq 3, ByteLevel
`use_regex`, NFC, full byte initial alphabet). Held-out efficiency
(`reports/tokenizer_comparison.json`):

| vocab | bytes/tok probes | bytes/tok fa | bytes/tok en | tokens/word |
|---|---|---|---|---|
| 8192 | 4.67 | 5.37 | 3.16 | 1.786 |
| 16384 | 5.31 | 5.89 | 3.65 | 1.571 |

Architecture benchmark (384d/6L/6H, seq 512, batch 4, fp32, 2 threads):

| vocab | params | train tok/s | bf16 tok/s |
|---|---|---|---|
| 8192 | 13.99 M | **1607** | — |
| 16384 | 17.14 M | 1350 | 486 |

**Decision: 8k** — best effective text throughput (~900 words/s), smallest
model for the token budget, 1.8× better tokenization than v1. bf16 autocast
measured *slower* (486 tok/s) — everything trained in fp32.
Special ids: `[UNK]`=0, bos/eos=`endoftext`=1, `<|user|>`=2, `<|assistant|>`=3,
`<|end|>`=4 (test-enforced).

## Phase 5 — pretraining (done)

Corpus **535,470,153 chars / 1,711,732 docs** (`corpus_stats.json`):
fawiki-2018 166,965 docs/401.9 M chars · mizan-fa 865,836/57.9 M · mizan-en
654,742/53.9 M · ganjoor-pd 24,150/21.9 M · starter 39/9,031. 1% content-hash
holdout (20,218 docs). Stream: `train.tok` 40,000,016 tokens / `valid.tok`
256,627 (seed 1234).

Run: 13,989,888 params (384/6L/6H, ctx 512, dropout 0), AdamW 6e-4 warmup 600
cosine→6e-5, batch 4×512, seed 4242, 13,000 steps = 25,088,000 tokens seen
(+750-step restart after an OOM kill during a concurrent compile — resumed via
`--resume`). Wall ≈ 4.6 h effective.

| step | train | val | val ppl |
|---|---|---|---|
| 500 | 6.09 | 6.26 | 522 |
| 2000 | 5.12 | 5.65 | 285 |
| 5500 | 4.21 | 5.07 | 160 |
| 9000 | 4.19 | 4.74 | 114 |
| 13000 | 4.13 | **4.564** | **95.97** |

## Phase 6 — SFT (done)

Data **121 train / 25 valid dialogs** (28 from v1 + 110/120 agent-authored;
v1's 8 valid pairs untouched for comparability; zero user turns matching the
21 eval probes — `sft_v2_stats.json`, test-enforced). Supervision: assistant
content + `\n` + `<|end|>` only.

Selection lesson (recorded before the run): **validation loss alone picks the
wrong checkpoint.** Val minimum was step 50 (4.75), yet its greedy answers
looped; generation quality peaked much later. We snapshotted every eval and
scored candidates with the probe suite (greedy *and* app-like sampling):

| SFT step | train | val | greedy empty/rep/ok | sampled empty/rep/ok | nll/byte | qualitative |
|---|---|---|---|---|---|---|
| 50 | 3.83 | **4.75** | 0 / 5 / 16 | 0 / 0 / 21 | **1.1018** | word salad |
| 100 | 2.42 | 4.90 | 0 / 0 / 21 | 0 / 0 / 21 | 1.1696 | fluent loops |
| 200 | 0.55 | 5.59 | — | 0 / 0 / 21 | 1.3401 | mixed fragments |
| **300 (selected)** | 0.15 | 5.99 | — | 0 / 0 / 21 | **1.4286** | coherent units, partial relevance |
| 400 | 0.07 | 6.12 | 0 / 0 / 21 | 0 / 0 / 21 | 1.4394 | similar to 300 |

**Selected: step 300** → `model/sft_v2/best` (snapshots cleaned up). Decode
policy for deployment: sampling temp 0.75 / top-p 0.9 / repeat-penalty 1.15
(matches the web app); greedy is reported only for baseline comparability.

## Phase 7 — GGUF (done)

`scripts/export_gguf_v2.py` computed our vocab's converter hash
`ed4475eb7413379481e643e9ff2908fb20d86e09486a61f69c53134b7d699b76`, patched
`conversion/base.py` → `gpt-2` pre-tokenizer, converted, quantized, restored.
Verified GGUF metadata: `tokenizer.ggml.model=gpt2`, `pre=gpt-2`,
`bos=1`, `eos=4` (`<|end|>` — natural turn-stop). Outputs:
`model/ravan-v2-f16.gguf` (28.8 MB), `model/ravan-v2-Q4_0.gguf` (10.6 MB,
5.85 BPW, 9.75 MiB tensors; 1/76 tensor fell back from f16 — embedding).

Native llama.cpp 0.7.0-dev test (Linux, 2 threads):
`llama-server` loads the Q4_0 file, `POST /completion` with the app payload
(stop `['<|end|>','<|user|>') returns Persian content and halts with
`stop_type=eos` at the end-of-turn token; ~96 tokens predicted per second on
this host. Transcript evidence: `reports/eval_sft_v2*.json` + this section.

## Phase 8 — app (done)

`app/index.html` emits v2 format (per-message `<|end|>`), n_predict 120,
repeat_penalty 1.15, stop `<|end|>`/`<|user|>`, memories folded into the first
user turn. `windows/run-chat-v2.bat` launches `llama-server.exe` on
`ravan-v2-Q4_0.gguf` (ctx 512, threads 2, loopback only).

## Phase 9 — evaluation vs baselines (final)

Metric policy: compare across tokenizers only via `nll_per_byte`; generate
with fixed seeds; probe set = the 21 held-out probes (leak-controlled).

| run | decode | empty_rate | repetitive | looks_ok | nll/byte |
|---|---|---|---|---|---|
| original v1 (`eval_original-v1.json`) | greedy | 0.952 | 21/21 | — | 1.9178 |
| fixed-pipeline exp (`eval_exp-v1fix.json`) | greedy | 0.048 | 0 | 20/21 | 1.5977 |
| **v2 final (`eval_sft_v2.json`)** | greedy | **0.000** | **0** | **21/21** | **1.4286** |
| **v2 final (`eval_sft_v2_samp.json`)** | sampled | **0.000** | **0** | **21/21** | 1.4286 |

Caveats (must be stated in any release): `looks_ok` is mechanical
(non-empty, no long token repeats); manual reading of samples shows answers
are grammatical, well-formatted Persian with partial topic relevance — **not
reliable answers**. The model is 14 M parameters trained on 25 M tokens; it
cannot be trusted for facts, education, health, or law. Outputs are
statistical patterns, not verifiable facts. Attribution obligations:
`NOTICE` / `data/DATA_MANIFEST.md` (CC BY-SA Wikipedia, CC BY MIZAN, PD
Ganjoor).

## Reproduce

`docs/REPRODUCE_v2.md` (data → corpus → tokenizer → streams → pretrain →
SFT → eval → GGUF → native). Environment snapshot: `reports/pip_freeze_v2.txt`,
`reports/environment_v2.txt`. Test suite: `scripts/test_pipeline.py` — 26/26.
