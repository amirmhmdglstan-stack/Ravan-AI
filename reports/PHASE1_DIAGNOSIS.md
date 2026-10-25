# Phase 1 — Independent diagnosis of the original Ravan failure

Date: 2026-10-24. Verified by re-running the prior artifacts in this repository with
`torch 2.14.1 / transformers 5.20.2 / tokenizers 0.23.3` on CPU. All numbers below were
reproduced locally from the committed checkpoints and scripts; nothing is taken on trust
from the previous report.

## What the prior agent claimed vs. what I verified

| Claim | Verdict | Evidence |
|---|---|---|
| Random-init GPT-2, 875,264 params | **True** | Fresh `GPT2Config` in `scripts/train.py`; param count re-verified (875,264); SFT weights differ from both pretrain (L1 2445.7) and a fresh random init (L1 17491.3) → weights are genuinely learned in-project. |
| Vocab 512 byte-level BPE, trained on project training split | **Partly false** | Tokenizer exists and round-trips, but `scripts/train_tokenizer.py` passes `str(path)` to `train_from_iterator()`, which treats each item as a *document string*, not a file to open. The pretraining corpus file was therefore **never read** during tokenizer training — only the literal path string plus the 31 instruction pairs' text were seen (≈6 KB). Micro-reproduction with the same API confirms the pattern produces character-level fragmentation. |
| Held-out instruction perplexity 188 → 164 | **True** | Re-ran `scripts/evaluate.py`: NLL 5.2380033/5.0972649 → PPL 188.2938/163.5739 (matches committed report to float32 noise). Caveat: this NLL is computed with the *correct* next-token shift, i.e. it measures an objective the model was **not trained on** (see bug #1). |
| Greedy empty answers "because it immediately emitted the end token" | **False mechanism** | Direct inspection: greedy decoding emits a loop of newline tokens (id 203, `Ċ`), **never** `<|end|>`; `evaluate.py` then `.strip()`s the whitespace → empty string. Top-1 logit after `<|assistant|>\n` is `Ċ` (4.72) far above any content token. The end token is not involved. |
| GGUF conversion + llama.cpp loading worked | Not re-verified yet (deferred to Phase 7) | GGUF files exist (F16 1.8 MB, Q4_0 621 KB). |

## Confirmed root-cause bugs

### Bug #1 (critical): double label shift — the model was trained to predict token *t+2*

`scripts/train.py::make_batch` returns `x = arr[:, :-1]`, `y = arr[:, 1:]` (already
right-shifted), then calls `model(input_ids=x, labels=y).loss`. HuggingFace's
`GPT2LMHeadModel` **shifts labels again internally** (`ForCausalLMLoss` pads and takes
`labels[..., 1:]`). Net effect: `logits[t]` is trained against `arr[t+2]`.

Proof (deterministic, eval mode, random model):
- reported loss with `labels = arr[:, 1:]` = **3.4486036**
- manual CE with correct shift (logits[t] → token t+1) = **3.4618163**
- manual CE with double shift (logits[t] → token t+2) = **3.4486036** ← exact match

Consequences: every trained checkpoint learned a systematically misaligned objective.
At inference the model's first answer token is really its estimate of the token *after*
next, which produces off-by-one garbage: newline loops after `<|assistant|>`,
repetition cycles (`مهلهی ملهی…`), and — after `.strip()` — the "empty answers" in
10/11 probes. This single bug explains most of the catastrophic generation failure.

### Bug #2 (critical): tokenizer never saw the pretraining corpus

`scripts/train_tokenizer.py` line: `files = [str(D / 'pretrain_train.txt')]` …
`tok.train_from_iterator(files, trainer)`. `train_from_iterator` does **not** open
paths; the corpus file content was excluded from BPE training. The 512-token vocabulary
was fitted to ≈6 KB of instruction text (+ a path string) only.

Consequences (measured on the committed tokenizer):
- Persian: **3.2 tokens/word** on casual/formal text; everyday words split into bytes
  (`نیم‌فاصله` → 9 tokens).
- English: **3.9 tokens/word**; `Hello, how are you doing today` → 20 tokens.
- Overall **1.51 characters/token** (roughly byte-level, i.e. the BPE bought almost
  nothing) — 1.5×–2× worse than a properly trained 8k–16k vocab would achieve, on top
  of the vocabulary being far too small for Persian morphology.

### Bug #3: evaluation/reporting misattributed the empty answers

The prior final report states answers were empty "because it immediately emitted the
end token". That is wrong (see table). The empty strings come from newline-loop
decoding plus `.strip()`. Fixing stopping criteria alone can never fix this checkpoint;
the weights themselves encode the wrong objective.

### Bug #4 (data): corpus far too small even if everything else were correct

8,140 training characters (~2–3k tokens) + 31 instruction pairs. A model cannot
acquire Persian syntax, vocabulary, or conversational behavior from this regardless of
architecture. This is a *fundamental limitation*, not a code bug.

### Secondary issues (confirmed but lower priority)

- **SFT loss not masked**: user turns and control tokens contribute to the loss; the
  model is trained as much to predict user text as assistant text.
- **Validation selected on noisy random windows**: `make_batch` samples one random
  window per eval → best-checkpoint selection is partly noise. (Both stages: reported
  best_val comes from a single random batch.)
- **Format inconsistency**: training joins examples with an extra `'\n'` between
  `format_dialog` blocks (each already ends with `<|end|>\n`), while inference prompts
  do not → small train/inference distribution mismatch.
- **No LR schedule / warmup**; constant LR; no gradient accumulation; no resume;
  dropout 0.05 on a 4-layer model with 350 steps of 1k tokens — undertrained *and*
  already overfitting by step 350 (train 3.42 vs val 4.61).
- **Context limit**: `n_positions = 128`; generation of 48 tokens + long prompts can
  exceed it → `generate` would fail on longer prompts (not hit by the 11 short probes).
- **Cosmetic**: `resid_pdrop` kwarg typo (`resid_pdrop` vs correct `resid_pdrop` is
  accepted positionally? → actually `GPT2Config` uses `resid_pdrop`; `scripts/train.py`
  passes `resid_pdrop=` correctly at model creation but `resid_pdrop` via `GPT2Config`
  as `resid_pdrop` — no issue; noted for clarity that dropout is 0.05 everywhere).

## What is *not* broken

- Architecture choice itself (standard decoder-only GPT-2; llama.cpp supports it).
- Label machinery of HF (correct; our call-site was wrong).
- Special-token IDs are consistent (0 `[UNK]`, 1 BOS, 2 `<|user|>`, 3 `<|assistant|>`,
  4 `<|end|>`) across tokenizer, configs, and generation code.
- Unicode normalization pipeline (NFC + Arabic→Persian yeh/kaf) is sound.
- Checkpoints contain finite, learned, distinct parameters.
- The evaluation perplexity computation in `evaluate.py` uses the correct shift.

## Prioritized plan

1. **Fix the label shift** (train.py) — non-negotiable; all future runs.
2. **Fix tokenizer training** to read actual corpus files; retrain tokenizer with a
   much larger vocabulary (8k–16k) on the expanded corpus (vocab change ⇒ model must
   retrain from scratch — acceptable, requirement is scratch training anyway).
3. **Fix generation/eval** to detect newline loops and repetition, use min-new-tokens,
   and stop on `<|user|>` as well as `<|end|>`; make eval deterministic.
4. **Fix SFT loss masking** (assistant turns + end tokens only), full-sequence
   deterministic validation, warmup + cosine schedule, checkpoint resume.
5. **Expand data by orders of magnitude** from lawful, documented sources.
6. Retrain from random init (bigger corpus, 8k+ vocab, ~5–15M params) and evaluate
   original vs new on a fixed held-out probe suite.
