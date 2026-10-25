# Why the scratch-trained model cannot form sensible sentences

Date: 2026-10-25. Companion to `PRODUCTION_REPORT_v2.md`. This answers the
question: *"the only difference is it now uses nonsense words instead of
nonsense letters — why can't it make a full sentence?"*

## Short answer

The v2 model learned **the shape of Persian** (grammar, morphology, script,
chat format — hence "real-looking words") but never learned **the mapping from
a user's meaning to a grounded answer**, because nothing in its size,
training-token count, or supervision budget can produce that mapping from
random initialization. A full sensible sentence requires all tokens in it to
serve one global intent; the model has no representation of intent that
survives more than a few tokens, so each next-token prediction drifts to
locally plausible, globally unrelated text.

## The numbers (all measured, artifacts in `reports/`)

| requirement for "understanding" | what v2 had | shortfall |
|---|---|---|
| Parameters (capacity for syntax+semantics+knowledge) | 13,989,888 | GPT-2-small (already too weak to chat) is 124M — **8.9× more**; usable chat models start ~0.5–1B — **36–70× more** |
| Pretraining tokens (Chinchilla: ~20 tok/param) | 25,088,000 seen | optimal ≈ 280M — **11× undertrained** |
| Instruction-following supervision | 11,430 masked tokens (121 dialogs) | enough to learn *format* (stop tokens 100% correct), not *content grounding* |
| Prior knowledge (a random net starts at zero) | none | pretrained LMs get their world knowledge from **10¹²–10¹³ tokens**; v2 got 2.5×10⁷ — **~100,000× less** |

Evidence from the evaluation artifacts:

- `eval_sft_v2.json`: held-out instruction **NLL/byte 1.4286**, empty 0%,
  no loops — i.e. the model produces *fluent Persian-shaped text*…
- …but reading `samples` shows topical drift: answers borrow frequent
  fragments from training data instead of addressing the prompt. Corpus
  perplexity fell 9.08 → 4.56 while prompt→answer grounding stayed ~0.
  **Likelihood modeling ≠ comprehension.**

## Why exactly a *sentence* breaks

1. **No global plan.** Generation is strictly left-to-right. A coherent
   sentence needs an intent ("I am explaining X because Y") held fixed while
   tokens are emitted. At 14 M parameters the model cannot store an intent
   *and* lexical knowledge *and* syntax simultaneously, so the intent decays
   within a sentence and the next tokens optimize only local plausibility.
2. **Topic drift compounds.** Each sampled token conditions the next; after
   10–20 tokens the conditioning context contains the model's own drift, and
   it reinforces itself (visible as loops or "nonsense words").
3. **Format ≠ semantics.** The 11K-token SFT set was enough for the *wrapper*
   (`<|user|> … <|assistant|> … <|end|>` — stop rates 100%, verified) because
   the wrapper has almost no information content. Meaning requires 3+ orders
   of magnitude more supervision, and supervision only shapes what pretraining
   already represents — a random init represents nothing.
4. **Why words look right now (v2) but didn't in v1:** fixing the tokenizer
   (8k BPE trained on 535M chars of real text) repaired the *inventory* of
   Persian building blocks. v1 broke at the piece level (byte-salad from a
   512-token vocab); v2 pieces are real words — the failure moved one level up,
   to *meaning*. That is exactly what "nonsense words instead of nonsense
   letters" means: each level of repair exposes the next missing ingredient.

## What actually fixes it (and what we did)

At a 4 GB RAM / no-GPU budget, training "understanding" from scratch is not
possible (would need ~10¹² tokens — months of multi-GPU time). The field's
answer — now explicitly authorized — is **start from a pretrained open model**
and only adapt deployment:

- **v3 = Qwen3 / Qwen2.5 / Gemma-3 class models (0.6–1.7 B params)**,
  Apache-2.0 / Gemma license, quantized to Q4_K_M GGUF (~0.4–1.1 GB),
  running on llama.cpp — the same serving stack proven in v2.
- Those models internalized grammar, grounding, and world knowledge during
  pretraining; the measured difference on identical probes is qualitative
  (correct multi-turn recall, correct step-by-step arithmetic on unseen
  problems, coherent Persian — see `reports/PRODUCTION_REPORT_v3.md`).
- Honest limits remain: 1–1.7 B models on 4 GB RAM can still be wrong or
  awkward; they are assistants, oracles no.

## Transport note (why GitHub Actions)

The build sandbox can reach only `github.com`/`api.github.com`/PyPI/npm —
**every model host (huggingface.co, release CDNs) is network-blocked**.
GitHub Actions runners are not blocked, so `.github/workflows/fetch-model.yml`
downloads the GGUFs on a runner, splits them into ≤95 MB chunks (GitHub's
per-file limit), commits them to this branch under `model-fetch/out/`, and
uploads the full files to the **`assets-v3` release** for direct download on
the user's PC. Reassembly anywhere: `model-fetch/assemble.py` (verifies sha256).
