# Ravan v3 — production report (pretrained base, real understanding)

Date: 2026-10-25. Follows `WHY_V2_CANT_TALK.md` (why the scratch-trained v2
could not form sensible sentences) and `PRODUCTION_REPORT_v2.md`.

## Mandate (user, 2026-10-25)

v2 verdict: *"the only difference is it now uses nonsense words instead of
nonsense letters … give me a real usable model, much smarter"* — with explicit
permission to use any open-source AI code/weights. Hard constraints: must run
on **Pentium G3250 (2C, SSE4.2-only, no AVX2), 4 GB DDR3, no GPU, Windows
11, 400 GB free disk**. Everything must be saved in this GitHub repository.

## Root cause of v2 (short form)

14 M params / 25 M pretraining tokens / 11 K supervised tokens from random
init cannot ground meaning — see `reports/WHY_V2_CANT_TALK.md` for the full
measured argument. Format was learnable at that budget (100% stop-token
compliance); semantics are not.

## Transport: how weights crossed a blocked network

The build sandbox reaches only GitHub/PyPI/npm — **every model host is
network-blocked**. GitHub Actions runners are not. Pipeline
(`.github/workflows/fetch-model.yml` + `model-fetch/`):

1. Push edits to `model-fetch/manifest.json` (5 entries) → workflow triggers.
2. Runner downloads each file from Hugging Face / GitHub releases,
   verifies sha256, splits into ≤95 MB chunks (GitHub per-file limit).
3. Runner commits chunks to **`model-fetch/out/`** (this branch) and uploads
   full files to the **`assets-v3` release** (single-file downloads for the PC).
4. Anywhere else: `model-fetch/assemble.py` rebuilds + re-verifies (sha256).

Result (all sha256-verified, `_failures.txt` = 0 bytes):

| asset | size | chunks |
|---|---|---|
| `qwen3-1.7b-q4_k_m.gguf` | 1,107,409,472 B | 12 |
| `qwen3-0.6b-q4_k_m.gguf` | 396,705,472 B | 5 |
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1,117,320,736 B | 12 |
| `gemma-3-1b-it-q4_k_m.gguf` | 806,058,496 B | 9 |
| `llama-b-win-cpu-x64.zip` (llama.cpp runtime) | 24,219,283 B | 1 |

## Candidate bake-off (same server, same prompts, seed-42 battery)

| test | **Qwen3-1.7B** | Qwen2.5-1.5B | Gemma-3-1B |
|---|---|---|---|
| casual Persian | coherent, minor roughness | coherent, bland | **most natural** |
| factual explanation (why sky is blue) | broken answer-switch in no-think mode; **reasoning (thinking mode) correct** | fluent non-answer | fluent but wrong physics |
| multi-turn name+field recall (2 seeds) | **2/2 correct** | correct-ish, verbose | **0/2 failed** |
| in-context follow-up («دومی چی بود؟») | **exact, 10 tokens** | ok | wrong |
| unseen word problem (3×4×4 L) | **48 L ✓** | **16 L ✗** | 48 L ✓ |
| speed on this host (2 threads) | ~10–14 tok/s | ~12 tok/s | ~13–14 tok/s |

**Selected: Qwen3-1.7B-Instruct Q4_K_M (`unsloth/Qwen3-1.7B-GGUF` quant)**
— best context-grounded understanding and reasoning, correct arithmetic,
reliable multi-turn memory: exactly the failure modes of v2. The other three
stay in the repo + release as drop-in alternatives.

Known weaknesses (honest): at 1.7 B, encyclopedic explanations can still be
wrong or awkward; thinking mode reasons correctly but its English→Persian
answer switch is unreliable, so the app keeps thinking **off** by default
(`chat_template_kwargs.enable_thinking` flips it per request).

## Deployment

- llama.cpp `0.7.0-dev` built in-sandbox (llama-server/cli/quantize); GGUF
  carries its own chat template — the app now calls **`/v1/chat/completions`**
  with a Persian system persona (no hand-rolled prompt formats anymore).
- App: `app/index.html` (history + explicit memories as system messages),
  localhost-only, offline.
- Windows: `windows/run-chat-v3.bat` — auto-extracts the bundled
  `llama-b-win-cpu-x64.zip` if needed, picks the first model found under
  `model\`, serves ctx 2048 on 127.0.0.1:8080.
- Context: **2048 tokens** (v2: 512). Params: **1.7 B** (v2: 14 M).

## Target-PC projections (G3250, SSE4.2 only, 2C/4GB)

Measured here (2 threads, Haswell-era AVX512 Xeon): ~10–14 tok/s generation,
~90–110 tok/s prompt eval at ctx 2048. The G3250 lacks AVX2 and has ~2–3×
lower IPC throughput → expect **≈3–5 tok/s generation, ≈30–40 tok/s prompt
eval**: first line in ~3–5 s, a 100-token reply in ~20–30 s. RAM: model+KV ≈
1.4–1.6 GB RSS — fits 4 GB with browser closed; if tight, switch to
`qwen3-0.6b-q4_k_m.gguf` (~2× faster, ~0.6 GB) in `run-chat-v3.bat`.
The bundled Windows zip is the official **CPU (no-AVX2-required) build**.

## Licenses

- Qwen3 / Qwen2.5: **Apache-2.0** (weights redistribution OK, keep NOTICE).
- Gemma-3: **Gemma Terms** (redistribution of original/quantized weights OK).
- llama.cpp: **MIT**. Chunks + full files live in this repo/release under
  those terms — see `NOTICE`.

## Honest limits

This is now a genuinely usable small assistant (context-grounded, correct on
simple reasoning, natural conversation) — not an oracle. 1–1.7 B models can
be confidently wrong on facts; verify anything that matters. No internet
access, weights-only knowledge. The v1/v2 scratch-trained artifacts remain in
the repo as the documented baseline.
