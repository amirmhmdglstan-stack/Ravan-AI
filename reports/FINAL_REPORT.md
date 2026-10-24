# Ravan-AI scratch-training report

Date: 2026-10-24. This report distinguishes implemented work from the original quality goal.

## Verdict

A real, newly initialized and newly trained model exists, and the project can export/load it as GGUF with native llama.cpp. It is **not a useful conversation model**. The user asked for excellent natural Iranian Persian and useful bilingual assistance; this experiment falls far short. The artifact should be regarded as a reproducible pipeline smoke test, not a finished AI product.

## Feasibility and architecture

Sandbox inventory: 2 logical CPUs, ~3.8 GiB RAM, no CUDA GPU (`torch.cuda.is_available() == False`), ~20 GiB free disk; training used the CPU. This does not provide the resources or data for meaningful LLM pretraining. Target device details were supplied by the user, not attached to the sandbox.

A decoder-only GPT-2-style Transformer was chosen over custom recurrent/state-space architectures because it has a straightforward random-init implementation in Transformers and a registered llama.cpp GPT-2 converter/runtime. Llama.cpp's current `conversion/gpt2.py` registers `GPT2LMHeadModel`; successful GGUF conversion and native load tested that exact compatibility route. The GGUF converter initially rejected the custom tokenizer's pretokenizer fingerprint; `scripts/export_gguf.py` maps the known project-trained ByteLevel regex to llama.cpp's GPT-2 pretokenizer label, restoring upstream files afterward.

Configuration: 4 decoder layers, hidden dimension 128, 4 heads, 128-token context, trained 512-entry byte-level BPE, tied token/output embeddings, 875,264 learned parameters. This size is a feasibility-driven toy model, not an evidence-based optimum for good language. It is very far smaller and less trained than a genuinely capable Persian chat model would need.

Alternatives considered at a high level: an RWKV/Mamba-style recurrent model may reduce long-context KV-cache cost, but would add architecture/conversion risk and would not solve the data/compute bottleneck. A larger Transformer has more capacity but cannot make scarce CPU training produce quality; 4 GB target RAM would constrain it. This proof-of-pipeline therefore used the smallest practical supported conventional architecture.

## Data and rights

Only 39 short original Persian/English text passages and 39 short original instruction/response pairs were used. Training data: 8,140 pretraining characters, 31 training dialogue pairs; validation: 891 characters, 8 pairs. No external website, benchmark, pretrained tokenizer, or pretrained weight file was used. No Hamyar text was copied. Hamyar was excluded because access alone is not permission and an appropriate training licence was not verified. Full processing and source/rights notes: `data/DATA_MANIFEST.md`.

A 512-entry byte-level BPE tokenizer was fitted from this project's training split only. Preprocessing includes NFC, mapping Arabic Yeh/Kaf to Persian Yeh/Kaf, whitespace cleanup, and duplicate removal. No educational-provider content, scraped social media, or purchased dataset was included. The internal corpus is too small and not independently reviewed by teachers/native reviewers.

## Actual training

- Initialization: fresh `GPT2Config` / `GPT2LMHeadModel`, all learned parameters random, seed 20261024. No checkpoint download or base-model load.
- Pretraining: 325 optimizer steps at 8 examples × 128 tokens; 332,800 token positions presented. Best recorded sampled holdout loss 4.4987 at step 325; step 1 validation loss 6.1259. The 350-step run showed validation loss 4.6063, while train loss continued falling (3.418), an overfitting warning.
- SFT: 175 steps at 8 × 128 tokens (179,200 positions); initialized only from this project's scratch-trained checkpoint. Best sampled validation loss 4.4792 at step 175. A later evaluation on the same held-out instruction stream (993 target tokens) measured NLL 5.2380 / perplexity 188.3 for pretraining and NLL 5.0973 / perplexity 163.6 for SFT. That small reduction is measurable but still represents extremely poor prediction quality; the holdout is tiny.
- The tokenizer/model were not evaluated against an external benchmark and no blinded human rating was collected. Preference optimization was not attempted because there is no reliable preference data and no credible indication it would benefit this run.
- Per-step metrics and exact configuration are in `reports/pretrain_metrics.json`, `reports/sft_metrics.json`; same-stream comparison and prompts are in `reports/model_evaluation.json`.

## Actual generation and native inference

The final HF model was run on 11 fixed probes: Iranian casual Persian, explanation, spelling, English conversation, translation, school math, science, creative writing, reasoning, unknown facts, and context. Greedy decoding produced an empty string in 10 of 11 cases because it immediately emitted the end token. The context sample produced malformed repetition (`‌ ملهی‌ ملهی‌ ...`). These unedited outputs are recorded in `reports/model_evaluation.json`.

Final Q4_0 GGUF size: ~608 KiB; F16 GGUF: ~1.8 MiB. A Linux llama.cpp `llama-server` build from commit `3f03b17` loaded the Q4_0 GGUF and served `/health` successfully. The app HTML returned HTTP 200. A native completion request to the Persian prompt `سلام` produced repeated malformed fragments: `‌ ملهی‌ ملهی‌ ملهی‌ ...`. Thus GGUF compatibility is verified; useful generation is not.

The target PC is specified as a Pentium G3250. Intel's official product specification lists SSE4.1 and SSE4.2; it does not list AVX or AVX2. The native sandbox test build explicitly disabled AVX, AVX2, FMA, F16C and BMI2, and enabled SSE4.2. It ran on this sandbox's Xeon, not the user's exact CPU. Peak server RSS measured here was ~22 MiB with context 128; Windows behavior and speed are unmeasured. No token-per-second claim is made.

## Interface, memory, and boundaries

`app/index.html` is a compact RTL/LTR local UI served by `llama-server --path app`. `windows/run-chat.bat` binds the runtime to loopback. User messages and explicit user-entered memories are stored in browser `localStorage` and can be inspected, edited, exported and deleted; conversation history can be exported or cleared. Browser memory is not encrypted. No external API is called. A saved note is context/memory, not new parameter training. There is no external knowledge retrieval collection and no incremental training process.

LM Studio compatibility is **not verified**. GGUF loading is verified in llama.cpp only. The Windows runtime binary is not present in this checkout because GitHub release-asset redirects were blocked by this sandbox's egress policy. The Persian guide links the official pinned CPU release and documents the G3250 ISA caveat. Consequently a complete ready-to-run Windows binary bundle was not delivered and Windows execution must be tested on the target before claiming portability.

## Next steps for a genuinely useful model

1. Secure licensed Persian contemporary dialogue, conversational and educational data plus sufficient English text; review/attribute each source. Preserve the source-level permissions and remove private/sensitive content.
2. Grow the deduplicated token corpus by orders of magnitude; use native Persian reviewers and an uncontaminated benchmark set.
3. Train on a GPU/cluster with token/step budgets appropriate to the model, compare model size and held-out loss, and retain best checkpoints.
4. Expand high-quality, human-reviewed SFT and preference data; test multi-turn behavior, uncertainty and safety with human ratings.
5. Validate a Windows runtime build without AVX2 on an actual G3250, measure RAM/speed, test the local UI, and test LM Studio separately.

Until these are done, the only justified claim is: **a tiny scratch-trained GPT-2-compatible Persian/English experiment was built and its weights converted to GGUF; quality goals remain unmet.**
