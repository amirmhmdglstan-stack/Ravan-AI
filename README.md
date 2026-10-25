# Ravan-AI — scratch-trained Persian-first model experiment

**This repository contains real model weights initialized randomly and trained here; it is not a fine-tune of Qwen, Llama, GPT, or another pretrained model.** It also contains a tokenizer trained on this project's small text corpus, pretraining and SFT checkpoints, and exported GGUF files. The work proves an end-to-end training/export pipeline, **but it does not meet the requested quality goal**: the trained model's answers are mostly gibberish or empty. Do not mistake it for a useful assistant or trust its answers.

## Honest status

- **Completed:** hardware/resource inspection; GPT-2 architecture selection; custom byte-level BPE tokenizer; original, source-documented Persian/English starter corpus; deterministic preparation and holdout; random initialization; causal-LM pretraining; SFT from that project's own checkpoint; training/validation loss tracking; HF SafeTensors checkpoints; F16 and Q4_0 GGUF export; native llama.cpp CPU loading and HTTP chat-completion endpoint test; Persian RTL web UI with local browser history and user-approved editable/exportable local memories.
- **Not completed:** broad legally licensed corpus acquisition, high-quality/full-scale training, human evaluation, target-PC testing, LM Studio verification, and inclusion of a Windows inference binary in this checkout. The sandbox could access GitHub's API/repository but could not retrieve release assets redirected to `release-assets.githubusercontent.com`; instructions below link the official release for one-time Windows runtime acquisition.
- **Quality:** inadequate. Greedy HF evaluation emitted empty strings for 10/11 probe cases; the remaining case repeated broken Persian fragments. A native GGUF request returned repeated malformed Persian fragments. The held-out instruction perplexity is still about 164 after SFT. This is not natural Iranian Persian and is not a functional answer engine.

## Files

- `model/sft/model.safetensors` — final post-trained weights (875,264 parameters).
- `model/pretrain/model.safetensors` — best pretraining checkpoint selected by held-out loss.
- `model/tokenizer.json` and tokenizer metadata in both checkpoint folders — trained byte-level BPE tokenizer, 512 vocabulary entries.
- `model/ravan-sft-f16.gguf`, `model/ravan-sft-Q4_0.gguf` — converted native inference formats. The Q4_0 file is about 0.6 MiB.
- `data/raw/`, `data/processed/`, `data/DATA_MANIFEST.md` — tiny original data, normalized splits, rights/provenance record.
- `scripts/` — deterministic data prep, tokenizer fitting, random-init training, evaluation, sample generation, GGUF export.
- `reports/` — recorded losses and actual model outputs.
- `app/index.html` — simple offline local-chat UI. Chat and explicit memories stay in the browser's local storage; no remote API is used.
- `windows/run-chat.bat`, `docs/INSTALL_FA.md` — local Windows launcher and Persian instructions. You must first place a compatible official llama.cpp Windows runtime next to the launcher.

## Model and training facts

GPT-2-style causal decoder: 4 layers, width 128, 4 attention heads, context 128, 512 BPE tokens, tied token/output embeddings, 875,264 parameters. It is an architecture choice, not imported GPT-2 weights. The model was constructed from a fresh `GPT2Config`; all parameters were initialized randomly (seed 20261024), pretrained for 325 steps, and SFT-trained for 175 steps using only this project's own checkpoint. Best checkpoints are selected on a held-out split. No preference optimization was run: there is no reliable preference dataset or quality evidence to justify it.

Training device: 2 logical CPU workers, 3.8 GiB visible RAM, ~20 GiB free filesystem, no usable CUDA GPU (`torch.cuda.is_available() == False`). Small run times were tens of seconds on the sandbox host, not on the target PC. Dataset and model are toy-scale; short-run loss reduction is not evidence of broad language competence.

Full tracked data: 8,140 pretraining characters for training plus 891 validation; 31 training and 8 validation instruction pairs. Exact counts and methods are recorded in `data/processed/preparation_stats.json` and `data/DATA_MANIFEST.md`.

## Training from scratch (research machine only)

The target Windows PC does **not** need Python. For research reproduction, use a separate machine with Python 3.11, PyTorch 2.2.2 executed on CPU, Transformers, and Tokenizers. (The standard PyPI wheel may install unused CUDA libraries; this experiment had no usable GPU.)

```bash
python -m venv .venv
# activate it, then:
pip install -r requirements-training.txt
python scripts/prepare_data.py
python scripts/train_tokenizer.py
python scripts/train.py --stage pretrain --steps 350
python scripts/train.py --stage sft --steps 200
python scripts/evaluate.py
```

The script keeps the best validation-loss checkpoint. SFT loads the current repository's pretrained checkpoint; it never downloads a model. More steps on this tiny dataset quickly overfit; use substantially larger licensed data and revise evaluation before scaling training.

GGUF conversion needs a llama.cpp checkout with the GPT-2 converter and a built `llama-quantize`; the helper applies a tokenizer-pretokenizer recognition mapping for this project's trained GPT-2 ByteLevel regex, then restores upstream converter code:

```bash
python scripts/export_gguf.py --llama-cpp /path/to/llama.cpp --quantize /path/to/llama.cpp/build/bin/llama-quantize
```

This conversion was tested against llama.cpp commit `3f03b17` (0.7.0-dev). Native load and generation were verified on Linux. LM Studio itself and Windows were not tested.

## CPU, memory, and portability

Intel's specification for the Pentium G3250 lists 64-bit, SSE4.1 and SSE4.2; it does **not** list AVX/AVX2. Do not run an AVX2-only binary. The sandbox native test build explicitly disabled AVX, AVX2, FMA, F16C and BMI2 and enabled SSE4.2, matching the target's listed ISA capability. The target must still be validated on-device. The sandbox's own Xeon is not the target CPU.

The 0.6 MiB Q4_0 weights are tiny; the tested Linux llama-server process used ~22 MiB RSS with the tiny context and app, but this is not a Windows measurement. A conservative target estimate for native model/runtime working memory is under a few hundred MiB; Windows, browser and background apps are the larger 4-GiB constraint. No honest response-speed estimate is available because the exact PC was not benchmarked, and this model has no practical language capability to justify a speed claim.

The official Windows runtime download is a one-time network step; after it, the model, UI, history and memories work locally. Download a Windows x64 CPU package from [llama.cpp release b11768](https://github.com/ggml-org/llama.cpp/releases/tag/b11768) (`llama-b11768-bin-win-cpu-x64.zip`) or a newer official release, extract `llama-server.exe` and every sibling DLL beside `windows/run-chat.bat`, then run it. The generic CPU package/build must not require AVX2. If it fails with `Illegal instruction`, stop using it and rebuild llama.cpp with SSE4.2 only (`GGML_NATIVE=OFF`, `GGML_AVX=OFF`, `GGML_AVX2=OFF`, `GGML_FMA=OFF`, `GGML_F16C=OFF`, `GGML_BMI2=OFF`, `GGML_SSE42=ON`) or use a confirmed baseline-compatible binary. This release asset was not downloaded/tested in this environment.

The local UI connects only to the same-origin llama-server at `127.0.0.1`; the supplied batch file binds it to loopback. Do not expose the server to your LAN or the public internet. Browser `localStorage` is not encrypted; do not put secrets there. Export memories/history before clearing browser data. A saved memory is a local note supplied as context, **not a change in model weights**. There is no retrieval corpus and no incremental learning loop.

## Research notes and sources

The choice favored a small conventional decoder that has a direct llama.cpp GPT-2 converter/runtime path, rather than a novel architecture that would need custom inference. The official llama.cpp `conversion/gpt2.py` registers `GPT2LMHeadModel`; this repository's GGUF load proves the selected converter route works. llama.cpp official runtime is separate from the trained weights.

Selected references (reviewed 2026-10-24):

- [Intel Pentium G3250 specifications](https://www.intel.com/content/www/us/en/products/sku/83538/intel-pentium-processor-g3250-3m-cache-3-20-ghz/specifications.html)
- [llama.cpp source and releases](https://github.com/ggml-org/llama.cpp)
- [Google Dakshina corpus repository](https://github.com/google-research-datasets/dakshina) — CC BY-SA 4.0 reported by the project; not downloaded/used.
- [Persian Wikipedia reuse conditions](https://creativecommons.org/licenses/by-sa/4.0/) — potentially reusable with attribution/share-alike obligations; not used.

See `data/DATA_MANIFEST.md` for why Hamyar and other publicly visible sites were not scraped. No blanket permission to ingest site content was assumed.
