# Reproducing the Ravan v2 pipeline (research machine)

All steps run on Linux/macOS with Python 3.11. The target Windows PC only needs
`llama-server` + the GGUF (see `windows/run-chat-v2.bat`).

```bash
python -m venv .venv && .venv/bin/pip install torch transformers tokenizers numpy gguf safetensors
```

## 1. Raw data (~2.1 GB, not committed)

Raw files live in `data_raw/` (gitignored). Re-downloading takes ~2 minutes with
network access to GitHub:

```bash
mkdir -p data_raw/fawiki
# Persian Wikipedia 2018 dump (5 zips, CC BY-SA 4.0 content) via the contents API:
for part in \
  "fawiki-20181001-pages-articles-multistream%201%20-%20100000.zip" \
  "fawiki-20181001-pages-articles-multistream%20100001%20-%20290169.zip" \
  "fawiki-20181001-pages-articles-multistream%20290170%20-%20580338.zip" \
  "fawiki-20181001-pages-articles-multistream%20580339%20-%20870507.zip" \
  "fawiki-20181001-pages-articles-multistream%20870508%20-%201160676.zip"; do
  gh api "repos/Text-Mining/Persian-Wikipedia-Corpus/contents/Json Format of Persian Wikipedia Pages/$part" \
    -H "Accept: application/vnd.github.raw" > "data_raw/fawiki/${part//%20/ }.zip"
  (cd data_raw/fawiki && unzip -o -q "${part//%20/ }")
done
rm -f data_raw/fawiki/*.zip            # extracted JSONL only; save disk

# MIZAN fa-en parallel corpus (CC BY 4.0):
gh api repos/omidkashefi/Mizan/contents/mizan.zip -H "Accept: application/vnd.github.raw" > data_raw/mizan.zip

# Ganjoor public-domain classical poetry (sparse clone: poets only):
git clone --filter=blob:none --no-checkout https://github.com/ganjoor/ganjoor.git data_raw/ganjoor
cd data_raw/ganjoor && git sparse-checkout init --cone && \
  git sparse-checkout set poets && git checkout && cd ../..
# ensure poets/{ferdousi,saadi,hafez,khayyam,attar,jami,saeb,khajoo,bahar,asadi,anvari} exist
```

## 2. Corpus → tokenizer → streams → SFT

```bash
.venv/bin/python scripts/build_corpus.py         # 535,470,153 chars → corpus_*.jsonl
.venv/bin/python scripts/train_tokenizer_v2.py   # 8k/16k candidates + reports/tokenizer_comparison.json
.venv/bin/python scripts/tokenize_corpus.py --tokenizer model/tok_v2_8k \
    --max-tokens 40000000 --valid-max-tokens 256000   # train.tok / valid.tok
.venv/bin/python scripts/prepare_sft.py          # 121 train / 25 valid, probe-leak controlled
.venv/bin/python scripts/test_pipeline.py        # 26 checks
```

Chosen tokenizer: **tok_v2_8k** — 4.67 bytes/token on probes, 5.37 fa / 3.16 en
bytes/token on held-out corpus, 1.79 tokens/word (v1 was 3.2/3.9). See
`reports/tokenizer_comparison.json`.

## 3. Pretrain (CPU)

```bash
.venv/bin/python scripts/train_v2.py --stage pretrain --arch small --seq 512 \
  --steps 13000 --batch 4 --lr 6e-4 --warmup 600 --dropout 0.0 --seed 4242 \
  --tokenizer model/tok_v2_8k --train-tok data/processed/train.tok \
  --valid-tok data/processed/valid.tok --out model/pretrain_v2 \
  --eval-every 500 --save-every 750
# resume after interruption: same command + --resume
```

Measured: 14.0M params (384d/6L/6H, ctx 512), ~1,600 train tok/s on 2 CPU
threads fp32 (bf16 autocast measured *slower*: 486 tok/s — do not enable).

## 4. SFT from the best pretrain checkpoint

```bash
.venv/bin/python scripts/train_v2.py --stage sft \
  --init-from model/pretrain_v2/best --tokenizer model/tok_v2_8k \
  --train-sft data/processed/instructions_train_v2.jsonl \
  --valid-sft data/processed/instructions_valid_v2.jsonl \
  --seq 512 --batch 2 --steps 600 --lr 3e-4 --warmup 60 --dropout 0.05 \
  --out model/sft_v2 --eval-every 50 --save-every 100
```

## 5. Evaluate

```bash
.venv/bin/python scripts/eval_suite.py --checkpoint model/sft_v2/best --tag sft_v2
.venv/bin/python scripts/eval_suite.py --compare reports/eval_original-v1.json reports/eval_sft_v2.json
.venv/bin/python scripts/generate_v2.py --checkpoint model/sft_v2/best --prompt 'سلام!'
```

## 6. GGUF export + native run (llama.cpp)

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ../llama-cpp
cd ../llama-cpp && cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF && \
  cmake --build build -j2 --target llama-cli llama-server llama-quantize llama-gguf-split
cd -
.venv/bin/python scripts/export_gguf_v2.py --llama-cpp ../llama-cpp \
  --quantize ../llama-cpp/build/bin/llama-quantize --checkpoint model/sft_v2/best
# → model/ravan-v2-f16.gguf, model/ravan-v2-Q4_0.gguf
../llama-cpp/build/bin/llama-cli -m model/ravan-v2-Q4_0.gguf -p "$(printf '<|user|>\nسلام! خوبی؟\n<|end|>\n<|assistant|>\n')" \
  -n 64 --temp 0.75 --top-p 0.9 --reverse-prompt '<|user|>' -s
```

Windows: copy official `llama-server.exe` (+DLLs) next to `windows/run-chat-v2.bat`,
place `model/ravan-v2-Q4_0.gguf`, double-click.

## Notes

- Chat format (v2): every message is `<|role|>\n{content}\n<|end|>\n`, the
  answer slot opens with `<|assistant|>\n`; generation must stop on `<|end|>` or `<|user|>`.
- Special ids: `[UNK]`=0, `endoftext`=1 (bos/eos), `<|user|>`=2, `<|assistant|>`=3, `<|end|>`=4.
- Metric policy: compare tokenizers/checkpoints only via `nll_per_byte`.
- Sandbox resets kill background processes and drop untracked data — commit+push
  code often; raw data is re-downloadable with the commands above.
