#!/usr/bin/env bash
# Final stages of the v2 pipeline: SFT -> eval -> GGUF -> native smoke test.
# Run after pretraining completes (model/pretrain_v2/best exists and training stopped).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LLAMA=${LLAMA_CPP:-../llama-cpp}

echo "== 4. SFT =="
$PY scripts/train_v2.py --stage sft \
  --init-from model/pretrain_v2/best --tokenizer model/tok_v2_8k \
  --train-sft data/processed/instructions_train_v2.jsonl \
  --valid-sft data/processed/instructions_valid_v2.jsonl \
  --seq 512 --batch 2 --steps "${SFT_STEPS:-600}" --lr 3e-4 --warmup 60 \
  --dropout 0.05 --seed 77 --out model/sft_v2 --eval-every 50 --save-every 100

echo "== 5. Eval =="
$PY scripts/eval_suite.py --checkpoint model/sft_v2/best --tag sft_v2 --max-new-tokens 64

echo "== 6. GGUF =="
$PY scripts/export_gguf_v2.py --llama-cpp "$LLAMA" \
  --quantize "$LLAMA/build/bin/llama-quantize" --checkpoint model/sft_v2/best

echo "== 7. Native smoke =="
PROMPT='<|user|>\nسلام! حالت چطوره؟\n<|end|>\n<|assistant|>\n'
printf "$PROMPT" > /tmp/ravan_prompt.txt
"$LLAMA/build/bin/llama-cli" -m model/ravan-v2-Q4_0.gguf -f /tmp/ravan_prompt.txt \
  -n 64 --temp 0.75 --top-p 0.9 --repeat-penalty 1.15 \
  --reverse-prompt '<|user|>' -r '<|end|>' --seed 7 -no-cnv 2>&1 | tail -30

echo "== done: model/ravan-v2-f16.gguf model/ravan-v2-Q4_0.gguf =="
