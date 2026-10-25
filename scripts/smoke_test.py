"""Local structural smoke test; does not imply language quality."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
required=[ROOT/'model/tokenizer.json',ROOT/'model/pretrain/model.safetensors',ROOT/'model/sft/model.safetensors',ROOT/'model/ravan-sft-f16.gguf',ROOT/'model/ravan-sft-Q4_0.gguf',ROOT/'app/index.html']
missing=[str(p.relative_to(ROOT)) for p in required if not p.is_file() or p.stat().st_size==0]
if missing: raise SystemExit(f'Missing/empty deliverables: {missing}')
import torch
from transformers import GPT2LMHeadModel,PreTrainedTokenizerFast
for name in ('pretrain','sft'):
    m=GPT2LMHeadModel.from_pretrained(ROOT/'model'/name)
    count=sum(p.numel() for p in m.parameters())
    assert count==875264,(name,count)
    assert m.config.vocab_size==512
    print(name,'parameters=',count,'vocab=',m.config.vocab_size)
t=PreTrainedTokenizerFast.from_pretrained(ROOT/'model/sft')
assert len(t)==512
print('tokenizer',len(t),'Persian round-trip:',t.decode(t.encode('سلام')))
for p in required: print('OK',p.relative_to(ROOT),p.stat().st_size,'bytes')
print('Structural smoke test passed. Quality is separately documented as poor.')
