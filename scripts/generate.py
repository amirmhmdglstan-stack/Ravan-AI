"""Generate a sample from the project's own checkpoint for evaluation."""
import argparse, torch
from pathlib import Path
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('--prompt',default='<|user|>\nسلام، خوبی؟\n<|assistant|>\n'); p.add_argument('--max-new-tokens',type=int,default=60); p.add_argument('--checkpoint',choices=['pretrain','sft'],default='sft'); a=p.parse_args()
torch.set_num_threads(2); d=ROOT/'model'/a.checkpoint
m=GPT2LMHeadModel.from_pretrained(d).eval(); t=PreTrainedTokenizerFast.from_pretrained(d)
ids=t(a.prompt,return_tensors='pt').input_ids
with torch.no_grad(): out=m.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=a.max_new_tokens,do_sample=True,temperature=.8,top_p=.9,repetition_penalty=1.1,pad_token_id=t.pad_token_id,eos_token_id=t.convert_tokens_to_ids('<|end|>'))
print(t.decode(out[0],skip_special_tokens=False))
