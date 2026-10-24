"""Evaluate the actual pretrained-from-scratch and SFT checkpoints on fixed prompts."""
import json, math, torch
from pathlib import Path
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
ROOT=Path(__file__).resolve().parents[1]; M=ROOT/'model'; R=ROOT/'reports'; D=ROOT/'data/processed'
PROBES=[
 ('iranian_casual','سلام، خوبی؟ امروزت چطور بود؟'),
 ('persian_explain','فرق آب‌وهوا و هوا چیه؟'),
 ('persian_grammar','در نوشتار معیار، می خواهم را چطور می‌نویسیم؟'),
 ('english_conversation','Hi! What do you like to do in your free time?'),
 ('translation','ترجمه‌ی طبیعی «دلم برایت تنگ شده» به انگلیسی چیه؟'),
 ('school_math','اگر ۴۰ تا از ۵۰ سؤال را درست جواب بدهم، چند درصد می‌شود؟'),
 ('science','چرا آسمان در روز معمولاً آبی دیده می‌شود؟'),
 ('creative','یک داستان دو جمله‌ای درباره‌ی یک گربه و ماه بنویس.'),
 ('reasoning','آیا همبستگی به‌تنهایی ثابت می‌کند یک چیز علت چیز دیگر است؟'),
 ('uncertainty','پایتخت کشور خیالی زارمان چیست؟'),
 ('context','سلام، اسم من نازنین است.\n<|assistant|>\nسلام نازنین!\n<|user|>\nاسم من چی بود؟'),
]
torch.set_num_threads(2)
tok=PreTrainedTokenizerFast.from_pretrained(M/'sft')
model=GPT2LMHeadModel.from_pretrained(M/'sft').eval()
rows=[]
for name,q in PROBES:
    prompt='<|user|>\n'+q+'\n<|assistant|>\n'
    enc=tok(prompt,return_tensors='pt')
    with torch.no_grad():
        out=model.generate(input_ids=enc['input_ids'],attention_mask=enc['attention_mask'],max_new_tokens=48,do_sample=False,eos_token_id=tok.convert_tokens_to_ids('<|end|>'),pad_token_id=tok.pad_token_id)
    ans=tok.decode(out[0][enc['input_ids'].shape[1]:],skip_special_tokens=False).strip()
    rows.append({'category':name,'prompt':q,'answer':ans})
# Same held-out instruction set, exact tokenizer and token stream for both checkpoints.
valid=[json.loads(s) for s in (D/'instructions_valid.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
text='\n'.join(''.join(f"<|{m['role']}|>\n{m['content']}\n" for m in x['messages'])+'<|end|>' for x in valid)
ids=tok.encode(text,add_special_tokens=False)
def eval_nll(path):
    m=GPT2LMHeadModel.from_pretrained(path).eval(); total=0.0; count=0; context=128
    with torch.no_grad():
        for st in range(0,len(ids)-1,context-1):
            arr=torch.tensor([ids[st:st+context]],dtype=torch.long)
            if arr.shape[1]<2: continue
            logits=m(input_ids=arr[:,:-1]).logits
            loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),arr[:,1:].reshape(-1),reduction='sum')
            total+=float(loss); count+=arr.shape[1]-1
    return {'tokens':count,'nll':total/count,'perplexity':math.exp(min(50,total/count))}
comparison={k:eval_nll(M/k) for k in ('pretrain','sft')}
result={'note':'Tiny internal diagnostic; not a human-rated benchmark. Validation data is project-authored and held out by pair/document; very small.', 'instruction_validation_comparison':comparison,'samples':rows}
R.mkdir(exist_ok=True); (R/'model_evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
