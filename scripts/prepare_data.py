"""Deterministic, conservative preparation of the project-authored starter corpus."""
import hashlib, json, re, unicodedata
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / 'data/raw', ROOT / 'data/processed'
OUT.mkdir(parents=True, exist_ok=True)

def normalize(s):
    s = unicodedata.normalize('NFC', s).replace('\u064a', '\u06cc').replace('\u0649', '\u06cc').replace('\u0643', '\u06a9')
    s = s.replace('\u200c ', '\u200c').replace(' \u200c', '\u200c')
    s = ''.join(c for c in s if c in '\n\t' or unicodedata.category(c) not in ('Cc','Cf') or c == '\u200c')
    s = re.sub(r'[ \t]+', ' ', s)
    return s.strip()

paras=[]; seen=set(); stats={'raw_paragraphs':0,'duplicate_paragraphs':0}
for p in (RAW/'pretrain.txt').read_text(encoding='utf-8').split('\n\n'):
    p=normalize(p)
    if not p or p.startswith('#'): continue
    stats['raw_paragraphs']+=1
    key=hashlib.sha256(re.sub(r'\s+',' ',p).casefold().encode()).hexdigest()
    if key in seen: stats['duplicate_paragraphs']+=1; continue
    seen.add(key); paras.append(p)
# Stable hold-out by document. Tokenizer and all training use train only.
valid=[]; train=[]
for i,p in enumerate(paras): (valid if i%10==0 else train).append(p)
(OUT/'pretrain_train.txt').write_text('\n\n'.join(train)+'\n',encoding='utf-8')
(OUT/'pretrain_valid.txt').write_text('\n\n'.join(valid)+'\n',encoding='utf-8')
items=[json.loads(x) for x in (RAW/'instructions.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
for x in items:
    for m in x['messages']: m['content']=normalize(m['content'])
# Reserve every fifth pair for validation, preventing exact QA-pair leakage.
itr=[x for i,x in enumerate(items) if i%5]; iva=[x for i,x in enumerate(items) if i%5==0]
for name, arr in [('instructions_train.jsonl',itr),('instructions_valid.jsonl',iva)]:
    (OUT/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in arr),encoding='utf-8')
stats.update({'unique_pretrain_documents':len(paras),'pretrain_train_documents':len(train),'pretrain_validation_documents':len(valid),'instruction_pairs':len(items),'instruction_train_pairs':len(itr),'instruction_validation_pairs':len(iva),'train_chars':sum(map(len,train)),'validation_chars':sum(map(len,valid))})
(OUT/'preparation_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False,indent=2))
