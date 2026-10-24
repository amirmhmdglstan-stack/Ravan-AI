"""Train a randomly initialized compact GPT-2 decoder, then SFT its own weights."""
import argparse, json, math, os, random, time
from pathlib import Path
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'data/processed'; M=ROOT/'model'; R=ROOT/'reports'

def text_examples(path, tok):
    s=path.read_text(encoding='utf8')
    ids=tok.encode(s,add_special_tokens=False)
    return torch.tensor(ids,dtype=torch.long)
def pairs(path): return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines() if x.strip()]
def format_dialog(x):
    return ''.join(f"<|{m['role']}|>\n{m['content']}\n" for m in x['messages'])+'<|end|>\n'
def make_batch(data, batch, seq, device):
    max_start=max(1,len(data)-seq-1)
    ix=torch.randint(max_start,(batch,))
    arr=torch.stack([data[int(i):int(i)+seq+1] for i in ix])
    return arr[:,:-1].to(device),arr[:,1:].to(device)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['pretrain','sft'],required=True); ap.add_argument('--steps',type=int,default=220); ap.add_argument('--seed',type=int,default=20261024); args=ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); torch.set_num_threads(2)
    device='cpu'; tok=PreTrainedTokenizerFast(tokenizer_file=str(M/'tokenizer.json'),unk_token='[UNK]',bos_token='<|endoftext|>',eos_token='<|end|>',pad_token='[UNK]',additional_special_tokens=['<|user|>','<|assistant|>'])
    seq=128; batch=8
    if args.stage=='pretrain':
        config=GPT2Config(vocab_size=len(tok),n_positions=seq,n_ctx=seq,n_embd=128,n_layer=4,n_head=4, resid_pdrop=0.05,embd_pdrop=0.05,attn_pdrop=0.05,bos_token_id=tok.bos_token_id,eos_token_id=tok.eos_token_id)
        model=GPT2LMHeadModel(config) # every learned parameter freshly randomized here
        train=text_examples(D/'pretrain_train.txt',tok); valid=text_examples(D/'pretrain_valid.txt',tok)
        out=M/'pretrain'; lr=3e-4
    else:
        model=GPT2LMHeadModel.from_pretrained(str(M/'pretrain')) # only this project's scratch-trained checkpoint
        train=text_examples(D/'instructions_train.jsonl',tok) if False else torch.tensor(tok.encode('\n'.join(format_dialog(x) for x in pairs(D/'instructions_train.jsonl'))),dtype=torch.long)
        valid=torch.tensor(tok.encode('\n'.join(format_dialog(x) for x in pairs(D/'instructions_valid.jsonl'))),dtype=torch.long)
        out=M/'sft'; lr=1e-4
    model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.1)
    out.mkdir(parents=True,exist_ok=True); R.mkdir(parents=True,exist_ok=True)
    metrics=[]; t0=time.time(); model.train(); best_val=float('inf'); best_step=0
    for step in range(1,args.steps+1):
        x,y=make_batch(train,batch,seq,device); opt.zero_grad(set_to_none=True)
        loss=model(input_ids=x,labels=y).loss; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step==1 or step%25==0 or step==args.steps:
            model.eval()
            with torch.no_grad():
                vx,vy=make_batch(valid,min(batch,max(1,len(valid)-seq-1)),seq,device); vloss=float(model(input_ids=vx,labels=vy).loss)
            row={'step':step,'train_loss':float(loss),'validation_loss':vloss,'elapsed_seconds':round(time.time()-t0,2),'processed_tokens':step*batch*seq}
            metrics.append(row); print(json.dumps(row),flush=True)
            if vloss < best_val:
                best_val, best_step = vloss, step
                model.save_pretrained(out,safe_serialization=True); tok.save_pretrained(out)
            model.train()
    (R/f'{args.stage}_metrics.json').write_text(json.dumps({'stage':args.stage,'steps':args.steps,'batch_size':batch,'sequence_length':seq,'seed':args.seed,'processed_tokens':args.steps*batch*seq,'best_validation_loss':best_val,'best_step':best_step,'metrics':metrics},indent=2)+'\n')
    print('saved',out,'parameters',sum(p.numel() for p in model.parameters()),'device',device)
if __name__=='__main__': main()
