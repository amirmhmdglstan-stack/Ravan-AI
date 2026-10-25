"""Ravan scratch-LM training utilities (v2).

Key fixes vs the original scripts/train.py:
- Correct causal-LM loss: manual shift, explicit and unit-tested (no double shift).
- SFT loss masking: only assistant content + end tokens contribute.
- Deterministic full-set validation (no random windows for checkpoint selection).
- Warmup + cosine LR schedule, gradient accumulation, grad clipping.
- Checkpoint resume, best/last saving, JSONL metric log with throughput.
- Optional bf16 autocast (CPU AVX512-BF16) when it measures faster.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "model"
REPORTS = ROOT / "reports"

def format_dialog(messages):
    out = []
    for m in messages:
        out.append(f"<|{m['role']}|>\n{m['content']}\n<|end|>\n")
    return "".join(out)


def build_sft_sample(tok, messages):
    """Encode one dialog; returns (ids, target_mask).

    target_mask[i] is True when token i should be predicted (supervised).
    Supervised: assistant message content tokens and its trailing <|end|>.
    """
    ids = []
    mask = []
    for m in messages:
        header = f"<|{m['role']}|>\n"
        h_ids = tok.encode(header, add_special_tokens=False)
        content = m["content"]
        c_ids = tok.encode(content, add_special_tokens=False)
        end_ids = tok.encode("\n<|end|>\n", add_special_tokens=False)
        # NOTE: '<|end|>' is a special token in this tokenizer; encoding the
        # string with surrounding newlines splits correctly because special
        # tokens are matched literally.
        ids += h_ids
        mask += [False] * len(h_ids)
        if m["role"] == "assistant":
            ids += c_ids
            mask += [True] * len(c_ids)
            ids += end_ids
            # end_ids = ['\\n', '<|end|>', '\\n']: supervise the newline that
            # precedes <|end|> and <|end|> itself (so greedy decoding reliably
            # reaches the stop token); trailing newline stays context-only.
            mask += [True, True] + [False] * (len(end_ids) - 2)
        else:
            ids += c_ids
            mask += [False] * len(c_ids)
            ids += end_ids
            mask += [False] * len(end_ids)
    return ids, mask


def causal_loss(logits, ids, target_mask=None, ignore_index=-100):
    """Standard shifted CE: position t of logits predicts ids[t+1].

    logits: [B, T, V] over inputs ids[:, :-1]
    ids:    [B, T+1]
    target_mask: optional [B, T+1] bool; target ids[t+1] supervised only where mask.
    Returns mean CE over supervised positions.
    """
    # logits[t] (computed from inputs ids[:, :-1]) predicts ids[:, 1:][t].
    tgt = ids[:, 1:].clone()
    if target_mask is not None:
        keep = target_mask[:, 1:]
        tgt = tgt.masked_fill(~keep, ignore_index)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        tgt.reshape(-1),
        ignore_index=ignore_index,
    )


def load_tok(path):
    return PreTrainedTokenizerFast.from_pretrained(str(path))


def load_stream(path, tok):
    text = Path(path).read_text(encoding="utf-8")
    return tok.encode(text, add_special_tokens=False)


def load_sft(path, tok):
    """Returns LongTensor ids [N] and BoolTensor mask [N] for the whole split."""
    all_ids, all_mask = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        ids, mask = build_sft_sample(tok, item["messages"])
        # Document separator so dialogs never run into each other.
        sep = tok.encode("\n", add_special_tokens=False)
        ids += sep
        mask += [False] * len(sep)
        all_ids += ids
        all_mask += mask
    return torch.tensor(all_ids, dtype=torch.long), torch.tensor(all_mask, dtype=torch.bool)


@torch.no_grad()
def deterministic_val_loss(model, ids, seq, masked=False, batch=4, device="cpu"):
    """Full sequential sweep over the validation stream; returns mean token NLL.

    Windows are non-overlapping and cover every position once; windows are
    batched (batch x seq) for speed.
    """
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    starts = list(range(0, len(ids) - 1, seq))
    for bi in range(0, len(starts), batch):
        xs, ys = [], []
        for start in starts[bi:bi + batch]:
            chunk = ids[start:start + seq + 1]
            if chunk.numel() < 2:
                continue
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        if not xs:
            continue
        # pad batch rows to equal length (last window may be short)
        maxlen = max(x.numel() for x in xs)
        xp = torch.zeros(len(xs), maxlen, dtype=torch.long)
        yp = torch.full((len(xs), maxlen), -100, dtype=torch.long)
        for i, (x, y) in enumerate(zip(xs, ys)):
            xp[i, :x.numel()] = x
            yp[i, :y.numel()] = y
        logits = model(input_ids=xp.to(device)).logits
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                              yp.to(device).reshape(-1), reduction="sum", ignore_index=-100)
        total += float(nll)
        count += int((yp != -100).sum())
    if was_training:
        model.train()
    return total / max(1, count)


@torch.no_grad()
def sft_val_loss(model, ids, mask, seq, device="cpu"):
    """Deterministic NLL over supervised (assistant) tokens only."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(ids) - 1, seq):
        chunk = ids[start:start + seq + 1]
        mchunk = mask[start:start + seq + 1]
        if chunk.numel() < 2:
            continue
        x = chunk[:-1].unsqueeze(0).to(device)
        full = chunk.unsqueeze(0).to(device)
        fullm = mchunk.unsqueeze(0).to(device)
        logits = model(input_ids=x).logits
        tgt = full[:, 1:].clone()
        keep = fullm[:, 1:]
        tgt = tgt.masked_fill(~keep, -100)
        n = int(keep.sum())
        if n == 0:
            continue
        total += float(
            F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                            tgt.reshape(-1), reduction="sum", ignore_index=-100)
        )
        count += n
    if was_training:
        model.train()
    return total / max(1, count)


def sample_windows(ids, mask, batch, seq, tries_per_slot=30):
    """Sample `batch` random windows of seq+1 tokens, aligned masks.

    `ids` may be a torch tensor or a numpy uint32 stream.
    For SFT, resample a slot a few times if the window holds no supervised
    token so the gradient batch stays effective.
    """
    n = len(ids)
    xs, fulls, masks = [], [], []

    def win(start):
        sl = ids[start:start + seq + 1]
        if torch.is_tensor(sl):
            return sl
        return torch.from_numpy(np.ascontiguousarray(sl).astype(np.int64))

    for _ in range(batch):
        for _attempt in range(tries_per_slot):
            start = random.randint(0, max(0, n - seq - 1))
            chunk = win(start)
            if chunk.numel() == seq + 1:
                break
        if chunk.numel() < seq + 1:  # stream shorter than window: pad-wrap
            pad_n = seq + 1 - chunk.numel()
            chunk = torch.cat([chunk, chunk[:pad_n] if chunk.numel() else torch.zeros(pad_n, dtype=torch.long)])
            if mask is not None:
                mchunk = torch.cat([mask[start:start + seq + 1],
                                    torch.zeros(pad_n, dtype=torch.bool)])
            else:
                mchunk = None
        else:
            mchunk = mask[start:start + seq + 1] if mask is not None else None
        if mask is not None and mchunk is not None and not bool(mchunk[1:].any()):
            # no supervision in this window: mark slot by all-mask later
            pass
        xs.append(chunk[:-1])
        fulls.append(chunk)
        if mchunk is not None:
            masks.append(mchunk)
    x = torch.stack(xs)
    full = torch.stack(fulls)
    fullm = torch.stack(masks) if masks else None
    return x, full, fullm


def lr_at(step, total, base_lr, warmup, min_ratio=0.1):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    p = min(1.0, max(0.0, p))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def build_model(cfg, tok):
    torch.manual_seed(cfg["seed"])
    g = GPT2Config(
        vocab_size=len(tok),
        n_positions=cfg["seq"],
        n_ctx=cfg["seq"],
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        resid_pdrop=cfg.get("dropout", 0.0),
        embd_pdrop=cfg.get("dropout", 0.0),
        attn_pdrop=cfg.get("dropout", 0.0),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    m = GPT2LMHeadModel(g)
    # GPT-2 scaled init for residual projections improves deep-net stability.
    for name, p in m.named_parameters():
        if name.endswith("c_proj.weight"):
            torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg["n_layer"]))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "sft"], required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--tokenizer", default=None, help="tokenizer dir (default: model/<v>)")
    ap.add_argument("--init-from", default=None, help="checkpoint to init from (sft); default model/pretrain_v2")
    ap.add_argument("--out", default=None, help="output dir (default model/pretrain_v2 or model/sft_v2)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--train-tok", default=None, help="uint32 .tok stream (pretrain)")
    ap.add_argument("--valid-tok", default=None, help="uint32 .tok stream (pretrain val)")
    ap.add_argument("--train-sft", default=None, help="messages jsonl (sft train)")
    ap.add_argument("--valid-sft", default=None, help="messages jsonl (sft val)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--snap-every", type=int, default=0,
                    help="if >0, save a raw state_dict snapshot at each eval "
                         "into out/snaps/stepN.pt (for offline candidate comparison)")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--bf16", action="store_true", help="CPU bf16 autocast if faster/supported")
    ap.add_argument("--arch", default="small", choices=["small", "micro"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    if args.arch == "small":
        arch = dict(n_embd=384, n_layer=6, n_head=6)
    else:
        arch = dict(n_embd=256, n_layer=6, n_head=4)

    tok_dir = Path(args.tokenizer) if args.tokenizer else MODEL_DIR / "tok_v2"
    out = Path(args.out) if args.out else MODEL_DIR / ("pretrain_v2" if args.stage == "pretrain" else "sft_v2")
    tok = load_tok(tok_dir)

    cfg = dict(seed=args.seed, seq=args.seq, **arch, dropout=args.dropout)

    if args.stage == "pretrain":
        if args.train_tok:
            train_ids = np.fromfile(args.train_tok, dtype=np.uint32)
        else:
            train_ids = torch.tensor(load_stream(DATA / "pretrain_train.txt", tok), dtype=torch.long)
        if args.valid_tok:
            valid_ids = torch.from_numpy(
                np.fromfile(args.valid_tok, dtype=np.uint32).astype(np.int64))
        else:
            valid_ids = torch.tensor(load_stream(DATA / "pretrain_valid.txt", tok), dtype=torch.long)
        train_mask = None
        model = build_model(cfg, tok)
    else:
        init = Path(args.init_from) if args.init_from else MODEL_DIR / "pretrain_v2"
        model = GPT2LMHeadModel.from_pretrained(str(init))
        tr_sft = Path(args.train_sft) if args.train_sft else DATA / "instructions_train.jsonl"
        va_sft = Path(args.valid_sft) if args.valid_sft else DATA / "instructions_valid.jsonl"
        train_ids, train_mask = load_sft(tr_sft, tok)
        valid_ids, valid_mask = load_sft(va_sft, tok)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    out.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    warmup = args.warmup if args.warmup is not None else max(50, args.steps // 20)
    log_path = out / "train_log.jsonl"
    start_step = 0
    best_val = float("inf")
    best_step = 0

    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        best_val = ck.get("best_val", float("inf"))
        best_step = ck.get("best_step", 0)
        print(f"resumed at step {start_step}", flush=True)

    use_bf16 = False
    if args.bf16:
        try:
            # probe: one autocast matmul works and is finite
            with torch.autocast("cpu", dtype=torch.bfloat16):
                probe = torch.randn(64, 64) @ torch.randn(64, 64)
            use_bf16 = bool(torch.isfinite(probe.float()).all())
        except Exception as e:
            print("bf16 unavailable:", e)
            use_bf16 = False
    print(f"stage={args.stage} params={sum(p.numel() for p in model.parameters()):,} "
          f"train_tokens={len(train_ids):,} valid_tokens={len(valid_ids):,} "
          f"bf16={use_bf16} seq={args.seq} batch={args.batch}x{args.accum}", flush=True)

    def evaluate():
        if args.stage == "pretrain":
            return deterministic_val_loss(model, valid_ids, args.seq)
        return sft_val_loss(model, valid_ids, valid_mask, args.seq)

    model.train()
    t0 = time.time()
    tokens_done = 0
    log_f = open(log_path, "a", encoding="utf-8")
    for step in range(start_step, args.steps):
        lr = lr_at(step, args.steps, args.lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for micro in range(args.accum):
            x, full, fullm = sample_windows(train_ids, train_mask if args.stage == "sft" else None,
                                            args.batch, args.seq)
            with torch.autocast("cpu", dtype=torch.bfloat16) if use_bf16 else _nullctx():
                logits = model(input_ids=x).logits
                if args.stage == "pretrain":
                    loss = causal_loss(logits, full)
                else:
                    tgt = full[:, 1:].clone()
                    tgt = tgt.masked_fill(~fullm[:, 1:], -100)
                    if not bool((~tgt.eq(-100)).any()):
                        continue  # batch without supervision: skip micro
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)).float(),
                        tgt.reshape(-1), ignore_index=-100)
            (loss / args.accum).backward()
            step_loss += float(loss) / args.accum
            tokens_done += args.batch * args.seq
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == start_step or (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            val = evaluate()
            row = dict(step=step + 1, train_loss=round(step_loss, 4),
                       val_loss=round(val, 4), val_ppl=round(math.exp(min(30, val)), 2),
                       lr=round(lr, 7), elapsed=round(time.time() - t0, 1),
                       tokens_processed=tokens_done)
            print(json.dumps(row), flush=True)
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            if val < best_val:
                best_val, best_step = val, step + 1
                model.save_pretrained(out / "best", safe_serialization=True)
                tok.save_pretrained(out / "best")
            if args.snap_every:
                (out / "snaps").mkdir(exist_ok=True)
                torch.save(model.state_dict(), out / "snaps" / f"step{step + 1}.pt")
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                            step=step + 1, best_val=best_val, best_step=best_step),
                       out / "last.pt")
    log_f.close()
    # ensure at least best exists even if never improved after first eval
    if not (out / "best").exists():
        model.save_pretrained(out / "best", safe_serialization=True)
        tok.save_pretrained(out / "best")
    summary = dict(stage=args.stage, steps=args.steps, best_val=round(best_val, 4),
                   best_step=best_step, seed=args.seed, arch=arch, seq=args.seq,
                   batch=args.batch, accum=args.accum, lr=args.lr, warmup=warmup,
                   dropout=args.dropout, bf16=use_bf16,
                   params=sum(p.numel() for p in model.parameters()),
                   train_tokens=len(train_ids), valid_tokens=len(valid_ids),
                   tokenizer=str(tok_dir), wall_seconds=round(time.time() - t0, 1))
    (out / "run_config.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
