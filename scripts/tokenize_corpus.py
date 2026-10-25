"""Tokenize the cleaned corpus into flat uint32 streams for fast training.

- shuffles documents with a fixed seed (source mix stays proportional)
- encodes doc + document separator ('\\n\\n') with the chosen tokenizer
- stops at --max-tokens for the train stream (bounded RAM / epoch definition)
- valid stream: all held-out docs up to --valid-max-tokens

Outputs: data/processed/train.tok, data/processed/valid.tok (uint32 LE)
plus data/processed/tokenized_stats.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'data' / 'processed'


def stream_encode(docs, tok, budget, shuffle_seed=None):
    """docs: list of plain text strings."""
    order = np.arange(len(docs))
    if shuffle_seed is not None:
        np.random.RandomState(shuffle_seed).shuffle(order)
    out = np.empty(min(budget, 8_000_000), dtype=np.uint32)
    total = 0
    n_docs = 0

    def push(arr):
        nonlocal out, total
        need = total + len(arr)
        if need > out.size and need > out.size:
            newcap = max(need, int(out.size * 1.6), budget)
            out = np.resize(out, newcap)
        out[total:total + len(arr)] = arr
        total += len(arr)

    sep = tok.encode('\n\n', add_special_tokens=False)
    for i in order:
        if total >= budget:
            break
        ids = tok.encode(docs[i], add_special_tokens=False)
        if not ids:
            continue
        push(np.asarray(ids, dtype=np.uint32))
        push(np.asarray(sep, dtype=np.uint32))
        n_docs += 1
    return out[:total], n_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', default='model/tok_v2_8k')
    ap.add_argument('--max-tokens', type=int, default=40_000_000)
    ap.add_argument('--valid-max-tokens', type=int, default=256_000)
    ap.add_argument('--seed', type=int, default=1234)
    a = ap.parse_args()

    tok = PreTrainedTokenizerFast.from_pretrained(a.tokenizer)
    train_docs = [json.loads(l)['text']
                  for l in (P / 'corpus_train.jsonl').open(encoding='utf-8')]
    valid_docs = [json.loads(l)['text']
                  for l in (P / 'corpus_valid.jsonl').open(encoding='utf-8')]
    tr, ntr = stream_encode(train_docs, tok, a.max_tokens, shuffle_seed=a.seed)
    va, nva = stream_encode(valid_docs, tok, a.valid_max_tokens)
    tr.tofile(P / 'train.tok')
    va.tofile(P / 'valid.tok')
    stats = dict(tokenizer=a.tokenizer, seed=a.seed,
                 train_tokens=int(tr.size), train_docs=ntr,
                 valid_tokens=int(va.size), valid_docs=nva,
                 note='docs are seed-shuffled before encoding; train stream is a '
                      'random document subset capped at max_tokens')
    (P / 'tokenized_stats.json').write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
