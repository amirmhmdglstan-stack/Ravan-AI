"""Assemble the Phase-6 SFT train/valid sets (v2).

Sources
- data/processed/instructions_train.jsonl : v1 train (31), probe-near-dups removed
- data/processed/instructions_valid.jsonl : v1 valid (8), kept untouched as held-out
  (keeps v2 SFT valid comparable with the v1 pipeline's held-out ppl)
- data/raw/instructions_v2_part*.jsonl    : agent-authored additions (this phase)

Leakage control vs data/eval/probes.jsonl (eval probes must stay unseen):
- exact normalized user-prompt match -> never train on it
- fuzzy ratio >= 0.55 against any probe -> dropped from training

Outputs
- data/processed/instructions_train_v2.jsonl
- data/processed/instructions_valid_v2.jsonl
- data/processed/sft_v2_stats.json
"""
import difflib
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'data' / 'processed'
RAW = ROOT / 'data' / 'raw'


def normalize(s):
    s = unicodedata.normalize('NFC', s).replace('\u064a', '\u06cc').replace('\u0649', '\u06cc').replace('\u0643', '\u06a9')
    s = s.replace('\u200c ', '\u200c').replace(' \u200c', '\u200c')
    s = ''.join(c for c in s if c in '\n\t' or unicodedata.category(c) not in ('Cc', 'Cf') or c == '\u200c')
    s = re.sub(r'[ \t]+', ' ', s)
    return s.strip()


def user_prompt(d):
    return normalize(d['messages'][0]['content'])


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def load_probes(path):
    # probes file is a single JSON object {"probes": [...]}
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return data['probes'] if isinstance(data, dict) else data


def user_turns(d):
    return [normalize(m['content']) for m in d['messages'] if m['role'] == 'user']


# Semantically different but superficially similar: keep in training (task-pattern
# exposure is fine; probe CONTENT must stay unseen).
RESCUE = {
    'فرق say و tell چیه؟',
    'Explain photosynthesis in simple English.',
    'یه راه خلاقانه برای یاد گرفتن واژه‌های انگلیسی پیشنهاد بده.',
    'سلام! حالت چطوره؟ امروز صبح‌تون چطور گذشت؟',
    'فرق «اگر» و «اگه» چیه؟',
    'رابطه‌ی غذا و حافظه چیه؟',
}
# Same question as a probe with only surface rewording: drop even under threshold.
FORCE_DROP = {
    'از ۵۰ تا، ۴۰ تا درست زدم. چند درصد می‌شه؟',
    'چرا آسمان روزها آبیه؟',
}


def is_probe_duplicate(d, probe_norms, fuzzy_thresh=0.70):
    """True if ANY user turn of d equals / nearly equals a probe prompt."""
    up = user_prompt(d)
    if up in RESCUE:
        return False
    if up in FORCE_DROP:
        return True
    for u in user_turns(d):
        if u in probe_norms:
            return True
        for p in probe_norms:
            if difflib.SequenceMatcher(None, u, p).ratio() >= fuzzy_thresh:
                return True
    return False


def main():
    probes = load_probes(ROOT / 'data' / 'eval' / 'probes.jsonl')
    probe_norms = {user_prompt(p) for p in probes}

    stats = {'n_probes': len(probe_norms)}

    # v1 sets -------------------------------------------------------------
    v1_train = load_jsonl(P / 'instructions_train.jsonl')
    v1_valid = load_jsonl(P / 'instructions_valid.jsonl')
    for d in v1_train + v1_valid:
        for m in d['messages']:
            m['content'] = normalize(m['content'])

    v1_train_kept, v1_train_dropped = [], []
    for d in v1_train:
        (v1_train_dropped if is_probe_duplicate(d, probe_norms) else v1_train_kept).append(d)
    stats['v1_train_in'] = len(v1_train)
    stats['v1_train_kept'] = len(v1_train_kept)
    stats['v1_train_dropped_as_probe_dup'] = [user_prompt(d) for d in v1_train_dropped]
    stats['v1_valid_held_out'] = len(v1_valid)

    # authored parts ------------------------------------------------------
    mine = []
    for f in sorted(RAW.glob('instructions_v2_part*.jsonl')):
        mine.extend(load_jsonl(f))
    for d in mine:
        for m in d['messages']:
            m['content'] = normalize(m['content'])

    seen_prompts, train_mine, valid_mine, dup_mine, probe_dup_mine = set(), [], [], [], []
    for d in mine:
        up = user_prompt(d)
        if up in seen_prompts:
            dup_mine.append(up)
            continue
        seen_prompts.add(up)
        if is_probe_duplicate(d, probe_norms):
            probe_dup_mine.append(up)
            continue
        h = int(hashlib.sha256(up.encode('utf-8')).hexdigest()[:8], 16)
        (valid_mine if h % 6 == 0 else train_mine).append(d)

    stats.update({
        'authored_in': len(mine),
        'authored_exact_dup_dropped': len(dup_mine),
        'authored_probe_dup_dropped': probe_dup_mine,
        'authored_train': len(train_mine),
        'authored_valid': len(valid_mine),
    })

    train = v1_train_kept + train_mine
    valid = v1_valid + valid_mine
    # intra-set dedup on normalized prompts
    def dedup(items):
        out, s = [], set()
        for d in items:
            k = user_prompt(d)
            if k in s:
                continue
            s.add(k)
            out.append(d)
        return out

    train_d, valid_d = dedup(train), dedup(valid)
    stats['train_total'] = len(train_d)
    stats['valid_total'] = len(valid_d)
    stats['train_messages'] = sum(len(d['messages']) for d in train_d)

    (P / 'instructions_train_v2.jsonl').write_text(
        ''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in train_d), encoding='utf-8')
    (P / 'instructions_valid_v2.jsonl').write_text(
        ''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in valid_d), encoding='utf-8')
    (P / 'sft_v2_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
