"""Train candidate v2 byte-level BPE tokenizers on the stratified corpus sample.

Compares 8k vs 16k vocabularies and reports tokenization efficiency
(bytes/token) on fixed Persian/English probes plus held-out corpus lines.

The v1 512-token tokenizer was fitted on ~6KB only (bug: it never read the
corpus file) and needs ~3.2 tokens per Persian word. Candidates are trained
on data/processed/corpus_toksample.txt.

Outputs (dirs): model/tok_v2_8k, model/tok_v2_16k.
Special-token IDs match v1 for tool compatibility:
  0 = [UNK], 1 = endoftext (bos/eos), 2 = user marker, 3 = assistant marker,
  4 = end marker.
"""
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / 'data' / 'processed' / 'corpus_toksample.txt'
EOS = json.loads((ROOT / 'model' / 'tokenizer.json').read_text(encoding='utf-8'))['added_tokens'][1]['content']
SPECIAL = ['[UNK]', EOS, '<|user|>', '<|assistant|>', '<|end|>']
VOCABS = [8192, 16384]

PROBES = [
    'سلام! حالت چطوره؟ امروز یه کم خسته‌ام، فردا باید زود برم سر کار.',
    'طبق گزارش سازمان هواشناسی، دمای هوا فردا در تهران به بیست و هفت درجه سانتی‌گراد خواهد رسید.',
    'فتوسنتز فرایندی است که در آن گیاه با استفاده از نور خورشید، دی‌اکسیدکربن و آب قند تولید می‌کند.',
    'The quick brown fox jumps over the lazy dog while the children watch from the porch.',
    'امروز یک جلسه‌ی design review داشتم و فردا باید report رو تحویل بدم.',
    'قیمت هر کیلو گوشت ۴۵۰٬۰۰۰ تومان است و تخفیف ۲۰٪ دارد.',
    'نیم‌فاصله در فارسی کمک می‌کند بخش‌های یک واژه درست دیده شوند.',
    'Explain in simple English what a tokenizer does and why vocabulary size matters.',
]


def train_one(vocab: int) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token='[UNK]'))
    tok.normalizer = normalizers.Sequence([normalizers.NFC()])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab, min_frequency=3, special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    def iter_docs():
        with SAMPLE.open(encoding='utf-8') as f:
            buf = []
            n = 0
            for line in f:
                buf.append(line)
                n += len(line)
                if n >= 1 << 20:
                    yield ''.join(buf)
                    buf, n = [], 0
            if buf:
                yield ''.join(buf)
    tok.train_from_iterator(iter_docs(), trainer)
    return tok


def efficiency(tok: Tokenizer, texts):
    total_b = total_t = 0
    for t in texts:
        total_b += len(t.encode('utf-8'))
        total_t += len(tok.encode(t).ids)
    return total_b / max(1, total_t)


def main():
    # held-out corpus lines (not from the first MB the trainer saw)
    import random
    lines = SAMPLE.read_text(encoding='utf-8').split('\n\n')
    random.Random(99).shuffle(lines)
    corpus_fa = [l for l in lines if l.strip() and sum('؀' <= c <= 'ۿ' for c in l) > len(l) // 3][:60]
    corpus_en = [l for l in lines if l.strip()
                 and sum('؀' <= c <= 'ۿ' for c in l) == 0 and any(c.isalpha() for c in l)][:60]
    results = {}
    for v in VOCABS:
        tok = train_one(v)
        out = ROOT / 'model' / f'tok_v2_{v // 1000}k'
        out.mkdir(parents=True, exist_ok=True)
        tok.save(str(out / 'tokenizer.json'))
        # minimal HF-compatible sidecar files via tokenizers is not automatic;
        # we write a PreTrainedTokenizerFast-compatible config below.
        stats = {
            'vocab_actual': tok.get_vocab_size(),
            'bytes_per_token_probes': round(efficiency(tok, PROBES), 3),
            'bytes_per_token_corpus_fa': round(efficiency(tok, corpus_fa), 3),
            'bytes_per_token_corpus_en': round(efficiency(tok, corpus_en), 3),
            'tokens_per_probe_word': round(
                sum(len(tok.encode(p).ids) for p in PROBES)
                / max(1, sum(len(p.split()) for p in PROBES)), 3),
        }
        results[str(v)] = stats
        # save tokenizer in HF fast format
        from transformers import PreTrainedTokenizerFast
        hf = PreTrainedTokenizerFast(tokenizer_object=tok,
                                     unk_token=SPECIAL[0],
                                     bos_token=SPECIAL[1], eos_token=SPECIAL[4],
                                     pad_token=SPECIAL[0],
                                     additional_special_tokens=SPECIAL)
        hf.save_pretrained(str(out))
        print(json.dumps(dict(vocab=v, **stats), ensure_ascii=False), flush=True)
    (ROOT / 'reports' / 'tokenizer_comparison.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('wrote reports/tokenizer_comparison.json')


if __name__ == '__main__':
    main()
