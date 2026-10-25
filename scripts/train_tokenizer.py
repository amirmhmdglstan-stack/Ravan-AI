"""Train this project's byte-level BPE tokenizer using only its training split."""
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'data/processed'; OUT=ROOT/'model'
OUT.mkdir(exist_ok=True)
tok=Tokenizer(models.BPE(unk_token='[UNK]'))
tok.normalizer=normalizers.Sequence([normalizers.NFC()])
tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder=decoders.ByteLevel()
trainer=trainers.BpeTrainer(vocab_size=512,min_frequency=2,special_tokens=['[UNK]','<|endoftext|>','<|user|>','<|assistant|>','<|end|>'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=True)
files=[str(D/'pretrain_train.txt')]
# Include SFT prompts/responses in tokenizer training without validation examples.
import json
for line in (D/'instructions_train.jsonl').read_text(encoding='utf8').splitlines():
    for m in json.loads(line)['messages']: files.append(m['content'])
tok.train_from_iterator(files,trainer=trainer)
tok.save(str(OUT/'tokenizer.json'))
print('vocab_size',tok.get_vocab_size())
