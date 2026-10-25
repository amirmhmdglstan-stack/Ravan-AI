"""Export a v2 Ravan checkpoint (ByteLevel BPE, custom vocab) to GGUF via llama.cpp.

llama.cpp identifies BPE pre-tokenizers by hashing the token ids a probe string
encodes to (sha256 of str(list_of_ids)). Our trained vocab produces an unknown
hash, so we patch conversion/base.py at runtime to map OUR hash to the "gpt-2"
pre-tokenizer (identical ByteLevel+regex behavior), run the converter, quantize
to Q4_0, then restore the original file.

Example:
  python scripts/export_gguf_v2.py --llama-cpp /path/to/llama.cpp \
      --quantize /path/to/build/bin/llama-quantize --checkpoint model/sft_v2/best
"""
import argparse
import re
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

from transformers import PreTrainedTokenizerFast

p = argparse.ArgumentParser()
p.add_argument('--llama-cpp', required=True)
p.add_argument('--quantize', required=True)
p.add_argument('--checkpoint', default='model/sft_v2/best')
p.add_argument('--out', default=None, help='output gguf base name (without f16/Q4_0 suffix)')
a = p.parse_args()

root = Path(__file__).resolve().parents[1]
repo = Path(a.llama_cpp)
base = repo / 'conversion' / 'base.py'
ckpt = root / a.checkpoint

# --- our hash: replicate get_vocab_base_pre's chktok computation -------------
src = base.read_text(encoding='utf8')
m = re.search(r"chktxt = ('(?:[^'\\]|\\.)*')\n", src)
if not m:
    raise SystemExit('Could not locate chktxt in conversion/base.py; inspect get_vocab_base_pre.')
chktxt = ast_literal = None
import ast
chktxt = ast.literal_eval(m.group(1))
tok = PreTrainedTokenizerFast.from_pretrained(str(ckpt))
chktok = tok.encode(chktxt)
chkhsh = sha256(str(chktok).encode()).hexdigest()
print(f'our chkhsh: {chkhsh} (vocab={len(tok)})')

if f'"{chkhsh}"' in src:
    print('hash already present in converter; no patch needed')
    patched = False
else:
    backup = src
    marker = '        if res is None:\n            logger.warning("\\n")'
    if marker not in src:
        raise SystemExit('Unrecognized llama.cpp converter revision; inspect get_vocab_base_pre.')
    patch = (f'        # Ravan-AI v2 trained ByteLevel BPE: same regex behavior as GPT-2.\n'
             f'        if chkhsh == "{chkhsh}":\n'
             f'            res = "gpt-2"\n\n')
    base.write_text(src.replace(marker, patch + marker, 1), encoding='utf8')
    patched = True

out_base = Path(a.out) if a.out else root / 'model' / 'ravan-v2'
f16 = out_base.with_name(out_base.name + '-f16.gguf')
q4 = out_base.with_name(out_base.name + '-Q4_0.gguf')
try:
    subprocess.run([sys.executable, str(repo / 'convert_hf_to_gguf.py'), str(ckpt),
                    '--outfile', str(f16), '--outtype', 'f16'], check=True)
    subprocess.run([a.quantize, str(f16), str(q4), 'Q4_0'], check=True)
finally:
    if patched:
        base.write_text(backup, encoding='utf8')

print(f'Wrote {f16} and {q4}')
