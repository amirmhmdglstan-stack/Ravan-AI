"""Export this custom ByteLevel-BPE GPT-2 checkpoint using llama.cpp.

Example: python scripts/export_gguf.py --llama-cpp C:/src/llama.cpp --quantize C:/src/llama.cpp/build/bin/llama-quantize.exe
"""
import argparse, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--llama-cpp',required=True); p.add_argument('--quantize',required=True); p.add_argument('--checkpoint',default='sft'); a=p.parse_args()
root=Path(__file__).resolve().parents[1]; repo=Path(a.llama_cpp); base=repo/'conversion'/'base.py'; backup=base.read_text(encoding='utf8')
marker='        if res is None:\n            logger.warning("\\n")'
patch='        # Ravan-AI trained ByteLevel BPE: same regex behavior as GPT-2.\n        if chkhsh == "6dc3f90b199a230f4c10a3c551b33ae59a9b5cffc297919a2fa1e1a5efc8b800":\n            res = "gpt-2"\n\n'
if marker not in backup: raise SystemExit('Unrecognized llama.cpp converter revision; inspect get_vocab_base_pre before exporting.')
try:
    base.write_text(backup.replace(marker,patch+marker,1),encoding='utf8')
    f16=root/'model'/'ravan-sft-f16.gguf'; q4=root/'model'/'ravan-sft-Q4_0.gguf'
    subprocess.run([sys.executable,str(repo/'convert_hf_to_gguf.py'),str(root/'model'/a.checkpoint),'--outfile',str(f16),'--outtype','f16'],check=True)
    subprocess.run([a.quantize,str(f16),str(q4),'Q4_0'],check=True)
finally:
    base.write_text(backup,encoding='utf8')
print(f'Wrote {f16} and {q4}')
