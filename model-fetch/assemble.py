"""Assemble full model files from in-repo chunks (model-fetch/out/*.part-*).

Runs in the sandbox: fetches each <=95MB chunk through the GitHub contents API
(allowed host), concatenates to model-fetch/assembled/<filename>, verifies the
committed sha256, and prints per-file status.

Usage: .venv/bin/python model-fetch/assemble.py [--only id]
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'model-fetch' / 'out'
DEST = ROOT / 'model-fetch' / 'assembled'
MANIFEST = ROOT / 'model-fetch' / 'manifest.json'
BRANCH = 'arena/3d3c3d6b-ravan-ai'
REPO = 'amirmhmdglstan-stack/Ravan-AI'


def gh_raw(path: str) -> bytes:
    r = subprocess.run(
        ['gh', 'api', f'repos/{REPO}/contents/{path}', '-H', 'Accept: application/vnd.github.raw'],
        capture_output=True, check=True)
    return r.stdout


def parts_for(entry_id: str):
    # list via API so we never need a local git pull of the chunk blobs
    r = subprocess.run(['gh', 'api', f'repos/{REPO}/contents/model-fetch/out?ref={BRANCH}'],
                       capture_output=True, check=True)
    names = [x['name'] for x in json.loads(r.stdout)]
    return sorted(n for n in names
                  if n.startswith(entry_id + '.part-') and n.endswith(tuple('0123456789')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    DEST.mkdir(parents=True, exist_ok=True)
    failures = 0
    for e in manifest['entries']:
        eid, fname = e['id'], e['filename']
        if a.only and a.only != eid:
            continue
        dest = DEST / fname
        try:
            sha_file = OUT / f'{eid}.sha256'
            if not sha_file.exists():
                # fetch sha from repo (may not be local after no-pull workflow)
                expected = gh_raw(f'model-fetch/out/{eid}.sha256').decode().split()[0]
            else:
                expected = sha_file.read_text().split()[0]
        except Exception as ex:
            print(f'[fail] {eid}: sha256 file not found: {ex}')
            failures += 1
            continue

        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == expected:
            print(f'[ok]   {eid}: already assembled at {dest}')
            continue

        try:
            plist = parts_for(eid)
        except Exception as ex:
            print(f'[fail] {eid}: cannot list parts: {ex}')
            failures += 1
            continue
        if not plist:
            print(f'[wait] {eid}: no chunks in repo yet (workflow may still be running)')
            failures += 1
            continue

        h = hashlib.sha256()
        tmp = dest.with_suffix(dest.suffix + '.part'
                               if dest.suffix else dest.name + '.part')
        with tmp.open('wb') as f:
            for i, pn in enumerate(plist, 1):
                data = gh_raw(f'model-fetch/out/{pn}?ref={BRANCH}')
                f.write(data)
                h.update(data)
                print(f'       {eid}: {pn} ({len(data)} B) [{i}/{len(plist)}]', flush=True)
        got = h.hexdigest()
        if got != expected:
            print(f'[fail] {eid}: sha256 mismatch got={got} want={expected}')
            tmp.unlink(missing_ok=True)
            failures += 1
            continue
        tmp.replace(dest)
        print(f'[ok]   {eid}: assembled {dest} ({dest.stat().st_size} B)')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
