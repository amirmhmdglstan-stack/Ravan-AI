"""Extract the uploaded curriculum PDFs with a dual-engine per-page chooser.

pypdf and PyMuPDF each win on different pages of these RTL textbooks (font
subsets with broken ToUnicode maps). For every page we extract with both,
score Persian sanity, and keep the better one. Then clean:

- NFC, Arabic yeh/kaf -> Persian, strip harakat, ZWNJ glue, whitespace
- drop page furniture: repeated header/footer lines (appear on >25% of pages)
  and lone page-number lines
- drop pages that are garbage in BOTH engines (private-use / replacement ratio)

Outputs: uploads/<name>.clean.txt, uploads/corpus_books.jsonl, uploads/stats.json
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf
from pypdf import PdfReader

RAW = Path('data_raw/pdfs')
OUT = Path('uploads')
OUT.mkdir(exist_ok=True)

PRIVATE = re.compile(r'[\ue000-\uf8ff\u0000-\u0008\ufffd]')
ARAB = re.compile(r'[\u064b-\u0652\u0670\u0640]')  # harakat + tatweel
HARAKAT_KEEP = re.compile(r'[\u064b-\u0652\u0670]')
FA_RE = re.compile(r'[\u0600-\u06ff]')
LINE_NUM = re.compile(r'^[\s\d۰-۹٠-٩\.\-–—/]{1,6}$')


def clean(t: str) -> str:
    t = unicodedata.normalize('NFC', t)
    t = t.replace('\u064a', '\u06cc').replace('\u0649', '\u06cc')
    t = t.replace('\u0643', '\u06a9').replace('\u0640', '')
    t = ARAB.sub('', t)                      # strip harakat/tatweel
    t = t.replace('\u200c ', '\u200c').replace(' \u200c', '\u200c')
    t = PRIVATE.sub(' ', t)
    t = re.sub(r'[ \t]+', ' ', t)
    return t.strip()


def score(t: str) -> float:
    """Sanity: share of Persian letters among word chars, penalize garbage."""
    if not t or len(t) < 30:
        return -1.0
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return -1.0
    fa = sum(1 for c in letters if '\u0600' <= c <= '\u06ff')
    priv = len(PRIVATE.findall(t))
    return fa / len(letters) - 3.0 * priv / max(200, len(t))


def extract_pages(name: str):
    """-> list of (page_text, engine) using the better engine per page."""
    doc = pymupdf.open(RAW / name)
    reader = PdfReader(str(RAW / name))
    n = len(doc)
    chosen, winners = [], Counter()
    for i in range(n):
        cands = []
        try:
            cands.append(('pypdf', clean(reader.pages[i].extract_text() or '')))
        except Exception:
            pass
        try:
            cands.append(('mupdf', clean(doc[i].get_text() or '')))
        except Exception:
            pass
        if not cands:
            chosen.append('')
            continue
        best_txt, best_eng, best_s = cands[0][1], cands[0][0], score(cands[0][1])
        for eng, txt in cands[1:]:
            s = score(txt)
            if s > best_s:
                best_txt, best_eng, best_s = txt, eng, s
        winners[best_eng] += 1
        chosen.append(best_txt if best_s > 0.15 else '')
    return n, chosen, winners


def drop_furniture(pages):
    """Remove repeated header/footer lines and lone page numbers."""
    line_pages = Counter()
    for pi, t in enumerate(pages):
        seen = set()
        for ln in t.split('\n'):
            k = ln.strip()
            if k and k not in seen:
                seen.add(k)
                line_pages[k] += 1
    thresh = max(4, int(0.25 * len(pages)))
    furniture = {k for k, c in line_pages.items() if c >= thresh and len(k) < 90}
    out = []
    for t in pages:
        kept = []
        for ln in t.split('\n'):
            s = ln.strip()
            if not s:
                continue
            if s in furniture:
                continue
            if LINE_NUM.match(s):
                continue
            kept.append(s)
        out.append('\n'.join(kept))
    return out, len(furniture)


def main():
    stats, corpus = {}, []
    for f in sorted(RAW.glob('*.pdf')):
        n, pages, winners = extract_pages(f.name)
        pages, n_furn = drop_furniture(pages)
        text = '\n\n'.join(p for p in pages if p.strip())
        nonempty = sum(1 for p in pages if p.strip())
        (OUT / (f.stem + '.clean.txt')).write_text(text, encoding='utf8')
        stats[f.name] = dict(pages=n, pages_with_text=nonempty, chars=len(text),
                             engines=dict(winners), furniture_lines_dropped=n_furn)
        # split to paragraphs for the corpus
        for para in re.split(r'\n{2,}', text):
            para = ' '.join(para.split())
            if len(para) >= 40 and sum(1 for c in para if '\u0600' <= c <= '\u06ff') >= 0.2 * len(para):
                corpus.append({'text': para, 'source': f.stem, 'lang': 'fa'})
        print(f'{f.name}: {stats[f.name]}', flush=True)

    (OUT / 'corpus_books.jsonl').write_text(
        ''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in corpus),
        encoding='utf8')
    total = sum(s['chars'] for s in stats.values())
    stats['_TOTAL'] = dict(chars=total, corpus_paragraphs=len(corpus))
    (OUT / 'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf8')
    print('TOTAL:', total, 'chars;', len(corpus), 'paragraphs')


if __name__ == '__main__':
    main()
