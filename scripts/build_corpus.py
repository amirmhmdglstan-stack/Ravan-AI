"""Build the Ravan v2 training corpus from documented, lawful sources.

Inputs (any missing source is skipped and recorded):
  - data_raw/fawiki/*.jsonl          Persian Wikipedia 2018 dump as JSONL
                                     (Text-Mining/Persian-Wikipedia-Corpus repo,
                                     Wikipedia content CC BY-SA 4.0)
  - data_raw/mizan.zip               Mizan fa-en parallel corpus (CC-BY-4.0)
  - data_raw/ganjoor/poets/*.json    Ganjoor classical public-domain poetry
                                     (only verse Text; AI summaries excluded)
  - data/raw/pretrain.txt            project-authored starter corpus

Outputs:
  - data/processed/corpus_train.jsonl  {"text","source","lang"} per line
  - data/processed/corpus_valid.jsonl  deterministic 1% content-hash holdout
  - data/processed/corpus_toksample.txt  ~48 MB stratified sample for BPE fitting
  - data/processed/corpus_stats.json  per-source counts/chars/lang

Normalization: NFC, Arabic Yeh/Kaf -> Persian, control chars stripped (ZWNJ
kept), whitespace collapsed, exact-duplicate removal (sha256 of normalized
text, per-source counters kept).
"""
import hashlib
import json
import random
import re
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("/home/user/data_raw")
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(4242)
VALID_EVERY = 100  # every 100th unique doc (by content hash) -> valid

FA_RE = re.compile(r"[؀-ۿ]")
LAT_RE = re.compile(r"[A-Za-z]")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u064a", "\u06cc").replace("\u0649", "\u06cc").replace("\u0643", "\u06a9")
    # ZWNJ glued to surrounding spaces (common normalization for Persian)
    s = s.replace("\u200c ", "\u200c").replace(" \u200c", "\u200c")
    # strip control/format chars except newline, tab, ZWNJ
    s = "".join(c for c in s if c in "\n\t" or c == "\u200c"
                or unicodedata.category(c) not in ("Cc", "Cf"))
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def detect_lang(text: str) -> str:
    fa = len(FA_RE.findall(text))
    lat = len(LAT_RE.findall(text))
    if fa == 0 and lat == 0:
        return "other"
    if lat > fa:
        return "en" if lat > 0.7 * (lat + fa) else "mixed"
    return "fa" if fa > 0.7 * (lat + fa) else "mixed"


class Writer:
    def __init__(self):
        self.seen = set()
        self.stats = {}
        self.tr = open(OUT / "corpus_train.jsonl", "w", encoding="utf-8")
        self.va = open(OUT / "corpus_valid.jsonl", "w", encoding="utf-8")

    def st(self, source):
        return self.stats.setdefault(source, dict(docs=0, chars=0, dup=0, short=0,
                                                  lang={"fa": 0, "en": 0, "mixed": 0, "other": 0}))

    def add(self, text, source, min_chars=1):
        text = normalize(text)
        if len(text) < min_chars:
            self.st(source)["short"] += 1
            return
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in self.seen:
            self.st(source)["dup"] += 1
            return
        self.seen.add(h)
        lang = detect_lang(text)
        st = self.st(source)
        st["docs"] += 1
        st["chars"] += len(text)
        st["lang"][lang] += 1
        rec = json.dumps({"text": text, "source": source, "lang": lang}, ensure_ascii=False)
        if int(h[-2:], 16) % VALID_EVERY == 0:
            self.va.write(rec + "\n")
        else:
            self.tr.write(rec + "\n")

    def close(self):
        self.tr.close()
        self.va.close()


def take_fawiki(w: Writer, limit_docs=None):
    files = sorted(RAW.glob("fawiki/*.jsonl"))
    n = 0
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if limit_docs and n >= limit_docs:
                    return n
                if '"Namespace":0' not in line and '"Namespace": 0' not in line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("Namespace") != 0 or d.get("IsDisambiguationPage"):
                    continue
                text = d.get("Text") or ""
                if len(text) < 400:
                    continue
                w.add(text, "fawiki-2018")
                n += 1
    return n


def take_mizan(w: Writer):
    zp = RAW / "mizan.zip"
    if not zp.exists():
        return 0
    with zipfile.ZipFile(zp) as z:
        with z.open("mizan/mizan_fa.txt") as f:
            import io
            for raw in io.TextIOWrapper(f, encoding="utf-8"):
                w.add(raw.rstrip("\n"), "mizan-fa", min_chars=24)
        with z.open("mizan/mizan_en.txt") as f:
            import io
            for raw in io.TextIOWrapper(f, encoding="utf-8"):
                w.add(raw.rstrip("\n"), "mizan-en", min_chars=40)
    return 1


def take_ganjoor(w: Writer):
    base = RAW / "ganjoor" / "poets"
    if not base.exists():
        return 0
    n = 0
    for fp in base.rglob("*.json"):
        if fp.name.startswith("_") or fp.name == "poet.json":
            continue
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        verses = d.get("Verses")
        if not verses:
            continue
        lines = [v.get("Text", "").strip() for v in sorted(verses, key=lambda v: v.get("VOrder", 0))]
        lines = [l for l in lines if l]
        if len(lines) < 4:
            continue
        # NOTE: only classical verse text; AI-generated CoupletSummary /
        # PoemSummary / biographies are deliberately excluded.
        w.add("\n".join(lines), "ganjoor-pd", min_chars=80)
        n += 1
    return n


def take_starter(w: Writer):
    p = ROOT / "data" / "raw" / "pretrain.txt"
    if not p.exists():
        return 0
    for para in p.read_text(encoding="utf-8").split("\n\n"):
        if para.strip() and not para.startswith("#"):
            w.add(para, "starter", min_chars=40)
    return 1


def build_toksample(w: Writer, target_bytes=48 * 1024 * 1024):
    """Stratified ~48MB sample of training docs for BPE fitting."""
    per_source_budget = {
        "fawiki-2018": int(target_bytes * 0.60),
        "mizan-fa": int(target_bytes * 0.10),
        "mizan-en": int(target_bytes * 0.12),
        "ganjoor-pd": int(target_bytes * 0.13),
        "starter": int(target_bytes * 0.05),
    }
    got = {k: 0 for k in per_source_budget}
    out = open(OUT / "corpus_toksample.txt", "w", encoding="utf-8")
    with open(OUT / "corpus_train.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            s = d["source"]
            budget = per_source_budget.get(s, 0)
            if budget <= 0 or got[s] >= budget:
                continue
            out.write(d["text"] + "\n\n")
            got[s] += len(d["text"]) + 2
            if all(got[k] >= v for k, v in per_source_budget.items()):
                break
    out.close()
    return got


def main():
    w = Writer()
    nf = take_fawiki(w)
    nm = take_mizan(w)
    ng = take_ganjoor(w)
    ns = take_starter(w)
    w.close()
    sample = build_toksample(w)
    stats = dict(
        note="content-hash exact dedup; valid = every 100th unique doc by sha256; "
             "fawiki = Persian Wikipedia 2018 dump JSONL (CC BY-SA 4.0, Text-Mining/"
             "Persian-Wikipedia-Corpus); mizan = CC-BY-4.0 parallel corpus (each side "
             "kept as separate plain-text docs); ganjoor = public-domain classical "
             "verse only (AI summaries excluded); starter = project-authored.",
        sources=w.stats,
        raw_scan=dict(fawiki_docs=nf, mizan=nm, ganjoor_poems=ng, starter=ns),
        toksample_bytes=sample,
    )
    (OUT / "corpus_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
