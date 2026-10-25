# Data manifest and rights notes

## v2 production corpus (Phase 3, current)

| Dataset/material | Source and licence | Use / processing | Split and size |
|---|---|---|---|
| `data_raw/fawiki/*.jsonl` | Persian Wikipedia 2018 dump, packaged by GitHub `Text-Mining/Persian-Wikipedia-Corpus`; Wikipedia text is **CC BY-SA 4.0** (attribution + ShareAlike). Raw dump files are not committed (re-download ~2 min; see `docs/REPRODUCE_v2.md`). | Namespace-0 articles ≥400 chars; NFC + Arabic Yeh/Kaf→Persian; control chars stripped (ZWNJ kept); sha256 exact-dedup; 1% hash holdout. | 166,965 docs, 401,868,353 chars (train+holdout combined). |
| `data_raw/mizan.zip` | **MIZAN** fa↔en parallel corpus by Omid Kashefi (2018, arXiv:1801.02107), github.com/omidkashefi/Mizan, licensed **CC BY 4.0** (zip layout `mizan/mizan_fa.txt`, `mizan/mizan_en.txt`). Not committed; re-download per `docs/REPRODUCE_v2.md`. | Parallel sides ingested separately as `mizan-fa` / `mizan-en`; min-length filter (fa 24 / en 40 chars); same normalization + dedup. | fa: 865,836 lines / 57,867,626 chars; en: 654,742 / 53,872,547. |
| `data_raw/ganjoor/` | Ganjoor.net public-domain classical poetry (ghazal/hexameter verse of classical poets: ferdousi, saadi, hafez, khayyam, attar, jami, saeb, khajoo, bahar, asadi, anvari). Only `Verses[].Text` by `VOrder` is used; AI summaries and biographies deliberately excluded. | Verse text joined per poem; same normalization + dedup. | 24,150 poems, 21,852,596 chars. |
| `data/raw/pretrain.txt` | Original short Persian/English text written for this repository. | Starter source in the same pipeline. | 39 docs, 9,031 chars. |
| **Total** | | | **535,470,153 chars / 1,711,732 docs** (`data/processed/corpus_stats.json`). `corpus_train.jsonl` 1,691,514 docs; hash holdout `corpus_valid.jsonl` 20,218 docs. |

Tokenized stream (`scripts/tokenize_corpus.py`, tokenizer `model/tok_v2_8k`, seed 1234):
`data/processed/train.tok` = 40,000,016 tokens (379,434 seed-shuffled docs, budget-capped),
`data/processed/valid.tok` = 256,627 tokens (204 held-out docs).

### Attribution and obligations (release requirement)

When distributing this repository, the trained weights, GGUF files, or derivatives, include:

- **Persian Wikipedia contributors** — content under CC BY-SA 4.0; source: https://fa.wikipedia.org (2018 dump via Text-Mining/Persian-Wikipedia-Corpus). ShareAlike and attribution apply to the corpus text; retain this notice with any redistributed corpus files.
- **Mizan corpus** — MIZAN, Omid Kashefi (2018), CC BY 4.0, https://github.com/omidkashefi/Mizan (cite: arXiv:1801.02107).
- **Ganjoor classical verse** — public-domain poems as digitized by Ganjoor (https://ganjoor.net); AI-generated summaries were not used.
- Model outputs are **statistical patterns, not verifiable facts**; the model has weights-only knowledge (no retrieval), no live data, and can be confidently wrong. Do not use it for medical, legal, or safety-critical decisions.

## v2 instruction data (Phase 6)

| File | Provenance | Size |
|---|---|---|
| `data/raw/instructions.jsonl` | Original authoring for this repository (v1 set). | 39 dialogs. |
| `data/raw/instructions_v2_part1..7.jsonl` | **Original dialogs authored for this repository** during the v2 phase (not copied from other assistants or datasets). Covers casual conversation, clarification/uncertainty, school QA, science, grammar, translation, English, creative, multi-turn, instruction-following, tech/health/travel. | 120 dialogs total. |
| `data/processed/instructions_train_v2.jsonl` | Assembled by `scripts/prepare_sft.py`: v1 train minus eval-probe near-duplicates (28 kept of 31) + authored set minus probe duplicates (110 of 120). **Zero user turns match the 21 eval probes.** | 121 dialogs / 250 messages. |
| `data/processed/instructions_valid_v2.jsonl` | v1's original 8 held-out pairs (kept untouched for cross-version comparability) + authored hash-split holdout (1/6). | 25 dialogs. |

Leakage control: exact and fuzzy (difflib ≥0.70) matching of **every** user turn against all 21 probe prompts; a rescue list keeps semantically distinct look-alikes; force-drop list removes rewordings of probes. Recorded in `data/processed/sft_v2_stats.json` and asserted by `scripts/test_pipeline.py` (`test_sft_leak_control`).

## v1 material (kept for comparison)

- `data/raw/pretrain.txt`, `data/raw/instructions.jsonl`, `data/processed/preparation_stats.json`, `data/processed/instructions_train|valid.jsonl` — as originally documented: project-authored, deterministic splits (35/4 and 31/8).

## Material researched, still not included

- Hamyar and other education sites were not scraped: public accessibility is not evidence of permission for machine-learning training.
- Large Persian news/tweet corpora were excluded where provenance, non-commercial restrictions, or derivatives clauses were not sufficiently verified.
- Persian Wikipedia and Mizan ARE now included under the licences above — attribution notices are mandatory (see above).

Before expanding further, record source-level licence compatibility, preserve attribution, and exclude personal/sensitive material. Training on a dataset does not erase its licence obligations.

## Reproducibility

```bash
python scripts/build_corpus.py       # corpus_*.jsonl + corpus_stats.json from data_raw/
python scripts/train_tokenizer_v2.py # model/tok_v2_8k | tok_v2_16k + reports/tokenizer_comparison.json
python scripts/tokenize_corpus.py    # train.tok / valid.tok uint32 streams
python scripts/prepare_sft.py        # instructions_train|valid_v2.jsonl + sft_v2_stats.json
python scripts/test_pipeline.py      # 26 checks incl. leak control
```
