# Data manifest and rights notes

## Material actually used

| Dataset/material | Source and licence | Use / processing | Split and size |
|---|---|---|---|
| `data/raw/pretrain.txt` | Original short Persian/English text written specifically for this repository; no external corpus passages copied. No external dataset licence applies. Repository itself currently has no declared code/data licence. | NFC normalization; Arabic ي/ك variants mapped to Persian ی/ک; whitespace normalized; exact normalized duplicate check; document-level deterministic 90/10 holdout. | 39 unique short passages; 35 train (8,140 chars), 4 validation (891 chars). |
| `data/raw/instructions.jsonl` | Original authoring for this repository, not copied from another assistant or dataset. | 39 short one-turn instruction/answer examples, spanning Persian colloquial chat, science, school explanations, translation, English, creative writing and uncertainty. 4/5 train, 1/5 holdout. | 31 train pairs, 8 validation pairs. |

The corpus is deliberately tiny and was not independently reviewed by native-language educators. It is enough to prove tokenizer/model/training/export plumbing, not to train a competent bilingual assistant. No third-party scraped text is included in weights or tracked data.

## Researched, but not included

- Hamyar and other education sites were not scraped: public accessibility is not evidence of permission for machine-learning training. No clear training licence/authorization was verified for Hamyar content.
- Persian Wikipedia is potentially reusable under CC BY-SA 4.0, with attribution, modification notices and ShareAlike obligations. It was not downloaded or included here; formal and encyclopedic prose also does not solve colloquial dialogue quality.
- The Dakshina corpus is described by Google Research as CC BY-SA 4.0; it was not fetched or used.
- Large Persian web/news/tweet corpora were excluded where source provenance, individual website rights, non-commercial restrictions, or derivatives clauses were not sufficiently verified.

Before expanding the dataset, obtain and record source-level permission and licence compatibility, preserve attribution, and exclude personal/sensitive material. Training on a dataset does not automatically erase its licence obligations.

## Reproducibility

`python scripts/prepare_data.py` creates the normalized split and `data/processed/preparation_stats.json`. `scripts/train_tokenizer.py` fits the project BPE tokenizer using only training passages/dialogues. No validation examples are fed into tokenizer training.
