# Business Entity Resolution: Amazon ML Challenge 2026

Pipeline: **normalise → token + embedding candidate search → learned pre-filter → pair features →
level-1 XGBoost + fine-tuned multilingual cross-encoder → level-2 XGBoost (3-fold out-of-fold by S1) →
one-to-one assignment → F0.5-tuned decision → `output/`**.
Best result: leaderboard 0.98046, train out-of-fold 0.98667. See `Documentation_template.md` for the
method and results.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.11
pip install -r code/business_entity_resolution/requirements.txt
# GPU steps (embedding neighbours / features, cross-encoder):
pip install torch --index-url https://download.pytorch.org/whl/cu126 && pip install sentence-transformers
```

Data layout: `dataset/train/train_source{1,2,3}.tsv`, `dataset/train/train_ground_truth.tsv`,
`dataset/test/test_source{1,2,3}.tsv`.

## Run (from the repo / submission root)

```bash
SRC=code/business_entity_resolution/src
python3 $SRC/pipeline.py data  --data-dir dataset --work-dir work                   # cached, incremental
python3 $SRC/pipeline.py stack --data-dir dataset --work-dir work --out-dir output  # cross-encoder + model
# (or: `crossenc`, then `model` = train + decide + predict)
python3 utils/validate_submission.py ...                                            # official checker
```

`full` runs `data` (which ends with `crossenc`) and then `model`. Every data step skips itself when its outputs are newer than its inputs, and
`--rebuild STEP` forces one step and everything downstream of it. The GPU steps (`embblock`,
`embed`, `crossenc`) are skipped automatically without a CUDA device. `decide` re-tunes the decision
rule and re-predicts with the trained models. For large runs on Modal, see `remote/launch.py`.

Main flags:

| flag | default | meaning |
|---|---|---|
| `--drop-s1` | 0.19 | share of train S1 entities removed, so train has test's distractor density |
| `--k-folds` | 3 | folds by S1 entity |
| `--gbm` | auto | XGBoost on GPU, otherwise LightGBM |
| `--k-emb` | 5 | embedding neighbours per record |
| `--budget` | 4 | pre-filter pairs per record |
| `--min-gain` | 2e-4 | out-of-fold gain a per-S1 decision rule needs over the global threshold |
| `--ce` | auto | cross-encoder features (auto = on when a GPU is present) |
| `--ce-train` | 2,000,000 | cross-encoder training pairs per fold |
| `--ce-k-top`, `--ce-p1-min` | 1, 0.05 | pairs the cross-encoder scores: each record's top pairs by pre-filter probability, plus every pair above this |
| `--ce-continue` | 0 | continue training the saved cross-encoder models on this many new pairs per fold (with `--rebuild crossenc`) |

## Source layout

| file | role |
|---|---|
| `src/normalize.py` | transliteration, legal / address abbreviation tables (US / IN / FR), core name, phonetic key, house number, postal code, DBA split, blocking tokens |
| `src/synonyms.py` | token synonyms learned from training pairs (Indic transliterations, city aliases) |
| `src/prep.py` | streams the TSVs, normalises in parallel, writes `work/prep/*.parquet`; train S1 subsampling |
| `src/block.py` | IDF-weighted inverted-index candidate search (reverse + forward top-k) per country |
| `src/embblock.py`, `src/embed.py` | multilingual MiniLM name embeddings: nearest-neighbour candidates and similarity features (GPU) |
| `src/prefilter.py` | cheap LightGBM keeping ~2.5 pairs per record |
| `src/features2.py` | 136 pair features (fuzzy, script, typo, IDF overlap, numbers, frequency, context ranks / gaps) |
| `src/stage2.py`, `src/gbm.py` | 2-level XGBoost / LightGBM with out-of-fold stacking, threshold tuning, prediction |
| `src/decide2.py` | per-S1 decision rules (threshold / two-threshold / expected-F0.5) tuned on out-of-fold probabilities |
| `src/crossenc.py` | multilingual MiniLM fine-tuned as a pair cross-encoder on raw name/address text, 2-fold out-of-fold; its probability and context feed level 2 (GPU) |
| `src/diagnose.py`, `src/errors.py`, `src/selftrain.py` | unseen-country simulation, error attribution, pseudo-label self-training |
| `src/eda.py` | EDA report |
| `src/pipeline.py` | orchestration (`data`, `model`, `full`, individual steps) |

Older single-machine prototype modules (`blocking.py`, `features.py`, `decide.py`, `metrics.py`)
remain for the `train` / `predict` commands of the first version.
