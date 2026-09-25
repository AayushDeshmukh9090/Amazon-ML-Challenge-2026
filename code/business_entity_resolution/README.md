# Business Entity Resolution: Amazon ML Challenge 2026

Pipeline: **data → normalisation → multi-blocker candidate generation → pairwise features →
LightGBM (5-fold OOF, 3-seed final ensemble) → F0.5-optimal decision layer → `output/`**.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.11
pip install -r code/business_entity_resolution/requirements.txt
```

Put the challenge data in `dataset/` (same layout as `student_resource/`):

```
dataset/train/train_source{1,2,3}.tsv  dataset/train/train_ground_truth.tsv
dataset/test/test_source{1,2,3}.tsv
```

## Run (from the repo / submission root)

```bash
SRC=code/business_entity_resolution/src

# 1. EDA -> reports/eda/eda_report.md (+ plots, example/hard-negative TSVs)
python3 $SRC/eda.py --data-dir dataset --out-dir reports/eda

# 2. Train: blocking report, 5-fold OOF, decision tuning, leave-one-country-out, final models -> work/
python3 $SRC/pipeline.py train --data-dir dataset --work-dir work

# 3. Predict test -> output/matching_results.tsv + output/candidate_pairs.tsv
python3 $SRC/pipeline.py predict --data-dir dataset --work-dir work --out-dir output

# 4. Validate with the official checker (from student_resource/)
python3 utils/validate_submission.py --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

`pipeline.py all` runs train + predict in one go. Other sub-commands: `blocking` (candidate recall
report only) and `features` (build + cache pair features in `work/cache_*.pkl`; `train`/`predict`
reuse the cache while the data files and blocking config are unchanged, and `--no-cache` forces a rebuild). No external data, APIs or
geocoders are used: everything is learned from the provided training files.

### Smoke test without the real data

```bash
python3 utils/make_synthetic_data.py --out dataset_synth
python3 $SRC/eda.py --data-dir dataset_synth --out-dir /tmp/eda
python3 $SRC/pipeline.py all --data-dir dataset_synth --work-dir /tmp/work --out-dir /tmp/out
```

## Source layout

| file | role |
|---|---|
| `src/normalize.py` | ASCII transliteration, `&`→and, legal-suffix + address abbreviation canonicalisation (US/IN/FR), core name, phonetic key, postal code, DBA split |
| `src/eda.py` | EDA report that drives the design decisions (auto "decision hints" section) |
| `src/blocking.py` | 6 blockers (char/word TF-IDF top-k, reverse top-k, phonetic key, postal+token key) within country blocks; union = candidate set |
| `src/features.py` | ~90 pair features: fuzzy, TF-IDF cosines, IDF-weighted token overlap / rare-token disagreement, postal/number agreement, name frequency, rank/gap context features |
| `src/decide.py` | threshold vs expected-F0.5 decision rules, optional 1-to-1 filter, tuned on OOF |
| `src/metrics.py` | exact macro F0.5 (singletons included), blocking recall / reduction ratio |
| `src/pipeline.py` | train / predict orchestration |

## Outputs in `work/`
`train_summary.json` (OOF score, per-bucket breakdown, blocking ceiling, LOCO scores, chosen decision rule),
`feature_importance.csv`, `oof_pairs.tsv`, `model.pkl`, `test_pairs_scored.tsv`.
