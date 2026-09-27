# Amazon ML Challenge 2026: Business Entity Resolution

Match each Source-1 business record to its Source-2 / Source-3 counterparts. Train covers the US and
India; test adds France. The metric is macro F0.5 per S1 entity, singletons included.

**Best result:** leaderboard **0.98046**, train out-of-fold **0.98667** (India 0.9835, US 0.9888).
Pipeline code and module reference: [`code/business_entity_resolution/`](code/business_entity_resolution/README.md).
Method write-up: [`Documentation_template.md`](Documentation_template.md).

## The final pipeline

```
normalise ─► token index search ─┐
             embedding neighbours ┴─► learned pre-filter ─► 136 pair features
                                                               │
      fine-tuned multilingual cross-encoder (5 features) ──────┤
                                                               ▼
         level 1 XGBoost (3-fold OOF by S1) ─► level 2 XGBoost (+ L1 aggregates + cross-encoder)
                                                               ▼
                  one-to-one per S2/S3 record ─► threshold tuned for macro F0.5 ─► output/
```

| stage | module | train result |
|---|---|---|
| normalisation + learned synonyms | `normalize.py`, `synonyms.py`, `prep.py` | — |
| token candidate search (reverse top-10 + forward top-20, per country) | `block.py` | pair recall 0.9809 |
| multilingual MiniLM name-embedding neighbours (GPU) | `embblock.py` | union recall 0.9842 |
| learned pre-filter (~2.5 pairs per record) | `prefilter.py` | 25.4M pairs, recall 0.9837 |
| pair features + embedding similarity features | `features2.py`, `embed.py` | 136 features |
| level-1 XGBoost | `stage2.py` | OOF 0.98314 |
| cross-encoder: MiniLM fine-tuned on raw name/address pairs, out-of-fold (GPU) | `crossenc.py` | held-out AUC 0.9987 |
| level-2 XGBoost with cross-encoder features | `stage2.py` | **OOF 0.98667** |
| decision rules compared on OOF (threshold / two-threshold / expected-F0.5) | `decide2.py` | threshold 0.72 kept |

Leaderboard history: 0.970 → 0.971 (distractor density matched) → 0.972 (embedding
neighbours, typo features) → **0.98046** (cross-encoder at level 2).

## Folder layout

```
Amazon-ML-Challenge-2026/
├── dataset/                 ← challenge data (git-ignored, see dataset/README.md)
├── code/business_entity_resolution/   ← submission code package (src/, README, requirements)
├── remote/
│   ├── launch.py            ← launch Modal jobs server-side (survives laptop sleep / network loss)
│   ├── modal_app.py         ← Modal app: 32 CPU / 128 GB + GPU, volume "amazon-ml-2026"
│   └── colab_runner.ipynb   ← older Colab alternative
├── utils/                   format checker, sample / synthetic data, per-country leaderboard probe
├── work/                    caches, models, reports (git-ignored)
├── output/                  matching_results.tsv, candidate_pairs.tsv, run_info.json (git-ignored)
├── run.py                   ← local entry point (setup, check-data, smoke, validate, zip, ...)
└── Documentation_template.md
```

## Local setup (once)

```bash
git clone -b claude/entity-resolution-ml-c287bq https://github.com/AayushDeshmukh9090/Amazon-ML-Challenge-2026.git
cd Amazon-ML-Challenge-2026
python -m venv .venv          # activate: source .venv/bin/activate  |  Windows: .venv\Scripts\activate
python run.py setup
pip install modal && modal setup
```
Copy the data into `dataset/`, and `student_resource/utils/validate_submission.py` into `utils/`.
Then run `python run.py check-data`.

## Running on Modal (full data)

```bash
modal run remote/modal_app.py --upload --task upload      # once: dataset/ -> Modal volume
python remote/launch.py full --gpu-type H100              # data part (cached) + model part
python remote/launch.py status                            # running / finished?
modal app logs amazon-ml-2026-er                          # live log
python remote/launch.py download                          # outputs + reports -> output/, work/
python run.py validate
```

| task | what it runs | typical time (H100) |
|---|---|---|
| `data` | synonyms → prep → block → embblock → prefilter → features2 → embed (each step skipped when up to date) | hours the first time, seconds when cached |
| `crossenc` | cross-encoder fine-tuning (2 folds) + OOF / test scoring | ~1.6 h |
| `model` | level 1 (resumes from checkpoints) → level 2 → decision tuning → predict | ~45 min (level 1 retrain adds ~1.4 h) |
| `stack` | `crossenc`, then `model` in a fresh process | ~2.5 h |
| `decide` | re-tune the decision rule on OOF + predict (no retraining) | ~30 min |
| `errors`, `diagnose` | loss attribution by error type; unseen-country simulation | ~15–40 min |

Useful flags go through `--extra`:
- `--rebuild STEP` forces one step and refreshes everything after it.
- `--ce-continue N` continues training the saved cross-encoder models on N new pairs per fold. Use it with `--rebuild crossenc`.
- `--min-gain 0` always takes the best out-of-fold decision rule.

Example:
```bash
python remote/launch.py stack --gpu-type H100 --extra "--rebuild crossenc --ce-continue 3000000"
```

## Local tasks

| task | command |
|---|---|
| code still runs? (synthetic data) | `python run.py smoke` |
| EDA report | `python run.py eda` |
| validate `output/` | `python run.py validate` |
| per-country leaderboard probe | `python utils/probe_country.py --country france` |
| final zip | `python run.py zip --team NAME` |

The zip contains `output/` (both TSVs), `code/business_entity_resolution/` (src, README,
requirements) and `Documentation_template.md`. `output/run_info.json` records which run produced
the files you are about to submit.

## Rules compliance

- No external data, APIs or geocoding: every table is built from the provided training files.
- The pretrained model is `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, ~118M parameters). It is used for name embeddings and fine-tuned as the cross-encoder. The other libraries: XGBoost (Apache-2.0), LightGBM (MIT), scikit-learn (BSD-3).
