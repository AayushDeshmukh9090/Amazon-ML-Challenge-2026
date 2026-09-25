# Amazon ML Challenge 2026: Business Entity Resolution

Match Source-1 business records to their Source-2 and Source-3 counterparts (US, India, and France in test only).
The metric is macro F0.5 per S1 entity, with singletons included.

Everything runnable lives in [`code/business_entity_resolution/`](code/business_entity_resolution/README.md).

## Quick start
```bash
pip install -r code/business_entity_resolution/requirements.txt
# put the challenge data in ./dataset/{train,test}/  (git-ignored)
SRC=code/business_entity_resolution/src
python3 $SRC/eda.py --data-dir dataset --out-dir reports/eda      # 1. EDA report
python3 $SRC/pipeline.py train   --data-dir dataset --work-dir work  # 2. CV + tune + fit
python3 $SRC/pipeline.py predict --data-dir dataset --work-dir work --out-dir output
python3 utils/check_submission.py --out-dir output --test-dir dataset/test
bash utils/make_submission_zip.sh <team_name>
```

| path | what |
|---|---|
| `code/business_entity_resolution/src/` | pipeline source (normalise → block → features → LightGBM → F0.5 decision) |
| `reports/eda/` | EDA report output (`eda_report.md`, plots, example TSVs) |
| `utils/make_synthetic_data.py` | synthetic look-alike data for smoke tests only |
| `utils/check_submission.py` | local format check (the official validator is authoritative) |
| `utils/make_submission_zip.sh` | builds the final zip in the required structure |
| `Documentation_template.md` | methodology write-up (fill the TODOs after the final run) |
