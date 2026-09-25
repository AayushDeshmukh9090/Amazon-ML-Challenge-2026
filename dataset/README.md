# Put the challenge data here (git-ignored)

Copy the files from `student_resource/dataset/` so the layout is exactly:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Then run `python run.py check-data`.
Also copy `student_resource/utils/validate_submission.py` into `utils/` so that `python run.py validate`
uses the official validator. It is git-ignored because it belongs to the organisers.
