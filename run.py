"""One entry point for every task, locally (VS Code) or remotely.

  python run.py <task> [--data dataset|dataset_sample] [extra pipeline args...]

LOCAL / cheap (seconds - few minutes on a laptop):
  setup        pip install the pinned requirements into the current interpreter
  check-data   verify dataset/ has every expected file + print row counts
  sample       build dataset_sample/ (10% of S1 + their matches + 10% distractors)
  eda          EDA report -> reports/eda/eda_report.md
  blocking     candidate generation + recall report only (no model)
  smoke        synthetic data -> full train+predict in a temp folder (checks the code runs)
  quick        full pipeline on dataset_sample/ (fast iteration on real-looking data)
  validate     check output/ (official validator if present in utils/, else ours)
  zip          build <team>_submission.zip   (python run.py zip --team NAME)

HEAVY (full data - prefer Modal / Colab, see remote/):
  features     build + cache pair features for train and test into work/
  train        5-fold OOF + decision tuning + LOCO + final models  -> work/
  predict      score test -> output/matching_results.tsv + candidate_pairs.tsv
  all          train + predict
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "code", "business_entity_resolution", "src")
REQ = os.path.join(ROOT, "code", "business_entity_resolution", "requirements.txt")
PY = sys.executable
EXPECTED = {
    "train": ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"],
    "test": ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"],
}


def sh(*args):
    print("$", " ".join(str(a) for a in args), flush=True)
    r = subprocess.run([str(a) for a in args], cwd=ROOT)
    if r.returncode:
        sys.exit(r.returncode)


def pipeline(cmd, data, rest, work="work", out="output"):
    sh(PY, os.path.join(SRC, "pipeline.py"), cmd, "--data-dir", data, "--work-dir", work, "--out-dir", out, *rest)


def check_data(data):
    ok = True
    for split, files in EXPECTED.items():
        for f in files:
            p = os.path.join(ROOT, data, split, f)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as fh:
                    n = sum(1 for _ in fh) - 1
                print(f"  OK   {data}/{split}/{f}  ({n:,} rows)")
            else:
                ok = False
                print(f"  MISSING  {data}/{split}/{f}")
    if not ok:
        print(f"\nCopy the challenge files into {data}/train and {data}/test (see dataset/README.md).")
        sys.exit(1)


def make_zip(team):
    for f in ("output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md"):
        if not os.path.exists(os.path.join(ROOT, f)):
            sys.exit(f"missing {f} - run predict first")
    name = os.path.join(ROOT, f"{team}_submission.zip")
    with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(os.path.join(ROOT, "output", f), f"output/{f}")
        base = os.path.join(ROOT, "code", "business_entity_resolution")
        for dp, dn, fn in os.walk(base):
            dn[:] = [d for d in dn if d != "__pycache__"]
            for f in fn:
                full = os.path.join(dp, f)
                z.write(full, os.path.join("code", os.path.relpath(full, os.path.join(ROOT, "code"))))
        z.write(os.path.join(ROOT, "Documentation_template.md"), "Documentation_template.md")
    with zipfile.ZipFile(name) as z:
        for n in z.namelist():
            print("  ", n)
    print("->", name)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    task, rest = sys.argv[1], sys.argv[2:]
    data = "dataset"
    if "--data" in rest:
        i = rest.index("--data")
        data = rest[i + 1]
        del rest[i:i + 2]

    if task == "setup":
        sh(PY, "-m", "pip", "install", "-r", REQ)
    elif task == "check-data":
        check_data(data)
    elif task == "sample":
        check_data("dataset")
        sh(PY, "utils/make_sample.py", "--src", "dataset", "--dst", "dataset_sample", *rest)
    elif task == "eda":
        check_data(data)
        sh(PY, os.path.join(SRC, "eda.py"), "--data-dir", data, "--out-dir", "reports/eda", *rest)
    elif task == "blocking":
        pipeline("blocking", data, rest)
    elif task == "smoke":
        tmp = tempfile.mkdtemp(prefix="er_smoke_")
        sh(PY, "utils/make_synthetic_data.py", "--out", os.path.join(tmp, "data"), "--n-train", "800",
           "--n-test", "500")
        pipeline("all", os.path.join(tmp, "data"), ["--folds", "3", "--skip-loco", *rest],
                 work=os.path.join(tmp, "work"), out=os.path.join(tmp, "out"))
        sh(PY, "utils/check_submission.py", "--out-dir", os.path.join(tmp, "out"),
           "--test-dir", os.path.join(tmp, "data", "test"))
        shutil.rmtree(tmp, ignore_errors=True)
        print("SMOKE TEST PASSED")
    elif task == "quick":
        if not os.path.isdir(os.path.join(ROOT, "dataset_sample")):
            sh(PY, "utils/make_sample.py", "--src", "dataset", "--dst", "dataset_sample")
        pipeline("all", "dataset_sample", ["--folds", "3", *rest], work="work_sample", out="output_sample")
    elif task in ("features", "train", "predict", "all"):
        check_data(data)
        pipeline(task, data, rest)
    elif task == "validate":
        official = os.path.join(ROOT, "utils", "validate_submission.py")
        if os.path.exists(official):
            sh(PY, official, "--matching", "output/matching_results.tsv",
               "--candidate", "output/candidate_pairs.tsv", "--test-dir", f"{data}/test")
        else:
            print("(official utils/validate_submission.py not found - using utils/check_submission.py)")
            sh(PY, "utils/check_submission.py", "--out-dir", "output", "--test-dir", f"{data}/test")
    elif task == "zip":
        team = rest[rest.index("--team") + 1] if "--team" in rest else "team"
        make_zip(team)
    else:
        print(__doc__)
        sys.exit(f"unknown task {task!r}")


if __name__ == "__main__":
    main()
