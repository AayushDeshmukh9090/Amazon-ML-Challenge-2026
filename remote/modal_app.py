"""Run the heavy stages on Modal (https://modal.com), CPU only, with many cores and plenty of RAM.

One-time setup (on your laptop):
    pip install modal
    modal setup                                   # browser login, stores a token

Every run is launched from the repo root:
    modal run remote/modal_app.py --upload        # first time / when dataset changes: push dataset/ to a Modal Volume
    modal run remote/modal_app.py                 # full large-scale pipeline (default task "full"), then download results
    modal run remote/modal_app.py --task model    # any task: eda | data | model | full | synonyms | prep | ...
    modal run remote/modal_app.py --task train --extra "--folds 5 --skip-loco"
    modal run remote/modal_app.py --gpu no        # force CPU only (default: GPU for full/model/train2/predict2)
    modal run --detach remote/modal_app.py        # keeps running if your laptop sleeps / disconnects;
                                                  # afterwards fetch results with:  --task download

Your *current local code* is shipped on every run (src/ is mounted), so edit in VS Code,
then launch here. Data, cached features, models and outputs live in the Modal Volume
"amazon-ml-2026" and are copied back into ./output, ./work, ./reports/eda.

Only compute is rented. No external data, APIs or lookups are used (fair-play rules).
"""
from __future__ import annotations

import os
import shlex
import subprocess

import modal

APP_NAME = "amazon-ml-2026-er"
VOL_NAME = "amazon-ml-2026"
CPU = 32            # cores: prep / blocking / features run in process pools
MEMORY_MB = 131072  # 128 GB: ~12M-record token matrices + ~100M candidate pairs
TIMEOUT_S = 6 * 3600
GPU = "L4"          # 24 GB NVIDIA GPU for XGBoost training/prediction (tasks in GPU_TASKS)
GPU_TASKS = {"full", "model", "train2", "predict2"}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_LOCAL = os.path.join(ROOT, "code", "business_entity_resolution", "src")
REQ_LOCAL = os.path.join(ROOT, "code", "business_entity_resolution", "requirements.txt")

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(REQ_LOCAL)
    .add_local_dir(SRC_LOCAL, "/root/src")        # mounted fresh on every run -> always your latest code
)

# files copied back to the laptop after a run (volume path -> local path)
DOWNLOADS = [
    ("output/matching_results.tsv", "output/matching_results.tsv"),
    ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
    ("work/train_summary.json", "work/train_summary.json"),
    ("work/feature_importance.csv", "work/feature_importance.csv"),
    ("work/model.pkl", "work/model.pkl"),
    ("work/oof_pairs.tsv", "work/oof_pairs.tsv"),
    ("work/test_pairs_scored.tsv", "work/test_pairs_scored.tsv"),
    ("work/blocking_report.md", "work/blocking_report.md"),
    ("work/blocking_grid.json", "work/blocking_grid.json"),
    ("work/synonyms.json", "work/synonyms.json"),
    ("work/stage2_report.md", "work/stage2_report.md"),
    ("logs/last_run.log", "work/modal_last_run.log"),
]


def _run(task: str, extra: str = "") -> int:
    vol.reload()
    os.makedirs("/vol/logs", exist_ok=True)
    if task == "eda":
        cmd = ["python", "/root/src/eda.py", "--data-dir", "/vol/dataset", "--out-dir", "/vol/reports/eda"]
    else:
        cmd = ["python", "-u", "/root/src/pipeline.py", task, "--data-dir", "/vol/dataset",
               "--work-dir", "/vol/work", "--out-dir", "/vol/output"]
    cmd += shlex.split(extra)
    print("$", " ".join(cmd), flush=True)
    with open("/vol/logs/last_run.log", "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:                     # stream to your terminal AND keep a copy in the volume
            print(line, end="", flush=True)
            log.write(line)
        rc = p.wait()
    vol.commit()
    return rc


@app.function(image=image, volumes={"/vol": vol}, cpu=CPU, memory=MEMORY_MB, timeout=TIMEOUT_S)
def run_remote(task: str, extra: str = "") -> int:
    return _run(task, extra)


# Same container size plus a GPU: string stages still use the 32 cores, model training uses the GPU
@app.function(image=image, volumes={"/vol": vol}, cpu=CPU, memory=MEMORY_MB, gpu=GPU, timeout=TIMEOUT_S)
def run_remote_gpu(task: str, extra: str = "") -> int:
    return _run(task, extra)


def _upload_dataset():
    src = os.path.join(ROOT, "dataset")
    for split in ("train", "test"):
        if not os.path.isdir(os.path.join(src, split)):
            raise SystemExit(f"{src}/{split} not found - copy the challenge data first")
    with vol.batch_upload(force=True) as batch:
        for split in ("train", "test"):
            for f in sorted(os.listdir(os.path.join(src, split))):
                if f.endswith(".tsv"):
                    batch.put_file(os.path.join(src, split, f), f"/dataset/{split}/{f}")
                    print("uploaded", f"{split}/{f}")


def _download(task: str):
    items = list(DOWNLOADS)
    if task == "eda":
        items = [(f"reports/eda/{e.path.split('/')[-1]}", f"reports/eda/{e.path.split('/')[-1]}")
                 for e in vol.listdir("reports/eda")]
    for remote, local in items:
        dst = os.path.join(ROOT, local)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                for chunk in vol.read_file(remote):
                    fh.write(chunk)
            print("downloaded", local)
        except Exception as e:  # file not produced by this task (e.g. no model after `features`)
            if os.path.exists(dst) and os.path.getsize(dst) == 0:
                os.remove(dst)
            print(f"skip {remote} ({type(e).__name__})")


@app.local_entrypoint()
def main(task: str = "full", extra: str = "", upload: bool = False, download: bool = True, gpu: str = "auto"):
    if upload:
        _upload_dataset()
        if task == "upload":
            return
    if task != "download":
        use_gpu = gpu == "yes" or (gpu == "auto" and task in GPU_TASKS)
        print(f"running '{task}' on {'CPU + ' + GPU + ' GPU' if use_gpu else 'CPU'}")
        rc = (run_remote_gpu if use_gpu else run_remote).remote(task, extra)
        if rc:
            print(f"remote task failed (exit {rc}); log copied to work/modal_last_run.log")
    if download:
        _download("eda" if task == "eda" else "pipeline")
