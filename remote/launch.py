"""Launch a pipeline task on Modal fully server-side: survives laptop sleep / network loss.

`modal run --detach` keeps the job attached to your client, and a dropped connection can cancel it
(it did: "Received a cancellation signal").  This script instead DEPLOYS the app and SPAWNS the job
on the deployed app, then exits.  The job runs on Modal until it finishes, whatever your laptop does.

  python remote/launch.py model            # train + predict (GPU)
  python remote/launch.py data             # data preparation (CPU)
  python remote/launch.py full --gpu yes   # anything modal_app.py accepts; --extra "..." for flags
  python remote/launch.py status           # is the last launched job still running / did it finish?
  python remote/launch.py download         # fetch outputs + reports once it finished
Watch live logs:  modal app logs amazon-ml-2026-er
"""
import argparse
import os
import subprocess
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
APP = "amazon-ml-2026-er"
GPU_TASKS = {"full", "model", "train2", "predict2"}
LAST = os.path.join(ROOT, "work", "last_modal_call.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--extra", default="")
    ap.add_argument("--gpu", default="auto", choices=["auto", "yes", "no"])
    a = ap.parse_args()

    if a.task == "download":
        sys.exit(subprocess.run(["modal", "run", os.path.join(HERE, "modal_app.py"), "--task", "download"],
                                cwd=ROOT).returncode)
    if a.task == "status":
        if not os.path.exists(LAST):
            sys.exit("no job launched from this machine yet")
        call_id, task = open(LAST).read().split()
        call = modal.FunctionCall.from_id(call_id)
        try:
            rc = call.get(timeout=0)
            print(f"'{task}' FINISHED with exit code {rc}. Next: python remote/launch.py download")
        except Exception as e:  # noqa: BLE001
            if "Timeout" in type(e).__name__:
                print(f"'{task}' is still RUNNING ({call_id}). Live logs: modal app logs {APP}")
            else:
                print(f"'{task}' FAILED or was cancelled: {type(e).__name__}: {e}")
        return

    # deploy the current local code (src/ is mounted into the image at deploy time)
    subprocess.run(["modal", "deploy", os.path.join(HERE, "modal_app.py")], cwd=ROOT, check=True)
    use_gpu = a.gpu == "yes" or (a.gpu == "auto" and a.task in GPU_TASKS)
    fn = modal.Function.from_name(APP, "run_remote_gpu" if use_gpu else "run_remote")
    call = fn.spawn(a.task, a.extra)
    os.makedirs(os.path.dirname(LAST), exist_ok=True)
    with open(LAST, "w") as fh:
        fh.write(f"{call.object_id} {a.task}")
    print(f"\nLaunched '{a.task}' on {'GPU' if use_gpu else 'CPU'} server-side (call {call.object_id}).")
    print("You can close this terminal / let the laptop sleep. Check with:")
    print("  python remote/launch.py status")
    print(f"  modal app logs {APP}          (live logs)")
    print("  python remote/launch.py download   (when finished)")


if __name__ == "__main__":
    main()
