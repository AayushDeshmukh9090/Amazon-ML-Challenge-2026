"""Make a small, *structurally faithful* sample of the real dataset for fast local iteration.

train: keep a fraction of S1 entities, ALL of their true S2/S3 matches, and the same
       fraction of the unmatched S2/S3 records (distractors), so match ratios,
       singleton rate and distractor density stay the same as the full data.
test : same fraction of S1 and of S2/S3 (no labels, used only to smoke-test predict).

  python utils/make_sample.py --src dataset --dst dataset_sample --frac 0.1
"""
import argparse
import csv
import os

import pandas as pd


def read(p):
    return pd.read_csv(p, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False)


def write(df, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:  # verbatim rows, no quoting/escaping
        f.write("\t".join(df.columns) + "\n")
        for row in df.itertuples(index=False):
            f.write("\t".join(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--dst", default="dataset_sample")
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # ---- train
    d = os.path.join(a.src, "train")
    s1, s2, s3 = (read(os.path.join(d, f"train_source{i}.tsv")) for i in (1, 2, 3))
    gt = read(os.path.join(d, "train_ground_truth.tsv"))
    keep1 = s1.sample(frac=a.frac, random_state=a.seed)
    gt_k = gt[gt.source1_entity_id.isin(keep1.entity_id)]
    matched_all = {x for m in gt.matched_entity_ids for x in m.split(",") if x}
    matched_keep = {x for m in gt_k.matched_entity_ids for x in m.split(",") if x}
    out = os.path.join(a.dst, "train")
    for i, df in ((2, s2), (3, s3)):
        distract = df[~df.entity_id.isin(matched_all)].sample(frac=a.frac, random_state=a.seed)
        write(pd.concat([df[df.entity_id.isin(matched_keep)], distract]).sample(frac=1, random_state=a.seed),
              os.path.join(out, f"train_source{i}.tsv"))
    write(keep1, os.path.join(out, "train_source1.tsv"))
    write(gt_k, os.path.join(out, "train_ground_truth.tsv"))
    print(f"train sample: S1={len(keep1):,}  matched ids kept={len(matched_keep):,}")

    # ---- test
    d = os.path.join(a.src, "test")
    if os.path.isdir(d):
        out = os.path.join(a.dst, "test")
        for i in (1, 2, 3):
            df = read(os.path.join(d, f"test_source{i}.tsv"))
            write(df.sample(frac=a.frac, random_state=a.seed), os.path.join(out, f"test_source{i}.tsv"))
        print("test sample written")
    print("->", a.dst)


if __name__ == "__main__":
    main()
