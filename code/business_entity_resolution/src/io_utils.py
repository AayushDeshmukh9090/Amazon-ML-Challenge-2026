"""Reading / writing the challenge TSVs."""
from __future__ import annotations

import csv
import os

import pandas as pd

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str) -> pd.DataFrame:
    # quoting=NONE: names/addresses can contain stray quotes; keep_default_na=False
    # so the literal string "NA"/"None" in a name is not turned into NaN.
    df = pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                     keep_default_na=False, na_values=[""])
    for c in COLS:
        if c in df.columns:
            df[c] = df[c].fillna("")
    return df


def load_split(data_dir: str, split: str):
    """Return (s1, s2, s3, gt_or_None) for split in {'train','test'}."""
    d = os.path.join(data_dir, split)
    s1 = read_tsv(os.path.join(d, f"{split}_source1.tsv"))
    s2 = read_tsv(os.path.join(d, f"{split}_source2.tsv"))
    s3 = read_tsv(os.path.join(d, f"{split}_source3.tsv"))
    gt = None
    gp = os.path.join(d, f"{split}_ground_truth.tsv")
    if os.path.exists(gp):
        gt = read_tsv(gp)
    return s1, s2, s3, gt


def gt_to_dict(gt: pd.DataFrame) -> dict[str, set[str]]:
    out = {}
    for s1, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"].fillna("")):
        out[s1] = {x.strip() for x in str(m).split(",") if x.strip()}
    return out


def gt_pairs(gt: pd.DataFrame) -> pd.DataFrame:
    rows = [(s1, x) for s1, ms in gt_to_dict(gt).items() for x in ms]
    return pd.DataFrame(rows, columns=["s1_id", "cand_id"])


def write_id_lists(path: str, s1_ids, mapping: dict, col: str):
    """Write one row per S1 id, comma-joined deduplicated list (empty allowed)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s in s1_ids:
            ids = list(dict.fromkeys(mapping.get(s, [])))
            f.write(f"{s}\t{','.join(ids)}\n")
