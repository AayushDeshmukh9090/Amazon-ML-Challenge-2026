"""Stream every raw TSV once, normalise it in parallel, write <work>/prep/<split>_s<k>.parquet.

Columns written per record:
  entity_id, ckey (country key), name, addr (raw),
  n_name, c_name (core name), n_addr, p_name (phonetic core name),
  hn (house number), nums (space-joined numbers), pc (postal code),
  btok (space-joined blocking tokens, see normalize.block_tokens)

  python src/pipeline.py prep --data-dir dataset --work-dir work [--jobs 32]
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import normalize as N

T0 = time.time()
COLS = ["entity_id", "ckey", "name", "addr", "n_name", "c_name", "n_addr", "p_name", "hn", "nums", "pc", "btok"]


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _init(syn_path):
    N.load_synonyms(syn_path)


def normalise_frame(ch: pd.DataFrame) -> pd.DataFrame:
    names = ch["business_name"].fillna("").astype(str).tolist()
    addrs = ch["business_address"].fillna("").astype(str).tolist()
    c_name = [N.core_name(x) for x in names]
    return pd.DataFrame({
        "entity_id": ch["entity_id"].astype(str).values,
        "ckey": ch["country"].fillna("").astype(str).str.strip().str.lower().values,
        "name": names, "addr": addrs,
        "n_name": [N.norm_name(x) for x in names],
        "c_name": c_name,
        "n_addr": [N.norm_addr(x) for x in addrs],
        "p_name": [N.phonetic_key(x) for x in c_name],
        "hn": [N.house_number(x) for x in addrs],
        "nums": [" ".join(sorted(N.numbers(x))) for x in addrs],
        "pc": [N.postal_code(x) for x in addrs],
        "btok": [" ".join(N.block_tokens(n, a)) for n, a in zip(names, addrs)],
    }, columns=COLS)


def keep_s1(ids, drop_frac: float) -> np.ndarray:
    """Deterministic S1 subsample for TRAINING (salted hash, independent of the CV folds)."""
    if drop_frac <= 0:
        return np.ones(len(ids), bool)
    h = pd.util.hash_array(np.asarray(["drop:" + x for x in ids], dtype=object))
    return (h % 100_000) / 100_000.0 >= drop_frac


def prep_file(src: str, dst: str, pool, chunksize: int, drop_frac: float = 0.0):
    reader = pd.read_csv(src, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False,
                         chunksize=chunksize)
    if drop_frac > 0:
        reader = (ch[keep_s1(ch["entity_id"].tolist(), drop_frac)] for ch in reader)
    writer, n = None, 0
    it = pool.imap(normalise_frame, reader) if pool else map(normalise_frame, reader)
    for out in it:
        tbl = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(dst + ".tmp", tbl.schema, compression="zstd")
        writer.write_table(tbl)
        n += len(out)
    if writer is not None:
        writer.close()
        os.replace(dst + ".tmp", dst)
    log(f"  {os.path.basename(src)} -> {os.path.basename(dst)}: {n:,} rows")


def prep_all(data_dir, work_dir, jobs=None, chunksize=100_000, splits=("train", "test"), force=False,
             drop_s1=0.0):
    """drop_s1: share of TRAIN S1 entities removed (their S2/S3 matches stay as no-match records), so
    that train has the same S2/S3-per-S1 density - i.e. the same share of unmatched distractor
    records - as test (train 4.68 vs test 5.75 records per S1 -> drop ~19%)."""
    syn = os.path.join(work_dir, "synonyms.json")
    if not N.load_synonyms(syn):
        log(f"WARNING: {syn} not found - run `pipeline.py synonyms` first for script/alias handling")
    out_dir = os.path.join(work_dir, "prep")
    os.makedirs(out_dir, exist_ok=True)
    jobs = jobs or os.cpu_count()
    cfg_path = os.path.join(out_dir, "prep_config.json")
    old_cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {"drop_s1": 0.0}
    drop_changed = float(old_cfg.get("drop_s1", 0.0)) != float(drop_s1)
    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    with ctx.Pool(jobs, initializer=_init, initargs=(syn,)) if jobs > 1 else _NoPool() as pool:
        for split in splits:
            for k in (1, 2, 3):
                src = os.path.join(data_dir, split, f"{split}_source{k}.tsv")
                dst = os.path.join(out_dir, f"{split}_s{k}.parquet")
                subsample = split == "train" and k == 1
                stale = subsample and drop_changed
                if os.path.exists(dst) and not force and not stale and os.path.getmtime(dst) > max(
                        os.path.getmtime(src), os.path.getmtime(syn) if os.path.exists(syn) else 0):
                    log(f"  {os.path.basename(dst)} up to date")
                    continue
                prep_file(src, dst, pool if jobs > 1 else None, chunksize, drop_s1 if subsample else 0.0)
    with open(cfg_path, "w") as fh:
        json.dump({"drop_s1": float(drop_s1)}, fh)
    if drop_s1 > 0:
        log(f"train S1 subsampled: {drop_s1:.0%} of entities removed (their matches stay as distractors)")


class _NoPool:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def load_prep(work_dir, split, sources=(1, 2, 3), columns=None) -> pd.DataFrame:
    parts = [pd.read_parquet(os.path.join(work_dir, "prep", f"{split}_s{k}.parquet"), columns=columns)
             for k in sources]
    return pd.concat(parts, ignore_index=True)
