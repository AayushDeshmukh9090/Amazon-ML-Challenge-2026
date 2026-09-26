"""Scalable candidate generation (blocking) for millions of records.

Per country block (country = open set of labels; matched pairs share country 100% in train):
  1. every record -> sparse vector of hashed blocking tokens (name skeletons, no-space core
     name, address skeletons, normalised numbers; see normalize.block_tokens)
  2. tokens with document frequency > --df-cap are dropped (stop words like 'road', 'delhi',
     'llc' carry no identity and would make the sparse products explode), remaining tokens are
     IDF-weighted and rows L2-normalised -> cosine = IDF-weighted token overlap
  3. REVERSE search: each S2/S3 record -> its top-k_rev S1 records   (GT is 1-to-1 from this side,
     so the true S1 only has to be in this short list)
     FORWARD search: each S1 record -> its top-k_fwd S2/S3 records
  4. union -> <work>/cands/<split>.parquet with columns
        s1 (row index into prep S1), o (row index into prep S2+S3), score, r_rev, r_fwd (255 = absent)

Everything is chunked sparse-matrix products run in a process pool (fork), so it scales with
cores; memory ~ nnz of the token matrices.  On train it writes a recall report for every
(k_rev, k_fwd) combination to <work>/blocking_report.md.

  python src/pipeline.py block --data-dir dataset --work-dir work --jobs 32
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2norm

from prep import load_prep

T0 = time.time()
NOPE = 255


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _split(s):
    return s.split()


_HV = HashingVectorizer(analyzer=_split, n_features=2 ** 24, binary=True, norm=None, alternate_sign=False,
                        dtype=np.float32)


def _hash_chunk(texts):
    return _HV.transform(texts)


def hash_tokens(texts, pool, chunk=200_000):
    parts = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    mats = pool.map(_hash_chunk, parts) if pool else [_hash_chunk(p) for p in parts]
    return sp.vstack(mats, format="csr") if mats else sp.csr_matrix((0, _HV.n_features), dtype=np.float32)


# ------------------------------------------------------------------ top-k worker (globals via fork)
_G = {}


def _topk_chunk(args):
    lo, hi, k = args
    Q, IT = _G["Q"], _G["IT"]
    S = (Q[lo:hi] @ IT).tocsr()
    S.sum_duplicates()
    n = hi - lo
    counts = np.diff(S.indptr)
    rows = np.repeat(np.arange(n, dtype=np.int64), counts)
    order = np.lexsort((-S.data, rows))
    srow = rows[order]
    rank = np.arange(len(order), dtype=np.int64) - S.indptr[srow]
    keep = rank < k
    return ((srow[keep] + lo).astype(np.int32), S.indices[order][keep].astype(np.int32),
            S.data[order][keep].astype(np.float32), (rank[keep] + 1).astype(np.uint8))


def _chunks_by_cost(Q, I, max_cost=15_000_000, max_rows=20_000):
    """Split query rows so each chunk's sparse product has <= ~max_cost entries (bounded memory
    per worker even when common tokens have long posting lists)."""
    df = np.asarray((I > 0).sum(0)).ravel().astype(np.int64)
    cost = (Q > 0).astype(np.int64) @ df
    bounds, lo, acc = [], 0, 0
    for r, c in enumerate(cost):
        if r > lo and (acc + c > max_cost or r - lo >= max_rows):
            bounds.append((lo, r))
            lo, acc = r, 0
        acc += c
    if lo < Q.shape[0]:
        bounds.append((lo, Q.shape[0]))
    return bounds, int(cost.sum())


def topk(Q, I, k, jobs):
    """For each row of Q, top-k rows of I by dot product (both L2-normalised) -> (q, i, score, rank)."""
    _G["Q"], _G["IT"] = Q, I.T.tocsr()
    bounds, total = _chunks_by_cost(Q, I)
    log(f"    {len(bounds)} chunks, {total / 1e9:.2f}G posting entries")
    tasks = [(lo, hi, k) for lo, hi in bounds]
    if jobs > 1 and len(tasks) > 1 and hasattr(os, "fork"):   # Windows: no fork -> single process
        ctx = mp.get_context("fork")
        with ctx.Pool(jobs) as pool:
            res = pool.map(_topk_chunk, tasks, chunksize=1)
    else:
        res = [_topk_chunk(t) for t in tasks]
    _G.clear()
    if not res:
        z = np.zeros(0, np.int32)
        return z, z, np.zeros(0, np.float32), np.zeros(0, np.uint8)
    return tuple(np.concatenate([r[j] for r in res]) for j in range(4))


def weight(A, B, df_cap):
    """Drop too-common tokens, IDF-weight, L2-normalise (fitted on A and B jointly: no labels)."""
    df = np.asarray((A > 0).sum(0)).ravel() + np.asarray((B > 0).sum(0)).ravel()
    n = A.shape[0] + B.shape[0]
    keep = (df >= 2) & (df <= df_cap)
    w = np.where(keep, np.log(n / np.maximum(df, 1)), 0).astype(np.float32)
    W = sp.diags(w)
    A, B = l2norm(A @ W), l2norm(B @ W)
    A.eliminate_zeros()
    B.eliminate_zeros()
    return A.tocsr(), B.tocsr(), int(keep.sum())


def block_split(work_dir, split, k_rev=10, k_fwd=20, df_cap=3000, jobs=None, df_frac=0.004):
    jobs = jobs or os.cpu_count()
    s1 = load_prep(work_dir, split, (1,), ["entity_id", "ckey", "btok"])
    oth = load_prep(work_dir, split, (2, 3), ["entity_id", "ckey", "btok"])
    log(f"{split}: S1 {len(s1):,}  S2+S3 {len(oth):,}")
    if not hasattr(os, "fork"):
        jobs = 1
    ctx = mp.get_context("fork") if jobs > 1 else None
    out = []
    with ctx.Pool(jobs) if jobs > 1 else _Null() as pool:
        A_all = hash_tokens(s1["btok"].tolist(), pool)
        B_all = hash_tokens(oth["btok"].tolist(), pool)
    log(f"hashed tokens: S1 nnz {A_all.nnz:,}  S2+S3 nnz {B_all.nnz:,}")
    g1, g2 = s1.groupby("ckey").indices, oth.groupby("ckey").indices
    for c, ia in g1.items():
        ib = g2.get(c)
        if ib is None or not len(ib):
            continue
        cap = max(df_cap, int(df_frac * (len(ia) + len(ib))))   # relative to block size
        A, B, nkeep = weight(A_all[ia], B_all[ib], cap)
        empty_b = float((np.diff(B.indptr) == 0).mean())
        log(f"[{c}] S1 {len(ia):,} x other {len(ib):,}; df cap {cap:,}; kept tokens {nkeep:,}; "
            f"records with no usable token: S1 {(np.diff(A.indptr) == 0).mean():.2%}, other {empty_b:.2%}")
        q, i, sc, rk = topk(B, A, k_rev, jobs)            # reverse: other -> S1
        rev = pd.DataFrame({"s1": ia[i], "o": ib[q], "score": sc, "r_rev": rk, "r_fwd": np.uint8(NOPE)})
        log(f"[{c}] reverse top-{k_rev}: {len(rev):,} pairs")
        q, i, sc, rk = topk(A, B, k_fwd, jobs)            # forward: S1 -> other
        fwd = pd.DataFrame({"s1": ia[q], "o": ib[i], "score": sc, "r_rev": np.uint8(NOPE), "r_fwd": rk})
        log(f"[{c}] forward top-{k_fwd}: {len(fwd):,} pairs")
        both = pd.concat([rev, fwd], ignore_index=True)
        both = both.groupby(["s1", "o"], sort=False, as_index=False).agg(
            score=("score", "max"), r_rev=("r_rev", "min"), r_fwd=("r_fwd", "min"))
        out.append(both)
        log(f"[{c}] union: {len(both):,} pairs")
        del A, B, rev, fwd
    cands = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["s1", "o", "score", "r_rev", "r_fwd"])
    for c, t in (("s1", np.int32), ("o", np.int32), ("score", np.float32), ("r_rev", np.uint8),
                 ("r_fwd", np.uint8)):
        cands[c] = cands[c].astype(t)
    os.makedirs(os.path.join(work_dir, "cands"), exist_ok=True)
    path = os.path.join(work_dir, "cands", f"{split}.parquet")
    cands.to_parquet(path, index=False)
    log(f"saved {len(cands):,} candidate pairs -> {path}")
    return s1, oth, cands


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ------------------------------------------------------------------ recall report (train only)
def true_pairs(data_dir, s1_ids: pd.Series, o_ids: pd.Series) -> pd.DataFrame:
    import csv
    gt = pd.read_csv(os.path.join(data_dir, "train", "train_ground_truth.tsv"), sep="\t", dtype=str,
                     keep_default_na=False, quoting=csv.QUOTE_NONE)
    lists = gt["matched_entity_ids"].str.split(",")
    sizes = lists.map(lambda x: sum(1 for y in x if y)).values
    ex = pd.DataFrame({"s1_id": gt["source1_entity_id"].repeat(sizes).values,
                       "o_id": [y for x in lists for y in x if y]})
    s1_pos = pd.Series(np.arange(len(s1_ids), dtype=np.int32), index=s1_ids.values)
    o_pos = pd.Series(np.arange(len(o_ids), dtype=np.int32), index=o_ids.values)
    ex["s1"] = s1_pos.reindex(ex.s1_id).values
    ex["o"] = o_pos.reindex(ex.o_id).values
    # S1 entities not in prep (dropped for training, see prep.drop_s1) do not exist for this run:
    # their pairs leave the truth, their S2/S3 records simply become unmatched records
    present = gt["source1_entity_id"].isin(s1_pos.index).to_numpy()
    ex = ex[ex["s1"].notna()].reset_index(drop=True)
    return ex, gt[present].reset_index(drop=True), sizes[present]


def recall_report(data_dir, work_dir, s1, oth, cands, k_rev, k_fwd):
    ex, gt, sizes = true_pairs(data_dir, s1["entity_id"], oth["entity_id"])
    ex = ex.merge(cands, on=["s1", "o"], how="left")
    ex["r_rev"] = ex["r_rev"].fillna(NOPE).astype(int)
    ex["r_fwd"] = ex["r_fwd"].fillna(NOPE).astype(int)
    n_true = len(ex)
    lines = ["# Blocking recall report (train)", "",
             f"S1 {len(s1):,}, S2+S3 {len(oth):,}, true pairs {n_true:,}, candidate pairs {len(cands):,} "
             f"({len(cands) / len(s1):.1f} per S1, {len(cands) / len(oth):.2f} per S2/S3 record)", "",
             "Pair recall / #pairs kept if we keep a pair when r_rev <= R OR r_fwd <= F:", "",
             "| R \\ F | " + " | ".join(str(f) for f in (0, 1, 3, 5, 10, k_fwd)) + " |",
             "|---|" + "---|" * 6]
    grid = {}
    for R in (1, 2, 3, 5, k_rev):
        cells = []
        for F in (0, 1, 3, 5, 10, k_fwd):
            hit = ((ex.r_rev <= R) | (ex.r_fwd <= F)).mean()
            npairs = int(((cands.r_rev <= R) | (cands.r_fwd <= F)).sum())
            grid[f"R{R}_F{F}"] = (float(hit), npairs)
            cells.append(f"{hit:.4f} / {npairs / 1e6:.1f}M")
        lines.append(f"| {R} | " + " | ".join(cells) + " |")
    # entity-level ceiling of macro F0.5 with a perfect matcher on the full candidate set
    ex["hit"] = (ex.r_rev <= k_rev) | (ex.r_fwd <= k_fwd)
    r = ex.groupby("s1_id")["hit"].mean()
    r = r.reindex(gt["source1_entity_id"]).values
    f = np.where(sizes == 0, 1.0, np.where(r > 0, 1.25 * r / (0.25 + np.nan_to_num(r)), 0.0))
    lines += ["", f"Macro-F0.5 ceiling (perfect matcher on ALL candidates): {np.nanmean(f):.4f}",
              f"S1 entities with every match covered: {np.mean(r[sizes > 0] == 1):.4f}"]
    miss = ex[~ex.hit].sample(min(40, int((~ex.hit).sum())), random_state=0) if (~ex.hit).any() else ex.head(0)
    if len(miss):
        pr1 = load_prep(work_dir, "train", (1,), ["entity_id", "name", "addr"]).set_index("entity_id")
        pro = load_prep(work_dir, "train", (2, 3), ["entity_id", "name", "addr"]).set_index("entity_id")
        lines += ["", "## Sample of MISSED true pairs", "", "| S1 name | S1 addr | match name | match addr |",
                  "|---|---|---|---|"]
        for a, b in zip(miss.s1_id, miss.o_id):
            if a in pr1.index and b in pro.index:
                lines.append(f"| {pr1.at[a, 'name']} | {pr1.at[a, 'addr']} | {pro.at[b, 'name']} | "
                             f"{pro.at[b, 'addr']} |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(work_dir, "blocking_report.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(work_dir, "blocking_grid.json"), "w") as fh:
        json.dump(grid, fh, indent=1)
    print(text)
