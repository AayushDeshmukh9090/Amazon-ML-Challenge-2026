"""Stage 1.5: cheap learned pre-filter over ALL blocking candidates (~110-120M pairs per split).

Blocking keeps up to 10 S1 per S2/S3 record (+20 per S1) for recall; scoring all of that with the
full ~120-feature model is too expensive.  A small LightGBM on cheap features (blocking score /
ranks, group maxima, 4 rapidfuzz scores, house-number equality, gaps to the best competitor)
ranks every pair; we keep the best few per record and per S1 entity, sized to a pair budget, and
report how much recall survives.  The kept set is what the final model scores (= candidate_pairs).

  python src/pipeline.py prefilter --data-dir dataset --work-dir work
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from block import true_pairs
from prep import load_prep

T0 = time.time()
_G: dict = {}
CHEAP = ["score", "r_rev", "r_fwd", "o_max", "o_ratio", "o_n", "s1_max", "s1_gap", "s1_n",
         "core_tset", "core_ratio", "addr_tset", "addr_ratio", "hn_eq", "core_tset_gap_o", "addr_tset_gap_o",
         "core_tset_gap_s1", "addr_tset_gap_s1"]
LGB = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=200, feature_fraction=0.9,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=0)


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _str_chunk(args):
    i, j = args
    P1, PO = _G["P1"], _G["PO"]
    ca, cb = P1["c_name"].take(i).tolist(), PO["c_name"].take(j).tolist()
    aa, ab = P1["n_addr"].take(i).tolist(), PO["n_addr"].take(j).tolist()
    ha, hb = P1["hn"].take(i).to_numpy(), PO["hn"].take(j).to_numpy()
    cp = lambda a, b, sc: process.cpdist(a, b, scorer=sc, workers=1, dtype=np.float32)  # noqa: E731
    return np.stack([cp(ca, cb, fuzz.token_set_ratio), cp(ca, cb, fuzz.ratio),
                     cp(aa, ab, fuzz.token_set_ratio), cp(aa, ab, fuzz.ratio),
                     np.where((ha != "") & (hb != ""), (ha == hb).astype(np.float32), np.float32(-1))], 1)


def cheap_features(work_dir, split, cands: pd.DataFrame, jobs: int, chunk=400_000) -> pd.DataFrame:
    X = cands.copy()
    for c in ("r_rev", "r_fwd"):
        X[c] = X[c].astype(np.float32)
    go, gs = X.groupby("o")["score"], X.groupby("s1")["score"]
    X["o_max"] = go.transform("max").astype(np.float32)
    X["o_ratio"] = (X["score"] / X["o_max"].clip(lower=1e-6)).astype(np.float32)
    X["o_n"] = go.transform("size").astype(np.float32)
    X["s1_max"] = gs.transform("max").astype(np.float32)
    X["s1_gap"] = (X["s1_max"] - X["score"]).astype(np.float32)
    X["s1_n"] = gs.transform("size").astype(np.float32)
    _G["P1"] = load_prep(work_dir, split, (1,), ["c_name", "n_addr", "hn"])
    _G["PO"] = load_prep(work_dir, split, (2, 3), ["c_name", "n_addr", "hn"])
    s1, o = X["s1"].to_numpy(), X["o"].to_numpy()
    tasks = [(s1[k:k + chunk], o[k:k + chunk]) for k in range(0, len(X), chunk)]
    log(f"{split}: string scores for {len(X):,} pairs in {len(tasks)} chunks")
    if jobs > 1 and hasattr(os, "fork"):
        with mp.get_context("fork").Pool(jobs) as pool:
            parts = pool.map(_str_chunk, tasks, chunksize=1)
    else:
        parts = [_str_chunk(t) for t in tasks]
    _G.clear()
    M = np.concatenate(parts)
    for n, c in enumerate(["core_tset", "core_ratio", "addr_tset", "addr_ratio", "hn_eq"]):
        X[c] = M[:, n]
    del M, parts
    for c in ("core_tset", "addr_tset"):
        X[f"{c}_gap_o"] = (X.groupby("o")[c].transform("max") - X[c]).astype(np.float32)
        X[f"{c}_gap_s1"] = (X.groupby("s1")[c].transform("max") - X[c]).astype(np.float32)
    return X


def _ranks(X, p):
    ro = pd.Series(p).groupby(X["o"].to_numpy()).rank(ascending=False, method="first").to_numpy()
    rs = pd.Series(p).groupby(X["s1"].to_numpy()).rank(ascending=False, method="first").to_numpy()
    return ro, rs


def _keep_mask(p, ranks, tau, k_o, k_s1, k_min=2):
    """Always keep each record's k_min best S1 (the final model needs its runner-up to judge
    near-duplicates); beyond that keep pairs above tau within the per-record / per-S1 top-k."""
    ro, rs = ranks
    return (ro <= k_min) | ((p >= tau) & ((ro <= k_o) | (rs <= k_s1)))


def run(data_dir, work_dir, jobs=None, budget=4.0, k_o=5, k_s1=15, sample=6_000_000):
    jobs = jobs or os.cpu_count()
    path_model = os.path.join(work_dir, "prefilter.pkl")
    for split in ("train", "test"):
        cands = pd.read_parquet(os.path.join(work_dir, "cands", f"{split}.parquet"))
        n_o = int(cands["o"].max()) + 1 if len(cands) else 0
        X = cheap_features(work_dir, split, cands, jobs)
        del cands
        if split == "train":
            s1_ids = load_prep(work_dir, "train", (1,), ["entity_id"])["entity_id"]
            o_ids = load_prep(work_dir, "train", (2, 3), ["entity_id"])["entity_id"]
            ex, _, _ = true_pairs(data_dir, s1_ids, o_ids)
            e2 = ex.dropna(subset=["s1", "o"])
            tkey = e2["s1"].astype(np.int64).to_numpy() * (1 << 32) + e2["o"].astype(np.int64).to_numpy()
            key = X["s1"].to_numpy().astype(np.int64) * (1 << 32) + X["o"].to_numpy().astype(np.int64)
            y = np.isin(key, tkey).astype(np.int8)
            n_true = len(ex)
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
            va = rng.choice(len(X), size=min(1_000_000, len(X)), replace=False)
            m = lgb.train(LGB, lgb.Dataset(X[CHEAP].iloc[idx], y[idx]), 400,
                          valid_sets=[lgb.Dataset(X[CHEAP].iloc[va], y[va])],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
            p = m.predict(X[CHEAP], num_threads=jobs)
            # tau: smallest threshold whose kept set fits the pair budget
            target = budget * n_o
            ranks = _ranks(X, p)
            best = None
            for tau in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
                keep = _keep_mask(p, ranks, tau, k_o, k_s1)
                rec = y[keep].sum() / n_true
                log(f"  tau {tau:<7} keep {keep.sum():>12,} pairs ({keep.sum() / max(n_o, 1):.2f}/record)  "
                    f"pair recall {rec:.4f}")
                if best is None or keep.sum() <= target:
                    best = (tau, keep, rec)
                if keep.sum() <= target:
                    break
            tau, keep, rec = best
            log(f"prefilter: tau={tau}, kept {keep.sum():,} of {len(X):,} pairs, pair recall "
                f"{rec:.4f} (all candidates: {y.sum() / n_true:.4f})")
            with open(path_model, "wb") as fh:
                pickle.dump({"model": m, "tau": tau, "k_o": k_o, "k_s1": k_s1}, fh)
        else:
            b = pickle.load(open(path_model, "rb"))
            p = b["model"].predict(X[CHEAP], num_threads=jobs)
            keep = _keep_mask(p, _ranks(X, p), b["tau"], b["k_o"], b["k_s1"])
            log(f"prefilter test: kept {keep.sum():,} of {len(X):,} pairs")
        out = X.loc[keep, ["s1", "o", "score", "r_rev", "r_fwd"]].copy()
        out["p1"] = p[keep].astype(np.float32)
        out.to_parquet(os.path.join(work_dir, "cands", f"{split}_pruned.parquet"), index=False)
        del X
