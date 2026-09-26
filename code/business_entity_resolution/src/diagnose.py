"""Diagnostics for the out-of-fold vs leaderboard gap (model part only: cached features + trained models).

1. Unseen-country simulation (proxy for France, absent from train):
   each train country's S1 entities are split in halves A/B (hash).  For target country T:
     in-country    : train on T-half-A                         -> score T-half-B
     cross-country : train on the OTHER country (same #pairs)  -> score T-half-B
   Same data size, same features, same threshold search -> the difference is the penalty a
   country pays when the model has never seen it.
2. Confidence profile per country, OOF (train) vs test (incl. France):
   share of S2/S3 records whose best candidate probability is uncertain (0.2-0.8), mean best
   probability, matches per S1, empty share.  A country far less certain on test than the train
   countries are OOF is where the leaderboard points are lost.
Writes work/diagnose_report.md.

  python src/pipeline.py diagnose --data-dir dataset --work-dir work
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
import pandas as pd

import gbm
from block import true_pairs
from prep import load_prep
from stage2 import (AGG, _folds, agg_features, feature_columns, load_features, macro_f05, one_to_one_best,
                    pair_thresholds)

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _labels(data_dir, work_dir, X):
    s1_ids = load_prep(work_dir, "train", (1,), ["entity_id"])["entity_id"]
    o_ids = load_prep(work_dir, "train", (2, 3), ["entity_id"])["entity_id"]
    ex, _, _ = true_pairs(data_dir, s1_ids, o_ids)
    n_true = np.zeros(len(s1_ids), np.int64)
    np.add.at(n_true, ex["s1"].astype(np.int64).to_numpy(), 1)
    e2 = ex.dropna(subset=["o"])
    tkey = e2["s1"].astype(np.int64).to_numpy() * (1 << 32) + e2["o"].astype(np.int64).to_numpy()
    key = X["s1"].to_numpy().astype(np.int64) * (1 << 32) + X["o"].to_numpy().astype(np.int64)
    return np.isin(key, tkey), n_true, s1_ids


def _best_f(s1, o, p, lab, n_true, ids):
    best = one_to_one_best(s1, o, p)
    res = [(macro_f05(s1, best & (p >= t), lab, n_true, ids)["macro"], float(t))
           for t in np.round(np.arange(0.3, 0.91, 0.02), 2)]
    return max(res)


def profile(s1, o, p, ck_s1, thresholds) -> pd.DataFrame:
    """Per-country confidence profile of a scored pair set."""
    order = np.lexsort((-p, o))
    first = np.ones(len(order), bool)
    first[1:] = o[order][1:] != o[order][:-1]
    top = order[first]                                   # best pair of every S2/S3 record
    best_p, rec_country = p[top], ck_s1[s1[top]]
    keep = one_to_one_best(s1, o, p) & (p >= pair_thresholds(thresholds, ck_s1[s1]))
    nm = np.bincount(s1[keep], minlength=len(ck_s1))
    rows = {}
    for c in np.unique(ck_s1):
        r = rec_country == c
        e = ck_s1 == c
        rows[c] = {"records": int(r.sum()), "mean_best_p": round(float(best_p[r].mean()), 4),
                   "uncertain_0.2-0.8": round(float(((best_p[r] >= 0.2) & (best_p[r] < 0.8)).mean()), 4),
                   "best_p<0.2": round(float((best_p[r] < 0.2).mean()), 4),
                   "matches_per_s1": round(float(nm[e].mean()), 3), "empty_s1": round(float((nm[e] == 0).mean()), 4)}
    return pd.DataFrame(rows).T


def run(data_dir, work_dir, max_rounds=3000, min_s1=50_000):
    b = pickle.load(open(os.path.join(work_dir, "model2.pkl"), "rb"))
    backend = b["l1"][0].backend
    use_emb = b.get("use_emb", False)
    rep = ["# Diagnose report", ""]

    # ------------------------------------------------------------------ train: OOF profile
    X = load_features(work_dir, "train", use_emb)
    y, n_true, s1_ids = _labels(data_dir, work_dir, X)
    s1, o = X["s1"].to_numpy(), X["o"].to_numpy()
    ck = load_prep(work_dir, "train", (1,), ["ckey"])["ckey"].to_numpy()
    k = len(b["l1"])
    fold_all, _ = _folds(s1_ids, k)
    fold = fold_all[s1]
    lvl = "L2" if b["use_l2"] else "L1"
    oof = np.zeros(len(X))
    for f in range(k):
        ck_f = pickle.load(open(os.path.join(work_dir, "ckpt", f"{lvl}_fold{f}.pkl"), "rb"))
        oof[fold == f] = ck_f["oof"]
    thr = b.get("thresholds", {"_global": b["threshold"]})
    rep += [f"## 1. Confidence profile - TRAIN out-of-fold ({lvl})", "", profile(s1, o, oof, ck, thr).to_markdown(), ""]
    log("train OOF profile done")

    # ------------------------------------------------------------------ unseen-country simulation (L1 features)
    feats = b["feats"]
    half = (pd.util.hash_array(np.asarray(["half:" + x for x in s1_ids], dtype=object)) % 2)[s1]
    u = (pd.util.hash_array(np.asarray(["es:" + x for x in s1_ids], dtype=object)) % 1000)[s1] / 1000.0
    countries = [c for c in np.unique(ck) if (ck == c).sum() > min_s1]
    rows = []
    rng = np.random.default_rng(0)
    for tgt in countries:
        in_c = ck[s1] == tgt
        tr_in = in_c & (half == 0)
        ev = in_c & (half == 1)
        ev_ids = np.where((ck == tgt) & ((pd.util.hash_array(
            np.asarray(["half:" + x for x in s1_ids], dtype=object)) % 2) == 1))[0]
        for src in countries:
            if src == tgt:
                tr = tr_in
            else:
                pool = np.where((ck[s1] == src) & (half == 0))[0]
                n = min(len(pool), int(tr_in.sum()))
                tr = np.zeros(len(X), bool)
                tr[rng.choice(pool, size=n, replace=False)] = True
            es = tr & (u < 0.05)
            fit = tr & ~es
            m = gbm.fit(backend, X.loc[fit, feats], y[fit].astype(np.int8), feats, max_rounds, seed=0,
                        Xva=X.loc[es, feats], yva=y[es].astype(np.int8), log=log)
            p = np.zeros(len(X))
            p[ev] = m.predict(X.loc[ev, feats])
            sc, t = _best_f(s1[ev], o[ev], p[ev], y[ev], n_true, ev_ids)
            sc71 = macro_f05(s1[ev], one_to_one_best(s1[ev], o[ev], p[ev]) & (p[ev] >= 0.71), y[ev], n_true,
                             ev_ids)["macro"]
            rows.append({"target": tgt, "trained_on": src, "train_pairs": int(fit.sum()),
                         "macroF05_best_t": round(sc, 5), "best_t": t, "macroF05_t0.71": round(sc71, 5)})
            log(f"unseen-country sim: train {src} -> score {tgt}: {sc:.5f} (t={t}), {sc71:.5f} @0.71")
    sim = pd.DataFrame(rows)
    rep += ["## 2. Unseen-country simulation (L1 features, equal train size, scored on held-out half)", "",
            sim.to_markdown(index=False), "",
            "Penalty = in-country minus cross-country score on the same target: an estimate of what France "
            "(never seen in training) loses.", ""]
    del X

    # ------------------------------------------------------------------ test profile (incl. France)
    Xt = load_features(work_dir, "test", use_emb)
    p = np.mean([m.predict(Xt[b["feats"]]) for m in b["l1"]], axis=0)
    if b["use_l2"]:
        A = agg_features(Xt, p)
        for c in AGG:
            Xt[c] = A[c].to_numpy()
        p = np.mean([m.predict(Xt[b["feats2"]]) for m in b["l2"]], axis=0)
    ckt = load_prep(work_dir, "test", (1,), ["ckey"])["ckey"].to_numpy()
    rep += [f"## 3. Confidence profile - TEST ({lvl}, final models)", "",
            profile(Xt["s1"].to_numpy(), Xt["o"].to_numpy(), p, ckt, thr).to_markdown(), "",
            "Compare with section 1: a country whose records are much more often 'uncertain' (or whose best "
            "probability is lower) on test than India/US are out-of-fold is where the leaderboard gap comes from.",
            ""]
    pd.DataFrame({"s1": Xt["s1"].to_numpy(), "o": Xt["o"].to_numpy(), "p": p.astype(np.float32)}).to_parquet(
        os.path.join(work_dir, "test_scored.parquet"), index=False)
    text = "\n".join(rep) + "\n"
    open(os.path.join(work_dir, "diagnose_report.md"), "w").write(text)
    print(text)
