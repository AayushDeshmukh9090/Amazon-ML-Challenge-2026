"""Per-S1 decision rules on top of the final model probabilities (no retraining).

The metric is F0.5 PER S1 ENTITY, averaged.  A single global probability threshold treats every pair
alike, but the right choice depends on the S1's whole candidate list:
  - an S1 with no pair above the threshold scores 0 unless it is a true singleton (only ~5.6% are),
    so its best candidate is often worth taking at a LOWER probability ('one' group: 0.941 OOF);
  - an S1 with several strong candidates can afford an extra uncertain one only if it barely
    lowers expected precision.
Rules compared on the train out-of-fold probabilities (exact macro F0.5), best one used on test:
  threshold  : 1-to-1 best record + p >= t                                   (current rule)
  two_thr    : 1-to-1 best record + (p >= t_rest  or  S1's top pick with p >= t_first)
  expected_f : per S1, sort its records by p, take the prefix k maximising
               1.25*sum(p_1..p_k) / (0.25*E|T| + k),  E|T| = beta * sum(p);
               empty if prod(1 - p) + bias beats it.  p is shifted in logit space by delta.

  python src/pipeline.py decide --data-dir dataset --work-dir work --out-dir output
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import pandas as pd

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


class _Groups:
    """Best-per-record pairs grouped by S1, sorted by p descending (computed once, reused per rule)."""

    def __init__(self, s1, o, p):
        from stage2 import one_to_one_best
        self.n = len(p)
        self.best = np.where(one_to_one_best(s1, o, p))[0]
        bs, bp = s1[self.best], p[self.best]
        order = np.lexsort((-bp, bs))
        self.rows = self.best[order]                 # pair indices, grouped by S1, p descending
        self.s = bs[order]
        self.p = bp[order]
        first = np.ones(len(self.s), bool)
        first[1:] = self.s[1:] != self.s[:-1]
        self.starts = np.where(first)[0]
        self.grp = np.cumsum(first) - 1
        self.k = np.arange(len(self.s)) - self.starts[self.grp] + 1       # rank within the S1

    def mask(self, keep_sorted):
        m = np.zeros(self.n, bool)
        m[self.rows[keep_sorted]] = True
        return m


def rule_threshold(G, t):
    return G.mask(G.p >= t)


def rule_two_thr(G, t_first, t_rest):
    return G.mask((G.p >= t_rest) | ((G.k == 1) & (G.p >= t_first)))


def rule_expected_f(G, delta=0.0, beta=1.0, bias=0.0):
    p = G.p.astype(np.float64)
    if delta:
        lg = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
        p = 1.0 / (1.0 + np.exp(-(lg + delta)))
    csum = np.cumsum(p)
    base = np.r_[0.0, csum][G.starts][G.grp]
    cs = csum - base                                          # prefix sum within the S1
    tot = np.add.reduceat(p, G.starts)[G.grp]
    score = 1.25 * cs / (0.25 * beta * tot + G.k)
    best = np.maximum.reduceat(score, G.starts)
    kstar = np.minimum.reduceat(np.where(score >= best[G.grp] - 1e-12, G.k, 1 << 30), G.starts)
    empty = np.exp(np.add.reduceat(np.log1p(-np.clip(p, 0, 1 - 1e-9)), G.starts)) + bias
    take = best > empty
    return G.mask(take[G.grp] & (G.k <= kstar[G.grp]))


def tune(s1, o, p, lab, n_true, log=log):
    from stage2 import macro_f05
    G = _Groups(s1, o, p)
    f = lambda m: macro_f05(s1, m, lab, n_true)["macro"]          # noqa: E731
    res = []
    for t in np.round(np.arange(0.5, 0.86, 0.02), 2):
        res.append((f(rule_threshold(G, t)), {"rule": "threshold", "t": float(t)}))
    base = max(res, key=lambda r: r[0])
    log(f"threshold rule: best {base[0]:.5f} {base[1]}")
    tr = base[1]["t"]
    for tf in np.round(np.arange(0.10, tr + 1e-9, 0.05), 2):
        for trest in np.round(np.arange(max(0.5, tr - 0.1), tr + 0.11, 0.02), 2):
            res.append((f(rule_two_thr(G, tf, trest)), {"rule": "two_thr", "t_first": float(tf), "t_rest": float(trest)}))
    b2 = max((r for r in res if r[1]["rule"] == "two_thr"), key=lambda r: r[0])
    log(f"two-threshold rule: best {b2[0]:.5f} {b2[1]}")
    for delta in (-1.0, -0.5, 0.0, 0.5):
        for beta in (1.0, 1.05):
            for bias in (-0.05, 0.0, 0.05, 0.1, 0.2):
                res.append((f(rule_expected_f(G, delta, beta, bias)),
                            {"rule": "expected_f", "delta": delta, "beta": beta, "bias": bias}))
    b3 = max((r for r in res if r[1]["rule"] == "expected_f"), key=lambda r: r[0])
    log(f"expected-F0.5 rule: best {b3[0]:.5f} {b3[1]}")
    return max(res, key=lambda r: r[0]), base, b2, b3


def apply(dec, s1, o, p):
    G = _Groups(s1, o, p)
    if dec["rule"] == "two_thr":
        return rule_two_thr(G, dec["t_first"], dec["t_rest"])
    if dec["rule"] == "expected_f":
        return rule_expected_f(G, dec["delta"], dec["beta"], dec["bias"])
    return rule_threshold(G, dec["t"])


def run(data_dir, work_dir, min_gain=2e-4):
    """Tune the decision on the train out-of-fold probabilities; store it in model2.pkl."""
    from block import true_pairs
    from prep import load_prep
    from stage2 import _folds, macro_f05
    mp = os.path.join(work_dir, "model2.pkl")
    b = pickle.load(open(mp, "rb"))
    lvl = "L2" if b["use_l2"] else "L1"
    X = pd.read_parquet(os.path.join(work_dir, "feat", "train.parquet"), columns=["s1", "o"])
    s1, o = X["s1"].to_numpy(), X["o"].to_numpy()
    P1 = load_prep(work_dir, "train", (1,), ["entity_id", "ckey"])
    PO = load_prep(work_dir, "train", (2, 3), ["entity_id"])
    ex, _, _ = true_pairs(data_dir, P1["entity_id"], PO["entity_id"])
    n_true = np.zeros(len(P1), np.int64)
    np.add.at(n_true, ex["s1"].dropna().astype(np.int64).to_numpy(), 1)
    e2 = ex.dropna(subset=["s1", "o"])
    tkey = e2["s1"].astype(np.int64).to_numpy() * (1 << 32) + e2["o"].astype(np.int64).to_numpy()
    lab = np.isin(s1.astype(np.int64) * (1 << 32) + o.astype(np.int64), tkey)
    k = len(b["l1"])
    fold_all, _ = _folds(P1["entity_id"], k)
    p = np.zeros(len(X))
    for f in range(k):
        ck = pickle.load(open(os.path.join(work_dir, "ckpt", f"{lvl}_fold{f}.pkl"), "rb"))
        p[fold_all[s1] == f] = ck["oof"]
    log(f"OOF {lvl} probabilities for {len(p):,} pairs")
    best, base, b2, b3 = tune(s1, o, p, lab, n_true)
    use = best if best[0] - base[0] > min_gain else base
    m = apply(use[1], s1, o, p)
    bd = macro_f05(s1, m, lab, n_true)
    ck1 = P1["ckey"].to_numpy()
    by_c = {c: round(macro_f05(s1, m, lab, n_true, np.where(ck1 == c)[0])["macro"], 5) for c in np.unique(ck1)}
    b["decision"] = use[1]
    with open(mp + ".tmp", "wb") as fh:
        pickle.dump(b, fh)
    os.replace(mp + ".tmp", mp)
    rep = ["# Decision report (train out-of-fold " + lvl + ")", "",
           f"- threshold rule   : {base[0]:.5f} {base[1]}",
           f"- two-threshold    : {b2[0]:.5f} {b2[1]}",
           f"- expected-F0.5    : {b3[0]:.5f} {b3[1]}",
           f"- **used: {use[1]} -> OOF macro F0.5 {bd['macro']:.5f}** (gain {bd['macro'] - base[0]:+.5f})",
           f"- breakdown {bd}", f"- by country {by_c}"]
    text = "\n".join(rep) + "\n"
    open(os.path.join(work_dir, "decide_report.md"), "w").write(text)
    log(text)
    return use[1]
