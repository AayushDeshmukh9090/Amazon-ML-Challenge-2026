"""Turn pair probabilities into per-S1 match lists that maximise macro F0.5.

Two decision rules, both tuned on out-of-fold predictions:

* threshold   : predict every candidate with p >= t (after optional 1-to-1 filter)
* expected_f  : per S1 entity pick the prefix of candidates (sorted by p) that
                maximises E[F0.5] under independent Bernoulli(p) labels,
                including the empty prediction (worth P(no true match)).
                Note F0.5 = 1.25*TP / (0.25*|truth| + |pred|).

Optional 1-to-1 filter: a S2/S3 record is kept only for the S1 entity with the
highest probability (EDA shows whether GT satisfies this).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import macro_f05


def one_to_one(df: pd.DataFrame, rel: float = 1.0) -> pd.DataFrame:
    """Keep pair only if p >= rel * (best p of that candidate over all S1)."""
    best = df.groupby("cand_id")["p"].transform("max")
    return df[df["p"] >= rel * best - 1e-12]


def decide_threshold(df: pd.DataFrame, t: float) -> dict:
    sel = df[df["p"] >= t]
    return sel.groupby("s1_id")["cand_id"].apply(list).to_dict()


def decide_expected_f(df: pd.DataFrame, n_samples: int = 400, max_c: int = 25, min_p: float = 0.01,
                      bias: float = 0.0, seed: int = 0) -> dict:
    """bias > 0 makes the rule more conservative (require E[F] gain over empty > bias)."""
    rng = np.random.default_rng(seed)
    out = {}
    d = df[df["p"] >= min_p].sort_values(["s1_id", "p"], ascending=[True, False])
    for s1, g in d.groupby("s1_id", sort=False):
        p = g["p"].values[:max_c].astype(np.float64)
        ids = g["cand_id"].values[:max_c]
        n = len(p)
        U = rng.random((n_samples, n)) < p  # samples x n
        T = U.sum(1)                        # |truth| within candidates
        # rest of mass (candidates below min_p) ignored - negligible
        tp_cum = np.cumsum(U, axis=1)       # TP if we predict top-k
        k = np.arange(1, n + 1)
        ef = (1.25 * tp_cum / (0.25 * T[:, None] + k[None, :])).mean(0)
        e0 = float(np.prod(1 - p))
        kb = int(np.argmax(ef))
        if ef[kb] > e0 + bias:
            out[s1] = list(ids[:kb + 1])
    return out


def apply_rule(df: pd.DataFrame, params: dict) -> dict:
    d = one_to_one(df, params.get("rel", 1.0)) if params.get("one_to_one") else df
    if params["rule"] == "threshold":
        return decide_threshold(d, params["t"])
    return decide_expected_f(d, bias=params.get("bias", 0.0), min_p=params.get("min_p", 0.01))


def tune(df: pd.DataFrame, truth: dict, allow_one_to_one: bool = True, log=print) -> dict:
    """Grid-search the decision rule on OOF predictions. Returns best params (+score)."""
    ids = list(truth.keys())
    results = []
    o2o_opts = [False, True] if allow_one_to_one else [False]
    for o2o in o2o_opts:
        for t in np.round(np.arange(0.2, 0.96, 0.05), 3):
            prm = {"rule": "threshold", "t": float(t), "one_to_one": o2o, "rel": 1.0}
            results.append((macro_f05(apply_rule(df, prm), truth, ids), prm))
        for bias in (0.0, 0.02, 0.05, 0.1):
            prm = {"rule": "expected_f", "bias": bias, "one_to_one": o2o, "rel": 1.0}
            results.append((macro_f05(apply_rule(df, prm), truth, ids), prm))
    results.sort(key=lambda x: -x[0])
    for sc, prm in results[:8]:
        log(f"  {sc:.5f}  {prm}")
    best_sc, best = results[0]
    # refine the threshold locally
    if best["rule"] == "threshold":
        for t in np.round(np.arange(best["t"] - 0.05, best["t"] + 0.051, 0.01), 3):
            prm = {**best, "t": float(t)}
            sc = macro_f05(apply_rule(df, prm), truth, ids)
            if sc > best_sc:
                best_sc, best = sc, prm
    best = dict(best)
    best["oof_score"] = float(best_sc)
    return best
