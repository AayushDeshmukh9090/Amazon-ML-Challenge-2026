"""Macro F0.5 exactly as the leaderboard defines it (singletons included)."""
from __future__ import annotations

import numpy as np


def f_beta_entity(pred: set, true: set, beta: float = 0.5) -> float:
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f05(pred: dict, truth: dict, ids=None) -> float:
    ids = list(truth.keys()) if ids is None else list(ids)
    return float(np.mean([f_beta_entity(set(pred.get(i, ())), set(truth.get(i, ()))) for i in ids]))


def breakdown(pred: dict, truth: dict, ids=None) -> dict:
    """Score split by singleton / 1 match / multi-match S1 entities."""
    ids = list(truth.keys()) if ids is None else list(ids)
    buckets = {"singleton": [], "one": [], "multi": []}
    for i in ids:
        t = set(truth.get(i, ()))
        k = "singleton" if not t else ("one" if len(t) == 1 else "multi")
        buckets[k].append(f_beta_entity(set(pred.get(i, ())), t))
    out = {k: (float(np.mean(v)) if v else float("nan"), len(v)) for k, v in buckets.items()}
    out["macro"] = macro_f05(pred, truth, ids)
    return out


def blocking_report(cands: dict, truth: dict, n_s1: int, n_other: int) -> dict:
    """Pair recall, entity-level full-recall, and reduction ratio of a candidate set."""
    tot_true = sum(len(v) for v in truth.values())
    hit = sum(len(set(cands.get(k, ())) & v) for k, v in truth.items())
    full = np.mean([set(v) <= set(cands.get(k, ())) for k, v in truth.items() if v]) if tot_true else float("nan")
    n_c = sum(len(v) for v in cands.values())
    return {
        "pair_recall": hit / max(tot_true, 1),
        "entities_fully_covered": float(full),
        "n_candidates": int(n_c),
        "avg_cands_per_s1": n_c / max(n_s1, 1),
        "reduction_ratio": 1 - n_c / max(n_s1 * n_other, 1),
    }
