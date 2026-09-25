"""Stage 2: prune candidates -> features -> LightGBM (2-fold OOF by S1 entity) -> 1-to-1 decision -> output.

  python src/pipeline.py features2 --work-dir work            # both splits, cached to work/feat/
  python src/pipeline.py train2    --data-dir dataset --work-dir work
  python src/pipeline.py predict2  --data-dir dataset --work-dir work --out-dir output

Pairs: the learned pre-filter's kept set (prefilter.py, default) or, with --R/--F, the blocking pairs
with r_rev <= R or r_fwd <= F.  That set is exactly what the model scores -> candidate_pairs.tsv.

Decision: GT is strictly 1-to-1 from the S2/S3 side (EDA: 0 of 7.6M matched ids are shared), so
each S2/S3 record keeps only its most probable S1; then a threshold / expected-F0.5 rule per S1
entity, tuned on out-of-fold predictions for macro F0.5 (singletons included).
"""
from __future__ import annotations

import json
import os
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from block import true_pairs
from decide import apply_rule
from features2 import PREP_COLS, build, feature_columns
from io_utils import write_id_lists
from metrics import breakdown
from prep import load_prep

T0 = time.time()
LGB = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=100,
           feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, max_bin=255,
           verbose=-1, num_threads=0)


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def feat_path(work_dir, split):
    return os.path.join(work_dir, "feat", f"{split}.parquet")


def features(work_dir, split, R, F, jobs, force=False):
    path = feat_path(work_dir, split)
    meta = os.path.join(work_dir, "feat", f"{split}.meta.json")
    pruned = os.path.join(work_dir, "cands", f"{split}_pruned.parquet")
    src = pruned if R is None and os.path.exists(pruned) else os.path.join(work_dir, "cands", f"{split}.parquet")
    if (os.path.exists(path) and not force and os.path.exists(meta) and json.load(open(meta)) == {"R": R, "F": F}
            and os.path.getmtime(path) > os.path.getmtime(src)):
        log(f"{split}: cached features {path}")
        return
    if R is None and os.path.exists(pruned):                      # learned pre-filter (prefilter.py)
        pairs = pd.read_parquet(pruned)
        log(f"{split}: {len(pairs):,} pairs from the learned pre-filter")
    else:
        R, F = (R, F) if R is not None else (5, 5)
        cands = pd.read_parquet(os.path.join(work_dir, "cands", f"{split}.parquet"))
        pairs = cands[(cands.r_rev <= R) | (cands.r_fwd <= F)].reset_index(drop=True)
        log(f"{split}: pruned {len(cands):,} -> {len(pairs):,} pairs (R={R}, F={F})")
        del cands
    P1 = load_prep(work_dir, split, (1,), PREP_COLS)
    PO = load_prep(work_dir, split, (2, 3), PREP_COLS)
    X = build(pairs, P1, PO, jobs or os.cpu_count(), log=log)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    X.to_parquet(path, index=False)
    json.dump({"R": R, "F": F}, open(meta, "w"))
    log(f"{split}: features {X.shape} -> {path}")


def _fold(ids: pd.Series) -> np.ndarray:
    h = pd.util.hash_array(ids.values.astype(str))
    return (h % 2).astype(np.int8), (h // 2 % 10_000) / 10_000.0


def _truth_ints(ex, n_s1):
    t = {i: set() for i in range(n_s1)}
    for a, b in zip(ex.s1.values, ex.o.values):
        if a == a and b == b:
            t[int(a)].add(int(b))
    return t


def tune_decision(P: pd.DataFrame, truth: dict, ids, log=log):
    d = P[["s1", "o", "p"]].rename(columns={"s1": "s1_id", "o": "cand_id"})
    res = []
    for t in np.round(np.arange(0.10, 0.91, 0.05), 2):
        prm = {"rule": "threshold", "t": float(t), "one_to_one": True, "rel": 1.0}
        res.append((breakdown(apply_rule(d, prm), truth, ids)["macro"], prm))
    res.sort(key=lambda x: -x[0])
    bt = res[0][1]["t"]
    for t in np.round(np.arange(bt - 0.04, bt + 0.041, 0.01), 2):
        prm = {"rule": "threshold", "t": float(t), "one_to_one": True, "rel": 1.0}
        res.append((breakdown(apply_rule(d, prm), truth, ids)["macro"], prm))
    for bias in (0.0, 0.02, 0.05):
        prm = {"rule": "expected_f", "bias": bias, "one_to_one": True, "rel": 1.0, "min_p": 0.02}
        res.append((breakdown(apply_rule(d, prm), truth, ids)["macro"], prm))
    res.sort(key=lambda x: -x[0])
    for sc, prm in res[:6]:
        log(f"  {sc:.5f} {prm}")
    return res[0][1], res[0][0]


def train(data_dir, work_dir, train_frac=0.25, tune_s1=400_000, max_rounds=2000):
    X = pd.read_parquet(feat_path(work_dir, "train"))
    s1_ids = load_prep(work_dir, "train", (1,), ["entity_id"])["entity_id"]
    o_ids = load_prep(work_dir, "train", (2, 3), ["entity_id"])["entity_id"]
    ex, gt, sizes = true_pairs(data_dir, s1_ids, o_ids)
    key = X["s1"].values.astype(np.int64) * (1 << 32) + X["o"].values.astype(np.int64)
    e2 = ex.dropna(subset=["s1", "o"])
    tkey = e2["s1"].astype(np.int64).values * (1 << 32) + e2["o"].astype(np.int64).values
    X["label"] = np.isin(key, tkey).astype(np.int8)
    fold_all, u_all = _fold(s1_ids)
    X["fold"] = fold_all[X["s1"].values]
    in_sample = u_all[X["s1"].values] < 2 * train_frac
    feats = feature_columns(X)
    log(f"train pairs {len(X):,} (pos {X.label.sum():,}); features {len(feats)}; "
        f"pair recall after pruning {X.label.sum() / len(ex):.4f}")

    # ---- 2-fold OOF by S1 entity
    X["p"] = np.nan
    iters = []
    for f in (0, 1):
        tr = (X.fold.values == f) & in_sample
        va_all = X.fold.values != f
        es = va_all & (u_all[X["s1"].values] < 0.04)          # small early-stopping slice of the other fold
        dtr = lgb.Dataset(X.loc[tr, feats], X.label[tr], free_raw_data=True)
        dva = lgb.Dataset(X.loc[es, feats], X.label[es], reference=dtr)
        m = lgb.train({**LGB, "seed": f}, dtr, max_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
        iters.append(m.best_iteration)
        X.loc[va_all, "p"] = m.predict(X.loc[va_all, feats], num_iteration=m.best_iteration)
        log(f"fold {f}: trained on {tr.sum():,} pairs, best_iter {m.best_iteration}")
    from sklearn.metrics import average_precision_score, roc_auc_score
    log(f"OOF pair AUC {roc_auc_score(X.label, X.p):.5f}  AP {average_precision_score(X.label, X.p):.5f}")

    # ---- tune decision on a random subset of S1 (all S1, incl. singletons / no candidates)
    truth = _truth_ints(ex, len(s1_ids))
    rng = np.random.default_rng(0)
    ids = rng.choice(len(s1_ids), size=min(tune_s1, len(s1_ids)), replace=False)
    idset = set(ids.tolist())
    # 1-to-1 must see ALL S1 competitors of each record -> keep every pair of records touching the subset
    touch = X["o"][X["s1"].isin(idset)].unique()
    P = X.loc[X["o"].isin(touch), ["s1", "o", "p"]]
    params, score = tune_decision(P, truth, ids)
    pred = apply_rule(P.rename(columns={"s1": "s1_id", "o": "cand_id"}), params)
    bd = breakdown(pred, truth, ids)
    sub = X.loc[X["s1"].isin(set(ids.tolist())), ["s1", "o"]].groupby("s1")["o"].apply(set).to_dict()
    ceil = breakdown({i: [o for o in truth[i] if o in sub.get(i, ())] for i in ids}, truth, ids)
    log(f"OOF macro F0.5 {score:.5f}  breakdown {bd}")
    log(f"ceiling after pruning (perfect matcher on scored pairs): {ceil}")

    # ---- final model on the union of both training samples
    n_rounds = int(np.mean(iters) * 1.15) + 1
    m = lgb.train({**LGB, "seed": 7}, lgb.Dataset(X.loc[in_sample, feats], X.label[in_sample]), n_rounds)
    imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
    with open(os.path.join(work_dir, "model2.pkl"), "wb") as fh:
        pickle.dump({"model": m, "feats": feats, "params": params}, fh)
    rep = ["# Stage-2 report", "", f"- train pairs {len(X):,}, positives {int(X.label.sum()):,}",
           f"- pair recall after pruning: {X.label.sum() / len(ex):.4f}",
           f"- OOF pair AUC {roc_auc_score(X.label, X.p):.5f}, AP {average_precision_score(X.label, X.p):.5f}",
           f"- best iterations {iters}, final rounds {n_rounds}",
           f"- decision: {params}", f"- **OOF macro F0.5 {score:.5f}**; breakdown {bd}",
           f"- ceiling after pruning: {ceil}", "", "## Top features (gain)", "",
           imp.head(40).round(0).to_string()]
    open(os.path.join(work_dir, "stage2_report.md"), "w").write("\n".join(rep) + "\n")
    log("\n".join(rep))


def predict(work_dir, out_dir):
    b = pickle.load(open(os.path.join(work_dir, "model2.pkl"), "rb"))
    X = pd.read_parquet(feat_path(work_dir, "test"))
    X["p"] = b["model"].predict(X[b["feats"]])
    s1_ids = load_prep(work_dir, "test", (1,), ["entity_id"])["entity_id"].values
    o_ids = load_prep(work_dir, "test", (2, 3), ["entity_id"])["entity_id"].values
    pred = apply_rule(X[["s1", "o", "p"]].rename(columns={"s1": "s1_id", "o": "cand_id"}), b["params"])
    os.makedirs(out_dir, exist_ok=True)
    cand = X.groupby("s1")["o"].apply(list).to_dict()
    write_id_lists(os.path.join(out_dir, "candidate_pairs.tsv"), s1_ids,
                   {s1_ids[k]: [o_ids[o] for o in v] for k, v in cand.items()}, "candidate_entity_ids")
    write_id_lists(os.path.join(out_dir, "matching_results.tsv"), s1_ids,
                   {s1_ids[k]: [o_ids[o] for o in v] for k, v in pred.items()}, "matched_entity_ids")
    n = sum(len(v) for v in pred.values())
    log(f"wrote {out_dir}: {len(s1_ids):,} S1 rows, {n:,} matches "
        f"({n / len(s1_ids):.2f}/S1), candidates {len(X):,}")
