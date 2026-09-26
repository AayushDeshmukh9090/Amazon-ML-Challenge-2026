"""Stage 2: pruned candidates -> features -> 2-level GBM (k-fold OOF by S1 entity) -> 1-to-1 decision -> output.

Level 1 scores each pair from its own features.  Level 2 re-scores it with the level-1 OUT-OF-FOLD
probabilities aggregated over the record's competing S1 entities and over the S1 entity's other
candidates (best other probability, number of strong candidates, ...), which is what decides
"no match" entities and near-duplicate distractors.  Test is scored by averaging the fold models,
so its probabilities follow the same distribution as the OOF ones the threshold is tuned on.

  python src/pipeline.py features2 --work-dir work            # both splits, cached to work/feat/
  python src/pipeline.py train2    --data-dir dataset --work-dir work
  python src/pipeline.py predict2  --data-dir dataset --work-dir work --out-dir output

Pairs: the learned pre-filter's kept set (prefilter.py, default) or, with --R/--F, the blocking pairs
with r_rev <= R or r_fwd <= F.  That set is exactly what the model scores -> candidate_pairs.tsv.

Decision: GT is strictly 1-to-1 from the S2/S3 side (EDA: 0 of 7.6M matched ids are shared), so
each S2/S3 record keeps only its most probable S1, then a probability threshold tuned for the exact
macro F0.5 over ALL training S1 entities (vectorised, singletons included).
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import pandas as pd

import gbm
from block import true_pairs
from features2 import PREP_COLS, build, feature_columns
from io_utils import write_id_lists
from prep import load_prep

T0 = time.time()
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


def _folds(ids: pd.Series, k: int) -> np.ndarray:
    """Deterministic fold per S1 entity (hash of its id)."""
    h = pd.util.hash_array(ids.values.astype(str))
    return (h % k).astype(np.int8), (h // k % 10_000) / 10_000.0


# ----------------------------------------------------------------------------- decision (vectorised)
def one_to_one_best(s1: np.ndarray, o: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Boolean mask: pair is the most probable S1 for its S2/S3 record (GT is 1-to-1 from that side)."""
    order = np.lexsort((-p, o))
    first = np.ones(len(order), bool)
    first[1:] = o[order][1:] != o[order][:-1]
    mask = np.zeros(len(p), bool)
    mask[order[first]] = True
    return mask


def macro_f05(s1, pred, label, n_true, ids=None):
    """Exact leaderboard metric: per S1 F0.5 = 1.25*TP / (0.25*|truth| + |pred|), singletons included.
    s1/pred/label are per-pair arrays; n_true[i] = true matches of S1 i (incl. ones not in candidates)."""
    n = len(n_true)
    npred = np.bincount(s1[pred], minlength=n)
    tp = np.bincount(s1[pred & label], minlength=n)
    f = np.where(n_true == 0, (npred == 0).astype(float),
                 np.where(npred == 0, 0.0, 1.25 * tp / (0.25 * n_true + np.maximum(npred, 1))))
    if ids is not None:
        f, nt = f[ids], n_true[ids]
    else:
        nt = n_true
    return {"macro": float(f.mean()),
            "singleton": (float(f[nt == 0].mean()) if (nt == 0).any() else float("nan"), int((nt == 0).sum())),
            "one": (float(f[nt == 1].mean()) if (nt == 1).any() else float("nan"), int((nt == 1).sum())),
            "multi": (float(f[nt > 1].mean()) if (nt > 1).any() else float("nan"), int((nt > 1).sum()))}


def tune_threshold(s1, o, p, label, n_true, log=log):
    best_mask = one_to_one_best(s1, o, p)
    res = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = best_mask & (p >= t)
        res.append((macro_f05(s1, pred, label, n_true)["macro"], float(t)))
    res.sort(reverse=True)
    for sc, t in res[:5]:
        log(f"  threshold {t:.2f}: macro F0.5 {sc:.5f}")
    return res[0][1], res[0][0]


def tune_country_thresholds(s1, o, p, label, n_true, ckey_s1, t_global, min_s1=50_000, min_gain=3e-4, log=log):
    """Per-country threshold, adopted only when it beats the global one on that country by > min_gain
    (hundreds of thousands of entities per country -> no noise fitting). Unseen countries (France)
    fall back to the global threshold."""
    best_mask = one_to_one_best(s1, o, p)
    out = {"_global": float(t_global)}
    for c in np.unique(ckey_s1):
        ids = np.where(ckey_s1 == c)[0]
        if len(ids) < min_s1:
            continue
        base = macro_f05(s1, best_mask & (p >= t_global), label, n_true, ids)["macro"]
        grid = [(macro_f05(s1, best_mask & (p >= t), label, n_true, ids)["macro"], float(t))
                for t in np.round(np.arange(0.05, 0.96, 0.01), 2)]
        sc, t = max(grid)
        log(f"  country {c}: global t={t_global} -> {base:.5f}; own t={t} -> {sc:.5f}")
        if sc - base > min_gain:
            out[str(c)] = t
    return out


def pair_thresholds(thresholds: dict, ckey_per_pair: np.ndarray) -> np.ndarray:
    t = np.full(len(ckey_per_pair), thresholds["_global"], dtype=np.float64)
    for c, v in thresholds.items():
        if c != "_global":
            t[ckey_per_pair == c] = v
    return t


# ----------------------------------------------------------------------------- level-2 (collective) features
AGG = ["l1_p", "l1_rank_o", "l1_best_other_o", "l1_gap_o", "l1_n_hi_o", "l1_sum_o",
       "l1_rank_s1", "l1_max_s1", "l1_best_other_s1", "l1_sum_s1", "l1_n_hi_s1", "l1_n_assigned_s1",
       "l1_is_best_o"]


def _best_other(key, v):
    """For each row: max of v over the OTHER rows with the same key (-1 if none)."""
    order = np.lexsort((-v, key))
    ks, vs = key[order], v[order]
    first = np.ones(len(ks), bool)
    first[1:] = ks[1:] != ks[:-1]
    grp = np.cumsum(first) - 1
    top = vs[first]
    second = np.full(len(top), -1.0)
    sec_idx = np.where(~first & np.r_[False, first[:-1]])[0]       # the 2nd row of each group
    second[grp[sec_idx]] = vs[sec_idx]
    out_sorted = np.where(first, second[grp], top[grp])
    out = np.empty(len(v))
    out[order] = out_sorted
    return out


def agg_features(X: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    s1, o = X["s1"].to_numpy(), X["o"].to_numpy()
    ps = pd.Series(p)
    A = pd.DataFrame({"l1_p": p.astype(np.float32)})
    go, gs = ps.groupby(o), ps.groupby(s1)
    A["l1_rank_o"] = go.rank(ascending=False, method="first").to_numpy(np.float32)
    A["l1_best_other_o"] = _best_other(o, p).astype(np.float32)
    A["l1_gap_o"] = (p - A["l1_best_other_o"].to_numpy()).astype(np.float32)
    hi = (p >= 0.5).astype(np.float32)
    A["l1_n_hi_o"] = pd.Series(hi).groupby(o).transform("sum").to_numpy(np.float32)
    A["l1_sum_o"] = go.transform("sum").to_numpy(np.float32)
    A["l1_rank_s1"] = gs.rank(ascending=False, method="first").to_numpy(np.float32)
    A["l1_max_s1"] = gs.transform("max").to_numpy(np.float32)
    A["l1_best_other_s1"] = _best_other(s1, p).astype(np.float32)
    A["l1_sum_s1"] = gs.transform("sum").to_numpy(np.float32)
    A["l1_n_hi_s1"] = pd.Series(hi).groupby(s1).transform("sum").to_numpy(np.float32)
    best = one_to_one_best(s1, o, p)
    A["l1_is_best_o"] = best.astype(np.float32)
    A["l1_n_assigned_s1"] = pd.Series((best & (p >= 0.5)).astype(np.float32)).groupby(s1).transform(
        "sum").to_numpy(np.float32)
    return A


# ----------------------------------------------------------------------------- training
def _kfold(backend, X, y, feats, fold, u, k, max_rounds, tag, ckpt_dir=None, sig=None, log=log):
    """k-fold by S1 entity -> (OOF predictions, models, best iterations).
    Each finished fold is checkpointed to ckpt_dir; a rerun with the same signature (same features
    file, folds, backend, settings) reloads it instead of retraining -> interrupted runs resume."""
    oof = np.zeros(len(X))
    models, iters = [], []
    for f in range(k):
        path = os.path.join(ckpt_dir, f"{tag}_fold{f}.pkl") if ckpt_dir else None
        if path and os.path.exists(path):
            with open(path, "rb") as fh:
                ck = pickle.load(fh)
            if ck.get("sig") == sig:
                oof[fold == f] = ck["oof"]
                models.append(ck["model"])
                iters.append(ck["model"].best_iteration)
                log(f"{tag} fold {f}: resumed from checkpoint (best_iter {ck['model'].best_iteration})")
                continue
        tr = fold != f
        es = (fold == f) & (u < 0.08)                         # early-stopping slice of the held-out fold
        m = gbm.fit(backend, X.loc[tr, feats], y[tr], feats, max_rounds, seed=f,
                    Xva=X.loc[es, feats], yva=y[es], log=log)
        part = m.predict(X.loc[fold == f, feats])
        oof[fold == f] = part
        models.append(m)
        iters.append(m.best_iteration)
        log(f"{tag} fold {f}: trained on {tr.sum():,} pairs, best_iter {m.best_iteration}")
        if path:
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(path + ".tmp", "wb") as fh:
                pickle.dump({"sig": sig, "oof": part, "model": m}, fh)
            os.replace(path + ".tmp", path)
    return oof, models, iters


def train(data_dir, work_dir, n_folds=3, max_rounds=3000, backend="auto", **_):
    backend = gbm.resolve(backend)
    log(f"model backend: {backend}, {n_folds} folds, max rounds {max_rounds}")
    X = pd.read_parquet(feat_path(work_dir, "train"))
    s1_ids = load_prep(work_dir, "train", (1,), ["entity_id"])["entity_id"]
    o_ids = load_prep(work_dir, "train", (2, 3), ["entity_id"])["entity_id"]
    ex, gt, sizes = true_pairs(data_dir, s1_ids, o_ids)
    n_true = np.zeros(len(s1_ids), np.int64)
    np.add.at(n_true, ex["s1"].dropna().astype(np.int64).to_numpy(), 1)
    e2 = ex.dropna(subset=["s1", "o"])
    tkey = e2["s1"].astype(np.int64).to_numpy() * (1 << 32) + e2["o"].astype(np.int64).to_numpy()
    key = X["s1"].to_numpy().astype(np.int64) * (1 << 32) + X["o"].to_numpy().astype(np.int64)
    y = np.isin(key, tkey).astype(np.int8)
    del key, tkey
    fold_all, u_all = _folds(s1_ids, n_folds)
    s1 = X["s1"].to_numpy()
    o = X["o"].to_numpy()
    fold, u = fold_all[s1], u_all[s1]
    feats = feature_columns(X)
    log(f"train pairs {len(X):,} (pos {int(y.sum()):,}); features {len(feats)}; "
        f"pair recall of scored pairs {y.sum() / len(ex):.4f}")
    from sklearn.metrics import average_precision_score, roc_auc_score
    lab = y.astype(bool)

    # ---- level 1  (fold checkpoints in work/ckpt: an interrupted run resumes at the next fold)
    ckpt = os.path.join(work_dir, "ckpt")
    sig = {"feat_mtime": os.path.getmtime(feat_path(work_dir, "train")), "folds": n_folds, "backend": backend,
           "max_rounds": max_rounds, "feats": feats, "params": gbm.XGB if backend == "xgb" else gbm.LGB}
    p1, m1, it1 = _kfold(backend, X, y, feats, fold, u, n_folds, max_rounds, "L1", ckpt, sig)
    t1, sc1 = tune_threshold(s1, o, p1, lab, n_true)
    log(f"L1 OOF AUC {roc_auc_score(y, p1):.5f} AP {average_precision_score(y, p1):.5f}; "
        f"macro F0.5 {sc1:.5f} @ t={t1}")

    # ---- level 2: level-1 OOF probabilities aggregated per record / per S1 entity + all features
    A = agg_features(X, p1)
    for c in AGG:
        X[c] = A[c].to_numpy()
    del A
    feats2 = feats + AGG
    p2, m2, it2 = _kfold(backend, X, y, feats2, fold, u, n_folds, max_rounds, "L2", ckpt, {**sig, "level": 2})
    t2, sc2 = tune_threshold(s1, o, p2, lab, n_true)
    log(f"L2 OOF AUC {roc_auc_score(y, p2):.5f} AP {average_precision_score(y, p2):.5f}; "
        f"macro F0.5 {sc2:.5f} @ t={t2}")

    use_l2 = sc2 >= sc1
    p_fin, t_fin, sc = (p2, t2, sc2) if use_l2 else (p1, t1, sc1)
    ck = load_prep(work_dir, "train", (1,), ["ckey"])["ckey"].to_numpy()
    thresholds = tune_country_thresholds(s1, o, p_fin, lab, n_true, ck, t_fin)
    pred = one_to_one_best(s1, o, p_fin) & (p_fin >= pair_thresholds(thresholds, ck[s1]))
    bd = macro_f05(s1, pred, lab, n_true)
    ceil = macro_f05(s1, lab, lab, n_true)
    by_c = {c: macro_f05(s1, pred, lab, n_true, np.where(ck == c)[0])["macro"] for c in np.unique(ck)}
    imp = m2[0].importance() if use_l2 else m1[0].importance()
    with open(os.path.join(work_dir, "model2.pkl"), "wb") as fh:
        pickle.dump({"l1": m1, "l2": m2 if use_l2 else None, "feats": feats, "feats2": feats2,
                     "threshold": t_fin, "thresholds": thresholds, "use_l2": use_l2}, fh)
    rep = ["# Stage-2 report", "", f"- train pairs {len(X):,}, positives {int(y.sum()):,}",
           f"- pair recall of scored pairs: {y.sum() / len(ex):.4f}",
           f"- backend {backend}, {n_folds} folds; L1 best iterations {it1}; L2 best iterations {it2}",
           f"- L1: AUC {roc_auc_score(y, p1):.5f}, macro F0.5 {sc1:.5f} @ t={t1}",
           f"- L2: AUC {roc_auc_score(y, p2):.5f}, macro F0.5 {sc2:.5f} @ t={t2}",
           f"- **used {'L2' if use_l2 else 'L1'}: OOF macro F0.5 {bd['macro']:.5f}** (all {len(s1_ids):,} train S1)",
           f"- thresholds {thresholds} (countries not listed, e.g. unseen France, use _global)",
           f"- breakdown {bd}", f"- by country {by_c}",
           f"- ceiling (perfect matcher on scored pairs) {ceil}", "", "## Top features (gain)", "",
           imp.head(40).round(0).to_string()]
    open(os.path.join(work_dir, "stage2_report.md"), "w").write("\n".join(rep) + "\n")
    log("\n".join(rep))


def predict(work_dir, out_dir):
    b = pickle.load(open(os.path.join(work_dir, "model2.pkl"), "rb"))
    X = pd.read_parquet(feat_path(work_dir, "test"))
    p = np.mean([m.predict(X[b["feats"]]) for m in b["l1"]], axis=0)     # same distribution as OOF
    log(f"L1 test scores from {len(b['l1'])} fold models")
    if b["use_l2"]:
        A = agg_features(X, p)
        for c in AGG:
            X[c] = A[c].to_numpy()
        p = np.mean([m.predict(X[b["feats2"]]) for m in b["l2"]], axis=0)
        log(f"L2 test scores from {len(b['l2'])} fold models")
    s1, o = X["s1"].to_numpy(), X["o"].to_numpy()
    ck_test = load_prep(work_dir, "test", (1,), ["ckey"])["ckey"].to_numpy()
    thresholds = b.get("thresholds", {"_global": b["threshold"]})
    keep = one_to_one_best(s1, o, p) & (p >= pair_thresholds(thresholds, ck_test[s1]))
    s1_ids = load_prep(work_dir, "test", (1,), ["entity_id"])["entity_id"].to_numpy()
    o_ids = load_prep(work_dir, "test", (2, 3), ["entity_id"])["entity_id"].to_numpy()
    os.makedirs(out_dir, exist_ok=True)
    cand = pd.Series(o_ids[o]).groupby(s1).apply(list).to_dict()
    match = pd.Series(o_ids[o[keep]]).groupby(s1[keep]).apply(list).to_dict()
    write_id_lists(os.path.join(out_dir, "candidate_pairs.tsv"), s1_ids,
                   {s1_ids[k]: v for k, v in cand.items()}, "candidate_entity_ids")
    write_id_lists(os.path.join(out_dir, "matching_results.tsv"), s1_ids,
                   {s1_ids[k]: v for k, v in match.items()}, "matched_entity_ids")
    ck = load_prep(work_dir, "test", (1,), ["ckey"])["ckey"].to_numpy()
    nm = np.bincount(s1[keep], minlength=len(s1_ids))
    per_country = {}
    for c in np.unique(ck):
        sel = ck == c
        per_country[str(c)] = {"s1": int(sel.sum()), "mean_matches": round(float(nm[sel].mean()), 4),
                               "empty_share": round(float(np.mean(nm[sel] == 0)), 4)}
        log(f"  test {c}: {sel.sum():,} S1, mean matches {nm[sel].mean():.2f}, empty {np.mean(nm[sel] == 0):.2%}")
    # identity card of this output, so an uploaded file can always be traced to its run
    info = {"written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "model_levels": "L1+L2" if b["use_l2"] else "L1", "n_fold_models": len(b["l1"]),
            "thresholds": thresholds, "n_matches": int(keep.sum()), "n_candidates": int(len(X)),
            "per_country": per_country}
    with open(os.path.join(out_dir, "run_info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    log(f"wrote {out_dir}: {len(s1_ids):,} S1 rows, {int(keep.sum()):,} matches "
        f"({keep.sum() / len(s1_ids):.2f}/S1), candidates {len(X):,}")
