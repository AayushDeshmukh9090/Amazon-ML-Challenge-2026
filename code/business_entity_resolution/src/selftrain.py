"""Self-training on test pairs (pseudo-labels) to adapt the model to France and to the harder test data.

Evidence (work/diagnose_report.md): a model that never saw a country loses 1.3-7.7 macro-F0.5 points
on it (US->India 0.900 vs 0.977 in-country), and France - absent from train - is the least certain
country on test (4.6% uncertain best candidates vs 2.8% India/US test, 1.6% train OOF).

Method (uses only the provided data):
  1. score test with the current models (work/model2.pkl)
  2. pseudo-labels: positive = the record's most probable S1 with p >= HI, negative = p <= LO
     (pairs in between are not used); France fully, India/US subsampled
  3. retrain level 1 on train + pseudo-labelled test pairs (same k folds), level 2 on train, re-tune
     the threshold on train OOF, predict test -> <out-dir>_selftrain/ (baseline output untouched)

Section A first SIMULATES it with labels: pretend a train country is unseen (train on the other
country only), self-train on it, and score it with its TRUE labels - before vs after, and the
accuracy of the pseudo-labels themselves.  Upload the self-trained file only if A shows a gain.

  python src/pipeline.py selftrain --data-dir dataset --work-dir work --out-dir output
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
import pandas as pd

import gbm
from diagnose import _labels
from io_utils import write_id_lists
from prep import load_prep
from stage2 import (AGG, _folds, _kfold, agg_features, load_features, macro_f05, one_to_one_best,
                    pair_thresholds, tune_country_thresholds, tune_threshold)

T0 = time.time()
HI, LO = 0.95, 0.05


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def pseudo_mask(s1, o, p):
    """(use, label): confident positives = the record's best S1 with p >= HI; negatives ONLY from those
    same records (their other candidates with p <= LO).  Records without a confident match are not used
    at all: labelling their low-probability pairs 'no match' would teach the model its own blind spot
    (on an unseen country those pairs contain exactly the true matches it misses)."""
    best = one_to_one_best(s1, o, p)
    pos = best & (p >= HI)
    rec_ok = pd.Series(pos).groupby(o).transform("any").to_numpy()
    neg = rec_ok & ~best & (p <= LO)
    return pos | neg, pos


def _best_f(s1, o, p, lab, n_true, ids):
    best = one_to_one_best(s1, o, p)
    return max((macro_f05(s1, best & (p >= t), lab, n_true, ids)["macro"], float(t))
               for t in np.round(np.arange(0.3, 0.91, 0.02), 2))


def simulate(data_dir, work_dir, b, max_rounds, rep, min_s1=50_000):
    X = load_features(work_dir, "train", b.get("use_emb", False))
    y, n_true, s1_ids = _labels(data_dir, work_dir, X)
    s1, o = X["s1"].to_numpy(), X["o"].to_numpy()
    ck = load_prep(work_dir, "train", (1,), ["ckey"])["ckey"].to_numpy()
    feats, backend = b["feats"], b["l1"][0].backend
    u = (pd.util.hash_array(np.asarray(["es:" + x for x in s1_ids], dtype=object)) % 1000)[s1] / 1000.0
    rows = []
    countries = [c for c in np.unique(ck) if (ck == c).sum() > min_s1]
    for tgt in countries:
        for src in countries:
            if src == tgt:
                continue
            tr = ck[s1] == src
            ev = ck[s1] == tgt
            ev_ids = np.where(ck == tgt)[0]
            es = tr & (u < 0.05)
            m0 = gbm.fit(backend, X.loc[tr & ~es, feats], y[tr & ~es].astype(np.int8), feats, max_rounds,
                         Xva=X.loc[es, feats], yva=y[es].astype(np.int8), log=log)
            p0 = m0.predict(X.loc[ev, feats])
            f0, t0 = _best_f(s1[ev], o[ev], p0, y[ev], n_true, ev_ids)
            use, lab = pseudo_mask(s1[ev], o[ev], p0)
            acc_pos = float(y[ev][use & lab].mean()) if (use & lab).any() else float("nan")
            acc_neg = float(1 - y[ev][use & ~lab].mean()) if (use & ~lab).any() else float("nan")
            ev_idx = np.where(ev)[0][use]
            Xa = pd.concat([X.loc[tr & ~es, feats], X.iloc[ev_idx][feats]], ignore_index=True)
            ya = np.concatenate([y[tr & ~es], lab[use]]).astype(np.int8)
            m1 = gbm.fit(backend, Xa, ya, feats, max_rounds, Xva=X.loc[es, feats], yva=y[es].astype(np.int8),
                         log=log)
            del Xa
            p1 = m1.predict(X.loc[ev, feats])
            f1, t1 = _best_f(s1[ev], o[ev], p1, y[ev], n_true, ev_ids)
            rows.append({"unseen": tgt, "trained_on": src, "F05_before": round(f0, 5), "F05_after": round(f1, 5),
                         "gain": round(f1 - f0, 5), "pseudo_pairs": int(use.sum()),
                         "pseudo_pos_precision": round(acc_pos, 5), "pseudo_neg_precision": round(acc_neg, 5)})
            log(f"SIM {src}->{tgt}: before {f0:.5f} (t={t0}) after {f1:.5f} (t={t1}); pseudo pos prec {acc_pos:.4f}")
    rep += ["## A. Simulation: self-training on a country the model never saw (scored with TRUE labels)", "",
            pd.DataFrame(rows).to_markdown(index=False), "",
            "gain > 0 in both directions = self-training helps an unseen country -> upload the self-trained file.",
            ""]
    return rows


def production(data_dir, work_dir, out_dir, b, n_folds, max_rounds, other_frac, rep):
    use_emb = b.get("use_emb", False)
    feats, feats2, backend = b["feats"], b["feats2"], b["l1"][0].backend
    # ---- 1. score test with the current models, derive pseudo-labels
    Xt = load_features(work_dir, "test", use_emb)
    st1, ot = Xt["s1"].to_numpy(), Xt["o"].to_numpy()
    p = np.mean([m.predict(Xt[feats]) for m in b["l1"]], axis=0)
    if b["use_l2"]:
        A = agg_features(Xt, p)
        for c in AGG:
            Xt[c] = A[c].to_numpy()
        p = np.mean([m.predict(Xt[feats2]) for m in b["l2"]], axis=0)
    ckt = load_prep(work_dir, "test", (1,), ["ckey"])["ckey"].to_numpy()
    tids = load_prep(work_dir, "test", (1,), ["entity_id"])["entity_id"]
    use, lab = pseudo_mask(st1, ot, p)
    known = set(np.unique(load_prep(work_dir, "train", (1,), ["ckey"])["ckey"]))
    r = (pd.util.hash_array(np.asarray(["ps:" + x for x in tids], dtype=object)) % 1000)[st1] / 1000.0
    unseen = ~np.isin(ckt[st1], list(known))
    use &= unseen | (r < other_frac)                  # unseen countries fully, others subsampled
    tfold_all, tu_all = _folds(tids, n_folds)
    log(f"pseudo-labels: {use.sum():,} test pairs ({lab[use].sum():,} positive); "
        f"unseen-country pairs {int((use & unseen).sum()):,}")
    P = Xt.loc[use, feats].reset_index(drop=True)
    yP = lab[use].astype(np.int8)
    foldP, uP = tfold_all[st1[use]], tu_all[st1[use]]

    # ---- 2. level 1 on train + pseudo (same folds)
    X = load_features(work_dir, "train", use_emb)
    y, n_true, s1_ids = _labels(data_dir, work_dir, X)
    s1, o = X["s1"].to_numpy(), X["o"].to_numpy()
    fold_all, u_all = _folds(s1_ids, n_folds)
    nt = len(X)
    Z = pd.concat([X[feats], P], ignore_index=True)
    del P
    yz = np.concatenate([y.astype(np.int8), yP])
    fz = np.concatenate([fold_all[s1], foldP])
    uz = np.concatenate([u_all[s1], uP])
    ckpt = os.path.join(work_dir, "ckpt")
    sig = {"st": True, "hi": HI, "lo": LO, "other_frac": other_frac, "n_pseudo": int(len(yP)),
           "feat_mtime": os.path.getmtime(os.path.join(work_dir, "feat", "train.parquet")),
           "folds": n_folds, "max_rounds": max_rounds, "feats": feats}
    pz, m1, it1 = _kfold(backend, Z, yz, feats, fz, uz, n_folds, max_rounds, "ST_L1", ckpt, sig)
    del Z
    p1 = pz[:nt]
    lab_tr = y.astype(bool)
    t1, sc1 = tune_threshold(s1, o, p1, lab_tr, n_true)
    # ---- 3. level 2 on train (aggregates of the new level-1 OOF)
    A = agg_features(X, p1)
    for c in AGG:
        X[c] = A[c].to_numpy()
    fold, u = fold_all[s1], u_all[s1]
    p2, m2, it2 = _kfold(backend, X, y.astype(np.int8), feats2, fold, u, n_folds, max_rounds, "ST_L2", ckpt,
                         {**sig, "level": 2})
    t2, sc2 = tune_threshold(s1, o, p2, lab_tr, n_true)
    use_l2 = sc2 >= sc1
    pf, tf = (p2, t2) if use_l2 else (p1, t1)
    ck = load_prep(work_dir, "train", (1,), ["ckey"])["ckey"].to_numpy()
    thresholds = tune_country_thresholds(s1, o, pf, lab_tr, n_true, ck, tf)
    del X
    # ---- 4. predict test
    q = np.mean([m.predict(Xt[feats]) for m in m1], axis=0)
    if use_l2:
        A = agg_features(Xt, q)
        for c in AGG:
            Xt[c] = A[c].to_numpy()
        q = np.mean([m.predict(Xt[feats2]) for m in m2], axis=0)
    keep = one_to_one_best(st1, ot, q) & (q >= pair_thresholds(thresholds, ckt[st1]))
    oids = load_prep(work_dir, "test", (2, 3), ["entity_id"])["entity_id"].to_numpy()
    sid = tids.to_numpy()
    os.makedirs(out_dir, exist_ok=True)
    write_id_lists(os.path.join(out_dir, "matching_results.tsv"), sid,
                   {sid[k]: v for k, v in pd.Series(oids[ot[keep]]).groupby(st1[keep]).apply(list).items()},
                   "matched_entity_ids")
    write_id_lists(os.path.join(out_dir, "candidate_pairs.tsv"), sid,
                   {sid[k]: v for k, v in pd.Series(oids[ot]).groupby(st1).apply(list).items()},
                   "candidate_entity_ids")
    nm = np.bincount(st1[keep], minlength=len(sid))
    per_c = {c: f"{nm[ckt == c].mean():.2f} matches/S1, empty {np.mean(nm[ckt == c] == 0):.2%}"
             for c in np.unique(ckt)}
    rep += ["## B. Production self-training", "",
            f"- pseudo-labelled test pairs used: {len(yP):,} ({int(yP.sum()):,} positive), HI={HI}, LO={LO}",
            f"- train OOF after self-training: L1 {sc1:.5f}, L2 {sc2:.5f} (used {'L2' if use_l2 else 'L1'}); "
            f"thresholds {thresholds}",
            f"- test predictions: {per_c}", f"- written to {out_dir}/ (baseline output/ untouched)", ""]


def run(data_dir, work_dir, out_dir, n_folds=3, max_rounds=3000, other_frac=0.3, skip_sim=False, min_s1=50_000):
    b = pickle.load(open(os.path.join(work_dir, "model2.pkl"), "rb"))
    rep = ["# Self-training report", ""]
    if not skip_sim:
        simulate(data_dir, work_dir, b, max_rounds, rep, min_s1)
        open(os.path.join(work_dir, "selftrain_report.md"), "w").write("\n".join(rep) + "\n")
    production(data_dir, work_dir, out_dir.rstrip("/\\") + "_selftrain", b, n_folds, max_rounds, other_frac, rep)
    text = "\n".join(rep) + "\n"
    open(os.path.join(work_dir, "selftrain_report.md"), "w").write(text)
    print(text)
