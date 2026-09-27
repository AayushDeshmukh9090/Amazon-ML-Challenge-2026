"""Error analysis on the out-of-fold predictions (model part, CPU): WHERE is macro F0.5 lost?

For every train S1 entity, the lost score (1 - F0.5) is attributed to error types:
  FN_blocking     a true pair never reached the model (not in the scored candidate set)
  FN_competition  the true record was assigned to ANOTHER S1 (1-to-1 picked a competitor)
  FN_threshold    the true S1 was the record's best candidate but p < threshold
  FP_wrong_s1     a predicted record truly belongs to another S1 entity
  FP_no_entity    a predicted record belongs to no S1 in this data (unmatched / dropped entity)
Writes work/errors_report.md (loss by type and country) and work/errors_sample.tsv (real examples
with names, addresses, probabilities and the competing S1), plus a row-order sanity check.

  python src/pipeline.py errors --data-dir dataset --work-dir work
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
import pandas as pd

from block import true_pairs
from prep import load_prep
from stage2 import _folds, macro_f05, one_to_one_best, pair_thresholds

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def run(data_dir, work_dir, n_examples=400):
    b = pickle.load(open(os.path.join(work_dir, "model2.pkl"), "rb"))
    lvl = "L2" if b["use_l2"] else "L1"
    X = pd.read_parquet(os.path.join(work_dir, "feat", "train.parquet"), columns=["s1", "o"])
    s1, o = X["s1"].to_numpy(), X["o"].to_numpy()
    P1 = load_prep(work_dir, "train", (1,), ["entity_id", "ckey", "name", "addr"])
    PO = load_prep(work_dir, "train", (2, 3), ["entity_id", "ckey", "name", "addr"])
    ex, _, _ = true_pairs(data_dir, P1["entity_id"], PO["entity_id"])
    k = len(b["l1"])
    fold_all, _ = _folds(P1["entity_id"], k)
    p = np.zeros(len(X))
    for f in range(k):
        ck = pickle.load(open(os.path.join(work_dir, "ckpt", f"{lvl}_fold{f}.pkl"), "rb"))
        p[fold_all[s1] == f] = ck["oof"]
    thr = b.get("thresholds", {"_global": b["threshold"]})
    ck1 = P1["ckey"].to_numpy()
    best = one_to_one_best(s1, o, p)
    pred = best & (p >= pair_thresholds(thr, ck1[s1]))
    n1 = len(P1)

    # truth: pair keys and record -> true S1 (1-to-1 from the record side)
    e2 = ex.dropna(subset=["o"])
    ts1, to = e2["s1"].astype(np.int64).to_numpy(), e2["o"].astype(np.int64).to_numpy()
    n_true = np.bincount(ex["s1"].astype(np.int64).to_numpy(), minlength=n1)
    rec_true = np.full(len(PO), -1, np.int64)
    rec_true[to] = ts1
    key = s1.astype(np.int64) * (1 << 32) + o.astype(np.int64)
    tkey = ts1 * (1 << 32) + to
    lab = np.isin(key, tkey)
    in_cand = np.isin(tkey, key)

    # per-pair error types
    rec_pred_s1 = np.full(len(PO), -1, np.int64)                 # S1 each record was assigned to
    rec_pred_s1[o[pred]] = s1[pred]
    rec_best_s1 = np.full(len(PO), -1, np.int64)
    rec_best_s1[o[best]] = s1[best]
    fn_block = ~in_cand
    fn_comp = in_cand & (rec_pred_s1[to] >= 0) & (rec_pred_s1[to] != ts1)
    fn_thr = in_cand & ~fn_comp & (rec_pred_s1[to] != ts1)
    fp = pred & ~lab
    fp_wrong = fp & (rec_true[o] >= 0)
    fp_none = fp & (rec_true[o] < 0)

    # per-entity F0.5 and loss attribution (loss split across the entity's error counts)
    f = np.zeros(n1)
    npred = np.bincount(s1[pred], minlength=n1)
    tp = np.bincount(s1[pred & lab], minlength=n1)
    f = np.where(n_true == 0, (npred == 0).astype(float),
                 np.where(npred == 0, 0.0, 1.25 * tp / (0.25 * n_true + np.maximum(npred, 1))))
    loss = 1 - f
    cnt = {"FN_blocking": np.bincount(ts1[fn_block], minlength=n1),
           "FN_competition": np.bincount(ts1[fn_comp], minlength=n1),
           "FN_threshold": np.bincount(ts1[fn_thr], minlength=n1),
           "FP_wrong_s1": np.bincount(s1[fp_wrong], minlength=n1),
           "FP_no_entity": np.bincount(s1[fp_none], minlength=n1)}
    tot = sum(cnt.values()).astype(float)
    rows = []
    for c in list(np.unique(ck1)) + ["ALL"]:
        sel = np.ones(n1, bool) if c == "ALL" else ck1 == c
        r = {"country": c, "S1": int(sel.sum()), "macroF05": round(float(f[sel].mean()), 5),
             "lost_points": round(float(loss[sel].sum() / n1 * 100), 3)}
        for t, v in cnt.items():
            share = np.where(tot > 0, v / np.maximum(tot, 1), 0) * loss
            r[t + " (pts)"] = round(float(share[sel].sum() / n1 * 100), 3)
            r[t + " (#)"] = int(v[sel].sum())
        rows.append(r)
    table = pd.DataFrame(rows)
    log("loss attribution done")

    # ---- examples
    rng = np.random.default_rng(0)
    nm1, ad1, id1 = P1["name"].to_numpy(), P1["addr"].to_numpy(), P1["entity_id"].to_numpy()
    nmo, ado = PO["name"].to_numpy(), PO["addr"].to_numpy()
    pmap = pd.Series(p, index=key)
    ex_rows = []

    def add(kind, a, r, other_s1):
        a, r, other_s1 = int(a), int(r), int(other_s1)          # int32 arrays -> Python ints (no overflow)
        pa = float(pmap.get(a * (1 << 32) + r, np.nan))
        po_ = float(pmap.get(other_s1 * (1 << 32) + r, np.nan)) if other_s1 >= 0 else np.nan
        ex_rows.append({"type": kind, "country": ck1[a], "s1_name": nm1[a], "s1_addr": ad1[a], "rec_name": nmo[r],
                        "rec_addr": ado[r], "p_this": round(pa, 4),
                        "other_s1_name": nm1[other_s1] if other_s1 >= 0 else "",
                        "other_s1_addr": ad1[other_s1] if other_s1 >= 0 else "", "p_other": round(po_, 4)})

    for kind, mask_t in (("FN_blocking", fn_block), ("FN_competition", fn_comp), ("FN_threshold", fn_thr)):
        idx = np.where(mask_t)[0]
        for i in rng.choice(idx, size=min(n_examples // 5, len(idx)), replace=False):
            add(kind, ts1[i], to[i], rec_pred_s1[to[i]] if kind == "FN_competition" else rec_best_s1[to[i]])
    for kind, mask_p in (("FP_wrong_s1", fp_wrong), ("FP_no_entity", fp_none)):
        idx = np.where(mask_p)[0]
        for i in rng.choice(idx, size=min(n_examples // 5, len(idx)), replace=False):
            add(kind, s1[i], o[i], rec_true[o[i]])
    samp = pd.DataFrame(ex_rows)
    samp.to_csv(os.path.join(work_dir, "errors_sample.tsv"), sep="\t", index=False)

    # ---- row-order sanity check (data artefact, not a feature)
    pos1 = np.arange(n1)[ts1] / n1
    po_all = np.arange(len(PO))
    n2 = int(PO["entity_id"].str.startswith("S2").sum())
    pos_o = np.where(to < n2, po_all[to] / max(n2, 1), (po_all[to] - n2) / max(len(PO) - n2, 1))
    rho = pd.Series(pos1).corr(pd.Series(pos_o), method="spearman")

    rep = ["# Error analysis (train, out-of-fold " + lvl + ")", "",
           f"OOF macro F0.5 {f.mean():.5f}; lost points = 100 x (1 - macroF). Loss of each entity is split "
           "across its error types in proportion to their counts.", "",
           table.to_markdown(index=False), "",
           "- FN_blocking: true pair never scored | FN_competition: record given to another S1 | "
           "FN_threshold: true S1 was the record's best but p < threshold | FP_wrong_s1: record belongs to "
           "another S1 | FP_no_entity: record has no S1 here", "",
           f"Row-order check: Spearman(position of S1 in its file, position of its match in its file) = "
           f"{rho:.4f} (|rho| near 0 = no ordering artefact)", "",
           "## Examples per type (first 8 each; all in errors_sample.tsv)", ""]
    for t in samp["type"].unique() if len(samp) else []:
        rep += [f"### {t}", "", samp[samp.type == t].head(8).drop(columns=["type"]).to_markdown(index=False), ""]
    text = "\n".join(rep) + "\n"
    open(os.path.join(work_dir, "errors_report.md"), "w").write(text)
    print(text)
