"""End-to-end pipeline:  data -> blocking -> features -> LightGBM -> decision -> output/.

  python src/pipeline.py train   --data-dir dataset --work-dir work
  python src/pipeline.py predict --data-dir dataset --work-dir work --out-dir output
  python src/pipeline.py all     ...   (train then predict)

`train` runs 5-fold GroupKFold (grouped by S1 entity) to get out-of-fold
probabilities, tunes the decision rule for macro F0.5 on them, reports a
leave-one-country-out check (proxy for the unseen test country), then fits
the final models on all training pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import BlockConfig, Space, cands_to_frame, generate_candidates, prepare  # noqa: E402
from decide import apply_rule, tune  # noqa: E402
from features import build_features, feature_columns  # noqa: E402
from io_utils import gt_to_dict, load_split, write_id_lists  # noqa: E402
from metrics import blocking_report, breakdown  # noqa: E402

LGB_PARAMS = dict(objective="binary", learning_rate=0.04, num_leaves=63, min_child_samples=40,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  max_bin=255, verbose=-1, n_jobs=-1)
SEEDS = (0, 1, 2)
MAX_ROUNDS = 3000
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def data_sig(data_dir, split):
    """Fingerprint of the split's files (name, size) so a cache is never reused for other data."""
    d = os.path.join(data_dir, split)
    return sorted((f, os.path.getsize(os.path.join(d, f))) for f in os.listdir(d) if f.endswith(".tsv"))


def build_split(data_dir, split, cfg: BlockConfig, work_dir=None, features=True, use_cache=True):
    """Blocking (+ features). With work_dir, the result is cached in work_dir/cache_<split>.pkl
    so the (slow) feature build can be done once - e.g. locally - and reused for training elsewhere."""
    cache = os.path.join(work_dir, f"cache_{split}.pkl") if work_dir else None
    if features and use_cache and cache and os.path.exists(cache):
        with open(cache, "rb") as fh:
            obj = pickle.load(fh)
        if obj.get("cfg") == cfg.__dict__ and obj.get("sig") == data_sig(data_dir, split):
            log(f"loaded cached {split} features from {cache}: {obj['X'].shape}")
            return obj["s1"], obj["oth"], obj["gt"], obj["cands"], obj["X"]
        log("cache exists but data or blocking config changed -> rebuilding")
    s1, s2, s3, gt = load_split(data_dir, split)
    s1 = prepare(s1)
    oth = prepare(pd.concat([s2, s3], ignore_index=True))
    log(f"{split}: S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}")
    space = Space(s1, oth)
    log("tf-idf spaces fitted")
    cands, per = generate_candidates(s1, oth, space, cfg, return_sources=True)
    pairs = cands_to_frame(cands)
    log(f"candidates: {len(pairs):,} pairs ({len(pairs) / len(s1):.1f}/S1)")
    if gt is not None:
        truth = gt_to_dict(gt)
        rep = blocking_report(cands, truth, len(s1), len(oth))
        log("blocking (union):", json.dumps({k: round(v, 5) for k, v in rep.items()}))
        for name, d in per.items():
            r = blocking_report({k: v for k, v in d.items()}, truth, len(s1), len(oth))
            log(f"  blocker {name}: pair_recall={r['pair_recall']:.4f} avg={r['avg_cands_per_s1']:.1f}")
            only = {k: v - set().union(*[per[o].get(k, set()) for o in per if o != name]) for k, v in d.items()}
            ro = blocking_report(only, truth, len(s1), len(oth))
            log(f"     unique contribution: {ro['pair_recall']:.4f}")
    if not features:
        return s1, oth, gt, cands, None
    X = build_features(pairs, s1, oth, space)
    log(f"features: {X.shape}")
    if cache:
        os.makedirs(work_dir, exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump({"cfg": cfg.__dict__, "sig": data_sig(data_dir, split), "s1": s1, "oth": oth, "gt": gt, "cands": cands, "X": X}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        log(f"cached -> {cache}")
    return s1, oth, gt, cands, X


def fit_lgb(X, y, params, seed, num_rounds, Xv=None, yv=None):
    p = {**params, "seed": seed}
    dtr = lgb.Dataset(X, y)
    if Xv is not None:
        dv = lgb.Dataset(Xv, yv, reference=dtr)
        return lgb.train(p, dtr, num_rounds, valid_sets=[dv],
                         callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.train(p, dtr, num_rounds)


def cmd_train(args):
    cfg = BlockConfig()
    s1, oth, gt, cands, X = build_split(args.data_dir, "train", cfg, args.work_dir, use_cache=not args.no_cache)
    X = X.copy()
    truth = gt_to_dict(gt)
    lab = {(a, b) for a, v in truth.items() for b in v}
    X["label"] = [int((a, b) in lab) for a, b in zip(X["s1_id"], X["cand_id"])]
    feats = feature_columns(X)
    log(f"{len(feats)} features, positives={X.label.sum():,} / {len(X):,}")
    os.makedirs(args.work_dir, exist_ok=True)

    # ---------------- 5-fold OOF
    gkf = GroupKFold(n_splits=args.folds)
    oof = np.zeros(len(X))
    iters = []
    for f, (tr, va) in enumerate(gkf.split(X, X.label, X.s1_id)):
        m = fit_lgb(X.iloc[tr][feats], X.label.iloc[tr], LGB_PARAMS, 0, MAX_ROUNDS,
                    X.iloc[va][feats], X.label.iloc[va])
        oof[va] = m.predict(X.iloc[va][feats], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        log(f"fold {f}: best_iter={m.best_iteration} logloss={m.best_score['valid_0']['binary_logloss']:.5f}")
    X["p"] = oof
    from sklearn.metrics import average_precision_score, roc_auc_score
    log(f"OOF pair AUC={roc_auc_score(X.label, oof):.5f}  AP={average_precision_score(X.label, oof):.5f}")

    log("tuning decision rule on OOF (macro F0.5 over ALL train S1 incl. no-candidate ones):")
    params = tune(X[["s1_id", "cand_id", "p"]], truth, log=log)
    log("best:", params)
    pred = apply_rule(X[["s1_id", "cand_id", "p"]], params)
    bd = breakdown(pred, truth)
    log("OOF breakdown:", bd)
    ceiling = breakdown({k: list(v & cands.get(k, set())) for k, v in truth.items()}, truth)
    log("blocking ceiling (perfect matcher on our candidates):", ceiling)

    # ---------------- leave-one-country-out (proxy for unseen France)
    ck = s1.set_index("entity_id")["ckey"]
    X["ck"] = X["s1_id"].map(ck)
    loco = {}
    countries = sorted(X["ck"].unique())
    if len(countries) > 1 and not args.skip_loco:
        for c in countries:
            tr, va = X[X.ck != c], X[X.ck == c]
            m = fit_lgb(tr[feats], tr.label, LGB_PARAMS, 0, int(np.mean(iters)))
            v = va[["s1_id", "cand_id"]].copy()
            v["p"] = m.predict(va[feats])
            ids = s1.loc[s1.ckey == c, "entity_id"]
            tr_ = {k: truth[k] for k in ids if k in truth}
            loco[c] = breakdown(apply_rule(v, params), tr_)
            log(f"LOCO train-without-{c} -> score on {c}: {loco[c]}")

    # ---------------- final models on all data
    n_rounds = int(np.mean(iters) * 1.1) + 1
    models = [fit_lgb(X[feats], X.label, LGB_PARAMS, sd, n_rounds) for sd in SEEDS]
    imp = pd.Series(models[0].feature_importance("gain"), index=feats).sort_values(ascending=False)
    imp.to_csv(os.path.join(args.work_dir, "feature_importance.csv"))
    log("top features:\n" + imp.head(25).to_string())
    with open(os.path.join(args.work_dir, "model.pkl"), "wb") as fh:
        pickle.dump({"models": models, "feats": feats, "params": params, "cfg": cfg.__dict__}, fh)
    X[["s1_id", "cand_id", "label", "p"]].to_csv(os.path.join(args.work_dir, "oof_pairs.tsv"), sep="\t",
                                                  index=False)
    summary = {"decision": params, "oof_breakdown": bd, "blocking_ceiling": ceiling, "loco": loco,
               "n_rounds": n_rounds, "n_features": len(feats)}
    with open(os.path.join(args.work_dir, "train_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log("saved model + summary to", args.work_dir)


def cmd_predict(args):
    with open(os.path.join(args.work_dir, "model.pkl"), "rb") as fh:
        bundle = pickle.load(fh)
    cfg = BlockConfig(**bundle["cfg"])
    s1, oth, _, cands, X = build_split(args.data_dir, "test", cfg, args.work_dir, use_cache=not args.no_cache)
    X = X.copy()
    feats = bundle["feats"]
    X["p"] = np.mean([m.predict(X[feats]) for m in bundle["models"]], axis=0)
    pred = apply_rule(X[["s1_id", "cand_id", "p"]], bundle["params"])
    s1_ids = s1["entity_id"].tolist()
    os.makedirs(args.out_dir, exist_ok=True)
    cand_lists = {k: sorted(v) for k, v in cands.items()}
    write_id_lists(os.path.join(args.out_dir, "candidate_pairs.tsv"), s1_ids, cand_lists, "candidate_entity_ids")
    write_id_lists(os.path.join(args.out_dir, "matching_results.tsv"), s1_ids, pred, "matched_entity_ids")
    X[["s1_id", "cand_id", "p"]].to_csv(os.path.join(args.work_dir, "test_pairs_scored.tsv"), sep="\t",
                                        index=False)
    n_match = sum(len(v) for v in pred.values())
    ck = s1.set_index("entity_id")["ckey"]
    by_c = pd.Series({k: len(pred.get(k, [])) > 0 for k in s1_ids}).groupby(ck).mean()
    log(f"wrote {args.out_dir}: {n_match:,} matches; share of S1 with >=1 match by country:\n{by_c.to_string()}")


def cmd_blocking(args):
    """Cheap: candidate generation + recall report only (no features / model)."""
    for split in ("train",) + (("test",) if args.with_test else ()):
        build_split(args.data_dir, split, BlockConfig(), features=False)


def cmd_features(args):
    """Build + cache pair features for train and test (reused by train / predict)."""
    for split in ("train", "test"):
        build_split(args.data_dir, split, BlockConfig(), args.work_dir, use_cache=not args.no_cache)


def cmd_synonyms(args):
    from synonyms import learn
    learn(args.data_dir, args.work_dir, n_pairs=args.syn_pairs)


def cmd_prep(args):
    from prep import prep_all
    prep_all(args.data_dir, args.work_dir, jobs=args.jobs, force=args.force)


def cmd_block(args):
    from block import block_split, recall_report
    for split in args.splits.split(","):
        s1, oth, cands = block_split(args.work_dir, split, args.k_rev, args.k_fwd, args.df_cap, args.jobs)
        if split == "train":
            recall_report(args.data_dir, args.work_dir, s1, oth, cands, args.k_rev, args.k_fwd)


def cmd_stage2(args, which=("features2", "train2", "predict2")):
    import stage2
    if "features2" in which:
        R, F = stage2.choose_RF(args.work_dir, args.R, args.F)
        for split in args.splits.split(","):
            stage2.features(args.work_dir, split, R, F, args.jobs, force=args.force)
    if "train2" in which:
        stage2.train(args.data_dir, args.work_dir, train_frac=args.train_frac, max_rounds=args.max_rounds)
    if "predict2" in which:
        stage2.predict(args.work_dir, args.out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["synonyms", "prep", "block", "stage1", "features2", "train2", "predict2",
                                    "stage2", "full",
                                    "blocking", "features", "train", "predict", "all"])
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--skip-loco", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached features in work-dir")
    ap.add_argument("--with-test", action="store_true", help="blocking: also run on test (no recall)")
    # large-scale stages (v2)
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: all cores)")
    ap.add_argument("--force", action="store_true", help="prep: rebuild even if up to date")
    ap.add_argument("--syn-pairs", type=int, default=800_000)
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--k-rev", type=int, default=10)
    ap.add_argument("--k-fwd", type=int, default=20)
    ap.add_argument("--df-cap", type=int, default=3000)
    ap.add_argument("--R", type=int, default=None, help="prune: keep r_rev <= R (default: auto)")
    ap.add_argument("--F", type=int, default=None, help="prune: keep r_fwd <= F (default: auto)")
    ap.add_argument("--train-frac", type=float, default=0.25, help="share of S1 entities used per fold model")
    ap.add_argument("--max-rounds", type=int, default=2000)
    args = ap.parse_args()
    if args.cmd in ("features2", "train2", "predict2"):
        return cmd_stage2(args, (args.cmd,))
    if args.cmd == "stage2":
        return cmd_stage2(args)
    if args.cmd == "full":  # the whole large-scale pipeline
        cmd_synonyms(args)
        cmd_prep(args)
        cmd_block(args)
        return cmd_stage2(args)
    if args.cmd == "synonyms":
        return cmd_synonyms(args)
    if args.cmd == "prep":
        return cmd_prep(args)
    if args.cmd == "block":
        return cmd_block(args)
    if args.cmd == "stage1":  # synonyms -> prep -> block in one go
        cmd_synonyms(args)
        cmd_prep(args)
        return cmd_block(args)
    if args.cmd == "blocking":
        return cmd_blocking(args)
    if args.cmd == "features":
        return cmd_features(args)
    if args.cmd in ("train", "all"):
        cmd_train(args)
    if args.cmd in ("predict", "all"):
        cmd_predict(args)


if __name__ == "__main__":
    main()
