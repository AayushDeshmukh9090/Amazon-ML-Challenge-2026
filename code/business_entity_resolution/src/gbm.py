"""Gradient-boosting backend: XGBoost on GPU when available (Apache-2.0), LightGBM on CPU otherwise (MIT).

Only model training / prediction benefits from a GPU in this pipeline; the string-heavy stages
(normalisation, fuzzy features) stay on CPU cores.
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pandas as pd


def has_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def resolve(backend: str) -> str:
    if backend == "auto":
        return "xgb" if has_gpu() else "lgb"
    return backend


LGB = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=100,
           feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, max_bin=255,
           verbose=-1, num_threads=0)
XGB = dict(objective="binary:logistic", eval_metric="logloss", eta=0.08, max_depth=0, grow_policy="lossguide",
           max_leaves=127, min_child_weight=5, subsample=0.7, colsample_bytree=0.7, reg_lambda=10.0,
           max_bin=256, tree_method="hist")


class Model:
    def __init__(self, backend, booster, feats, best_iteration):
        self.backend, self.booster, self.feats, self.best_iteration = backend, booster, feats, best_iteration

    def predict(self, X: pd.DataFrame, chunk: int = 5_000_000) -> np.ndarray:
        out = []
        for k in range(0, len(X), chunk):
            part = X.iloc[k:k + chunk][self.feats]
            if self.backend == "lgb":
                out.append(self.booster.predict(part, num_iteration=self.best_iteration))
            else:
                out.append(self.booster.inplace_predict(part, iteration_range=(0, self.best_iteration)))
        return np.concatenate(out) if out else np.zeros(0)

    def importance(self) -> pd.Series:
        if self.backend == "lgb":
            return pd.Series(self.booster.feature_importance("gain"), index=self.feats).sort_values(ascending=False)
        g = self.booster.get_score(importance_type="total_gain")
        return pd.Series({f: g.get(f, 0.0) for f in self.feats}).sort_values(ascending=False)


def fit(backend, Xtr, ytr, feats, rounds, seed=0, Xva=None, yva=None, log=print) -> Model:
    if backend == "lgb":
        import lightgbm as lgb
        dtr = lgb.Dataset(Xtr[feats], ytr, free_raw_data=True)
        kw = {}
        if Xva is not None:
            kw = dict(valid_sets=[lgb.Dataset(Xva[feats], yva, reference=dtr)],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
        m = lgb.train({**LGB, "seed": seed}, dtr, rounds, **kw)
        return Model("lgb", m, feats, m.best_iteration or rounds)
    import xgboost as xgb
    device = "cuda" if has_gpu() else "cpu"
    dtr = xgb.QuantileDMatrix(Xtr[feats], ytr, max_bin=XGB["max_bin"])
    kw = {}
    if Xva is not None:
        kw = dict(evals=[(xgb.QuantileDMatrix(Xva[feats], yva, ref=dtr), "valid")], early_stopping_rounds=100,
                  verbose_eval=200)
    log(f"xgboost on {device}: {len(ytr):,} rows x {len(feats)} features")
    m = xgb.train({**XGB, "device": device, "seed": seed}, dtr, rounds, **kw)
    best = (m.best_iteration + 1) if Xva is not None else rounds
    return Model("xgb", m, feats, best)
