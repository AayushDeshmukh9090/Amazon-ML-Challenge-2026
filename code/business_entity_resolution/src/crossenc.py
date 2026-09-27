"""Data step 'crossenc' (GPU): a fine-tuned transformer CROSS-ENCODER that reads both records together.

The GBM only sees hand-made similarity numbers.  A cross-encoder reads the raw pair
  "<S1 name> | <S1 address>"  [SEP]  "<S2/S3 name> | <S2/S3 address>"
with full attention between the two sides, so it can learn typo / transliteration / abbreviation /
reordering patterns directly from text - and it starts from a multilingual pretrained model
(paraphrase-multilingual-MiniLM-L12-v2, Apache-2.0, ~118M params) that already knows French,
Hindi, Tamil... words, which is what an unseen country (France) needs.  Its probability is stacked
into the GBM as extra features (it does not replace it).

Leak-free stacking:
  train: 2 folds by S1 entity (hash); the fold-f model is fine-tuned on the OTHER fold's pairs and
         scores fold f  -> every train pair gets an out-of-fold score.
  test : scored by the fold models (average), same text -> same distribution.
Only the plausible pairs are scored (top-K per S2/S3 record by the pre-filter probability p1, or
p1 >= P1_MIN); the rest get NaN (GBM handles missing), identically on train and test.

Writes work/feat/<split>_ce.parquet (row-aligned with work/feat/<split>.parquet) + _ce.json:
  ce_p, ce_rank_o, ce_best_other_o, ce_gap_o, ce_best_other_s1
Fold models + scores are checkpointed in work/ce/, so an interrupted job resumes.
ER_FAKE_CE=1 (tests): tiny randomly initialised BERT + hashed char-trigram tokenizer on CPU.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from prep import load_prep

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CE_COLS = ["ce_p", "ce_rank_o", "ce_best_other_o", "ce_gap_o", "ce_best_other_s1"]
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def ce_path(work_dir, split):
    return os.path.join(work_dir, "feat", f"{split}_ce.parquet")


def available() -> bool:
    if os.environ.get("ER_FAKE_CE") == "1":
        return True
    try:
        import torch
        import transformers  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ----------------------------------------------------------------------------- tokenizer / model
class _FakeTok:
    """Test stand-in: hashed character trigrams -> ids (vocab 4096), BERT-style pair layout."""
    vocab = 4096

    def __call__(self, a, b, max_length, **_):
        import torch
        ids, tt = [], []
        for x, y in zip(a, b):
            ta = [2 + hash(x[i:i + 3]) % (self.vocab - 3) for i in range(max(len(x) - 2, 1))]
            tb = [2 + hash(y[i:i + 3]) % (self.vocab - 3) for i in range(max(len(y) - 2, 1))]
            half = (max_length - 3) // 2
            seq = [1] + ta[:half] + [1] + tb[:half] + [1]
            ids.append(seq)
            tt.append([0] * (len(ta[:half]) + 2) + [1] * (len(tb[:half]) + 1))
        n = max(map(len, ids))
        pad = lambda v: [s + [0] * (n - len(s)) for s in v]          # noqa: E731
        return {"input_ids": torch.tensor(pad(ids)), "token_type_ids": torch.tensor(pad(tt)),
                "attention_mask": torch.tensor(pad([[1] * len(s) for s in ids]))}


def _load(model_name):
    import torch
    if os.environ.get("ER_FAKE_CE") == "1":
        from transformers import BertConfig, BertForSequenceClassification
        os.environ.setdefault("PYTHONHASHSEED", "0")
        cfg = BertConfig(vocab_size=_FakeTok.vocab, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64, num_labels=1, max_position_embeddings=256)
        torch.manual_seed(0)
        return _FakeTok(), BertForSequenceClassification(cfg), "cpu"
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    return tok, model, dev


def _texts(work_dir, split):
    P1 = load_prep(work_dir, split, (1,), ["name", "addr"])
    PO = load_prep(work_dir, split, (2, 3), ["name", "addr"])
    f = lambda P: (P["name"].fillna("").str.slice(0, 120) + " | " + P["addr"].fillna("").str.slice(0, 160)).to_numpy()  # noqa: E731
    return f(P1), f(PO)


class _Pairs:
    """Map-style dataset of (text_a, text_b, label) for a torch DataLoader."""

    def __init__(self, ta, tb, y=None):
        self.ta, self.tb, self.y = ta, tb, y

    def __len__(self):
        return len(self.ta)

    def __getitem__(self, i):
        return self.ta[i], self.tb[i], (self.y[i] if self.y is not None else 0.0)


def _collate(tok, max_len):
    import torch

    def f(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), max_length=max_len, truncation=True, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        return enc
    return f


def _loader(ds, tok, max_len, batch, shuffle, workers):
    import torch
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle,
                                       num_workers=workers, collate_fn=_collate(tok, max_len),
                                       persistent_workers=False, pin_memory=workers > 0)


def fine_tune(tok, model, dev, ta, tb, y, epochs=1, batch=256, lr=5e-5, max_len=96, workers=8, log=log):
    import torch
    model.to(dev).train()
    dl = _loader(_Pairs(ta, tb, y.astype(np.float32)), tok, max_len, batch, True, workers)
    steps = max(1, epochs * len(dl))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm + 1))))
    lossf = torch.nn.BCEWithLogitsLoss()
    use_amp = dev == "cuda"
    step, t, run = 0, time.time(), 0.0
    for ep in range(epochs):
        for enc in dl:
            lab = enc.pop("labels").to(dev)
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logit = model(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), lab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            lv = loss.item()
            run = 0.98 * run + 0.02 * lv if step > 1 else lv
            if step % 500 == 0 or step == steps:
                log(f"    step {step:,}/{steps:,} loss {run:.4f} ({step * batch / (time.time() - t):,.0f} pairs/s)")
    model.eval()
    return model


def score(tok, model, dev, ta, tb, batch=1024, max_len=96, workers=8, log=log):
    """Probabilities for the pairs, batched by text length (less padding), original order kept."""
    import torch
    model.to(dev).eval()
    order = np.argsort(np.fromiter((len(a) + len(b) for a, b in zip(ta, tb)), np.int32, len(ta)), kind="stable")
    dl = _loader(_Pairs(ta[order], tb[order]), tok, max_len, batch, False, workers)
    out = np.empty(len(ta), np.float32)
    pos, t = 0, time.time()
    with torch.inference_mode():
        for i, enc in enumerate(dl):
            enc.pop("labels")
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                logit = model(**enc).logits.squeeze(-1).float()
            n = logit.shape[0]
            out[order[pos:pos + n]] = torch.sigmoid(logit).cpu().numpy()
            pos += n
            if i % 2000 == 0 and i:
                log(f"    scored {pos:,}/{len(ta):,} ({pos / (time.time() - t):,.0f} pairs/s)")
    return out


# ----------------------------------------------------------------------------- pipeline step
def _selected(X, k_top, p1_min):
    """Pairs worth a transformer pass: top-k per record by p1, or p1 >= p1_min."""
    p1 = X["p1"].to_numpy() if "p1" in X.columns else X["score"].to_numpy()
    r = pd.Series(p1).groupby(X["o"].to_numpy()).rank(ascending=False, method="first").to_numpy()
    return (r <= k_top) | (p1 >= p1_min)


def _features(X, p):
    from stage2 import _best_other
    s1, o = X["s1"].to_numpy(), X["o"].to_numpy()
    q = np.where(np.isnan(p), -1.0, p).astype(np.float64)
    F = pd.DataFrame({"ce_p": p.astype(np.float32)})
    F["ce_rank_o"] = pd.Series(q).groupby(o).rank(ascending=False, method="first").to_numpy(np.float32)
    F["ce_best_other_o"] = _best_other(o, q).astype(np.float32)
    F["ce_gap_o"] = np.where(np.isnan(p), np.nan, q - F["ce_best_other_o"].to_numpy()).astype(np.float32)
    F["ce_best_other_s1"] = _best_other(s1, q).astype(np.float32)
    return F


def _write(work_dir, split, F):
    from stage2 import feat_path
    path = ce_path(work_dir, split)
    F.to_parquet(path + ".tmp.parquet", index=False)
    os.replace(path + ".tmp.parquet", path)
    with open(path.replace(".parquet", ".json"), "w") as fh:
        json.dump({"feat_bytes": os.path.getsize(feat_path(work_dir, split)), "rows": int(len(F))}, fh)
    log(f"{split}: cross-encoder features {F.shape} -> {path}")


def run(data_dir, work_dir, model_name=MODEL, n_train=3_000_000, epochs=1, k_top=2, p1_min=0.02,
        batch=256, lr=5e-5, max_len=96, test_models=2, workers=8):
    import torch
    from block import true_pairs
    from stage2 import _folds, feat_path
    if os.environ.get("ER_FAKE_CE") == "1":
        workers = 0                                         # tests: CPU, hashed tokenizer in-process
    ce_dir = os.path.join(work_dir, "ce")
    os.makedirs(ce_dir, exist_ok=True)
    sig = {"feat_bytes": os.path.getsize(feat_path(work_dir, "train")), "model": model_name, "n_train": n_train,
           "epochs": epochs, "k_top": k_top, "p1_min": p1_min, "lr": lr, "max_len": max_len}

    # ---- train split: labels, selection, 2 folds by S1 entity
    X = pd.read_parquet(feat_path(work_dir, "train"), columns=["s1", "o", "p1"])
    s1_ids = load_prep(work_dir, "train", (1,), ["entity_id"])["entity_id"]
    o_ids = load_prep(work_dir, "train", (2, 3), ["entity_id"])["entity_id"]
    ex, _, _ = true_pairs(data_dir, s1_ids, o_ids)
    e2 = ex.dropna(subset=["s1", "o"])
    tkey = e2["s1"].astype(np.int64).to_numpy() * (1 << 32) + e2["o"].astype(np.int64).to_numpy()
    key = X["s1"].to_numpy().astype(np.int64) * (1 << 32) + X["o"].to_numpy().astype(np.int64)
    y = np.isin(key, tkey).astype(np.int8)
    del key, tkey
    sel = _selected(X, k_top, p1_min)
    fold_all, _ = _folds(pd.Series(["ce:" + x for x in s1_ids]), 2)
    fold = fold_all[X["s1"].to_numpy()]
    log(f"train: {len(X):,} pairs, {sel.sum():,} selected for the cross-encoder "
        f"(recall of positives {y[sel].sum() / max(y.sum(), 1):.4f})")
    t1, to = _texts(work_dir, "train")
    s1, o = X["s1"].to_numpy(), X["o"].to_numpy()
    p_tr = np.full(len(X), np.nan, np.float32)
    models = []
    rng = np.random.default_rng(0)
    for f in (0, 1):
        mdir, sp = os.path.join(ce_dir, f"fold{f}"), os.path.join(ce_dir, f"oof{f}.npy")
        meta = os.path.join(ce_dir, f"fold{f}.json")
        ev = np.where(sel & (fold == f))[0]
        if os.path.exists(sp) and os.path.exists(meta) and json.load(open(meta)) == sig:
            p_tr[ev] = np.load(sp)
            models.append(mdir)
            log(f"fold {f}: resumed from checkpoint")
            continue
        tr = np.where(sel & (fold != f))[0]
        if len(tr) > n_train:
            tr = np.sort(rng.choice(tr, n_train, replace=False))
        tok, model, dev = _load(model_name)
        log(f"fold {f}: fine-tuning on {len(tr):,} pairs (pos {int(y[tr].sum()):,}) on {dev}")
        model = fine_tune(tok, model, dev, t1[s1[tr]], to[o[tr]], y[tr], epochs=epochs, batch=batch, lr=lr,
                          max_len=max_len, workers=workers)
        log(f"fold {f}: scoring {len(ev):,} held-out pairs")
        p = score(tok, model, dev, t1[s1[ev]], to[o[ev]], max_len=max_len, workers=workers)
        p_tr[ev] = p
        from sklearn.metrics import roc_auc_score
        if 0 < y[ev].sum() < len(ev):
            log(f"fold {f}: held-out AUC {roc_auc_score(y[ev], p):.5f}")
        model.save_pretrained(mdir)
        if hasattr(tok, "save_pretrained"):
            tok.save_pretrained(mdir)
        np.save(sp, p)
        json.dump(sig, open(meta, "w"))
        models.append(mdir)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    _write(work_dir, "train", _features(X, p_tr))
    del X, t1, to

    # ---- test split: scored by the fold models (averaged)
    Xt = pd.read_parquet(feat_path(work_dir, "test"), columns=["s1", "o", "p1"])
    selt = _selected(Xt, k_top, p1_min)
    t1, to = _texts(work_dir, "test")
    idx = np.where(selt)[0]
    a, b = t1[Xt["s1"].to_numpy()[idx]], to[Xt["o"].to_numpy()[idx]]
    log(f"test: {len(Xt):,} pairs, {len(idx):,} selected")
    ps = []
    for f, mdir in enumerate(models[:max(1, test_models)]):
        tok, model, dev = _load(model_name)
        model = type(model).from_pretrained(mdir)          # the fine-tuned fold model
        log(f"test: scoring with fold-{f} model")
        ps.append(score(tok, model, dev, a, b, max_len=max_len, workers=workers))
        del model
        if dev == "cuda":                                   # hand the GPU back to XGBoost (same process)
            torch.cuda.empty_cache()
    p_te = np.full(len(Xt), np.nan, np.float32)
    p_te[idx] = np.mean(ps, axis=0)
    _write(work_dir, "test", _features(Xt, p_te))


def done(work_dir) -> bool:
    from stage2 import ce_ok
    return ce_ok(work_dir, "train") and ce_ok(work_dir, "test")
