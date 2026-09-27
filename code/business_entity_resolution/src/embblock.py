"""Data step 'embblock' (GPU): multilingual-embedding nearest neighbours as an extra candidate source.

Error analysis (work/errors_report.md): 39% of the lost macro-F0.5 comes from true pairs the token
blocker never proposed - mostly Indic-script names with truncated addresses ('मॉडर्न सॉल्यूशंस…' +
'26, MUMBAI CITY': the transliteration 'monddrn' never produces the token of 'modern') and typo'd
names with empty addresses ('Memorial Atsociatieon'). The multilingual MiniLM embedding places those
names next to their S1 name, so for every S2/S3 record we add its top-K S1 by name-embedding cosine
(within the country block) to the candidate set.

Embeddings of every unique name are cached in work/emb/<split>_U.npy (+ _codes.npy) and reused by
the 'embed' feature step.  Adds columns r_emb (rank, 255 = not proposed) and emb_score to
work/cands/<split>.parquet.  Runs only with a GPU (or ER_FAKE_EMB=1 in tests).
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from prep import load_prep

T0 = time.time()
NOPE = 255


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def emb_cache_paths(work_dir, split):
    d = os.path.join(work_dir, "emb")
    return os.path.join(d, f"{split}_codes.npy"), os.path.join(d, f"{split}_U.npy")


def get_embeddings(work_dir, split):
    """(codes for S1 rows then S2+S3 rows, float16 unique-name embeddings); computed once, cached."""
    import embed
    cp, up = emb_cache_paths(work_dir, split)
    n1 = load_prep(work_dir, split, (1,), ["name"])["name"].tolist()
    no = load_prep(work_dir, split, (2, 3), ["name"])["name"].tolist()
    if os.path.exists(cp) and os.path.exists(up):
        codes = np.load(cp)
        if len(codes) == len(n1) + len(no):
            log(f"{split}: name embeddings from cache {up}")
            return codes, np.load(up, mmap_mode="r"), len(n1)
    model, dev = embed._encoder()
    log(f"{split}: embedding {len(n1) + len(no):,} names on {dev}")
    codes, U = embed.encode_unique(model, n1 + no)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    np.save(up, U)
    np.save(cp, codes.astype(np.int32))
    return codes.astype(np.int32), U, len(n1)


def _topk(Q, B, k):
    """Row-wise top-k of Q @ B.T (both L2-normalised). GPU via torch when available."""
    try:
        import torch
        if torch.cuda.is_available():
            Bt = torch.from_numpy(np.ascontiguousarray(B, dtype=np.float16)).cuda()
            idx_out, val_out = [], []
            for st in range(0, len(Q), 2048):
                q = torch.from_numpy(np.ascontiguousarray(Q[st:st + 2048], dtype=np.float16)).cuda()
                v, i = torch.topk(q @ Bt.T, min(k, Bt.shape[0]), dim=1)
                idx_out.append(i.cpu().numpy())
                val_out.append(v.float().cpu().numpy())
            del Bt
            torch.cuda.empty_cache()
            return np.concatenate(idx_out), np.concatenate(val_out)
    except ImportError:
        pass
    B32 = np.asarray(B, dtype=np.float32)
    idx_out, val_out = [], []
    kk = min(k, len(B32))
    for st in range(0, len(Q), 4096):
        S = np.asarray(Q[st:st + 4096], dtype=np.float32) @ B32.T
        i = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        v = np.take_along_axis(S, i, 1)
        o = np.argsort(-v, axis=1)
        idx_out.append(np.take_along_axis(i, o, 1))
        val_out.append(np.take_along_axis(v, o, 1))
    return np.concatenate(idx_out), np.concatenate(val_out)


def done(work_dir, split) -> bool:
    import pyarrow.parquet as pq
    p = os.path.join(work_dir, "cands", f"{split}.parquet")
    return os.path.exists(p) and "r_emb" in pq.read_schema(p).names


def run(work_dir, split, k=5):
    codes, U, n1 = get_embeddings(work_dir, split)
    c1, co = codes[:n1], codes[n1:]
    ck1 = load_prep(work_dir, split, (1,), ["ckey"])["ckey"].to_numpy()
    cko = load_prep(work_dir, split, (2, 3), ["ckey"])["ckey"].to_numpy()
    parts = []
    for c in np.unique(ck1):
        ia, ib = np.where(ck1 == c)[0], np.where(cko == c)[0]
        if not len(ib):
            continue
        idx, val = _topk(U[co[ib]], U[c1[ia]], k)
        parts.append(pd.DataFrame({"s1": ia[idx.ravel()].astype(np.int32),
                                   "o": np.repeat(ib, idx.shape[1]).astype(np.int32),
                                   "emb_score": val.ravel().astype(np.float32),
                                   "r_emb": np.tile(np.arange(1, idx.shape[1] + 1, dtype=np.uint8), len(ib))}))
        log(f"  [{c}] {len(ib):,} records x {len(ia):,} S1: top-{k} embedding neighbours")
    E = pd.concat(parts, ignore_index=True)
    path = os.path.join(work_dir, "cands", f"{split}.parquet")
    C = pd.read_parquet(path)
    C = C.drop(columns=[x for x in ("r_emb", "emb_score") if x in C.columns])
    n_before = len(C)
    C = C.merge(E, on=["s1", "o"], how="outer")
    C["score"] = C["score"].fillna(0).astype(np.float32)
    for col in ("r_rev", "r_fwd", "r_emb"):
        C[col] = C[col].fillna(NOPE).astype(np.uint8)
    C["emb_score"] = C["emb_score"].fillna(-1).astype(np.float32)
    C.to_parquet(path + ".tmp.parquet", index=False)
    os.replace(path + ".tmp.parquet", path)
    log(f"{split}: candidates {n_before:,} -> {len(C):,} (+{len(C) - n_before:,} from embedding neighbours)")
    return C


def recall_gain(data_dir, work_dir, C):
    from block import true_pairs
    s1_ids = load_prep(work_dir, "train", (1,), ["entity_id"])["entity_id"]
    o_ids = load_prep(work_dir, "train", (2, 3), ["entity_id"])["entity_id"]
    ex, _, _ = true_pairs(data_dir, s1_ids, o_ids)
    ex = ex.dropna(subset=["o"])
    tkey = ex["s1"].astype(np.int64).to_numpy() * (1 << 32) + ex["o"].astype(np.int64).to_numpy()
    key = C["s1"].to_numpy().astype(np.int64) * (1 << 32) + C["o"].to_numpy().astype(np.int64)
    tok = (C["r_rev"].to_numpy() < NOPE) | (C["r_fwd"].to_numpy() < NOPE)
    emb = C["r_emb"].to_numpy() < NOPE
    r_tok = np.isin(tkey, key[tok]).mean()
    r_all = np.isin(tkey, key).mean()
    r_emb = np.isin(tkey, key[emb]).mean()
    log(f"train pair recall: token blocker {r_tok:.4f} | embedding neighbours alone {r_emb:.4f} | "
        f"union {r_all:.4f} (+{r_all - r_tok:.4f})")
