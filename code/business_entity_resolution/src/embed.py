"""Optional data step: multilingual sentence-embedding similarity of business NAMES (GPU).

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, ~118M params). It reads
Devanagari / Tamil / Bengali / Latin text natively, so 'इनोवेटिव इंफ्रास्ट्रक्चर प्राइवेट लिमिटेड' and
'Innovative Infrastructure Private Limited' land close together without transliteration.
No external data or lookup: a pretrained open model applied to the provided names only.

Writes work/feat/<split>_emb.parquet, row-aligned with work/feat/<split>.parquet:
  emb_cos                cosine(name S1, name S2/S3)
  emb_cos_rank_c         rank of this S1 among the record's candidates by emb_cos
  emb_cos_best_other_c   best emb_cos of the record's OTHER candidate S1 entities
  emb_cos_gap_c          emb_cos - emb_cos_best_other_c
  emb_cos_best_other_s1  best emb_cos among the S1 entity's other candidates
Runs only when a GPU is present (CPU would take hours on ~24M names); the model part uses these
columns only if they exist for BOTH train and test.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from prep import load_prep

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMB_COLS = ["emb_cos", "emb_cos_rank_c", "emb_cos_best_other_c", "emb_cos_gap_c", "emb_cos_best_other_s1"]
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def emb_path(work_dir, split):
    return os.path.join(work_dir, "feat", f"{split}_emb.parquet")


class _FakeEncoder:
    """Deterministic stand-in (tests only, ER_FAKE_EMB=1): hashed char-trigram random projection."""

    def encode(self, texts, **_):
        from sklearn.feature_extraction.text import HashingVectorizer
        hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2 ** 12, norm="l2")
        R = np.random.default_rng(0).standard_normal((2 ** 12, 64)).astype(np.float32)
        E = hv.transform(texts) @ R
        return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)


def _encoder():
    if os.environ.get("ER_FAKE_EMB") == "1":
        return _FakeEncoder(), "cpu"
    import sentence_transformers
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"torch {torch.__version__} (CUDA {torch.version.cuda}, available={torch.cuda.is_available()}), "
        f"sentence-transformers {sentence_transformers.__version__}")
    m = SentenceTransformer(MODEL, device=dev)
    m.max_seq_length = 48
    if dev == "cuda":
        m.half()
    return m, dev


def encode_unique(model, names: list[str], batch=1024, chunk=500_000) -> tuple[np.ndarray, np.ndarray]:
    """Embed each distinct name once -> (codes per row, float16 matrix of unique embeddings)."""
    codes, uniq = pd.factorize(pd.Series(names), sort=False)
    uniq = list(uniq)
    out = None
    t = time.time()
    for k in range(0, len(uniq), chunk):
        e = model.encode(uniq[k:k + chunk], batch_size=batch, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        if out is None:
            out = np.empty((len(uniq), e.shape[1]), np.float16)
        out[k:k + len(e)] = e
        done = k + len(e)
        log(f"    embedded {done:,}/{len(uniq):,} unique names ({done / max(time.time() - t, 1e-6):,.0f}/s)")
    return codes, out


def run(work_dir, split):
    from stage2 import _best_other, feat_path   # reuse the vectorised group helpers
    model, dev = _encoder()
    name = "FAKE test encoder" if os.environ.get("ER_FAKE_EMB") == "1" else MODEL
    log(f"{split}: embedding names with {name} on {dev}")
    n1 = load_prep(work_dir, split, (1,), ["name"])["name"].tolist()
    no = load_prep(work_dir, split, (2, 3), ["name"])["name"].tolist()
    codes, U = encode_unique(model, n1 + no)
    c1, co = codes[:len(n1)], codes[len(n1):]
    pairs = pd.read_parquet(feat_path(work_dir, split), columns=["s1", "o"])
    s1, o = pairs["s1"].to_numpy(), pairs["o"].to_numpy()
    cos = np.empty(len(pairs), np.float32)
    for k in range(0, len(pairs), 2_000_000):
        a = U[c1[s1[k:k + 2_000_000]]].astype(np.float32)
        b = U[co[o[k:k + 2_000_000]]].astype(np.float32)
        cos[k:k + len(a)] = (a * b).sum(1)
    E = pd.DataFrame({"emb_cos": cos})
    E["emb_cos_rank_c"] = pd.Series(cos).groupby(o).rank(ascending=False, method="first").to_numpy(np.float32)
    E["emb_cos_best_other_c"] = _best_other(o, cos.astype(np.float64)).astype(np.float32)
    E["emb_cos_gap_c"] = (cos - E["emb_cos_best_other_c"].to_numpy()).astype(np.float32)
    E["emb_cos_best_other_s1"] = _best_other(s1, cos.astype(np.float64)).astype(np.float32)
    path = emb_path(work_dir, split)
    E.to_parquet(path + ".tmp.parquet", index=False)
    os.replace(path + ".tmp.parquet", path)
    log(f"{split}: embedding features {E.shape} -> {path} "
        f"(mean cos {cos.mean():.3f}; pairs {len(pairs):,})")


def available() -> bool:
    if os.environ.get("ER_FAKE_EMB") == "1":
        return True
    try:
        import torch
        import sentence_transformers  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False
