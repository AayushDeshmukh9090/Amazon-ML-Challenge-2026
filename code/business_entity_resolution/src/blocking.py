"""Candidate generation (blocking).

Union of several complementary, high-recall blockers, all run *within* a
country block (country treated as an open set of string labels):

  A  char 2-4gram TF-IDF on the normalised name            top-k  (S1 -> S2/S3)
  B  char 2-4gram TF-IDF on core-name + address            top-k
  C  word TF-IDF on the normalised address                 top-k  (catches DBA / renamed)
  D  reverse name neighbours: S2/S3 record -> its top-k S1 (catches crowded S1 rows)
  E  exact phonetic core-name key                          (transliteration variants)
  F  postal code + first core-name token                   (exact key)

The union is what the matching model scores, and it is what we write to
candidate_pairs.tsv.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from normalize import core_name, norm_addr, norm_name, phonetic_key, postal_code


@dataclass
class BlockConfig:
    k_name: int = 25
    k_full: int = 25
    k_addr: int = 10
    k_reverse: int = 5
    max_key_block: int = 30        # skip exact-key blocks bigger than this (stop-word keys)
    block_by_country: bool = True
    min_sim: float = 0.05          # drop neighbours with cosine below this


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalised columns used by blocking + features (idempotent)."""
    if "n_name" in df.columns:
        return df
    df = df.copy()
    df["n_name"] = df["business_name"].map(norm_name)
    df["c_name"] = df["business_name"].map(core_name)
    df["n_addr"] = df["business_address"].map(norm_addr)
    df["p_name"] = df["c_name"].map(phonetic_key)
    df["pc"] = df["business_address"].map(postal_code)
    df["ckey"] = df["country"].fillna("").str.strip().str.lower()
    df["full"] = df["c_name"] + " | " + df["n_addr"]
    df["src"] = df["entity_id"].str[:2]
    return df.reset_index(drop=True)


class Space:
    """TF-IDF spaces fitted on all records of one split (S1+S2+S3, no labels)."""

    def __init__(self, s1: pd.DataFrame, oth: pd.DataFrame):
        allx = pd.concat([s1, oth], ignore_index=True)
        self.v_name = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True,
                                      dtype=np.float32)
        self.v_full = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True,
                                      dtype=np.float32, max_features=2_000_000)
        self.v_addr = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=1, sublinear_tf=True,
                                      dtype=np.float32)
        self.v_nword = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=1, sublinear_tf=True,
                                       dtype=np.float32)
        self.v_name.fit(allx["n_name"])
        self.v_full.fit(allx["full"])
        self.v_addr.fit(allx["n_addr"].replace("", "_empty_"))
        self.v_nword.fit(allx["n_name"])
        self.mats = {}
        for tag, df in (("s1", s1), ("oth", oth)):
            self.mats[tag] = {
                "name": self.v_name.transform(df["n_name"]),
                "full": self.v_full.transform(df["full"]),
                "addr": self.v_addr.transform(df["n_addr"]),
                "nword": self.v_nword.transform(df["n_name"]),
            }
        # word idf lookup for "rare shared token" features
        self.name_idf = dict(zip(self.v_nword.get_feature_names_out(), self.v_nword.idf_))
        self.addr_idf = dict(zip(self.v_addr.get_feature_names_out(), self.v_addr.idf_))


def _topk(A, B, k, min_sim, row_ids, col_ids, out):
    """For each row of A, top-k columns of B by cosine (dense chunked)."""
    if A.shape[0] == 0 or B.shape[0] == 0:
        return
    k = min(k, B.shape[0])
    chunk = max(1, int(3e7 // max(B.shape[0], 1)))
    BT = B.T.tocsr()
    for st in range(0, A.shape[0], chunk):
        S = (A[st:st + chunk] @ BT).toarray()
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        for r in range(S.shape[0]):
            for c in idx[r]:
                if S[r, c] >= min_sim:
                    out[row_ids[st + r]].add(col_ids[c])


def _groups(s1, oth, by_country):
    if not by_country:
        yield np.arange(len(s1)), np.arange(len(oth))
        return
    og = oth.groupby("ckey").indices
    for c, ia in s1.groupby("ckey").indices.items():
        ib = og.get(c)
        if ib is not None:
            yield ia, ib


def generate_candidates(s1: pd.DataFrame, oth: pd.DataFrame, space: Space, cfg: BlockConfig,
                        return_sources: bool = False):
    """Return dict s1_id -> set(candidate ids) (and optionally which blockers fired)."""
    s1_ids = s1["entity_id"].values
    o_ids = oth["entity_id"].values
    per = {name: defaultdict(set) for name in "ABCDEF"}
    M1, MO = space.mats["s1"], space.mats["oth"]

    for ia, ib in _groups(s1, oth, cfg.block_by_country):
        ra, rb = s1_ids[ia], o_ids[ib]
        _topk(M1["name"][ia], MO["name"][ib], cfg.k_name, cfg.min_sim, ra, rb, per["A"])
        _topk(M1["full"][ia], MO["full"][ib], cfg.k_full, cfg.min_sim, ra, rb, per["B"])
        _topk(M1["addr"][ia], MO["addr"][ib], cfg.k_addr, 0.2, ra, rb, per["C"])
        rev = defaultdict(set)
        _topk(MO["name"][ib], M1["name"][ia], cfg.k_reverse, cfg.min_sim, rb, ra, rev)
        for o, ss in rev.items():
            for s in ss:
                per["D"][s].add(o)

    # exact-key blockers
    def key_join(kfun, tag):
        kb = defaultdict(list)
        for eid, row in zip(o_ids, oth.itertuples(index=False)):
            k = kfun(row)
            if k:
                kb[k].append(eid)
        for eid, row in zip(s1_ids, s1.itertuples(index=False)):
            k = kfun(row)
            if k and k in kb and len(kb[k]) <= cfg.max_key_block:
                per[tag][eid].update(kb[k])

    cc = (lambda r: r.ckey) if cfg.block_by_country else (lambda r: "")
    key_join(lambda r: (cc(r), r.p_name) if r.p_name else None, "E")
    key_join(lambda r: (cc(r), r.pc, r.c_name.split()[0]) if r.pc and r.c_name else None, "F")

    cands = defaultdict(set)
    for d in per.values():
        for s, os_ in d.items():
            cands[s] |= os_
    cands = {s: cands.get(s, set()) for s in s1_ids}
    if return_sources:
        return cands, per
    return cands


def cands_to_frame(cands: dict) -> pd.DataFrame:
    rows = [(s, o) for s, os_ in cands.items() for o in os_]
    return pd.DataFrame(rows, columns=["s1_id", "cand_id"])


def tfidf_topk_probe(s1, oth, truth, k):
    """Used by EDA: pair recall of name-only char TF-IDF top-k within country."""
    s1, oth = prepare(s1), prepare(oth)
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True, dtype=np.float32)
    v.fit(pd.concat([s1["n_name"], oth["n_name"]]))
    A, B = v.transform(s1["n_name"]), v.transform(oth["n_name"])
    out = defaultdict(set)
    for ia, ib in _groups(s1, oth, True):
        _topk(A[ia], B[ib], k, 0.0, s1["entity_id"].values[ia], oth["entity_id"].values[ib], out)
    tot = sum(len(v) for v in truth.values())
    hit = sum(len(out.get(s, set()) & v) for s, v in truth.items())
    return hit / max(tot, 1)
