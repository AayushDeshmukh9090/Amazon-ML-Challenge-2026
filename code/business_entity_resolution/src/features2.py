"""Pair features at scale (tens of millions of candidate pairs), computed in a fork pool.

Per pair (S1 row i, S2/S3 row j), all country-agnostic:
  name     rapidfuzz scores on normalised / core / phonetic names, char-2-4gram cosine,
           no-space equality + containment (domain names), acronym, DBA/formerly best score,
           IDF-weighted token overlap and IDF mass of tokens present on ONE side only
           ("Inc Partners", "Group", "Industries" added to near-duplicate distractors)
  address  rapidfuzz scores, char cosine, IDF token overlap, postal code, house-number agreement
           (exact / prefix-suffix / edit distance / numeric gap), number-set Jaccard
  blocking candidate-search score and ranks in both directions
  freq     how common the core name is (chains / generic names)
  context  rank, gap to best and second best of key scores within the S1 group AND within the
           S2/S3 record's group of competing S1 entities (GT is 1-to-1 from that side, so the
           decisive question is "is there a better S1 for this record?")
"""
from __future__ import annotations

import multiprocessing as mp
import os
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import HashingVectorizer

import normalize as N

PREP_COLS = ["entity_id", "ckey", "name", "addr", "n_name", "c_name", "n_addr", "p_name", "hn", "nums", "pc"]
_G: dict = {}
_CHAR = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2 ** 20, norm="l2",
                          alternate_sign=False, dtype=np.float32)


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=1, dtype=np.float32)


def _row_cos(a_texts, b_texts):
    A, B = _CHAR.transform(a_texts), _CHAR.transform(b_texts)
    return np.asarray(A.multiply(B).sum(1)).ravel().astype(np.float32)


def idf_table(frames, col, sample=1_500_000, seed=0):
    """Token -> IDF from a sample of records (unseen tokens get the max IDF)."""
    s = pd.concat([f[col] for f in frames], ignore_index=True)
    if len(s) > sample:
        s = s.sample(sample, random_state=seed)
    df = Counter(t for x in s for t in set(x.split()))
    n = len(s)
    idf = {t: float(np.log(n / c)) for t, c in df.items()}
    return idf, float(np.log(n))


def _tok_feats(ta, tb, idf, dflt, p):
    n = len(ta)
    out = {k: np.zeros(n, np.float32) for k in
           (f"{p}_jac", f"{p}_wjac", f"{p}_max_shared", f"{p}_only_s1", f"{p}_only_c", f"{p}_max_only",
            f"{p}_n_only_s1", f"{p}_n_only_c")}
    for k, (sa, sb) in enumerate(zip(ta, tb)):
        if not sa and not sb:
            continue
        inter, uni = sa & sb, sa | sb
        w = {t: idf.get(t, dflt) for t in uni}
        wu = sum(w.values())
        oa, ob = [w[t] for t in sa - sb], [w[t] for t in sb - sa]
        out[f"{p}_jac"][k] = len(inter) / len(uni)
        out[f"{p}_wjac"][k] = sum(w[t] for t in inter) / wu if wu else 0
        out[f"{p}_max_shared"][k] = max((w[t] for t in inter), default=0)
        out[f"{p}_only_s1"][k], out[f"{p}_only_c"][k] = sum(oa), sum(ob)
        out[f"{p}_max_only"][k] = max(oa + ob, default=0)
        out[f"{p}_n_only_s1"][k], out[f"{p}_n_only_c"][k] = len(oa), len(ob)
    return out


def _hn_feats(ha, hb):
    n = len(ha)
    both = np.zeros(n, np.float32)
    eq = np.full(n, np.nan, np.float32)
    affix = np.full(n, np.nan, np.float32)
    lev = np.full(n, np.nan, np.float32)
    gap = np.full(n, np.nan, np.float32)
    for k, (a, b) in enumerate(zip(ha, hb)):
        if a and b:
            both[k] = 1
            eq[k] = float(a == b)
            affix[k] = float(a != b and (a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)))
            lev[k] = Levenshtein.distance(a, b)
            gap[k] = np.log1p(abs(int(a[:9]) - int(b[:9])))
    return {"hn_both": both, "hn_equal": eq, "hn_affix": affix, "hn_lev": lev, "hn_log_gap": gap}


def _set_feats(sa, sb, p):
    n = len(sa)
    both = np.zeros(n, np.float32)
    jac = np.full(n, np.nan, np.float32)
    conflict = np.zeros(n, np.float32)
    for k, (a, b) in enumerate(zip(sa, sb)):
        if a and b:
            both[k] = 1
            jac[k] = len(a & b) / len(a | b)
            conflict[k] = float(not (a & b))
    return {f"{p}_both": both, f"{p}_jac": jac, f"{p}_conflict": conflict}


def _dba_best(na, nb):
    out = np.full(len(na), np.nan, np.float32)
    for k, (a, b) in enumerate(zip(na, nb)):
        va, vb = N.split_dba(a), N.split_dba(b)
        if len(va) > 1 or len(vb) > 1:
            out[k] = max(fuzz.token_set_ratio(N.core_name(x), N.core_name(y)) for x in va for y in vb)
    return out


def pair_features(A: pd.DataFrame, B: pd.DataFrame, idf_n, dn, idf_a, da) -> dict:
    F = {}
    na, nb = A["n_name"].tolist(), B["n_name"].tolist()
    ca, cb = A["c_name"].tolist(), B["c_name"].tolist()
    pa, pb = A["p_name"].tolist(), B["p_name"].tolist()
    aa, ab = A["n_addr"].tolist(), B["n_addr"].tolist()
    for nm, sc in (("ratio", fuzz.ratio), ("partial", fuzz.partial_ratio), ("tsort", fuzz.token_sort_ratio),
                   ("tset", fuzz.token_set_ratio)):
        F[f"name_{nm}"] = _cp(na, nb, sc)
        F[f"core_{nm}"] = _cp(ca, cb, sc)
        F[f"addr_{nm}"] = _cp(aa, ab, sc)
    F["name_wratio"] = _cp(na, nb, fuzz.WRatio)
    F["core_jw"] = _cp(ca, cb, JaroWinkler.normalized_similarity)
    F["core_lev"] = _cp(ca, cb, Levenshtein.normalized_similarity)
    F["phon_ratio"] = _cp(pa, pb, fuzz.ratio)
    F["phon_tset"] = _cp(pa, pb, fuzz.token_set_ratio)
    F["cos_core_char"] = _row_cos(ca, cb)
    F["cos_addr_char"] = _row_cos(aa, ab)
    F["core_equal"] = (A["c_name"].values == B["c_name"].values).astype(np.float32)
    F["phon_equal"] = (A["p_name"].values == B["p_name"].values).astype(np.float32)
    nsa, nsb = [x.replace(" ", "") for x in ca], [x.replace(" ", "") for x in cb]
    F["ns_equal"] = np.array([x == y and len(x) > 0 for x, y in zip(nsa, nsb)], np.float32)
    F["ns_contain"] = np.array([len(x) >= 4 and len(y) >= 4 and (x in y or y in x) for x, y in zip(nsa, nsb)],
                               np.float32)
    acr_a = ["".join(t[0] for t in x.split()) for x in ca]
    acr_b = ["".join(t[0] for t in x.split()) for x in cb]
    F["acronym_match"] = np.array([(len(x) >= 2 and x == ny) or (len(y) >= 2 and y == nx)
                                   for x, y, nx, ny in zip(acr_a, acr_b, nsa, nsb)], np.float32)
    F["first_tok_eq"] = np.array([x.split()[:1] == y.split()[:1] for x, y in zip(ca, cb)], np.float32)
    F["core_len_a"] = np.array([len(x) for x in ca], np.float32)
    F["core_len_b"] = np.array([len(x) for x in cb], np.float32)
    F["core_ntok_diff"] = np.array([len(x.split()) - len(y.split()) for x, y in zip(ca, cb)], np.float32)
    F["name_nonascii_b"] = np.array([any(ord(c) > 127 for c in x) for x in B["name"]], np.float32)
    F["dba_best"] = _dba_best(A["name"].tolist(), B["name"].tolist())
    F.update(_tok_feats([set(x.split()) for x in ca], [set(x.split()) for x in cb], idf_n, dn, "ntok"))
    F.update(_tok_feats([set(x.split()) for x in aa], [set(x.split()) for x in ab], idf_a, da, "atok"))
    F["addr_empty_a"] = np.array([not x for x in aa], np.float32)
    F["addr_empty_b"] = np.array([not x for x in ab], np.float32)
    F["addr_len_ratio"] = np.array([min(len(x), len(y)) / max(len(x), len(y), 1) for x, y in zip(aa, ab)],
                                   np.float32)
    pca, pcb = A["pc"].values, B["pc"].values
    both = (pca != "") & (pcb != "")
    F["pc_both"] = both.astype(np.float32)
    F["pc_equal"] = np.where(both, (pca == pcb).astype(np.float32), np.nan).astype(np.float32)
    F.update(_hn_feats(A["hn"].tolist(), B["hn"].tolist()))
    F.update(_set_feats([set(x.split()) for x in A["nums"]], [set(x.split()) for x in B["nums"]], "nums"))
    F.update(_set_feats([N.numbers(x) for x in A["name"]], [N.numbers(x) for x in B["name"]], "name_nums"))
    return F


def _chunk(args):
    i, j = args
    A = _G["P1"].iloc[i].reset_index(drop=True)
    B = _G["PO"].iloc[j].reset_index(drop=True)
    return pd.DataFrame(pair_features(A, B, _G["idf_n"], _G["dn"], _G["idf_a"], _G["da"]))


CTX_COLS = ["quick", "name_tset", "core_tset", "addr_tset", "cos_addr_char", "hn_equal", "atok_wjac", "score"]


def context_features(X: pd.DataFrame) -> dict:
    out = {}
    for c in CTX_COLS:
        v = X[c].fillna(-1).astype(np.float32)
        for key, tag in (("s1", "s1"), ("o", "c")):
            g = v.groupby(X[key].values)
            mx = g.transform("max").values
            out[f"{c}_rank_{tag}"] = g.rank(ascending=False, method="min").astype(np.float32).values
            out[f"{c}_gap_{tag}"] = (mx - v.values).astype(np.float32)
            # best OTHER value in the group (second best if this is the best)
            srt = pd.DataFrame({"k": X[key].values, "v": v.values}).sort_values(["k", "v"], ascending=[True, False])
            first = ~srt["k"].duplicated()
            second = srt[~first].drop_duplicates("k").set_index("k")["v"]
            sec = pd.Series(X[key].values).map(second).fillna(-1).values.astype(np.float32)
            out[f"{c}_best_other_{tag}"] = np.where(v.values >= mx, sec, mx).astype(np.float32)
    out["n_cands_s1"] = X.groupby("s1")["o"].transform("size").astype(np.float32).values
    out["n_cands_c"] = X.groupby("o")["s1"].transform("size").astype(np.float32).values
    return out


def build(pairs: pd.DataFrame, P1: pd.DataFrame, PO: pd.DataFrame, jobs: int, chunk: int = 50_000, log=print):
    """pairs: DataFrame[s1, o, score, r_rev, r_fwd] (row indices into P1 / PO)."""
    _G.update(P1=P1, PO=PO)
    _G["idf_n"], _G["dn"] = idf_table([P1, PO], "c_name")
    _G["idf_a"], _G["da"] = idf_table([P1, PO], "n_addr")
    log(f"idf tables: name {len(_G['idf_n']):,} tokens, addr {len(_G['idf_a']):,} tokens")
    s1i, oi = pairs["s1"].values, pairs["o"].values
    tasks = [(s1i[k:k + chunk], oi[k:k + chunk]) for k in range(0, len(pairs), chunk)]
    if jobs > 1 and hasattr(os, "fork") and len(tasks) > 1:
        with mp.get_context("fork").Pool(jobs) as pool:
            parts = []
            for n, part in enumerate(pool.imap(_chunk, tasks, chunksize=1)):
                parts.append(part)
                if n % 100 == 0:
                    log(f"  features chunk {n + 1}/{len(tasks)}")
    else:
        parts = [_chunk(t) for t in tasks]
    X = pd.concat(parts, ignore_index=True)
    X.insert(0, "o", oi)
    X.insert(0, "s1", s1i)
    for c in ("score", "r_rev", "r_fwd"):
        X[c] = pairs[c].values.astype(np.float32)
    X["is_s3"] = PO["entity_id"].str.startswith("S3").values[oi].astype(np.float32)
    # frequency of the core name within the split (chains / generic names)
    h1 = pd.util.hash_array(P1["c_name"].values.astype(str))
    ho = pd.util.hash_array(PO["c_name"].values.astype(str))
    vc_o = pd.Series(ho).value_counts()
    vc_1 = pd.Series(h1).value_counts()
    X["cname_freq_oth"] = np.log1p(vc_o.reindex(h1[s1i]).fillna(0).values).astype(np.float32)
    X["cname_freq_s1"] = np.log1p(vc_1.reindex(h1[s1i]).fillna(0).values).astype(np.float32)
    X["quick"] = (0.3 * X["name_tset"] / 100 + 0.4 * X["addr_tset"] / 100 + 0.3 * X["score"]).astype(np.float32)
    log("context features")
    for k, v in context_features(X).items():
        X[k] = v
    _G.clear()
    return X


def feature_columns(X):
    return [c for c in X.columns if c not in ("s1", "o", "label", "p", "fold", "s1_id", "o_id")]
