"""Pairwise features for (S1 record, S2/S3 candidate) pairs.

Groups:
  * name string similarity (rapidfuzz, several normalisations, phonetic, acronym, DBA)
  * TF-IDF cosines (char name, word name, char name+addr, word addr)
  * IDF-weighted token overlap + "rare token only on one side" (distinctive differences)
  * address: fuzzy scores, postal-code agreement, house/street number agreement
  * frequency of the core name (chains / common names are risky merges)
  * context: rank / gap of the pair among the S1 entity's candidates AND among
    the candidate record's S1 competitors (models the ~1-to-1 structure)

No feature encodes the country label itself, so an unseen country (France)
is scored with the same function the model learned on US + India.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from normalize import acronym, core_name, norm_name, numbers, split_dba


def _cpdist(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def _row_cos(A, B, i, j, chunk=200_000):
    out = np.empty(len(i), dtype=np.float32)
    for st in range(0, len(i), chunk):
        a, b = A[i[st:st + chunk]], B[j[st:st + chunk]]
        out[st:st + chunk] = np.asarray(a.multiply(b).sum(1)).ravel()
    return out


def _tok_feats(ta_list, tb_list, idf, default_idf, prefix):
    n = len(ta_list)
    jac = np.zeros(n, np.float32)
    wjac = np.zeros(n, np.float32)
    max_shared = np.zeros(n, np.float32)
    only_a = np.zeros(n, np.float32)
    only_b = np.zeros(n, np.float32)
    max_only = np.zeros(n, np.float32)
    for k, (sa, sb) in enumerate(zip(ta_list, tb_list)):
        if not sa and not sb:
            continue
        inter, uni = sa & sb, sa | sb
        jac[k] = len(inter) / len(uni)
        w = {t: idf.get(t, default_idf) for t in uni}
        wu = sum(w.values())
        wjac[k] = sum(w[t] for t in inter) / wu if wu else 0
        max_shared[k] = max((w[t] for t in inter), default=0)
        oa = [w[t] for t in sa - sb]
        ob = [w[t] for t in sb - sa]
        only_a[k] = sum(oa)
        only_b[k] = sum(ob)
        max_only[k] = max(oa + ob, default=0)
    return {f"{prefix}_jac": jac, f"{prefix}_wjac": wjac, f"{prefix}_max_shared_idf": max_shared,
            f"{prefix}_only_s1_idf": only_a, f"{prefix}_only_c_idf": only_b, f"{prefix}_max_only_idf": max_only}


def _dba_best(names_a, names_b):
    out = np.full(len(names_a), np.nan, dtype=np.float32)
    for k, (a, b) in enumerate(zip(names_a, names_b)):
        va, vb = split_dba(a), split_dba(b)
        if len(va) == 1 and len(vb) == 1:
            continue
        out[k] = max(fuzz.token_set_ratio(core_name(x), core_name(y)) for x in va for y in vb)
    return out


def _num_feats(sa_list, sb_list, prefix):
    n = len(sa_list)
    both = np.zeros(n, np.float32)
    jac = np.full(n, np.nan, np.float32)
    conflict = np.zeros(n, np.float32)
    for k, (a, b) in enumerate(zip(sa_list, sb_list)):
        if a and b:
            both[k] = 1
            jac[k] = len(a & b) / len(a | b)
            conflict[k] = float(not (a & b))
    return {f"{prefix}_num_both": both, f"{prefix}_num_jac": jac, f"{prefix}_num_conflict": conflict}


def _context(df: pd.DataFrame, cols):
    """Rank/gap of each pair within its S1 group and within its candidate group."""
    out = {}
    for c in cols:
        g1 = df.groupby("s1_id")[c]
        out[f"{c}_rank_s1"] = g1.rank(ascending=False, method="min").astype(np.float32).values
        out[f"{c}_gap_s1"] = (g1.transform("max") - df[c]).astype(np.float32).values
        g2 = df.groupby("cand_id")[c]
        out[f"{c}_rank_c"] = g2.rank(ascending=False, method="min").astype(np.float32).values
        out[f"{c}_gap_c"] = (g2.transform("max") - df[c]).astype(np.float32).values
        # second best in S1 group: is there a close competitor?
        srt = df[["s1_id", c]].sort_values(c, ascending=False, kind="stable")
        sec = srt[srt.groupby("s1_id").cumcount() == 1].set_index("s1_id")[c]
        out[f"{c}_second_s1"] = df["s1_id"].map(sec).fillna(0).astype(np.float32).values
    out["n_cands_s1"] = df.groupby("s1_id")["cand_id"].transform("size").astype(np.float32).values
    out["n_cands_c"] = df.groupby("cand_id")["s1_id"].transform("size").astype(np.float32).values
    return out


def build_features(pairs: pd.DataFrame, s1: pd.DataFrame, oth: pd.DataFrame, space) -> pd.DataFrame:
    """pairs: DataFrame[s1_id, cand_id]; s1/oth already passed through blocking.prepare."""
    pairs = pairs.reset_index(drop=True)
    i = s1.reset_index().set_index("entity_id").loc[pairs["s1_id"], "index"].values
    j = oth.reset_index().set_index("entity_id").loc[pairs["cand_id"], "index"].values
    A, B = s1.iloc[i].reset_index(drop=True), oth.iloc[j].reset_index(drop=True)
    F = {}

    # ---------------- name
    na, nb = A["n_name"].tolist(), B["n_name"].tolist()
    ca, cb = A["c_name"].tolist(), B["c_name"].tolist()
    F["name_ratio"] = _cpdist(na, nb, fuzz.ratio)
    F["name_partial"] = _cpdist(na, nb, fuzz.partial_ratio)
    F["name_tsort"] = _cpdist(na, nb, fuzz.token_sort_ratio)
    F["name_tset"] = _cpdist(na, nb, fuzz.token_set_ratio)
    F["name_wratio"] = _cpdist(na, nb, fuzz.WRatio)
    F["name_jw"] = _cpdist(na, nb, JaroWinkler.normalized_similarity)
    F["core_ratio"] = _cpdist(ca, cb, fuzz.ratio)
    F["core_tset"] = _cpdist(ca, cb, fuzz.token_set_ratio)
    F["core_tsort"] = _cpdist(ca, cb, fuzz.token_sort_ratio)
    F["core_partial"] = _cpdist(ca, cb, fuzz.partial_ratio)
    F["core_lev"] = _cpdist(ca, cb, Levenshtein.normalized_similarity)
    F["core_jw"] = _cpdist(ca, cb, JaroWinkler.normalized_similarity)
    F["phon_ratio"] = _cpdist(A["p_name"].tolist(), B["p_name"].tolist(), fuzz.ratio)
    F["phon_tset"] = _cpdist(A["p_name"].tolist(), B["p_name"].tolist(), fuzz.token_set_ratio)
    F["core_equal"] = (A["c_name"].values == B["c_name"].values).astype(np.float32)
    F["phon_equal"] = (A["p_name"].values == B["p_name"].values).astype(np.float32)
    ca_ns = [x.replace(" ", "") for x in ca]
    cb_ns = [x.replace(" ", "") for x in cb]
    F["core_nospace_equal"] = np.array([x == y for x, y in zip(ca_ns, cb_ns)], np.float32)
    acr_a = [acronym(x) for x in A["business_name"]]
    acr_b = [acronym(x) for x in B["business_name"]]
    F["acronym_match"] = np.array([(len(x) >= 2 and x == ny) or (len(y) >= 2 and y == nx)
                                   for x, y, nx, ny in zip(acr_a, acr_b, ca_ns, cb_ns)], np.float32)
    F["first_tok_equal"] = np.array([(x.split()[:1] == y.split()[:1]) for x, y in zip(ca, cb)], np.float32)
    F["core_len_a"] = np.array([len(x) for x in ca], np.float32)
    F["core_len_b"] = np.array([len(x) for x in cb], np.float32)
    F["core_ntok_diff"] = np.array([abs(len(x.split()) - len(y.split())) for x, y in zip(ca, cb)], np.float32)
    F["dba_best_tset"] = _dba_best(A["business_name"].tolist(), B["business_name"].tolist())
    # legal-form agreement (Pvt Ltd vs LLC ... ) - weak signal
    lf = lambda s: set(norm_name(s).split()) - set(core_name(s).split())  # noqa: E731
    F["legal_conflict"] = np.array([float(bool(x) and bool(y) and not (x & y)) for x, y in
                                    zip(map(lf, A["business_name"]), map(lf, B["business_name"]))], np.float32)

    # ---------------- tf-idf cosines
    M1, MO = space.mats["s1"], space.mats["oth"]
    for key in ("name", "nword", "full", "addr"):
        F[f"cos_{key}"] = _row_cos(M1[key], MO[key], i, j)

    # ---------------- token idf overlap
    dn = float(np.max(list(space.name_idf.values()))) if space.name_idf else 10.0
    da = float(np.max(list(space.addr_idf.values()))) if space.addr_idf else 10.0
    F.update(_tok_feats([set(x.split()) for x in na], [set(x.split()) for x in nb], space.name_idf, dn, "ntok"))
    aa, ab = A["n_addr"].tolist(), B["n_addr"].tolist()
    F.update(_tok_feats([set(x.split()) for x in aa], [set(x.split()) for x in ab], space.addr_idf, da, "atok"))

    # ---------------- address
    F["addr_ratio"] = _cpdist(aa, ab, fuzz.ratio)
    F["addr_partial"] = _cpdist(aa, ab, fuzz.partial_ratio)
    F["addr_tset"] = _cpdist(aa, ab, fuzz.token_set_ratio)
    F["addr_tsort"] = _cpdist(aa, ab, fuzz.token_sort_ratio)
    F["addr_empty_a"] = np.array([not x for x in aa], np.float32)
    F["addr_empty_b"] = np.array([not x for x in ab], np.float32)
    F["addr_len_ratio"] = np.array([min(len(x), len(y)) / max(len(x), len(y), 1) for x, y in zip(aa, ab)],
                                   np.float32)
    pa, pb = A["pc"].values, B["pc"].values
    both_pc = (pa != "") & (pb != "")
    F["pc_both"] = both_pc.astype(np.float32)
    F["pc_equal"] = np.where(both_pc, (pa == pb).astype(np.float32), np.nan)
    F["pc_prefix3"] = np.where(both_pc, np.array([x[:3] == y[:3] for x, y in zip(pa, pb)], np.float32), np.nan)
    F.update(_num_feats([numbers(x) - {p} for x, p in zip(A["business_address"], pa)],
                        [numbers(x) - {p} for x, p in zip(B["business_address"], pb)], "addr"))
    F.update(_num_feats([numbers(x) for x in A["business_name"]], [numbers(x) for x in B["business_name"]], "name"))
    F["country_equal"] = (A["ckey"].values == B["ckey"].values).astype(np.float32)
    F["is_s3"] = (B["src"].values == "S3").astype(np.float32)

    # ---------------- frequency of the (core) name: chains & generic names
    cn_o = oth.groupby(["ckey", "c_name"]).size()
    cn_1 = s1.groupby(["ckey", "c_name"]).size()
    F["cname_freq_oth"] = np.log1p(cn_o.reindex(list(zip(A["ckey"], A["c_name"]))).fillna(0).values).astype(
        np.float32)
    F["cname_freq_s1"] = np.log1p(cn_1.reindex(list(zip(A["ckey"], A["c_name"]))).fillna(0).values).astype(
        np.float32)

    # combined quick score used for context features
    F["quick"] = (0.45 * F["cos_name"] + 0.35 * F["cos_full"] + 0.2 * F["cos_addr"]).astype(np.float32)

    X = pd.DataFrame(F)
    X.insert(0, "cand_id", pairs["cand_id"].values)
    X.insert(0, "s1_id", pairs["s1_id"].values)
    ctx = _context(X, ["quick", "name_tset", "cos_full", "addr_tset", "core_lev"])
    for k, v in ctx.items():
        X[k] = v
    return X


def feature_columns(X: pd.DataFrame):
    return [c for c in X.columns if c not in ("s1_id", "cand_id", "label")]
