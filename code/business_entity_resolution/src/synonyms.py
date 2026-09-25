"""Learn token synonyms from TRAINING matched pairs (no external data).

For a sampled set of true (S1, S2/S3) pairs we look at the tokens that differ:
    A = tokens only in the S1 record, B = tokens only in the matched record
and count co-occurring replacements (a, b) when both sides differ by at most 3 tokens.
A replacement b -> a is kept when it is frequent and consistent, e.g.
    limittedd -> ltd, praaivett -> pvt, elelpii -> llp          (Devanagari names, transliterated)
    mhaaraassttr -> mh, tmilllnaattu -> tn, pshcimbngg -> wb   (state names in Indic scripts)
    bombay -> mumbai, bengaluru -> bangalore                    (city aliases)
Separate dictionaries for names and addresses; written to <work>/synonyms.json and
applied by normalize.py to every record (S1, S2, S3, train and test).

  python src/pipeline.py synonyms --data-dir dataset --work-dir work
"""
from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

import normalize as N

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _read_filtered(path, ids, chunksize=500_000):
    out = []
    for ch in pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False,
                          chunksize=chunksize):
        out.append(ch[ch["entity_id"].isin(ids)])
    return pd.concat(out, ignore_index=True)


def _name_toks(x):
    return set(N._name_tokens(x))


def _addr_toks(x):
    return set(N.norm_addr(x).split())


_LEGAL = set(N.LEGAL_CANON) | set(N.LEGAL_CANON.values()) | set(N.LEGAL_FORMS) | N._NAME_NOISE


def _ok(t):
    # legal forms / noise words are handled by normalize.core_name - never learn them as synonyms
    return not any(c.isdigit() for c in t) and t not in _LEGAL


def mine(pairs, rec, tok_fn, field, min_count, min_ratio, max_diff=3):
    co, seen_b = Counter(), Counter()
    for a_id, b_id in pairs:
        ta, tb = tok_fn(rec[a_id][field]), tok_fn(rec[b_id][field])
        A = [t for t in ta - tb if _ok(t)]
        B = [t for t in tb - ta if _ok(t) and len(t) >= 2]
        if not B or len(A) > max_diff or len(B) > max_diff:
            continue
        seen_b.update(B)
        for b in B:
            for a in A:
                co[(b, a)] += 1
    best = {}
    for (b, a), c in co.items():
        if c >= min_count and c / seen_b[b] >= min_ratio and c > best.get(b, (None, 0))[1]:
            best[b] = (a, c)
    return {b: a for b, (a, c) in best.items()}, co, seen_b


def learn(data_dir, work_dir, n_pairs=800_000, min_count=15, min_ratio=0.5, seed=0):
    rng = np.random.default_rng(seed)
    tr = lambda f: os.path.join(data_dir, "train", f)  # noqa: E731
    gt = pd.read_csv(tr("train_ground_truth.tsv"), sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE)
    lists = gt["matched_entity_ids"].str.split(",")
    sizes = lists.map(lambda x: sum(1 for y in x if y)).values
    ex = pd.DataFrame({"s1": gt["source1_entity_id"].repeat(sizes).values,
                       "m": [y for x in lists for y in x if y]})
    ex = ex.iloc[rng.choice(len(ex), size=min(n_pairs, len(ex)), replace=False)]
    log(f"mining synonyms on {len(ex):,} sampled training pairs")
    frames = [_read_filtered(tr("train_source1.tsv"), set(ex.s1))]
    oth = set(ex.m)
    frames += [_read_filtered(tr(f"train_source{i}.tsv"), oth) for i in (2, 3)]
    rec = pd.concat(frames).set_index("entity_id")[["business_name", "business_address"]].to_dict("index")
    pairs = [(a, b) for a, b in zip(ex.s1, ex.m) if a in rec and b in rec]
    log(f"rows loaded: {len(rec):,}; usable pairs {len(pairs):,}")
    N.set_synonyms()  # learn on un-mapped tokens
    name_map, nco, _ = mine(pairs, rec, _name_toks, "business_name", min_count, min_ratio)
    addr_map, aco, _ = mine(pairs, rec, _addr_toks, "business_address", min_count, min_ratio)
    # a name token that maps to itself via a chain (b->a->c) is resolved one step
    for m in (name_map, addr_map):
        for b, a in list(m.items()):
            if a in m and m[a] != b:
                m[b] = m[a]
    out = {"name": name_map, "addr": addr_map}
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, "synonyms.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=0, sort_keys=True)
    log(f"learned {len(name_map):,} name + {len(addr_map):,} address synonyms -> {path}")
    for tag, m, co in (("name", name_map, nco), ("addr", addr_map, aco)):
        top = sorted(((co[(b, a)], b, a) for b, a in m.items()), reverse=True)[:40]
        log(f"top {tag} synonyms: " + ", ".join(f"{b}->{a}({c})" for c, b, a in top))
    return out
