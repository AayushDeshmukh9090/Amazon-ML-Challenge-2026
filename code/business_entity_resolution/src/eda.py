"""Memory-safe exploratory data analysis -> reports/eda/eda_report.md (+ plot, sample TSVs).

Run:  python src/eda.py --data-dir dataset --out-dir reports/eda

Built for the real data scale (~24M records): every source file is STREAMED in
chunks exactly once. File-level statistics (row counts, missingness, country
mix, id uniqueness) are exact; pair-level analyses use random samples:
  * --n-pairs   true (S1, match) pairs to profile               (default 20k)
  * --n-probe   S1 entities used for the blocking-recall probe   (default 3k)
  * --pool      random S2+S3 distractors in the probe pool       (default 300k)
  * --row-sample rows kept per file for text profiling           (default 50k)
The report is re-written after every section, so a crash still leaves output.
Peak RAM stays around 2-3 GB (the ground-truth file is the largest thing loaded).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import basic_clean, core_name, norm_addr, norm_name, postal_code, split_dba  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # plotting is optional
    plt = None

T0 = time.time()
TEXT = ["business_name", "business_address", "country"]


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


class Report:
    def __init__(self, path):
        self.path, self.lines = path, []

    def h(self, t, lvl=2):
        self.lines += ["", "#" * lvl + " " + t, ""]
        log(t)

    def p(self, t=""):
        self.lines.append(str(t))

    def table(self, df: pd.DataFrame, max_rows=40):
        df = df.head(max_rows)
        try:
            md = df.to_markdown(index=True)
        except Exception:  # tabulate not installed
            md = "```\n" + df.to_string() + "\n```"
        self.lines += [md, ""]

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def pct(x):
    return f"{100 * x:.2f}%"


def read_chunks(path, chunksize, usecols=None):
    return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False,
                       chunksize=chunksize, usecols=usecols)


# --------------------------------------------------------------------------- streaming pass
class FileStats:
    """Exact per-file stats + reservoir-like row sample + optional id-filtered rows."""

    def __init__(self, name, rng, row_sample, want_ids=None, pool_frac=0.0):
        self.name, self.rng = name, rng
        self.n = 0
        self.empty = Counter()
        self.len_sum = Counter()
        self.country = Counter()
        self.ids_dup = 0
        self._seen_hash = set()
        self.row_sample_n = row_sample
        self.samples = []
        self.wanted = []
        self.pool = []
        self.want_ids = want_ids
        self.pool_frac = pool_frac

    def update(self, ch: pd.DataFrame, total_rows_est: int):
        self.n += len(ch)
        for c in TEXT:
            s = ch[c]
            self.empty[c] += int((s.str.strip() == "").sum())
            self.len_sum[c] += int(s.str.len().sum())
        self.country.update(ch["country"].value_counts().to_dict())
        h = pd.util.hash_pandas_object(ch["entity_id"], index=False).values
        before = len(self._seen_hash)
        self._seen_hash.update(h.tolist())
        self.ids_dup += len(h) - (len(self._seen_hash) - before)
        frac = min(1.0, self.row_sample_n / max(total_rows_est, 1))
        self.samples.append(ch[self.rng.random(len(ch)) < frac])
        if self.want_ids is not None:
            self.wanted.append(ch[ch["entity_id"].isin(self.want_ids)])
        if self.pool_frac > 0:
            self.pool.append(ch[self.rng.random(len(ch)) < self.pool_frac])

    def done(self):
        self._seen_hash = None
        cat = lambda L: pd.concat(L, ignore_index=True) if L else pd.DataFrame(columns=["entity_id"] + TEXT)  # noqa
        self.sample_df, self.wanted_df, self.pool_df = cat(self.samples), cat(self.wanted), cat(self.pool)
        self.samples = self.wanted = self.pool = None

    def summary_row(self):
        return {"file": self.name, "rows": self.n, "dup_ids": self.ids_dup,
                **{f"empty_{c[9:] if c.startswith('business_') else c}": pct(self.empty[c] / max(self.n, 1))
                   for c in TEXT},
                **{f"avglen_{c[9:]}": round(self.len_sum[c] / max(self.n, 1), 1) for c in TEXT[:2]}}


def count_lines(path):
    with open(path, "rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 24), b"")) - 1


def stream(path, name, rng, args, want_ids=None, pool_frac=0.0):
    n_est = count_lines(path)
    fs = FileStats(name, rng, args.row_sample, want_ids, pool_frac)
    for ch in read_chunks(path, args.chunksize):
        fs.update(ch, n_est)
    fs.done()
    log(f"  streamed {name}: {fs.n:,} rows (sample {len(fs.sample_df):,}, wanted {len(fs.wanted_df):,}, "
        f"pool {len(fs.pool_df):,})")
    return fs


# --------------------------------------------------------------------------- pair features
def sim_row(a, b):
    na, nb = norm_name(a["business_name"]), norm_name(b["business_name"])
    aa, ab = norm_addr(a["business_address"]), norm_addr(b["business_address"])
    pa, pb = postal_code(a["business_address"]), postal_code(b["business_address"])
    return {
        "name_ratio": fuzz.ratio(na, nb), "name_tset": fuzz.token_set_ratio(na, nb),
        "core_equal": float(core_name(a["business_name"]) == core_name(b["business_name"])),
        "addr_ratio": fuzz.ratio(aa, ab) if aa and ab else np.nan,
        "addr_tset": fuzz.token_set_ratio(aa, ab) if aa and ab else np.nan,
        "pc_equal": np.nan if not pa or not pb else float(pa == pb),
        "country_equal": float(a["country"] == b["country"]),
    }


def char_profile(df):
    if not len(df):
        return {}
    s = df["business_name"] + " " + df["business_address"]
    pcs = df["business_address"].map(postal_code)
    lm = r"(?i)\b(?:near|nr|opp|opposite|behind|beside)\b"
    return {"non_ascii": pct(s.map(lambda x: any(ord(ch) > 127 for ch in x)).mean()),
            "digits_in_name": pct(df["business_name"].str.contains(r"\d").mean()),
            "amp_in_name": pct(df["business_name"].str.contains("&", regex=False).mean()),
            "dba_marker": pct(df["business_name"].map(lambda x: len(split_dba(x)) > 1).mean()),
            "all_caps_name": pct(df["business_name"].map(lambda x: x.isupper()).mean()),
            "postal_code": pct((pcs != "").mean()),
            "landmark_addr": pct(df["business_address"].str.contains(lm).mean())}


def top_tokens(r, df, col, k=30):
    rows = {}
    for c, g in df.groupby("country"):
        cnt = Counter(t for x in g[col] for t in basic_clean(x).split())
        rows[c] = [f"{t}({n})" for t, n in cnt.most_common(k)]
    r.table(pd.DataFrame({k_: pd.Series(v) for k_, v in rows.items()}), max_rows=k)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="reports/eda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-pairs", type=int, default=20000)
    ap.add_argument("--n-probe", type=int, default=3000)
    ap.add_argument("--pool", type=int, default=300000)
    ap.add_argument("--row-sample", type=int, default=50000)
    ap.add_argument("--chunksize", type=int, default=500000)
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    r = Report(os.path.join(args.out_dir, "eda_report.md"))
    hints = []
    tr = lambda f: os.path.join(args.data_dir, "train", f)  # noqa: E731
    te = lambda f: os.path.join(args.data_dir, "test", f)  # noqa: E731
    r.h("Business Entity Resolution - EDA report", 1)
    r.p(f"Pair-level sections use samples: n_pairs={args.n_pairs:,}, n_probe={args.n_probe:,}, "
        f"pool={args.pool:,}, row_sample/file={args.row_sample:,}. File-level stats are exact.")

    # ---------------------------------------------------------------- ground truth (ids only)
    log("loading ground truth")
    gt = pd.read_csv(tr("train_ground_truth.tsv"), sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE)
    lists = gt["matched_entity_ids"].str.split(",")
    sizes = lists.map(lambda x: sum(1 for y in x if y)).values
    ex = pd.DataFrame({"s1": gt["source1_entity_id"].repeat(sizes).values,
                       "m": [y for x in lists for y in x if y]})
    log(f"GT: {len(gt):,} S1 rows, {len(ex):,} matched pairs")

    # choose samples up front so each big file is streamed once
    pair_idx = rng.choice(len(ex), size=min(args.n_pairs, len(ex)), replace=False)
    pairs = ex.iloc[pair_idx].reset_index(drop=True)
    probe_s1 = set(rng.choice(gt["source1_entity_id"].values, size=min(args.n_probe, len(gt)), replace=False))
    probe_pairs = ex[ex.s1.isin(probe_s1)]
    want_s1 = set(pairs.s1) | probe_s1
    want_oth = set(pairs.m) | set(probe_pairs.m)

    # ---------------------------------------------------------------- stream train files
    r.h("1. Files, sizes, missingness (exact)")
    log("streaming train files")
    n_oth_est = count_lines(tr("train_source2.tsv")) + count_lines(tr("train_source3.tsv"))
    pool_frac = min(1.0, args.pool / max(n_oth_est, 1))
    f1 = stream(tr("train_source1.tsv"), "train S1", rng, args, want_s1)
    f2 = stream(tr("train_source2.tsv"), "train S2", rng, args, want_oth, pool_frac)
    f3 = stream(tr("train_source3.tsv"), "train S3", rng, args, want_oth, pool_frac)
    files = [f1, f2, f3]
    if not args.skip_test:
        log("streaming test files")
        files += [stream(te(f"test_source{i}.tsv"), f"test S{i}", rng, args) for i in (1, 2, 3)]
    r.table(pd.DataFrame([f.summary_row() for f in files]).set_index("file"))
    n1, n2, n3 = f1.n, f2.n, f3.n
    r.p(f"ground truth rows: {len(gt):,} (train S1 rows: {n1:,}); matched pairs: {len(ex):,}")
    r.p(f"(S2+S3)/S1: train {(n2 + n3) / n1:.2f}"
        + (f", test {(files[4].n + files[5].n) / files[3].n:.2f}" if not args.skip_test else ""))
    r.p(f"Brute-force pair space train: {n1 * (n2 + n3):.3e}  -> blocking is mandatory")
    r.save()

    # ---------------------------------------------------------------- 2 country
    r.h("2. Country distribution per file (exact)")
    cc = pd.DataFrame({f.name: pd.Series(f.country) for f in files}).fillna(0).astype(int)
    r.table(cc)
    tr_c = set(f1.country) | set(f2.country) | set(f3.country)
    te_c = set().union(*[set(f.country) for f in files[3:]]) if not args.skip_test else set()
    unseen = sorted(te_c - tr_c)
    r.p(f"Countries in test but not train: {unseen}")
    hints.append(f"Unseen test countries {unseen}; test share by country: "
                 + (", ".join(f"{c}={pct(v / files[3].n)}" for c, v in files[3].country.items())
                    if not args.skip_test else "n/a"))
    for c in sorted(cc.index):
        blk = cc.loc[c]
        if not args.skip_test:
            r.p(f"- block '{c}': train {blk['train S1']:,} S1 x {blk['train S2'] + blk['train S3']:,} other; "
                f"test {blk['test S1']:,} x {blk['test S2'] + blk['test S3']:,}")
    r.save()

    # ---------------------------------------------------------------- 3 match structure
    r.h("3. Match structure (ground truth, exact)")
    sz = pd.Series(sizes)
    single = float((sz == 0).mean())
    r.p(f"singleton S1 entities: {pct(single)}  -> an all-empty submission scores {single:.4f}")
    hints.append(f"Singleton rate {pct(single)} = score of predicting nothing (baseline to beat).")
    r.table(sz.clip(upper=10).value_counts().sort_index().rename("n_S1 (10 = 10+)").to_frame())
    is2 = ex["m"].str.startswith("S2-")
    per_s1 = ex.assign(s2=is2, s3=~is2).groupby("s1")[["s2", "s3"]].sum()
    per_s1 = per_s1.reindex(gt["source1_entity_id"]).fillna(0).astype(int)
    r.table(pd.crosstab(per_s1.s2.clip(upper=5), per_s1.s3.clip(upper=5),
                        rownames=["#S2 (5=5+)"], colnames=["#S3 (5=5+)"]))
    r.p(f"mean matches/S1 {sz.mean():.3f}; given >=1: {sz[sz > 0].mean():.3f}; max {sz.max()}; "
        f"p99 {np.percentile(sz, 99):.0f}")
    vc = ex["m"].value_counts()
    multi = int((vc > 1).sum())
    r.p(f"S2/S3 ids matched to >1 S1 entity: {multi:,} of {len(vc):,} matched ids ({pct(multi / max(len(vc), 1))})")
    hints.append("GT is one-to-one from the S2/S3 side -> enforce 'best S1 per S2/S3 record'." if multi == 0 else
                 f"{multi:,} S2/S3 ids belong to several S1 ({pct(multi / len(vc))}) -> 1-to-1 only as a soft rule.")
    m2, m3 = int(is2.sum()), int((~is2).sum())
    r.p(f"share of S2 records with a match: {pct(vc.index.str.startswith('S2-').sum() / n2)}; "
        f"S3: {pct(vc.index.str.startswith('S3-').sum() / n3)}  (the rest are distractors)")
    hints.append(f"Matched share: S2 {pct(vc.index.str.startswith('S2-').sum() / n2)}, "
                 f"S3 {pct(vc.index.str.startswith('S3-').sum() / n3)}; S2 pairs {m2:,}, S3 pairs {m3:,}.")
    del vc
    r.save()

    # ---------------------------------------------------------------- lookup for sampled rows
    lk = pd.concat([f1.wanted_df, f2.wanted_df, f3.wanted_df]).drop_duplicates("entity_id").set_index("entity_id")
    rec = lk[TEXT].to_dict("index")
    pairs = pairs[pairs.s1.isin(rec) & pairs.m.isin(rec)].reset_index(drop=True)
    log(f"sampled pairs with rows found: {len(pairs):,}")

    # ---------------------------------------------------------------- 4 country safety
    r.h("4. Do matched records share country? (sample)")
    ce = float(np.mean([rec[a]["country"] == rec[b]["country"] for a, b in zip(pairs.s1, pairs.m)]))
    r.p(f"matched pairs with identical country label: {pct(ce)}")
    hints.append("Country is " + ("SAFE" if ce > 0.999 else f"NOT fully safe ({pct(ce)})") + " as a hard block key.")
    s1c = pd.Series({k: v["country"] for k, v in rec.items() if k.startswith("S1-")})
    r.p("singleton rate by country (sampled S1): ")
    gs = gt.assign(n=sizes).set_index("source1_entity_id")["n"]
    by = pd.DataFrame({"country": s1c, "n": gs.reindex(s1c.index)})
    r.table(by.groupby("country")["n"].agg(count="count", mean_matches="mean",
                                           singleton_rate=lambda x: (x == 0).mean()))
    r.save()

    # ---------------------------------------------------------------- 5 similarity
    r.h("5. Similarity: true matches vs random same-country non-matches (sample)")
    log("computing pair similarities")
    pos = pd.DataFrame([sim_row(rec[a], rec[b]) for a, b in zip(pairs.s1, pairs.m)])
    pos["src"] = pairs.m.str[:2].values
    pos["country"] = [rec[a]["country"] for a in pairs.s1]
    s1s = f1.sample_df
    oths = pd.concat([f2.sample_df, f3.sample_df], ignore_index=True)
    neg = []
    by_c = {c: g for c, g in oths.groupby("country")}
    for _, a in s1s.sample(min(len(s1s), len(pos)), random_state=args.seed).iterrows():
        g = by_c.get(a["country"])
        if g is not None and len(g):
            neg.append(sim_row(a, g.iloc[rng.integers(len(g))]))
    neg = pd.DataFrame(neg)
    cols = ["name_ratio", "name_tset", "core_equal", "addr_ratio", "addr_tset", "pc_equal"]
    r.table(pd.concat({"match": pos[cols].describe().T[["mean", "25%", "50%", "75%"]],
                       "random": neg[cols].describe().T[["mean", "25%", "50%", "75%"]]}, axis=1))
    r.p("By source and country (match means):")
    r.table(pos.groupby(["country", "src"])[cols].mean().round(3))
    r.p(f"- true matches with name_ratio < 60: {pct((pos.name_ratio < 60).mean())}")
    r.p(f"- exact normalised-name equality: {pct((pos.name_ratio == 100).mean())}; core-name equality: "
        f"{pct(pos.core_equal.mean())}")
    r.p(f"- postal code in both: {pct(pos.pc_equal.notna().mean())}; equal when both present: "
        f"{pct(pos.pc_equal.mean())}")
    hints.append(f"Postal code equal in {pct(pos.pc_equal.mean())} of matches when both present "
                 f"(both present {pct(pos.pc_equal.notna().mean())}).")
    hints.append(f"{pct((pos.name_ratio < 60).mean())} of true matches have name_ratio<60 (hard recall cases).")
    if plt is not None:
        fig, axes = plt.subplots(1, 4, figsize=(18, 3.5))
        for ax, c in zip(axes, ["name_ratio", "name_tset", "addr_ratio", "addr_tset"]):
            ax.hist(neg[c].dropna(), bins=50, alpha=.6, label="random", density=True)
            ax.hist(pos[c].dropna(), bins=50, alpha=.6, label="match", density=True)
            ax.set_title(c)
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "sim_match_vs_random.png"), dpi=90)
        plt.close(fig)
        r.p("![sim](sim_match_vs_random.png)")
    r.save()

    # ---------------------------------------------------------------- 6 examples
    r.h("6. Raw examples of true matches (sample)")
    exm = pd.DataFrame([{"country": rec[a]["country"], "s1_name": rec[a]["business_name"],
                         "m_name": rec[b]["business_name"], "s1_addr": rec[a]["business_address"],
                         "m_addr": rec[b]["business_address"], "src": b[:2], **sim_row(rec[a], rec[b])}
                        for a, b in zip(pairs.s1[:3000], pairs.m[:3000])])
    exm.to_csv(os.path.join(args.out_dir, "match_examples.tsv"), sep="\t", index=False)
    for c, g in exm.groupby("country"):
        r.p(f"\n**{c}**")
        r.table(g.head(10)[["src", "s1_name", "m_name", "s1_addr", "m_addr"]])
    r.p("Lowest name-similarity true matches:")
    r.table(exm.sort_values("name_tset").head(20)[["country", "s1_name", "m_name", "s1_addr", "m_addr",
                                                   "name_tset", "addr_tset"]])
    r.save()

    # ---------------------------------------------------------------- 7 blocking probe + hard negatives
    r.h("7. Blocking-recall probe + hard negatives (sampled S1 vs sampled pool)")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        pool = pd.concat([f2.pool_df, f3.pool_df], ignore_index=True)
        truth_probe = probe_pairs.groupby("s1")["m"].apply(set).to_dict()
        pr_s1 = f1.wanted_df[f1.wanted_df.entity_id.isin(probe_s1)].reset_index(drop=True)
        tm = pd.concat([f2.wanted_df, f3.wanted_df])
        tm = tm[tm.entity_id.isin(set(probe_pairs.m))]
        pool = pd.concat([pool, tm]).drop_duplicates("entity_id").reset_index(drop=True)
        r.p(f"probe: {len(pr_s1):,} S1 vs pool {len(pool):,} (= {pct(len(pool) / (n2 + n3))} of real S2+S3). "
            "Recall on the full pool will be LOWER: more distractors compete for the top-k.")
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True, dtype=np.float32)
        nn1, nnp = pr_s1["business_name"].map(norm_name), pool["business_name"].map(norm_name)
        v.fit(pd.concat([nn1, nnp]))
        A, B = v.transform(nn1), v.transform(nnp)
        ks = [1, 5, 10, 20, 50]
        hits = {k: 0 for k in ks}
        tot = sum(len(truth_probe.get(s, ())) for s in pr_s1.entity_id)
        hard = []
        pool_ids = pool.entity_id.values
        for c in pr_s1.country.unique():
            ia = np.where(pr_s1.country.values == c)[0]
            ib = np.where(pool.country.values == c)[0]
            if not len(ib):
                continue
            Bt = B[ib].T.tocsr()
            for st in range(0, len(ia), 200):
                rows = ia[st:st + 200]
                S = (A[rows] @ Bt).toarray()
                kk = min(max(ks), S.shape[1])
                part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]           # top-kk unordered
                order = np.argsort(-np.take_along_axis(S, part, 1), axis=1)  # then sort only those
                top = np.take_along_axis(part, order, 1)
                for ri, rrow in enumerate(rows):
                    sid = pr_s1.entity_id.values[rrow]
                    t = truth_probe.get(sid, set())
                    ranked = pool_ids[ib[top[ri]]]
                    for k in ks:
                        hits[k] += len(t & set(ranked[:k]))
                    for j, cid in enumerate(ranked[:5]):
                        if cid not in t:
                            b = pool.iloc[ib[top[ri, j]]]
                            a = pr_s1.iloc[rrow]
                            hard.append({"s1_name": a.business_name, "neg_name": b.business_name,
                                         "s1_addr": a.business_address, "neg_addr": b.business_address,
                                         "s1_has_match": bool(t), "neg_rank": j + 1,
                                         **sim_row(a, b)})
                            break
        for k in ks:
            r.p(f"- name char-TF-IDF top-{k}: pair recall {pct(hits[k] / max(tot, 1))}")
        hints.append("Name-only TF-IDF recall on sampled pool: "
                     + ", ".join(f"@{k}={pct(hits[k] / max(tot, 1))}" for k in ks))
        hard = pd.DataFrame(hard)
        if len(hard):
            hard.sort_values("name_tset", ascending=False).to_csv(
                os.path.join(args.out_dir, "hard_negatives_sample.tsv"), sep="\t", index=False)
            r.p(f"closest non-match has name_tset==100 for {pct((hard.name_tset == 100).mean())} of probe S1 "
                "(same-name different business: chains / generic names -> address must decide)")
            r.p(f"closest non-match: postal equal {pct(hard.pc_equal.mean())} when both present")
            r.table(hard.sort_values("name_tset", ascending=False)[
                        ["s1_name", "neg_name", "s1_addr", "neg_addr", "name_tset", "addr_tset"]].head(15))
    except Exception as e:  # never lose the rest of the report
        r.p(f"(probe failed: {type(e).__name__}: {e})")
    r.save()

    # ---------------------------------------------------------------- 8 text profile
    r.h("8. Character / format profile (row samples)")
    r.table(pd.DataFrame({f.name: char_profile(f.sample_df) for f in files}).T)
    r.p("Postal code extractable by country (train samples):")
    allsamp = pd.concat([f.sample_df.assign(file=f.name) for f in files])
    allsamp["has_pc"] = allsamp["business_address"].map(postal_code) != ""
    r.table(allsamp.pivot_table(index="country", columns="file", values="has_pc", aggfunc="mean").round(3))
    r.save()

    # ---------------------------------------------------------------- 9 vocab
    r.h("9. Token vocabularies (extend abbreviation tables)")
    trs = pd.concat([f.sample_df for f in files[:3]])
    r.p("NAME tokens per country (train samples):")
    top_tokens(r, trs, "business_name")
    r.p("ADDRESS tokens per country (train samples):")
    top_tokens(r, trs, "business_address")
    if unseen:
        tes = pd.concat([f.sample_df for f in files[3:]])
        tes = tes[tes.country.isin(unseen)]
        r.p(f"Unseen-country ({unseen}) tokens (test samples):")
        top_tokens(r, tes, "business_name")
        top_tokens(r, tes, "business_address")
        r.table(tes.sample(min(25, len(tes)), random_state=0)[["entity_id"] + TEXT])
    r.save()

    # ---------------------------------------------------------------- 10 systematic diffs
    r.h("10. Tokens systematically dropped/added from S1 to its matches (sample)")
    diffs = Counter()
    for a, b in zip(pairs.s1[:5000], pairs.m[:5000]):
        ta = set(basic_clean(rec[a]["business_name"]).split())
        tb = set(basic_clean(rec[b]["business_name"]).split())
        diffs.update((b[:2], "dropped", t) for t in ta - tb)
        diffs.update((b[:2], "added", t) for t in tb - ta)
    r.table(pd.DataFrame([(s, k, t, n) for (s, k, t), n in diffs.most_common(40)],
                         columns=["src", "kind", "token", "count"]))

    # ---------------------------------------------------------------- 11 leakage
    r.h("11. ID ordering sanity check")
    num = lambda x: int(re.sub(r"\D", "", x) or 0)  # noqa: E731
    r.p(f"Spearman(S1 id number, matched id number) on sample = "
        f"{pd.Series(pairs.s1.map(num)).corr(pd.Series(pairs.m.map(num)), method='spearman'):.4f} "
        "(ids are never used as features either way)")

    r.h("DECISION HINTS (auto-generated)")
    for h in hints:
        r.p(f"- {h}")
    r.save()
    log(f"report written to {r.path}")
    print("\n".join(f"- {h}" for h in hints))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
