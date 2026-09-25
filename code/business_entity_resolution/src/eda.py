"""Exploratory data analysis -> reports/eda/eda_report.md (+ PNG plots, sample TSVs).

Run:  python src/eda.py --data-dir dataset --out-dir reports/eda

Every section answers a question that changes a pipeline decision; the
"DECISION HINTS" section at the end summarises them automatically.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import gt_to_dict, load_split  # noqa: E402
from normalize import basic_clean, core_name, norm_addr, norm_name, postal_code, split_dba  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # plotting is optional
    plt = None


class Report:
    def __init__(self, path):
        self.path, self.lines = path, []

    def h(self, t, lvl=2):
        self.lines += ["", "#" * lvl + " " + t, ""]
        print("\n" + "#" * lvl + " " + t)

    def p(self, t=""):
        self.lines.append(str(t))
        print(t)

    def table(self, df: pd.DataFrame, max_rows=40):
        df = df.head(max_rows)
        try:
            md = df.to_markdown(index=True)
        except Exception:  # tabulate not installed
            md = "```\n" + df.to_string() + "\n```"
        self.lines += [md, ""]
        print(df.to_string())

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def pct(x):
    return f"{100 * x:.2f}%"


def describe_source(r: Report, name: str, df: pd.DataFrame):
    r.p(f"**{name}**: {len(df):,} rows, columns = {list(df.columns)}")
    stats = []
    for c in ["business_name", "business_address", "country"]:
        s = df[c].astype(str)
        stats.append({"col": c, "empty": pct((s.str.strip() == "").mean()),
                      "n_unique": s.nunique(), "len_mean": round(s.str.len().mean(), 1),
                      "len_p50": s.str.len().median(), "len_p99": s.str.len().quantile(.99),
                      "n_tokens_mean": round(s.str.split().str.len().mean(), 2)})
    r.table(pd.DataFrame(stats).set_index("col"))
    dup_id = df["entity_id"].duplicated().sum()
    dup_rec = df.duplicated(["business_name", "business_address", "country"]).sum()
    bad_prefix = (~df["entity_id"].str.match(r"^S[123]-")).sum()
    r.p(f"- duplicate entity_id: {dup_id}; exact duplicate (name,address,country) records: {dup_rec}; "
        f"ids with unexpected prefix: {bad_prefix}")
    r.p(f"- id format examples: {df['entity_id'].head(3).tolist()}")


def char_profile(r: Report, name: str, df: pd.DataFrame):
    s = (df["business_name"] + " " + df["business_address"]).astype(str)
    non_ascii = s.map(lambda x: any(ord(ch) > 127 for ch in x)).mean()
    has_digit_name = df["business_name"].str.contains(r"\d").mean()
    has_amp = df["business_name"].str.contains("&").mean()
    has_dba = df["business_name"].map(lambda x: len(split_dba(x)) > 1).mean()
    upper = df["business_name"].map(lambda x: x.isupper()).mean()
    r.p(f"- {name}: non-ASCII records {pct(non_ascii)}, names with digits {pct(has_digit_name)}, "
        f"'&' in name {pct(has_amp)}, DBA/aka markers {pct(has_dba)}, ALL-CAPS names {pct(upper)}")


def top_tokens(r: Report, df: pd.DataFrame, col: str, by_country=True, k=30):
    groups = df.groupby("country") if by_country else [("all", df)]
    rows = {}
    for c, g in groups:
        cnt = Counter(t for x in g[col] for t in basic_clean(x).split())
        rows[c] = [f"{t}({n})" for t, n in cnt.most_common(k)]
    r.table(pd.DataFrame(dict([(k, pd.Series(v)) for k, v in rows.items()])), max_rows=k)


def sim_row(a, b):
    na, nb = norm_name(a.business_name), norm_name(b.business_name)
    aa, ab = norm_addr(a.business_address), norm_addr(b.business_address)
    return {
        "name_ratio": fuzz.ratio(na, nb), "name_tset": fuzz.token_set_ratio(na, nb),
        "core_equal": float(core_name(a.business_name) == core_name(b.business_name)),
        "addr_ratio": fuzz.ratio(aa, ab) if aa and ab else np.nan,
        "addr_tset": fuzz.token_set_ratio(aa, ab) if aa and ab else np.nan,
        "pc_equal": (lambda p, q: np.nan if not p or not q else float(p == q))(
            postal_code(a.business_address), postal_code(b.business_address)),
        "country_equal": float(a.country == b.country),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="reports/eda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    r = Report(os.path.join(args.out_dir, "eda_report.md"))
    hints = []

    s1, s2, s3, gt = load_split(args.data_dir, "train")
    t1, t2, t3, _ = load_split(args.data_dir, "test")
    r.h("Business Entity Resolution - EDA report", 1)

    # ------------------------------------------------------------------ 1
    r.h("1. Files, sizes, missingness")
    for nm, df in [("train S1", s1), ("train S2", s2), ("train S3", s3),
                   ("test S1", t1), ("test S2", t2), ("test S3", t3)]:
        describe_source(r, nm, df)
    r.p(f"\nground truth: {len(gt):,} rows, S1 rows covered: "
        f"{gt['source1_entity_id'].isin(s1['entity_id']).mean():.4f}, "
        f"S1 ids missing from GT: {(~s1['entity_id'].isin(gt['source1_entity_id'])).sum()}")
    ratio_tr = (len(s2) + len(s3)) / len(s1)
    ratio_te = (len(t2) + len(t3)) / len(t1)
    r.p(f"(S2+S3)/S1 ratio: train {ratio_tr:.2f}, test {ratio_te:.2f}")
    r.p(f"Brute-force pair space: train {len(s1) * (len(s2) + len(s3)):,}, "
        f"test {len(t1) * (len(t2) + len(t3)):,}")

    # ------------------------------------------------------------------ 2
    r.h("2. Country distribution per source (train vs test)")
    cc = pd.DataFrame({nm: df["country"].value_counts() for nm, df in
                       [("tr_S1", s1), ("tr_S2", s2), ("tr_S3", s3), ("te_S1", t1), ("te_S2", t2), ("te_S3", t3)]}
                      ).fillna(0).astype(int)
    r.table(cc)
    unseen = sorted(set(pd.concat([t1, t2, t3])["country"]) - set(pd.concat([s1, s2, s3])["country"]))
    r.p(f"Countries in test but not in train: {unseen}")
    hints.append(f"Unseen test countries {unseen}: features must be country-agnostic; check France samples below.")

    # ------------------------------------------------------------------ 3
    r.h("3. Match structure (ground truth)")
    truth = gt_to_dict(gt)
    sizes = pd.Series({k: len(v) for k, v in truth.items()})
    n2 = pd.Series({k: sum(x.startswith("S2-") for x in v) for k, v in truth.items()})
    n3 = pd.Series({k: sum(x.startswith("S3-") for x in v) for k, v in truth.items()})
    r.p(f"singleton S1 entities (no match): {pct((sizes == 0).mean())}  "
        f"-> an all-empty submission scores {(sizes == 0).mean():.4f}")
    hints.append(f"Singleton rate {pct((sizes == 0).mean())} = score of predicting nothing (baseline to beat).")
    r.table(sizes.value_counts().sort_index().rename("n_S1_entities").to_frame())
    r.table(pd.crosstab(n2.clip(upper=5), n3.clip(upper=5), rownames=["#S2 (clip5)"], colnames=["#S3 (clip5)"]))
    r.p(f"mean matches per S1: {sizes.mean():.3f}; mean given >=1: {sizes[sizes > 0].mean():.3f}; "
        f"max: {sizes.max()}")

    all_matched = [x for v in truth.values() for x in v]
    mc = Counter(all_matched)
    multi = {k: c for k, c in mc.items() if c > 1}
    r.p(f"S2/S3 ids matched to >1 S1 entity: {len(multi)} (of {len(mc)} matched ids)")
    if not multi:
        hints.append("Every S2/S3 record belongs to at most ONE S1 entity -> enforce 1-to-1 assignment "
                     "(keep only the best S1 per S2/S3 record).")
    else:
        hints.append(f"{len(multi)} S2/S3 ids are shared across S1 entities - do NOT enforce strict 1-to-1.")
    unknown = [x for x in mc if x not in set(s2["entity_id"]) | set(s3["entity_id"])]
    r.p(f"GT ids not present in S2/S3 files: {len(unknown)}")
    frac_s2 = s2["entity_id"].isin(mc).mean()
    frac_s3 = s3["entity_id"].isin(mc).mean()
    r.p(f"fraction of S2 records that match some S1: {pct(frac_s2)}; S3: {pct(frac_s3)}")
    hints.append(f"Only {pct(frac_s2)} of S2 and {pct(frac_s3)} of S3 have a match: the rest are distractors.")

    # ------------------------------------------------------------------ 4
    r.h("4. Do matched records share country? (blocking-by-country safety)")
    lk = pd.concat([s1, s2, s3]).set_index("entity_id")
    pairs = [(a, b) for a, v in truth.items() for b in v if b in lk.index]
    ce = np.mean([lk.at[a, "country"] == lk.at[b, "country"] for a, b in pairs]) if pairs else float("nan")
    r.p(f"matched pairs with identical country label: {pct(ce)}")
    hints.append("Country is " + ("SAFE" if ce > 0.999 else f"NOT fully safe ({pct(ce)})")
                 + " as a hard blocking key.")
    r.p("country breakdown of singletons / match counts:")
    tmp = s1.assign(n=s1["entity_id"].map(sizes).fillna(0))
    r.table(tmp.groupby("country")["n"].agg(["count", "mean", lambda x: (x == 0).mean()])
            .rename(columns={"<lambda_0>": "singleton_rate"}))

    # ------------------------------------------------------------------ 5
    r.h("5. Similarity of true matches vs random non-matches")
    samp = [pairs[i] for i in rng.choice(len(pairs), size=min(4000, len(pairs)), replace=False)]
    pos = pd.DataFrame([sim_row(lk.loc[a], lk.loc[b]) for a, b in samp])
    others = pd.concat([s2, s3])["entity_id"].values
    neg_pairs = []
    s1_ids = s1["entity_id"].values
    while len(neg_pairs) < len(samp):
        a, b = rng.choice(s1_ids), rng.choice(others)
        if b not in truth.get(a, ()) and lk.at[a, "country"] == lk.at[b, "country"]:
            neg_pairs.append((a, b))
    neg = pd.DataFrame([sim_row(lk.loc[a], lk.loc[b]) for a, b in neg_pairs])
    r.table(pd.concat({"match": pos.describe().T[["mean", "25%", "50%", "75%"]],
                       "random_same_country": neg.describe().T[["mean", "25%", "50%", "75%"]]}, axis=1))
    r.p(f"true matches with name_ratio < 60: {pct((pos.name_ratio < 60).mean())} "
        f"(hard cases: need address / phonetic / token features)")
    r.p(f"true matches with EXACT normalized name equality: {pct((pos.name_ratio == 100).mean())}")
    r.p(f"true matches with postal code present in both: {pct(pos.pc_equal.notna().mean())}, "
        f"equal when both present: {pct(pos.pc_equal.mean())}")
    hints.append(f"Postal code equal in {pct(pos.pc_equal.mean())} of matched pairs when both present "
                 f"(present in both for {pct(pos.pc_equal.notna().mean())}).")
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

    # ------------------------------------------------------------------ 6
    r.h("6. Hard negatives: how similar are the *closest* non-matches?")
    r.p("For sampled S1 entities: best name_tset among its true matches vs best among same-country non-matches "
        "that share the first core-name token (these are the dangerous false merges).")
    by_first = {}
    oth = pd.concat([s2, s3])
    for eid, nm, ctry in zip(oth.entity_id, oth.business_name, oth.country):
        toks = core_name(nm).split()
        if toks:
            by_first.setdefault((ctry, toks[0]), []).append(eid)
    hard = []
    for a in rng.choice(s1_ids, size=min(1500, len(s1_ids)), replace=False):
        ra = lk.loc[a]
        toks = core_name(ra.business_name).split()
        if not toks:
            continue
        cands = [b for b in by_first.get((ra.country, toks[0]), []) if b not in truth.get(a, ())][:50]
        if not cands:
            continue
        best = max(cands, key=lambda b: fuzz.token_set_ratio(norm_name(ra.business_name),
                                                             norm_name(lk.at[b, "business_name"])))
        hard.append({"s1": a, "s1_name": ra.business_name, "s1_addr": ra.business_address,
                     "neg": best, "neg_name": lk.at[best, "business_name"], "neg_addr": lk.at[best, "business_address"],
                     **sim_row(ra, lk.loc[best])})
    hard = pd.DataFrame(hard)
    if len(hard):
        r.p(f"closest non-match with name_tset==100: {pct((hard.name_tset == 100).mean())} of sampled S1 "
            f"(chains/franchises/common names -> address features are decisive)")
        hard.sort_values("name_tset", ascending=False).head(300).to_csv(
            os.path.join(args.out_dir, "hard_negatives_sample.tsv"), sep="\t", index=False)
        r.table(hard.sort_values("name_tset", ascending=False)[
                    ["s1_name", "neg_name", "s1_addr", "neg_addr", "name_tset", "addr_tset"]].head(15))

    # ------------------------------------------------------------------ 7
    r.h("7. Raw examples of true matches (eyeball the noise)")
    ex = []
    for a, b in samp[:400]:
        ex.append({"s1": a, "s1_name": lk.at[a, "business_name"], "s1_addr": lk.at[a, "business_address"],
                   "m": b, "m_name": lk.at[b, "business_name"], "m_addr": lk.at[b, "business_address"],
                   "country": lk.at[a, "country"]})
    ex = pd.DataFrame(ex)
    ex.to_csv(os.path.join(args.out_dir, "match_examples.tsv"), sep="\t", index=False)
    for c, g in ex.groupby("country"):
        r.p(f"\n**{c}**")
        r.table(g.head(12).drop(columns=["s1", "m", "country"]))
    r.p("Lowest-similarity true matches (the ones a threshold will miss):")
    pos_ex = pd.DataFrame([{**sim_row(lk.loc[a], lk.loc[b]), "a": lk.at[a, "business_name"],
                            "b": lk.at[b, "business_name"], "aa": lk.at[a, "business_address"],
                            "ab": lk.at[b, "business_address"]} for a, b in samp[:1500]])
    r.table(pos_ex.sort_values("name_tset").head(20)[["a", "b", "aa", "ab", "name_tset", "addr_tset"]])

    # ------------------------------------------------------------------ 8
    r.h("8. Token vocabularies (to extend abbreviation tables)")
    r.p("Most common NAME tokens per country (train, all sources):")
    top_tokens(r, pd.concat([s1, s2, s3]), "business_name")
    r.p("Most common ADDRESS tokens per country (train, all sources):")
    top_tokens(r, pd.concat([s1, s2, s3]), "business_address")
    r.p("Most common NAME / ADDRESS tokens in TEST for countries unseen in train:")
    te = pd.concat([t1, t2, t3])
    if unseen:
        top_tokens(r, te[te.country.isin(unseen)], "business_name")
        top_tokens(r, te[te.country.isin(unseen)], "business_address")
        smp = te[te.country.isin(unseen)].sample(min(25, int(te.country.isin(unseen).sum())), random_state=0)
        r.table(smp[["entity_id", "business_name", "business_address"]])

    r.h("9. Character / format profile")
    for nm, df in [("train S1", s1), ("train S2", s2), ("train S3", s3), ("test S1", t1), ("test S2", t2),
                   ("test S3", t3)]:
        char_profile(r, nm, df)
    for nm, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        pcs = df["business_address"].map(postal_code)
        r.p(f"- train {nm}: postal code extractable {pct((pcs != '').mean())}; by country: "
            + ", ".join(f"{c}={pct((pcs[df.country == c] != '').mean())}" for c in df.country.unique()))
    lm_re = r"(?i)\b(?:near|nr|opp|opposite|behind|beside)\b"
    r.p("Address 'landmark' phrases (near/opp/behind) share per source: " + ", ".join(
        f"{nm}={pct(df.business_address.str.contains(lm_re).mean())}"
        for nm, df in [("S1", s1), ("S2", s2), ("S3", s3)]))

    # ------------------------------------------------------------------ 10
    r.h("10. Source-specific systematic differences (S1 vs its S2/S3 matches)")
    diffs = Counter()
    for a, b in samp[:3000]:
        ta = set(basic_clean(lk.at[a, "business_name"]).split())
        tb = set(basic_clean(lk.at[b, "business_name"]).split())
        for t in ta - tb:
            diffs[(b[:2], "dropped", t)] += 1
        for t in tb - ta:
            diffs[(b[:2], "added", t)] += 1
    r.table(pd.DataFrame([(s, k, t, n) for (s, k, t), n in diffs.most_common(40)],
                         columns=["src", "kind", "token", "count"]))

    # ------------------------------------------------------------------ 11
    r.h("11. ID leakage sanity check")
    num = lambda x: int(re.sub(r"\D", "", x) or 0)  # noqa: E731
    if pairs:
        a_num = np.array([num(a) for a, b in pairs])
        b_num = np.array([num(b) for a, b in pairs])
        r.p(f"Spearman(S1 id number, matched id number) = "
            f"{pd.Series(a_num).corr(pd.Series(b_num), method='spearman'):.4f} "
            "(~0 means ids are shuffled; we never use ids as features either way)")

    # ------------------------------------------------------------------ 12
    r.h("12. Quick blocking-recall probe (char 3-gram TF-IDF on names, same country)")
    try:
        from blocking import tfidf_topk_probe
        for k in (5, 10, 20, 50):
            rec = tfidf_topk_probe(s1, pd.concat([s2, s3]), truth, k)
            r.p(f"- top-{k} name neighbours: pair recall {pct(rec)}")
    except Exception as e:  # probe is best-effort
        r.p(f"(probe skipped: {e})")

    r.h("DECISION HINTS (auto-generated)")
    for h in hints:
        r.p(f"- {h}")
    r.save()
    print(f"\nreport written to {r.path}")


if __name__ == "__main__":
    main()
