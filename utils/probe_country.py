"""Leaderboard probe: how good is our submission on ONE country (e.g. France, never seen in training)?

Makes a copy of a matching_results.tsv in which every S1 entity of the chosen country gets an EMPTY
match list.  An empty list scores 1 for a true singleton and 0 otherwise, so

    LB_full - LB_probe = w_c * (F_c - s_c)      ->      F_c = s_c + (LB_full - LB_probe) / w_c

w_c = share of test S1 entities in country c (printed below), s_c = singleton share of country c
(~0.17-0.18 in train for both countries; the same generator made test).  Two probes (blank france,
blank us) give France, US and - by subtraction from LB_full - India.

  python utils/probe_country.py --country france
  python utils/probe_country.py --estimate --lb 0.9720 --lb-probe 0.6600 --country france
"""
import argparse
import csv
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def weights(s1_path):
    s1 = pd.read_csv(s1_path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False,
                     usecols=["entity_id", "country"])
    s1["c"] = s1["country"].str.strip().str.lower()
    return s1, s1["c"].value_counts(normalize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True, help="country label to blank, e.g. france / us / india")
    ap.add_argument("--results", default=os.path.join(ROOT, "output", "matching_results.tsv"))
    ap.add_argument("--s1", default=os.path.join(ROOT, "dataset", "test", "test_source1.tsv"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--estimate", action="store_true", help="compute F_country from two leaderboard scores")
    ap.add_argument("--lb", type=float, help="leaderboard score of the full submission")
    ap.add_argument("--lb-probe", type=float, help="leaderboard score of the probe submission")
    ap.add_argument("--singleton", type=float, default=0.1745, help="singleton share assumed for the country")
    a = ap.parse_args()
    c = a.country.strip().lower()
    s1, w = weights(a.s1)
    if c not in w:
        raise SystemExit(f"country '{c}' not in {list(w.index)}")
    if a.estimate:
        f = a.singleton + (a.lb - a.lb_probe) / w[c]
        rest = (a.lb - w[c] * f) / (1 - w[c])
        print(f"estimated macro F0.5 on {c}: {f:.4f}  (other countries together: {rest:.4f}); "
              f"(an error of x in the assumed singleton share {a.singleton} moves this by x)")
        return
    res = pd.read_csv(a.results, sep="\t", dtype=str, keep_default_na=False)
    ids = set(s1.loc[s1["c"] == c, "entity_id"])
    blank = res["source1_entity_id"].isin(ids)
    res.loc[blank, res.columns[1]] = ""
    out = a.out_dir or os.path.join(ROOT, f"output_probe_{c}")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "matching_results.tsv")
    res.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE)
    print(f"country shares of test S1: {w.round(4).to_dict()}")
    print(f"blanked {int(blank.sum()):,} {c} S1 rows -> {path}")
    print(f"upload it, then:  python utils/probe_country.py --estimate --country {c} --lb <full score> "
          f"--lb-probe <probe score>")


if __name__ == "__main__":
    main()
