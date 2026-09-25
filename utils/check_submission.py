"""Local mirror of the submission rules (the official utils/validate_submission.py is authoritative).

  python utils/check_submission.py --out-dir output --test-dir dataset/test
"""
import argparse
import csv
import os
import sys


def read_ids(p):
    with open(p, encoding="utf-8") as f:
        next(f)
        return [line.rstrip("\n").split("\t")[0] for line in f]


def read_lists(p, col):
    rows = {}
    errs = []
    with open(p, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        hdr = next(r)
        if hdr != ["source1_entity_id", col]:
            errs.append(f"{p}: bad header {hdr}")
        for row in r:
            if len(row) != 2:
                errs.append(f"{p}: bad row {row}")
                continue
            s, ids = row
            if s in rows:
                errs.append(f"{p}: duplicate row {s}")
            lst = [x for x in ids.split(",") if x] if ids else []
            if len(lst) != len(set(lst)):
                errs.append(f"{p}: duplicate ids in list for {s}")
            rows[s] = lst
    return rows, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--test-dir", default="dataset/test")
    a = ap.parse_args()
    s1 = read_ids(os.path.join(a.test_dir, "test_source1.tsv"))
    valid = set(read_ids(os.path.join(a.test_dir, "test_source2.tsv"))) | set(
        read_ids(os.path.join(a.test_dir, "test_source3.tsv")))
    m, e1 = read_lists(os.path.join(a.out_dir, "matching_results.tsv"), "matched_entity_ids")
    c, e2 = read_lists(os.path.join(a.out_dir, "candidate_pairs.tsv"), "candidate_entity_ids")
    errs = e1 + e2
    for name, d in (("matching", m), ("candidate", c)):
        miss = set(s1) - set(d)
        extra = set(d) - set(s1)
        if miss:
            errs.append(f"{name}: {len(miss)} S1 ids missing")
        if extra:
            errs.append(f"{name}: {len(extra)} unknown S1 ids")
        bad = sum(1 for v in d.values() for x in v if x not in valid)
        if bad:
            errs.append(f"{name}: {bad} ids not in test S2/S3")
    notcand = sum(1 for k, v in m.items() for x in v if x not in set(c.get(k, [])))
    if notcand:
        errs.append(f"{notcand} matched ids are not in candidate list")
    n_m = sum(map(len, m.values()))
    print(f"S1={len(s1)} matches={n_m} non-empty={sum(1 for v in m.values() if v)} "
          f"candidates={sum(map(len, c.values()))}")
    if errs:
        print("FAIL"), [print(" -", x) for x in errs[:50]]
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
