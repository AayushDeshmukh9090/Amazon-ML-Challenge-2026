"""Generate a small SYNTHETIC dataset in the challenge format, only for smoke-testing
the pipeline when the real data is not at hand.  Never used for training the
submitted model.

  python utils/make_synthetic_data.py --out dataset_synth --n-train 3000 --n-test 2000
"""
import argparse
import os
import random

WORDS_US = ["Summit", "Pioneer", "Blue", "River", "Eagle", "Liberty", "Golden", "Harbor", "Maple", "Atlas",
            "Crest", "Evergreen", "Granite", "Silver", "North", "Star", "Oak", "Bright", "Prime", "Coastal"]
KIND_US = ["Dental", "Auto Repair", "Consulting", "Bakery", "Logistics", "Realty", "Plumbing", "Law Group",
           "Pharmacy", "Fitness", "Printing", "Insurance"]
SUF_US = [("Corporation", "Corp"), ("Incorporated", "Inc"), ("Limited Liability Company", "LLC"),
          ("Company", "Co"), ("", "")]
STREETS_US = ["Main", "Oak", "Elm", "Washington", "Lake", "Hill", "Park", "Cedar", "Pine", "Sunset"]
ST_TYPES_US = [("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Boulevard", "Blvd"), ("Drive", "Dr")]
CITIES_US = [("Austin", "TX", "787"), ("Denver", "CO", "802"), ("Seattle", "WA", "981"), ("Boston", "MA", "021"),
             ("Chicago", "IL", "606")]

WORDS_IN = ["Shree", "Ganesh", "Laxmi", "Sai", "Krishna", "Balaji", "Om", "Durga", "Jai", "Mahalaxmi", "Vinayak",
            "Sharma", "Patel", "Agarwal", "Reddy", "Iyer", "Gupta", "Mehta", "Singh", "Rao"]
KIND_IN = ["Traders", "Enterprises", "Textiles", "Electricals", "Pharma", "Steels", "Hardware", "Sweets",
           "Motors", "Jewellers", "Agencies", "Infotech"]
SUF_IN = [("Private Limited", "Pvt Ltd"), ("Limited", "Ltd"), ("", ""), ("LLP", "LLP")]
AREAS_IN = ["Andheri East", "Koramangala", "T Nagar", "Salt Lake", "Banjara Hills", "Navrangpura", "Kothrud",
            "Sector 18", "Laxmi Nagar", "Hazratganj"]
CITIES_IN = [("Mumbai", "Maharashtra", "400"), ("Bengaluru", "Karnataka", "560"), ("Chennai", "Tamil Nadu", "600"),
             ("Kolkata", "West Bengal", "700"), ("Hyderabad", "Telangana", "500"), ("Pune", "Maharashtra", "411")]
LANDMARKS = ["Near SBI ATM", "Opp Railway Station", "Behind City Mall", "Near Hanuman Mandir"]
TRANSLIT = [("ee", "i"), ("sh", "s"), ("aa", "a"), ("v", "w"), ("ph", "f"), ("ksh", "x")]

WORDS_FR = ["Boulangerie", "Maison", "Atelier", "Garage", "Cabinet", "Pharmacie", "Societe", "Bistrot", "Galerie"]
KIND_FR = ["Dupont", "Martin", "Lefevre", "du Soleil", "des Alpes", "Saint-Michel", "Moreau", "de la Gare",
           "Bernard", "Rousseau"]
SUF_FR = [("SARL", "SARL"), ("SAS", "S.A.S."), ("", ""), ("SA", "SA")]
STREETS_FR = [("Rue", "R."), ("Avenue", "Av."), ("Boulevard", "Bd"), ("Place", "Pl.")]
NAMES_FR = ["de la Republique", "Victor Hugo", "Jean Jaures", "du General de Gaulle", "Pasteur", "des Lilas"]
CITIES_FR = [("Paris", "75"), ("Lyon", "69"), ("Marseille", "13"), ("Toulouse", "31"), ("Lille", "59")]


def typo(s, r):
    if len(s) < 4 or r.random() > 0.5:
        return s
    i = r.randrange(1, len(s) - 1)
    op = r.choice("dsi")
    if op == "d":
        return s[:i] + s[i + 1:]
    if op == "s":
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + r.choice("aeiourn") + s[i:]


def make_entity(r, country):
    if country == "US":
        core = f"{r.choice(WORDS_US)} {r.choice(WORDS_US)} {r.choice(KIND_US)}"
        suf = r.choice(SUF_US)
        city, st, z = r.choice(CITIES_US)
        addr = dict(num=str(r.randint(1, 9999)), street=r.choice(STREETS_US), stype=r.choice(ST_TYPES_US),
                    city=city, state=st, pc=z + f"{r.randint(0, 99):02d}")
    elif country == "India":
        core = f"{r.choice(WORDS_IN)} {r.choice(WORDS_IN)} {r.choice(KIND_IN)}"
        suf = r.choice(SUF_IN)
        city, st, z = r.choice(CITIES_IN)
        addr = dict(num=f"{r.randint(1, 300)}/{r.randint(1, 20)}", street=r.choice(AREAS_IN),
                    stype=("Road", "Rd"), city=city, state=st, pc=z + f"{r.randint(0, 999):03d}")
    else:
        core = f"{r.choice(WORDS_FR)} {r.choice(KIND_FR)}"
        suf = r.choice(SUF_FR)
        city, dep = r.choice(CITIES_FR)
        addr = dict(num=str(r.randint(1, 200)), street=r.choice(NAMES_FR), stype=r.choice(STREETS_FR),
                    city=city, state="", pc=dep + f"{r.randint(0, 999):03d}")
    return dict(core=core, suf=suf, addr=addr, country=country)


def render(e, r, noisy):
    core, (sf, sa), a, c = e["core"], e["suf"], e["addr"], e["country"]
    name = core
    if noisy:
        if c == "India" and r.random() < 0.4:
            for x, y in TRANSLIT:
                if x in name.lower() and r.random() < 0.5:
                    name = name.replace(x, y).replace(x.capitalize(), y.capitalize())
        if r.random() < 0.3:
            toks = name.split()
            i = r.randrange(len(toks))
            toks[i] = typo(toks[i], r)
            name = " ".join(toks)
        if r.random() < 0.15:
            toks = name.split()
            toks[0], toks[1] = toks[1], toks[0]
            name = " ".join(toks)
        if r.random() < 0.3:
            name = name.upper()
    sfx = (sa if noisy and r.random() < 0.6 else sf) if sf else ""
    if noisy and r.random() < 0.25:
        sfx = ""
    name = (name + " " + sfx).strip()
    if noisy and r.random() < 0.05:
        name = name + " DBA " + r.choice(WORDS_US) + " Store"
    stype = a["stype"][1] if (noisy and r.random() < 0.6) else a["stype"][0]
    if c == "France":
        parts = [f"{a['num']} {stype} {a['street']}", f"{a['pc']} {a['city']}"]
    else:
        parts = [f"{a['num']} {a['street']} {stype}", a["city"], a["state"], a["pc"]]
    if noisy:
        if r.random() < 0.3:
            parts = [p for p in parts if p != a["pc"] and not p.endswith(a["pc"])] or parts
        if r.random() < 0.2 and len(parts) > 2:
            parts = parts[:-2] + [parts[-1]]
        if c == "India" and r.random() < 0.3:
            parts = [r.choice(LANDMARKS)] + parts
    return name, ", ".join(p for p in parts if p)


def build(r, n, countries, prefix_dir, split, with_gt):
    ents = [make_entity(r, r.choice(countries)) for _ in range(n)]
    s1, s2, s3, gt = [], [], [], []
    oth = {"S2": [], "S3": []}
    for i, e in enumerate(ents):
        sid = f"S1-{i:06d}"
        nm, ad = render(e, r, noisy=False)
        s1.append((sid, nm, ad, e["country"]))
        matches = []
        roll = r.random()
        n2 = 0 if roll < 0.35 else (1 if roll < 0.8 else r.randint(2, 3))
        n3 = 0 if r.random() < 0.5 else (1 if r.random() < 0.85 else 2)
        for src, k in (("S2", n2), ("S3", n3)):
            for _ in range(k):
                oth[src].append((e, len(matches)))
                matches.append(src)
        gt.append([sid, []])
    # distractors (entities not in S1)
    for _ in range(n // 2):
        e = make_entity(r, r.choice(countries))
        oth[r.choice(["S2", "S3"])].append((e, None))
    ent_index = {id(e): i for i, e in enumerate(ents)}
    for src, lst in oth.items():
        r.shuffle(lst)
        rows = s2 if src == "S2" else s3
        for j, (e, m) in enumerate(lst):
            oid = f"{src}-{j:06d}"
            nm, ad = render(e, r, noisy=True)
            rows.append((oid, nm, ad, e["country"]))
            if m is not None:
                gt[ent_index[id(e)]][1].append(oid)
    d = os.path.join(prefix_dir, split)
    os.makedirs(d, exist_ok=True)
    hdr = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    for name, rows in (("source1", s1), ("source2", s2), ("source3", s3)):
        with open(os.path.join(d, f"{split}_{name}.tsv"), "w") as f:
            f.write(hdr + "".join("\t".join(x) + "\n" for x in rows))
    if with_gt:
        with open(os.path.join(d, f"{split}_ground_truth.tsv"), "w") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("".join(f"{a}\t{','.join(b)}\n" for a, b in gt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_synth")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=2000)
    a = ap.parse_args()
    rnd = random.Random(42)
    build(rnd, a.n_train, ["US", "India"], a.out, "train", True)
    build(rnd, a.n_test, ["US", "India", "France"], a.out, "test", False)
    print("synthetic data written to", a.out)
