"""Text normalisation for business names and addresses.

Everything here is country-agnostic: the abbreviation tables are unions over
US / India / France conventions and are applied to every record, so an unseen
country label (France in test) goes through the same code path as the
training countries.
"""
from __future__ import annotations

import re
from functools import lru_cache

from unidecode import unidecode

# --------------------------------------------------------------------------
# Legal-form / corporate suffix tokens.  Mapped to a canonical form so that
# "Pvt Ltd" == "Private Limited" and can be stripped to get a "core" name.
# --------------------------------------------------------------------------
LEGAL_CANON = {
    # English / US
    "corporation": "corp", "corp": "corp", "incorporated": "inc", "inc": "inc",
    "company": "co", "co": "co", "cos": "co", "limited": "ltd", "ltd": "ltd", "lmtd": "ltd",
    "llc": "llc", "l.l.c": "llc", "llp": "llp", "lp": "lp", "plc": "plc",
    "pllc": "pllc", "pc": "pc", "pa": "pa", "na": "na",
    "group": "group", "grp": "group", "holdings": "holdings", "hldgs": "holdings",
    "enterprises": "enterprises", "enterprise": "enterprises", "ent": "enterprises",
    "associates": "associates", "assoc": "associates", "assocs": "associates",
    "international": "intl", "intl": "intl", "industries": "industries", "inds": "industries",
    "services": "services", "svcs": "services", "svc": "services", "srvcs": "services",
    "solutions": "solutions", "soln": "solutions", "solns": "solutions",
    "technologies": "technologies", "technology": "technologies", "tech": "technologies",
    "brothers": "bros", "bros": "bros", "trust": "trust",
    # India
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "prv": "pvt",
    "opc": "opc", "huf": "huf",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "scop": "scop", "societe": "ste", "ste": "ste", "cie": "co", "compagnie": "co",
    "etablissements": "ets", "ets": "ets", "et": "and",
}
# the subset that are purely legal forms (stripped for the core name)
LEGAL_FORMS = {
    "corp", "inc", "co", "ltd", "llc", "llp", "lp", "plc", "pllc", "pc", "pa", "na",
    "pvt", "opc", "huf", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "ste",
    "ets", "the", "and", "of", "le", "la", "les", "de", "des", "du", "d", "l",
}

# --------------------------------------------------------------------------
# Address abbreviations (union of US / India / France)
# --------------------------------------------------------------------------
ADDR_CANON = {
    # US street types
    "street": "st", "st": "st", "str": "st", "avenue": "ave", "ave": "ave", "av": "ave", "avn": "ave",
    "road": "rd", "rd": "rd", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd",
    "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl", "square": "sq", "sq": "sq", "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy", "circle": "cir", "cir": "cir", "terrace": "ter", "ter": "ter",
    "trail": "trl", "trl": "trl", "way": "way", "expressway": "expy", "expy": "expy",
    "freeway": "fwy", "fwy": "fwy", "turnpike": "tpke", "tpke": "tpke", "plaza": "plz", "plz": "plz",
    "suite": "ste", "ste": "ste", "ste.": "ste", "apartment": "apt", "apt": "apt", "unit": "unit",
    "floor": "fl", "fl": "fl", "flr": "fl", "building": "bldg", "bldg": "bldg", "room": "rm", "rm": "rm",
    "number": "no", "no": "no", "num": "no", "nbr": "no",
    "north": "n", "south": "s", "east": "e", "west": "w", "n": "n", "s": "s", "e": "e", "w": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "mount": "mt", "mt": "mt", "saint": "st", "fort": "ft", "ft": "ft",
    # India
    "marg": "marg", "mg": "marg", "nagar": "ngr", "ngr": "ngr", "nager": "ngr",
    "colony": "col", "col": "col", "sector": "sec", "sec": "sec", "sect": "sec",
    "phase": "ph", "ph": "ph", "block": "blk", "blk": "blk", "opposite": "opp", "opp": "opp",
    "near": "nr", "nr": "nr", "behind": "bh", "bhd": "bh", "bh": "bh",
    "chowk": "chk", "chk": "chk", "cross": "crs", "crs": "crs", "main": "mn", "mn": "mn",
    "layout": "lyt", "lyt": "lyt", "extension": "extn", "extn": "extn", "ext": "extn",
    "industrial": "indl", "indl": "indl", "ind": "indl", "estate": "est", "est": "est",
    "area": "ar", "complex": "cmplx", "cmplx": "cmplx", "apartments": "apts", "apts": "apts",
    "society": "soc", "soc": "soc", "house": "hse", "hse": "hse", "tower": "twr", "twr": "twr",
    "post": "po", "po": "po", "district": "dist", "dist": "dist", "dt": "dist", "distt": "dist",
    "taluk": "tq", "taluka": "tq", "tq": "tq", "tehsil": "teh", "teh": "teh",
    "village": "vill", "vill": "vill", "vil": "vill", "gali": "gali", "galli": "gali",
    "bazaar": "bazar", "bazar": "bazar", "bzr": "bazar", "puram": "puram", "pur": "pur",
    "halli": "halli", "bagh": "bagh", "ganj": "ganj", "wadi": "wadi", "peth": "peth",
    "pin": "pin", "pincode": "pin",
    # France
    "rue": "rue", "r": "rue", "chemin": "ch", "ch": "ch", "che": "ch", "impasse": "imp", "imp": "imp",
    "allee": "all", "all": "all", "route": "rte", "rte": "rte", "quai": "qu", "qu": "qu",
    "cours": "crs", "faubourg": "fbg", "fbg": "fbg", "passage": "pass", "pass": "pass",
    "residence": "res", "res": "res", "lieu": "ld", "lieudit": "ld", "ld": "ld",
    "zone": "z", "za": "za", "zi": "zi", "zac": "zac", "cedex": "cedex", "bis": "bis", "ter.": "ter",
}

# Indian state names / common abbreviations -> canonical
STATE_CANON = {
    "maharashtra": "mh", "mh": "mh", "karnataka": "ka", "ka": "ka", "tamil nadu": "tn", "tamilnadu": "tn",
    "tn": "tn", "delhi": "dl", "new delhi": "dl", "dl": "dl", "uttar pradesh": "up", "up": "up",
    "gujarat": "gj", "gj": "gj", "rajasthan": "rj", "rj": "rj", "west bengal": "wb", "wb": "wb",
    "telangana": "ts", "ts": "ts", "tg": "ts", "andhra pradesh": "ap", "ap": "ap", "kerala": "kl", "kl": "kl",
    "madhya pradesh": "mp", "mp": "mp", "haryana": "hr", "hr": "hr", "punjab": "pb", "pb": "pb",
    "bihar": "br", "br": "br", "odisha": "od", "orissa": "od", "od": "od",
}

_AMP = re.compile(r"\s*&\s*|\s+\+\s+")
_DBA = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|dba|d\.b\.a\.?|t\s*/\s*a|trading as|doing business as|aka|a\.k\.a\.?)\b",
                  re.I)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")
_ORD = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|e|eme|ème)\b")
_US_ZIP = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_IN_PIN = re.compile(r"\b(\d{3})\s?(\d{3})\b")


def basic_clean(s) -> str:
    """lowercase, strip accents/transliterate to ASCII, map & -> and, drop punctuation."""
    if s is None or (isinstance(s, float) and s != s):
        return ""
    s = unidecode(str(s)).lower()
    s = _AMP.sub(" and ", s)
    s = s.replace("'", "").replace("`", "")
    # keep dots inside abbreviations together: "p.v.t." -> "pvt"
    s = re.sub(r"\b((?:[a-z]\.){2,})", lambda m: m.group(1).replace(".", ""), s)
    s = _NON_ALNUM.sub(" ", s)
    s = _ORD.sub(r"\1", s)
    return _WS.sub(" ", s).strip()


def _canon_tokens(tokens, table):
    return [table.get(t, t) for t in tokens]


@lru_cache(maxsize=None)
def norm_name(name: str) -> str:
    toks = basic_clean(name).split()
    return " ".join(_canon_tokens(toks, LEGAL_CANON))


@lru_cache(maxsize=None)
def core_name(name: str) -> str:
    """Name with legal forms / stop words removed (falls back to norm name)."""
    toks = norm_name(name).split()
    core = [t for t in toks if t not in LEGAL_FORMS]
    return " ".join(core) if core else " ".join(toks)


def split_dba(name: str) -> list[str]:
    """Return [full, legal part, trade part] when a DBA marker is present."""
    if not isinstance(name, str):
        return [""]
    parts = [p.strip(" ,;-()") for p in _DBA.split(name) if p and p.strip(" ,;-()")]
    return [name] + parts if len(parts) > 1 else [name]


@lru_cache(maxsize=None)
def norm_addr(addr: str) -> str:
    s = basic_clean(addr)
    for full, ab in STATE_CANON.items():
        if " " in full:
            s = s.replace(full, ab)
    toks = s.split()
    return " ".join(_canon_tokens(toks, ADDR_CANON))


def acronym(name: str) -> str:
    toks = core_name(name).split()
    return "".join(t[0] for t in toks if t and not t.isdigit())


@lru_cache(maxsize=None)
def phonetic_key(s: str) -> str:
    """Crude transliteration-robust skeleton (Hindi/English/French spellings).

    sharma/sharmaa/shurma -> srm ; ph/f, v/w, z/j, k/c/q, ee/i, oo/u collapse.
    """
    s = basic_clean(s)
    for a, b in (("ph", "f"), ("sh", "s"), ("ch", "c"), ("kh", "k"), ("gh", "g"), ("th", "t"),
                 ("dh", "d"), ("bh", "b"), ("jh", "j"), ("ck", "k"), ("q", "k"), ("c", "k"),
                 ("w", "v"), ("z", "j"), ("x", "ks"), ("y", "i")):
        s = s.replace(a, b)
    out = []
    for tok in s.split():
        if tok.isdigit():
            out.append(tok)
            continue
        first = tok[0]
        rest = re.sub(r"[aeiouh]", "", tok[1:])
        rest = re.sub(r"(.)\1+", r"\1", rest)
        out.append(first + rest)
    return " ".join(out)


def numbers(s: str) -> set[str]:
    return set(_DIGITS.findall(basic_clean(s)))


def postal_code(addr: str) -> str:
    """Best-effort postal code: US ZIP5, India PIN6, France CP5.  '' if absent."""
    if not isinstance(addr, str):
        return ""
    a = unidecode(addr)
    m = re.search(r"\b(\d{6})\b", a)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{3})\s(\d{3})\b", a)  # "400 001"
    if m:
        return m.group(1) + m.group(2)
    # 5-digit: take the LAST 5-digit number (house numbers usually come first)
    ms = re.findall(r"\b(\d{5})(?:-\d{4})?\b", a)
    return ms[-1] if ms else ""
