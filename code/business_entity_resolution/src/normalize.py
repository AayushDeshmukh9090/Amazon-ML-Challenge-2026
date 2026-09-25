"""Text normalisation for business names and addresses.

Everything here is country-agnostic: the abbreviation tables are unions over
US / India / France conventions and are applied to every record, so an unseen
country label (France in test) goes through the same code path as the
training countries.
"""
from __future__ import annotations

import re
from functools import lru_cache
import json
import os

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

# US states: full name -> USPS code (applied to addresses; only full names are mapped, so
# 2-letter codes are never re-interpreted inside non-US text)
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
# "washington"/"virginia" also occur as street / city names: only map them when they are the
# LAST address component (see _map_states)
_STATE_TAIL_ONLY = {"washington", "virginia"}
_MULTI_STATES = sorted([k for k in list(US_STATES) + list(STATE_CANON) if " " in k], key=len, reverse=True)
_MULTI_RE = re.compile(r"\b(" + "|".join(map(re.escape, _MULTI_STATES)) + r")\b")

# Learned token synonyms (see synonyms.py): transliterated-script / alias token -> canonical
# token, mined from TRAINING matched pairs only.  Loaded once per process.
SYN_NAME: dict[str, str] = {}
SYN_ADDR: dict[str, str] = {}


def set_synonyms(name_map: dict | None = None, addr_map: dict | None = None):
    global SYN_NAME, SYN_ADDR
    SYN_NAME, SYN_ADDR = dict(name_map or {}), dict(addr_map or {})
    for f in (norm_name, core_name, norm_addr):
        f.cache_clear()


def load_synonyms(path: str | None) -> bool:
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        set_synonyms(d.get("name"), d.get("addr"))
        return True
    return False


_AMP = re.compile(r"\s*&\s*|\s+\+\s+")
_DBA = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|dba|d\.b\.a\.?|t\s*/\s*a|trading as|doing business as|aka|a\.k\.a\.?|"
                  r"f\s*/\s*k\s*/\s*a|fka|formerly known as|formerly)\b\s*:?", re.I)
_DOMAIN = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|net|org|in|co\.in|fr|biz|us|info)\b",
                     re.I)
_NULLS = re.compile(r"<\s*null\s*>|\bnull\b|\bnone\b|\bn/?a\b|\bnil\b", re.I)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")
_ORD = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|e|eme)\b")
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "8": "b"})
_NAME_NOISE = {"m", "s", "ms", "mr", "mrs", "id", "www", "com", "the",   # "M/s", "Mr", "(ID: x.com 123)"
               "dba", "formerly", "fka", "aka", "known", "as"}
_COUNTRY_WORDS = {"india", "france", "usa", "us", "america", "bharat"}


def basic_clean(s) -> str:
    """lowercase, transliterate to ASCII, & -> and, drop punctuation, strip ordinal suffixes."""
    if s is None or (isinstance(s, float) and s != s):
        return ""
    s = unidecode(str(s)).lower()
    s = _AMP.sub(" and ", s)
    s = s.replace("'", "").replace("`", "")
    s = re.sub(r"\b((?:[a-z]\.){2,})", lambda m: m.group(1).replace(".", ""), s)   # p.v.t. -> pvt
    s = _NON_ALNUM.sub(" ", s)
    s = _ORD.sub(r"\1", s)
    return _WS.sub(" ", s).strip()


def _fix_leet(tok: str) -> str:
    """'8rothers' -> 'brothers', 'c0mite' -> 'comite': single digit inside a mostly-letter token."""
    if len(tok) >= 4 and sum(c.isdigit() for c in tok) == 1 and sum(c.isalpha() for c in tok) >= 3:
        return tok.translate(_LEET)
    return tok


def _syn(toks, table):
    return [table.get(t, t) for t in toks] if table else toks


def _canon_tokens(tokens, table):
    return [table.get(t, t) for t in tokens]


def _name_tokens(name: str) -> list[str]:
    if not isinstance(name, str):
        return []
    s = _DOMAIN.sub(lambda m: " " + m.group(1) + " ", name)          # bydynamic.com -> bydynamic
    toks = [_fix_leet(t) for t in basic_clean(s).split()]
    return _canon_tokens(_syn(toks, SYN_NAME), LEGAL_CANON)


@lru_cache(maxsize=1 << 16)
def norm_name(name: str) -> str:
    return " ".join(_name_tokens(name))


def _is_legal(t: str) -> bool:
    # exact legal token, or a transliterated / misspelt long legal word ('limittedd', 'praaivett')
    return t in LEGAL_FORMS or (len(t) >= 5 and _skel_tok(t) in _LEGAL_SKELS)


@lru_cache(maxsize=1 << 16)
def core_name(name: str) -> str:
    """Name without legal forms, stop words, 'M/s', '(India)', domain residue. Falls back to norm_name."""
    toks = _name_tokens(name)
    core = [t for t in toks if not _is_legal(t) and t not in _NAME_NOISE and t not in _COUNTRY_WORDS
            and not t.isdigit()]
    return " ".join(core) if core else " ".join(toks)


def split_dba(name: str) -> list[str]:
    """[full, part1, part2] when a DBA / formerly / fka marker is present, else [full]."""
    if not isinstance(name, str):
        return [""]
    parts = [p.strip(" ,;:-()|") for p in _DBA.split(name) if p and p.strip(" ,;:-()|")]
    if "|" in name:  # "mv trace private limited | www.mvtracep.com"
        parts += [p.strip() for p in name.split("|") if p.strip()]
    return [name] + parts if len(parts) > 1 else [name]


def _map_states(s: str) -> str:
    s = _MULTI_RE.sub(lambda m: US_STATES.get(m.group(1)) or STATE_CANON.get(m.group(1)), s)
    toks = s.split()
    out = []
    for i, t in enumerate(toks):
        if t in US_STATES and (t not in _STATE_TAIL_ONLY or i == len(toks) - 1):
            out.append(US_STATES[t])
        else:
            out.append(STATE_CANON.get(t, t) if len(t) > 3 else t)
    return " ".join(out)


@lru_cache(maxsize=1 << 16)
def norm_addr(addr: str) -> str:
    if not isinstance(addr, str):
        return ""
    s = basic_clean(_NULLS.sub(" ", addr))
    s = " ".join(_syn(s.split(), SYN_ADDR))
    s = _map_states(s)
    return " ".join(_canon_tokens(s.split(), ADDR_CANON))


def acronym(name: str) -> str:
    toks = core_name(name).split()
    return "".join(t[0] for t in toks if t and not t.isdigit())


def _skel_tok(tok: str) -> str:
    if tok.isdigit():
        return tok
    for a, b in (("ph", "f"), ("sh", "s"), ("ch", "c"), ("kh", "k"), ("gh", "g"), ("th", "t"),
                 ("dh", "d"), ("bh", "b"), ("jh", "j"), ("ck", "k"), ("q", "k"), ("c", "k"),
                 ("w", "v"), ("z", "j"), ("x", "ks"), ("y", "i")):
        tok = tok.replace(a, b)
    rest = re.sub(r"[aeiouh]", "", tok[1:])
    return tok[0] + re.sub(r"(.)\1+", r"\1", rest)


def phonetic_key(s: str) -> str:
    """Transliteration/typo-robust skeleton: sharma/shurma -> srm, limittedd/limited -> lmtd."""
    return " ".join(_skel_tok(t) for t in basic_clean(s).split())


def numbers(s: str) -> set[str]:
    """Digit runs with leading zeros stripped ('0094' -> '94', '3503-3507' -> {3503, 3507})."""
    if not isinstance(s, str):
        return set()
    return {d.lstrip("0") or "0" for d in _DIGITS.findall(unidecode(s))}


def house_number(addr: str) -> str:
    """First number of the address that is not a postal code (leading zeros stripped)."""
    if not isinstance(addr, str):
        return ""
    for d in _DIGITS.findall(unidecode(_NULLS.sub(" ", addr))):
        if len(d) <= 5 or len(d.lstrip("0")) <= 5:
            return d.lstrip("0") or "0"
    return ""


def postal_code(addr: str) -> str:
    """Best-effort postal code: India PIN6, US ZIP5 / France CP5 (last 5-digit number). '' if absent."""
    if not isinstance(addr, str):
        return ""
    a = unidecode(addr)
    m = re.search(r"\b(\d{6})\b", a)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{3})\s(\d{3})\b", a)
    if m:
        return m.group(1) + m.group(2)
    ms = re.findall(r"\b(\d{5})(?:-\d{4})?\b", a)
    return ms[-1] if len(ms) >= 1 and len(re.findall(r"\d+", a)) > 1 else ""


def block_tokens(name: str, addr: str) -> list[str]:
    """Tokens for the inverted-index blocker: name skeletons, no-space core name, address
    skeletons, and normalised numbers - each namespaced so they never collide."""
    cn = core_name(name)
    nsk = [_skel_tok(t) for t in cn.split() if len(t) > 1]
    toks = {"n:" + t for t in nsk}
    toks.update("nb:" + a + "_" + b for a, b in zip(nsk, nsk[1:]))          # name bigrams
    ns = cn.replace(" ", "")
    if len(ns) >= 4:
        toks.add("ns:" + ns)                                                # 'vangregionalfinance'
    ask = []
    for t in norm_addr(addr).split():
        if t.isdigit():
            ask.append("#" + (t.lstrip("0") or "0"))
        elif len(t) > 1:
            ask.append(_skel_tok(t))
    toks.update(("#:" + t[1:]) if t[0] == "#" else ("a:" + t) for t in ask)
    toks.update("ab:" + a + "_" + b for a, b in zip(ask, ask[1:]))          # '1020_fldng', 'fldng_ln'
    return sorted(toks)


_LEGAL_SKELS = {_skel_tok(w) for w in ("private", "limited", "corporation", "incorporated", "company",
                                       "societe", "compagnie", "etablissements")}
