"""ClauseBench dimension extractors.

Pure regex-based extractors that pull structured benchmark dimensions out of
contract clause text (CUAD answer spans, LEDGAR provisions).

Every function is total on strings (never raises on ordinary text) and returns
None (or an empty result) when the dimension is *unfixable* from the text, i.e.
when no honest value can be extracted. Unfixable does NOT mean "rare": it
covers redactions ("[***]"), defined terms that are never expanded in the span
("the Territory"), qualitative-only language, and genuinely absent dimensions.

Worldwide geo-breadth sentinel: ``WORLDWIDE_BREADTH = 200`` (documented here;
used by ``extract_geo_scope`` when a span says worldwide/global).

Doctests below are the hand-written sanity + regression examples (2026-10-17
fix pass); run with ``python -m doctest src/dimensions.py -v``.

>>> round(extract_duration_months("for a period of 24 months following termination"), 4)
24.0
>>> round(extract_duration_months("during the two-year non-compete term of this Agreement"), 4)
24.0
>>> extract_duration_months("Employee shall not compete for twelve (12) months from the Effective Date")
12.0
>>> extract_duration_months("no earlier than three (3) years prior to the expiration of the Noncompetition Period") is None
True
>>> extract_duration_months("failure to cure such violation within thirty (30) days of notification") is None
True
>>> extract_amount_usd("shall not exceed $1,000,000 in the aggregate")
1000000.0
>>> extract_amount_usd("USD 5 million per occurrence")
5000000.0
>>> extract_amount_usd("five million dollars")
5000000.0
>>> extract_amount_usd("$500k per claim")
500000.0
>>> extract_geo_scope("shall not compete worldwide for two years")
{'jurisdictions': ['worldwide'], 'breadth': 200}
>>> extract_geo_scope("shall not compete anywhere in the world for two years")
{'jurisdictions': ['worldwide'], 'breadth': 200}
>>> extract_indemnity_mutuality("Each party shall indemnify the other party against losses.")
'mutual'
>>> extract_indemnity_mutuality("The Company shall indemnify and hold harmless the Executive.")
'unilateral'
>>> extract_indemnity_mutuality("you shall be indemnified and held harmless by the Company")
'unilateral'
>>> extract_indemnity_mutuality("The Company shall promptly indemnify Executive")
'unilateral'
>>> extract_indemnity_mutuality("The Executive shall at all times during employment be indemnified by the Bank")
'unilateral'
>>> extract_notice_days("terminate upon thirty (30) days' written notice")
30.0
>>> extract_notice_days("provides a thirty-day notice of written termination")
30.0
>>> extract_notice_days("at any time upon two weeks notice")
14.0
>>> round(extract_notice_days("MediWound may terminate at any time, by 6 (six) months prior notice in writing"), 2)
182.64
>>> extract_governing_law("This Agreement shall be governed by the laws of the State of Texas.")
'State of Texas'
>>> extract_governing_law("This Agreement shall be construed in accordance with and governed by the substantive internal laws of the State of New York.")
'State of New York'
>>> extract_duration_months("See Section 12 for the renewal mechanics.") is None
True
>>> extract_amount_usd("dated as of July 1, 2015") is None
True
"""

import re

# ---------------------------------------------------------------------------
# Shared number-word machinery
# ---------------------------------------------------------------------------

_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
}
# Alternation for regexes (longest-first so "eighteen" beats "eight").
_WORD_ALT = "|".join(sorted(_NUM_WORDS, key=len, reverse=True))

# ---------------------------------------------------------------------------
# Duration extraction
# ---------------------------------------------------------------------------

_DAYS_PER_MONTH = 30.44
_WEEKS_PER_MONTH = 4.345

_UNIT_TO_MONTHS = {
    "month": 1.0, "months": 1.0,
    "year": 12.0, "years": 12.0,
    "week": 1.0 / _WEEKS_PER_MONTH, "weeks": 1.0 / _WEEKS_PER_MONTH,
    "day": 1.0 / _DAYS_PER_MONTH, "days": 1.0 / _DAYS_PER_MONTH,
}

# Word-number + parenthesised digits ("twelve (12) months") -> keep digits.
_PAREN_NUM_RE = re.compile(
    r"(?<![\w$])(?:" + _WORD_ALT + r")(?:[\s-]+(?:" + _WORD_ALT + r"))*"
    r"\s*\(\s*(\d+(?:\.\d+)?)\s*\)"
    r"(?=\s*(?:business\s+|calendar\s+)?(?:months?|years?|weeks?|days?)\b)",
    re.IGNORECASE,
)

# Numeric duration: "24 months", "90 days", "18-month", "two-year" is word form.
# Guards: no "$" directly before (dollar amounts); not preceded by
# Section/Clause/Article/Exhibit/Schedule/Page (clause/page numbers); not a
# fragment of a longer dotted number ("12.2.2").
_NUM_DUR_RE = re.compile(
    r"(?<![\$\d])"
    r"(?<!Section\s)(?<!Clause\s)(?<!Article\s)(?<!Exhibit\s)"
    r"(?<!Schedule\s)(?<!Page\s)(?<!\d\.)"
    r"(\d+(?:\.\d+)?)"
    r"\s*-?\s*(?:business\s+|calendar\s+)?"
    r"(months?|years?|weeks?|days?)\b",
    re.IGNORECASE,
)

# Word-number duration: "two-year", "twenty four months", "one hundred eighty days".
_WORD_DUR_RE = re.compile(
    r"(?<![\w$-])"
    r"((?:" + _WORD_ALT + r")(?:[\s-]+(?:" + _WORD_ALT + r"))*)"
    r"\s*-?\s*(?:business\s+|calendar\s+)?"
    r"(months?|years?|weeks?|days?)\b",
    re.IGNORECASE,
)

# Restriction triggers for trigger-scoped duration extraction: a duration is
# only accepted as the restriction duration when it sits within
# _DURATION_WINDOW chars of one of these. The "for ... period" pattern is
# deliberately loose ("for a period of", "for an additional period of",
# "for a two-year period", "for a period commencing") because restriction
# durations in the gold sample are always introduced by such phrasing.
_DURATION_TRIGGERS = [
    re.compile(r"shall\s+not\s+compete", re.IGNORECASE),
    re.compile(r"covenant\s+not\s+to\s+compete", re.IGNORECASE),
    re.compile(r"(?:agrees?|agree)\s+not\s+to\s+compete", re.IGNORECASE),
    re.compile(r"\bnot\s+to\s+compete\b", re.IGNORECASE),
    re.compile(r"non[\s-]?compete\w*", re.IGNORECASE),  # non-compete, noncompete, noncompetition
    re.compile(r"restricted\s+period", re.IGNORECASE),
    re.compile(r"for\s+(?:an?\s+)?(?:[\w-]+\s+){0,2}period\b", re.IGNORECASE),
]
# Window choice: measured on the round-1 gold sample, the three stated
# restriction durations sit 6/11/54 chars from the nearest trigger; all
# false-positive clocks either have no trigger in the span at all (cure
# period, look-back window, ad-removal deadline) or are defeated by the
# prior-to guard below (lead-time windows). Any window >= 54 chars keeps
# all true positives; 120 gives ~2x margin while still requiring real
# proximity to a restriction trigger.
_DURATION_WINDOW = 120
# A duration followed by "prior to the expiration/termination ..." is a
# lead-time or look-back window anchored to that event ("three (3) years
# prior to the expiration of the Noncompetition Period"; "twelve (12)
# months prior to the Termination Date"), not the restriction duration.
_DURATION_PRIOR_TO_RE = re.compile(
    r"\bprior\s+to\s+the\s+(?:expiration|expiry|termination)\b", re.IGNORECASE)


def _words_to_num(text):
    """Convert an English number phrase ("twenty four", "one hundred") to a number."""
    total = 0.0
    current = 0.0
    for tok in re.split(r"[\s-]+", text.strip().lower()):
        v = _NUM_WORDS.get(tok)
        if v is None:
            return None
        if v == 100:
            current = (current or 1.0) * 100
        else:
            current += v
    return total + current


def extract_duration_months(text):
    """Extract a restriction duration from clause text, normalised to months.

    Trigger-scoped: only a duration sitting within 120 characters of a
    restriction trigger ("shall not compete", "covenant not to compete",
    "non-compete"/"noncompetition", "restricted period", "for a period of"
    and variants like "for an additional period of") is accepted. This
    excludes adjacent clocks of the wrong kind: look-back windows, cure
    periods, preparation lead-time windows, and ad-removal deadlines (the
    latter three additionally guarded by the prior-to and cure-context
    checks below).

    Matches: "24 months", "2 years", "twelve (12) months", "two-year",
    "one (1) year", "90 days", "18-month", "one hundred eighty days".
    Word numbers one..ninety (plus "hundred") are supported.
    Conversion: months*1, years*12, weeks/4.345, days/30.44.

    Returns the FIRST duration that passes the trigger and exclusion
    guards, or None when unfixable.

    Unfixable (returns None): no number+time-unit within 120 chars of a
    restriction trigger ("during the two-year term of this Agreement" with
    no non-compete trigger); a duration followed by "prior to the
    expiration/termination" (lead-time or look-back window); a duration in
    cure-period context ("failure to cure ... within thirty (30) days");
    bare section/page/clause numbers ("Section 12", "page 4 of 9"); dollar
    amounts ("$5") which never match a time unit; dates ("July 1, 2015")
    which carry no time unit; fractional words ("one-half year") which are
    not parsed; redacted durations ("[***] days"). Multiple durations in
    one span (e.g. initial term + renewal term) are ambiguous beyond the
    first acceptable one: only the first acceptable duration is returned,
    by design.
    """
    if not text:
        return None
    norm = _PAREN_NUM_RE.sub(r"\1", text)
    trig_spans = [(m.start(), m.end())
                  for rx in _DURATION_TRIGGERS for m in rx.finditer(norm)]

    def near_trigger(s, e):
        return any(ts - _DURATION_WINDOW <= s and e <= te + _DURATION_WINDOW
                   for ts, te in trig_spans)

    cands = []
    for m in _NUM_DUR_RE.finditer(norm):
        cands.append((m.start(), m.end(), m.group(1), m.group(2)))
    for m in _WORD_DUR_RE.finditer(norm):
        cands.append((m.start(), m.end(), m.group(1), m.group(2)))
    cands.sort(key=lambda c: (c[0], c[1]))

    for s, e, num_txt, unit in cands:
        if not near_trigger(s, e):
            continue
        if _DURATION_PRIOR_TO_RE.search(norm[e:e + 80]):
            continue
        if re.search(r"\bcure\b.{0,40}$", norm[max(0, s - 60):s],
                     re.IGNORECASE):
            continue
        try:
            n = float(num_txt)
        except ValueError:
            n = _words_to_num(num_txt)
        if n is not None:
            return n * _UNIT_TO_MONTHS[unit.lower()]
    return None


# ---------------------------------------------------------------------------
# Amount extraction
# ---------------------------------------------------------------------------

_MULTIPLIERS = {
    "thousand": 1e3, "million": 1e6, "billion": 1e9,
    "k": 1e3, "m": 1e6, "b": 1e9,
}

# "$1,000,000", "$500k", "$5 million"
_DOLLAR_RE = re.compile(
    r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(thousand|million|billion|[kmb])?\b",
    re.IGNORECASE,
)
# "USD 5 million", "US$ 10,000", "U.S. dollars 3 million"
_USD_RE = re.compile(
    r"\bU\.?\s?S\.?(?:\s?D\.?)?\s?\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?!\s*\d)"
    r"\s*(thousand|million|billion)?\b",
    re.IGNORECASE,
)
# "5 million dollars", "1.5 billion"
_MULT_WORD_RE = re.compile(
    r"(?<![\w$])(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(million|billion|thousand)"
    r"(?:\s*(?:U\.?\s?S\.?\s*)?dollars?)?\b",
    re.IGNORECASE,
)
# "five million dollars", "one hundred thousand dollars"
_WORD_AMOUNT_RE = re.compile(
    r"(?<![\w$-])"
    r"((?:" + _WORD_ALT + r")(?:[\s-]+(?:" + _WORD_ALT + r"))*)"
    r"\s*(million|billion|thousand)"
    r"\s*(?:U\.?\s?S\.?\s*)?dollars?\b",
    re.IGNORECASE,
)


def _num(s):
    return float(s.replace(",", ""))


def extract_amount_usd(text):
    """Extract a monetary amount from clause text, normalised to USD.

    Matches: "$1,000,000", "USD 5 million", "$500k", "five million dollars",
    "1.5 billion" (bare million/billion reads as USD in contract context).
    Multipliers: thousand/k = 1e3, million/m = 1e6, billion/b = 1e9.

    Returns the FIRST amount found, or None when unfixable.

    Unfixable (returns None): no currency marker and no magnitude word
    ("2015", "4.1.2", "Section 10" never match); dates ("July 1, 2015");
    clause/section numbers; redacted amounts ("[***]", "[ * ]"); purely
    qualitative caps ("uncapped", "unlimited liability", "sole remedy is
    termination") with no number; amounts in non-USD currencies written
    without a USD marker are not converted and return None rather than a
    wrong-currency number.
    """
    if not text:
        return None
    m = _DOLLAR_RE.search(text)
    if m:
        mult = _MULTIPLIERS.get((m.group(2) or "").lower(), 1.0)
        return _num(m.group(1)) * mult
    m = _USD_RE.search(text)
    if m:
        mult = _MULTIPLIERS.get((m.group(2) or "").lower(), 1.0)
        return _num(m.group(1)) * mult
    m = _WORD_AMOUNT_RE.search(text)
    if m:
        n = _words_to_num(m.group(1))
        if n is not None:
            return n * _MULTIPLIERS[m.group(2).lower()]
    m = _MULT_WORD_RE.search(text)
    if m:
        return _num(m.group(1)) * _MULTIPLIERS[m.group(2).lower()]
    return None


# ---------------------------------------------------------------------------
# Geographic scope extraction
# ---------------------------------------------------------------------------

WORLDWIDE_BREADTH = 200  # sentinel breadth for worldwide/global scope

_WORLDWIDE_RE = re.compile(
    r"\bworld-?wide\b|\bglobal(?:ly)?\b|throughout the world\b"
    r"|\bentire world\b|anywhere in the world\b",
    re.IGNORECASE,
)

_US_STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC",
}
_POSTAL_TO_STATE = {v: k for k, v in _US_STATES.items()}

_COUNTRIES = [
    "United States", "USA", "U.S.A.", "U.S.", "US",
    "United Kingdom", "UK", "England", "Scotland", "Wales",
    "Canada", "Australia", "Germany", "France", "Japan", "China",
    "India", "Singapore", "Ireland", "Netherlands", "Switzerland",
    "Brazil", "Mexico", "Spain", "Italy", "Hong Kong", "South Korea",
    "European Union", "EU", "UAE", "United Arab Emirates", "Israel",
    "New Zealand", "Sweden", "Norway", "Denmark", "Belgium", "Austria",
    "Poland", "Taiwan", "Malaysia", "Thailand", "Philippines", "Indonesia",
]
_US_ALIASES = {"USA", "U.S.A.", "U.S.", "US"}

_POSTAL_PAREN_RE = re.compile(r"\(\s*([A-Z]{2})\s*\)")
_POSTAL_AFTER_STATE_RE = re.compile(
    r"\bState\s+of\s+([A-Z]{2})\b|\bState\s+([A-Z]{2})\b"
)


def _canon_country(name):
    return "United States" if name in _US_ALIASES else name


def extract_geo_scope(text):
    """Extract geographic scope: {"jurisdictions": [...], "breadth": int}.

    Detects US states (full names, plus postal abbreviations in context such
    as "(MN)" or "State of MN"), countries from a built-in list, and
    worldwide markers ("worldwide", "global", "throughout the world",
    "anywhere in the world", "throughout the United States").
    ``breadth`` = count of distinct jurisdictions; a worldwide marker yields
    the ``WORLDWIDE_BREADTH`` sentinel (200), documented here.

    Unfixable (returns {"jurisdictions": [], "breadth": 0}): scope expressed
    only via defined terms never expanded in the span ("the Territory", "the
    Restricted Area"); no geographic reference at all; postal codes outside
    parentheses/state context are ignored (a bare "MN" mid-sentence is not
    trusted).
    """
    if not text:
        return {"jurisdictions": [], "breadth": 0}
    if _WORLDWIDE_RE.search(text):
        return {"jurisdictions": ["worldwide"], "breadth": WORLDWIDE_BREADTH}
    found = set()
    for state in _US_STATES:
        if re.search(r"\b" + re.escape(state) + r"\b", text, re.IGNORECASE):
            found.add(state)
    for code in _POSTAL_PAREN_RE.findall(text):
        if code in _POSTAL_TO_STATE:
            found.add(_POSTAL_TO_STATE[code])
    for m in _POSTAL_AFTER_STATE_RE.finditer(text):
        code = m.group(1) or m.group(2)
        if code in _POSTAL_TO_STATE:
            found.add(_POSTAL_TO_STATE[code])
    for country in _COUNTRIES:
        if country in _US_ALIASES:
            continue
        if re.search(r"\b" + re.escape(country) + r"\b", text):
            found.add(country)
    for alias in _US_ALIASES:
        if re.search(r"\b" + re.escape(alias) + r"\b", text):
            found.add("United States")
            break
    if re.search(r"throughout the United States\b", text, re.IGNORECASE):
        found.add("United States")
    jurisdictions = sorted(found)
    return {"jurisdictions": jurisdictions, "breadth": len(jurisdictions)}


# ---------------------------------------------------------------------------
# Indemnity mutuality
# ---------------------------------------------------------------------------

_MUTUAL_RES = [
    re.compile(r"\bmutual(?:ly)?\b.{0,40}\bindemnif\w*", re.IGNORECASE),
    re.compile(r"\bindemnif\w*.{0,40}\bmutual(?:ly)?\b", re.IGNORECASE),
    re.compile(r"\beach\s+(?:of\s+the\s+)?part(?:y|ies)\s+(?:shall|agrees?|hereby)\b.{0,60}\bindemnif\w*", re.IGNORECASE),
    re.compile(r"\bindemnif\w*.{0,40}\beach\s+other\b", re.IGNORECASE),
    re.compile(r"\beach\s+other\b.{0,40}\bindemnif\w*", re.IGNORECASE),
    re.compile(r"\breciprocal\b.{0,40}\bindemnif\w*", re.IGNORECASE),
    re.compile(r"\bindemnif\w*.{0,40}\breciprocal\b", re.IGNORECASE),
    re.compile(r"\beither\s+party\s+shall\s+indemnif\w*\s+the\s+other\b", re.IGNORECASE),
    re.compile(r"\beach\s+(?:of\s+the\s+)?part(?:y|ies)\b.{0,50}\bindemnif\w*", re.IGNORECASE),
    re.compile(r"\bindemnif\w*.{0,60}\bthe\s+other\s+part", re.IGNORECASE),
    re.compile(r"\bthe\s+other\s+part.{0,60}\bindemnif\w*", re.IGNORECASE),
]
_NEGATION_RE = re.compile(
    r"\b(?:no|neither)\s+(?:party|parties)\b.{0,60}\bindemnif\w*"
    r"|\b(?:shall|will)\s+not\s+be\s+indemnif\w*",
    re.IGNORECASE,
)
_UNILATERAL_RE = re.compile(
    r"\b(?:shall|will|hereby\s+agrees?\s+to|agrees?\s+to)\s+"
    r"(?:[a-z]+ly[,\s]+){0,2}"
    r"(?:(?:defend|reimburse|protect|hold|pay|and|or)[,\s]+){0,5}indemnif\w*",
    re.IGNORECASE,
)
# Passive voice: "shall/will be indemnified (and held harmless) (by X)",
# with an optional adverbial gap of up to 120 chars containing no sentence
# punctuation ("the Executive shall at all times during the Executive's
# employment by Bancorp or Bank, and after such employment, be indemnified
# by such entities"). The indemnified party is named one-directionally,
# so this classifies as unilateral.
_PASSIVE_RE = re.compile(
    r"\b(?:shall|will)\b[^.;()]{0,120}?\bbe\s+indemnif\w*",
    re.IGNORECASE,
)


def extract_indemnity_mutuality(text):
    """Classify an indemnity provision as "mutual", "unilateral", or None.

    mutual: "mutually indemnify", "mutual indemnification", "each party shall
    indemnify (the other)", "reciprocal", "indemnify each other".
    unilateral: a one-directional indemnify verb ("shall indemnify and hold
    harmless") naming the indemnifying party, with no bilateral signal;
    passive voice ("shall/will be indemnified (and held harmless) (by X)")
    likewise names the indemnified party one-directionally. An adverb may
    sit between the modal and the verb ("shall promptly indemnify").

    Unfixable (returns None): no indemnify-family language at all (e.g.
    definitions-only text like "Indemnitee means..."); negated indemnity
    ("no party shall indemnify"; "shall not be indemnified"); indemnity
    mentioned only inside a defined term or cross-reference with no
    operative verb; direction genuinely ambiguous.
    """
    if not text or "indemnif" not in text.lower():
        return None
    if _NEGATION_RE.search(text):
        return None
    for rx in _MUTUAL_RES:
        if rx.search(text):
            return "mutual"
    if _UNILATERAL_RE.search(text) or _PASSIVE_RE.search(text):
        return "unilateral"
    return None


# ---------------------------------------------------------------------------
# Notice days
# ---------------------------------------------------------------------------

_NOTICE_KW_RE = re.compile(r"notic\w*|notif\w*", re.IGNORECASE)
_TERMINATE_KW_RE = re.compile(r"terminat\w*", re.IGNORECASE)
_BANKRUPTCY_RE = re.compile(r"insolven|bankrupt|filing|receivership", re.IGNORECASE)

_NOTICE_DUR_RE = re.compile(
    r"(?<![\$\d])"
    r"(?<!\d\.)"
    r"(\d+(?:\.\d+)?)"
    r"\s*(?:business\s+|calendar\s+)?"
    r"(days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)

# Word-number duration: "thirty-day", "two weeks", "one hundred eighty days".
_WORD_NOTICE_DUR_RE = re.compile(
    r"(?<![\w$-])"
    r"((?:" + _WORD_ALT + r")(?:[\s-]+(?:" + _WORD_ALT + r"))*)"
    r"\s*-?\s*(?:business\s+|calendar\s+)?"
    r"(days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)

# Digit-then-word parenthetical ("6 (six) months") -> keep the digits.
# (Word-then-digit, "twelve (12) months", is already handled by _PAREN_NUM_RE.)
_PAREN_WORD_NUM_RE = re.compile(
    r"(?<![\w$])(\d+(?:\.\d+)?)"
    r"\s*\(\s*(?:" + _WORD_ALT + r")(?:[\s-]+(?:" + _WORD_ALT + r"))*\s*\)"
    r"(?=\s*(?:business\s+|calendar\s+)?(?:days?|weeks?|months?|years?)\b)",
    re.IGNORECASE,
)

_UNIT_TO_DAYS = {
    "day": 1.0, "days": 1.0,
    "week": 7.0, "weeks": 7.0,
    "month": _DAYS_PER_MONTH, "months": _DAYS_PER_MONTH,
    "year": 365.25, "years": 365.25,
}


def extract_notice_days(text):
    """Extract a notice/termination waiting period, normalised to days.

    Finds day/week/month/year counts that sit in notice/termination context
    (within ~100 chars of "notice"/"notify"/"terminate" language), e.g.
    "thirty (30) days' written notice", "at least 180 days prior written
    notice to terminate", "a thirty-day notice of written termination",
    "upon two weeks notice", "6 (six) months prior notice". Word-number
    durations ("thirty-day", "two weeks") and digit-then-word
    parentheticals ("6 (six) months") are parsed. Conversion: days*1,
    weeks*7, months*30.44, years*365.25.

    Returns the FIRST qualifying count, or None when unfixable.

    Unfixable (returns None): no count in notice context ("may terminate at
    any time" with no period; redacted "[**] days"); counts that belong to
    other clocks nearby (insolvency/bankruptcy filing windows such as
    "not dismissed within ninety (90) days of filing" are explicitly
    excluded); agreement lifetimes ("terminates 50 years from the date
    hereof") which are not waiting periods; renewal-term lengths
    ("renewed for one (1) year periods unless notice is given") which are
    term lengths, not notice periods. A bare "terminate" keyword is not
    enough on its own: the count must sit near notice language, or between
    terminate and notice language.
    """
    if not text:
        return None
    norm = _PAREN_WORD_NUM_RE.sub(r"\1", _PAREN_NUM_RE.sub(r"\1", text))
    notice_spans = [(m.start(), m.end()) for m in _NOTICE_KW_RE.finditer(norm)]
    term_spans = [(m.start(), m.end()) for m in _TERMINATE_KW_RE.finditer(norm)]

    def near_notice(s, e, radius=100):
        return any(ks - radius <= s and e <= ke + radius for ks, ke in notice_spans)

    def near_terminate(s, e, radius=60):
        return any(ks - radius <= s and e <= ke + radius for ks, ke in term_spans)

    cands = []
    for m in _NOTICE_DUR_RE.finditer(norm):
        cands.append((m.start(), m.end(), float(m.group(1)), m.group(2)))
    for m in _WORD_NOTICE_DUR_RE.finditer(norm):
        cands.append((m.start(), m.end(), _words_to_num(m.group(1)), m.group(2)))
    cands.sort(key=lambda c: (c[0], c[1]))

    for s, e, num, unit in cands:
        if num is None:
            continue
        if not (near_notice(s, e) or (near_terminate(s, e) and near_notice(s, e, 200))):
            continue
        window = norm[max(0, s - 40): e + 40]
        if _BANKRUPTCY_RE.search(window):
            continue
        # Renewal-term length, not a notice period.
        if re.search(r"\brenew(?:ed|al|s)?\b.{0,30}$", norm[max(0, s - 60): s], re.IGNORECASE) and \
           re.search(r"\bperiods?\b", norm[e: e + 20], re.IGNORECASE):
            continue
        return num * _UNIT_TO_DAYS[unit.lower()]
    return None


# ---------------------------------------------------------------------------
# Governing law
# ---------------------------------------------------------------------------

# Middle verbs between "governed by" and "the laws of": handles
# "governed by, and interpreted in accordance with, the law of X",
# "governed exclusively by, and determined in accordance with, the laws of X",
# "governed by and construed and enforced in accordance with the laws of X".
_GOV_BY = r"governed\s+(?:in\s+all\s+respects(?:,[^.;()]{0,120})?,?\s*)?(?:exclusively\s+)?by\s*,?\s*"
_GOV_MID = (
    r"(?:and\s+(?:construed|interpreted|determined)\s+"
    r"(?:and\s+enforced\s+)?(?:in\s+accordance\s+with|under)\s*,?\s*)?"
)
# Up to two stacked adjectives ("substantive internal laws of X",
# "general substantive laws of X").
_GOV_ADJ = r"(?:(?:internal|substantive|general)\s+){0,2}"
_GOV_LAWS_OF = (
    r"the\s+" + _GOV_ADJ + r"laws?\s+"
    r"(?:and\s+judicial\s+decisions\s+)?of\s+(?:the\s+)?"
)
_GOV_JUR = r"([A-Z][^.,;()]{2,100})"

_GOVLAW_RES = [
    re.compile(_GOV_BY + _GOV_MID + _GOV_LAWS_OF + _GOV_JUR, re.IGNORECASE),
    re.compile(
        r"(?:construed|interpreted)(?:\s+and\s+enforced)?\s+"
        r"(?:in\s+accordance\s+with|according\s+to)\s+"
        r"the\s+" + _GOV_ADJ + r"laws?\s+"
        r"(?:and\s+judicial\s+decisions\s+)?of\s+"
        r"(?:the\s+)?" + _GOV_JUR,
        re.IGNORECASE,
    ),
    re.compile(
        r"in\s+accordance\s+with\s+the\s+" + _GOV_ADJ + r"laws?\s+of\s+"
        r"(?:the\s+)?" + _GOV_JUR,
        re.IGNORECASE,
    ),
    re.compile(
        r"the\s+laws?\s+of\s+(?:the\s+)?([A-Z][^.,;()]{2,100}?)"
        r"\s*(?:\([^()]*\))?\s+shall\s+govern\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"governed by\s+(?:the\s+)?law\s+of\s+(?:the\s+)?" + _GOV_JUR,
        re.IGNORECASE,
    ),
    re.compile(
        r"governed by(?:,)?\s+"
        r"(?:and\s+(?:construed|interpreted)\s+in\s+accordance\s+with\s+)?"
        r"((?-i:[A-Z])[\w.'-]{1,60}"
        r"(?:\s+(?-i:[A-Z])[\w.'-]{1,60}){0,4})\s+laws?\b",
        re.IGNORECASE,
    ),
    # Conservative "reference to the laws of X" trigger (checked LAST, so
    # standard triggers take precedence): fires only when the span also
    # applies a governing verb (construed/interpreted/governed) to the SAME
    # jurisdiction X, i.e. the reference unambiguously denotes governing
    # law and is not a conflicts-of-law reference. Kept conservative on
    # purpose: a bare "with reference to the laws of X" with no governing
    # verb still returns None.
    re.compile(
        r"\bwith\s+reference\s+to\s+the\s+"
        r"(?:(?:internal|substantive|general)\s+){0,2}laws?\s+of\s+(?:the\s+)?"
        r"([A-Z][^.,;()]{2,100})"
        r"(?=[^.;()]{0,250}?\b(?:construed|interpreted|governed)\b"
        r"[^.;()]{0,250}?\1)",
        re.IGNORECASE,
    ),
]
_GOVLAW_TRIM_RE = re.compile(
    r"\s*\b(?:applicable|without reference|without regard|without giving effect|"
    r"excluding|irrespective|notwithstanding)\b.*$",
    re.IGNORECASE,
)
_GOVLAW_STOPWORDS = {
    "the", "a", "an", "internal", "substantive", "such", "said", "and",
}


def extract_governing_law(text):
    """Extract the governing-law jurisdiction from clause text.

    Matches "governed by the laws of X" (with up to two stacked adjectives
    such as "substantive internal"), "construed in accordance with /
    according to the laws of X", "governed by and construed ... the laws of
    X" (incl. ", and interpreted in accordance with X law" and "under the
    substantive laws of X" variants), "governed by English law"-style
    phrasing, and — conservatively — "contracted with reference to the
    laws of X" only when the span also applies a governing verb
    (construed/interpreted/governed) to the same jurisdiction X. Returns the
    jurisdiction string with a leading "the " stripped (e.g. "State of
    Texas").

    Unfixable (returns None): no governing-law trigger phrase (jurisdiction-
    only clauses like "exclusive jurisdiction of the English courts" are not
    governing law; a bare "with reference to the laws of X" with no
    governing verb is not trusted either); redacted jurisdiction ("laws of
    [***]"); the laws of a defined term ("laws of the Territory") which
    cannot be resolved from the span alone.
    """
    if not text:
        return None
    for rx in _GOVLAW_RES:
        m = rx.search(text)
        if m:
            jur = m.group(1).strip()
            jur = _GOVLAW_TRIM_RE.sub("", jur).strip()
            jur = re.sub(r"^(?:the|The)\s+", "", jur)
            jur = re.sub(r"\s+", " ", jur).strip(" ,;")
            if (jur and "[***]" not in jur and "[ * ]" not in jur
                    and jur.lower() not in _GOVLAW_STOPWORDS):
                return jur
    return None
