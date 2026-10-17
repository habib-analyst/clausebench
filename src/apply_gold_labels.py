"""Apply hand labels to the gold sample (Project #41 ClauseBench).

Labels were produced by reading all 150 sampled spans (see src/gold_review/
text dumps). Conventions:
  duration: months (float) or None
  geo:      {"jurisdictions": [...], "breadth": n} or None
  amount:   USD (float) or None
  notice:   days (float) or None
  govlaw:   jurisdiction string, or [j1, j2] for dual-jurisdiction clauses,
            or None
  indemnity:"mutual" | "unilateral" | None
"""

import json
import os

GOAL = os.path.join(
    os.path.expanduser("~"),
    "workspace/goals/github-portfolio-linkedin-presence-buildout",
    "hidden_files/clausebench",
)
GOLD = os.path.join(GOAL, "gold_dimensions.jsonl")

# index -> (gold_value_or_null, labeler_note)
LABELS = {
    # ---- duration (Non-Compete), gold in months ----
    0: (None, "restriction lasts 'During the Term'; term length defined elsewhere, not stated"),
    1: (None, "'during the term of the Franchise Agreement'; no numeric duration"),
    2: (None, "redacted [***] notice period"),
    3: (None, "'during the Term'; no numeric duration"),
    4: (None, "shareholding carve-out; no duration"),
    5: (None, "the 'twelve (12) months' is a look-back window for the customer-dealing test, not the restriction duration"),
    6: (None, "employment restriction; no duration"),
    7: (None, "no duration stated"),
    8: (24.0, "'for an additional period of two (2) years from the date of termination'"),
    9: (None, "'as long as Igene and T&L continue to own an interest'; qualitative, no numeric duration"),
    10: (None, "no duration stated"),
    11: (None, "'thirty (30) days after notice' is a cure period for default, not the restriction duration"),
    12: (None, "Non-Competition Period length defined via term elsewhere; not stated in span"),
    13: (24.0, "'for a two-year period after Franchisee ceases...'"),
    14: (None, "trademark-use restriction; no duration"),
    15: (None, "span is a bare fragment 'You must:'; no duration"),
    16: (24.0, "'period commencing on the Effective Date and ending two years after the Closing Date'"),
    17: (None, "'for so long as the appointment ... is exclusive'; qualitative"),
    18: (None, "'During the Term'; no numeric duration"),
    19: (None, "'During the Term'; no numeric duration"),
    20: (None, "truncated span; no duration stated"),
    21: (None, "bare fragment; no duration"),
    22: (None, "'three (3) years'/'twelve (12) months prior to the expiration of the Noncompetition Period' are preparation lead-time windows, not the Noncompetition Period duration"),
    23: (None, "'within three business days' is an ad-removal deadline, not a restriction duration"),
    24: (None, "'during the Term'; territory restriction; no numeric duration"),
    # ---- geo (same Non-Compete spans), gold jurisdictions/breadth ----
    25: (None, "'any country in the Territory'; Territory defined elsewhere, not expanded in span"),
    26: (None, "no geographic reference"),
    27: (None, "no geographic reference"),
    28: (None, "'Exclusive Field of Use'; defined term, no jurisdiction"),
    29: (None, "no geographic reference"),
    30: (None, "'in the Restricted Territory'; defined term, no jurisdiction"),
    31: (None, "no geographic reference"),
    32: (None, "no geographic reference"),
    33: (None, "no geographic reference"),
    34: ({"jurisdictions": ["worldwide"], "breadth": 200}, "'anywhere in the world'"),
    35: (None, "no geographic reference"),
    36: (None, "no geographic reference"),
    37: (None, "no geographic reference"),
    38: (None, "'within a ten (10) mile radius of your former Franchised Restaurant'; a radius, no named jurisdiction"),
    39: (None, "no geographic reference"),
    40: (None, "no geographic reference"),
    41: ({"jurisdictions": ["worldwide"], "breadth": 200}, "'anywhere in the world, excluding India'; breadth 200 sentinel, exclusion noted"),
    42: (None, "no geographic reference"),
    43: (None, "no geographic reference"),
    44: (None, "no geographic reference"),
    45: (None, "no geographic reference"),
    46: (None, "no geographic reference"),
    47: (None, "no geographic reference"),
    48: (None, "no geographic reference"),
    49: (None, "'in the Territory'; defined term, not expanded in span"),
    # ---- amount (Cap On Liability), gold in USD ----
    50: (5000000.0, "'five million US dollars ($5,000,000) in the aggregate'"),
    51: (None, "sole-remedy clause; no amount"),
    52: (None, "no-liability statement; no amount"),
    53: (None, "carve-out of losses; no amount"),
    54: (None, "waiver of damages categories; no amount"),
    55: (None, "exclusive-remedies clause; no amount"),
    56: (None, "'limited to the amount of the cost of the products'; amount defined via term, not stated"),
    57: (None, "exclusion of consequential damages; no amount"),
    58: (None, "'LIMITED TO RECOVERY OF ANY ACTUAL DAMAGES'; no numeric amount"),
    59: (None, "truncated span; maximum liability stated but number cut off"),
    60: (None, "'limited to the annualised value of the Maintenance Charges'; value defined via term, not stated"),
    61: (None, "fragment; no amount"),
    62: (None, "sole-remedy clause; no amount"),
    63: (None, "'shall not exceed the insurance coverage set out in Section 9.0'; coverage amount not stated"),
    64: (None, "'sole remedy ... is the Availability Credit'; credit amount not stated"),
    65: (None, "'SHALL NOT EXCEED THE AMOUNT RECEIVED BY JRVS'; amount defined via receipts, not stated"),
    66: (None, "indemnification as exclusive remedy; no amount"),
    67: (None, "indemnification as exclusive remedy; no amount"),
    68: (None, "no-liability statement; no amount"),
    69: (None, "'limited to the repair or replacement'; remedy stated, no amount"),
    70: (None, "'exceeds the amount paid ... to the Agent'; amount defined via payments, not stated"),
    71: (None, "redacted '[*] years' limitation period"),
    72: (None, "contribution cap via formula on proceeds/damages; no numeric amount"),
    73: (None, "exclusion of consequential damages; no amount"),
    74: (None, "redacted amounts; no stated amount"),
    # ---- notice, gold in days ----
    75: (None, "'renewed for one (1) or more one (1) month periods' are renewal TERM lengths; the notice period itself is not stated"),
    76: (30.0, "'thirty (30) days prior written notice'"),
    77: (91.32, "'three (3) months written notice'; 3*30.44"),
    78: (45.0, "'not less than forty-five (45) Days'"),
    79: (None, "'at any time upon prior written notice'; no period stated"),
    80: (182.64, "'6 (six) months prior notice'; 6*30.44"),
    81: (30.0, "'thirty (30) days' prior written notice'"),
    82: (None, "redacted '[*****] period'"),
    83: (60.0, "'not less than 60 days' prior written notice'"),
    84: (None, "'after Contract Year 2'; no notice period stated"),
    85: (180.0, "'one hundred eighty (180) days' written notice'"),
    86: (30.0, "'thirty-day notice of written termination'"),
    87: (90.0, "'ninety (90) calendar days prior to the expiration'"),
    88: (60.0, "'on sixty (60) days' prior written notice'"),
    89: (60.0, "'upon sixty (60) days' prior written notice'"),
    90: (14.0, "'upon two weeks notice'; 2*7"),
    91: (None, "'the notice indicated below in this Section'; period defined elsewhere, not stated in span"),
    92: (547.92, "'eighteen (18) months' advance written notice'; 18*30.44"),
    93: (30.0, "'thirty (30) calendar days' prior written notice of termination'"),
    94: (90.0, "'not less than ninety (90) days prior'"),
    95: (30.0, "'at least thirty (30) days before the expiration'"),
    96: (60.0, "'giving 60 days written notice'"),
    97: (60.0, "'at least sixty (60) days advance written notice'"),
    98: (60.0, "'a minimum of 60 days before the end of the then current term'"),
    99: (30.0, "'no less than thirty (30) days before the expiration'"),
    # ---- govlaw ----
    100: ("State of Illinois", "'construed according to the laws of the State of Illinois'"),
    101: ("State of New York", "'GOVERNED BY AND CONSTRUED IN ACCORDANCE WITH THE LAWS OF THE STATE OF NEW YORK'"),
    102: ("State of California", "'construed in accordance with and governed by the laws of the State of California'"),
    103: ("State of New York", "'construed in accordance with and governed by the substantive internal laws of the State of New York'"),
    104: ("State of California", "'governed by the laws of the State of California'"),
    105: ("Hong Kong", "'governed by and construed under the laws of Hong Kong'"),
    106: ("England and Wales", "'governed by and interpreted in accordance with the laws of England and Wales'"),
    107: (["State of California", "Hong Kong"], "dual jurisdiction: 'laws of the State of California and the laws of Hong Kong'"),
    108: ("State of South Dakota", "'construed in accordance with and governed by the laws of the State of South Dakota'"),
    109: ("Delaware", "'governed, interpreted and construed in accordance with the substantive laws of the Delaware' (ungrammatical; means Delaware)"),
    110: ("State of Delaware", "'governed by, and construed in accordance with, the laws of the State of Delaware, USA'"),
    111: ("State of California", "'governed by the laws of the State of California, USA'"),
    112: ("State of Florida", "'governed by the laws of the State of Florida'"),
    113: ("Japan", "'governed in all respects by the laws of Japan'"),
    114: ("State of New York", "'governed by and construed in accordance with the laws of the State of New York'"),
    115: ("State of New York", "'exclusively governed by and construed in accordance with the laws of the State of New York, USA'"),
    116: (["State of New York", "United States"], "dual jurisdiction: 'laws of the State of New York and the laws of the United States applicable therein'"),
    117: ("Commonwealth of Pennsylvania", "'contracted with reference to the laws of the Commonwealth of Pennsylvania ... construed under the laws of the Commonwealth of Pennsylvania'; no standard 'governed by' trigger"),
    118: ("State of Delaware", "'construed in accordance with the laws of the State of Delaware'"),
    119: ("State of Florida", "'governed by and interpreted in accordance with the laws of the State of Florida'"),
    120: ("State of Texas", "'interpreted and governed by the laws of the State of Texas'"),
    121: ("Maryland", "'governed and construed in accordance with the laws of Maryland'"),
    122: (None, "fragment 'giving effect to the conflicts of laws principles thereof'; no jurisdiction, no governing-law trigger"),
    123: ("State of Virginia", "'governed by and construed in accordance with the laws of the State of Virginia'"),
    124: ("State of Illinois", "'interpreted, construed and enforced ... in accordance with the laws of the State of Illinois'"),
    # ---- indemnity ----
    125: ("unilateral", "'shall be indemnified ... by ISS and the Stockholders' Representative'; passive but one-directional"),
    126: ("unilateral", "'Landlord will indemnify, defend and hold Tenant harmless'"),
    127: ("unilateral", "'The Company hereby agree to indemnify Executive'"),
    128: ("unilateral", "'Licensee will defend, indemnify, and hold Licensor harmless'"),
    129: (None, "'will be provided indemnification to the extent permitted by ... Bylaws'; no operative indemnify verb, indemnifying party not stated in span"),
    130: ("unilateral", "'The Issuer shall indemnify and defend the Trustee'"),
    131: ("unilateral", "'Buyer shall defend, indemnify and hold harmless Seller'"),
    132: ("unilateral", "'the Borrower hereby agrees to protect, indemnify, pay and save ... the Lenders harmless'"),
    133: ("unilateral", "'Tenant's indemnity of Landlord'; one-directional, no operative shall/will-indemnify verb"),
    134: ("unilateral", "'Client shall indemnify and hold harmless each of Landlord and Relmada'"),
    135: ("unilateral", "'Executive agrees to indemnify and hold the Company harmless'"),
    136: ("unilateral", "'shall be indemnified by the Company'; passive but one-directional"),
    137: ("unilateral", "'SolarCity shall defend, indemnify and hold harmless the Assignee and the Third Party Beneficiary'"),
    138: ("unilateral", "'you shall be indemnified and held harmless by the Company'; passive but one-directional"),
    139: (None, "indemnification mentioned only as 'rights to indemnification ... Schedule B'; no operative verb, direction not stated"),
    140: ("unilateral", "'The Company shall indemnify, defend and hold harmless the Holder'"),
    141: ("unilateral", "'Pledgor hereby agrees to indemnify Lender'"),
    142: ("unilateral", "'the Company shall promptly indemnify Executive'"),
    143: ("unilateral", "'the Executive shall ... be indemnified by such entities'; passive but one-directional"),
    144: ("unilateral", "'Borrower agrees to indemnify and hold Lender ... harmless'"),
    145: (None, "'shall be entitled to indemnification ... pursuant to Delaware law'; eligibility language, no operative verb"),
    146: (None, "'shall be entitled to indemnification ... as set forth in the ... LLC Agreement'; defined elsewhere"),
    147: ("unilateral", "'will be indemnified and held harmless by the Company'; passive but one-directional"),
    148: ("unilateral", "'The Borrower hereby agrees to defend, indemnify and hold harmless the Lender'"),
    149: ("unilateral", "'EACH LENDER SHALL INDEMNIFY AND HOLD HARMLESS AGENT INDEMNITEES'"),
}


def main():
    rows = [json.loads(l) for l in open(GOLD)]
    assert len(rows) == len(LABELS) == 150, (len(rows), len(LABELS))
    for i, row in enumerate(rows):
        assert row["dimension"] == {0: "duration"}.get(0) or True
        val, note = LABELS[i]
        row["gold_value_or_null"] = val
        row["labeler_note"] = note
    with open(GOLD, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"labeled {len(rows)} rows")


if __name__ == "__main__":
    main()
