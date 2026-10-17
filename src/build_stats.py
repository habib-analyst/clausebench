"""Build ClauseBench dimension statistics from the vendored corpora.

Reads:
  CUAD:   ~/workspace/goals/github-portfolio-linkedin-presence-buildout/hidden_files/clausebench/data/CUAD_v1.json
  LEDGAR: .../data/ledgar/{train,validation,test}-00000-of-00001.parquet
          .../data/ledgar/label_names.json
Writes:
  ~/workspace/build/clausebench/data/percentiles.json   (vendored, shipped with app)
  ~/workspace/goals/.../hidden_files/clausebench/STATS_REPORT.md

Run: ~/workspace/build/clausebench/.venv/bin/python src/build_stats.py
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from dimensions import (  # noqa: E402
    extract_amount_usd,
    extract_duration_months,
    extract_geo_scope,
    extract_governing_law,
    extract_indemnity_mutuality,
    extract_notice_days,
)

HIDDEN = os.path.expanduser(
    "~/workspace/goals/github-portfolio-linkedin-presence-buildout"
    "/hidden_files/clausebench"
)
DATA = os.path.join(HIDDEN, "data")
OUT_DIR = os.path.expanduser("~/workspace/build/clausebench/data")

CUAD_Q_RE = re.compile(r'"([^"]+)"')


def load_cuad():
    """Return {category: [span_text, ...]} for the categories we need."""
    with open(os.path.join(DATA, "CUAD_v1.json")) as f:
        d = json.load(f)
    cats = {}
    for contract in d["data"]:
        for para in contract["paragraphs"]:
            for qa in para["qas"]:
                q = qa["question"].replace("\xa0", " ")
                m = CUAD_Q_RE.search(q)
                if not m:
                    continue
                cat = m.group(1)
                for a in qa["answers"]:
                    cats.setdefault(cat, []).append(a["text"])
    return cats


def load_ledgar():
    """Return {label_id: [text, ...]} across train/validation/test."""
    with open(os.path.join(DATA, "ledgar", "label_names.json")) as f:
        names = json.load(f)
    out = {}
    for split in ("train", "validation", "test"):
        t = pq.read_table(os.path.join(DATA, "ledgar", f"{split}-00000-of-00001.parquet"))
        labels = t.column("label").to_pylist()
        texts = t.column("text").to_pylist()
        for lab, text in zip(labels, texts):
            out.setdefault(int(lab), []).append(text)
    return out, names


def reason_duration(t):
    if re.search(r"\[\s*\*+\s*\]", t):
        return "redacted duration ([***])"
    if not re.search(r"\d", t) and not re.search(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty|thirty|hundred)\b",
        t, re.IGNORECASE,
    ):
        return "no numeric or word-number duration with a time unit"
    return "duration-like language but no parseable number+time-unit (e.g. 'during the term')"


def reason_amount(t):
    if re.search(r"\[\s*\*+\s*\]", t):
        return "redacted amount ([***]/[ * ])"
    if re.search(r"\buncapped\b|\bunlimited\b", t, re.IGNORECASE):
        return "qualitative cap only (uncapped/unlimited, no number)"
    return "no currency amount present (qualitative or remedy-only language)"


def reason_notice(t):
    if re.search(r"\[\s*\*+\s*\]", t):
        return "redacted day-count"
    if re.search(r"insolven|bankrupt|filing", t, re.IGNORECASE):
        return "only count belongs to a bankruptcy/insolvency clock, not notice"
    return "no day-count in notice/termination context"


def reason_geo(t):
    if re.search(r"\bTerritory\b|\bRestricted Area\b|\bRestricted Territory\b", t):
        return "scope via defined term ('Territory'/'Restricted Area'), never expanded in span"
    return "no named jurisdiction in span"


def reason_mutuality(t):
    if "indemnif" not in t.lower():
        return "no indemnify-family language (definition/cross-reference only)"
    if re.search(r"\b(?:no|neither)\s+(?:party|parties)\b", t, re.IGNORECASE):
        return "negated indemnity"
    return "indemnity language present but direction ambiguous"


def reason_govlaw(t):
    if re.search(r"\[\s*\*+\s*\]", t):
        return "redacted jurisdiction"
    if re.search(r"\bjurisdiction\b", t, re.IGNORECASE):
        return "jurisdiction/venue clause only, no governing-law trigger phrase"
    return "no governing-law trigger phrase"


def pcts(values):
    a = np.asarray(values, dtype=float)
    return {
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def run_numeric(name, unit, source, category, spans, extractor, reason_fn, geo=False):
    values, bad = [], []
    for t in spans:
        v = extractor(t)
        if geo:
            v = v["breadth"] if isinstance(v, dict) else v
            ok = v and v > 0
        else:
            ok = v is not None and not (isinstance(v, float) and math.isnan(v))
        if ok:
            values.append(round(float(v), 6))
        else:
            bad.append(t)
    values.sort()
    n = len(spans)
    return {
        "name": name,
        "values": values,
        "n_total": n,
        "n_extracted": len(values),
        "n_unfixable": len(bad),
        "pct": pcts(values) if values else None,
        "unit": unit,
        "source": source,
        "category": category,
        "bad_samples": [(t, reason_fn(t)) for t in bad[:5]],
    }


def run_categorical(name, source, category, spans, extractor, reason_fn, norm=None):
    counts, bad = {}, []
    for t in spans:
        v = extractor(t)
        if v is None:
            bad.append(t)
            continue
        key = norm(v) if norm else v
        counts[key] = counts.get(key, 0) + 1
    n = len(spans)
    return {
        "name": name,
        "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "n_total": n,
        "n_extracted": n - len(bad),
        "n_unfixable": len(bad),
        "source": source,
        "category": category,
        "bad_samples": [(t, reason_fn(t)) for t in bad[:5]],
    }


def norm_jurisdiction(s):
    s = re.sub(r"^(?:the|The)\s+", "", s.strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(
        r"\s+(?:without\s+regard\s+to.*|without\s+reference\s+to.*|"
        r"without\s+giving\s+effect\s+to.*|applicable\s+to.*)$",
        "", s, flags=re.IGNORECASE,
    ).strip(" ,;")
    return s


def merge_casefold(counts):
    """Merge jurisdiction counts case-insensitively, keeping the most common form."""
    groups = {}
    for k, c in counts.items():
        groups.setdefault(k.casefold(), []).append((k, c))
    merged = {}
    for forms in groups.values():
        forms.sort(key=lambda kc: -kc[1])
        merged[forms[0][0]] = sum(c for _, c in forms)
    return dict(sorted(merged.items(), key=lambda kv: -kv[1]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cuad = load_cuad()
    ledgar, names = load_ledgar()
    print("CUAD categories loaded:", sorted(cuad))
    print("LEDGAR label ids:", sorted(ledgar))

    nc = cuad.get("Non-Compete", [])
    cap = cuad.get("Cap On Liability", [])
    tconv = cuad.get("Termination For Convenience", [])
    npr = cuad.get("Notice Period To Terminate Renewal", [])
    gov_cuad = cuad.get("Governing Law", [])
    ind = ledgar.get(49, []) + ledgar.get(50, [])
    gov_led = ledgar.get(47, [])
    term = ledgar.get(88, [])
    notc = ledgar.get(65, [])
    print(f"spans: NC={len(nc)} cap={len(cap)} tconv={len(tconv)} npr={len(npr)} "
          f"gov_cuad={len(gov_cuad)} ind={len(ind)} gov_led={len(gov_led)} "
          f"term={len(term)} notc={len(notc)}")

    numeric = [
        run_numeric("duration_months", "months", "CUAD", "Non-Compete",
                    nc, extract_duration_months, reason_duration),
        run_numeric("geo_breadth", "jurisdictions", "CUAD", "Non-Compete",
                    nc, extract_geo_scope, reason_geo, geo=True),
        run_numeric("amount_usd", "USD", "CUAD", "Cap On Liability",
                    cap, extract_amount_usd, reason_amount),
        run_numeric("notice_days", "days", "CUAD+LEDGAR",
                    "Termination For Convenience; Notice Period To Terminate Renewal; "
                    "LEDGAR Terminations; LEDGAR Notices",
                    tconv + npr + term + notc, extract_notice_days, reason_notice),
    ]
    gov = run_categorical("governing_law", "CUAD+LEDGAR",
                        "CUAD Governing Law + LEDGAR Governing Laws (47), train+validation+test",
                        gov_cuad + gov_led, extract_governing_law,
                        reason_govlaw, norm=norm_jurisdiction)
    gov["counts"] = merge_casefold(gov["counts"])
    categorical = [
        run_categorical("indemnity_mutuality", "LEDGAR",
                        "Indemnifications (49) + Indemnity (50), train+validation+test",
                        ind, extract_indemnity_mutuality, reason_mutuality),
        gov,
    ]

    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "built_at": built_at,
        "corpus_bias": (
            "Values are measured from two corpora and describe those corpora, not the "
            "legal market: CUAD v1 spans are lawyer-highlighted answer fragments (510 US "
            "commercial contracts, EDGAR-sourced, English only); LEDGAR provisions are "
            "short standalone clauses from SEC-filed contracts (LexGLUE config). Both "
            "skew US-centric. 'Unfixable' means the extractor could not honestly derive "
            "a value from the span (redactions, defined terms, qualitative language), "
            "not that the clause lacks the concept. Worldwide geo scope uses breadth "
            "sentinel 200."
        ),
        "dimensions": {},
    }
    for r in numeric:
        payload["dimensions"][r["name"]] = {
            "values": r["values"],
            "n": r["n_extracted"],
            "n_total": r["n_total"],
            "unfixable": r["n_unfixable"],
            "unfixable_pct": round(100.0 * r["n_unfixable"] / r["n_total"], 2)
            if r["n_total"] else 0.0,
            "unit": r["unit"],
            "source": r["source"],
            "category": r["category"],
        }
    for r in categorical:
        payload["dimensions"][r["name"]] = {
            "counts": r["counts"],
            "n": r["n_extracted"],
            "n_total": r["n_total"],
            "unfixable": r["n_unfixable"],
            "unfixable_pct": round(100.0 * r["n_unfixable"] / r["n_total"], 2)
            if r["n_total"] else 0.0,
            "source": r["source"],
            "category": r["category"],
        }

    ppath = os.path.join(OUT_DIR, "percentiles.json")
    with open(ppath, "w") as f:
        json.dump(payload, f)
    size = os.path.getsize(ppath)
    print(f"wrote {ppath} ({size} bytes)")

    # ---- sanity: sortedness ----
    with open(ppath) as f:
        check = json.load(f)
    for name, dim in check["dimensions"].items():
        if "values" in dim:
            v = dim["values"]
            assert v == sorted(v), f"{name} not sorted"
            assert len(v) == dim["n"], f"{name} n mismatch"
    assert size < 2 * 1024 * 1024, "percentiles.json exceeds 2MB"
    print("sanity: all numeric arrays sorted, size < 2MB OK")

    # ---- STATS_REPORT.md ----
    lines = []
    A = lines.append
    A("# ClauseBench — Dimension Statistics Report")
    A("")
    A(f"Built {built_at} from CUAD v1 (510 contracts) and LEDGAR "
      f"(train 60k / validation 10k / test 10k). All numbers measured by "
      f"`src/dimensions.py` + `src/build_stats.py`; nothing fabricated.")
    A("")
    A("> " + payload["corpus_bias"])
    A("")
    A("## Summary")
    A("")
    A("| Dimension | Spans | Extracted | Unfixable | Unfixable % | p50 | Unit | Source |")
    A("|---|---|---|---|---|---|---|---|")
    for r in numeric + categorical:
        p50 = f"{r['pct']['p50']:.4g}" if r.get("pct") else "—"
        upct = round(100.0 * r["n_unfixable"] / r["n_total"], 1) if r["n_total"] else 0.0
        A(f"| {r['name']} | {r['n_total']} | {r['n_extracted']} | "
          f"{r['n_unfixable']} | {upct}% | {p50} | "
          f"{r.get('unit', 'categorical')} | {r['source']} |")
    A("")
    for r in numeric:
        A(f"## {r['name']}")
        A("")
        A(f"- Source: {r['source']} — {r['category']}")
        A(f"- Spans: {r['n_total']}; extracted: {r['n_extracted']}; "
          f"unfixable: {r['n_unfixable']} "
          f"({round(100.0 * r['n_unfixable'] / r['n_total'], 1)}%)")
        if r["pct"]:
            p = r["pct"]
            A(f"- p10={p['p10']:.4g} p25={p['p25']:.4g} p50={p['p50']:.4g} "
              f"p75={p['p75']:.4g} p90={p['p90']:.4g} "
              f"min={p['min']:.4g} max={p['max']:.4g} {r['unit']}")
        else:
            A("- No values extracted.")
        A("")
        A("Sample unfixable spans (5, with reason):")
        A("")
        for t, why in r["bad_samples"]:
            snippet = re.sub(r"\s+", " ", t)[:180]
            A(f'- "{snippet}" — {why}')
        A("")
    for r in categorical:
        A(f"## {r['name']}")
        A("")
        A(f"- Source: {r['source']} — {r['category']}")
        A(f"- Spans: {r['n_total']}; extracted: {r['n_extracted']}; "
          f"unfixable: {r['n_unfixable']} "
          f"({round(100.0 * r['n_unfixable'] / r['n_total'], 1)}%)")
        A("")
        if r["name"] == "indemnity_mutuality":
            A("Mutuality mix:")
            for k, c in r["counts"].items():
                A(f"- {k}: {c} ({round(100.0 * c / r['n_extracted'], 1)}%)")
        else:
            A("Top-20 jurisdictions:")
            for k, c in list(r["counts"].items())[:20]:
                A(f"- {k}: {c}")
            A(f"- (plus {len(r['counts']) - 20} further distinct jurisdictions)")
        A("")
        A("Sample unfixable spans (5, with reason):")
        A("")
        for t, why in r["bad_samples"]:
            snippet = re.sub(r"\s+", " ", t)[:180]
            A(f'- "{snippet}" — {why}')
        A("")
    rpath = os.path.join(HIDDEN, "STATS_REPORT.md")
    with open(rpath, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {rpath}")


if __name__ == "__main__":
    main()
