"""ClauseBench gold-sample evaluation (Project #41).

Runs src/dimensions.py extractors over the hand-labeled gold sample and
computes, per dimension:
  - extraction precision  = correct / attempted
      (attempted = extractor returned non-None; for geo, breadth > 0)
  - not-stated rate       = gold null / n   (corpus reality, not failure)
  - unfixable+wrong-on-stated = (gold stated but extractor None-or-wrong)
      / gold-stated n    <-- the kill-gate metric
Overall verdict: micro-average of unfixable+wrong over all gold-stated
spans; > 30% -> FAIL.

Correctness rules:
  numerics (duration_months, amount_usd, notice_days): |ext - gold|/gold <= 5%
  govlaw: normalized exact match; for dual-jurisdiction golds a list is
      accepted and the extraction counts correct when it contains (or
      exactly equals) one gold jurisdiction (documented tolerance)
  indemnity: exact 'mutual'/'unilateral' match
  geo: breadth exact match (documented); jurisdictions-set agreement reported
      separately

Run: .venv/bin/python src/eval.py  -> prints tables, writes data/eval_metrics.json
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dimensions as D  # noqa: E402

GOAL = os.path.join(
    os.path.expanduser("~"),
    "workspace/goals/github-portfolio-linkedin-presence-buildout",
    "hidden_files/clausebench",
)
GOLD = os.path.join(GOAL, "gold_dimensions.jsonl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "eval_metrics.json")
REL_TOL = 0.05
KILL_THRESHOLD = 0.30


def _norm_jur(s):
    s = re.sub(r"^(?:the|The)\s+", "", s.strip())
    return re.sub(r"\s+", " ", s).strip().lower()


def _geo_out(r):
    d = r if isinstance(r, dict) else {}
    return d.get("jurisdictions", []) or [], int(d.get("breadth", 0) or 0)


def evaluate(rows):
    results = []
    for r in rows:
        dim, gold = r["dimension"], r["gold_value_or_null"]
        txt = r["text"]
        if dim == "duration":
            ext = D.extract_duration_months(txt)
        elif dim == "geo":
            ext = D.extract_geo_scope(txt)
        elif dim == "amount":
            ext = D.extract_amount_usd(txt)
        elif dim == "notice":
            ext = D.extract_notice_days(txt)
        elif dim == "govlaw":
            ext = D.extract_governing_law(txt)
        elif dim == "indemnity":
            ext = D.extract_indemnity_mutuality(txt)
        else:
            raise ValueError(dim)

        if dim == "geo":
            jur, breadth = _geo_out(ext)
            attempted = breadth > 0
            if gold is None:
                correct = None
            else:
                correct = (breadth == gold["breadth"]) and (
                    sorted(jur) == sorted(gold["jurisdictions"]))
            results.append({
                "span_id": r["span_id"], "dimension": dim, "gold": gold,
                "extracted": {"jurisdictions": jur, "breadth": breadth},
                "attempted": attempted, "correct": correct,
            })
        else:
            attempted = ext is not None
            if gold is None:
                correct = None
            elif ext is None:
                correct = False  # gold stated but extractor returned None: a miss
            elif dim in ("duration", "amount", "notice"):
                correct = abs(ext - gold) / abs(gold) <= REL_TOL
            elif dim == "govlaw":
                ne = _norm_jur(ext)
                golds = [_norm_jur(g) for g in (gold if isinstance(gold, list) else [gold])]
                correct = any(ne == g or g in ne for g in golds)
            elif dim == "indemnity":
                correct = ext == gold
            results.append({
                "span_id": r["span_id"], "dimension": dim, "gold": gold,
                "extracted": ext, "attempted": attempted, "correct": correct,
            })
    return results


def summarize(results):
    dims = {}
    tot_stated = tot_bad = 0
    for dim in ["duration", "geo", "amount", "notice", "govlaw", "indemnity"]:
        rs = [x for x in results if x["dimension"] == dim]
        n = len(rs)
        stated = [x for x in rs if x["gold"] is not None]
        unstated = n - len(stated)
        attempted = [x for x in rs if x["attempted"]]
        correct = [x for x in attempted if x["correct"]]
        bad = [x for x in stated if (not x["attempted"]) or (not x["correct"])]
        tot_stated += len(stated)
        tot_bad += len(bad)
        dims[dim] = {
            "n": n,
            "n_stated": len(stated),
            "not_stated_rate": unstated / n,
            "attempted": len(attempted),
            "correct": len(correct),
            "precision": (len(correct) / len(attempted)) if attempted else None,
            "unfixable_wrong_on_stated": len(bad) / len(stated) if stated else None,
            "unfixable_wrong_n": len(bad),
        }
    overall = tot_bad / tot_stated if tot_stated else None
    return {
        "dimensions": dims,
        "kill": {
            "total_gold_stated": tot_stated,
            "total_unfixable_wrong": tot_bad,
            "rate": overall,
            "threshold": KILL_THRESHOLD,
            "verdict": "FAIL" if (overall is not None and overall > KILL_THRESHOLD) else "PASS",
        },
    }


def main():
    rows = [json.loads(l) for l in open(GOLD)]
    results = evaluate(rows)
    summary = summarize(results)
    with open(OUT, "w") as f:
        json.dump({"per_row": results, "summary": summary}, f, indent=2)

    print(f"{'dim':<10}{'n':>4}{'stated':>8}{'notst%':>8}{'att':>5}{'corr':>6}{'prec':>8}{'uw/st':>8}")
    for dim, s in summary["dimensions"].items():
        prec = f"{s['precision']:.3f}" if s["precision"] is not None else "  n/a"
        uw = f"{s['unfixable_wrong_on_stated']:.3f}" if s["unfixable_wrong_on_stated"] is not None else "  n/a"
        print(f"{dim:<10}{s['n']:>4}{s['n_stated']:>8}{s['not_stated_rate']:>7.1%}"
              f"{s['attempted']:>5}{s['correct']:>6}{prec:>8}{uw:>8}")
    k = summary["kill"]
    print(f"\nKILL GATE: {k['total_unfixable_wrong']}/{k['total_gold_stated']} = "
          f"{k['rate']:.3f} vs threshold {k['threshold']} -> {k['verdict']}")

    print("\nExtractor failures on gold-stated spans (5 examples):")
    shown = 0
    for x in results:
        if x["gold"] is not None and ((not x["attempted"]) or (not x["correct"])):
            print(f"  [{x['dimension']}] {x['span_id'][:80]}")
            print(f"    gold={x['gold']!r} extracted={x['extracted']!r}")
            shown += 1
            if shown >= 5:
                break
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
