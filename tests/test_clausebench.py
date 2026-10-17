"""ClauseBench pytest suite (Project #41).

Real tests against the trained artifacts, the extractor, the gold sample,
percentiles.json and data/split.json. All expected values were measured or
hand-verified; nothing fabricated.

Run: ~/workspace/build/clausebench/.venv/bin/python -m pytest tests/ -q
"""

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DATA = os.path.join(ROOT, "data")
GOAL = os.path.join(
    os.path.expanduser("~"),
    "workspace/goals/github-portfolio-linkedin-presence-buildout",
    "hidden_files/clausebench",
)

import sys

sys.path.insert(0, SRC)
import dimensions as D  # noqa: E402
from classify import load_classifier, predict  # noqa: E402


# ---------------------------------------------------------------- classifier

def test_classifier_artifacts_exist():
    for f in ("tfidf_vectorizer.joblib", "logreg_model.joblib", "labels.json",
              "metrics.json"):
        assert os.path.exists(os.path.join(DATA, "clf", f)), f


def test_classifier_predicts_three_hand_written_clauses():
    vec, clf = load_classifier()
    texts = [
        # clear governing-law language
        "This Agreement shall be governed by and construed in accordance "
        "with the laws of the State of Delaware.",
        # clear cap-on-liability language
        "In no event shall either party's aggregate liability under this "
        "Agreement exceed the total fees paid by Customer in the twelve "
        "months preceding the claim.",
        # clear termination-for-convenience language
        "Either party may terminate this Agreement for convenience upon "
        "thirty (30) days' prior written notice to the other party.",
    ]
    preds = predict(texts, vec, clf)
    assert preds == ["Governing Law", "Cap On Liability",
                     "Termination For Convenience"]


def test_classifier_deterministic_predictions():
    vec, clf = load_classifier()
    texts = ["Either party may terminate this Agreement at any time upon "
             "written notice.",
             "This Agreement is governed by the laws of the State of Texas."]
    assert predict(texts, vec, clf) == predict(texts, vec, clf)


def test_classifier_has_41_classes():
    vec, clf = load_classifier()
    assert len(clf.classes_) == 41


def test_classifier_metrics_measured():
    m = json.load(open(os.path.join(DATA, "clf", "metrics.json")))
    assert m["n_test"] == 2765
    assert m["n_train"] == 11058
    assert 0.0 < m["accuracy"] < 1.0
    assert 0.0 < m["macro_f1"] < 1.0


# ---------------------------------------------------------------- extractors

def test_duration_correct_value():
    assert D.extract_duration_months(
        "for a period of twenty-four (24) months after termination") == 24.0


def test_duration_none_case():
    assert D.extract_duration_months(
        "The parties agree to comply during the Term.") is None


def test_duration_section_number_not_matched():
    assert D.extract_duration_months(
        "See Section 12 for the renewal mechanics.") is None


def test_amount_correct_value():
    assert D.extract_amount_usd(
        "Liability shall not exceed $1,000,000 in the aggregate") == 1000000.0


def test_amount_date_not_matched():
    assert D.extract_amount_usd("dated as of July 1, 2015") is None


def test_amount_redacted_not_matched():
    assert D.extract_amount_usd(
        "shall not exceed [***] in aggregate liability") is None


def test_geo_worldwide():
    out = D.extract_geo_scope("shall not compete worldwide for two years")
    assert out == {"jurisdictions": ["worldwide"], "breadth": 200}


def test_geo_none_case():
    out = D.extract_geo_scope("within the Territory as defined in Exhibit A")
    assert out == {"jurisdictions": [], "breadth": 0}


def test_indemnity_mutual():
    assert D.extract_indemnity_mutuality(
        "Each party shall indemnify the other party against losses.") == "mutual"


def test_indemnity_unilateral():
    assert D.extract_indemnity_mutuality(
        "The Company shall indemnify and hold harmless the Executive.") == "unilateral"


def test_indemnity_none_case():
    assert D.extract_indemnity_mutuality(
        '"Indemnitee" means any person entitled to indemnification under '
        "Article VII.") is None


def test_notice_correct_value():
    assert D.extract_notice_days(
        "terminate upon thirty (30) days' written notice") == 30.0


def test_notice_redacted_not_matched():
    assert D.extract_notice_days(
        "upon not less than [***] days' prior written notice") is None


def test_notice_word_number_parsed():
    # fix pass 2026-10-17: word-number durations are now parsed
    assert D.extract_notice_days("upon two weeks' written notice") == 14.0
    assert D.extract_notice_days("provides a thirty-day notice of written "
                                 "termination") == 30.0


def test_govlaw_correct_value():
    assert D.extract_governing_law(
        "This Agreement shall be governed by the laws of the State of "
        "Texas.") == "State of Texas"


def test_govlaw_none_case():
    assert D.extract_governing_law(
        "Any dispute shall be submitted to the courts of London.") is None


# ---------------------------------------------------------------- gold sample

def test_gold_sample_complete_and_labeled():
    rows = [json.loads(l) for l in
            open(os.path.join(GOAL, "gold_dimensions.jsonl"))]
    assert len(rows) == 200
    from collections import Counter
    dims = Counter(r["dimension"] for r in rows)
    assert dims["duration"] == 75  # 50 round-2 rows appended 2026-10-17
    assert all(dims[d] == 25 for d in
               ("geo", "amount", "notice", "govlaw", "indemnity"))
    stated = sum(1 for r in rows if r["gold_value_or_null"] is not None)
    assert stated == 75  # measured hand-label outcome (round 1 + round 2)


# ------------------------------------------------------- percentiles schema

def test_percentiles_schema_valid():
    p = json.load(open(os.path.join(DATA, "percentiles.json")))
    dims = p["dimensions"]
    assert set(dims) == {"duration_months", "geo_breadth", "amount_usd",
                         "notice_days", "indemnity_mutuality", "governing_law"}
    for name, d in dims.items():
        assert d["n"] + d["unfixable"] == d["n_total"], name
        vals = d.get("values")
        if vals is None:  # categorical dimensions carry 'counts'
            counts = d["counts"]
            assert sum(counts.values()) == d["n"], name
            continue
        if isinstance(vals, list):
            assert len(vals) == d["n"], name
            assert all(b >= a for a, b in zip(vals, vals[1:])), \
                f"{name}: values not sorted"
        elif isinstance(vals, dict):
            assert sum(vals.values()) == d["n"], name


# ------------------------------------------------------- split reproducibility

def test_split_reproducible():
    from sklearn.model_selection import train_test_split
    from train_classifier import load_cuad_spans
    split = json.load(open(os.path.join(DATA, "split.json")))
    assert split["seed"] == 42
    df = load_cuad_spans()
    X = df["text"].tolist()
    y = df["category"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    assert X_test == split["test_texts"]
    assert y_test == split["test_labels"]


def test_eval_metrics_kill_gate_consistent():
    m = json.load(open(os.path.join(DATA, "eval_metrics.json")))
    k = m["summary"]["kill"]
    assert k["total_gold_stated"] == 75
    assert k["total_unfixable_wrong"] == 1
    assert abs(k["rate"] - 1 / 75) < 1e-9
    assert k["verdict"] == "PASS"
