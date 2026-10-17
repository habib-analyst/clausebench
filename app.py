"""ClauseBench: "Is your clause normal?" — Gradio demo app (Project #41).

Paste one contract clause -> a conservative keyword router maps it to a
dimension extractor -> an honest benchmark card computed from precomputed
corpus statistics (data/percentiles.json).

No network at runtime. CPU only. Percentile tables are loaded once at module
level. No model weights ship with the app: the 41-class TF-IDF+logreg clause
classifier (held-out accuracy 0.7222, n=2765, macro-F1 0.5046) lives in
src/train_classifier.py as the research artifact with its honest eval in
hidden_files/clausebench/EVAL_REPORT.md; its 18MB weights are deliberately
excluded from the deployable package (text-only repo rule), so the deployed
router is keyword-based and conservative: it says "not benchmarked" rather
than guessing. The demo cards are identical either way.

Measured facts (see README.md): extractor kill gate 1/75 = 0.013 vs 0.30 ->
PASS. Percentile cards ship for notice_days, governing_law,
indemnity_mutuality only; duration/amount/geo cards are dropped (corpus too
thin) and get honest "not stated" / value-only cards.
"""

import base64
import io
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")  # headless-safe before pyplot import
import matplotlib.pyplot as plt
import gradio as gr

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))

from dimensions import (  # noqa: E402
    extract_amount_usd,
    extract_duration_months,
    extract_governing_law,
    extract_indemnity_mutuality,
    extract_notice_days,
)

# --------------------------------------------------------------------------
# Module-level loads (once; no network)
# --------------------------------------------------------------------------
with open(os.path.join(BASE, "data", "percentiles.json")) as f:
    PCT = json.load(f)["dimensions"]

# --------------------------------------------------------------------------
# Routing: conservative keyword router (no model weights ship with the app)
# --------------------------------------------------------------------------
# The research classifier (src/train_classifier.py, 0.7222 held-out) is not
# deployed: its 18MB weights violate the text-only repo rule. This router is
# deliberately conservative -- unknown phrasing -> "not benchmarked", never a
# guessed percentile card.
_ROUTE_PATTERNS = [
    (re.compile(r"indemnif\w*|hold harmless", re.IGNORECASE),
     "(indemnification keywords)", "indemnity_mutuality"),
    (re.compile(r"governed by (the|and)|construed in accordance with|laws of the State",
                re.IGNORECASE),
     "Governing Law", "governing_law"),
    # Non-compete BEFORE terminat*: non-compete clauses often mention
    # "termination" ("for 24 months after termination").
    (re.compile(r"non[\s-]?compete|not (?:to )?compete|noncompetition|covenant not to",
                re.IGNORECASE),
     "Non-Compete", "duration_months"),      # dropped card: value-only fallback
    (re.compile(r"terminat\w*", re.IGNORECASE),
     "Termination / Notice", "notice_days"),
    (re.compile(r"cap on liabilit|liability.{0,40}(cap|limited to|shall not exceed)",
                re.IGNORECASE),
     "Cap On Liability", "amount_usd"),      # dropped card: value-only fallback
]

TYPE_TO_DIM = {
    "Termination / Notice": "notice_days",
    "Governing Law": "governing_law",
    "(indemnification keywords)": "indemnity_mutuality",
    "Non-Compete": "duration_months",
    "Cap On Liability": "amount_usd",
}

BIAS_NOTE = (
    "Corpus bias: CUAD = 510 US commercial contracts, EDGAR-sourced, English "
    "only; LEDGAR = SEC-filed provisions. US-centric; not representative of "
    "non-US or consumer contracts."
)
DISCLAIMER = "For information only. Not legal advice."

EXAMPLES = {
    "termination": (
        "Either party may terminate this Agreement upon thirty (30) days' "
        "prior written notice to the other party."
    ),
    "governing law": (
        "This Agreement shall be governed by and construed in accordance "
        "with the laws of the State of Delaware."
    ),
    "indemnity": (
        "Each party shall indemnify, defend and hold harmless the other "
        "party from and against any third-party claims."
    ),
}

CARD_CSS = (
    "font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;"
    "max-width: 720px; border: 1px solid #d7dce3; border-radius: 12px;"
    "padding: 22px 24px; margin: 12px 0; background: #ffffff;"
)
HEADLINE_CSS = "font-size: 19px; font-weight: 700; color: #1a2332; margin: 0 0 10px 0;"
SUB_CSS = "font-size: 14px; color: #4b5563; margin: 8px 0;"
FINE_CSS = "font-size: 12px; color: #6b7280; margin: 10px 0 0 0;"
DISC_CSS = "font-size: 12px; color: #9ca3af; font-style: italic; margin: 6px 0 0 0;"


def _card(headline, body_html, fine_print):
    return (
        f"<div style=\"{CARD_CSS}\">"
        f"<p style=\"{HEADLINE_CSS}\">{headline}</p>"
        f"{body_html}"
        f"<p style=\"{FINE_CSS}\">{fine_print}</p>"
        f"<p style=\"{FINE_CSS}\">{BIAS_NOTE}</p>"
        f"<p style=\"{DISC_CSS}\">{DISCLAIMER}</p>"
        "</div>"
    )


def _fig_to_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def notice_sparkline(values, user_v):
    fig, ax = plt.subplots(figsize=(4.2, 1.7))
    ax.hist(values, bins=10, color="#93b4d8", edgecolor="#ffffff")
    ax.axvline(user_v, color="#1d4ed8", linewidth=2.5)
    ax.set_xlabel("Notice days (corpus)", fontsize=8)
    ax.set_ylabel("Clauses", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.6)
    return _fig_to_uri(fig)


def govlaw_bars(top5, user_jur):
    labels = [k.replace("State of ", "") for k, _ in top5]
    counts = [c for _, c in top5]
    colors = ["#1d4ed8" if k == user_jur else "#93b4d8" for k, _ in top5]
    fig, ax = plt.subplots(figsize=(4.4, 1.9))
    ax.barh(labels[::-1], counts[::-1], color=colors[::-1], edgecolor="#ffffff")
    ax.set_xlabel("Provisions", fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.6)
    return _fig_to_uri(fig)


def _fmt_num(v):
    return str(int(v)) if v == int(v) else f"{v:.1f}"


# --------------------------------------------------------------------------
# Dimension cards
# --------------------------------------------------------------------------
def card_notice_days(text):
    d = PCT["notice_days"]
    v = extract_notice_days(text)
    if v is None:
        return _card(
            "No notice period stated.",
            f"<p style=\"{SUB_CSS}\">This clause does not state a notice "
            "period; no percentile computed.</p>",
            f"Sample: {d['n']:,} comparable notice clauses "
            f"({d['source']}).",
        )
    vals = d["values"]
    n = d["n"]
    below = sum(1 for x in vals if x < v)
    pct = 100.0 * below / n
    headline = (
        f"Your {_fmt_num(v)}-day termination notice is longer than "
        f"{pct:.1f}% of {n:,} comparable notice clauses."
    )
    body = (
        f"<img src=\"{notice_sparkline(vals, v)}\" "
        "style=\"max-width:100%; display:block; margin:6px 0;\"/>"
    )
    fine = (
        f"Sample: n={n:,} comparable notice clauses with a stated notice "
        f"period ({d['source']}). Vertical line marks your clause."
    )
    return _card(headline, body, fine)


def card_governing_law(text):
    d = PCT["governing_law"]
    jur = extract_governing_law(text)
    if jur is None:
        return _card(
            "No governing law stated.",
            f"<p style=\"{SUB_CSS}\">This clause does not state a governing "
            "law; no share computed.</p>",
            f"Sample: {d['n']:,} comparable provisions ({d['source']}).",
        )
    counts = d["counts"]
    n = d["n"]
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    rank = next(i for i, (k, _) in enumerate(ranked, 1) if k == jur)
    top = ranked[0][0]
    share = 100.0 * counts[jur] / n
    jur_short = jur.replace("State of ", "")
    top_short = top.replace("State of ", "")
    if rank == 1:
        rank_txt = f"rank #1 of {len(ranked):,} distinct jurisdictions"
    else:
        rank_txt = f"rank #{rank} after {top_short}"
    headline = (
        f"{jur_short} is the governing law in {share:.1f}% of {n:,} "
        f"comparable provisions ({rank_txt})."
    )
    body = (
        f"<img src=\"{govlaw_bars(ranked[:5], jur)}\" "
        "style=\"max-width:100%; display:block; margin:6px 0;\"/>"
    )
    fine = (
        f"Sample: n={n:,} comparable provisions with an extracted "
        f"governing law ({d['source']})."
    )
    return _card(headline, body, fine)


def card_indemnity_mutuality(text):
    d = PCT["indemnity_mutuality"]
    m = extract_indemnity_mutuality(text)
    if m is None:
        return _card(
            "No indemnity direction stated.",
            f"<p style=\"{SUB_CSS}\">This clause does not state indemnity "
            "mutuality; no share computed.</p>",
            f"Sample: {d['n']:,} comparable indemnification provisions "
            f"({d['source']}).",
        )
    counts = d["counts"]
    n = d["n"]
    other = "unilateral" if m == "mutual" else "mutual"
    share = 100.0 * counts.get(other, 0) / n
    headline = (
        f"Your clause is {m}. {share:.1f}% of {n:,} comparable "
        f"indemnification provisions are {other}."
    )
    body = (
        f"<p style=\"{SUB_CSS}\">Corpus mix: unilateral "
        f"{counts.get('unilateral', 0):,} / mutual {counts.get('mutual', 0):,} "
        f"(n={n:,}).</p>"
    )
    fine = (
        f"Sample: n={n:,} comparable indemnification provisions "
        f"({d['source']}). Note: the gold sample contained zero mutual "
        "examples, so mutual-class precision is unmeasured."
    )
    return _card(headline, body, fine)


def card_dropped_dimension(name, text, value_txt, noun):
    d = PCT[name]
    art = "an" if noun == "amount" else "a"
    if value_txt is None:
        return _card(
            f"No {noun} stated.",
            f"<p style=\"{SUB_CSS}\">This clause does not state {art} {noun}; "
            "no percentile computed.</p>",
            f"Sample: {d['n_total']:,} comparable clauses ({d['source']}).",
        )
    return _card(
        f"Extracted {noun}: {value_txt}.",
        f"<p style=\"{SUB_CSS}\">The corpus is too thin for an honest "
        f"percentile: only {d['n']:,} of {d['n_total']:,} comparable clauses "
        f"state {art} {noun}.</p>",
        f"Sample: n={d['n']:,} with a stated {noun} of "
        f"{d['n_total']:,} comparable clauses ({d['source']}).",
    )


def card_unsupported(clause_type):
    return _card(
        "Clause type not benchmarked.",
        f"<p style=\"{SUB_CSS}\">ClauseBench does not benchmark "
        f"<b>{clause_type}</b> clauses. Supported benchmarks: termination "
        "notice, governing law, indemnification mutuality.</p>",
        "CUAD spans across 41 clause types were classified; only supported "
        "types get percentile cards.",
    )


def card_empty():
    return _card(
        "Paste a clause to benchmark it.",
        f"<p style=\"{SUB_CSS}\">One contract clause at a time; try one of "
        "the examples above.</p>",
        "Nothing is sent anywhere; everything runs locally on CPU.",
    )


DIM_CARD = {
    "notice_days": card_notice_days,
    "governing_law": card_governing_law,
    "indemnity_mutuality": card_indemnity_mutuality,
}


def dropped_card(name, text):
    if name == "duration_months":
        v = extract_duration_months(text)
        return card_dropped_dimension(
            name, text, f"{_fmt_num(v)} months" if v is not None else None,
            "duration")
    if name == "amount_usd":
        v = extract_amount_usd(text)
        return card_dropped_dimension(
            name, text, f"${v:,.0f}" if v is not None else None, "amount")
    return None


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def route(text):
    """Conservative keyword route -> (label, dimension|None)."""
    for rx, label, dim in _ROUTE_PATTERNS:
        if rx.search(text):
            return label, dim, "keyword router"
    return None, None, "keyword router"


def analyze(text, override):
    """Route one clause -> dimension card. Returns (status_html, card_html)."""
    text = (text or "").strip()
    if not text:
        return "", card_empty()

    if override and override != "Auto (router)":
        # User-corrected type: re-run the pipeline on the override.
        clause_type = override
        dim = TYPE_TO_DIM.get(clause_type)
        routed_by = "type override"
    else:
        clause_type, dim, routed_by = route(text)

    if dim is None:
        return (
            f"Detected type: {clause_type or 'unknown'} &rarr; no benchmarked "
            f"dimension ({routed_by}; the router is conservative -- correct "
            "it with the override if wrong)",
            card_unsupported(clause_type or "unknown"),
        )
    if dim in DIM_CARD:
        card = DIM_CARD[dim](text)
    else:
        card = dropped_card(dim, text)
    return (
        f"Detected type: {clause_type} &rarr; benchmarked dimension: {dim} "
        f"({routed_by})",
        card,
    )


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def build_demo():
    classes = ["Auto (router)"] + sorted(TYPE_TO_DIM.keys())
    with gr.Blocks(title="ClauseBench - Is your clause normal?") as demo:
        gr.Markdown(
            "# ClauseBench: Is your clause normal?\n"
            "Paste one contract clause. A keyword router maps it to a "
            "dimension extractor, and you get an honest benchmark card "
            "computed from public corpora (CUAD + LEDGAR). "
            "Percentile cards ship for termination notice, governing law, "
            "and indemnification mutuality. If your clause does not state "
            "the value, the card says so instead of guessing."
        )
        clause_box = gr.Textbox(
            label="Paste one contract clause",
            lines=5,
            placeholder="e.g. Either party may terminate this Agreement upon "
                        "thirty (30) days' prior written notice...",
        )
        with gr.Row():
            b_term = gr.Button("Example: termination")
            b_gov = gr.Button("Example: governing law")
            b_ind = gr.Button("Example: indemnity")
        b_term.click(lambda: EXAMPLES["termination"], outputs=clause_box)
        b_gov.click(lambda: EXAMPLES["governing law"], outputs=clause_box)
        b_ind.click(lambda: EXAMPLES["indemnity"], outputs=clause_box)

        bench_btn = gr.Button("Benchmark this clause", variant="primary")
        override = gr.Dropdown(
            label="Type override (the router is conservative; correct it if wrong)",
            choices=classes,
            value="Auto (router)",
        )
        status = gr.Markdown()
        card = gr.HTML()

        bench_btn.click(analyze, inputs=[clause_box, override],
                        outputs=[status, card])
        override.change(analyze, inputs=[clause_box, override],
                        outputs=[status, card])

        gr.Markdown(
            "Decision support, not legal advice. Routing is keyword-based "
            "(conservative by design). The research clause classifier "
            "(TF-IDF + logistic regression, 41 CUAD classes, held-out "
            "accuracy 0.7222, n=2,765) is in src/train_classifier.py; its "
            "weights are not shipped (text-only repo). Corpus stats "
            "precomputed at build time; runs on CPU with no network."
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
