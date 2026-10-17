"""Build the dimension-extraction gold sample (Project #41 ClauseBench).

For each dimension, samples 25 spans (seed 42) from the appropriate corpus
population. Writes one JSONL line per span to the goal hidden_files gold file
with placeholder null labels; hand-labeling fills gold_value_or_null.

Populations (matching the stats builder):
  duration, geo:      CUAD Non-Compete (259)
  amount:             CUAD Cap On Liability (672)
  notice:             CUAD Termination For Convenience (246)
                      + Notice Period To Terminate Renewal (122)
  govlaw:             CUAD Governing Law (464)
  indemnity:          LEDGAR train+validation+test, labels 49+50

span_id: cuad:<qa_id>:<answer_idx>  or  ledgar:<split>:<row_idx>
"""

import json
import os
import re

import numpy as np
import pandas as pd

GOAL = os.path.join(
    os.path.expanduser("~"),
    "workspace/goals/github-portfolio-linkedin-presence-buildout",
    "hidden_files/clausebench",
)
GOAL_DATA = os.path.join(GOAL, "data")
OUT = os.path.join(GOAL, "gold_dimensions.jsonl")
SEED = 42
N = 25

_CAT_RE = re.compile(r'related to "(.+?)"')


def cuad_spans(categories):
    with open(os.path.join(GOAL_DATA, "CUAD_v1.json")) as f:
        raw = json.load(f)
    out = []
    for doc in raw["data"]:
        for para in doc["paragraphs"]:
            for qa in para["qas"]:
                q = qa["question"].replace(" ", " ").strip()
                m = _CAT_RE.search(q)
                if not m or m.group(1) not in categories:
                    continue
                for ai, ans in enumerate(qa.get("answers", [])):
                    t = (ans.get("text") or "").strip()
                    if t:
                        out.append({
                            "span_id": f"cuad:{qa['id']}:{ai}",
                            "text": t,
                            "category": m.group(1),
                        })
    return out


def ledgar_spans(label_ids):
    out = []
    names = json.load(open(os.path.join(GOAL_DATA, "ledgar", "label_names.json")))
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(
            os.path.join(GOAL_DATA, "ledgar", f"{split}-00000-of-00001.parquet"),
            columns=["text", "label"],
        )
        sub = df[df["label"].isin(label_ids)]
        for idx, row in sub.iterrows():
            out.append({
                "span_id": f"ledgar:{split}:{idx}",
                "text": row["text"],
                "category": names.get(str(int(row["label"])), str(row["label"])),
            })
    return out


def sample(pop, n=N):
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(pop), size=min(n, len(pop)), replace=False)
    return [pop[i] for i in sorted(idx)]


def main():
    dims = []
    dims += [("duration", r) for r in sample(cuad_spans({"Non-Compete"}))]
    dims += [("geo", r) for r in sample(cuad_spans({"Non-Compete"}))]
    dims += [("amount", r) for r in sample(cuad_spans({"Cap On Liability"}))]
    dims += [("notice", r) for r in sample(
        cuad_spans({"Termination For Convenience",
                    "Notice Period To Terminate Renewal"}))]
    dims += [("govlaw", r) for r in sample(cuad_spans({"Governing Law"}))]
    dims += [("indemnity", r) for r in sample(ledgar_spans({49, 50}))]

    with open(OUT, "w") as f:
        for dim, r in dims:
            f.write(json.dumps({
                "span_id": r["span_id"],
                "text": r["text"],
                "category": r["category"],
                "dimension": dim,
                "gold_value_or_null": None,
                "labeler_note": "",
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(dims)} rows to {OUT}")


if __name__ == "__main__":
    main()
