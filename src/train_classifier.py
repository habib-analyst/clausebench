"""Train the CUAD clause-type classifier (Project #41 ClauseBench).

Pipeline: CUAD answer spans -> TF-IDF word 1-2 grams (sublinear, max_features
50000) -> multinomial logistic regression (lbfgs, C=1.0). Deterministic:
stratified 80/20 train/test split with seed 42, saved to data/split.json;
random_state fixed everywhere.

Saves: data/clf/tfidf_vectorizer.joblib, data/clf/logreg_model.joblib,
       data/clf/labels.json, data/split.json.

Run: ~/workspace/build/clausebench/.venv/bin/python src/train_classifier.py
(writes measured metrics to data/clf/metrics.json)
"""

import json
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA = os.path.join(ROOT, "data")
GOAL_DATA = os.path.join(
    os.path.expanduser("~"),
    "workspace/goals/github-portfolio-linkedin-presence-buildout",
    "hidden_files/clausebench/data",
)
CUAD_PATH = os.path.join(GOAL_DATA, "CUAD_v1.json")
SEED = 42

_CAT_RE = re.compile(r'related to "(.+?)"')


def load_cuad_spans(cuad_path=CUAD_PATH):
    """Return DataFrame(text, category) of all non-empty CUAD answer spans."""
    with open(cuad_path) as f:
        raw = json.load(f)
    rows = []
    for doc in raw["data"]:
        for para in doc["paragraphs"]:
            for qa in para["qas"]:
                q = qa["question"].replace(" ", " ").strip()
                m = _CAT_RE.search(q)
                if not m:
                    continue
                cat = m.group(1)
                for ans in qa.get("answers", []):
                    t = (ans.get("text") or "").strip()
                    if t:
                        rows.append((t, cat))
    return pd.DataFrame(rows, columns=["text", "category"])


def main():
    os.makedirs(os.path.join(DATA, "clf"), exist_ok=True)
    df = load_cuad_spans()
    print(f"spans={len(df)} categories={df['category'].nunique()}")

    counts = df["category"].value_counts().sort_values()
    print("class counts (min -> max):")
    print(counts.to_string())
    small = counts[counts < 5]
    if len(small):
        print("CLASSES WITH <5 EXAMPLES (kept, documented):")
        print(small.to_string())

    X = df["text"].tolist()
    y = df["category"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    with open(os.path.join(DATA, "split.json"), "w") as f:
        json.dump(
            {"seed": SEED, "test_size": 0.2, "stratify": True,
             "n_train": len(X_train), "n_test": len(X_test),
             "test_texts": X_test, "test_labels": y_test},
            f,
        )
    print(f"train={len(X_train)} test={len(X_test)}")

    vec = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, max_features=50000,
        lowercase=True, strip_accents="unicode",
    )
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED, n_jobs=-1,
    )
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    labels = sorted(df["category"].unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    off = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                off.append((labels[i], labels[j], int(cm[i, j])))
    off.sort(key=lambda t: -t[2])
    print(f"Held-out accuracy: {acc:.4f} (n={len(y_test)})")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Top-5 confused pairs (true -> predicted, count):")
    for a, b, c in off[:5]:
        print(f"  {a} -> {b}: {c}")

    joblib.dump(vec, os.path.join(DATA, "clf", "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(DATA, "clf", "logreg_model.joblib"))
    with open(os.path.join(DATA, "clf", "labels.json"), "w") as f:
        json.dump({"classes": labels}, f)
    with open(os.path.join(DATA, "clf", "metrics.json"), "w") as f:
        json.dump(
            {"accuracy": acc, "macro_f1": macro_f1, "n_test": len(y_test),
             "n_train": len(X_train), "top_confused": [
                 {"true": a, "pred": b, "count": c} for a, b, c in off[:10]],
             "classes_below_5": small.index.tolist(), "seed": SEED},
            f, indent=2,
        )
    print("saved to data/clf/")


if __name__ == "__main__":
    main()
