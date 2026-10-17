"""Load the trained ClauseBench clause-type classifier and predict.

The model is trained by src/train_classifier.py (TF-IDF word 1-2 grams +
multinomial logistic regression) and stored under data/clf/.
"""

import os

import joblib

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BASE)
_CLF_DIR = os.path.join(_ROOT, "data", "clf")


def load_classifier(clf_dir=_CLF_DIR):
    """Return (vectorizer, model) loaded from disk."""
    vec = joblib.load(os.path.join(clf_dir, "tfidf_vectorizer.joblib"))
    clf = joblib.load(os.path.join(clf_dir, "logreg_model.joblib"))
    return vec, clf


def predict(texts, vec=None, clf=None):
    """Predict CUAD categories for a list of clause texts."""
    if vec is None or clf is None:
        vec, clf = load_classifier()
    X = vec.transform(texts)
    return clf.predict(X).tolist()


def predict_proba(texts, vec=None, clf=None):
    """Return (classes, probability matrix) for a list of clause texts."""
    if vec is None or clf is None:
        vec, clf = load_classifier()
    X = vec.transform(texts)
    return list(clf.classes_), clf.predict_proba(X).tolist()
