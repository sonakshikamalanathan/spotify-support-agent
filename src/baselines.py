"""Baselines the agent must beat.

Trivial:  majority intent | the brand's single most common reply | never escalate
Simple:   TF-IDF + logistic regression trained on cluster-derived weak labels
          | copy the brand reply of the most similar past message | keyword rules only
"""
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from agent import ESCALATION_RULES, Retriever
from config import DATA_PROCESSED, LABELS_DIR, SEED

CLUSTER_MAP_PATH = LABELS_DIR / "cluster_intent_map.json"


def train_weak_label_classifier(history):
    """Human maps each k-means cluster to an intent; every history message inherits its
    cluster's intent. Free, noisy supervision for a classic classifier."""
    cluster_map = json.loads(CLUSTER_MAP_PATH.read_text(encoding="utf-8"))
    clusters = pd.read_csv(DATA_PROCESSED / "history_clusters.csv", dtype={"conv_id": str})
    clusters["intent"] = clusters["cluster"].astype(str).map(cluster_map)
    train = clusters.dropna(subset=["intent"]).merge(history[["conv_id", "customer_text"]], on="conv_id")
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    )
    model.fit(train["customer_text"], train["intent"])
    return model


def keyword_escalation(message):
    return [f"rule:{rid}" for rid, pattern in ESCALATION_RULES if pattern.search(message)]


def run_baselines(frame, history, majority_intent):
    history = history.copy()
    history["conv_id"] = history["conv_id"].astype(str)
    most_common_reply = history["brand_reply"].value_counts().idxmax()
    retriever = Retriever(history)
    weak_clf = train_weak_label_classifier(history)
    weak_preds = weak_clf.predict(frame["customer_text"])

    trivial, simple = [], []
    for row, weak_intent in zip(frame.itertuples(index=False), weak_preds):
        trivial.append({
            "conv_id": str(row.conv_id), "pred_intent": majority_intent, "reply": most_common_reply,
            "escalate": False, "escalation_reasons": "",
        })
        reasons = keyword_escalation(row.customer_text)
        nearest = retriever.search(row.customer_text, k=1)[0]
        simple.append({
            "conv_id": str(row.conv_id), "pred_intent": weak_intent, "reply": nearest["brand_reply"],
            "escalate": bool(reasons), "escalation_reasons": ";".join(reasons),
        })
    return pd.DataFrame(trivial), pd.DataFrame(simple)
