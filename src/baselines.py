"""Baselines the agent must beat.

Trivial:  majority intent | the brand's single most common reply | never escalate
Simple:   TF-IDF + logistic regression trained on cluster-derived weak labels
          | copy the brand reply of the most similar past message | keyword rules only
"""
import pandas as pd

from agent import ESCALATION_RULES, Retriever
from weak_classifier import train_weak_label_classifier


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
