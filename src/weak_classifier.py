"""TF-IDF + logistic regression intent classifier trained on weak labels.

Each k-means cluster that clearly belongs to one intent was mapped to it by hand
(labels/cluster_intent_map.json), and every historical tweet in that cluster inherits the intent:
free, noisy supervision with no LLM involved. Used as the simple baseline, to stratify the golden
set sample, and as the agent's second opinion.
"""
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from config import DATA_PROCESSED, LABELS_DIR, SEED

CLUSTER_MAP_PATH = LABELS_DIR / "cluster_intent_map.json"


def train_weak_label_classifier(history):
    cluster_map = json.loads(CLUSTER_MAP_PATH.read_text(encoding="utf-8"))
    clusters = pd.read_csv(DATA_PROCESSED / "history_clusters.csv", dtype={"conv_id": str})
    clusters["intent"] = clusters["cluster"].astype(str).map(cluster_map)
    history = history.assign(conv_id=history["conv_id"].astype(str))
    train = clusters.dropna(subset=["intent"]).merge(history[["conv_id", "customer_text"]], on="conv_id")
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    )
    model.fit(train["customer_text"], train["intent"])
    return model
