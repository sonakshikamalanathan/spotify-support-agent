"""Shared file helpers for the labelling tools (label_app.py and fast_label.py)."""
import pandas as pd

from config import LABELS_DIR, SEED

CODEBOOK = LABELS_DIR / "codebook.json"
CANDIDATES = LABELS_DIR / "golden_candidates.csv"
GOLDEN = LABELS_DIR / "golden_labels.csv"
RELABEL_ITEMS = LABELS_DIR / "relabel_items.csv"
RELABEL = LABELS_DIR / "relabel_labels.csv"
RATING_ITEMS = LABELS_DIR / "human_rating_items.csv"
RATINGS = LABELS_DIR / "human_reply_ratings.csv"
N_RELABEL = 30
MIN_LABELS_BEFORE_RELABEL = 150


def read_csv(path, key):
    if path.exists():
        return pd.read_csv(path, dtype={key: str}, keep_default_na=False)
    return pd.DataFrame(columns=[key])


def upsert(path, key, row):
    df = read_csv(path, key)
    df = df[df[key] != str(row[key])]
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def relabel_items():
    """The finished labels to re-label blind, sampled once and then fixed. None until enough labels exist."""
    if not RELABEL_ITEMS.exists():
        labels = read_csv(GOLDEN, "conv_id")
        if len(labels) < MIN_LABELS_BEFORE_RELABEL:
            return None
        labels.sample(N_RELABEL, random_state=SEED)[["conv_id"]].to_csv(RELABEL_ITEMS, index=False)
    candidates = read_csv(CANDIDATES, "conv_id").set_index("conv_id")
    return candidates.loc[read_csv(RELABEL_ITEMS, "conv_id")["conv_id"]].reset_index()
