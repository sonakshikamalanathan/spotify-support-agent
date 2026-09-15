"""Build the blind human-rating set used to measure judge-human agreement.

Samples replies from all three systems on the test split, shuffles them, and hides which
system wrote each one. The item -> (conv_id, system) key is stored separately.
"""
import pandas as pd

from config import EVAL_DIR, LABELS_DIR, SEED

PER_SYSTEM = {"agent": 30, "simple": 20, "trivial": 10}


def main():
    judged = pd.read_csv(EVAL_DIR / "judgments_test.csv", dtype={"conv_id": str}, keep_default_na=False)
    candidates = pd.read_csv(LABELS_DIR / "golden_candidates.csv", dtype={"conv_id": str}, keep_default_na=False)
    picks = pd.concat([
        judged[judged["system"] == system].sample(n, random_state=SEED) for system, n in PER_SYSTEM.items()
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    picks["item_id"] = [f"r{n:03d}" for n in range(1, len(picks) + 1)]
    items = picks.merge(candidates[["conv_id", "context", "customer_text"]], on="conv_id")
    items[["item_id", "context", "customer_text", "reply"]].to_csv(LABELS_DIR / "human_rating_items.csv", index=False)
    picks[["item_id", "conv_id", "system"]].to_csv(LABELS_DIR / "human_rating_key.csv", index=False)
    print(f"Wrote {len(items)} blind rating items. Do not open human_rating_key.csv until you finish rating.")


if __name__ == "__main__":
    main()
