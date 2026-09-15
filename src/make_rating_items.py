"""Build the blind human-rating set used to measure judge-human agreement.

Takes test-split replies from all three systems (different tweets per system, so the rater never
sees the same tweet twice), shuffles them, and hides which system wrote each one. The
item -> (conv_id, system) key is stored separately. Needs no labels, and runs offline from the
cached agent responses, so rating can start as soon as labelling is done.
"""
import pandas as pd

from agent import run_on
from baselines import run_baselines
from config import LABELS_DIR, PAIRS_CSV, SEED

PER_SYSTEM = {"agent": 30, "simple": 20, "trivial": 10}


def main():
    pairs = pd.read_csv(PAIRS_CSV, dtype={"conv_id": str}, keep_default_na=False)
    history = pairs[pairs["split"] == "history"]
    candidates = pd.read_csv(LABELS_DIR / "golden_candidates.csv", dtype={"conv_id": str}, keep_default_na=False)
    test = candidates[candidates["golden_split"] == "test"].reset_index(drop=True)

    # The majority intent only affects the trivial baseline's intent, not its reply.
    trivial, simple = run_baselines(test, history, majority_intent="other")
    replies = {"agent": run_on(test, history), "simple": simple, "trivial": trivial}

    picks, used = [], set()
    for system, n in PER_SYSTEM.items():
        pool = replies[system][~replies[system]["conv_id"].isin(used)]
        chosen = pool.sample(n, random_state=SEED).assign(system=system)[["conv_id", "system", "reply"]]
        used |= set(chosen["conv_id"])
        picks.append(chosen)
    picks = pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)
    picks["item_id"] = [f"r{n:03d}" for n in range(1, len(picks) + 1)]

    items = picks.merge(test[["conv_id", "context", "customer_text"]], on="conv_id")
    items[["item_id", "context", "customer_text", "reply"]].to_csv(LABELS_DIR / "human_rating_items.csv", index=False)
    picks[["item_id", "conv_id", "system"]].to_csv(LABELS_DIR / "human_rating_key.csv", index=False)
    print(f"Wrote {len(items)} blind rating items. Do not open human_rating_key.csv until you finish rating.")


if __name__ == "__main__":
    main()
