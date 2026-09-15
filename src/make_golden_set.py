"""Sample golden-set candidates from the held-out (most recent) time period.

Strata (recorded per row so metrics can be reported on the natural distribution too):
  random          - uniform sample of the test period: estimates real-world performance
  intent_balanced - equal draws per weak-label intent: makes rare intents measurable
  hard            - escalation-keyword hits and multi-turn threads: stresses the escalation logic
Each candidate is also assigned to dev (prompt/threshold tuning) or test (reported numbers).
"""
import pandas as pd

from baselines import keyword_escalation, train_weak_label_classifier
from config import LABELS_DIR, PAIRS_CSV, SEED

N_RANDOM, N_BALANCED, N_HARD = 90, 80, 40
N_DEV = 50


def main():
    pairs = pd.read_csv(PAIRS_CSV, dtype={"conv_id": str}, keep_default_na=False)
    history = pairs[pairs["split"] == "history"]
    pool = pairs[pairs["split"] == "test"].copy()
    pool["weak_intent"] = train_weak_label_classifier(history).predict(pool["customer_text"])

    random_part = pool.sample(N_RANDOM, random_state=SEED).assign(stratum="random")
    pool = pool.drop(random_part.index)

    per_intent = max(1, N_BALANCED // pool["weak_intent"].nunique())
    balanced_part = (pool.groupby("weak_intent", group_keys=False)
                     .apply(lambda g: g.sample(min(per_intent, len(g)), random_state=SEED))
                     .assign(stratum="intent_balanced"))
    pool = pool.drop(balanced_part.index)

    keyword_hits = pool[pool["customer_text"].map(lambda t: bool(keyword_escalation(t)))]
    kw_part = keyword_hits.sample(min(N_HARD // 2, len(keyword_hits)), random_state=SEED)
    pool = pool.drop(kw_part.index)
    multi_turn = pool[pool["n_prior_turns"].astype(int) > 0]
    mt_part = multi_turn.sample(min(N_HARD - len(kw_part), len(multi_turn)), random_state=SEED)
    hard_part = pd.concat([kw_part, mt_part]).assign(stratum="hard")

    golden = pd.concat([random_part, balanced_part, hard_part]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    golden["golden_split"] = "test"
    golden.loc[:N_DEV - 1, "golden_split"] = "dev"

    # The brand's actual reply is deliberately excluded so the labeller judges the message as the agent sees it.
    out = golden[["conv_id", "golden_split", "stratum", "weak_intent", "n_prior_turns", "context", "customer_text"]]
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["weak_intent"]).to_csv(LABELS_DIR / "golden_candidates.csv", index=False)
    print(out.groupby(["golden_split", "stratum"]).size())
    print(f"Wrote {len(out)} candidates to {LABELS_DIR / 'golden_candidates.csv'}")


if __name__ == "__main__":
    main()
