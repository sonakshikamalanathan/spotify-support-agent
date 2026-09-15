"""Turn evaluation results into a deployment decision: which intents may the agent answer on its
own, and what does each rollout policy cost per 100 tickets?

Tiers are chosen on the dev split and priced on the test split, so the policy is not tuned on the
numbers used to judge it. Requires run_eval.py to have been run for both splits.
"""
import json

import pandas as pd

from config import EVAL_DIR
from run_eval import load_golden

# Relative cost of one ticket, where a human writing a reply from scratch = 1.
COST = {"human_writes": 1.0, "human_approves_draft": 0.3, "bad_auto_reply": 3.0, "missed_escalation": 10.0}
# An intent may auto-send only if (failures + 1) / (auto-handled + 2) on dev is at or below this.
# The +1/+2 smoothing stops an intent with two lucky dev examples from qualifying.
MAX_SMOOTHED_FAILURE = 0.2
MISSED_ESCALATION_SENSITIVITY = [5.0, 10.0, 25.0]


def load_split(split):
    gold = load_golden(split)
    pred = pd.read_csv(EVAL_DIR / f"predictions_{split}_agent.csv", dtype={"conv_id": str}, keep_default_na=False)
    judged = pd.read_csv(EVAL_DIR / f"judgments_{split}.csv", dtype={"conv_id": str}, keep_default_na=False)
    judged = judged[judged["system"] == "agent"][["conv_id", "judge_acceptable"]]
    df = gold.merge(pred, on="conv_id").merge(judged, on="conv_id")
    df["escalate"] = df["escalate"].astype(str).eq("True")
    df["judge_acceptable"] = df["judge_acceptable"].astype(str).eq("True")
    df["good_reply"] = (df["intent"] == df["pred_intent"]) & df["judge_acceptable"]
    return df


def per_intent(df):
    rows = []
    for intent, group in df.groupby("pred_intent"):
        auto = group[~group["escalate"]]
        failures = int((auto["should_escalate"] | ~auto["good_reply"]).sum())
        rows.append({
            "pred_intent": intent, "n": len(group), "auto_handled": len(auto), "auto_failures": failures,
            "missed_escalations": int(auto["should_escalate"].sum()),
            "smoothed_failure_rate": round((failures + 1) / (len(auto) + 2), 3),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def ticket_cost(row, mode, cost):
    """mode: 'human' (human writes), 'suggest' (human sees the AI draft) or 'auto' (AI may send)."""
    if mode == "human":
        return cost["human_writes"]
    if mode == "suggest" or row["escalate"]:
        usable_draft = not row["should_escalate"] and row["good_reply"]
        return cost["human_approves_draft"] if usable_draft else cost["human_writes"]
    if row["should_escalate"]:
        return cost["missed_escalation"]
    return 0.0 if row["good_reply"] else cost["bad_auto_reply"]


def price_policy(test, modes, cost):
    return 100 * sum(ticket_cost(row, mode, cost) for (_, row), mode in zip(test.iterrows(), modes)) / len(test)


def main():
    dev, test = load_split("dev"), load_split("test")
    dev_table = per_intent(dev)
    auto_intents = sorted(dev_table.loc[dev_table["smoothed_failure_rate"] <= MAX_SMOOTHED_FAILURE, "pred_intent"])

    policies = {
        "all_human": lambda r: "human",
        "suggest_only": lambda r: "suggest",
        "auto_everything": lambda r: "auto",
        "tiered": lambda r: "auto" if r["pred_intent"] in auto_intents else "suggest",
    }
    results = {}
    for name, choose in policies.items():
        modes = test.apply(choose, axis=1)
        sent = (modes == "auto") & ~test["escalate"]
        results[name] = {
            "cost_per_100_tickets": round(price_policy(test, modes, COST), 1),
            "sent_without_human": round(float(sent.mean()), 3),
            "missed_escalations": int((sent & test["should_escalate"]).sum()),
            "bad_replies_sent": int((sent & ~test["should_escalate"] & ~test["good_reply"]).sum()),
            "cost_if_missed_escalation_costs": {
                str(c): round(price_policy(test, modes, {**COST, "missed_escalation": c}), 1) for c in MISSED_ESCALATION_SENSITIVITY
            },
        }

    out = {
        "cost_model": COST, "max_smoothed_failure": MAX_SMOOTHED_FAILURE, "n_test": len(test),
        "auto_intents_chosen_on_dev": auto_intents,
        "dev_per_intent": dev_table.to_dict("records"),
        "test_per_intent": per_intent(test).to_dict("records"),
        "test_policies": results,
    }
    (EVAL_DIR / "trust_policy.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("auto_intents_chosen_on_dev", "test_policies")}, indent=2))


if __name__ == "__main__":
    main()
