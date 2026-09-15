"""Break agent failures down so the report's failure analysis rests on counts, not anecdotes.

Usage: python src/failure_analysis.py --split test
Writes eval/failure_analysis_<split>.json and prints a summary with examples.
"""
import argparse
import json

import pandas as pd

from config import EVAL_DIR
from run_eval import load_golden

EXAMPLE_COLUMNS = ["conv_id", "customer_text", "context", "intent", "pred_intent", "confidence", "should_escalate",
                   "escalation_reason", "escalation_reasons", "reply", "judge_overall", "judge_rationale"]


def load(split):
    gold = load_golden(split)
    pred = pd.read_csv(EVAL_DIR / f"predictions_{split}_agent.csv", dtype={"conv_id": str}, keep_default_na=False)
    judged = pd.read_csv(EVAL_DIR / f"judgments_{split}.csv", dtype={"conv_id": str}, keep_default_na=False)
    judged = judged[judged["system"] == "agent"][["conv_id", "judge_overall", "judge_acceptable", "judge_hallucination", "judge_rationale"]]
    df = gold.merge(pred, on="conv_id").merge(judged, on="conv_id")
    for col in ("escalate", "judge_acceptable", "judge_hallucination"):
        df[col] = df[col].astype(str).eq("True")
    df["confidence"] = df["confidence"].astype(float)
    df["intent_correct"] = df["intent"] == df["pred_intent"]
    df["has_thread"] = df["n_prior_turns"].astype(int) > 0
    df["confidence_bucket"] = pd.cut(df["confidence"], [-0.01, 0.6, 0.8, 0.9, 1.0], labels=["<0.6", "0.6-0.8", "0.8-0.9", ">=0.9"])
    # Escalation layers that fired, e.g. "rule:payment_dispute;low_confidence" -> ["low_confidence", "rule"]
    df["layers"] = df["escalation_reasons"].map(lambda r: sorted({x.split(":")[0] for x in r.split(";") if x}))
    return df


def examples(frame, n):
    return frame[EXAMPLE_COLUMNS].head(n).to_dict("records")


def rate_by(df, column, value_col):
    grouped = df.groupby(column, observed=True)[value_col].agg(["mean", "size"]).reset_index()
    return [{column: str(r[column]), "rate": round(float(r["mean"]), 3), "n": int(r["size"])} for _, r in grouped.iterrows()]


def analyse(df):
    wrong = df[~df["intent_correct"]]
    missed = df[df["should_escalate"] & ~df["escalate"]]
    unneeded = df[~df["should_escalate"] & df["escalate"]]
    bad_reply = df[~df["judge_acceptable"]].sort_values("judge_overall")
    confusions = wrong.groupby(["intent", "pred_intent"]).size().sort_values(ascending=False)
    return {
        "n": len(df),
        "intent_confusions": [{"gold": g, "pred": p, "count": int(c)} for (g, p), c in confusions.items()],
        "intent_confusion_examples": examples(wrong, 10),
        "intent_accuracy_by_gold_intent": rate_by(df, "intent", "intent_correct"),
        "intent_accuracy_by_confidence": rate_by(df, "confidence_bucket", "intent_correct"),
        "intent_accuracy_by_stratum": rate_by(df, "stratum", "intent_correct"),
        "intent_accuracy_thread_vs_single": rate_by(df, "has_thread", "intent_correct"),
        "escalation_recall_by_gold_reason": rate_by(df[df["should_escalate"]], "escalation_reason", "escalate"),
        "missed_escalations": {
            "count": len(missed), "by_gold_reason": missed["escalation_reason"].value_counts().to_dict(),
            "examples": examples(missed, 10),
        },
        "unneeded_escalations": {
            "count": len(unneeded),
            "layers_fired": pd.Series([layer for layers in unneeded["layers"] for layer in layers], dtype=object).value_counts().to_dict(),
            "only_low_confidence": int(unneeded["layers"].map(lambda ls: ls == ["low_confidence"]).sum()),
            "examples": examples(unneeded, 5),
        },
        "unacceptable_replies": {
            "count": len(bad_reply), "hallucinations": int(df["judge_hallucination"].sum()),
            "acceptable_rate_by_pred_intent": rate_by(df, "pred_intent", "judge_acceptable"),
            "examples": examples(bad_reply, 10),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    split = parser.parse_args().split
    result = {"split": split, **analyse(load(split))}
    (EVAL_DIR / f"failure_analysis_{split}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(f"=== Failure analysis ({split}, n={result['n']}) ===")
    print("Top intent confusions (gold -> predicted):", [(c["gold"], c["pred"], c["count"]) for c in result["intent_confusions"][:6]])
    print("Intent accuracy by confidence:", [(r["confidence_bucket"], r["rate"], r["n"]) for r in result["intent_accuracy_by_confidence"]])
    print("Intent accuracy, thread vs single tweet:", [(r["has_thread"], r["rate"], r["n"]) for r in result["intent_accuracy_thread_vs_single"]])
    print("Escalation recall by gold reason:", [(r["escalation_reason"], r["rate"], r["n"]) for r in result["escalation_recall_by_gold_reason"]])
    missed = result["missed_escalations"]
    print(f"Missed escalations: {missed['count']} by reason {missed['by_gold_reason']}")
    for e in missed["examples"]:
        print(f"  - [{e['escalation_reason']}] predicted {e['pred_intent']} ({e['confidence']:.2f}): {e['customer_text'][:120]}")
    unneeded = result["unneeded_escalations"]
    print(f"Unneeded escalations: {unneeded['count']} | layers fired: {unneeded['layers_fired']} | only low confidence: {unneeded['only_low_confidence']}")
    bad = result["unacceptable_replies"]
    print(f"Unacceptable replies: {bad['count']} ({bad['hallucinations']} flagged hallucinations)")
    for e in bad["examples"][:6]:
        print(f"  - {e['judge_overall']}/5 [{e['pred_intent']}] {e['reply'][:100]} | judge: {e['judge_rationale'][:110]}")


if __name__ == "__main__":
    main()
