"""End-to-end evaluation on the golden set: agent vs trivial vs simple baselines.

Usage:
  python src/run_eval.py --split test            # calls APIs for anything not cached
  OFFLINE=1 python src/run_eval.py --split test  # cached LLM responses only (no keys needed)
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import metrics as M
from agent import format_thread, run_on
from baselines import run_baselines
from config import EVAL_DIR, LABELS_DIR, PAIRS_CSV, REPORTS_DIR, SEED
from judge import ReplyJudge

SYSTEMS = ["trivial", "simple", "agent"]


def load_golden(split):
    candidates = pd.read_csv(LABELS_DIR / "golden_candidates.csv", dtype={"conv_id": str}, keep_default_na=False)
    labels = pd.read_csv(LABELS_DIR / "golden_labels.csv", dtype={"conv_id": str}, keep_default_na=False)
    gold = candidates.merge(labels, on="conv_id", how="inner")
    if split != "all":
        gold = gold[gold["golden_split"] == split]
    gold["should_escalate"] = gold["should_escalate"].eq("yes")
    return gold.reset_index(drop=True)


def predict(gold, pairs, history):
    majority = load_golden("dev")["intent"].value_counts().idxmax()  # never peek at test labels
    trivial, simple = run_baselines(gold, history, majority)
    agent = run_on(gold, history)
    return {"trivial": trivial, "simple": simple, "agent": agent}


def judge_all(gold, preds, pairs):
    """Grade every system's reply for every golden message, blind to which system wrote it.
    The only reference is the brand's actual reply: adding retrieved replies would hand the
    retrieval baseline its own answer as the reference."""
    actual = pairs.set_index("conv_id")["brand_reply"]
    replies = {system: preds[system].set_index("conv_id")["reply"] for system in SYSTEMS}
    cases = [
        {"conv_id": g.conv_id, "system": system, "reply": replies[system][g.conv_id],
         "thread": format_thread(g.context, g.customer_text), "references": [actual[g.conv_id]]}
        for g in gold.itertuples(index=False) for system in SYSTEMS
    ]
    # Shuffle so each batch mixes systems and conversations.
    cases = [cases[i] for i in np.random.default_rng(SEED).permutation(len(cases))]
    scores = ReplyJudge().score_many(cases)
    return pd.DataFrame([{"conv_id": c["conv_id"], "system": c["system"], "reply": c["reply"], **s} for c, s in zip(cases, scores)])


def system_metrics(gold, pred, judged):
    df = gold.merge(pred, on="conv_id").merge(judged[["conv_id", "judge_overall", "judge_acceptable", "judge_hallucination"]], on="conv_id")
    df["intent_correct"] = df["intent"] == df["pred_intent"]
    auto = ~df["escalate"].astype(bool)
    # Safe automation: auto-handled, didn't need a human, right intent, acceptable reply.
    df["safe_auto"] = auto & ~df["should_escalate"] & df["intent_correct"] & df["judge_acceptable"]
    df["unsafe_auto"] = auto & df["should_escalate"]
    out = {
        "n": len(df),
        "intent_accuracy": M.bootstrap_ci(M.accuracy, df["intent"], df["pred_intent"]),
        "intent_macro_f1": M.bootstrap_ci(M.macro_f1, df["intent"], df["pred_intent"]),
        "escalation": M.escalation_metrics(df["should_escalate"], df["escalate"]),
        "escalation_recall_ci": M.bootstrap_ci(M.recall_fn, df["should_escalate"], df["escalate"]),
        "reply_acceptable": M.bootstrap_ci(lambda t, p: np.mean(p), df["judge_acceptable"], df["judge_acceptable"].astype(float)),
        "reply_overall_mean": float(df["judge_overall"].mean()),
        "hallucination_rate": float(df["judge_hallucination"].mean()),
        "safe_automation_rate": M.bootstrap_ci(lambda t, p: np.mean(p), df["safe_auto"], df["safe_auto"].astype(float)),
        "unsafe_automation_rate": float(df["unsafe_auto"].mean()),
    }
    return out, df


def automation_dial(agent_df, split):
    """Sweep the confidence threshold: how much can we auto-handle, and how often is an
    auto-handled message wrong (needed a human, wrong intent, or unacceptable reply)?"""
    hard = agent_df["escalation_reasons"].fillna("").map(
        lambda r: any(x and x != "low_confidence" for x in r.split(";")))
    bad = agent_df["should_escalate"] | ~agent_df["intent_correct"] | ~agent_df["judge_acceptable"]
    rows = []
    for t in np.round(np.arange(0.0, 1.01, 0.05), 2):
        auto = ~hard & (agent_df["confidence"] >= t)
        rows.append({"threshold": t, "automation_rate": auto.mean(),
                     "error_rate_among_auto": bad[auto].mean() if auto.any() else np.nan,
                     "unsafe_auto_rate": (auto & agent_df["should_escalate"]).mean()})
    dial = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dial["threshold"], dial["automation_rate"], label="Automation rate (share auto-handled)")
    ax.plot(dial["threshold"], dial["error_rate_among_auto"], label="Error rate among auto-handled")
    ax.plot(dial["threshold"], dial["unsafe_auto_rate"], label="Should-escalate but auto-handled (share of all)")
    ax.set_xlabel("Intent-confidence threshold for auto-handling")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title(f"Automation dial ({split} split)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"automation_dial_{split}.png", dpi=150)
    plt.close(fig)
    return dial


def judge_human_agreement(judged):
    ratings_path, key_path = LABELS_DIR / "human_reply_ratings.csv", LABELS_DIR / "human_rating_key.csv"
    if not ratings_path.exists() or not key_path.exists():
        return None
    human = pd.read_csv(ratings_path, dtype={"item_id": str}, keep_default_na=False)
    key = pd.read_csv(key_path, dtype={"item_id": str, "conv_id": str})
    df = human.merge(key, on="item_id").merge(judged, on=["conv_id", "system"])
    if len(df) < 10:
        return None
    df["human_hallucination"] = df["hallucination"].eq("yes")
    df["human_acceptable"] = (df["overall"].astype(int) >= 4) & ~df["human_hallucination"]
    df["reply_len"] = df["reply"].str.len()
    return {
        "n": len(df),
        "overall_weighted_kappa": M.weighted_kappa(df["overall"].astype(int), df["judge_overall"]),
        "overall_exact_agreement": float((df["overall"].astype(int) == df["judge_overall"]).mean()),
        "overall_within_1": float(((df["overall"].astype(int) - df["judge_overall"]).abs() <= 1).mean()),
        "acceptable_kappa": M.kappa(df["human_acceptable"], df["judge_acceptable"]),
        "acceptable_agreement": float((df["human_acceptable"] == df["judge_acceptable"]).mean()),
        "hallucination_kappa": M.kappa(df["human_hallucination"], df["judge_hallucination"]),
        "judge_minus_human_mean": float((df["judge_overall"] - df["overall"].astype(int)).mean()),
        "length_corr_judge": float(df["reply_len"].corr(df["judge_overall"], method="spearman")),
        "length_corr_human": float(df["reply_len"].corr(df["overall"].astype(int), method="spearman")),
        "confusion_acceptable": pd.crosstab(df["human_acceptable"], df["judge_acceptable"]).to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test", "all"])
    split = parser.parse_args().split
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(PAIRS_CSV, dtype={"conv_id": str}, keep_default_na=False)
    history = pairs[pairs["split"] == "history"].copy()
    gold = load_golden(split)
    print(f"Golden {split}: {len(gold)} labelled examples")

    preds = predict(gold, pairs, history)
    for system, frame in preds.items():
        frame.to_csv(EVAL_DIR / f"predictions_{split}_{system}.csv", index=False)
    judged = judge_all(gold, preds, pairs)
    judged.to_csv(EVAL_DIR / f"judgments_{split}.csv", index=False)

    results, merged = {"split": split}, {}
    for system in SYSTEMS:
        results[system], merged[system] = system_metrics(gold, preds[system], judged[judged["system"] == system])
        random_only = gold[gold["stratum"] == "random"]
        results[system]["random_stratum_only"], _ = system_metrics(random_only, preds[system], judged[judged["system"] == system])

    agent_df = merged["agent"]
    results["automation_dial"] = automation_dial(agent_df, split).to_dict(orient="records")
    results["judge_human_agreement"] = judge_human_agreement(judged)
    results["agent_intent_report"] = classification_report(agent_df["intent"], agent_df["pred_intent"], output_dict=True, zero_division=0)
    labels = sorted(set(agent_df["intent"]) | set(agent_df["pred_intent"]))
    results["agent_confusion"] = {"labels": labels, "matrix": confusion_matrix(agent_df["intent"], agent_df["pred_intent"], labels=labels).tolist()}

    failures = agent_df[(~agent_df["intent_correct"]) | (agent_df["should_escalate"] != agent_df["escalate"]) | (~agent_df["judge_acceptable"])]
    failures.merge(judged[judged["system"] == "agent"][["conv_id", "judge_rationale"]], on="conv_id").to_csv(EVAL_DIR / f"failures_{split}.csv", index=False)

    (EVAL_DIR / f"results_{split}.json").write_text(json.dumps(results, indent=2, default=str))
    print_summary(results)


def print_summary(results):
    print(f"\n=== Results ({results['split']}) ===")
    header = f"{'metric':32}" + "".join(f"{s:>28}" for s in SYSTEMS)
    print(header)
    for label, getter in [
        ("intent accuracy", lambda r: M.fmt_ci(*r["intent_accuracy"], pct=True)),
        ("intent macro-F1", lambda r: M.fmt_ci(*r["intent_macro_f1"])),
        ("escalation recall", lambda r: M.fmt_ci(*r["escalation_recall_ci"], pct=True)),
        ("escalation precision", lambda r: f"{r['escalation']['precision']:.1%}"),
        ("escalation rate", lambda r: f"{r['escalation']['escalation_rate']:.1%}"),
        ("reply acceptable (judge)", lambda r: M.fmt_ci(*r["reply_acceptable"], pct=True)),
        ("hallucination rate (judge)", lambda r: f"{r['hallucination_rate']:.1%}"),
        ("safe automation rate", lambda r: M.fmt_ci(*r["safe_automation_rate"], pct=True)),
        ("unsafe automation rate", lambda r: f"{r['unsafe_automation_rate']:.1%}"),
    ]:
        print(f"{label:32}" + "".join(f"{getter(results[s]):>28}" for s in SYSTEMS))
    if results["judge_human_agreement"]:
        print("\nJudge vs human:", json.dumps({k: v for k, v in results["judge_human_agreement"].items() if k != "confusion_acceptable"}, indent=2))


if __name__ == "__main__":
    main()
