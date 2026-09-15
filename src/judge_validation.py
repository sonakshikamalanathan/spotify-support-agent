"""Validate the LLM judge beyond human agreement.

1. Stress test: real test-period conversations paired with the brand's own reply (control) and six
   deliberately broken replies. A judge worth trusting must reject the broken variants.
2. Batching check: re-grade a subset one case per call and compare with the batched grades.
"""
import json

import numpy as np
import pandas as pd

import metrics as M
from agent import Retriever, format_thread
from config import EVAL_DIR, LABELS_DIR, PAIRS_CSV, SEED
from judge import ReplyJudge

N_CONVERSATIONS = 12
N_SINGLE_CHECK = 24

DEFECTS = {
    "invented_refund": lambda reply, other: reply + " We've also refunded your last 3 payments and added 2 free months of Premium.",
    "asks_for_password": lambda reply, other: "Hey! Reply here with your account email and password and we'll log in and fix it for you.",
    "wrong_topic": lambda reply, other: other,
    "rude": lambda reply, other: "That's not our problem. Read the help pages before tweeting at us.",
    "fake_policy_link": lambda reply, other: reply + " Under our new loyalty policy you get Premium free for life at spotify.com/free-forever",
    "ignores_issue": lambda reply, other: "Thanks for reaching out! Have a great day :)",
}
MUST_FLAG_HALLUCINATION = {"invented_refund", "fake_policy_link"}


def pick_conversations(pairs):
    golden_ids = set(pd.read_csv(LABELS_DIR / "golden_candidates.csv", dtype={"conv_id": str})["conv_id"])
    pool = pairs[
        (pairs["split"] == "test")
        & ~pairs["conv_id"].isin(golden_ids)
        & (pairs["n_prior_turns"].astype(int) == 0)
        & (pairs["customer_text"].str.len() >= 40)
        & (pairs["brand_reply"].str.len() >= 60)
        & ~pairs["brand_reply"].str.contains(r"\bDM\b", case=False)
    ]
    return pool.sample(N_CONVERSATIONS, random_state=SEED).reset_index(drop=True)


def least_similar_reply(convs, i):
    """Brand reply from the sampled conversation sharing the fewest words with conversation i."""
    words = [set(t.lower().split()) for t in convs["customer_text"]]
    overlap = [len(words[i] & w) / len(words[i] | w) if j != i else 2 for j, w in enumerate(words)]
    return convs.at[int(np.argmin(overlap)), "brand_reply"]


def build_cases(convs, retriever):
    cases = []
    for i, conv in convs.iterrows():
        base = {"conv_id": conv["conv_id"], "thread": format_thread("", conv["customer_text"]),
                "references": [r["brand_reply"] for r in retriever.search(conv["customer_text"], k=3)]}
        cases.append({**base, "variant": "control", "reply": conv["brand_reply"]})
        other = least_similar_reply(convs, i)
        for name, corrupt in DEFECTS.items():
            cases.append({**base, "variant": name, "reply": corrupt(conv["brand_reply"], other)})
    order = np.random.default_rng(SEED).permutation(len(cases))
    return [cases[k] for k in order]


def main():
    pairs = pd.read_csv(PAIRS_CSV, dtype={"conv_id": str}, keep_default_na=False)
    retriever = Retriever(pairs[pairs["split"] == "history"])
    cases = build_cases(pick_conversations(pairs), retriever)

    batched = ReplyJudge().score_many(cases)
    df = pd.DataFrame([{k: v for k, v in c.items() if k not in ("thread", "references")} | s for c, s in zip(cases, batched)])
    df.to_csv(EVAL_DIR / "judge_stress_test.csv", index=False)

    stress = {}
    for variant, group in df.groupby("variant"):
        stress[variant] = {
            "n": len(group),
            "judged_acceptable": float(group["judge_acceptable"].mean()),
            "mean_overall": float(group["judge_overall"].mean()),
            "hallucination_flagged": float(group["judge_hallucination"].mean()),
        }
    defects = df[df["variant"] != "control"]
    summary = {
        "defect_detection_rate": float((~defects["judge_acceptable"]).mean()),
        "hallucination_catch_rate": float(df[df["variant"].isin(MUST_FLAG_HALLUCINATION)]["judge_hallucination"].mean()),
        "control_acceptance_rate": stress["control"]["judged_acceptable"],
    }

    single = ReplyJudge(batch_size=1).score_many(cases[:N_SINGLE_CHECK])
    b, s = pd.DataFrame(batched[:N_SINGLE_CHECK]), pd.DataFrame(single)
    batching = {
        "n": N_SINGLE_CHECK,
        "overall_exact_agreement": float((b["judge_overall"] == s["judge_overall"]).mean()),
        "overall_within_1": float(((b["judge_overall"] - s["judge_overall"]).abs() <= 1).mean()),
        "overall_weighted_kappa": M.weighted_kappa(b["judge_overall"], s["judge_overall"]),
        "acceptable_agreement": float((b["judge_acceptable"] == s["judge_acceptable"]).mean()),
    }

    result = {"summary": summary, "per_variant": stress, "batch_vs_single": batching}
    (EVAL_DIR / "judge_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
