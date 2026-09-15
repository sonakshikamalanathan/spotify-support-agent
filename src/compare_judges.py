"""Compare the two LLM judges on the dev split: gpt-oss-120b (archived) vs gpt-oss-20b (final).

Both graded the same 150 dev replies against the same rubric, so their agreement shows how much the
reply-quality numbers depend on which judge model was used.

Usage: python src/compare_judges.py   (needs eval/judge_120b/judgments_dev.csv and eval/judgments_dev.csv)
"""
import json

import pandas as pd

import metrics as M
from config import EVAL_DIR


def load(path):
    df = pd.read_csv(path, dtype={"conv_id": str}, keep_default_na=False)
    df["judge_acceptable"] = df["judge_acceptable"].astype(str).eq("True")
    return df[["conv_id", "system", "judge_overall", "judge_acceptable"]]


def main():
    df = load(EVAL_DIR / "judge_120b" / "judgments_dev.csv").merge(
        load(EVAL_DIR / "judgments_dev.csv"), on=["conv_id", "system"], suffixes=("_120b", "_20b"))
    result = {
        "n": len(df),
        "overall_exact_agreement": float((df["judge_overall_120b"] == df["judge_overall_20b"]).mean()),
        "overall_within_1": float(((df["judge_overall_120b"] - df["judge_overall_20b"]).abs() <= 1).mean()),
        "overall_weighted_kappa": M.weighted_kappa(df["judge_overall_120b"], df["judge_overall_20b"]),
        "acceptable_agreement": float((df["judge_acceptable_120b"] == df["judge_acceptable_20b"]).mean()),
        "acceptable_kappa": M.kappa(df["judge_acceptable_120b"], df["judge_acceptable_20b"]),
        "mean_score_20b_minus_120b": float((df["judge_overall_20b"] - df["judge_overall_120b"]).mean()),
        "acceptable_rate_by_system": {
            system: {"gpt-oss-120b": round(float(g["judge_acceptable_120b"].mean()), 3),
                     "gpt-oss-20b": round(float(g["judge_acceptable_20b"].mean()), 3)}
            for system, g in df.groupby("system")
        },
    }
    (EVAL_DIR / "judge_comparison_dev.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
