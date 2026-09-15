"""Bundle evaluation outputs into docs/data.js for the static interactive report (GitHub Pages).

Usage: python src/build_report_site.py [--out docs/data.js]
"""
import argparse
import json
import math
from datetime import datetime, timezone

import pandas as pd

from agent import CONFIDENCE_THRESHOLD
from config import BRAND, DATA_PROCESSED, EVAL_DIR, LABELS_DIR, ROOT
from metrics import kappa
from run_eval import load_golden

SYSTEMS = ["agent", "simple", "trivial"]
CI_KEYS = ["intent_macro_f1", "intent_accuracy", "escalation_recall_ci", "reply_acceptable", "safe_automation_rate"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def clean(value):
    """Replace NaN with None so the output is valid JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def pick(result):
    out = {key: result[key] for key in CI_KEYS}
    out.update({
        "n": result["n"],
        "escalation_precision": result["escalation"]["precision"],
        "escalation_rate": result["escalation"]["escalation_rate"],
        "missed_escalations": result["escalation"]["missed_escalations"],
        "hallucination_rate": result["hallucination_rate"],
        "unsafe_automation_rate": result["unsafe_automation_rate"],
        "reply_overall_mean": result["reply_overall_mean"],
    })
    return out


def examples(split):
    gold = load_golden(split)
    pred = pd.read_csv(EVAL_DIR / f"predictions_{split}_agent.csv", dtype={"conv_id": str}, keep_default_na=False)
    judged = pd.read_csv(EVAL_DIR / f"judgments_{split}.csv", dtype={"conv_id": str}, keep_default_na=False)
    judged = judged[judged["system"] == "agent"][["conv_id", "judge_overall", "judge_acceptable", "judge_rationale"]]
    rows = []
    for r in gold.merge(pred, on="conv_id").merge(judged, on="conv_id").itertuples(index=False):
        escalate, acceptable = str(r.escalate) == "True", str(r.judge_acceptable) == "True"
        failures = [name for name, failed in [
            ("wrong_intent", r.intent != r.pred_intent),
            ("missed_escalation", r.should_escalate and not escalate),
            ("unneeded_escalation", escalate and not r.should_escalate),
            ("unacceptable_reply", not acceptable),
        ] if failed]
        rows.append({
            "conv_id": r.conv_id, "stratum": r.stratum, "context": r.context, "message": r.customer_text,
            "gold_intent": r.intent, "pred_intent": r.pred_intent, "confidence": float(r.confidence),
            "should_escalate": bool(r.should_escalate), "gold_reason": r.escalation_reason,
            "escalate": escalate, "escalation_reasons": r.escalation_reasons, "reply": r.reply,
            "judge_overall": int(r.judge_overall), "judge_rationale": r.judge_rationale, "failures": failures,
        })
    return rows


def label_consistency():
    first, second = LABELS_DIR / "golden_labels.csv", LABELS_DIR / "relabel_labels.csv"
    if not second.exists():
        return None
    both = pd.read_csv(first, dtype=str, keep_default_na=False).merge(
        pd.read_csv(second, dtype=str, keep_default_na=False), on="conv_id", suffixes=("_first", "_second"))
    if len(both) < 10:
        return None
    result = {
        "n": len(both),
        "intent_agreement": float((both["intent_first"] == both["intent_second"]).mean()),
        "intent_kappa": kappa(both["intent_first"], both["intent_second"]),
        "escalation_agreement": float((both["should_escalate_first"] == both["should_escalate_second"]).mean()),
        "escalation_kappa": kappa(both["should_escalate_first"], both["should_escalate_second"]),
    }
    (EVAL_DIR / "label_consistency.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "docs" / "data.js"))
    out_path = parser.parse_args().out

    test, dev = read_json(EVAL_DIR / "results_test.json"), read_json(EVAL_DIR / "results_dev.json")
    stats = read_json(DATA_PROCESSED / "prepare_stats.json") or {}
    codebook = read_json(LABELS_DIR / "codebook.json")
    labelled = pd.read_csv(LABELS_DIR / "golden_labels.csv", dtype=str)
    report = test["agent_intent_report"]

    data = {
        "meta": {
            "brand": BRAND, "pairs": stats.get("after_dedup"), "history": stats.get("history"),
            "test_pool": stats.get("test"), "split_cutoff": stats.get("split_cutoff"),
            "golden_labelled": len(labelled), "test_n": test["agent"]["n"], "dev_n": dev["agent"]["n"] if dev else None,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        },
        "intents": {i["id"]: i["name"] for i in codebook["intents"]},
        "results": {
            "test": {s: pick(test[s]) for s in SYSTEMS},
            "test_random": {s: pick(test[s]["random_stratum_only"]) for s in SYSTEMS},
        },
        "dial": test["automation_dial"],
        "agent_threshold": CONFIDENCE_THRESHOLD,
        "intent_report": [
            {"intent": k, "precision": v["precision"], "recall": v["recall"], "f1": v["f1-score"], "support": v["support"]}
            for k, v in report.items() if isinstance(v, dict) and k not in ("macro avg", "weighted avg")
        ],
        "judge_human": test["judge_human_agreement"],
        "judge_validation": read_json(EVAL_DIR / "judge_validation.json"),
        "label_consistency": label_consistency(),
        "trust": read_json(EVAL_DIR / "trust_policy.json"),
        "examples": examples("test"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("window.REPORT = " + json.dumps(clean(data), ensure_ascii=False) + ";\n")
    print(f"Wrote {out_path} ({len(data['examples'])} examples)")


if __name__ == "__main__":
    main()
