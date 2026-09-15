"""CI check: re-running the evaluation offline must reproduce the committed headline numbers."""
import json
import subprocess
import sys

from config import EVAL_DIR, ROOT

SYSTEMS = ["trivial", "simple", "agent"]
HEADLINE = ["intent_accuracy", "intent_macro_f1", "escalation_recall_ci", "reply_acceptable", "safe_automation_rate"]
TOLERANCE = 0.005


def main():
    committed = json.loads(subprocess.run(["git", "show", "HEAD:eval/results_test.json"], cwd=ROOT,
                                          capture_output=True, text=True, check=True).stdout)
    current = json.loads((EVAL_DIR / "results_test.json").read_text())
    failures = []
    for system in SYSTEMS:
        for metric in HEADLINE:
            expected, actual = committed[system][metric][0], current[system][metric][0]
            status = "ok" if abs(expected - actual) <= TOLERANCE else "MISMATCH"
            print(f"{status:9} {system:8} {metric:24} committed={expected:.4f} reproduced={actual:.4f}")
            if status != "ok":
                failures.append((system, metric))
    if failures:
        sys.exit(f"{len(failures)} headline metrics did not reproduce")
    print("All headline metrics reproduced.")


if __name__ == "__main__":
    main()
