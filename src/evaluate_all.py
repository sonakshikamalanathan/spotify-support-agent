"""Run the complete evaluation once the golden set is labelled: dev and test splits, the
rollout-policy analysis, and the data file for the interactive report page.

Usage: python src/evaluate_all.py            (OFFLINE=1 replays cached LLM responses only)
"""
import sys

import build_report_site
import run_eval
import trust_policy


def main():
    for split in ("dev", "test"):
        sys.argv = ["run_eval.py", "--split", split]
        run_eval.main()
    trust_policy.main()
    sys.argv = ["build_report_site.py"]
    build_report_site.main()


if __name__ == "__main__":
    main()
