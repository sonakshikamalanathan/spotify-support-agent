# Spotify support agent: can we trust it?

An AI support agent for **@SpotifyCares**, built from real customer-support tweets (Kaggle *Customer Support on Twitter*). For every incoming tweet it:

1. **classifies** it into one of 9 intents derived from the data,
2. **drafts a reply** grounded in how @SpotifyCares handled similar tweets before, and
3. **decides** whether to auto-handle or escalate to a human, with the reasons.

Most of this repo is about the second question: **is it good enough to trust, and where exactly is it not?**

> Hiver SDE Intern take-home · Sonakshi Kamalanathan

## Headline results

Test split: 157 hand-labelled tweets from the held-out period, with 95% bootstrap confidence intervals in brackets.

| Metric | Trivial baseline | Simple baseline | **Agent** |
|---|---|---|---|
| Intent accuracy | 14.0% | 51.6% | **79.0%** [72.6, 84.7] |
| Intent macro-F1 | 0.03 | 0.41 | **0.76** [0.68, 0.82] |
| Escalation recall (needed a human → got one) | 0% | 37.8% | **64.9%** [48.5, 80.0] |
| Acceptable replies (LLM judge) | 3.8% | 45.9% | **86.6%** [80.9, 91.7] |
| Hallucination rate (LLM judge) | 8.9% | 6.4% | **0.6%** |
| Unsafe automation (auto-answered but needed a human) | 23.6% | 14.6% | **8.3%** |

The agent is much better than both baselines, and still not safe to run unsupervised: on natural traffic alone, escalation recall is only 50%. The full story, including what is misleading about these numbers, is in [REPORT.md](REPORT.md).

## Reproduce the headline numbers (about 2 minutes, no API keys)

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
OFFLINE=1 python src/run_eval.py --split test     # PowerShell: $env:OFFLINE="1"; python src/run_eval.py --split test
```

`OFFLINE=1` replays the committed LLM responses in `cache/`, so the evaluation is deterministic and needs no keys. CI runs the same command on every push and fails if any headline number drifts ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## Run everything from scratch

Needs a Kaggle download of the dataset, and free `GROQ_API_KEY` / `GEMINI_API_KEY` values in `.env` (see `.env.example`).

| Step | Command | Output |
|---|---|---|
| 1. Get data | `python src/download.py` | `data/raw/twcs/twcs.csv` |
| 2. Build brand pairs | `python src/prepare_data.py` | `data/processed/spotify_pairs.csv` |
| 3. Discover intents | `python src/discover_intents.py` | `reports/intent_clusters.md` |
| 4. Sample golden set | `python src/make_golden_set.py` | `labels/golden_candidates.csv` |
| 5. Label | `python src/fast_label.py` (keyboard) or `streamlit run src/label_app.py` | `labels/golden_labels.csv` |
| 6. Evaluate | `python src/run_eval.py --split dev` and `--split test` | `eval/results_*.json` |
| 7. Validate the judge | `python src/judge_validation.py` | `eval/judge_validation.json` |
| 8. Blind human ratings | `python src/make_rating_items.py`, then the app's "Rate replies" task | `labels/human_reply_ratings.csv` |
| 9. Rollout policy | `python src/trust_policy.py` | `eval/trust_policy.json` |

Unit tests: `pytest -q`

## Repo layout

```
src/        pipeline, agent, baselines, judge, evaluation
labels/     codebook, golden set candidates and hand labels, human ratings
eval/       predictions, judge scores, metrics (JSON)
reports/    cluster report, plots
cache/      committed LLM responses (offline reproduction)
tests/      unit tests
REPORT.md   the report
DECISIONS.md  decision log
```

## Citations and borrowed work
- **Dataset:** *Customer Support on Twitter*, Thought Vector, Kaggle (`thoughtvector/customer-support-on-twitter`). Only a processed @SpotifyCares subset is committed.
- **Models:** `openai/gpt-oss-120b` and `qwen/qwen3.8-27b` via the Groq API.
- **Libraries:** pandas, scikit-learn (TF-IDF, k-means, logistic regression, Cohen's kappa), matplotlib, Streamlit, google-genai, groq, pytest.
- **Methods:** LLM-as-judge with human agreement checks follows common practice (e.g. Zheng et al., 2023, *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*). Percentile bootstrap confidence intervals, Wilson/Laplace smoothing and Cohen's (weighted) kappa are standard statistics.
- **AI assistance:** built with Claude Code as a coding assistant. All design decisions are listed in `DECISIONS.md`, and the golden-set labels and human ratings were produced by hand.
