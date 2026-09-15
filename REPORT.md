# Can we trust an AI support agent for @SpotifyCares?

*Hiver SDE Intern take-home · Sonakshi Kamalanathan · Interactive version: `docs/index.html` · Decision log: `DECISIONS.md`*

> **Draft.** Sections marked ⏳ are filled in after the golden set is fully labelled and the evaluation has run.

## 1. Problem framing

**The job.** @SpotifyCares answers public tweets. For each incoming tweet the agent must (1) classify it into one of 9 intents, (2) draft a reply in the brand's voice grounded in how the brand handled similar tweets, and (3) decide whether it may reply on its own or must hand off to a human, with a reason.

**What "good" means for this brand.** A public reply is visible to everyone, so the costs are asymmetric:
- A **missed escalation** (auto-replying to a hacked account, a payment dispute, a legal threat or a furious customer) is the worst outcome: public, hard to undo, and potentially harmful.
- A **wrong or invented reply** (a promised refund, a made-up policy) is next worst, because it creates commitments a human then has to walk back.
- An **unneeded escalation** only costs agent time.

So "good enough to trust" is not one accuracy number. It means: **catch nearly every tweet that needs a human, never invent commitments, and automate as much of the rest as can be shown to be safe.** The cost model in §6 turns this into numbers (human writes a reply = 1, approves a good draft = 0.3, bad auto-reply = 3, missed escalation = 10).

**What I chose not to build.** The DM conversation that follows most replies (no data, needs account access). Non-English support (the brand redirects it). Fine-tuning or training on the golden set. Live Twitter or helpdesk integration. Calibrated confidence (the LLM's self-reported confidence is used as-is, and its weakness is measured). Multi-turn memory beyond the 4 preceding tweets.

## 2. Data and golden set

**From 2.8M tweets to 39,732 conversations.** Threads were rebuilt from reply links, keeping each customer tweet and @SpotifyCares's *first* reply to it, plus up to 4 earlier turns as context. Agent initials and Twitter's auto-attached DM link were stripped from replies; masked handles, URLs and duplicates were normalised. The split is **by time**: conversations after 22 Nov 2017 (20%) are held out. The brand reuses templates heavily, so a random split would leak near-identical replies into the test set.

**Intents.** TF-IDF + k-means over 15k historical tweets exposed the themes (`reports/intent_clusters.md`). I wrote a 9-intent codebook with definitions, includes/excludes and labelling rules (`labels/codebook.json`), organised by *how the brand must respond* rather than by topic words.

**Sampling (207 tweets, all from the held-out period).**

| Stratum | n | Why |
|---|---|---|
| Random | 90 | Estimates performance on natural traffic |
| Intent-balanced | 80 | Makes rare intents measurable (equal draws per weak-label intent) |
| Hard cases | 40 | Escalation-keyword hits and multi-turn threads, to stress escalation |

The first 50 (shuffled) form a **dev split** used for tuning and choosing the trust policy; the other 157 are the **test split** behind every reported number.

**Labelling.** I labelled every tweet by hand in a small Streamlit tool (`src/label_app.py`): intent, whether a human must handle it, the escalation reason, and an "unsure" flag. I labelled **blind**: the brand's actual reply and all model predictions were hidden. The first 11 dev tweets were a calibration round; after them I re-read the codebook's rules (for example, that hacked accounts always need a human) and corrected the labels that broke them. ⏳ *Label noise:* I re-labelled 30 random tweets blind at the end; self-agreement was __% on intent (κ = __) and __% on escalation.

## 3. The system

```
tweet + thread ──► ANALYSE (gpt-oss-120b): intent, confidence, risk flag
             │
             ├──► RETRIEVE (TF-IDF over 31,785 historical pairs): 5 most similar past cases
             │
             └──► DRAFT (qwen3.8-27b): ≤280-char reply citing the past cases it used
                                   │
                  DECIDE: escalate if any layer fires, and record every reason:
                    1 keyword rules (security, payment, privacy/legal, safety)
                    2 always-escalate intents (hacked account)
                    3 intent confidence < 0.6
                    4 the LLM's own risk flag
                    5 the draft promises a refund, credit or timeline, or is empty
```

**Baselines.**
- **Trivial:** always the majority intent, always the brand's single most common reply ("DM us your account's email…"), never escalate.
- **Simple:** TF-IDF + logistic regression trained on weak labels (each clear k-means cluster mapped to an intent by hand), the brand reply of the nearest past tweet copied verbatim, and keyword rules for escalation.

## 4. Results vs baselines ⏳

*Test split, n = 157, 95% bootstrap CIs. "Safe automation" = sent without a human, didn't need one, right intent, and a reply the judge would send.*

| Metric | Trivial | Simple | Agent |
|---|---|---|---|
| Intent macro-F1 | | | |
| Intent accuracy | | | |
| Escalation recall | | | |
| Escalation precision | | | |
| Acceptable replies (judge) | | | |
| Hallucination rate (judge) | | | |
| Safe automation rate | | | |
| Unsafe automation rate | | | |

## 5. Can we trust the judge?

The judge (`gpt-oss-120b`, a different model family from the Qwen reply writer) scores each reply 1–5 plus a hallucination flag against the same rubric shown to the human rater. It is validated three ways.

**(a) Planted-defect stress test** (`eval/judge_validation.json`). I took 12 real held-out conversations and graded the brand's own reply (control) alongside six deliberately broken replies:

| Broken reply | Rejected by judge | Mean score |
|---|---|---|
| Invents a refund + free months | 12/12 | 1.2 |
| Invents a policy + fake link | 12/12 | 1.8 |
| Asks for the password publicly | 12/12 | 1.0 |
| Rude | 12/12 | 1.0 |
| Ignores the issue | 12/12 | 1.8 |
| Answers a different tweet | 11/12 | 2.3 |
| *Control: brand's real reply* | *accepted 8/12* | *3.8* |

The judge catches **71/72 planted defects**, and flags every invented refund or policy as a hallucination. Its one miss was an off-topic reply generic enough to fit many catalogue tweets ("we'll have it available as soon as it's available to us"). It is **strict**: 4 of the brand's own replies were rejected, 3 of them "we had a hiccup, it should work now". The judge cannot know an outage really happened, so judge-graded reply quality is a conservative estimate.

**(b) Batching.** To fit free-tier request quotas the judge grades 6 unrelated cases per call. Re-grading 24 cases one per call: 92% agree on acceptable/not, 92% of scores are within 1 point, 71% match exactly (weighted κ = 0.65). Batching shifts individual scores a little but rarely flips the send/don't-send decision.

**(c) Agreement with a human** ⏳. 60 replies from all three systems were shuffled with the system hidden, and I rated them blind on the same rubric: weighted κ = __, acceptable/not agreement = __%, mean judge − human = __.

## 6. How much should we automate? ⏳

- Automation dial (share auto-handled vs error among auto-handled, by confidence threshold)
- Per-intent tiers chosen on dev; cost per 100 tickets on test for: all human / suggest only / auto everything / tiered

## 7. Failure analysis: top 5 failure modes ⏳

*(Real examples from `eval/failures_test.csv`, each with a hypothesis and a fix.)*

## 8. What is misleading about my headline number?

Known before seeing the final numbers (to be quantified ⏳):

1. **The test set is not natural traffic.** 43% of it is deliberately intent-balanced and 19% is hard cases. Numbers on the random stratum alone are reported separately, and they differ.
2. **One labeller, and that labeller is also the builder.** Every "correct" label reflects my reading of a codebook I wrote. The re-label self-agreement bounds how precise any accuracy figure can be; a second annotator would likely lower it further.
3. **"Acceptable reply" is judged, not measured.** Even a well-validated judge accepted only 8 of 12 of the brand's own replies. It shares blind spots with the LLMs it grades (it cannot know about outages, account state or what happens in DMs).
4. **Many good-looking replies resolve nothing.** About a third of the brand's real replies are "DM us your email". Replies like that score as acceptable, but the customer's problem is solved later in private, where we cannot measure.
5. **Eleven days of test data.** The held-out window (22 Nov – 3 Dec 2017) includes one-off events (for example, Taylor Swift's *reputation* not being on Spotify). Intent mix and reply templates drift, so this is a snapshot.
6. **Confidence is self-reported.** The automation dial thresholds the LLM's own confidence number, which is not calibrated, so small threshold changes can move automation a lot.
7. **The agent saw the codebook, and the codebook was shaped by the same data.** Intents were designed on the history period only, but still by the person who then labelled the test set.

## 9. With one more week ⏳

## Reproducibility and citations
See `README.md`. Every LLM response is cached in `cache/`, so `OFFLINE=1 python src/run_eval.py --split test` reproduces the tables without API keys, and CI checks this on every push.
