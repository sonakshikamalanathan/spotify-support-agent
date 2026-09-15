# Can we trust an AI support agent for @SpotifyCares?

*Hiver SDE Intern take-home · Sonakshi Kamalanathan · Interactive results: https://sonakshikamalanathan.github.io/spotify-support-agent/ · Decision log: [DECISIONS.md](DECISIONS.md)*

## 1. Problem framing

For each tweet to @SpotifyCares, the agent (1) classifies it into one of 9 intents, (2) drafts a reply grounded in how the brand handled similar tweets, and (3) decides whether it may reply on its own or must hand off to a human, with a reason.

**What "good" means here.** Replies are public, so errors are not equal:
- A **missed escalation** (auto-replying to a hacked account, a payment dispute or a furious customer) is the worst: public and hard to undo.
- A **wrong or invented reply** (a promised refund, a made-up policy) comes next, because a human has to walk it back.
- An **unneeded escalation** only costs agent time.

So "trustworthy" means catching nearly every tweet that needs a human, never inventing commitments, and automating only what can be shown to be safe. Section 6 prices this with a cost model: a human writing a reply = 1, approving a good draft = 0.3, a bad auto-reply = 3, a missed escalation = 10.

**Not built:** the DM conversation that follows most replies (no data, needs account access), non-English support (the brand redirects it), fine-tuning, live Twitter or helpdesk integration, calibrated confidence, and memory beyond the 4 preceding tweets.

## 2. Data and golden set

**Data.** From 2.8M tweets I rebuilt threads and kept each customer tweet with @SpotifyCares's *first* reply, plus up to 4 earlier turns: 39,732 conversations. Agent initials and Twitter's auto-attached DM link were stripped, and handles, URLs and duplicates normalised. The split is **by time**: the newest 20% (after 22 Nov 2017) is held out, because the brand reuses templates and a random split would leak near-identical replies into the test set.

**Intents.** TF-IDF + k-means over 15k historical tweets surfaced the themes (`reports/intent_clusters.md`). From them I wrote a 9-intent codebook (`labels/codebook.json`), organised by *how the brand must respond* rather than by topic words.

**Sampling.** 207 held-out tweets in three strata: random (90, estimates natural traffic), intent-balanced (80, makes rare intents measurable) and hard cases (40, escalation keywords and threads). The first 50 form a **dev split** for tuning; the other 157 are the **test split** behind every reported number.

**Labelling.** I labelled every tweet by hand in two local tools (`src/label_app.py`, `src/fast_label.py`): the intent, whether a human must handle it, the reason, and an "unsure" flag. Labelling was **blind** to the brand's reply and to all model output. The first 11 dev labels were a calibration round: an AI assistant checked them against the codebook's rules (not against model output) and I fixed rule violations, such as a hacked account I hadn't escalated. Every other label was made without assistance.

**Label noise.** Re-labelling 30 random tweets blind at the end, I gave the same intent 80% of the time (κ = 0.76) and the same escalation decision 87% of the time (κ = 0.63).

## 3. The system

```
tweet + thread ─► ANALYSE (gpt-oss-120b): intent, confidence, risk flag
               ├─► SECOND OPINION (TF-IDF + logistic regression on weak labels): intent
               ├─► RETRIEVE (TF-IDF over 31,785 historical pairs): 5 most similar past cases
               └─► DRAFT (qwen3.8-27b): reply of at most 280 characters, citing the cases used
                   DECIDE: escalate if any layer fires, recording every reason
                     1 keyword rules (security, payment, privacy/legal, safety)
                     2 always-escalate intent (hacked account)
                     3 intent confidence < 0.6
                     4 the second opinion disagrees with the LLM            (added after dev)
                     5 the LLM's own risk flag
                     6 draft promises a refund, credit or timeline, is empty,
                       or claims an action the agent can't take             (added after dev)
```

**Baselines.**
- *Trivial:* always the majority intent, always the brand's single most common reply, never escalate.
- *Simple:* TF-IDF + logistic regression on weak labels (clear k-means clusters mapped to intents by hand), the nearest past tweet's reply copied verbatim, and keyword-rule escalation.

## 4. Results vs baselines

*Test split: 157 hand-labelled tweets, 95% bootstrap confidence intervals. Safe automation = sent without a human, didn't need one, right intent, acceptable reply. Unsafe automation = sent without a human although one was needed.*

| Metric | Trivial | Simple | **Agent** |
|---|---|---|---|
| Intent accuracy | 14.0% [8.9, 19.7] | 51.6% [43.9, 59.9] | **79.0% [72.6, 84.7]** |
| Intent macro-F1 | 0.03 | 0.41 [0.35, 0.47] | **0.76 [0.68, 0.82]** |
| Escalation recall | 0% | 37.8% [21.9, 53.9] | **64.9% [48.5, 80.0]** |
| Escalation precision | n/a | 77.8% | 27.3% |
| Share sent to a human | 0% | 11.5% | 56.1% |
| Acceptable replies (judge) | 3.8% [1.3, 7.0] | 45.9% [38.2, 54.1] | **86.6% [80.9, 91.7]** |
| Hallucination rate (judge) | 8.9% | 6.4% | **0.6%** |
| Safe automation | 0% | 16.6% [10.8, 22.9] | **24.2% [17.2, 31.2]** |
| Unsafe automation | 23.6% | 14.6% | **8.3%** |

**Natural traffic only** (the 71 uniformly sampled tweets): agent intent accuracy 76.1% [66, 86], acceptable replies 81.7%, escalation recall 50.0%, unsafe automation 8.5%. The simple baseline scores 50.7%, 40.8%, 8.3% and 15.5%.

The agent understands tweets and writes sendable replies far better than both baselines: +27 points of intent accuracy with non-overlapping intervals, and 87% vs 46% acceptable replies. It is **safer, but not safe**: 13 of the 37 tweets that needed a human would have been answered automatically. It also escalates 56% of tweets, and only 27% of those escalations were needed.

## 5. Can we trust the judge?

The judge, `gpt-oss-20b`, is from a different model family than the Qwen reply writer. It scores each reply 1–5 plus a hallucination flag, using the same rubric the human rater used. `gpt-oss-120b` was the original judge, but it ran out of Groq's free 200K tokens per day partway through the test split, so every split was re-graded with `gpt-oss-20b`. That forced switch became check (d).

**(a) Planted defects.** I took 12 held-out conversations and graded each with the brand's real reply and six broken ones (`eval/judge_validation.json`):

| Broken reply | Rejected | Mean score |
|---|---|---|
| Invents a refund + free months | 12/12 | 1.5 |
| Invents a policy + fake link | 12/12 | 1.5 |
| Asks for the password publicly | 12/12 | 1.0 |
| Rude | 12/12 | 1.1 |
| Ignores the issue | 12/12 | 2.3 |
| Answers a different tweet | 9/12 | 2.8 |
| *Control: brand's real reply* | *accepted 8/12* | *3.8* |

The judge rejects **69 of 72** defects and flags every invented refund or policy as a hallucination. All three misses are generic off-topic replies ("we'll have it available as soon as it's available to us"). It also rejected 4 of the brand's own 12 replies, so judged quality is, if anything, conservative.

**(b) Batching.** The judge grades 12 cases per call. Re-grading 24 of them one at a time gave 92% agreement on acceptable vs not and 92% of scores within one point, but only 50% exact matches (weighted κ = 0.37). Scores move; send-or-don't decisions rarely do.

**(c) Human agreement.** I rated 60 shuffled replies from all three systems blind. On acceptable vs not, the judge and I agree **87% of the time (κ = 0.73)**. On the 1–5 score, weighted κ = 0.47: 40% exact and 83% within one point. On hallucination, agreement is weak (κ = 0.17), because I flagged far more replies than the judge did. The judge also scores 0.32 points higher than I do, and its scores track reply length a little more than mine (ρ = 0.27 vs 0.17).

**(d) A different judge.** `gpt-oss-120b` and `gpt-oss-20b` graded the same 150 dev replies. They agree on acceptability 86% of the time (κ = 0.71), with no overall leniency gap, yet rate the agent's replies 84% vs 70% acceptable.

## 6. How much should we automate?

**The confidence dial is flat.** 147 of 157 test predictions report confidence of at least 0.9 (`reports/automation_dial_test.png`), so no threshold can buy safety. Instead, I priced rollout policies on the test set, per 100 tweets, using the cost model from section 1:

| Policy | Cost | Sent with no human | Missed escalations | Bad replies sent |
|---|---|---|---|---|
| Humans write every reply | 100 | 0% | 0 | 0 |
| **AI drafts, a human approves** | **67** | 0% | 0 | 0 |
| AI sends unless it escalates | 157 | 44% | 13 | 18 |
| Tiered: auto-send only intents proven on dev | 67 | 0% | 0 | 0 |

**No intent earned auto-send.** With only 2 to 12 dev tweets per intent, none reached the required smoothed failure rate of 20% or less (the best, billing, was at 29%), so the tiered policy is the same as drafting.

**Recommendation: ship it as a drafting assistant, not an auto-responder.** Drafting cuts handling cost by a third with zero unreviewed replies. Auto-sending costs *more* than using no AI at all (157 vs 100), and that holds whether a missed escalation is priced at 5 (cost 116) or 25 (cost 281).

**Rollout:**
1. **Shadow:** log the agent's drafts next to the replies humans actually send.
2. **Suggest:** agents edit or approve drafts, and the edit rate is tracked per intent.
3. **Auto-send per intent:** only once the evidence supports it, starting with *Thanks / resolved* (12 auto-handled on test, 2 failures). *Playback* is the furthest away (13 failures in 16).

## 7. Failure analysis: top 5 failure modes

*Counts from `eval/failure_analysis_test.json`; examples are real test tweets.*

**1. Subjective escalations are missed.** Every hacked-account (9/9), payment-dispute (7/7) and privacy (1/1) case was caught. "High frustration" (caught 5 of 10) and "needs investigation" (2 of 10) account for all 13 misses, e.g. *"What's the point when you have 4000 votes from 4 years ago and you haven't done it."*, which was auto-handled as feature feedback.
- *Hypothesis:* rules and the risk flag react to explicit words, while frustration lives in tone, persistence and thread history, which are also the most subjective labels.
- *Fix:* thread-level signals ("still", "again", repeat complaints) and few-shot examples of these two reasons.

**2. Thread follow-ups.** Intent accuracy is 87.1% on standalone tweets but 67.2% inside threads (93 vs 64 tweets). An example is *"9 pesos?"* replying to a "3 months of Premium for ₱9" advert. Several of these are ambiguous for a human too.
- *Hypothesis:* classification and retrieval lean on the short latest message.
- *Fix:* classify and retrieve on the whole thread, and add a codebook rule for replies to adverts.

**3. The second opinion chosen on dev didn't generalise.** On test it was the only reason for 51 unnecessary escalations (in 35 of them the agent's intent was already right), and it rescued just 3 real escalations. The weak classifier behind it is only 52% accurate.
- *Hypothesis:* it was chosen from 21 dev disagreements, too few to separate signal from noise.
- *Fix:* use it only as a review flag on high-risk intents, or replace it with self-consistency sampling.

**4. Confidence carries no information.** 147 of 157 predictions report confidence of at least 0.9, so the low-confidence layer almost never fires.
- *Fix:* a calibrated trust signal (section 9).

**5. Safe but generic replies, and a judge blind spot.** Of 21 unacceptable agent replies, only 1 was a hallucination. Most ask the customer to DM without the troubleshooting steps the retrieved history offered. In 3 cases the judge wrongly treated "can you DM us your account's email?" as a public request for sensitive data. Separately, 7 drafts claimed "we've sent you a DM" and were stopped by the claimed-action check.
- *Hypothesis:* the drafter copies the most frequent retrieved template.
- *Fix:* rank retrieved examples by specificity, and tell the judge that asking to DM account details is fine.

## 8. What is misleading about my headline number?

"79% intent accuracy and 87% acceptable replies, beating both baselines" is true on this test set and still misleading:

1. **The test set isn't natural traffic.** 43% of it is intent-balanced and 19% hard cases, by design. On the 71 random tweets alone, escalation recall is **50%**, not 65%.
2. **Escalation recall rests on 37 tweets, and dev overstated it** (92% on 13 dev cases). A handful of tweets moves it by 15 points.
3. **"8% unsafe automation" is 1 tweet in 12.** In production, that is a public, unreviewed reply to someone who needed a person, many times a day.
4. **One labeller, who also wrote the codebook.** My blind self-agreement on intent is 80%, which limits how precise any accuracy figure can be, and some follow-up "errors" are arguable.
5. **"Acceptable" is judged, not measured.** The judge agrees with me on acceptability 87% of the time (κ = 0.73), but swapping the judge model alone moved the agent's dev rate from 84% to 70%. The judge also cannot see outages, account state or DMs.
6. **Good-looking replies often resolve nothing.** Many route the customer to DMs, where the real fix happens in private and cannot be measured.
7. **Eleven days, one brand, 2017.** The window includes one-off events (*reputation* missing from Spotify, a ₱9 / Rp 4,990 promo campaign), so this is a snapshot.

## 9. With one more week

1. **A second annotator** on 60 tweets, with disagreements settled into a second version of the codebook.
2. **A trust signal that tracks mistakes:** self-consistency sampling, or a small calibrated model over existing signals, checked with a reliability curve.
3. **A shadow-mode pilot:** log would-be replies next to human ones, and promote an intent to auto-send only with enough evidence.
4. **Better grounding:** embedding retrieval, template de-duplication, and weighting past replies by whether the customer stopped complaining.
5. **A sturdier judge:** more human ratings, tests for position, length and batch-size effects, a second judge family, and a small paid budget so free-tier limits can't force model changes.
6. **Robustness and drift:** tests with typos, sarcasm and code-switching, and weekly monitoring of the intent mix and escalation rate.

## Reproducibility

Every LLM response and retrieval ranking is committed, so `OFFLINE=1 python src/run_eval.py --split test` reproduces every table without API keys, and CI checks this on every push. See `README.md`.
