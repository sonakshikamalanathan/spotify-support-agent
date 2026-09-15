# Can we trust an AI support agent for @SpotifyCares?

*Hiver SDE Intern take-home · Sonakshi Kamalanathan · Interactive results page: https://sonakshikamalanathan.github.io/spotify-support-agent/ · Decision log: [DECISIONS.md](DECISIONS.md)*

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

**Labelling.** I labelled every tweet by hand in two small local tools, a Streamlit app (`src/label_app.py`) and a keyboard-driven page (`src/fast_label.py`): intent, whether a human must handle it, the escalation reason, and an "unsure" flag. I labelled **blind**: the brand's actual reply and all model predictions were hidden. The first 11 dev tweets were a calibration round: an AI assistant checked those labels against the codebook's rules (not against any model output), and I corrected the ones that broke a rule, for example a hacked account I had not escalated. Every other label was made without assistance. *Label noise:* at the end I re-labelled 30 random tweets blind. I gave the same intent 80% of the time (κ = 0.76, substantial) and the same escalation decision 87% of the time (κ = 0.63).

## 3. The system

```
tweet + thread ──► ANALYSE (gpt-oss-120b): intent, confidence, risk flag
             │
             ├──► SECOND OPINION (TF-IDF + logistic regression on weak labels): intent
             │
             ├──► RETRIEVE (TF-IDF over 31,785 historical pairs): 5 most similar past cases
             │
             └──► DRAFT (qwen3.8-27b): ≤280-char reply citing the past cases it used
                                   │
                  DECIDE: escalate if any layer fires, and record every reason:
                    1 keyword rules (security, payment, privacy/legal, safety)
                    2 always-escalate intents (hacked account)
                    3 intent confidence < 0.6
                    4 the second opinion disagrees with the LLM's intent      (added after dev)
                    5 the LLM's own risk flag
                    6 the draft promises a refund, credit or timeline, is empty,
                      or claims an action the agent cannot take               (added after dev)
```

**Baselines.**
- **Trivial:** always the majority intent, always the brand's single most common reply ("DM us your account's email…"), never escalate.
- **Simple:** TF-IDF + logistic regression trained on weak labels (each clear k-means cluster mapped to an intent by hand), the brand reply of the nearest past tweet copied verbatim, and keyword rules for escalation.

## 4. Results vs baselines

*Test split: 157 hand-labelled tweets from the held-out period, with 95% bootstrap confidence intervals. "Safe automation" = sent without a human, didn't need one, right intent, and a reply the judge would send. "Unsafe automation" = sent without a human even though one was needed.*

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

**Natural traffic only** (the 71 uniformly sampled test tweets): agent intent accuracy 76.1% [66, 86], acceptable replies 81.7%, escalation recall 50.0%, unsafe automation 8.5%. The simple baseline scores 50.7%, 40.8%, 8.3% and 15.5% on the same tweets.

What the numbers say:
- **The agent understands tweets and writes sendable replies far better than both baselines**: +27 points of intent accuracy over the simple baseline, with non-overlapping intervals, and 87% vs 46% acceptable replies.
- **It is safer, but not safe.** 13 of the 37 tweets that needed a human would have been answered automatically (8.3% of all tweets).
- **It escalates a lot.** 56% of tweets go to a human, and only 27% of those needed one; section 6 prices that trade-off.

## 5. Can we trust the judge?

The judge scores each reply 1–5 plus a hallucination flag, against the same rubric shown to the human rater. It is `gpt-oss-20b`, a different model family from the Qwen reply writer. It was not the first choice: `gpt-oss-120b` graded dev and the first stress test, then hit Groq's free limit of 200K tokens per day partway through the test split, so every split was re-graded with `gpt-oss-20b`. That forced switch became a fourth check, (d).

**(a) Planted-defect stress test** (`eval/judge_validation.json`). I took 12 real held-out conversations and graded the brand's own reply (control) alongside six deliberately broken replies:

| Broken reply | Rejected by judge | Mean score |
|---|---|---|
| Invents a refund + free months | 12/12 | 1.5 |
| Invents a policy + fake link | 12/12 | 1.5 |
| Asks for the password publicly | 12/12 | 1.0 |
| Rude | 12/12 | 1.1 |
| Ignores the issue | 12/12 | 2.3 |
| Answers a different tweet | 9/12 | 2.8 |
| *Control: brand's real reply* | *accepted 8/12* | *3.8* |

The judge rejects **69 of 72 planted defects** and flags every invented refund or policy as a hallucination. All three misses are off-topic replies generic enough to fit many tweets ("we'll have it available as soon as it's available to us"), so its weakest spot is noticing a reply that answers a *different* question. It is strict with real replies too, rejecting 4 of the brand's own 12, so judge-graded reply quality is if anything a conservative estimate.

**(b) Batching.** To stay within free-tier limits the judge grades 12 unrelated cases per call. Re-grading 24 cases one per call: 92% agree on acceptable vs not and 92% of scores are within 1 point, but only 50% match exactly (weighted κ = 0.37). Batching moves individual scores, but rarely flips the send/don't-send decision the headline numbers use.

**(c) Agreement with a human.** 60 replies from all three systems were shuffled with the system hidden, and I rated them blind on the same rubric. On the 1–5 score the judge and I agree at weighted κ = 0.47 (moderate): 40% exactly and 83% within one point. On the decision the headline uses, acceptable or not, we agree 87% of the time (κ = 0.73); on hallucination, κ = 0.17. The judge scores 0.32 points higher than I do on average, and reply length correlates with the judge's score at ρ = 0.27 and with mine at ρ = 0.17.

**(d) A different judge.** `gpt-oss-120b` and `gpt-oss-20b` graded the same 150 dev replies. They agree on acceptable vs not 86% of the time (κ = 0.71), with no overall leniency difference (mean score +0.03). On the agent's replies, though, they differ more: 84% vs 70% acceptable. The choice of judge alone moves the agent's reply-quality number by about 14 points.

## 6. How much should we automate?

**The confidence dial is flat.** Sweeping the agent's confidence threshold changes almost nothing below 0.9, because 147 of 157 test predictions report at least 0.9 (`reports/automation_dial_test.png`). A threshold cannot buy safety here, so the decision is made by rollout mode and per intent instead.

**Cost of four rollout policies on the test set**, per 100 tweets. A human writing a reply costs 1, a human approving a usable AI draft 0.3, a bad auto-reply 3, and a missed escalation 10.

| Policy | Cost / 100 tweets | Sent with no human | Missed escalations | Bad replies sent |
|---|---|---|---|---|
| Humans write every reply | 100 | 0% | 0 | 0 |
| **AI drafts, a human approves every reply** | **67** | 0% | 0 | 0 |
| AI sends unless it escalates | 157 | 44% | 13 | 18 |
| Tiered: auto-send only intents proven on dev | 67 | 0% | 0 | 0 |

**No intent earned auto-send.** Tiers were chosen on dev: an intent qualifies only if its smoothed failure rate among auto-handled tweets is at most 20%. With 2 to 12 dev tweets per intent none got there (the best, billing, was at 29%), so the tiered policy collapses to "draft for approval".

**Recommendation for Hiver: ship it as a drafting assistant, not an auto-responder.** Drafting cuts handling cost by a third with zero unreviewed public replies. Letting it send on its own costs *more* than using no AI at all (157 vs 100), because 13 missed escalations and 18 bad replies outweigh the automation. That conclusion holds whether a missed escalation is priced at 5 (cost 116) or 25 (cost 281). The closest candidate for future auto-send is *Thanks / resolved* (12 auto-handled on test, 2 failures); the furthest is *Playback* (16 auto-handled, 13 failures).

**Rollout plan.**
1. *Shadow*: the agent drafts silently while humans work as usual, and every draft is logged next to what the human actually sent.
2. *Suggest*: agents see the draft and edit or approve it; track the edit rate per intent.
3. *Auto-send per intent*: promote an intent only once shadow and suggest data give it enough examples with a low failure rate, starting with *Thanks / resolved*.

## 7. Failure analysis: top 5 failure modes

*Counts from `eval/failure_analysis_test.json`; the examples are real test tweets.*

**1. Objective escalations are caught; subjective ones are missed.** Every hacked-account (9/9), payment-dispute (7/7) and privacy (1/1) escalation was caught. "High frustration" was caught 5/10 and "needs investigation" 2/10, and those two reasons account for all 13 misses. Examples: *"Unfortunately I've had that problem for a very long time."* (a follow-up, auto-handled as a playback problem); *"What's the point when you have 4000 votes from 4 years ago and you haven't done it."* (auto-handled as feature feedback).
*Hypothesis:* rules and the LLM's risk flag react to explicit words (hacked, refund, lawyer). Frustration and "a person needs to look at this" live in tone, persistence and thread history, which the prompt barely describes and which are also the most subjective labels. *Fix:* thread-level signals (a second complaint, "still", "again", "for weeks") and few-shot examples of these two reasons.

**2. Thread follow-ups.** Intent accuracy is 87.1% on standalone tweets but 67.2% on replies inside a thread (93 vs 64 tweets). Many are short replies to Spotify's own promo tweets (*"9 pesos?"* under a "3 months of Premium for ₱9" ad), or updates that mix two intents (*"thanks! It worked for about 24 hours and upon another restart the searching... no longer works"*). Several are arguably ambiguous for a human too.
*Hypothesis:* classification and retrieval lean on the short latest message, and replies to adverts look like billing questions. *Fix:* retrieve and classify on the whole thread, and add a codebook rule for replies to brand adverts.

**3. The second-opinion layer chosen on dev did not generalise.** On test it was the *only* reason for 51 unnecessary escalations, and on 35 of those the agent's intent was already correct. It rescued just 3 real escalations the other layers missed. The weak classifier is itself only 52% accurate, so its disagreements are mostly its own mistakes. Example: *"Put 'Is Your Love Enough' on Today's Top Hits playlist, please"* was correctly read as feature feedback, flagged as playback by the second opinion, and escalated.
*Hypothesis:* the dev evidence was 21 disagreements, too few to separate signal from noise. *Fix:* use disagreement only as a review flag on high-risk intents, or replace it with self-consistency sampling.

**4. The model's confidence carries no information.** 147 of 157 test predictions report confidence of at least 0.9. The "low confidence" layer almost never fires, and the automation dial (`reports/automation_dial_test.png`) is flat until 0.9, so a threshold cannot trade automation for safety.
*Fix:* a calibrated signal (section 9, item 2).

**5. Replies that are safe but generic, plus a judge blind spot.** 21 agent replies were judged unacceptable, but only 1 was flagged as a hallucination. Most ask the customer to DM their email without the troubleshooting steps the retrieved history offered. In 3 cases the judge penalised "can you DM us your account's email address?" as a *public* request for sensitive data, which it is not: it is exactly what @SpotifyCares does. Separately, 7 drafts claimed "we've sent you a DM", an action the agent cannot take, and were stopped by the claimed-action check.
*Hypothesis:* the drafter copies the most frequent retrieved template, and the judge reads "email address" as sensitive. *Fix:* rank retrieved examples by specificity, and state in the rubric that asking to DM account details is acceptable.

## 8. What is misleading about my headline number?

The headline, "79% intent accuracy and 87% acceptable replies, beating both baselines", is true on this test set and still misleading in at least seven ways:

1. **The test set is harder, and shaped differently, than real traffic.** 43% of it is intent-balanced and 19% hard cases, on purpose. On the 71 uniformly sampled tweets alone, intent accuracy is 76%, acceptable replies 82%, and escalation recall **50%** (not 65%), with wider intervals.
2. **Escalation recall rests on 37 tweets, and dev overstated it.** It was 92% on the 13 dev escalations and 65% on test; the second-opinion layer chosen on dev mostly added noise on test (section 7, item 3). A handful of tweets moves this number by 15 points.
3. **"8% unsafe automation" sounds small, but it is 1 tweet in 12.** In production that is a public, unreviewed reply to someone who needed a person, many times a day.
4. **One labeller, who also wrote the codebook.** Every "correct" answer is one person's reading of their own definitions, and several follow-up "errors" in section 7 are arguably label disagreements. Re-labelling 30 tweets blind gave 80% intent self-agreement, which bounds how precise these accuracies can be.
5. **"Acceptable reply" is judged, not measured.** The judge agreed with my blind ratings on acceptability 87% of the time (κ = 0.73). Swapping the judge model alone moved the agent's acceptable-reply rate on dev from 84% to 70%. The judge also has concrete blind spots: it penalised "DM us your email" as a public data request in 3 cases, and it cannot know about outages, account state or what happens in DMs.
6. **A good-looking reply often resolves nothing.** Most acceptable replies route the customer to DMs, which is what the brand does, but the actual fix happens later in private, where we cannot measure it.
7. **Eleven days, one brand, 2017.** The held-out window includes one-off events (Taylor Swift's *reputation* missing from Spotify) and a promo campaign in the Philippines and Indonesia that produced many "is this price real?" replies. Intent mix and templates drift, so this is a snapshot.

## 9. With one more week

1. **A second annotator.** Have someone else label 60 of the golden tweets blind, then measure inter-annotator agreement and settle disagreements into codebook v2. Today every "correct" label is one person's reading.
2. **A trust signal that actually tracks mistakes.** The LLM's self-reported confidence was useless (≥ 0.92 on 48 of 50 dev tweets). Next: sample the analyser 3–5 times and use self-consistency, or fit a small calibrated model on dev over signals we already have (classifier agreement, keyword-rule hits, retrieval similarity), and check its reliability curve before trusting any threshold.
3. **A shadow-mode pilot.** Run the agent silently on live traffic for a week, log the reply it *would* have sent next to what the human sent, and promote an intent to auto-send only once it has enough examples with a low failure rate. The per-intent tiers here rest on a few dozen dev tweets each.
4. **Better grounding.** Embedding retrieval with template de-duplication and intent-filtered examples. Weight past replies by whether they worked: a customer who never had to tweet again is a better example than one who replied "still broken".
5. **A sturdier judge.** Grade a larger human-rated set, test position, length and batch-size effects at scale, and compare a second judge family. Free-tier limits forced two judge-model changes in this project; a small paid budget would remove that source of risk.
6. **Robustness and drift.** Perturbation tests (typos, sarcasm, emoji-only, code-switching), and weekly monitoring of the intent mix and escalation rate, since catalogue questions spike around album releases.

## Reproducibility and citations
See `README.md`. Every LLM response is cached in `cache/`, so `OFFLINE=1 python src/run_eval.py --split test` reproduces the tables without API keys, and CI checks this on every push.
