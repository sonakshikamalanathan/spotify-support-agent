# Decision log

Non-obvious decisions, in the order they were made. (Draft: numbers are filled in after evaluation.)

1. **Brand: @SpotifyCares.** About 43k brand tweets, a well-bounded product, and replies that often contain real troubleshooting steps rather than only "DM us", so drafting a grounded reply is meaningful. Airlines had richer escalation cases but mostly "DM your confirmation number" replies.

2. **Unit of work = one inbound customer tweet, target = the brand's first reply to it**, with up to 4 earlier thread turns as context. Later brand replies in a thread depend on private DM information the agent can never see.

3. **Time-based split, not random.** The newest 20% of conversations (after 22 Nov 2017) are held out. The brand reuses templates heavily, so a random split would put near-identical replies in both history and test and inflate every retrieval-based number.

4. **Cleaning decisions.** Agent initials (`/NQ`) and Twitter's auto-attached "send a DM" link are removed from brand replies, because the drafter would otherwise learn to sign as a fake agent. Content links are kept as `<url>`. Language filtering is a cheap ASCII-ratio heuristic, and exact duplicate customer texts are dropped.

5. **Intent taxonomy: clusters as a starting point, a human codebook as the result.** TF-IDF + k-means (24 clusters) exposed the themes, but 26% of messages fell into one mixed cluster. The final 9 intents are defined by *how the brand has to respond*: Family/Student plans merge into billing, country-launch requests into feature feedback, and hacked accounts get their own intent because they change the handling.

6. **"Escalate" means a human must review before anything is posted publicly.** A reply asking the customer to DM counts as auto-handleable for routine issues, because that is exactly what the brand does and a human picks it up in DMs anyway.

7. **Layered, explainable escalation**: keyword rules → always-escalate intents → low classifier confidence → a second-opinion classifier → the LLM's risk judgement → checks that the draft doesn't promise refunds, credits or timelines, or claim an action the agent can't take. Each escalation records every layer that fired, so failures can be attributed to a layer. **Two layers were added after the dev run, and chosen on dev only:** the LLM reported confidence ≥ 0.92 on 48 of 50 dev tweets, so "low confidence" could never catch its mistakes. Where an independent TF-IDF classifier disagreed with the LLM's intent, the failure rate doubled (52% vs 24%), so disagreement now escalates. Priced with the cost model on dev, this cut the cost from 154 to 94 per 100 tickets and raised escalation recall from 77% to 92%, at the price of escalating 54% of tweets instead of 22%.

8. **TF-IDF retrieval instead of embeddings.** No extra API, GPU or rate limit; deterministic; fast; and the evaluation reproduces offline. Semantic misses are a known weakness, noted in failure analysis.

9. **Two different Groq models inside the agent.** `gpt-oss-120b` (a reasoning model) classifies and `qwen3.8-27b` drafts. Each model has its own free-tier token budget, which doubles throughput.

10. **Golden set sampled in three strata from the test period**: random (90, the natural distribution), intent-balanced (80, so rare intents are measurable) and hard cases (40: escalation keywords and multi-turn threads). Metrics are also reported on the random stratum alone, because the stratified mix is harder than real traffic. The brand's actual reply and all model predictions are hidden from the labeller, and every label was entered by hand (a keyboard-driven page made this about 15 seconds per tweet). After the first 11 (dev) labels, an AI assistant checked them against the codebook's rules once, as a calibration round; labels that broke a rule were corrected. The labelling page also enforces the codebook's one hard rule (a hacked-account tweet always needs a human), the same way it requires a reason whenever "needs a human" is chosen. The first 50 items are a dev split for tuning; the rest are the reported test set.

11. **Simple baseline trained on weak labels.** Each k-means cluster that clearly belongs to one intent was mapped by hand, giving free, noisy training labels for TF-IDF + logistic regression, with no LLM involved. It cannot predict intents that no cluster captured (hacked account), which is a real limitation of that approach.

12. **Judge design.** The judge must not be the reply writer (Qwen), so it is `gpt-oss-120b`. Gemini 2.5 Flash was the first choice, but its free tier allows 20 requests/day. The judge's only reference is the brand's actual reply: giving it retrieved replies would hand the retrieval baseline its own answer. Cases are graded in shuffled batches of 6 to fit request quotas.

13. **The judge is validated three ways, not one**: agreement with blind human ratings, a planted-defect stress test (invented refunds, asking for passwords, rude or off-topic replies), and batched-vs-single grading agreement.

14. **Every LLM response is cached and committed.** `OFFLINE=1` re-runs the full evaluation without API keys in about a minute, and CI checks on every push that the headline numbers reproduce. Retrieval rankings are cached too. Many historical tweets tie on similarity, and numpy's default sort breaks ties differently on different CPUs, so on CI's Linux runner the drafter saw different examples and missed its cache. The committed rankings are replayed, and any new query breaks ties by history order.

15. **Trust is granted per intent, not globally.** A cost model (human writes = 1, human approves draft = 0.3, bad auto-reply = 3, missed escalation = 10) prices rollout policies. Intents qualify for auto-send on dev and are priced on test. No intent qualified (dev has 2 to 12 tweets per intent), so the recommendation is a drafting assistant: on test it costs 67 per 100 tweets versus 100 for humans alone and 157 for auto-sending, with zero unreviewed replies.

## Deliberately not built
- Handling the DM conversation itself (no data, and it needs account access)
- Non-English support (the brand itself redirects these)
- Fine-tuning, or any model trained on the golden set
- Real Twitter or helpdesk integration
- Calibrated confidence: the LLM's self-reported confidence is used as-is and its weakness is measured, not fixed
