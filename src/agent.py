"""The support agent: analyse (intent + risk) -> retrieve similar past cases -> draft reply
-> decide whether to auto-handle or escalate, with reasons.

Escalation is layered so that each decision is explainable:
  1. deterministic rules on the message (safety, security, payments, legal)
  2. intents that policy always routes to a human
  3. low classifier confidence
  4. a second-opinion classifier (TF-IDF on weak labels) disagreeing with the LLM's intent
  5. the LLM's own risk judgement
  6. the drafted reply making a commitment we cannot verify (refund, credit, timeline),
     or claiming an action the agent cannot take ("we've sent you a DM")
"""
import json
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from config import ANALYSER, BRAND, DRAFTER, LABELS_DIR, RETRIEVAL_CACHE
from llm import LLM
from weak_classifier import train_weak_label_classifier

CODEBOOK_PATH = LABELS_DIR / "codebook.json"
CONFIDENCE_THRESHOLD = 0.6
TOP_K = 5

ESCALATION_RULES = [
    ("account_security", re.compile(r"\bhack(ed|er|ing)?\b|compromised|someone (else )?(is )?(using|logged into|in) my account|stolen account|unauthori[sz]ed (login|access)|changed my (email|password)", re.I)),
    ("payment_dispute", re.compile(r"\brefund|charged (me )?(twice|double|again|after|without)|double charged|unauthori[sz]ed (charge|payment|transaction)|fraud|chargeback|money back|stole (my )?money", re.I)),
    # "__email__" is the dataset's mask for an email address the customer posted publicly.
    ("privacy_legal", re.compile(r"\blawyer|\bsu(e|ing)\b|legal action|\bgdpr\b|data protection|consumer (rights|protection)|trading standards|ombudsman|\bbbb\b|__email__|card number", re.I)),
    ("safety_wellbeing", re.compile(r"suicid|kill (myself|me)|self[- ]harm|want to die|threat(en)?|harass", re.I)),
]
# The agent cannot send DMs or take account actions, so a draft claiming it did is false.
CLAIMED_ACTION_RE = re.compile(r"\b(?:we've|we have|we just)\s+(?:just\s+)?(?:sent|replied|dm'?d|messaged)", re.I)
COMMITMENT_RE = re.compile(r"\brefund|credit (your|to your)|compensat|free (month|premium|trial)|we('ll| will) (fix|refund|credit|reimburse)|within \d+ (hours|days)", re.I)


def load_codebook():
    return json.loads(CODEBOOK_PATH.read_text(encoding="utf-8"))


def format_codebook(codebook):
    lines = ["INTENTS:"]
    for intent in codebook["intents"]:
        lines.append(f"- {intent['id']}: {intent['definition']}")
        if intent.get("includes"):
            lines.append(f"    includes: {'; '.join(intent['includes'])}")
        if intent.get("excludes"):
            lines.append(f"    excludes: {'; '.join(intent['excludes'])}")
    lines.append("\nESCALATE TO A HUMAN WHEN:")
    for reason in codebook["escalation_reasons"]:
        lines.append(f"- {reason['id']}: {reason['definition']}")
    return "\n".join(lines)


def format_thread(context, message):
    # Normalise Windows line endings so prompts, and therefore cache keys, match on every OS.
    context = context.replace("\r\n", "\n") if isinstance(context, str) else ""
    thread = f"Earlier in the thread:\n{context}\n\n" if context.strip() else ""
    return f"{thread}Customer's latest message:\n{message}"


class Retriever:
    """TF-IDF nearest neighbours over historical customer messages (history split only).

    Many historical tweets tie on similarity, and numpy's default sort orders ties differently on
    different CPUs, which changed the drafter's prompts on CI. So the rankings behind the committed
    results are replayed from RETRIEVAL_CACHE, and any other query breaks ties by history order.
    """

    def __init__(self, history, cache_path=RETRIEVAL_CACHE):
        self.history = history.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.history["customer_text"])
        self.row_of = {str(cid): row for row, cid in enumerate(self.history["conv_id"])}
        self.cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path and cache_path.exists() else {}
        self.cache_misses = 0

    def search(self, text, k=TOP_K):
        scores = linear_kernel(self.vectorizer.transform([text]), self.matrix).ravel()
        cached = self.cache.get(str(k), {}).get(text)
        if cached and all(cid in self.row_of for cid in cached):
            top = [self.row_of[cid] for cid in cached]
        else:
            self.cache_misses += 1
            top = (-scores).argsort(kind="stable")[:k]
        return [
            {
                "conv_id": str(self.history.at[i, "conv_id"]),
                "customer_text": self.history.at[i, "customer_text"],
                "brand_reply": self.history.at[i, "brand_reply"],
                "score": round(float(scores[i]), 3),
            }
            for i in top
        ]


ANALYSE_SYSTEM = f"""You triage tweets sent to @{BRAND}, Spotify's customer support account.
Classify the customer's latest message into exactly one intent from the codebook and decide
whether a human agent must handle it. Respond with a JSON object only."""

ANALYSE_TEMPLATE = """{codebook}

{thread}

Return JSON with these keys:
{{"rationale": "<one short sentence>",
  "intent": "<one intent id from the codebook>",
  "confidence": <number 0-1: how sure you are about the intent>,
  "escalate": <true|false>,
  "escalation_reason": "<escalation reason id, or \\"none\\">"}}"""

DRAFT_SYSTEM = f"""You write public Twitter replies as the @{BRAND} support team.
Ground every reply in how @{BRAND} has handled similar messages before (examples provided).
Rules:
- At most 280 characters. Friendly, concise, casual like the examples; no hashtags.
- Only suggest troubleshooting steps or links (shown as <url>) that appear in the examples.
- Never promise refunds, credits, compensation, free Premium, or fix timelines.
- If the issue needs account-specific details, ask the customer to send a DM, as the brand does.
- Never ask for passwords or full card numbers.
- If the message is not a support request (praise, jokes), reply briefly and warmly.
Respond with a JSON object only."""

DRAFT_TEMPLATE = """{thread}

Predicted intent: {intent}

How @{brand} replied to similar messages:
{examples}

Return JSON: {{"reply": "<the reply text>", "used_examples": ["E1", ...]}}"""


class SupportAgent:
    def __init__(self, history, codebook=None, confidence_threshold=CONFIDENCE_THRESHOLD):
        self.codebook = codebook or load_codebook()
        self.intent_ids = [i["id"] for i in self.codebook["intents"]]
        self.escalate_intents = set(self.codebook.get("always_escalate_intents", []))
        self.threshold = confidence_threshold
        self.retriever = Retriever(history)
        self.second_opinion = train_weak_label_classifier(history)
        self.analyser = LLM(ANALYSER["provider"], ANALYSER["model"], cache_name="analyser")
        self.drafter = LLM(DRAFTER["provider"], DRAFTER["model"], cache_name="drafter")

    def analyse(self, message, context):
        prompt = ANALYSE_TEMPLATE.format(codebook=format_codebook(self.codebook), thread=format_thread(context, message))
        result = self.analyser.complete_json(ANALYSE_SYSTEM, prompt, max_tokens=1500)
        if result.get("intent") not in self.intent_ids:
            result["intent"], result["confidence"] = "other", 0.0
        result["confidence"] = float(result.get("confidence") or 0.0)
        return result

    def draft(self, message, context, intent, examples):
        example_text = "\n".join(
            f"[E{n}] Customer: {ex['customer_text']}\n     @{BRAND}: {ex['brand_reply']}" for n, ex in enumerate(examples, 1)
        )
        prompt = DRAFT_TEMPLATE.format(thread=format_thread(context, message), intent=intent, brand=BRAND, examples=example_text)
        result = self.drafter.complete_json(DRAFT_SYSTEM, prompt, max_tokens=600)
        cited = [examples[int(e[1:]) - 1]["conv_id"] for e in result.get("used_examples", [])
                 if isinstance(e, str) and e[1:].isdigit() and 0 < int(e[1:]) <= len(examples)]
        return str(result.get("reply", "")).strip(), cited

    def decide(self, message, analysis, reply, second_opinion=None):
        reasons = [f"rule:{rid}" for rid, pattern in ESCALATION_RULES if pattern.search(message)]
        if analysis["intent"] in self.escalate_intents:
            reasons.append(f"policy_intent:{analysis['intent']}")
        if analysis["confidence"] < self.threshold:
            reasons.append("low_confidence")
        # The LLM's self-reported confidence was >= 0.92 on 48 of 50 dev tweets, so it cannot flag its
        # own mistakes; disagreement with an independent classifier doubled the dev failure rate.
        if second_opinion is not None and second_opinion != analysis["intent"]:
            reasons.append(f"second_opinion:{second_opinion}")
        if analysis.get("escalate") and analysis.get("escalation_reason", "none") != "none":
            reasons.append(f"llm:{analysis['escalation_reason']}")
        if not reply or COMMITMENT_RE.search(reply):
            reasons.append("draft_commitment_or_empty")
        if CLAIMED_ACTION_RE.search(reply):
            reasons.append("draft_claims_action")
        return reasons

    def handle(self, message, context=""):
        analysis = self.analyse(message, context)
        examples = self.retriever.search(message)
        reply, cited = self.draft(message, context, analysis["intent"], examples)
        second_opinion = self.second_opinion.predict([message])[0]
        reasons = self.decide(message, analysis, reply, second_opinion)
        return {
            "pred_intent": analysis["intent"],
            "confidence": analysis["confidence"],
            "second_opinion_intent": second_opinion,
            "llm_rationale": analysis.get("rationale", ""),
            "reply": reply,
            "cited_examples": ";".join(cited),
            "escalate": bool(reasons),
            "escalation_reasons": ";".join(reasons),
        }


def run_on(frame, history):
    """Run the agent over a DataFrame with customer_text/context columns."""
    agent = SupportAgent(history)
    rows = []
    for n, row in enumerate(frame.itertuples(index=False), 1):
        out = agent.handle(row.customer_text, row.context)
        rows.append({"conv_id": str(row.conv_id), **out})
        if n % 10 == 0:
            print(f"  agent: {n}/{len(frame)}")
    return pd.DataFrame(rows)
