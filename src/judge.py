"""LLM-as-judge for reply quality.

The rubric text is shown verbatim to the human rater in the labelling app, so judge and human
score against identical definitions. To fit the free-tier daily request quota, unrelated cases
are graded in small shuffled batches; judge_validation.py measures whether batching changes scores.
"""
from config import BRAND, JUDGE, JUDGE_MIN_INTERVAL_S
from llm import LLM

BATCH_SIZE = 12

RUBRIC = f"""OVERALL (1-5) - would a @{BRAND} support lead send this reply?
  5 = Send as-is: addresses the customer's actual issue, correct, on-brand.
  4 = Good: minor wording/tone edits at most; nothing wrong.
  3 = Partial: generic or misses part of the issue, but not harmful.
  2 = Poor: ignores or misreads the issue, confusing, or unhelpful.
  1 = Harmful: false or invented information, unkept promises, asks for sensitive data publicly, or rude.

HALLUCINATION (yes/no) - does the reply state a policy, feature, link, timeline, price or promise
that is not supported by the reference replies or by how Spotify support normally works?

A reply is ACCEPTABLE if OVERALL >= 4 and HALLUCINATION = no."""

JUDGE_SYSTEM = f"""You are a strict quality reviewer for the @{BRAND} Twitter support team.
You grade candidate replies against a rubric. Grade every case independently: the cases are
unrelated, so never compare them with each other. Do not reward length. Respond with JSON only."""

CASE_TEMPLATE = """=== CASE {number} ===
CONVERSATION
{thread}

REFERENCE - how @{brand} handled this or similar messages (acceptable approaches, not the only ones):
{references}

CANDIDATE REPLY
{reply}"""

JUDGE_TEMPLATE = """RUBRIC
{rubric}

{cases}

Return JSON: {{"results": [{{"case": <number>, "rationale": "<1-2 sentences>", "addresses_issue": <1-5>, "tone": <1-5>, "hallucination": <true|false>, "overall": <1-5>}}]}}
with exactly one entry per case ({n} in total), in case order."""


class ReplyJudge:
    def __init__(self, batch_size=BATCH_SIZE, cache_name="judge"):
        self.batch_size = batch_size
        self.llm = LLM(JUDGE["provider"], JUDGE["model"], cache_name=cache_name, min_interval=JUDGE_MIN_INTERVAL_S)

    def score(self, thread, references, reply):
        return self._score_chunk([{"thread": thread, "references": references, "reply": reply}])[0]

    def score_many(self, cases):
        """cases: dicts with thread, references (list of str) and reply. Returns scores in order."""
        results = []
        for start in range(0, len(cases), self.batch_size):
            chunk = cases[start:start + self.batch_size]
            try:
                results.extend(self._score_chunk(chunk))
            except (ValueError, KeyError, TypeError):
                if len(chunk) == 1:
                    raise
                for case in chunk:  # malformed batch output: grade each case on its own
                    results.extend(self._score_chunk([case]))
        return results

    def _score_chunk(self, chunk):
        cases = "\n\n".join(
            CASE_TEMPLATE.format(number=n, thread=c["thread"], brand=BRAND, reply=c["reply"] or "(empty)",
                                 references="\n".join(f"- {r}" for r in c["references"]))
            for n, c in enumerate(chunk, 1)
        )
        prompt = JUDGE_TEMPLATE.format(rubric=RUBRIC, cases=cases, n=len(chunk))
        parsed = self.llm.complete_json(JUDGE_SYSTEM, prompt, max_tokens=220 * len(chunk) + 100)
        entries = parsed["results"] if isinstance(parsed, dict) and "results" in parsed else [parsed]
        if len(entries) != len(chunk):
            raise ValueError(f"judge returned {len(entries)} results for {len(chunk)} cases")
        return [self._normalise(entry) for entry in entries]

    @staticmethod
    def _normalise(entry):
        overall = int(entry["overall"])
        hallucination = str(entry.get("hallucination", False)).lower() == "true"
        return {
            "judge_overall": overall,
            "judge_hallucination": hallucination,
            "judge_acceptable": overall >= 4 and not hallucination,
            "judge_addresses_issue": entry.get("addresses_issue"),
            "judge_tone": entry.get("tone"),
            "judge_rationale": entry.get("rationale", ""),
        }
