"""LLM-as-judge for reply quality. The same rubric text is shown to the human rater in the
labelling app, so judge and human score against identical definitions."""
from config import BRAND, JUDGE
from llm import LLM

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
You grade one candidate reply at a time against a rubric. Be consistent; do not reward length.
Respond with a JSON object only."""

JUDGE_TEMPLATE = """RUBRIC
{rubric}

CONVERSATION
{thread}

REFERENCE - how @{brand} actually handled this and similar messages (acceptable approaches, not the only ones):
{references}

CANDIDATE REPLY
{reply}

Return JSON: {{"rationale": "<1-2 sentences>", "addresses_issue": <1-5>, "tone": <1-5>,
"hallucination": <true|false>, "overall": <1-5>}}"""


class ReplyJudge:
    def __init__(self):
        self.llm = LLM(JUDGE["provider"], JUDGE["model"], cache_name="judge")

    def score(self, thread, references, reply):
        prompt = JUDGE_TEMPLATE.format(rubric=RUBRIC, thread=thread, brand=BRAND,
                                       references="\n".join(f"- {r}" for r in references), reply=reply or "(empty)")
        result = self.llm.complete_json(JUDGE_SYSTEM, prompt, max_tokens=300)
        overall = int(result.get("overall", 1))
        hallucination = bool(result.get("hallucination", False))
        return {
            "judge_overall": overall,
            "judge_hallucination": hallucination,
            "judge_acceptable": overall >= 4 and not hallucination,
            "judge_addresses_issue": result.get("addresses_issue"),
            "judge_tone": result.get("tone"),
            "judge_rationale": result.get("rationale", ""),
        }
