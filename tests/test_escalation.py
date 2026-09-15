import pytest

from agent import CLAIMED_ACTION_RE, COMMITMENT_RE, ESCALATION_RULES, SupportAgent, format_thread


def test_thread_formatting_ignores_line_ending_style():
    unix = format_thread("Customer: songs skip\nSpotifyCares: which device?", "iPhone 8")
    assert format_thread("Customer: songs skip\r\nSpotifyCares: which device?", "iPhone 8") == unix
    assert format_thread("", "hi") == "Customer's latest message:\nhi"


def rule_hits(text):
    return {rid for rid, pattern in ESCALATION_RULES if pattern.search(text)}


@pytest.mark.parametrize("text, expected", [
    ("someone hacked my account and changed my email", "account_security"),
    ("I was charged twice this month", "payment_dispute"),
    ("I want a refund now", "payment_dispute"),
    ("delete all my data under GDPR", "privacy_legal"),
    ("my email is __email__ please fix", "privacy_legal"),
])
def test_rules_fire(text, expected):
    assert expected in rule_hits(text)


def test_rules_stay_quiet_on_routine_messages():
    assert rule_hits("songs keep skipping on my android phone") == set()


def make_agent(threshold=0.6, escalate_intents=("account_security",)):
    agent = SupportAgent.__new__(SupportAgent)  # skip retriever and LLM setup
    agent.threshold = threshold
    agent.escalate_intents = set(escalate_intents)
    return agent


def test_confident_routine_message_is_auto_handled():
    analysis = {"intent": "playback_technical", "confidence": 0.9, "escalate": False, "escalation_reason": "none"}
    assert make_agent().decide("songs keep skipping", analysis, "Hey! Try logging out and back in") == []


def test_each_escalation_layer_adds_its_reason():
    analysis = {"intent": "account_security", "confidence": 0.3, "escalate": True, "escalation_reason": "high_frustration"}
    reasons = make_agent().decide("I was hacked", analysis, "We'll refund you, we've just sent you a DM",
                                  second_opinion="account_access")
    assert reasons == ["rule:account_security", "policy_intent:account_security", "low_confidence",
                       "second_opinion:account_access", "llm:high_frustration", "draft_commitment_or_empty",
                       "draft_claims_action"]


def test_second_opinion_agreement_adds_nothing():
    analysis = {"intent": "playback_technical", "confidence": 0.95, "escalate": False, "escalation_reason": "none"}
    assert make_agent().decide("songs skip", analysis, "Try reinstalling", second_opinion="playback_technical") == []


def test_claimed_action_detector():
    assert CLAIMED_ACTION_RE.search("Hey! We've just sent a DM your way")
    assert not CLAIMED_ACTION_RE.search("Can you send us a DM with your email?")


def test_commitment_detector():
    assert COMMITMENT_RE.search("We'll refund you right away")
    assert not COMMITMENT_RE.search("Can you DM us your account email?")
