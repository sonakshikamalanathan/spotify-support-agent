import pytest

import judge
import llm


def test_parse_json_variants():
    assert llm.parse_json('{"a": 1}') == {"a": 1}
    assert llm.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json('Here you go: {"a": 1} hope that helps') == {"a": 1}


def test_cache_hit_and_offline_miss(tmp_path, monkeypatch):
    monkeypatch.delenv("OFFLINE", raising=False)
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    client = llm.LLM("groq", "test-model", cache_name="t")
    calls = []
    monkeypatch.setattr(client, "_call_with_retry", lambda *args: calls.append(args) or '{"ok": true}')
    assert client.complete_json("sys", "user") == {"ok": True}
    assert client.complete_json("sys", "user") == {"ok": True}
    assert len(calls) == 1  # second call served from the cache

    monkeypatch.setenv("OFFLINE", "1")
    offline = llm.LLM("groq", "test-model", cache_name="t")
    assert offline.complete_json("sys", "user") == {"ok": True}  # persisted to disk
    with pytest.raises(llm.CacheMiss):
        offline.complete("sys", "a prompt never seen before")


class StubLLM:
    def __init__(self, responses):
        self.responses, self.prompts = list(responses), []

    def complete_json(self, system, user, **kwargs):
        self.prompts.append(user)
        return self.responses.pop(0)


def make_judge(responses, batch_size=2):
    j = judge.ReplyJudge.__new__(judge.ReplyJudge)
    j.llm, j.batch_size = StubLLM(responses), batch_size
    return j


CASE = {"thread": "Customer: songs skip", "references": ["Try reinstalling"], "reply": "Try reinstalling the app"}


def test_judge_batches_and_applies_acceptability_rule():
    j = make_judge([
        {"results": [{"overall": 5, "hallucination": False}, {"overall": 5, "hallucination": True}]},
        {"results": [{"overall": 3, "hallucination": False}]},
    ])
    scores = j.score_many([CASE, CASE, CASE])
    assert [s["judge_acceptable"] for s in scores] == [True, False, False]
    assert len(j.llm.prompts) == 2


def test_judge_falls_back_to_single_cases_when_batch_is_malformed():
    j = make_judge([{"results": [{"overall": 4}]}, {"overall": 4, "hallucination": False}, {"overall": 2, "hallucination": False}])
    assert [s["judge_overall"] for s in j.score_many([CASE, CASE])] == [4, 2]
