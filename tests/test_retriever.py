import json

import pandas as pd

from agent import Retriever

HISTORY = pd.DataFrame({
    "conv_id": ["a", "b", "c", "d"],
    "customer_text": ["songs skip on android", "songs skip on android", "billing charged twice", "songs skip on iphone"],
    "brand_reply": ["reply a", "reply b", "reply c", "reply d"],
})


def test_uncached_ties_break_by_history_order(tmp_path):
    retriever = Retriever(HISTORY, cache_path=tmp_path / "missing.json")
    assert [r["conv_id"] for r in retriever.search("songs skip on android", k=2)] == ["a", "b"]  # a and b tie exactly
    assert retriever.cache_misses == 1


def test_cached_rankings_are_replayed(tmp_path):
    cache = tmp_path / "retrieval_cache.json"
    cache.write_text(json.dumps({"2": {"songs skip on android": ["b", "a"]}}))
    retriever = Retriever(HISTORY, cache_path=cache)
    assert [r["conv_id"] for r in retriever.search("songs skip on android", k=2)] == ["b", "a"]
    assert retriever.cache_misses == 0
