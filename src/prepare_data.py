"""Turn the raw 3M-tweet dump into a clean table of (customer message -> brand's first reply)
for one brand, with prior thread context and a time-based train/test split.

Output: data/processed/spotify_pairs.csv (small enough to commit).
"""
import json
import re

import pandas as pd

from config import BRAND, DATA_PROCESSED, PAIRS_CSV, TEST_FRACTION
from download import find_raw_csv

MAX_CONTEXT_TURNS = 4

URL_RE = re.compile(r"https?://\S+")
HANDLE_RE = re.compile(r"@\w+")
BRAND_HANDLE_RE = re.compile(rf"@{BRAND}\b", re.IGNORECASE)
# Agent initials at the end of a reply ("... backstage /NQ"), usually followed by Twitter's
# auto-attached "Send a private message" link, which carries no content.
SIGNATURE_RE = re.compile(r"\s*[/^~][A-Z]{2,3}\s*(<url>)?\s*$")
LEADING_HANDLES_RE = re.compile(r"^(@user\s*)+")
WS_RE = re.compile(r"\s+")


def _norm_id(value):
    """Ids are sometimes read as floats ('123.0'); normalise to plain strings."""
    if pd.isna(value):
        return None
    return str(value).split(".")[0]


def clean_customer_text(text):
    text = URL_RE.sub("<url>", text)
    text = BRAND_HANDLE_RE.sub("", text)
    text = HANDLE_RE.sub("@user", text)
    return WS_RE.sub(" ", text).strip()


def clean_brand_text(text):
    text = LEADING_HANDLES_RE.sub("", clean_customer_text(text))
    return SIGNATURE_RE.sub("", text).strip()


def is_mostly_english(text):
    """Cheap language filter: most letters are ASCII and there are a few words."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 8:
        return False
    ascii_ratio = sum(c.isascii() for c in letters) / len(letters)
    return ascii_ratio >= 0.9


def load_raw():
    path = find_raw_csv()
    if path is None:
        raise SystemExit("twcs.csv not found - run `python src/download.py` first.")
    print(f"Reading {path} ...")
    df = pd.read_csv(path, dtype=str)
    df["tweet_id"] = df["tweet_id"].map(_norm_id)
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].map(_norm_id)
    df["inbound"] = df["inbound"].str.lower().eq("true")
    df["created_at"] = pd.to_datetime(df["created_at"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    return df


def build_pairs(df):
    by_id = df.set_index("tweet_id")
    brand = df[df["author_id"] == BRAND].dropna(subset=["in_response_to_tweet_id"])

    # The brand's FIRST reply to each tweet is the target we learn from.
    first_replies = brand.sort_values("created_at").drop_duplicates("in_response_to_tweet_id")
    first_replies = first_replies[first_replies["in_response_to_tweet_id"].isin(by_id.index)]

    rows = []
    for reply in first_replies.itertuples(index=False):
        cust = by_id.loc[reply.in_response_to_tweet_id]
        if not cust["inbound"]:
            continue  # brand replying to itself (multi-tweet answers)

        # Walk up the thread to collect prior turns as context.
        context, parent_id = [], cust["in_response_to_tweet_id"]
        while parent_id is not None and parent_id in by_id.index and len(context) < MAX_CONTEXT_TURNS:
            parent = by_id.loc[parent_id]
            speaker = BRAND if parent["author_id"] == BRAND else ("Customer" if parent["inbound"] else parent["author_id"])
            context.append(f"{speaker}: {clean_brand_text(parent['text'])}")
            parent_id = parent["in_response_to_tweet_id"]
        context.reverse()

        # Did the same customer tweet again under the brand's reply? (weak "not resolved yet" signal)
        followed_up = False
        if isinstance(reply.response_tweet_id, str):
            for child_id in reply.response_tweet_id.split(","):
                child_id = _norm_id(child_id)
                if child_id in by_id.index and by_id.loc[child_id, "author_id"] == cust["author_id"]:
                    followed_up = True
                    break

        rows.append({
            "conv_id": reply.in_response_to_tweet_id,
            "reply_id": reply.tweet_id,
            "created_at": cust["created_at"],
            "customer_text_raw": cust["text"],
            "customer_text": clean_customer_text(cust["text"]),
            "context": "\n".join(context),
            "n_prior_turns": len(context),
            "brand_reply_raw": reply.text,
            "brand_reply": clean_brand_text(reply.text),
            "reply_delay_min": round((reply.created_at - cust["created_at"]).total_seconds() / 60, 1),
            "customer_followed_up": followed_up,
        })
    return pd.DataFrame(rows)


def main():
    df = load_raw()
    print(f"Raw tweets: {len(df):,}  |  {BRAND} tweets: {(df['author_id'] == BRAND).sum():,}")
    pairs = build_pairs(df)
    stats = {"pairs_built": len(pairs)}

    pairs = pairs[pairs["customer_text"].map(is_mostly_english)]
    stats["after_english_filter"] = len(pairs)
    pairs = pairs[pairs["customer_text"].str.split().str.len() >= 3]
    stats["after_min_length"] = len(pairs)
    pairs = pairs.sort_values("created_at").drop_duplicates("customer_text")
    stats["after_dedup"] = len(pairs)

    cutoff = pairs["created_at"].quantile(1 - TEST_FRACTION)
    pairs["split"] = (pairs["created_at"] >= cutoff).map({True: "test", False: "history"})
    stats["split_cutoff"] = str(cutoff)
    stats["history"] = int((pairs["split"] == "history").sum())
    stats["test"] = int((pairs["split"] == "test").sum())
    stats["date_range"] = [str(pairs["created_at"].min()), str(pairs["created_at"].max())]

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(PAIRS_CSV, index=False)
    (DATA_PROCESSED / "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Wrote {PAIRS_CSV}")


if __name__ == "__main__":
    main()
