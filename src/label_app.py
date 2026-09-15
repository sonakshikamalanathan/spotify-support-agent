"""Local labelling tool for (1) the golden set and (2) blind human ratings of replies.

Run:  .venv/Scripts/streamlit run src/label_app.py
Labels are saved to labels/*.csv after every click.
"""
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from config import LABELS_DIR
from judge import RUBRIC

CODEBOOK = LABELS_DIR / "codebook.json"
CANDIDATES = LABELS_DIR / "golden_candidates.csv"
GOLDEN = LABELS_DIR / "golden_labels.csv"
RATING_ITEMS = LABELS_DIR / "human_rating_items.csv"
RATINGS = LABELS_DIR / "human_reply_ratings.csv"


def read_csv(path, key):
    if path.exists():
        return pd.read_csv(path, dtype={key: str}, keep_default_na=False)
    return pd.DataFrame(columns=[key])


def upsert(path, key, row):
    df = read_csv(path, key)
    df = df[df[key] != str(row[key])]
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def show_thread(context, message):
    if context:
        st.caption("Earlier in the thread")
        st.text(context)
    st.markdown("**Customer's latest message**")
    st.info(message)


def next_unlabelled(items, key, done, start):
    order = list(range(start + 1, len(items))) + list(range(0, start + 1))
    for i in order:
        if items.at[i, key] not in done:
            return i
    return start


def golden_page():
    codebook = json.loads(CODEBOOK.read_text(encoding="utf-8"))
    intents = {i["id"]: i for i in codebook["intents"]}
    reasons = ["none"] + [r["id"] for r in codebook["escalation_reasons"]]
    items = pd.read_csv(CANDIDATES, dtype={"conv_id": str}, keep_default_na=False)
    labels = read_csv(GOLDEN, "conv_id")
    done = set(labels["conv_id"])

    st.progress(len(done) / len(items), text=f"{len(done)} / {len(items)} labelled")
    if "g_idx" not in st.session_state:
        st.session_state.g_idx = next_unlabelled(items, "conv_id", done, -1)
    st.sidebar.number_input("Item #", 0, len(items) - 1, key="g_idx")
    with st.sidebar.expander("Codebook", expanded=False):
        for i in codebook["intents"]:
            st.markdown(f"**{i['id']}** - {i['definition']}")
        st.markdown("---")
        for r in codebook["escalation_reasons"]:
            st.markdown(f"**{r['id']}** - {r['definition']}")

    item = items.iloc[st.session_state.g_idx]
    cid = item["conv_id"]
    existing = labels[labels["conv_id"] == cid]
    prev = existing.iloc[0] if len(existing) else None
    st.caption(f"conv_id {cid} · split {item['golden_split']} · {'already labelled' if prev is not None else 'new'}")
    show_thread(item["context"], item["customer_text"])

    intent_ids = list(intents)
    st.radio("Intent", intent_ids, key=f"intent_{cid}", horizontal=True,
             index=intent_ids.index(prev["intent"]) if prev is not None else None,
             format_func=lambda x: intents[x]["name"])
    st.radio("Should a human handle this?", ["no", "yes"], key=f"esc_{cid}", horizontal=True,
             index=["no", "yes"].index(prev["should_escalate"]) if prev is not None else 0)
    st.selectbox("Escalation reason", reasons, key=f"reason_{cid}",
                 index=reasons.index(prev["escalation_reason"]) if prev is not None else 0)
    st.checkbox("I'm unsure about this label", key=f"unsure_{cid}",
                value=(prev is not None and str(prev["unsure"]) == "True"))
    st.text_input("Notes (optional)", key=f"notes_{cid}", value=prev["notes"] if prev is not None else "")

    def save():
        s = st.session_state
        if s[f"intent_{cid}"] is None:
            st.session_state.flash = "Pick an intent first."
            return
        upsert(GOLDEN, "conv_id", {
            "conv_id": cid, "intent": s[f"intent_{cid}"], "should_escalate": s[f"esc_{cid}"],
            "escalation_reason": s[f"reason_{cid}"] if s[f"esc_{cid}"] == "yes" else "none",
            "unsure": s[f"unsure_{cid}"], "notes": s[f"notes_{cid}"],
            "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        s.g_idx = next_unlabelled(items, "conv_id", done | {cid}, s.g_idx)

    st.button("Save & next", type="primary", on_click=save)
    if st.session_state.pop("flash", None):
        st.warning("Pick an intent first.")


def rating_page():
    items = pd.read_csv(RATING_ITEMS, dtype={"item_id": str}, keep_default_na=False)
    ratings = read_csv(RATINGS, "item_id")
    done = set(ratings["item_id"])
    st.progress(len(done) / len(items), text=f"{len(done)} / {len(items)} rated")
    if "r_idx" not in st.session_state:
        st.session_state.r_idx = next_unlabelled(items, "item_id", done, -1)
    st.sidebar.number_input("Item #", 0, len(items) - 1, key="r_idx")
    with st.sidebar.expander("Rubric", expanded=True):
        st.text(RUBRIC)

    item = items.iloc[st.session_state.r_idx]
    iid = item["item_id"]
    existing = ratings[ratings["item_id"] == iid]
    prev = existing.iloc[0] if len(existing) else None
    show_thread(item["context"], item["customer_text"])
    st.markdown("**Candidate reply** (system hidden)")
    st.success(item["reply"] or "(empty)")

    st.radio("Overall (1-5)", [1, 2, 3, 4, 5], key=f"overall_{iid}", horizontal=True,
             index=int(prev["overall"]) - 1 if prev is not None else None)
    st.radio("Hallucination?", ["no", "yes"], key=f"hall_{iid}", horizontal=True,
             index=["no", "yes"].index(prev["hallucination"]) if prev is not None else 0)
    st.text_input("Notes (optional)", key=f"rnotes_{iid}", value=prev["notes"] if prev is not None else "")

    def save():
        s = st.session_state
        if s[f"overall_{iid}"] is None:
            return
        upsert(RATINGS, "item_id", {
            "item_id": iid, "overall": s[f"overall_{iid}"], "hallucination": s[f"hall_{iid}"],
            "notes": s[f"rnotes_{iid}"], "rated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        s.r_idx = next_unlabelled(items, "item_id", done | {iid}, s.r_idx)

    st.button("Save & next", type="primary", on_click=save)


st.set_page_config(page_title="Spotify support labelling", layout="centered")
mode = st.sidebar.radio("Task", ["Golden set labels", "Rate replies (judge agreement)"])
if mode == "Golden set labels":
    if CANDIDATES.exists() and CODEBOOK.exists():
        golden_page()
    else:
        st.write("Golden candidates or codebook not generated yet.")
elif RATING_ITEMS.exists():
    rating_page()
else:
    st.write("Rating items not generated yet.")
