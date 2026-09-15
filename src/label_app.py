"""Local labelling tool:
  1. Golden set labels (intent + escalation)
  2. Consistency re-label: 30 of your own finished items again, blind, to measure label noise
  3. Blind reply ratings, used to measure judge-human agreement

Run:  .venv/Scripts/streamlit run src/label_app.py      (keyboard-driven alternative: src/fast_label.py)
Everything is saved to labels/*.csv after each click.
"""
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from judge import RUBRIC
from labels_io import (CANDIDATES, CODEBOOK, GOLDEN, MIN_LABELS_BEFORE_RELABEL, N_RELABEL, RATING_ITEMS, RATINGS,
                       RELABEL, read_csv, relabel_items, upsert)


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


def codebook_sidebar(codebook):
    with st.sidebar.expander("Codebook", expanded=False):
        for rule in codebook["labelling_rules"]:
            st.markdown(f"- {rule}")
        st.markdown("---")
        for i in codebook["intents"]:
            st.markdown(f"**{i['name']}** (`{i['id']}`): {i['definition']}")
        st.markdown("---")
        for r in codebook["escalation_reasons"]:
            st.markdown(f"**{r['id']}**: {r['definition']}")


def intent_label_page(items, out_path, prefix):
    codebook = json.loads(CODEBOOK.read_text(encoding="utf-8"))
    intents = {i["id"]: i for i in codebook["intents"]}
    intent_ids = list(intents)
    reasons = ["none"] + [r["id"] for r in codebook["escalation_reasons"]]
    labels = read_csv(out_path, "conv_id")
    done = set(labels["conv_id"])
    idx_key = f"{prefix}_idx"

    st.progress(min(1.0, len(done & set(items["conv_id"])) / len(items)),
                text=f"{len(done & set(items['conv_id']))} / {len(items)} labelled")
    if idx_key not in st.session_state:
        st.session_state[idx_key] = next_unlabelled(items, "conv_id", done, -1)
    st.sidebar.number_input("Item #", 0, len(items) - 1, key=idx_key)
    codebook_sidebar(codebook)

    item = items.iloc[st.session_state[idx_key]]
    cid = item["conv_id"]
    existing = labels[labels["conv_id"] == cid]
    prev = existing.iloc[0] if len(existing) else None
    st.caption(f"Item {st.session_state[idx_key] + 1} · {'already labelled (you can change it)' if prev is not None else 'new'}")
    show_thread(item["context"], item["customer_text"])

    st.radio("Intent", intent_ids, key=f"{prefix}_intent_{cid}",
             index=intent_ids.index(prev["intent"]) if prev is not None else None,
             format_func=lambda x: intents[x]["name"])
    st.radio("Should a human handle this?", ["no", "yes"], key=f"{prefix}_esc_{cid}", horizontal=True,
             index=["no", "yes"].index(prev["should_escalate"]) if prev is not None else 0)
    st.selectbox("Escalation reason (if yes)", reasons, key=f"{prefix}_reason_{cid}",
                 index=reasons.index(prev["escalation_reason"]) if prev is not None else 0)
    st.checkbox("I'm unsure about this label", key=f"{prefix}_unsure_{cid}",
                value=(prev is not None and str(prev["unsure"]) == "True"))
    st.text_input("Notes (optional)", key=f"{prefix}_notes_{cid}", value=prev["notes"] if prev is not None else "")

    def save():
        s = st.session_state
        if s[f"{prefix}_intent_{cid}"] is None:
            s[f"{prefix}_flash"] = "Pick an intent first."
            return
        escalate = s[f"{prefix}_esc_{cid}"]
        upsert(out_path, "conv_id", {
            "conv_id": cid, "intent": s[f"{prefix}_intent_{cid}"], "should_escalate": escalate,
            "escalation_reason": s[f"{prefix}_reason_{cid}"] if escalate == "yes" else "none",
            "unsure": s[f"{prefix}_unsure_{cid}"], "notes": s[f"{prefix}_notes_{cid}"],
            "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        s[idx_key] = next_unlabelled(items, "conv_id", done | {cid}, s[idx_key])

    st.button("Save & next", type="primary", on_click=save)
    flash = st.session_state.pop(f"{prefix}_flash", None)
    if flash:
        st.warning(flash)


def golden_page():
    items = pd.read_csv(CANDIDATES, dtype={"conv_id": str}, keep_default_na=False)
    intent_label_page(items, GOLDEN, "g")


def relabel_page():
    items = relabel_items()
    if items is None:
        st.info(f"Finish the golden set first ({len(read_csv(GOLDEN, 'conv_id'))} labelled so far; need "
                f"{MIN_LABELS_BEFORE_RELABEL}). This check re-samples {N_RELABEL} of your finished labels, so do it last.")
        return
    st.warning("Blind re-label: your earlier answers are hidden. Label each message from scratch.")
    intent_label_page(items, RELABEL, "rl")


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
    st.markdown("**Candidate reply** (which system wrote it is hidden)")
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
mode = st.sidebar.radio("Task", ["1. Golden set labels", "2. Consistency re-label", "3. Rate replies"])
if mode.startswith("1"):
    golden_page() if CANDIDATES.exists() else st.write("Golden candidates not generated yet.")
elif mode.startswith("2"):
    relabel_page()
elif RATING_ITEMS.exists():
    rating_page()
else:
    st.write("Rating items are generated after the agent and judge run. I'll tell you when they're ready.")
