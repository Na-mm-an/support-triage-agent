"""
Phase 6b: Streamlit Demo UI
-------------------------------
Chat-style frontend for the Support Triage Agent. Calls the FastAPI
backend (api.py) over HTTP.

Setup:
  pip install streamlit requests

Run (with api.py already running separately on port 8000):
  streamlit run streamlit_app.py

For deployment on Hugging Face Spaces: set API_URL below to your deployed
backend's public URL (or see the note at the bottom for a single-file,
no-separate-backend alternative that's simpler to deploy on Spaces).
"""

import streamlit as st
import requests
import os

API_URL = os.environ.get("SUPPORT_AGENT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Support Triage Agent", page_icon="🎧", layout="centered")

st.title("🎧 Support Triage Agent")
st.caption(
    "A multi-agent RAG system that classifies, retrieves, and answers support "
    "queries with cited sources — and escalates to a human when it isn't confident."
)

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        if turn["escalated"]:
            st.warning(f"🚩 **Escalated to human review**\n\nReason: `{turn['escalation_reason']}`")
        else:
            st.write(turn["answer"])
            if turn.get("intent"):
                st.caption(f"Intent: `{turn['intent']}` · Confidence: {turn['confidence']:.2f}")

        if turn.get("sources"):
            with st.expander(f"View {len(turn['sources'])} retrieved sources"):
                for i, src in enumerate(turn["sources"], 1):
                    st.text(f"[{i}] {src}...")

        st.caption(f"⏱ {turn['latency_ms']:.0f} ms")


# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
query = st.chat_input("Ask a support question...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query", json={"query": query}, timeout=30
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the backend API: {e}")
                st.stop()

        if result["escalated"]:
            st.warning(
                f"🚩 **Escalated to human review**\n\nReason: `{result['escalation_reason']}`"
            )
        else:
            st.write(result["answer"])
            if result.get("intent"):
                st.caption(f"Intent: `{result['intent']}` · Confidence: {result['confidence']:.2f}")

        if result.get("sources"):
            with st.expander(f"View {len(result['sources'])} retrieved sources"):
                for i, src in enumerate(result["sources"], 1):
                    st.text(f"[{i}] {src}...")

        st.caption(f"⏱ {result['latency_ms']:.0f} ms")

    st.session_state.history.append({"query": query, **result})

# ---------------------------------------------------------------------------
# Sidebar: example queries + reset
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Try an example")
    examples = [
        "I want to cancel my order",
        "my payment failed, what should I do",
        "the thing I bought never showed up",
        "can you recommend a good pizza place",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["_prefill"] = ex
            st.rerun()

    st.divider()
    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Handle example button prefill (Streamlit doesn't support programmatically
# submitting chat_input directly, so this re-injects it as a query on rerun)
if "_prefill" in st.session_state:
    prefill_query = st.session_state.pop("_prefill")
    with st.chat_message("user"):
        st.write(prefill_query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query", json={"query": prefill_query}, timeout=30
                )
                response.raise_for_status()
                result = response.json()
                st.session_state.history.append({"query": prefill_query, **result})
                st.rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the backend API: {e}")
