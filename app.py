"""
Phase 6/7 (Option A): Standalone Streamlit App
----------------------------------------------------
Runs the full multi-agent pipeline IN-PROCESS -- no separate FastAPI
backend needed. This is the simplest architecture to deploy as a single
Hugging Face Space.

Setup (local):
  export GROQ_API_KEY="your-key-here"
  pip install streamlit langgraph groq chromadb rank_bm25
              sentence-transformers scikit-learn joblib pandas numpy

Run locally:
  streamlit run app.py

Deploy on Hugging Face Spaces:
  1. Create a new Space (SDK: Streamlit)
  2. Push this file as app.py, plus requirements.txt, plus the entire
     artifacts/ folder (classifier, chroma_db, bm25 index, etc. -- these
     are NOT rebuilt on the server, they must be committed/uploaded)
  3. In Space Settings -> Repository secrets, add GROQ_API_KEY
"""

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import time
import pickle
import numpy as np
import pandas as pd
import joblib
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ARTIFACTS_DIR = "artifacts"
CHROMA_DIR = os.path.join(ARTIFACTS_DIR, "chroma_db")
COLLECTION_NAME = "support_corpus"

TOP_K = 5
RRF_K = 60
MIN_RRF_SCORE = 0.010
MIN_DENSE_SIMILARITY = 0.35
GROQ_MODEL = "llama-3.3-70b-versatile"


class SupportState(TypedDict):
    query: str
    intent: Optional[str]
    classifier_confidence: Optional[float]
    retrieved_docs: Optional[List[dict]]
    answer: Optional[str]
    escalated: bool
    escalation_reason: Optional[str]


# ---------------------------------------------------------------------------
# Load everything ONCE per server process (not per user session) using
# st.cache_resource -- this is the Streamlit-native way to avoid reloading
# the classifier/embedding model/indexes on every single query or rerun.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models and indexes (first load only)...")
def load_all_artifacts():
    clf = joblib.load(os.path.join(ARTIFACTS_DIR, "classifier.joblib"))
    label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
    abstention_threshold = joblib.load(os.path.join(ARTIFACTS_DIR, "abstention_threshold.joblib"))

    corpus_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "retrieval_corpus.csv"))
    with open(os.path.join(ARTIFACTS_DIR, "bm25_index.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    groq_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    if not groq_key:
        st.error("GROQ_API_KEY not found in environment or Streamlit secrets.")
        st.stop()
    llm_client = Groq(api_key=groq_key)

    return {
        "clf": clf,
        "label_encoder": label_encoder,
        "abstention_threshold": abstention_threshold,
        "corpus_df": corpus_df,
        "bm25": bm25,
        "collection": collection,
        "embed_model": embed_model,
        "llm_client": llm_client,
    }


# ---------------------------------------------------------------------------
# Retrieval helpers (same logic as Phase 3/4/5)
# ---------------------------------------------------------------------------
def simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def dense_search_with_similarity(collection, embed_model, query, top_k=TOP_K):
    query_emb = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=top_k)
    ids = results["ids"][0]
    docs = results["documents"][0]
    distances = results["distances"][0]
    similarities = [1 / (1 + d) for d in distances]
    return list(zip(ids, docs, similarities))


def bm25_search(bm25, corpus_df, query, top_k=TOP_K):
    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        (corpus_df.iloc[i]["doc_id"], corpus_df.iloc[i]["doc_text"], scores[i])
        for i in top_indices
    ]


def hybrid_search(collection, bm25, corpus_df, embed_model, query, top_k=TOP_K, rrf_k=RRF_K):
    dense_results = dense_search_with_similarity(collection, embed_model, query, top_k=top_k * 2)
    bm25_results = bm25_search(bm25, corpus_df, query, top_k=top_k * 2)

    rrf_scores, doc_text_lookup, dense_sim_lookup = {}, {}, {}

    for rank, (doc_id, text, sim) in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
        doc_text_lookup[doc_id] = text
        dense_sim_lookup[doc_id] = sim

    for rank, (doc_id, text, _) in enumerate(bm25_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
        doc_text_lookup[doc_id] = text

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "doc_id": doc_id,
            "text": doc_text_lookup[doc_id],
            "rrf_score": score,
            "dense_similarity": dense_sim_lookup.get(doc_id, 0.0),
        }
        for doc_id, score in ranked
    ]


GROUNDED_PROMPT_TEMPLATE = """You are a customer support assistant. Answer the customer's question using ONLY the information in the retrieved context below. Do not use any outside knowledge.

Rules:
- Every claim in your answer must be traceable to one of the numbered context sources.
- Cite sources inline like [1], [2] after each claim.
- If the retrieved context does NOT contain enough information to answer the question, respond with EXACTLY: "INSUFFICIENT_CONTEXT" and nothing else.
- Be concise and direct.

Retrieved context:
{context_block}

Customer question: {query}

Answer:"""


def build_context_block(results):
    return "\n\n".join(f"[{i}] {r['text']}" for i, r in enumerate(results, 1))


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------
def make_classifier_node(artifacts):
    def classifier_agent(state: SupportState) -> SupportState:
        query = state["query"]
        emb = artifacts["embed_model"].encode([query])
        probs = artifacts["clf"].predict_proba(emb)[0]

        # Top-2 cumulative confidence -- fixes cases where confidence splits
        # across two valid, related intents (e.g. track_refund vs get_refund)
        top2_conf = np.sort(probs)[::-1][:2].sum()
        pred_idx = probs.argmax()

        if top2_conf < artifacts["abstention_threshold"]:
            state["intent"] = "out_of_scope"
            state["classifier_confidence"] = float(top2_conf)
            state["escalated"] = True
            state["escalation_reason"] = "classifier_out_of_scope"
        else:
            state["intent"] = artifacts["label_encoder"].inverse_transform([pred_idx])[0]
            state["classifier_confidence"] = float(top2_conf)
            state["escalated"] = False
            state["escalation_reason"] = None

        return state
    return classifier_agent


def make_retriever_node(artifacts):
    def retriever_agent(state: SupportState) -> SupportState:
        results = hybrid_search(
            artifacts["collection"], artifacts["bm25"], artifacts["corpus_df"],
            artifacts["embed_model"], state["query"],
        )
        state["retrieved_docs"] = results

        if not results:
            state["escalated"] = True
            state["escalation_reason"] = "no_results_retrieved"
            return state

        top_rrf = results[0]["rrf_score"]
        top_sim = results[0]["dense_similarity"]

        if top_rrf < MIN_RRF_SCORE or top_sim < MIN_DENSE_SIMILARITY:
            state["escalated"] = True
            state["escalation_reason"] = (
                f"low_retrieval_confidence (rrf={top_rrf:.4f}, sim={top_sim:.4f})"
            )

        return state
    return retriever_agent


def make_responder_node(artifacts):
    def responder_agent(state: SupportState) -> SupportState:
        results = state["retrieved_docs"]
        context_block = build_context_block(results)
        prompt = GROUNDED_PROMPT_TEMPLATE.format(context_block=context_block, query=state["query"])

        response = artifacts["llm_client"].chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_answer = response.choices[0].message.content.strip()

        if raw_answer == "INSUFFICIENT_CONTEXT":
            state["escalated"] = True
            state["escalation_reason"] = "llm_flagged_insufficient_context"
            state["answer"] = None
        else:
            state["answer"] = raw_answer

        return state
    return responder_agent


def escalate_node(state: SupportState) -> SupportState:
    state["answer"] = None
    return state


def route_after_classifier(state):
    return "escalate" if state["escalated"] else "retrieve"


def route_after_retriever(state):
    return "escalate" if state["escalated"] else "respond"


def route_after_responder(state):
    return "escalate" if state["escalated"] else "end"


@st.cache_resource(show_spinner=False)
def build_graph(_artifacts):
    # Note: leading underscore on _artifacts tells st.cache_resource not to
    # try hashing the dict (it contains unhashable objects like models)
    graph = StateGraph(SupportState)
    graph.add_node("classify", make_classifier_node(_artifacts))
    graph.add_node("retrieve", make_retriever_node(_artifacts))
    graph.add_node("respond", make_responder_node(_artifacts))
    graph.add_node("escalate", escalate_node)
    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route_after_classifier, {"retrieve": "retrieve", "escalate": "escalate"})
    graph.add_conditional_edges("retrieve", route_after_retriever, {"respond": "respond", "escalate": "escalate"})
    graph.add_conditional_edges("respond", route_after_responder, {"end": END, "escalate": "escalate"})
    graph.add_edge("escalate", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Support Triage Agent", page_icon="🎧", layout="centered")
st.title("🎧 Support Triage Agent")
st.caption(
    "A multi-agent RAG system that classifies, retrieves, and answers support "
    "queries with cited sources — and escalates to a human when it isn't confident."
)

artifacts = load_all_artifacts()
app = build_graph(artifacts)

if "history" not in st.session_state:
    st.session_state.history = []


def run_query(query: str):
    start = time.perf_counter()
    initial_state: SupportState = {
        "query": query, "intent": None, "classifier_confidence": None,
        "retrieved_docs": None, "answer": None, "escalated": False, "escalation_reason": None,
    }
    final_state = app.invoke(initial_state)
    latency_ms = (time.perf_counter() - start) * 1000
    sources = [d["text"][:120] for d in final_state["retrieved_docs"]] if final_state.get("retrieved_docs") else []
    return {**final_state, "sources": sources, "latency_ms": latency_ms}


def render_turn(query, result):
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        if result["escalated"]:
            st.warning(f"🚩 **Escalated to human review**\n\nReason: `{result['escalation_reason']}`")
        else:
            st.write(result["answer"])
            if result.get("intent"):
                st.caption(f"Intent: `{result['intent']}` · Confidence: {result['classifier_confidence']:.2f}")
        if result.get("sources"):
            with st.expander(f"View {len(result['sources'])} retrieved sources"):
                for i, src in enumerate(result["sources"], 1):
                    st.text(f"[{i}] {src}...")
        st.caption(f"⏱ {result['latency_ms']:.0f} ms")


for turn in st.session_state.history:
    render_turn(turn["query"], turn)

query = st.chat_input("Ask a support question...")
if query:
    with st.spinner("Thinking..."):
        result = run_query(query)
    st.session_state.history.append({"query": query, **result})
    st.rerun()

with st.sidebar:
    st.subheader("Try an example")
    examples = [
        "I want to cancel my order",
        "my payment failed, what should I do",
        "where's my refund",
        "can you recommend a good pizza place",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            with st.spinner("Thinking..."):
                result = run_query(ex)
            st.session_state.history.append({"query": ex, **result})
            st.rerun()

    st.divider()
    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()
