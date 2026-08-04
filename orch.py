"""
Phase 5: Multi-Agent Orchestration (LangGraph)
--------------------------------------------------
Wires the classifier agent (Phase 2), retriever (Phase 3), and responder
agent (Phase 4) into a single LangGraph state graph with conditional
routing: any agent can trigger escalation, and once triggered, the graph
short-circuits straight to the escalation node instead of continuing.

Flow:
  classifier_agent -> (in-scope?) -> retriever_agent -> responder_agent -> END
        |                                                      |
        v (out-of-scope)                                       v (low confidence / insufficient context)
  escalate_node <------------------------------------------------

Setup:
  export GROQ_API_KEY="your-key-here"
  pip install langgraph groq chromadb rank_bm25 sentence-transformers
              scikit-learn joblib pandas numpy

Run:
  python phase5_orchestration.py
"""

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import pickle
import numpy as np
import pandas as pd
import joblib
import chromadb
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


# ---------------------------------------------------------------------------
# Shared state passed between all agent nodes
# ---------------------------------------------------------------------------
class SupportState(TypedDict):
    query: str
    intent: Optional[str]
    classifier_confidence: Optional[float]
    retrieved_docs: Optional[List[dict]]
    answer: Optional[str]
    escalated: bool
    escalation_reason: Optional[str]


# ---------------------------------------------------------------------------
# Load all artifacts once, shared across every agent call
# ---------------------------------------------------------------------------
def load_all_artifacts():
    print("Loading classifier artifacts (Phase 2)...")
    clf = joblib.load(os.path.join(ARTIFACTS_DIR, "classifier.joblib"))
    label_names = joblib.load(os.path.join(ARTIFACTS_DIR, "label_names.joblib"))
    abstention_threshold = joblib.load(os.path.join(ARTIFACTS_DIR, "abstention_threshold.joblib"))

    print("Loading retrieval artifacts (Phase 3)...")
    corpus_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "retrieval_corpus.csv"))
    with open(os.path.join(ARTIFACTS_DIR, "bm25_index.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    print("Loading embedding model (shared by classifier + retriever)...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Initializing Groq client...")
    if "GROQ_API_KEY" not in os.environ:
        raise RuntimeError("GROQ_API_KEY not set. Run: export GROQ_API_KEY='your-key-here'")
    llm_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    return {
        "clf": clf,
        "label_names": label_names,
        "abstention_threshold": abstention_threshold,
        "corpus_df": corpus_df,
        "bm25": bm25,
        "collection": collection,
        "embed_model": embed_model,
        "llm_client": llm_client,
    }


# ---------------------------------------------------------------------------
# Helper functions (retrieval logic reused from Phase 3/4)
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
# Agent nodes -- each takes and returns the shared SupportState
# ---------------------------------------------------------------------------
def make_classifier_node(artifacts):
    def classifier_agent(state: SupportState) -> SupportState:
        query = state["query"]
        emb = artifacts["embed_model"].encode([query])
        probs = artifacts["clf"].predict_proba(emb)[0]

        # Use top-2 cumulative confidence instead of top-1 alone --
        # fixes cases where confidence splits across two valid, related
        # intents (e.g. track_refund vs get_refund)
        top2_conf = np.sort(probs)[::-1][:2].sum()
        pred_idx = probs.argmax()

        if top2_conf < artifacts["abstention_threshold"]:
            state["intent"] = "out_of_scope"
            state["classifier_confidence"] = float(top2_conf)
            state["escalated"] = True
            state["escalation_reason"] = "classifier_out_of_scope"
        else:
            state["intent"] = artifacts["label_names"][pred_idx]
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
    # Terminal node -- in a real system this would create a support ticket,
    # notify a human agent, etc. Here it just finalizes the state.
    state["answer"] = None
    return state


# ---------------------------------------------------------------------------
# Conditional routing functions
# ---------------------------------------------------------------------------
def route_after_classifier(state: SupportState) -> str:
    return "escalate" if state["escalated"] else "retrieve"


def route_after_retriever(state: SupportState) -> str:
    return "escalate" if state["escalated"] else "respond"


def route_after_responder(state: SupportState) -> str:
    return "escalate" if state["escalated"] else "end"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph(artifacts):
    graph = StateGraph(SupportState)

    graph.add_node("classify", make_classifier_node(artifacts))
    graph.add_node("retrieve", make_retriever_node(artifacts))
    graph.add_node("respond", make_responder_node(artifacts))
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify", route_after_classifier, {"retrieve": "retrieve", "escalate": "escalate"}
    )
    graph.add_conditional_edges(
        "retrieve", route_after_retriever, {"respond": "respond", "escalate": "escalate"}
    )
    graph.add_conditional_edges(
        "respond", route_after_responder, {"end": END, "escalate": "escalate"}
    )
    graph.add_edge("escalate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Main -- run a batch of test queries through the full pipeline
# ---------------------------------------------------------------------------
def main():
    artifacts = load_all_artifacts()

    print("\nBuilding LangGraph state graph...")
    app = build_graph(artifacts)

    test_queries = [
        "I want to cancel my order",
        "my payment failed, what should I do",
        "the thing I bought never showed up",
        "can you recommend a good pizza place",   # should be caught at classifier stage
    ]

    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"Query: {query!r}")
        print("=" * 70)

        initial_state: SupportState = {
            "query": query,
            "intent": None,
            "classifier_confidence": None,
            "retrieved_docs": None,
            "answer": None,
            "escalated": False,
            "escalation_reason": None,
        }

        final_state = app.invoke(initial_state)

        print(f"Intent: {final_state['intent']} (confidence: {final_state['classifier_confidence']:.4f})")

        if final_state["escalated"]:
            print(f"🚩 ESCALATED — reason: {final_state['escalation_reason']}")
        else:
            print(f"✅ ANSWERED:\n{final_state['answer']}")


if __name__ == "__main__":
    main()