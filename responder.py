"""
Phase 4: Responder Agent + Grounding (Groq version)
------------------------------------------------------
Takes a query + retrieved documents (from Phase 3's hybrid retriever) and
generates a grounded, cited answer -- or abstains and escalates to a human
if retrieval confidence is too low to answer safely.

Two-layer safety design:
  1. Retrieval-confidence check (BEFORE calling the LLM at all) -- if the
     hybrid RRF scores / dense similarity are too low, skip generation
     entirely and escalate. Catches cases where retrieval itself failed.
  2. Grounded-generation prompt -- the LLM is instructed to answer ONLY
     from the retrieved context and cite which source supports each claim,
     and to explicitly say "I don't have enough information" if the
     retrieved context doesn't actually answer the question.

Setup:

  pip install groq chromadb rank_bm25 sentence-transformers pandas numpy

Run:
  python phase4_responder.py
"""

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import pickle
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ARTIFACTS_DIR = "artifacts"
CHROMA_DIR = os.path.join(ARTIFACTS_DIR, "chroma_db")
COLLECTION_NAME = "support_corpus"

TOP_K = 5
RRF_K = 60

# Rough starting thresholds -- calibrate properly later the same way you
# swept Phase 2's abstention threshold, rather than trusting these blindly.
MIN_RRF_SCORE = 0.010
MIN_DENSE_SIMILARITY = 0.35

GROQ_MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Load Phase 3 artifacts
# ---------------------------------------------------------------------------
def load_retrieval_artifacts():
    corpus_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "retrieval_corpus.csv"))

    with open(os.path.join(ARTIFACTS_DIR, "bm25_index.pkl"), "rb") as f:
        bm25 = pickle.load(f)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    return corpus_df, bm25, collection, embed_model


# ---------------------------------------------------------------------------
# Retrieval (same hybrid logic as Phase 3, plus similarity tracking)
# ---------------------------------------------------------------------------
def simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def dense_search_with_similarity(collection, embed_model, query, top_k=TOP_K):
    query_emb = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=top_k)
    ids = results["ids"][0]
    docs = results["documents"][0]
    distances = results["distances"][0]
    similarities = [1 / (1 + d) for d in distances]  # monotonic transform, not true cosine sim
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

    rrf_scores = {}
    doc_text_lookup = {}
    dense_sim_lookup = {}

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


# ---------------------------------------------------------------------------
# Layer 1: Retrieval-confidence check (pre-generation gate)
# ---------------------------------------------------------------------------
def retrieval_confidence_check(results):
    if not results:
        return False, "no_results_retrieved"

    top_rrf = results[0]["rrf_score"]
    top_dense_sim = results[0]["dense_similarity"]

    if top_rrf < MIN_RRF_SCORE:
        return False, f"low_rrf_score ({top_rrf:.4f} < {MIN_RRF_SCORE})"

    if top_dense_sim < MIN_DENSE_SIMILARITY:
        return False, f"low_dense_similarity ({top_dense_sim:.4f} < {MIN_DENSE_SIMILARITY})"

    return True, "confident"


# ---------------------------------------------------------------------------
# Layer 2: Grounded generation with citation (Groq)
# ---------------------------------------------------------------------------
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
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['text']}")
    return "\n\n".join(lines)


def generate_grounded_answer(client, query, results):
    context_block = build_context_block(results)
    prompt = GROUNDED_PROMPT_TEMPLATE.format(context_block=context_block, query=query)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Full responder pipeline (this is what Phase 5's LangGraph node will call)
# ---------------------------------------------------------------------------
def respond_to_query(query, collection, bm25, corpus_df, embed_model, llm_client):
    results = hybrid_search(collection, bm25, corpus_df, embed_model, query)

    is_confident, reason = retrieval_confidence_check(results)
    if not is_confident:
        return {
            "answer": None,
            "escalated": True,
            "escalation_reason": f"retrieval_confidence_gate: {reason}",
            "sources": [],
        }

    raw_answer = generate_grounded_answer(llm_client, query, results)

    if raw_answer.strip() == "INSUFFICIENT_CONTEXT":
        return {
            "answer": None,
            "escalated": True,
            "escalation_reason": "llm_flagged_insufficient_context",
            "sources": [r["text"][:80] for r in results],
        }

    return {
        "answer": raw_answer,
        "escalated": False,
        "escalation_reason": None,
        "sources": [r["text"][:80] for r in results],
    }


# ---------------------------------------------------------------------------
# Main -- sanity check against Phase 3's known good AND known bad queries
# ---------------------------------------------------------------------------
def main():
    print("Loading Phase 3 retrieval artifacts...")
    corpus_df, bm25, collection, embed_model = load_retrieval_artifacts()

    print("Initializing Groq client...")
    if "GROQ_API_KEY" not in os.environ:
        raise RuntimeError(
            "GROQ_API_KEY not set. Run: export GROQ_API_KEY before running this script."
        )
    llm_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    test_queries = [
        "I want to cancel my order",              # should answer confidently
        "my payment failed, what should I do",     # should answer confidently
        "the thing I bought never showed up",       # should ESCALATE (weak retrieval from Phase 3)
    ]

    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"Query: {query!r}")
        print("=" * 70)
        result = respond_to_query(query, collection, bm25, corpus_df, embed_model, llm_client)

        if result["escalated"]:
            print(f"🚩 ESCALATED — reason: {result['escalation_reason']}")
        else:
            print(f"✅ ANSWERED:\n{result['answer']}")
            print(f"\nSources used: {len(result['sources'])}")


if __name__ == "__main__":
    main()