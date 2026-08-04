"""
Phase 3: Retrieval Layer
-------------------------
Builds a hybrid (BM25 + dense) retrieval index over the Bitext customer
support corpus, so the retriever agent can pull relevant past resolutions
for a given incoming query.

Pipeline:
  1. Load the Bitext dataset -- each (instruction, response, category) row
     becomes a retrievable "document" (the response text).
  2. Chunk/clean the corpus (Bitext responses are already short, so this is
     light-touch -- mainly dedup + whitespace cleanup).
  3. Build a dense vector index (Chroma) over the corpus using the same
     sentence-transformer model as Phase 2, for consistency.
  4. Build a BM25 keyword index over the same corpus.
  5. Combine both into hybrid retrieval via reciprocal rank fusion (RRF).
  6. Sanity-check retrieval on a few sample queries.

Run:
  python phase3_retrieval.py
"""

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import re
import pickle
import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ARTIFACTS_DIR = "artifacts"
CHROMA_DIR = os.path.join(ARTIFACTS_DIR, "chroma_db")
COLLECTION_NAME = "support_corpus"
TOP_K = 5           # how many docs to retrieve per query
RRF_K = 60          # reciprocal rank fusion constant (standard default)

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Load the Bitext corpus
# ---------------------------------------------------------------------------
def load_bitext_corpus():
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    df = pd.DataFrame(ds["train"])
    print(f"Loaded Bitext: {len(df)} rows | columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Clean and prepare the corpus
# ---------------------------------------------------------------------------
def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def prepare_corpus(df):
    """
    Turns raw Bitext rows into a clean list of retrievable documents.
    Each document = the response text (what gets retrieved and shown as
    a grounding source). We keep the paired instruction + category as
    metadata for citation/debugging purposes.
    """
    # Adjust column names below if Bitext's schema differs -- verify with
    # df.columns before relying on this (same discipline as Phase 1/2).
    text_col = "response" if "response" in df.columns else "instruction"
    query_col = "instruction" if "instruction" in df.columns else None
    category_col = "category" if "category" in df.columns else None
    intent_col = "intent" if "intent" in df.columns else None

    df = df.copy()
    df["doc_text"] = df[text_col].apply(clean_text)

    if query_col:
        df["source_query"] = df[query_col].apply(clean_text)
    else:
        df["source_query"] = ""

    df["category"] = df[category_col] if category_col else "unknown"
    df["intent"] = df[intent_col] if intent_col else "unknown"

    # Dedup identical response texts (Bitext has templated responses that
    # repeat across many instruction variants)
    before = len(df)
    df = df.drop_duplicates(subset="doc_text").reset_index(drop=True)
    print(f"Deduplicated corpus: {before} -> {len(df)} unique documents")

    df["doc_id"] = [f"doc_{i}" for i in range(len(df))]
    return df[["doc_id", "doc_text", "source_query", "category", "intent"]]


# ---------------------------------------------------------------------------
# Step 3: Build dense vector index (Chroma)
# ---------------------------------------------------------------------------
def build_dense_index(corpus_df, embed_model):
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Fresh collection each run -- delete if it already exists to avoid
    # duplicate/stale entries across repeated script runs
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    print("Embedding corpus for dense index...")
    embeddings = embed_model.encode(
        list(corpus_df["doc_text"]), show_progress_bar=True, batch_size=64
    )

    # Chroma requires batched adds to stay well under its per-call limits
    batch_size = 500
    for start in range(0, len(corpus_df), batch_size):
        end = start + batch_size
        batch_df = corpus_df.iloc[start:end]
        collection.add(
            ids=batch_df["doc_id"].tolist(),
            embeddings=embeddings[start:end].tolist(),
            documents=batch_df["doc_text"].tolist(),
            metadatas=batch_df[["category", "intent", "source_query"]].to_dict("records"),
        )

    print(f"Dense index built: {collection.count()} documents indexed")
    return collection


# ---------------------------------------------------------------------------
# Step 4: Build BM25 keyword index
# ---------------------------------------------------------------------------
def simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25_index(corpus_df):
    tokenized_corpus = [simple_tokenize(doc) for doc in corpus_df["doc_text"]]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"BM25 index built over {len(tokenized_corpus)} documents")
    return bm25


# ---------------------------------------------------------------------------
# Step 5: Hybrid retrieval via Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------
def dense_search(collection, embed_model, query, top_k=TOP_K):
    query_emb = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=top_k)
    return list(zip(results["ids"][0], results["documents"][0], results["distances"][0]))


def bm25_search(bm25, corpus_df, query, top_k=TOP_K):
    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        (corpus_df.iloc[i]["doc_id"], corpus_df.iloc[i]["doc_text"], scores[i])
        for i in top_indices
    ]


def hybrid_search(collection, bm25, corpus_df, embed_model, query, top_k=TOP_K, rrf_k=RRF_K):
    """
    Reciprocal Rank Fusion: combines BM25 and dense rankings by rank
    position rather than raw score (avoids needing to normalize two very
    different score scales -- BM25 scores and cosine distances aren't
    directly comparable).
    """
    dense_results = dense_search(collection, embed_model, query, top_k=top_k * 2)
    bm25_results = bm25_search(bm25, corpus_df, query, top_k=top_k * 2)

    rrf_scores = {}
    doc_text_lookup = {}

    for rank, (doc_id, text, _) in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
        doc_text_lookup[doc_id] = text

    for rank, (doc_id, text, _) in enumerate(bm25_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
        doc_text_lookup[doc_id] = text

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"doc_id": doc_id, "text": doc_text_lookup[doc_id], "rrf_score": score}
        for doc_id, score in ranked
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading Bitext corpus...")
    raw_df = load_bitext_corpus()

    print("\nPreparing corpus...")
    corpus_df = prepare_corpus(raw_df)
    corpus_df.to_csv(os.path.join(ARTIFACTS_DIR, "retrieval_corpus.csv"), index=False)
    print(f"Saved cleaned corpus to '{ARTIFACTS_DIR}/retrieval_corpus.csv'")

    print("\nLoading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nBuilding dense index (Chroma)...")
    collection = build_dense_index(corpus_df, embed_model)

    print("\nBuilding BM25 index...")
    bm25 = build_bm25_index(corpus_df)

    # Save BM25 index + corpus reference so later phases (agent nodes) can
    # load them without rebuilding
    with open(os.path.join(ARTIFACTS_DIR, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    print(f"Saved BM25 index to '{ARTIFACTS_DIR}/bm25_index.pkl'")

    # ------------------------------------------------------------------
    # Sanity check: run a few sample queries through hybrid search
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SANITY CHECK: Sample retrieval results")
    print("=" * 70)

    test_queries = [
    "I want to cancel my order",
    "how do I get a refund",
    "my payment failed, what should I do",
    "the thing I bought never showed up",   # no obvious keyword overlap with "order"/"delivery"
    ]

    for query in test_queries:
        print(f"\nQuery: {query!r}")
        results = hybrid_search(collection, bm25, corpus_df, embed_model, query, top_k=3)
        for rank, r in enumerate(results, 1):
            preview = r["text"][:100] + ("..." if len(r["text"]) > 100 else "")
            print(f"  [{rank}] (RRF score: {r['rrf_score']:.4f}) {preview}")

    print("\nPhase 3 complete. Retrieval index ready for the responder agent (Phase 4).")


if __name__ == "__main__":
    main()