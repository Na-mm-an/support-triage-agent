"""
Confidence Fix: Top-K Cumulative Confidence
------------------------------------------------
PROBLEM: "where's my refund" was escalated as out-of-scope even though
it's clearly a refund question -- because confidence split almost evenly
between two valid intents (track_refund: 0.41, get_refund: 0.39), so
neither alone cleared the single-class threshold.

FIX: Use the SUM of the top-2 class probabilities as the confidence
signal instead of just the top-1 class.
"""

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

ARTIFACTS_DIR = "artifacts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def top_k_confidence(probs, k=2):
    """Sum of the top-k class probabilities -- the new confidence signal."""
    top_k = np.sort(probs)[::-1][:k]
    return top_k.sum()


def classify_query_v2(text, embed_model, clf, label_encoder, threshold, top_k=2):
    emb = embed_model.encode([text])
    probs = clf.predict_proba(emb)[0]

    conf = top_k_confidence(probs, k=top_k)
    pred_idx = probs.argmax()  # still report the single best-guess intent

    if conf < threshold:
        return {
            "intent": "out_of_scope",
            "confidence": float(conf),
            "top1_confidence": float(probs[pred_idx]),
            "escalate": True,
        }
    return {
        "intent": label_encoder.inverse_transform([pred_idx])[0],
        "confidence": float(conf),
        "top1_confidence": float(probs[pred_idx]),
        "escalate": False,
    }


def main():
    print("Loading classifier + threshold...")
    clf = joblib.load(os.path.join(ARTIFACTS_DIR, "classifier.joblib"))
    label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
    threshold = joblib.load(os.path.join(ARTIFACTS_DIR, "abstention_threshold.joblib"))
    print(f"Current threshold (calibrated for top-1 confidence): {threshold:.4f}")
    print("NOTE: top-2 sums run naturally higher than top-1 alone --")
    print("we may need to raise the threshold too. Checking below.\n")

    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    test_cases = [
        ("how do I change my shipping address", False),
        ("where's my refund", False),           # the failing case -- should NOT escalate
        ("I want to cancel my order", False),
        ("my payment failed, what should I do", False),
        ("what's the weather like today", True),
        ("can you recommend a good pizza place", True),
        ("tell me a joke", True),
    ]

    print("=" * 90)
    print(f"{'Query':<45}{'Top1':<10}{'Top2 Sum':<12}{'Expected':<12}{'Result'}")
    print("=" * 90)

    for query, expected_escalate in test_cases:
        result = classify_query_v2(query, embed_model, clf, label_encoder, threshold, top_k=2)
        expected_str = "ESCALATE" if expected_escalate else "ANSWER"
        actual_str = "ESCALATE" if result["escalate"] else "ANSWER"
        match = "✅" if (result["escalate"] == expected_escalate) else "❌"

        print(
            f"{query:<45}{result['top1_confidence']:<10.4f}"
            f"{result['confidence']:<12.4f}{expected_str:<12}{actual_str} {match}"
        )


if __name__ == "__main__":
    main()