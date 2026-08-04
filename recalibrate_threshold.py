"""
Threshold Recalibration: Using Realistic Paraphrases
--------------------------------------------------------
FIX: The original abstention threshold (0.8438) was calibrated on a
held-out slice of Bitext's OWN templated training data -- which made the
classifier look artificially confident during calibration, but didn't
generalize to naturally-phrased queries (discovered via manual UI testing).

This script recalibrates using a small hand-written set of realistically
paraphrased IN-SCOPE queries (not copied from Bitext's templates) plus
genuine OOS queries, so the threshold reflects real generalization
instead of the model's confidence on its own training-like phrasing.

Run:
  python recalibrate_threshold.py
"""

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pandas as pd

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ARTIFACTS_DIR = "artifacts"

# Hand-written, naturally-phrased IN-SCOPE queries -- deliberately worded
# differently from Bitext's templated style, to test real generalization.
# Each is labeled with which Bitext intent it should map to, for reference
# (not used for training, just for your own sanity-checking of output).
PARAPHRASED_IN_SCOPE_QUERIES = [
    ("how do I change my shipping address", "change_shipping_address"),
    ("where's my refund", "track_refund"),
    ("I need to update my delivery address", "change_shipping_address"),
    ("can I get my money back", "get_refund"),
    ("my order hasn't arrived yet", "track_order"),
    ("I want to talk to a real person", "contact_human_agent"),
    ("how much does it cost to cancel", "check_cancellation_fee"),
    ("I forgot my password", "recover_password"),
    ("can you send me my invoice", "get_invoice"),
    ("what payment methods do you accept", "check_payment_methods"),
    ("I want to leave a review", "review"),
    ("how do I close my account", "delete_account"),
    ("having trouble signing up", "registration_problems"),
    ("when will my order ship", "delivery_period"),
    ("I'd like to change what's in my order", "change_order"),
]

# Genuinely OOS queries -- unrelated to customer support at all
GENUINE_OOS_QUERIES = [
    "what's the weather like today",
    "can you recommend a good pizza place",
    "what's the capital of France",
    "tell me a joke",
    "set an alarm for 7am",
    "translate hello into Spanish",
    "who won the game last night",
    "what time is it in Tokyo",
]


def main():
    print("Loading classifier...")
    clf = joblib.load(os.path.join(ARTIFACTS_DIR, "classifier.joblib"))
    label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nEmbedding paraphrased in-scope queries...")
    in_scope_texts = [q for q, _ in PARAPHRASED_IN_SCOPE_QUERIES]
    in_scope_embeddings = embed_model.encode(in_scope_texts)
    in_scope_probs = clf.predict_proba(in_scope_embeddings)
    in_scope_max_conf = in_scope_probs.max(axis=1)
    in_scope_preds = in_scope_probs.argmax(axis=1)

    print("\nEmbedding genuine OOS queries...")
    oos_embeddings = embed_model.encode(GENUINE_OOS_QUERIES)
    oos_probs = clf.predict_proba(oos_embeddings)
    oos_max_conf = oos_probs.max(axis=1)

    print("\n" + "=" * 80)
    print("IN-SCOPE (paraphrased) query confidences:")
    print("=" * 80)
    for (query, expected_intent), conf, pred_idx in zip(
        PARAPHRASED_IN_SCOPE_QUERIES, in_scope_max_conf, in_scope_preds
    ):
        predicted_intent = label_encoder.inverse_transform([pred_idx])[0]
        match = "✅" if predicted_intent == expected_intent else "❓"
        print(f"{match} conf={conf:.4f} | expected={expected_intent:<28} predicted={predicted_intent}")
        print(f"   query: {query!r}")

    print("\n" + "=" * 80)
    print("GENUINE OOS query confidences:")
    print("=" * 80)
    for query, conf in zip(GENUINE_OOS_QUERIES, oos_max_conf):
        print(f"conf={conf:.4f} | {query!r}")

    print("\n" + "=" * 80)
    print("THRESHOLD RECOMMENDATION")
    print("=" * 80)
    print(f"Min confidence among paraphrased in-scope queries: {in_scope_max_conf.min():.4f}")
    print(f"Max confidence among genuine OOS queries:          {oos_max_conf.max():.4f}")

    if in_scope_max_conf.min() > oos_max_conf.max():
        # Clean separation exists -- pick a threshold in the gap
        new_threshold = (in_scope_max_conf.min() + oos_max_conf.max()) / 2
        print(f"\n✅ Clean separation found. Recommended threshold: {new_threshold:.4f}")
    else:
        # Overlap exists -- pick a threshold that favors catching OOS
        # (a few percentile points below the in-scope minimum, erring
        # toward not wrongly escalating genuine queries)
        new_threshold = np.percentile(in_scope_max_conf, 10)
        print(f"\n⚠️  Overlap found between in-scope and OOS confidence ranges.")
        print(f"Some genuine queries may be as uncertain as some OOS queries --")
        print(f"this is expected with a lightweight classifier on paraphrased input.")
        print(f"Recommended threshold (10th percentile of in-scope confidences): {new_threshold:.4f}")

    # Save the new threshold, overwriting the old one
    joblib.dump(new_threshold, os.path.join(ARTIFACTS_DIR, "abstention_threshold.joblib"))
    print(f"\nSaved new threshold ({new_threshold:.4f}) to artifacts/abstention_threshold.joblib")
    print("Old threshold (0.8438, calibrated on templated data) has been replaced.")


if __name__ == "__main__":
    main()
