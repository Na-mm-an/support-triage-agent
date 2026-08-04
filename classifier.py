"""
Phase 2 (v2): Classifier Agent -- retrained on Bitext's own domain
----------------------------------------------------------------------
FIX: The original classifier was trained on Banking77, a different domain
than the Bitext retrieval corpus -- this caused genuinely answerable
queries ("my payment failed") to be wrongly escalated as out-of-scope,
discovered during Phase 5 integration testing.

This version trains on Bitext's own `intent` labels (27 classes,
e.g. cancel_order, track_order, payment_issue), so the classifier's
notion of "in scope" now matches the same domain the retriever searches.

CLINC150 OOS examples are still used for the abstention threshold --
they remain a good out-of-scope proxy since they're from an unrelated
general-assistant domain (weather, alarms, translation, etc.), genuinely
outside customer support.

Run:
  python phase2_classifier_v2.py
"""

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OUTPUT_DIR = "artifacts"
ABSTENTION_PERCENTILE = 10  # carried over from the earlier sweep decision

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------
def load_bitext_for_classification():
    """
    Loads Bitext and uses its own `intent` column as the classification
    target -- this is the fix: same domain as the retrieval corpus.
    """
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    df = pd.DataFrame(ds["train"])
    df = df.dropna(subset=["instruction", "intent"]).drop_duplicates(subset="instruction")

    print(f"Bitext: {len(df)} unique instruction rows")
    print(f"Number of intent classes: {df['intent'].nunique()}")
    print(f"Intent classes: {sorted(df['intent'].unique())}")

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["intent"])

    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), le


def load_clinc_oos():
    """
    CLINC150 OOS examples still serve as the out-of-scope calibration set --
    they're from an unrelated general-assistant domain (alarms, weather,
    translation, reservations, etc.), which is a legitimate proxy for
    'not a customer support query' regardless of which support corpus
    we're using downstream.
    """
    ds = load_dataset("cmaldona/All-Generalization-OOD-CLINC150")

    def extract_oos(split):
        df = pd.DataFrame(ds[split])
        oos_df = df[df["labels"] == "ood"].copy().rename(columns={"data": "text"})
        return oos_df[["text"]]

    return extract_oos("train"), extract_oos("validation"), extract_oos("test")


# ---------------------------------------------------------------------------
# Step 2: Embed text
# ---------------------------------------------------------------------------
def embed_texts(model, texts):
    return model.encode(list(texts), show_progress_bar=True, batch_size=64)


# ---------------------------------------------------------------------------
# Step 3: Train classifier
# ---------------------------------------------------------------------------
def train_classifier(X_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf


# ---------------------------------------------------------------------------
# Step 4: Calibrate abstention threshold
# ---------------------------------------------------------------------------
def calibrate_threshold(clf, X_val, percentile=ABSTENTION_PERCENTILE):
    probs = clf.predict_proba(X_val)
    max_conf = probs.max(axis=1)
    threshold = np.percentile(max_conf, percentile)
    return threshold


# ---------------------------------------------------------------------------
# Step 5: Evaluate
# ---------------------------------------------------------------------------
def evaluate_in_scope(clf, X_test, y_test):
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    print(f"\n[Bitext Test] Accuracy: {acc:.4f} | Macro-F1: {f1_macro:.4f}")
    return acc, f1_macro


def evaluate_oos_detection(clf, X_oos, threshold):
    probs = clf.predict_proba(X_oos)
    max_conf = probs.max(axis=1)
    flagged_as_oos = (max_conf < threshold).mean()
    print(f"[CLINC150 OOS] Correctly abstained on {flagged_as_oos:.2%} of true OOS queries")
    print(f"[CLINC150 OOS] False negative rate: {1 - flagged_as_oos:.2%}")
    return flagged_as_oos


# ---------------------------------------------------------------------------
# Step 6: Inference wrapper (used by the LangGraph classifier node)
# ---------------------------------------------------------------------------
def classify_query(text, embed_model, clf, label_encoder, threshold):
    emb = embed_model.encode([text])
    probs = clf.predict_proba(emb)[0]
    max_conf = probs.max()
    pred_idx = probs.argmax()

    if max_conf < threshold:
        return {"intent": "out_of_scope", "confidence": float(max_conf), "escalate": True}
    return {
        "intent": label_encoder.inverse_transform([pred_idx])[0],
        "confidence": float(max_conf),
        "escalate": False,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading Bitext for classification (own domain, not Banking77)...")
    train_df, test_df, label_encoder = load_bitext_for_classification()

    print("\nLoading CLINC150 OOS examples for abstention calibration...")
    _, oos_val, oos_test = load_clinc_oos()

    print("\nLoading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nEmbedding Bitext train/test...")
    X_train_full = embed_texts(embed_model, train_df["instruction"])
    y_train_full = train_df["label"].values
    X_test = embed_texts(embed_model, test_df["instruction"])
    y_test = test_df["label"].values

    # Hold out a slice purely for abstention-threshold calibration
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    print("\nTraining classifier on Bitext intents...")
    clf = train_classifier(X_tr, y_tr)

    print("\nCalibrating abstention threshold...")
    threshold = calibrate_threshold(clf, X_cal)
    print(f"Abstention threshold set at: {threshold:.4f}")

    print("\nEvaluating on Bitext test set...")
    evaluate_in_scope(clf, X_test, y_test)

    print("\nEmbedding CLINC150 OOS test set...")
    X_oos_test = embed_texts(embed_model, oos_test["text"])

    print("\nEvaluating OOS detection...")
    evaluate_oos_detection(clf, X_oos_test, threshold)

    # Overwrite the same artifact filenames Phase 5 already loads --
    # no downstream code changes needed
    joblib.dump(clf, os.path.join(OUTPUT_DIR, "classifier.joblib"))
    joblib.dump(label_encoder.classes_.tolist(), os.path.join(OUTPUT_DIR, "label_names.joblib"))
    joblib.dump(threshold, os.path.join(OUTPUT_DIR, "abstention_threshold.joblib"))
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))  # kept for reuse
    print(f"\nSaved updated classifier, label names, and threshold to '{OUTPUT_DIR}/'")

    print("\n--- Sanity check ---")
    test_queries = [
        "I want to cancel my order",
        "my payment failed, what should I do",
        "the thing I bought never showed up",
        "can you recommend a good pizza place",
    ]
    for q in test_queries:
        result = classify_query(q, embed_model, clf, label_encoder, threshold)
        print(f"Query: {q!r} -> {result}")


if __name__ == "__main__":
    main()