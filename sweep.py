"""
Threshold Sweep: Abstention Percentile Tradeoff
-------------------------------------------------
Reuses the already-trained classifier (from artifacts/) to test multiple
abstention thresholds without retraining, so you can pick a threshold based
on an actual precision/recall tradeoff instead of one arbitrary value.

Run:
  python threshold_sweep.py
"""

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import numpy as np
import pandas as pd
import joblib
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ARTIFACTS_DIR = "artifacts"
PERCENTILES_TO_TEST = [3, 5, 8, 10, 15, 20]


def load_banking77():
    ds = load_dataset("legacy-datasets/banking77")
    label_names = ds["train"].features["label"].names
    train_df = pd.DataFrame(ds["train"])
    test_df = pd.DataFrame(ds["test"])
    return train_df, test_df, label_names


def load_clinc_oos_test():
    ds = load_dataset("cmaldona/All-Generalization-OOD-CLINC150")
    df = pd.DataFrame(ds["test"])
    oos_df = df[df["labels"] == "ood"].copy().rename(columns={"data": "text"})
    return oos_df[["text"]]


def main():
    print("Loading trained classifier from artifacts/ (no retraining needed)...")
    clf = joblib.load(os.path.join(ARTIFACTS_DIR, "classifier.joblib"))

    print("Loading data...")
    train_df, test_df, label_names = load_banking77()
    oos_test_df = load_clinc_oos_test()

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Recreate the same train/calibration split used during training
    # (same random_state=42 ensures this matches what the classifier was calibrated on originally)
    print("Embedding Banking77 train set (for calibration split)...")
    X_all_train = embed_model.encode(
        list(train_df["text"]), show_progress_bar=True, batch_size=64
    )
    y_all_train = train_df["label"].values

    _, X_cal, _, y_cal = train_test_split(
        X_all_train, y_all_train, test_size=0.1, random_state=42, stratify=y_all_train
    )

    print("Embedding Banking77 test set...")
    X_test = embed_model.encode(list(test_df["text"]), show_progress_bar=True, batch_size=64)
    y_test = test_df["label"].values

    print("Embedding CLINC150 OOS test set...")
    X_oos_test = embed_model.encode(
        list(oos_test_df["text"]), show_progress_bar=True, batch_size=64
    )

    # Precompute confidence scores once -- reused across all threshold values
    cal_probs = clf.predict_proba(X_cal)
    cal_max_conf = cal_probs.max(axis=1)

    test_probs = clf.predict_proba(X_test)
    test_preds = test_probs.argmax(axis=1)
    test_max_conf = test_probs.max(axis=1)

    oos_probs = clf.predict_proba(X_oos_test)
    oos_max_conf = oos_probs.max(axis=1)

    # Base accuracy/F1 (threshold-independent, computed once)
    base_acc = accuracy_score(y_test, test_preds)
    base_f1 = f1_score(y_test, test_preds, average="macro")

    print("\n" + "=" * 70)
    print(f"{'Percentile':<12}{'Threshold':<12}{'In-Scope':<14}{'OOS Caught':<14}{'OOS Missed':<12}")
    print(f"{'':12}{'':12}{'Escalated %':<14}{'(Recall)':<14}{'(FN Rate)':<12}")
    print("=" * 70)

    results = []
    for pct in PERCENTILES_TO_TEST:
        threshold = np.percentile(cal_max_conf, pct)

        # Of Banking77 test queries (all genuinely in-scope), what % get
        # wrongly escalated at this threshold? Lower is better -- this is
        # the "cost" side of the tradeoff.
        in_scope_escalated_rate = (test_max_conf < threshold).mean()

        # Of true OOS queries, what % get correctly escalated?
        # Higher is better -- this is the "safety" side of the tradeoff.
        oos_caught_rate = (oos_max_conf < threshold).mean()
        oos_missed_rate = 1 - oos_caught_rate

        results.append({
            "percentile": pct,
            "threshold": threshold,
            "in_scope_escalated_pct": in_scope_escalated_rate * 100,
            "oos_caught_pct": oos_caught_rate * 100,
            "oos_missed_pct": oos_missed_rate * 100,
        })

        print(
            f"{pct:<12}{threshold:<12.4f}"
            f"{in_scope_escalated_rate*100:<14.2f}"
            f"{oos_caught_rate*100:<14.2f}"
            f"{oos_missed_rate*100:<12.2f}"
        )

    print("=" * 70)
    print(f"\nBanking77 base accuracy (threshold-independent): {base_acc:.4f}")
    print(f"Banking77 base macro-F1 (threshold-independent): {base_f1:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(ARTIFACTS_DIR, "threshold_sweep_results.csv"), index=False)
    print(f"\nSaved full sweep results to '{ARTIFACTS_DIR}/threshold_sweep_results.csv'")

    print("\nHow to read this:")
    print("- Higher percentile -> higher threshold -> escalates more often overall")
    print("- 'In-Scope Escalated %' = cost: genuine queries wrongly sent to a human")
    print("- 'OOS Caught %' = safety: true out-of-scope queries correctly escalated")
    print("- Pick the row where OOS Caught is high enough for your risk tolerance,")
    print("  without pushing In-Scope Escalated too high (diminishing returns beyond ~15-20%ile).")


if __name__ == "__main__":
    main()