import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

from datasets import load_dataset
import pandas as pd

print("="*60)
print("CHECK 1: Banking77")
print("="*60)
banking77 = load_dataset("legacy-datasets/banking77")
label_names = banking77["train"].features["label"].names

print(f"Train size: {len(banking77['train'])} (expected 10,003)")
print(f"Test size: {len(banking77['test'])} (expected 3,080)")
print(f"Number of intent classes: {len(label_names)} (expected 77)")
print(f"Sample row: {banking77['train'][0]}")
print(f"First 5 label names: {label_names[:5]}")

# Check label distribution isn't badly broken (e.g. all one class)
train_df = pd.DataFrame(banking77["train"])
print(f"\nLabel distribution (should be roughly even across 77 classes):")
print(train_df["label"].value_counts().describe())

print("\n" + "="*60)
print("CHECK 2: CLINC150 OOS source")
print("="*60)
clinc = load_dataset("cmaldona/All-Generalization-OOD-CLINC150")
print(f"Splits available: {list(clinc.keys())}")
print(f"Columns: {clinc['train'].column_names}")
print(f"Sample row: {clinc['train'][0]}")

# Confirm 'ood' actually exists as a label value
clinc_train_df = pd.DataFrame(clinc["train"])
unique_labels = clinc_train_df["labels"].unique()
print(f"\nTotal unique labels: {len(unique_labels)} (expect 151 = 150 intents + ood)")
print(f"Is 'ood' present? {'ood' in unique_labels}")

oos_count = (clinc_train_df["labels"] == "ood").sum()
print(f"OOS example count in train: {oos_count} (should be > 0)")

print("\n" + "="*60)
print("CHECK 3: Cross-check label overlap")
print("="*60)
# Sanity check: Banking77 and CLINC150 shouldn't have identical label sets
# (they're different domains) -- just confirming they're actually distinct
banking77_labels = set(label_names)
clinc_labels = set(unique_labels) - {"ood"}
overlap = banking77_labels & clinc_labels
print(f"Label name overlap between Banking77 and CLINC150: {len(overlap)}")
print("(Some overlap is fine/expected since both cover some banking-adjacent intents)")

print("\n" + "="*60)
print("ALL CHECKS COMPLETE")
print("="*60)