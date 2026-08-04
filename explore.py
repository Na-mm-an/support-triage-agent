import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

from datasets import load_dataset
banking77 = load_dataset("legacy-datasets/banking77")
print(banking77)
print(banking77['train'][0])
print(banking77['train'].features)