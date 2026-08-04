import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

from datasets import load_dataset

intents = load_dataset("DeepPavlov/clinc150", "intents")
print(intents)
print(intents['intents'][:10])          # correct split name
print(intents['intents'].column_names)