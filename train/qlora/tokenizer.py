# tokenize.py

from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
import os

# Load tokenizer
model_path = "deepseek-ai/deepseek-coder-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load dataset
data_files = {
    "train": "datasets/nova_train.json",
    "test": "datasets/nova_test.json"
}
raw_datasets = load_dataset("json", data_files=data_files)

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=2048)

# Apply tokenizer
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])

# Save to disk for training
output_path = "datasets/tokenized"
os.makedirs(output_path, exist_ok=True)

tokenized_datasets["train"].save_to_disk(os.path.join(output_path, "train"))
tokenized_datasets["test"].save_to_disk(os.path.join(output_path, "test"))

print(f"Tokenized datasets saved to {output_path}/train and /test")