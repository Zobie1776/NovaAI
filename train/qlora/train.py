# train/qlora/train.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Load DeepSeek base model
model_path = "deepseek-ai/deepseek-coder-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Load dataset (NovaScript/NovaCore code samples and CLI logs)
dataset = load_dataset("json", data_files={
    "train": "datasets/nova_train.json",
    "test": "datasets/nova_test.json"
})

# Apply QLoRA config
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="outputs/qlora-novaai",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    evaluation_strategy="epoch",
    num_train_epochs=3,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=2e-4,
    warmup_steps=100,
    weight_decay=0.01,
    report_to="none"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# Start training
trainer.train()
