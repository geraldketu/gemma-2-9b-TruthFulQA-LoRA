#!/usr/bin/env python3
"""
train_gemma2_lora_dpo.py

Fine‑tune Gemma‑2 9B on the TruthfulQA “generation” split using
LoRA + Direct Preference Optimization (DPO), then merge and push
the resulting model & tokenizer to HuggingFace Hub.
"""

import os
from dotenv import load_dotenv

# 1. Environment setup
load_dotenv()  # expects .env in working dir with ACCESS_TOKEN=
os.environ["HF_HOME"] = os.getcwd() + "/hf_cache"
os.environ["TORCHDYNAMO_DISABLE"]   = "1"
os.environ["TORCH_COMPILE_BACKEND"] = "eager"

# 2. Imports
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType, merge_adapter
from trl import DPOConfig, DPOTrainer
from huggingface_hub import HfApi

# 3. Load & preprocess data
ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
ds = ds.remove_columns(["type", "category", "correct_answers", "source"])

def make_pairs(batch):
    chosen, rejected = [], []
    for q, best, wrongs in zip(
        batch["question"],
        batch["best_answer"],
        batch["incorrect_answers"]
    ):
        for w in wrongs:
            chosen.append(f"Question: {q}\nAnswer: {best}")
            rejected.append(f"Question: {q}\nAnswer: {w}")
    return {"chosen": chosen, "rejected": rejected}

train_ds = ds.map(make_pairs, batched=True, remove_columns=ds.column_names)

# 4. Load base model & tokenizer
model_name = "google/gemma-2-9b-it"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="bfloat16"
)

# 5. Configure and apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["self_attn.q_proj", "self_attn.v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)

# 6. Set up DPO training
dpo_args = DPOConfig(
    output_dir="models/",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    alpha=1.0,
    save_strategy="no",
)
trainer = DPOTrainer(
    model=model,
    args=dpo_args,
    train_dataset=train_ds
)

# 7. Train
trainer.train()

# 8. Merge LoRA adapters into base
merged_model = merge_adapter(model)
merged_dir   = "Gemma-DPO-Merged"
merged_model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)

# 9. Push to HuggingFace Hub
api = HfApi()
repo_id = "GeraldNdawula/gemma-2b-it-lora-dpo-tfQA"
api.create_repo(repo_id=repo_id, exist_ok=True)
merged_model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)

print("Training complete. Merged model pushed to:", repo_id)
