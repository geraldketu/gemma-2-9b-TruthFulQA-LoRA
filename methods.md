
# Methods & Configuration

This document describes how LoRA and DPO were implemented to fine‑tune Gemma‑2 9B on the TruthfulQA “generation” split, including all formulas in Markdown math syntax.

---

### 1. Environment Setup

- **Hardware constraints:** Gemma‑2 9B requires ∼40 GB of GPU RAM to load. Full fine‑tuning (updating all parameters) is infeasible on standard hardware.
- **Solution:**  
  - Use **LoRA** adapters to reduce trainable parameters.  
  - Use **DPO** to align outputs with human preferences without RL.

Install core packages:

```bash
pip install transformers datasets trl peft accelerate huggingface_hub python-dotenv
```
## Create a .env file containing:

```text

ACCESS_TOKEN=your_hf_api_token_here
```
## Disable TorchDynamo for stability:

```python
import os
os.environ["TORCHDYNAMO_DISABLE"]   = "1"
os.environ["TORCH_COMPILE_BACKEND"] = "eager"
```
### 2. Data Preparation
## 2.1 Load Dataset
```python
from datasets import load_dataset

ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
ds = ds.remove_columns(["type", "category", "correct_answers", "source"])
```
## 2.2 Format for DPO
For each question \($x$\), build pairs \(($y^+$, $y^-$)\) where:

- $y^+$ = human‑judged best answer  
- $y^-$ = each incorrect alternative  



```python
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
```
### 3. LoRA Implementation

LoRA injects low‑rank adapters into attention weights, freezing the base model:

- **Original weight matrix**:  
   $W_0 \in \mathbb{R}^{d \times k}$
- **Adapter update**:  
  $\Delta W = A\,B,\quad A \in \mathbb{R}^{d \times r},\; B \in \mathbb{R}^{r \times k},\; r \ll \min(d,k)$
- **Effective weight**:  
  $W = W_0 + \Delta W$

This reduces trainable parameters from $\(d \times k\)$ to $\(r(d + k)\)$. We used:

```python
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["self_attn.q_proj", "self_attn.v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(base_model, lora_config)
```
- **Rank** \($r = 8$\) balances expressivity and memory.  
- **Alpha** \($\alpha = 16$\) scales the adapter update.  
- **Dropout** $= 0.05$ regularizes adapters.

### 4. DPO Implementation

Direct Preference Optimization minimizes the loss  
$\mathcal{L}_{\mathrm{DPO}} = \mathbb{E}_{(x,y^+,y^-)}\bigl[\log\bigl(1 + \exp\bigl(-\alpha\,(s_\theta(x,y^+) - s_\theta(x,y^-))\bigr)\bigr)\bigr]$

where  
$s_\theta(x,y) = \log p_\theta(y \mid x)$, and $\alpha > 0$ is a temperature hyperparameter.

Implement with `sklearn.linear_model.LinearRegression`:

```python
from trl import DPOConfig, DPOTrainer

dpo_args = DPOConfig(
    output_dir="models/",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    alpha=1.0,            # temperature for preference margin
    save_strategy="no"
)

trainer = DPOTrainer(
    model=model,
    args=dpo_args,
    train_dataset=train_ds
)
trainer.train()
```

- **Batch size/device**\ (= 2\), accumulation \(= 4 →\) effective batch of \(8\)
- **Epochs** \( = 1\) for quick alignment on validation data
- **Alpha** \(\$alpha =1.0\) standard scaling for the logistic preference loss

## 5. Merge & Deployment
After training, merge LoRA adapters into the base model:

```python

from peft import merge_adapter

merged_model = merge_adapter(model)
merged_model.save_pretrained("Gemma-DPO-Merged")
tokenizer.save_pretrained("Gemma-DPO-Merged")
```
Push to HuggingFace Hub:

```python

from huggingface_hub import HfApi

api = HfApi()
repo_id = "GeraldNdawula/gemma-2b-it-lora-dpo-tfQA"
api.create_repo(repo_id=repo_id, exist_ok=True)

merged_model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
```
Thanks for reading, you can check out my model [here](https://huggingface.co/GeraldNdawula/gemma-2b-it-lora-dpo-tfQA) 

