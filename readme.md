

## Overview  
This repository demonstrates how to align a large language model’s outputs with human judgments of correctness by combining two techniques:

- **LoRA (Low‑Rank Adaptation):**  
  A parameter‑efficient fine‑tuning method that inserts small trainable “adapter” matrices into each attention layer. Instead of updating all model weights, LoRA freezes the base model and only learns a low‑rank decomposition. This dramatically reduces memory and compute requirements while retaining the expressivity needed to adapt the model to a new task.

- **DPO (Direct Preference Optimization):**  
  A fine‑tuning algorithm that trains directly on human preference pairs (“chosen” vs. “rejected” responses) without requiring a separate reward model or reinforcement‐learning loop. DPO maximizes the probability that the model prefers the chosen answer over the rejected one, aligning its behavior with human judgments of quality and truthfulness.

### Why Use Them Together?  
Fine‑tuning a 9‑billion‑parameter model like Gemma‑2 on preference data can be prohibitively expensive if all parameters are updated. By combining LoRA with DPO:

1. **Efficiency:** LoRA reduces the number of trainable parameters by orders of magnitude, making DPO training feasible on limited hardware.  
2. **Alignment:** DPO ensures the model’s outputs reflect human preferences for correct answers.  
3. **Flexibility:** The small LoRA adapters can be merged back into the base model after training, yielding a standalone, aligned model.

### Goal  
Our aim is to produce a version of Gemma‑2 9B that:

1. **Understands** the difference between correct and incorrect answers on open‑ended questions (TruthfulQA “generation” split).  
2. **Generates** responses that are more truthful and aligned with human preferences, without full‐scale reinforcement learning.  
3. **Maintains** parameter efficiency and ease of deployment by using LoRA adapters.

## Repository Structure  
├── notebooks/
│ └── modeling.ipynb # End-to-end: data formatting → LoRA → DPO → evaluation
├── requirements.txt # Python dependencies
├── .env # ACCESS_TOKEN for HuggingFace Hub
├── README.md # Project overview (this file)
└── methods.md # Detailed methods & configuration


## Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/geraldketu/gemma-2-9b-lora-dpo-tfQA.git
   cd gemma-2-9b-lora-dpo-tfQA```
Install dependencies

```bash

pip install -r requirements.txt
```
Configure credentials
Create a file named `.env` containing:
```
ini

ACCESS_TOKEN=your_hf_api_token_here
```
Usage
After training, load and use the aligned model:

```python

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("GeraldNdawula/gemma-2b-it-lora-dpo-tfQA")
model     = AutoModelForCausalLM.from_pretrained("GeraldNdawula/gemma-2b-it-lora-dpo-tfQA")
```
