# Vedaz Model Fine-Tuning & Deployment

Fine-tune Qwen2.5-7B-Instruct (or Qwen3-8B) on Vedaz AI Astrologer
chat data using QLoRA, then serve it with vLLM on a GPU VPS.

---

## Project structure

```
scripts/
  prepare_data.py     — clean & split the raw dataset
  finetune.py         — QLoRA training (Qwen2.5/Qwen3 + SFTTrainer)
  merge_model.py      — fold LoRA adapters into base weights for vLLM

data/
  raw_chats.jsonl     — 55 parsed training chats
  train.jsonl         — 49 training chats (written by prepare_data.py)
  val.jsonl           — 6 validation chats

output/
  vedaz-qwen/
    lora_adapters/    — written by finetune.py
  vedaz-qwen-merged/  — written by merge_model.py; ready for vLLM

vllm_hosting_guide.md — full VPS + vLLM deployment walkthrough
```

---

## Step 0 — Install dependencies (on a GPU machine, not the VPS yet)

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.47.0 peft==0.13.0 trl==0.12.0 \
            bitsandbytes==0.44.0 datasets accelerate einops
```

Minimum GPU for training: **16GB VRAM** (e.g. RTX 4090, A10G, A100).
For Qwen3-4B / Qwen2.5-3B: **12GB VRAM** is enough.

Free option: Google Colab Pro (A100 40GB) — upload the `scripts/` and
`data/` folders, run the three steps below in a notebook.

---

## Step 1 — Prepare data

```bash
python scripts/prepare_data.py \
    --input data/raw_chats.jsonl \
    --output-dir data
```

Output: `data/train.jsonl` (49 chats), `data/val.jsonl` (6 chats).

**What this does differently from the raw data:**
The original 55 chats each have a custom per-topic system prompt (37
different variants). This script replaces all of them with one canonical
Vedaz system prompt — so the model learns to apply all safety rules
from a single consistent instruction, not only when the relevant rule
is freshly named in context.

---

## Step 2 — Fine-tune

```bash
python scripts/finetune.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train-data data/train.jsonl \
    --val-data data/val.jsonl \
    --output-dir output/vedaz-qwen \
    --epochs 3 \
    --lr 2e-4 \
    --lora-r 16 \
    --lora-alpha 32
```

**Alternative models:**

| Model | VRAM needed | Notes |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | 16 GB | Default — good balance |
| `Qwen/Qwen3-8B` | 18 GB | Slightly better reasoning |
| `Qwen/Qwen2.5-3B-Instruct` | 10 GB | Fits on smaller GPUs |
| `Qwen/Qwen3-4B` | 12 GB | Good quality for size |

Training time (Qwen2.5-7B, 3 epochs, A100 40GB): ~10-15 minutes for 49 chats.

Output: `output/vedaz-qwen/lora_adapters/` (LoRA weights, ~80MB)

---

## Step 3 — Merge adapters

```bash
python scripts/merge_model.py \
    --base Qwen/Qwen2.5-7B-Instruct \
    --adapters output/vedaz-qwen/lora_adapters \
    --output output/vedaz-qwen-merged
```

This folds the LoRA adapters mathematically into the base weights,
producing a standalone model (~14GB for 7B). No PEFT dependency needed
at inference time.

Output: `output/vedaz-qwen-merged/` — a standard HuggingFace model folder
ready to upload to your VPS.

---

## Step 4 — Deploy on VPS with vLLM

See **`vllm_hosting_guide.md`** for the full walkthrough. Summary:

```bash
# On VPS — install vLLM
pip install vllm==0.6.3

# Upload model (from training machine)
rsync -avzP output/vedaz-qwen-merged/ user@your-vps:~/vedaz-server/models/vedaz-qwen-merged/

# Start server
vllm serve models/vedaz-qwen-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name vedaz-astrologer \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90

# Test
curl http://localhost:8000/health
```

---

## Design decisions

**QLoRA, not full fine-tune:**
Full fine-tuning Qwen2.5-7B needs ~56GB VRAM. QLoRA (4-bit NF4 base +
LoRA adapters) reduces this to ~10-12GB with minimal quality loss for
domain-adaptation tasks. LoRA adapters are also ~80MB vs ~14GB for the
full weights — much easier to version and roll back.

**Train only on assistant turns:**
We use `DataCollatorForCompletionOnlyLM` to mask loss on system and user
turns. The model only learns *how to respond* from the Vedaz examples,
not to mimic user messages.

**Merging for vLLM:**
vLLM supports dynamic LoRA adapter loading, but merging is simpler and
slightly faster at inference time (one model file, no adapter config).
For a setup where you want to A/B test different adapters on the same
base, dynamic LoRA is better — see the vLLM docs.

**55 chats is a small dataset:**
This fine-tune is a domain-adaptation / tone-alignment task, not teaching
the model new knowledge. 50 high-quality examples is actually reasonable
for that — research shows diminishing returns past a few hundred for
style/voice adaptation. The bigger risk is overfitting, which is why we
use only 3 epochs and validate on the 6-chat held-out set.
