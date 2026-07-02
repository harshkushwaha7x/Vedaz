"""
Step 2: Fine-tuning — finetune.py

Fine-tunes Qwen2.5-7B-Instruct (or Qwen3-8B) on Vedaz chat data
using QLoRA (4-bit quantization + LoRA adapters).

Why QLoRA:
  - A full fine-tune of a 7B model needs ~56GB VRAM. QLoRA reduces this
    to ~10-12GB, making it runnable on a single A100 40GB or even an
    A10G 24GB (available cheaply on Lambda Labs, Vast.ai, or Google Colab).
  - LoRA adapters are small (~50-100MB) — easy to version, swap, or roll back.
  - Quality is close to full fine-tuning for domain-adaptation tasks like this.

Minimum GPU requirement: 16GB VRAM (e.g. A10G, RTX 3090/4090, A100 40GB).
For Qwen3-4B, 12GB VRAM is enough.

Install dependencies first:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers==4.47.0 peft==0.13.0 trl==0.12.0 bitsandbytes==0.44.0
    pip install datasets accelerate einops

Usage:
    # Basic run (Qwen2.5-7B)
    python finetune.py

    # Use Qwen3-8B instead
    python finetune.py --model Qwen/Qwen3-8B-Instruct

    # Smaller model for limited VRAM (12GB)
    python finetune.py --model Qwen/Qwen2.5-3B-Instruct

    # Resume from checkpoint
    python finetune.py --resume-from-checkpoint output/vedaz-qwen/checkpoint-50

    # Custom data path
    python finetune.py --train-data data/train.jsonl --val-data data/val.jsonl
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# ── Imports (deferred inside main so the file can be syntax-checked without GPU) ──
def main(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 1. Load tokenizer ────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="right",   # right-pad for SFT with attention mask
    )
    # Qwen models may not have a pad token set — use eos as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 2. Dataset preparation ───────────────────────────────────────────────
    import json

    def load_jsonl(path: str) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    train_raw = load_jsonl(args.train_data)
    val_raw   = load_jsonl(args.val_data)
    print(f"Train: {len(train_raw)} chats  |  Val: {len(val_raw)} chats")

    def format_chat(example: dict) -> dict:
        """Convert a messages-format chat to a single 'text' string using
        the model's native chat template. This produces the exact token
        sequence the model was pre-trained on, so fine-tuning stays in-domain.

        We mask the loss on system and user turns (train on assistant turns
        only) by using DataCollatorForCompletionOnlyLM below.
        """
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = Dataset.from_list(train_raw).map(format_chat)
    val_ds   = Dataset.from_list(val_raw).map(format_chat)

    # Sanity check — show one formatted example
    print("\nSample formatted text (first 400 chars):")
    print(train_ds[0]["text"][:400])
    print("...")

    # ── 3. Quantisation config (4-bit NF4) ──────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NF4 is better than FP4 for LLMs
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,       # nested quantization saves ~0.4 GB
    )

    # ── 4. Load base model ───────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",                   # spread across all available GPUs
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
    )
    model.config.use_cache = False           # required for gradient checkpointing
    model.enable_input_require_grads()       # required for PEFT with quantized model

    # ── 5. LoRA config ───────────────────────────────────────────────────────
    # Target modules for Qwen2.5 / Qwen3. These are the attention projection
    # matrices. Adding gate/up/down (MLP) targets improves quality slightly
    # at the cost of more parameters.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",    # attention
            "gate_proj", "up_proj", "down_proj",         # MLP
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Typical output: trainable params ~20-80M out of 7B total (~0.3-1.1%)

    # ── 6. Data collator — train only on assistant turns ────────────────────
    # The response template token sequence marks the start of each assistant
    # turn. DataCollatorForCompletionOnlyLM sets all tokens before it to -100
    # so the loss is computed only on the assistant's words, not on system
    # or user content.
    #
    # Qwen2.5 / Qwen3 chat template uses "<|im_start|>assistant\n" to open
    # assistant turns. We find its token IDs once and reuse.
    response_template = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    # ── 7. Training arguments ────────────────────────────────────────────────
    output_dir = args.output_dir
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,

        # Batch size: 1 per device with gradient accumulation simulates a
        # larger effective batch. 1 * 8 = 8 is a good starting point.
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,

        # Optimizer
        optim="paged_adamw_8bit",            # 8-bit Adam saves VRAM
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,

        # Memory efficiency
        gradient_checkpointing=True,          # trade compute for memory
        fp16=False,
        bf16=True,                            # bfloat16 is stable on A100/H100

        # Sequence length
        max_seq_length=args.max_seq_len,

        # Logging & checkpointing
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,                   # keep only last 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        report_to="none",                     # disable wandb/tensorboard by default
                                              # set to "wandb" if you have wandb configured

        # Misc
        dataloader_num_workers=2,
        remove_unused_columns=False,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    # ── 8. Trainer ───────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    # ── 9. Train ─────────────────────────────────────────────────────────────
    print("\nStarting training...")
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    # ── 10. Save LoRA adapters ───────────────────────────────────────────────
    adapter_path = os.path.join(output_dir, "lora_adapters")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA adapters saved to: {adapter_path}")
    print("Next step: run merge_model.py to merge adapters into the base model for vLLM.")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5/Qwen3 on Vedaz chat data.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="HuggingFace model ID. Options: Qwen/Qwen2.5-7B-Instruct, "
                         "Qwen/Qwen3-8B, Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen3-4B")
    ap.add_argument("--train-data", default="data/train.jsonl")
    ap.add_argument("--val-data", default="data/val.jsonl")
    ap.add_argument("--output-dir", default="output/vedaz-qwen")
    ap.add_argument("--epochs", type=int, default=3,
                    help="Number of training epochs. 3-5 works well for 50 chats.")
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="Learning rate. 2e-4 is standard for QLoRA.")
    ap.add_argument("--lora-r", type=int, default=16,
                    help="LoRA rank. Higher = more parameters = better quality but more VRAM.")
    ap.add_argument("--lora-alpha", type=int, default=32,
                    help="LoRA alpha. Typically 2x the rank.")
    ap.add_argument("--max-seq-len", type=int, default=2048,
                    help="Max sequence length. Most Vedaz chats fit in 1024 tokens.")
    ap.add_argument("--use-flash-attn", action="store_true",
                    help="Use Flash Attention 2 (requires pip install flash-attn). "
                         "Speeds up training ~2x and reduces VRAM.")
    ap.add_argument("--resume-from-checkpoint", default=None,
                    help="Path to a checkpoint directory to resume training from.")
    args = ap.parse_args()
    main(args)
