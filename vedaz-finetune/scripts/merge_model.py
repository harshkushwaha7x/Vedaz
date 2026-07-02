"""
Step 3: Merge LoRA adapters into base model — merge_model.py

vLLM can serve LoRA adapters dynamically, but merging them into the base
model is simpler and slightly faster at inference time. This script does
the merge and saves a complete, standalone model ready to load into vLLM.

Why merge:
  - vLLM's dynamic LoRA loading requires keeping the base model in memory
    plus the adapter on top — no extra overhead at runtime, but more complex
    to deploy and manage.
  - A merged model is a single folder: copy it to the VPS and vllm serve it.
    No adapter paths, no LoRA flags, no adapter compatibility concerns.

Output size:
  Qwen2.5-7B merged:  ~14GB (bfloat16) or ~7GB (int8 with llm-int8)
  Qwen2.5-3B merged:  ~6GB (bfloat16)
  Qwen3-8B merged:    ~16GB (bfloat16)

Usage:
    python merge_model.py
    python merge_model.py --base Qwen/Qwen3-8B --adapters output/vedaz-qwen/lora_adapters --output output/vedaz-qwen-merged
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Base model:    {args.base}")
    print(f"LoRA adapters: {args.adapters}")
    print(f"Output path:   {args.output}")

    # ── 1. Load base model in full precision (not quantized) ─────────────────
    # We merge in bfloat16 — the result is a regular weights file that vLLM
    # can load normally. Loading the quantized version (4-bit) and merging
    # would produce a broken merged model because the scale factors from
    # quantization don't survive the merge correctly.
    print("\nLoading base model in bfloat16 (this needs ~14GB RAM, not VRAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",          # merge on CPU to avoid VRAM limits
        trust_remote_code=True,
    )

    # ── 2. Load LoRA adapters on top ─────────────────────────────────────────
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(model, args.adapters)

    # ── 3. Merge and unload ───────────────────────────────────────────────────
    # merge_and_unload() mathematically folds the LoRA matrices (A·B * alpha/r)
    # into the base weight matrices, then removes the LoRA structure.
    # The result is a standard HuggingFace model with no PEFT dependency.
    print("Merging adapters into base weights (this takes 1-3 minutes on CPU)...")
    model = model.merge_and_unload()

    # ── 4. Save merged model ─────────────────────────────────────────────────
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {out}...")
    model.save_pretrained(out, safe_serialization=True)  # saves as .safetensors

    # Copy tokenizer from adapter directory (it's the same as the base tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.adapters, trust_remote_code=True)
    tokenizer.save_pretrained(out)

    # Write a small README so anyone picking up this folder knows what it is
    (out / "ABOUT.txt").write_text(
        f"Vedaz fine-tuned model\n"
        f"Base: {args.base}\n"
        f"Adapters: {args.adapters}\n"
        f"Merged: bfloat16 safetensors\n"
        f"Ready for: vllm serve {out}\n"
    )

    print(f"\nDone! Merged model saved to: {out}")
    size_gb = sum(f.stat().st_size for f in out.rglob("*.safetensors")) / 1e9
    print(f"Model size (safetensors): {size_gb:.1f} GB")
    print("\nNext step: upload to VPS and run vLLM (see vllm_hosting_guide.md)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge LoRA adapters into base model for vLLM.")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct",
                    help="HuggingFace base model ID (same as used in finetune.py)")
    ap.add_argument("--adapters", default="output/vedaz-qwen/lora_adapters",
                    help="Path to the LoRA adapter directory saved by finetune.py")
    ap.add_argument("--output", default="output/vedaz-qwen-merged",
                    help="Where to save the merged full model")
    args = ap.parse_args()
    main(args)
