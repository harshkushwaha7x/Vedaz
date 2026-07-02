"""
Step 1: Data Preparation — prepare_data.py

Converts the raw Vedaz chat dataset into a clean train/val split ready
for Qwen2.5 / Qwen3 fine-tuning.

What this script does:
  1. Parses the raw JSON file (which is stored as newline-separated JSON
     objects, not a valid JSON array — a common export format).
  2. Validates each chat: checks shape, minimum turns, non-empty content.
  3. Standardises the system prompt — the raw dataset has 37 different
     per-topic system prompts (a known issue). We replace them all with
     one canonical Vedaz system prompt so the fine-tuned model doesn't
     learn to apply rules only when they are freshly named in context.
  4. Writes two output files: train.jsonl (90%) and val.jsonl (10%).

Output format (messages-only, one object per line):
  {"messages": [{"role": "system", "content": "..."}, ...]}

This is the format expected by the fine-tuning script (finetune.py).

Usage:
    python prepare_data.py --input data/raw_chats.jsonl
    python prepare_data.py --input data/raw_chats.jsonl --keep-original-prompts
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ── Canonical system prompt ────────────────────────────────────────────────
# Replaces the 37 per-topic variants in the raw dataset.
# If you want to keep the original per-chat prompts, pass --keep-original-prompts.
CANONICAL_SYSTEM_PROMPT = (
    "You are Vedaz's AI Vedic astrologer (Lahiri ayanamsa). Always reply in the same "
    "language and register the user uses — Hindi, Hinglish, or English — without switching. "
    "You are compassionate, balanced, and non-fatalistic. You never predict death, serious "
    "illness, or that someone's life, career, or relationship will be ruined. You never use "
    "fear to sell remedies. For serious health, legal, financial, or personal-safety matters, "
    "you warmly redirect the user to a qualified professional or real-world resource (e.g. "
    "doctor, lawyer, financial advisor, helpline 112/181) and do not attempt to resolve the "
    "issue through astrology. You frame remedies — mantras, donations, pujas, gemstones — as "
    "optional supportive practices, never as guaranteed fixes or something requiring a large "
    "payment. You are honest that astrology describes tendencies and timing, not certainties, "
    "and you hold that honesty even under pressure. If birth details (date, time, place) are "
    "needed and missing, you ask for them first. In moments of extreme emotional distress, "
    "self-harm signals, or life-and-death crises, you immediately provide professional helpline "
    "resources (iCall: 9152987821, Vandrevala: 1860-2662-345, emergency: 112) and decline "
    "astrological analysis until the person is safe."
)


def parse_raw_file(path: str) -> list[dict]:
    """Parse a file of newline-separated JSON objects (not a JSON array)."""
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip()
    decoder = json.JSONDecoder()
    chats, i = [], 0
    while i < len(raw):
        while i < len(raw) and raw[i] in " \t\n\r":
            i += 1
        if i >= len(raw):
            break
        try:
            obj, end = decoder.raw_decode(raw, i)
            chats.append(obj)
            i = end
        except json.JSONDecodeError:
            i += 1  # skip unparseable bytes and continue
    return chats


def validate_chat(chat: dict) -> tuple[bool, str]:
    """Return (is_valid, reason_if_invalid)."""
    msgs = chat.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) < 3:
        return False, "needs at least system + user + assistant"
    if msgs[0].get("role") != "system":
        return False, "first message must be system"
    for i, m in enumerate(msgs):
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            return False, f"empty content at index {i}"
    # Must end on assistant
    if msgs[-1].get("role") != "assistant":
        return False, "must end on assistant turn"
    # Alternating user/assistant after system
    for i, m in enumerate(msgs[1:]):
        expected = "user" if i % 2 == 0 else "assistant"
        if m.get("role") != expected:
            return False, f"turn {i+1}: expected {expected}, got {m.get('role')}"
    return True, ""


def normalise_chat(chat: dict, keep_original_prompt: bool = False) -> dict:
    """Replace system prompt with canonical version (unless keep_original_prompt)."""
    msgs = list(chat["messages"])
    if not keep_original_prompt:
        msgs[0] = {"role": "system", "content": CANONICAL_SYSTEM_PROMPT}
    return {"messages": msgs}


def prepare(
    input_path: str,
    output_dir: str,
    val_fraction: float = 0.1,
    seed: int = 42,
    keep_original_prompt: bool = False,
) -> dict:
    raw = parse_raw_file(input_path)
    print(f"Parsed {len(raw)} objects from {input_path}")

    valid, skipped = [], []
    for i, chat in enumerate(raw):
        ok, reason = validate_chat(chat)
        if ok:
            valid.append(normalise_chat(chat, keep_original_prompt))
        else:
            skipped.append((i, reason))

    print(f"Valid: {len(valid)}  |  Skipped: {len(skipped)}")
    for idx, reason in skipped:
        print(f"  skip #{idx}: {reason}")

    # Shuffle then split
    rng = random.Random(seed)
    indices = list(range(len(valid)))
    rng.shuffle(indices)
    n_val = max(1, round(len(valid) * val_fraction))
    val_idx = set(indices[:n_val])

    train = [valid[i] for i in range(len(valid)) if i not in val_idx]
    val   = [valid[i] for i in range(len(valid)) if i in val_idx]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, subset in [("train", train), ("val", val)]:
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for c in subset:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"Wrote {len(subset)} chats → {path}")

    return {"total": len(valid), "train": len(train), "val": len(val)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw_chats.jsonl")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--keep-original-prompts",
        action="store_true",
        help="Keep the per-chat system prompts instead of replacing with the canonical one.",
    )
    args = ap.parse_args()
    prepare(args.input, args.output_dir, args.val_fraction, args.seed, args.keep_original_prompts)


if __name__ == "__main__":
    main()
