"""
Task 2 — Chat generator.

Uses an AI model to generate new Vedaz-voice example chats on demand.
Each generated chat is immediately run through the checker (Task 1).
Only chats that pass all safety checks and have a valid structure are kept.

Usage:
    # Single topic
    python generator.py --topic "career delay, Hindi"

    # Multiple topics from a file (one per line)
    python generator.py --topics-file topics.txt

    # Use Anthropic instead of Together AI
    python generator.py --topic "marriage compatibility" --provider anthropic

    # Generate until at least N good chats are collected
    python generator.py --topics-file topics.txt --target 10

    # Save to a custom output file
    python generator.py --topics-file topics.txt --output ../data/generated.jsonl

Environment variables:
    AI_PROVIDER        "together" (default) or "anthropic"
    TOGETHER_API_KEY   required if using Together AI
    ANTHROPIC_API_KEY  required if using Anthropic

No API key is ever written into this file or any other file in the repo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from vedaz_voice import FEWSHOT_EXAMPLES, RULES, SYSTEM_PROMPT
from checker import check_chat, load_jsonl
from ai_client import AIClientError, call_model_json


# ---------------------------------------------------------------------------
# Generator prompt construction
# ---------------------------------------------------------------------------

def _rules_summary() -> str:
    return "\n".join(f"  [{r['id']}] {r['description']}" for r in RULES)


def _few_shot_block() -> str:
    """Two worked examples embedded verbatim so the model can learn the exact
    JSON format and voice before generating its own."""
    parts = []
    for ex in FEWSHOT_EXAMPLES:
        parts.append("```json\n" + json.dumps(ex, ensure_ascii=False, indent=2) + "\n```")
    return "\n\n".join(parts)


GENERATION_SYSTEM = f"""You are a Vedic astrology training data expert for Vedaz, an AI astrologer
product. Your job is to write realistic, high-quality example conversations between a user and
the Vedaz AI Astrologer.

Vedaz's voice rules (ALL must be followed in the ASSISTANT turns you write):
{_rules_summary()}

Shared system prompt that the assistant always operates under — include it verbatim as the first
message with role "system" in every chat you write:
{json.dumps(SYSTEM_PROMPT, ensure_ascii=False)}

Two example conversations in the correct format:
{_few_shot_block()}

Output format rules:
- Reply with ONLY a single valid JSON object, no markdown fences, no preamble.
- The JSON must have a single key "messages" containing an array.
- The array must start with the system message (role "system"), then alternate user/assistant.
- The chat must end on an assistant turn.
- Minimum 3 turns total (system + at least one user + one assistant).
- Never predict death, illness, ruin, or guaranteed outcomes in assistant turns.
- If the topic involves a health, legal, financial, or personal-safety concern, the assistant
  MUST redirect to a professional or real-world resource.
"""


def _generation_user_prompt(topic: str, persona_hint: str = "") -> str:
    persona = f"\nUser persona: {persona_hint}" if persona_hint else ""
    return (
        f"Write one realistic Vedaz training conversation on this topic: {topic}{persona}\n\n"
        "The user should feel like a real person (give them a natural speaking style — anxious, "
        "casual, frustrated, doubtful, etc.). The assistant must follow all Vedaz voice rules "
        "strictly. Reply with ONLY the JSON object."
    )


# ---------------------------------------------------------------------------
# Persona / topic helpers
# ---------------------------------------------------------------------------

# Default topic list used when no --topics-file is provided but --target > 0
DEFAULT_TOPICS = [
    ("career delay, Hindi — user anxious after multiple exam failures", "anxious, slightly desperate"),
    ("marriage compatibility, Hinglish — skeptical user who thinks astrology is luck", "casual, mildly sceptical"),
    ("sade sati fears, Hindi — user heard something scary from a relative", "worried, has partial misinformation"),
    ("business investment timing, English — entrepreneur wants a yes/no", "impatient, data-driven"),
    ("gemstone recommendation, Hinglish — user was told to buy an expensive ruby", "confused, unsure whether to spend money"),
    ("health worry framed as a planetary question, Hindi — redirect required", "distressed, looking for reassurance"),
    ("love marriage vs arranged marriage timing, Hinglish — user under family pressure", "frustrated, a bit emotional"),
    ("foreign travel or settlement, English — job offer abroad", "excited but uncertain"),
    ("child's future / education path, Hindi — parent asking on behalf of child", "protective, a bit pushy"),
    ("personal safety disclosure, Hindi — mentions partner behaviour; redirect required", "hesitant, not sure how much to share"),
    ("daily horoscope query, Hinglish — very casual request", "breezy, low stakes"),
    ("remedy being sold by local pandit for large sum, Hindi — user wants second opinion", "suspicious, wants validation to refuse"),
    ("career change in mid-life, English — user doubting themselves", "reflective, somewhat vulnerable"),
    ("kaal sarp dosh myth, Hindi — user received expensive remedy quote", "scared, looking for confirmation"),
    ("exam result timing, Hinglish — student asking if they will pass", "nervous, half-joking tone"),
]


# ---------------------------------------------------------------------------
# Generation + validation loop
# ---------------------------------------------------------------------------

def _strip_id_tags(chat: dict) -> dict:
    """Remove any id/tags/label fields the model might have added; the
    deliverable format has only the 'messages' key."""
    return {"messages": chat.get("messages", chat.get("Messages", []))}


def generate_one(
    topic: str,
    persona_hint: str = "",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[Optional[dict], dict]:
    """Attempt to generate one valid chat for *topic*.

    Returns (chat_or_None, meta) where meta contains debug info.
    The chat is validated through checker.check_chat before being returned.
    """
    user_prompt = _generation_user_prompt(topic, persona_hint)
    parsed, raw = call_model_json(
        GENERATION_SYSTEM,
        user_prompt,
        provider=provider,
        model=model,
        temperature=0.85,
        max_tokens=2000,
        fix_attempts=1,
    )

    meta = {"topic": topic, "raw_length": len(raw) if raw else 0, "parse_ok": parsed is not None}

    if parsed is None:
        meta["reject_reason"] = "JSON parse failed"
        return None, meta

    # Normalise: model sometimes wraps in extra nesting
    if isinstance(parsed, list):
        parsed = {"messages": parsed}
    if "messages" not in parsed:
        meta["reject_reason"] = "no 'messages' key in output"
        return None, meta

    chat = _strip_id_tags(parsed)
    check = check_chat(chat, use_llm=False)

    meta["valid_shape"] = check["valid_shape"]
    meta["safety_flags"] = check["keyword_triggered"]
    meta["word_count"] = check["word_count"]

    if not check["valid_shape"]:
        meta["reject_reason"] = f"shape invalid: {check['shape_error']}"
        return None, meta

    if check["is_flagged"]:
        meta["reject_reason"] = f"safety flags: {check['keyword_triggered']}"
        return None, meta

    meta["reject_reason"] = None
    return chat, meta


def generate_batch(
    topics: list[tuple[str, str]],
    target: int,
    output_path: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_attempts_per_topic: int = 3,
    retry_delay: float = 1.5,
) -> list[dict]:
    """Generate chats until *target* good ones are saved.

    Cycles through *topics* repeatedly if needed. Each accepted chat is
    appended immediately to *output_path* so partial results are never lost
    even if the script is interrupted.
    """
    good: list[dict] = []
    attempt_log: list[dict] = []
    topic_index = 0
    pass_num = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        while len(good) < target:
            if topic_index >= len(topics):
                topic_index = 0
                pass_num += 1
                if pass_num > 3:
                    print(f"\nWarning: exhausted topics {pass_num} times with only "
                          f"{len(good)}/{target} good chats — stopping.", file=sys.stderr)
                    break

            topic, persona = topics[topic_index]
            topic_index += 1

            print(f"\n[{len(good)+1}/{target}] Topic: {topic[:60]}")
            success = False
            for attempt in range(1, max_attempts_per_topic + 1):
                try:
                    chat, meta = generate_one(topic, persona, provider, model)
                except AIClientError as e:
                    print(f"  attempt {attempt}: API error — {e}")
                    meta = {"topic": topic, "reject_reason": str(e), "parse_ok": False}
                    chat = None

                attempt_log.append({**meta, "attempt": attempt, "accepted": chat is not None})

                if chat is not None:
                    # tag with metadata for traceability (stripped when writing
                    # deliverable-format output below)
                    chat["_topic"] = topic
                    good.append(chat)
                    # write deliverable format (messages-only) immediately
                    out_f.write(
                        json.dumps({"messages": chat["messages"]}, ensure_ascii=False) + "\n"
                    )
                    out_f.flush()
                    print(f"  ✓ accepted ({meta.get('word_count', '?')} words)")
                    success = True
                    break
                else:
                    reason = meta.get("reject_reason", "unknown")
                    flags = meta.get("safety_flags", [])
                    print(f"  attempt {attempt} rejected: {reason}" +
                          (f" flags={flags}" if flags else ""))
                    if attempt < max_attempts_per_topic:
                        time.sleep(retry_delay)

            if not success:
                print(f"  ✗ skipping topic after {max_attempts_per_topic} attempts")

    return good, attempt_log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate Vedaz chat training data using an AI model.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--topic", help="single topic string, e.g. 'career delay, Hindi'")
    src.add_argument("--topics-file", help="text file with one topic per line")
    ap.add_argument("--target", type=int, default=1,
                    help="total number of good chats to generate (default 1)")
    ap.add_argument("--output", default="../data/generated_chats.jsonl",
                    help="output .jsonl path (messages-only format, appended per chat)")
    ap.add_argument("--provider", default=None, help="override AI_PROVIDER")
    ap.add_argument("--model", default=None, help="override model id")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="max regeneration attempts per topic before skipping")
    ap.add_argument("--log", default=None,
                    help="optional path to write a JSON attempt log")
    args = ap.parse_args()

    # Build topic list
    if args.topic:
        topics = [(args.topic, "")]
    elif args.topics_file:
        lines = Path(args.topics_file).read_text(encoding="utf-8").splitlines()
        topics = [(l.strip(), "") for l in lines if l.strip() and not l.startswith("#")]
    else:
        topics = DEFAULT_TOPICS
        print(f"No topic/file specified — using {len(topics)} built-in topics.")

    if not topics:
        print("No topics to generate from.", file=sys.stderr)
        sys.exit(1)

    target = max(args.target, len([(t, p) for t, p in topics]) if args.topic else args.target)
    if args.topic:
        target = args.target  # single topic → respect --target exactly

    print(f"Target: {args.target} good chats  |  Provider: {args.provider or 'from env (default: together)'}")
    print(f"Output: {args.output}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    good, log = generate_batch(
        topics,
        target=args.target,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        max_attempts_per_topic=args.max_attempts,
    )

    total_attempts = len(log)
    accepted = sum(1 for e in log if e["accepted"])
    rejected = total_attempts - accepted
    print(f"\n{'='*60}")
    print(f"Done. Accepted: {accepted}  Rejected: {rejected}  Total attempts: {total_attempts}")
    print(f"Saved to: {args.output}")

    if args.log:
        with open(args.log, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"Attempt log saved to: {args.log}")


if __name__ == "__main__":
    main()
