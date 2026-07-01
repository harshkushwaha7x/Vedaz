"""
Task 1 — Chat checker.

Reads a .jsonl file of chats (one {"messages": [...]} object per line, each
message a {"role", "content"} pair) and prints a report covering:

  1. Shape validation     — system, then strictly alternating user/assistant
  2. Length                — word count + a rough token estimate per chat
  3. Duplicates            — exact and near-duplicate chats
  4. Train/test split      — for reuse by generator.py / quality_tester.py
  5. Safety rule flagging  — keyword layer (always on) + optional LLM-judge
                              layer (--use-llm-judge, needs an API key)

Usage:
    python checker.py --input ../data/seed_chats.jsonl
    python checker.py --input ../data/seed_chats.jsonl --use-llm-judge
    python checker.py --input ../data/seed_chats.jsonl --write-splits

See README.md "Task 1" section for why this detection method was chosen and
where it is known to miss things.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from vedaz_voice import RULES, SYSTEM_PROMPT  # noqa: E402

try:
    from ai_client import call_model_json
except ImportError:  # pragma: no cover
    call_model_json = None


# ---------------------------------------------------------------------------
# Loading & shape validation
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    chats = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chats.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ! skipping malformed line {line_no}: {e}", file=sys.stderr)
    return chats


def validate_shape(chat: dict) -> tuple[bool, Optional[str]]:
    msgs = chat.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 3:
        return False, "needs a 'messages' list with at least system+user+assistant"
    if msgs[0].get("role") != "system":
        return False, "first message must have role 'system'"
    expected = ["user", "assistant"]
    for i, m in enumerate(msgs[1:], start=0):
        role = m.get("role")
        if "content" not in m or not isinstance(m.get("content"), str) or not m["content"].strip():
            return False, f"message {i+1} (after system) is missing non-empty 'content'"
        want = expected[i % 2]
        if role != want:
            return False, f"message {i+1} (after system) should be '{want}', got '{role}'"
    if expected[(len(msgs) - 2) % 2] != "assistant":
        # last turn should be assistant, not a dangling user question
        return False, "chat must end on an assistant turn"
    return True, None


# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------

def chat_text(chat: dict, roles=("user", "assistant")) -> str:
    msgs = chat.get("messages", [])
    return "\n".join(m.get("content", "") for m in msgs if m.get("role") in roles)


def word_count(chat: dict) -> int:
    text = chat_text(chat)
    return len(text.split())


def estimate_tokens(chat: dict) -> int:
    # Rough, provider-agnostic heuristic (~4 chars/token). This under-counts
    # for Devanagari text, where one "character" often costs more than one
    # BPE token — treat this as a ballpark, not an exact figure. For an
    # exact count against a specific model, use that model's own tokenizer.
    text = chat_text(chat)
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Duplicate / near-duplicate detection
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def shingles(text: str, n: int = 3) -> set[str]:
    words = normalize(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def find_duplicates(chats: list[dict], threshold: float = 0.6) -> list[dict]:
    """O(n^2) pairwise comparison — fine for the hundreds-of-chats scale this
    project deals with. For a real production corpus (tens of thousands+),
    this should move to MinHash/LSH bucketing instead of pairwise Jaccard;
    flagged as a known limitation in README rather than built speculatively
    here.
    """
    shingle_sets = [shingles(chat_text(c)) for c in chats]
    norm_texts = [normalize(chat_text(c)) for c in chats]
    results = []
    for i in range(len(chats)):
        for j in range(i + 1, len(chats)):
            if norm_texts[i] == norm_texts[j] and norm_texts[i] != "":
                results.append({"i": i, "j": j, "similarity": 1.0, "type": "exact"})
                continue
            sim = jaccard(shingle_sets[i], shingle_sets[j])
            if sim >= threshold:
                results.append({"i": i, "j": j, "similarity": round(sim, 3), "type": "near"})
    return results


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def train_test_split(chats: list[dict], test_fraction: float = 0.2, seed: int = 42):
    indices = list(range(len(chats)))
    random.Random(seed).shuffle(indices)
    n_test = max(1, round(len(chats) * test_fraction)) if chats else 0
    test_idx = set(indices[:n_test])
    train = [c for i, c in enumerate(chats) if i not in test_idx]
    test = [c for i, c in enumerate(chats) if i in test_idx]
    return train, test


# ---------------------------------------------------------------------------
# Safety rule flagging — keyword layer
# ---------------------------------------------------------------------------

# Words that signal the immediately surrounding phrase is a negation/refusal
# rather than an assertion. When a keyword appears within NEGATION_WINDOW
# words of one of these, we do NOT treat it as a violation for rules where
# the concern is *promising* something (guaranteed_outcome, fear_prediction).
#
# Limitation: this catches the most common pattern ("guarantee nahi deta")
# but will still miss creative negations ("main aise vaade nahi karta"),
# cross-sentence negations, and sarcasm — those cases are why the optional
# LLM-judge layer exists.
_NEGATION_TERMS = [
    "nahi", "nahin", "नहीं", "nah", "no", "not", "never", "na",
    "don't", "doesn't", "cannot", "can't", "refuse", "refusal",
    "nahi deta", "nahi karunga", "nahi karti", "nahi karta",
    "guarantee nahi", "गारंटी नहीं", "pakka nahi", "पक्का नहीं",
    # Condemnation / disapproval context — "X is wrong / false / I'm against X"
    "galat", "wrong", "khilaf", "against it", "jhooti", "jhootha",
    "jhoothi", "fake", "false", "baadh nahi", "badhya nahi", "majboor nahi",
]

# Pre-compiled patterns: each negation term as a whole-word regex so that
# "na" does not match inside "karwana", "never" inside "whenever", etc.
_NEGATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?<!\w)" + re.escape(t.lower()) + r"(?!\w)", re.UNICODE)
    for t in _NEGATION_TERMS
]
NEGATION_WINDOW = 8   # words on either side of the match to check


def _in_negation_context(text: str, match_start: int, match_end: int) -> bool:
    """Return True if any negation term appears (as a whole word) within
    NEGATION_WINDOW words on either side of the matched keyword span."""
    words_before = text[:match_start].split()
    words_after = text[match_end:].split()
    window_text = (
        " ".join(words_before[-NEGATION_WINDOW:])
        + " "
        + " ".join(words_after[:NEGATION_WINDOW])
    ).lower()
    return any(pat.search(window_text) for pat in _NEGATION_PATTERNS)


def _find_matches(text: str, terms: list[str], negation_aware: bool = False) -> list[str]:
    """Return terms that appear in *text*.

    Single-word terms use whole-word (word-boundary) matching to avoid
    substring false positives like 'die' inside 'studies', 'na' inside
    'karwana', or 'maut' inside a longer compound.
    Multi-word terms (containing spaces) use substring matching since
    word-boundary regex is harder to apply correctly across scripts.

    If negation_aware=True, skip terms whose only occurrences are inside a
    negation context (e.g. 'guarantee nahi hai' correctly dismisses a
    guarantee rather than making one).
    """
    low = text.lower()
    hits = []
    for t in terms:
        tl = t.lower()
        # Choose match strategy: whole-word for single tokens, substring for phrases
        if " " in tl:
            positions = []
            pos = 0
            while True:
                idx = low.find(tl, pos)
                if idx == -1:
                    break
                positions.append(idx)
                pos = idx + 1
        else:
            # word-boundary regex so 'die' doesn't hit 'studies'
            pat = re.compile(r"(?<!\w)" + re.escape(tl) + r"(?!\w)", re.UNICODE)
            positions = [m.start() for m in pat.finditer(low)]

        if not positions:
            continue

        if not negation_aware:
            hits.append(t)
            continue

        # negation-aware: only flag if at least one occurrence is NOT negated
        any_non_negated = any(
            not _in_negation_context(low, idx, idx + len(tl))
            for idx in positions
        )
        if any_non_negated:
            hits.append(t)
    return hits


def keyword_flags(chat: dict) -> dict:
    msgs = chat.get("messages", [])
    user_text = chat_text(chat, roles=("user",))
    assistant_text = chat_text(chat, roles=("assistant",))

    flags = {}
    for rule in RULES:
        rid = rule["id"]
        if rid == "missing_redirect":
            triggers = _find_matches(user_text, rule["trigger_keywords"])
            redirects = _find_matches(assistant_text, rule["redirect_keywords"])
            matched = bool(triggers) and not redirects
            flags[rid] = {
                "matched": matched,
                "evidence": triggers if matched else [],
                "note": "user raised a serious topic with no matching redirect language in the reply"
                if matched
                else None,
            }
        elif rid in ("guaranteed_outcome", "fear_prediction", "fear_based_remedy_sale"):
            # Use negation-awareness for all three rules that are prone to
            # false positives from safe chats that *quote or condemn* a bad
            # behaviour rather than asserting it.
            hits = _find_matches(assistant_text, rule["keywords"], negation_aware=True)
            flags[rid] = {"matched": bool(hits), "evidence": hits}
        else:
            hits = _find_matches(assistant_text, rule["keywords"], negation_aware=False)
            flags[rid] = {"matched": bool(hits), "evidence": hits}
    return flags


# ---------------------------------------------------------------------------
# Safety rule flagging — optional LLM-judge layer
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a strict content-safety reviewer for Vedaz, an AI Vedic astrology "
    "product. You will be shown one full conversation. Judge ONLY the assistant's "
    "turns against the rules below. Reply with ONLY a JSON object, no commentary."
)


def _judge_prompt(chat: dict) -> str:
    rules_text = "\n".join(f"- {r['id']}: {r['description']}" for r in RULES if r["id"] != "missing_redirect")
    rules_text += "\n- missing_redirect: for serious health/legal/financial/personal-safety questions, the assistant must direct the user to a qualified professional or real resource instead of answering through astrology alone."
    convo = json.dumps(chat.get("messages", []), ensure_ascii=False, indent=2)
    return (
        f"Rules:\n{rules_text}\n\n"
        f"Conversation:\n{convo}\n\n"
        'Reply with JSON exactly like: {"violations": ["rule_id", ...], "reasoning": "1-2 sentences"}. '
        'Use an empty list for "violations" if nothing is broken.'
    )


def llm_judge_flags(chat: dict, provider: Optional[str] = None, model: Optional[str] = None) -> dict:
    if call_model_json is None:
        return {"error": "ai_client not available"}
    parsed, raw = call_model_json(JUDGE_SYSTEM, _judge_prompt(chat), provider=provider, model=model)
    if parsed is None:
        return {"error": "could not parse judge reply", "raw": raw[:300] if raw else None}
    violations = parsed.get("violations", [])
    if not isinstance(violations, list):
        violations = []
    return {"violations": violations, "reasoning": parsed.get("reasoning", "")}


# ---------------------------------------------------------------------------
# Combined per-chat check
# ---------------------------------------------------------------------------

def check_chat(chat: dict, use_llm: bool = False, provider: Optional[str] = None, model: Optional[str] = None) -> dict:
    valid, shape_err = validate_shape(chat)
    kw_flags = keyword_flags(chat) if valid else {}
    kw_triggered = [rid for rid, v in kw_flags.items() if v.get("matched")]

    result = {
        "id": chat.get("id", "(no id)"),
        "valid_shape": valid,
        "shape_error": shape_err,
        "word_count": word_count(chat) if valid else 0,
        "token_estimate": estimate_tokens(chat) if valid else 0,
        "keyword_flags": kw_flags,
        "keyword_triggered": kw_triggered,
        "llm_judge": None,
        "is_flagged": bool(kw_triggered),
    }

    if use_llm and valid:
        judge = llm_judge_flags(chat, provider=provider, model=model)
        result["llm_judge"] = judge
        if judge.get("violations"):
            result["is_flagged"] = True

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(chats: list[dict], use_llm: bool = False, test_fraction: float = 0.2,
                     dup_threshold: float = 0.6, provider: Optional[str] = None,
                     model: Optional[str] = None) -> dict:
    checks = [check_chat(c, use_llm=use_llm, provider=provider, model=model) for c in chats]
    valid_chats = [c for c, chk in zip(chats, checks) if chk["valid_shape"]]
    dup_pairs = find_duplicates(chats, threshold=dup_threshold)
    train, test = train_test_split(chats, test_fraction=test_fraction)

    lengths = [chk["word_count"] for chk in checks if chk["valid_shape"]]
    flagged = [chk for chk in checks if chk["is_flagged"]]

    return {
        "n_total": len(chats),
        "n_valid_shape": len(valid_chats),
        "n_invalid_shape": len(chats) - len(valid_chats),
        "length_stats": {
            "min_words": min(lengths) if lengths else 0,
            "max_words": max(lengths) if lengths else 0,
            "avg_words": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        },
        "duplicate_pairs": dup_pairs,
        "n_flagged": len(flagged),
        "flagged_ids": [chk["id"] for chk in flagged],
        "train_size": len(train),
        "test_size": len(test),
        "checks": checks,
        "_train": train,
        "_test": test,
    }


def print_report(report: dict, chats: list[dict]) -> None:
    print("=" * 70)
    print("VEDAZ CHAT CHECKER REPORT")
    print("=" * 70)
    print(f"Total chats:           {report['n_total']}")
    print(f"Valid shape:           {report['n_valid_shape']}")
    print(f"Invalid shape:         {report['n_invalid_shape']}")
    ls = report["length_stats"]
    print(f"Length (words):        min={ls['min_words']}  avg={ls['avg_words']}  max={ls['max_words']}")
    print(f"Train / test split:    {report['train_size']} / {report['test_size']}")
    print()

    if report["n_invalid_shape"]:
        print("-- Shape problems " + "-" * 50)
        for chk in report["checks"]:
            if not chk["valid_shape"]:
                print(f"  [{chk['id']}] {chk['shape_error']}")
        print()

    print("-- Duplicates / near-duplicates " + "-" * 36)
    if not report["duplicate_pairs"]:
        print("  none found")
    else:
        for dup in report["duplicate_pairs"]:
            a = chats[dup["i"]].get("id", f"#{dup['i']}")
            b = chats[dup["j"]].get("id", f"#{dup['j']}")
            print(f"  {dup['type']:5s}  sim={dup['similarity']:.2f}   {a}  <->  {b}")
    print()

    print("-- Safety flags " + "-" * 53)
    if not report["flagged_ids"]:
        print("  none found")
    else:
        for chk in report["checks"]:
            if not chk["is_flagged"]:
                continue
            print(f"  [{chk['id']}]")
            for rid, v in chk["keyword_flags"].items():
                if v.get("matched"):
                    ev = v.get("evidence") or []
                    print(f"      keyword: {rid}  evidence={ev}")
            if chk["llm_judge"] and chk["llm_judge"].get("violations"):
                print(f"      llm-judge: {chk['llm_judge']['violations']}  -- {chk['llm_judge'].get('reasoning','')}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Check a .jsonl file of Vedaz chats.")
    ap.add_argument("--input", required=True, help="path to .jsonl file of chats")
    ap.add_argument("--use-llm-judge", action="store_true", help="also run the AI-judge safety layer (needs an API key)")
    ap.add_argument("--provider", default=None, help="override AI_PROVIDER for this run")
    ap.add_argument("--model", default=None, help="override the model id for this run")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--dup-threshold", type=float, default=0.6, help="Jaccard similarity threshold for near-duplicates")
    ap.add_argument("--write-splits", action="store_true", help="write <input>_train.jsonl / <input>_test.jsonl")
    ap.add_argument("--write-report", default=None, help="optional path to write the full report as JSON")
    args = ap.parse_args()

    chats = load_jsonl(args.input)
    if not chats:
        print(f"No chats loaded from {args.input}", file=sys.stderr)
        sys.exit(1)

    report = generate_report(
        chats,
        use_llm=args.use_llm_judge,
        test_fraction=args.test_fraction,
        dup_threshold=args.dup_threshold,
        provider=args.provider,
        model=args.model,
    )
    print_report(report, chats)

    if args.write_splits:
        stem = Path(args.input).with_suffix("")
        train_path = f"{stem}_train.jsonl"
        test_path = f"{stem}_test.jsonl"
        for path, subset in ((train_path, report["_train"]), (test_path, report["_test"])):
            with open(path, "w", encoding="utf-8") as f:
                for c in subset:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"\nWrote splits: {train_path} ({len(report['_train'])}), {test_path} ({len(report['_test'])})")

    if args.write_report:
        serializable = {k: v for k, v in report.items() if not k.startswith("_")}
        with open(args.write_report, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"Wrote JSON report: {args.write_report}")


if __name__ == "__main__":
    main()
