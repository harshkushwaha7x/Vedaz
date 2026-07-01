"""Run checker.py's keyword layer against the hand-labeled golden set and
report how many of the deliberately-bad chats got caught, and how many of
the clearly-clean chats got (incorrectly) flagged.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from checker import check_chat

golden_path = Path(__file__).parent / "golden_set.jsonl"
chats = [json.loads(l) for l in open(golden_path, encoding="utf-8")]

tp = fn = tn = fp = 0
print(f"{'id':28s} {'expected':10s} {'predicted':10s} {'rule(s) hit'}")
print("-" * 80)
for c in chats:
    chk = check_chat(c, use_llm=False)
    predicted = "violates" if chk["is_flagged"] else "clean"
    expected = c["label"]
    hit_rules = chk["keyword_triggered"]
    print(f"{c['id']:28s} {expected:10s} {predicted:10s} {hit_rules}")
    if expected == "violates" and predicted == "violates":
        tp += 1
    elif expected == "violates" and predicted == "clean":
        fn += 1
    elif expected == "clean" and predicted == "clean":
        tn += 1
    else:
        fp += 1

print("-" * 80)
print(f"True positives (bad chats caught):     {tp}/{tp+fn}")
print(f"False negatives (bad chats missed):    {fn}/{tp+fn}")
print(f"True negatives (clean chats passed):   {tn}/{tn+fp}")
print(f"False positives (clean chats flagged): {fp}/{tn+fp}")
