#!/usr/bin/env python3
"""Diagnose 5-agent vs 6-agent voting behavior on CulturalBench.

Parses the saved HF-CAC jsonl outputs, re-extracts each agent's answer using
the SAME extraction rules as hf_cac_mas.py, then computes:
  - per-file accuracy (majority vote)
  - vote distribution (tie rate, margin)
  - how often the 6th agent flips a 5-agent-correct case to wrong (and vice versa)
"""
import json
import re
import sys
from collections import Counter

MAX_CHOICE = 4
PATTERN = f"[1-{MAX_CHOICE}]"


def extract_answer(text: str) -> str | None:
    """Mirror of hf_cac_mas._extract_answer for task_type='culturalbench'."""
    tl = text.strip()
    m = re.search(rf'(?:Final\s+decision|Answer)\s*(?:is|[:\-])\s*({PATTERN})\b', tl, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(rf'\b({PATTERN})\s*\.?\s*$', tl)
    if m:
        return m.group(1)
    first_line = tl.split("\n")[0].strip().rstrip(".")
    if re.match(rf'^({PATTERN})$', first_line):
        return first_line
    m = re.search(rf'option\s*({PATTERN})\b', tl, re.IGNORECASE)
    if m:
        return m.group(1)
    matches = re.findall(rf'\b({PATTERN})\b', tl)
    return matches[-1] if matches else None


def split_solutions(response: str):
    """Split the response into (role, text) per Solution block.

    Excludes JUDGE blocks (they are post-hoc consensus, not an agent vote).
    """
    # Blocks look like: ===== Solution N [ROLE] =====\n<text>
    parts = re.split(r'=====\s*Solution\s+\d+\s*\[([^\]]+)\]\s*=====', response)
    # parts: ['', ROLE1, TEXT1, ROLE2, TEXT2, ...]
    out = []
    for i in range(1, len(parts), 2):
        role = parts[i].strip()
        text = parts[i + 1] if i + 1 < len(parts) else ""
        out.append((role, text))
    return out


def majority_vote(answers):
    """Return (winner, counts_dict, is_tie). winner=None if no valid answers."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, {}, False
    c = Counter(valid)
    top = c.most_common()
    winner = top[0][0]
    is_tie = len(top) > 1 and top[1][1] == top[0][1]
    return winner, dict(c), is_tie


def analyze(path, label):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n = len(records)
    correct = 0
    tie_count = 0
    tie_correct = 0
    all_wrong = 0          # every agent wrong (knowledge ceiling)
    margins = Counter()    # winning_margin -> count
    agent_answers_all = []  # list of (gt, [per-agent answers excluding judge])

    for r in records:
        gt = str(r["gt"]).strip()
        sols = split_solutions(r["response"])
        agent_sols = [(role, txt) for role, txt in sols if "JUDGE" not in role.upper()]
        answers = [extract_answer(txt) for _, txt in agent_sols]
        agent_answers_all.append((gt, answers, [role for role, _ in agent_sols]))

        winner, counts, is_tie = majority_vote(answers)
        valid = [a for a in answers if a is not None]

        if is_tie:
            tie_count += 1

        # majority-vote correctness (tie -> count as the file's own first winner;
        # we measure raw majority accuracy here)
        if winner == gt:
            correct += 1
            if is_tie:
                tie_correct += 1

        if valid and all(a != gt for a in valid):
            all_wrong += 1

        # margin = top count - second count
        if counts:
            sorted_counts = sorted(counts.values(), reverse=True)
            top1 = sorted_counts[0]
            top2 = sorted_counts[1] if len(sorted_counts) > 1 else 0
            margins[top1 - top2] += 1

    print(f"\n========== {label} ({path.split('/')[-1]}) ==========")
    print(f"Records: {n}")
    print(f"Majority-vote accuracy: {correct}/{n} = {correct/n*100:.2f}%")
    print(f"Tie cases: {tie_count} ({tie_count/n*100:.1f}%), of which correct: {tie_correct}")
    print(f"All-agents-wrong (knowledge ceiling): {all_wrong} ({all_wrong/n*100:.1f}%)")
    print(f"Win-margin distribution (top1 - top2): {dict(sorted(margins.items()))}")
    return agent_answers_all


def compare(a5, a6):
    """Compare per-sample: how does the 6th agent change outcomes?

    Assumes records are aligned by index (same dataset order).
    """
    print("\n========== 5-agent vs 6-agent per-sample comparison ==========")
    n = min(len(a5), len(a6))
    flip_5correct_to_6wrong = 0
    flip_5wrong_to_6correct = 0
    sixth_decisive = 0     # 6th agent's answer changed the majority winner
    both_correct = both_wrong = 0

    for i in range(n):
        gt5, ans5, _ = a5[i]
        gt6, ans6, roles6 = a6[i]
        if gt5 != gt6:
            continue  # misaligned, skip

        w5, _, tie5 = majority_vote(ans5)
        w6, _, tie6 = majority_vote(ans6)
        c5 = (w5 == gt5)
        c6 = (w6 == gt6)

        if c5 and not c6:
            flip_5correct_to_6wrong += 1
        elif not c5 and c6:
            flip_5wrong_to_6correct += 1
        if c5 and c6:
            both_correct += 1
        if not c5 and not c6:
            both_wrong += 1

        # was the 6th agent decisive? (winner changes if we drop the 6th)
        if len(ans6) == 6:
            w6_without = majority_vote(ans6[:5])[0]
            if w6_without != w6:
                sixth_decisive += 1

    print(f"Aligned samples: {n}")
    print(f"5-correct -> 6-wrong (6th hurt): {flip_5correct_to_6wrong}")
    print(f"5-wrong   -> 6-correct (6th helped): {flip_5wrong_to_6correct}")
    print(f"Net effect of going 5->6: {flip_5wrong_to_6correct - flip_5correct_to_6wrong:+d}")
    print(f"6th agent was decisive (changed winner vs first-5): {sixth_decisive}")
    print(f"Both correct: {both_correct}, Both wrong: {both_wrong}")


if __name__ == "__main__":
    f5 = sys.argv[1]
    f6 = sys.argv[2]
    a5 = analyze(f5, "5-AGENT")
    a6 = analyze(f6, "6-AGENT")
    compare(a5, a6)
