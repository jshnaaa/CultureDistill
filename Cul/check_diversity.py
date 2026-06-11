#!/usr/bin/env python3
"""Confirm agent homogeneity: measure answer diversity per question on the
6-agent file, and quantify the headroom an ideal aggregator could capture.
"""
import json
import math
import re
import sys
from collections import Counter

MAX_CHOICE = 4
PATTERN = f"[1-{MAX_CHOICE}]"


def extract_answer(text: str) -> str | None:
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
    parts = re.split(r'=====\s*Solution\s+\d+\s*\[([^\]]+)\]\s*=====', response)
    out = []
    for i in range(1, len(parts), 2):
        role = parts[i].strip()
        text = parts[i + 1] if i + 1 < len(parts) else ""
        out.append((role, text))
    return out


def entropy(counter, total):
    h = 0.0
    for c in counter.values():
        p = c / total
        h -= p * math.log2(p)
    return h


def analyze(path, label):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    n = len(recs)
    unanimous = 0          # all agents same answer
    distinct_hist = Counter()  # number of distinct answers per question
    # headroom analysis
    oracle_correct = 0     # at least one agent correct
    majority_correct = 0   # plain majority correct
    minority_correct = 0   # majority wrong BUT some agent correct (recoverable!)
    avg_entropy = 0.0

    for r in recs:
        gt = str(r["gt"]).strip()
        sols = split_solutions(r["response"])
        agent_sols = [(role, txt) for role, txt in sols if "JUDGE" not in role.upper()]
        answers = [extract_answer(txt) for _, txt in agent_sols]
        valid = [a for a in answers if a is not None]
        if not valid:
            distinct_hist[0] += 1
            continue

        c = Counter(valid)
        distinct_hist[len(c)] += 1
        if len(c) == 1:
            unanimous += 1
        avg_entropy += entropy(c, len(valid))

        winner = c.most_common(1)[0][0]
        any_correct = any(a == gt for a in valid)
        if any_correct:
            oracle_correct += 1
        if winner == gt:
            majority_correct += 1
        elif any_correct:
            minority_correct += 1

    print(f"\n========== {label} ==========")
    print(f"Records: {n}")
    print(f"Unanimous (all agents identical): {unanimous} ({unanimous/n*100:.1f}%)")
    print(f"Distinct-answer-count histogram (per question): {dict(sorted(distinct_hist.items()))}")
    print(f"Average answer entropy (bits): {avg_entropy/n:.3f}  (0 = total agreement, 2 = max for 4 options)")
    print(f"--- Headroom analysis ---")
    print(f"Majority correct:  {majority_correct}/{n} = {majority_correct/n*100:.2f}%")
    print(f"Oracle (any agent correct): {oracle_correct}/{n} = {oracle_correct/n*100:.2f}%  <-- CEILING for any aggregator")
    print(f"Minority-correct (majority wrong but recoverable): {minority_correct} ({minority_correct/n*100:.1f}%)")
    print(f"Hard-wrong (NO agent correct, unrecoverable): {n-oracle_correct} ({(n-oracle_correct)/n*100:.1f}%)")


if __name__ == "__main__":
    f5 = sys.argv[1]
    f6 = sys.argv[2]
    analyze(f5, "5-AGENT")
    analyze(f6, "6-AGENT")
