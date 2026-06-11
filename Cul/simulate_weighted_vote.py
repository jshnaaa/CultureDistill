#!/usr/bin/env python3
"""Plan C: offline simulation of affinity-weighted / drop-worst voting on the
saved 6-agent CulturalBench output. No GPU, no re-run.

Goal: can a smarter aggregation over the SAME 6 agents beat the 5-agent
plain-majority baseline (70.17%)?

Strategies tested (all on the 6-agent file):
  1. plain      : equal-weight majority (baseline reproduction)
  2. affinity   : weight each agent by affinity(guardian_idx, agent_idx)
  3. drop_worst : drop the single least-affine agent, then majority
  4. home_boost : home-field (guardian) agent gets extra weight (1.5/2.0)
  5. drop2_worst: drop the 2 least-affine agents -> mimic a 4-agent ensemble
"""
import json
import re
import sys
from collections import defaultdict

MAX_CHOICE = 4
PATTERN = f"[1-{MAX_CHOICE}]"

# Affinity matrix from hf_cac_config_culturalbench.yaml (row = guardian/home culture)
AFFINITY = [
    [1.0, 0.4, 0.1, 0.2, 0.2, 0.1],
    [0.4, 1.0, 0.3, 0.1, 0.2, 0.2],
    [0.1, 0.3, 1.0, 0.1, 0.5, 0.2],
    [0.2, 0.1, 0.1, 1.0, 0.2, 0.5],
    [0.2, 0.2, 0.5, 0.2, 1.0, 0.4],
    [0.1, 0.2, 0.2, 0.5, 0.4, 1.0],
]


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


# role-name -> agent index (order matches config culture_roles)
ROLE_ORDER = [
    "Western & Anglo-Saxon Culture",
    "Latin American Culture",
    "Sub-Saharan African Culture",
    "East-Asian Culture",
    "Islamic & Middle-Eastern Culture",
    "South & Southeast Asian Culture",
]
ROLE_TO_IDX = {name: i for i, name in enumerate(ROLE_ORDER)}


def weighted_winner(agent_answers, weights):
    """agent_answers: list of (agent_idx, answer). weights: list indexed by agent_idx.
    Returns winning answer (ties broken by higher total weight, then by lower digit)."""
    score = defaultdict(float)
    for idx, ans in agent_answers:
        if ans is None:
            continue
        score[ans] += weights[idx]
    if not score:
        return None
    best = max(score.items(), key=lambda kv: (kv[1], -int(kv[0])))
    return best[0]


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def build_agent_answers(rec):
    """Return list of (agent_idx, answer) for non-judge solutions, using the
    GUARDIAN-first ordering present in the file. We reconstruct agent_idx from
    role name; the guardian's role appears first."""
    sols = split_solutions(rec["response"])
    agent_sols = [(role, txt) for role, txt in sols if "JUDGE" not in role.upper()]
    guardian_idx = rec["guardian_idx"]

    # The file lists Guardian first, then auditors. Auditor role labels are
    # generic "AUDITOR" not culture names, so we cannot always map each auditor
    # to a specific culture. Strategy: the FIRST solution is the guardian
    # (home-field, idx=guardian_idx). Remaining auditors are the OTHER cultures
    # in config order (excluding guardian).
    other_indices = [i for i in range(6) if i != guardian_idx]
    result = []
    for pos, (role, txt) in enumerate(agent_sols):
        ans = extract_answer(txt)
        if pos == 0:
            result.append((guardian_idx, ans))
        else:
            # map the (pos-1)-th auditor to the (pos-1)-th non-guardian culture
            ai = other_indices[pos - 1] if (pos - 1) < len(other_indices) else other_indices[-1]
            result.append((ai, ans))
    return result, guardian_idx


def evaluate(recs, weight_fn, label):
    correct = 0
    n = len(recs)
    for rec in recs:
        gt = str(rec["gt"]).strip()
        agent_answers, gidx = build_agent_answers(rec)
        weights = weight_fn(gidx)
        w = weighted_winner(agent_answers, weights)
        if w == gt:
            correct += 1
    print(f"  {label:28s}: {correct}/{n} = {correct/n*100:.2f}%")
    return correct / n


def main():
    f6 = sys.argv[1]
    recs = load(f6)
    print(f"6-agent file: {len(recs)} records\n")
    print("Strategy accuracies (all on the SAME 6-agent outputs):")

    # 1. plain equal weight
    evaluate(recs, lambda g: [1.0] * 6, "1. plain (equal weight)")

    # 2. affinity weighting
    evaluate(recs, lambda g: AFFINITY[g], "2. affinity-weighted")

    # 3. drop single worst (least affine) agent -> weight 0
    def drop_worst(g):
        w = list(AFFINITY[g])
        worst = min(range(6), key=lambda i: w[i])
        w2 = [1.0] * 6
        w2[worst] = 0.0
        return w2
    evaluate(recs, drop_worst, "3. drop-1-worst (equal rest)")

    # 4. home-field boost
    for boost in (1.5, 2.0, 3.0):
        def hb(g, b=boost):
            w = [1.0] * 6
            w[g] = b
            return w
        evaluate(recs, hb, f"4. home-boost x{boost}")

    # 5. drop 2 worst -> 4-agent ensemble
    def drop2(g):
        w = list(AFFINITY[g])
        order = sorted(range(6), key=lambda i: w[i])
        w2 = [1.0] * 6
        w2[order[0]] = 0.0
        w2[order[1]] = 0.0
        return w2
    evaluate(recs, drop2, "5. drop-2-worst (4-agent)")

    # 6. affinity + home boost combo
    def aff_home(g):
        w = list(AFFINITY[g])
        w[g] *= 2.0
        return w
    evaluate(recs, aff_home, "6. affinity + home x2")


if __name__ == "__main__":
    main()
