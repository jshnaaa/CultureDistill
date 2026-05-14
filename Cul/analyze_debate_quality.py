"""Analyze quality differences between nodebate, 1debate, 2debate data files."""
import json
import re
from collections import Counter


def extract_answer(text):
    m = re.search(r'Answer\s*:\s*([1-4])', text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'answer\s+is\s*:?\s*([1-4])\b', text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'option\s*:?\s*([1-4])\b', text, re.IGNORECASE)
    if m:
        return m.group(1)
    digits = re.findall(r'\b([1-4])\b', text)
    return digits[-1] if digits else None


def analyze_file(name, filepath):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    with open(filepath, 'r') as f:
        samples = [json.loads(line) for line in f]

    print(f"Total samples: {len(samples)}")

    agent_correct = 0
    agent_total = 0
    judge_correct = 0
    judge_total = 0
    unique_answers_list = []
    full_agreement = 0

    for sample in samples:
        gt = sample['gt']
        response = sample['response']
        solutions = response.split('===== Solution ')

        agent_answers = []
        judge_answer = None

        for sol in solutions[1:]:
            sol_num = sol.split(' =====')[0]
            sol_text = sol.split('=====\n', 1)[1] if '=====\n' in sol else sol
            ans = extract_answer(sol_text)

            if sol_num == '6':
                judge_answer = ans
            else:
                agent_answers.append(ans)

        # Agent accuracy
        for ans in agent_answers:
            agent_total += 1
            if ans == gt:
                agent_correct += 1

        # Judge accuracy
        judge_total += 1
        if judge_answer == gt:
            judge_correct += 1

        # Diversity
        valid_answers = [a for a in agent_answers if a is not None]
        unique = len(set(valid_answers))
        unique_answers_list.append(unique)

        if unique == 1:
            full_agreement += 1

    print(f"\nAgent accuracy: {agent_correct}/{agent_total} = {agent_correct/agent_total*100:.1f}%")
    print(f"Judge accuracy: {judge_correct}/{judge_total} = {judge_correct/judge_total*100:.1f}%")
    avg_unique = sum(unique_answers_list) / len(unique_answers_list)
    print(f"Average unique answers per sample: {avg_unique:.2f}")
    print(f"Full agreement (all agents same answer): {full_agreement}/{len(samples)} = {full_agreement/len(samples)*100:.1f}%")

    print("\nDiversity distribution:")
    dist = Counter(unique_answers_list)
    for k in sorted(dist.keys()):
        print(f"  unique={k}: {dist[k]} samples ({dist[k]/len(samples)*100:.1f}%)")

    return {
        'name': name,
        'agent_acc': agent_correct / agent_total * 100,
        'judge_acc': judge_correct / judge_total * 100,
        'avg_unique': avg_unique,
        'full_agreement': full_agreement / len(samples) * 100,
    }


if __name__ == '__main__':
    files = {
        'nodebate (0 rounds)': 'Cul/data/normad_mas_inference_nodebate_20260514_102211.jsonl',
        '1debate (1 round)': 'Cul/data/normad_mas_inference_1debate_20260514_101819.jsonl',
        '2debate (2 rounds)': 'Cul/data/normad_mas_inference_2debate_20260514_102337.jsonl',
    }

    results = []
    for name, filepath in files.items():
        r = analyze_file(name, filepath)
        results.append(r)

    print("\n\n" + "="*60)
    print("  COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Config':<20} {'Agent Acc':<12} {'Judge Acc':<12} {'Avg Unique':<12} {'Full Agree':<12}")
    print("-"*68)
    for r in results:
        print(f"{r['name']:<20} {r['agent_acc']:.1f}%{'':<6} {r['judge_acc']:.1f}%{'':<6} {r['avg_unique']:.2f}{'':<8} {r['full_agreement']:.1f}%")
