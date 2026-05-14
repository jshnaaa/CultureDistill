"""Detailed analysis: which samples went wrong in 2debate vs nodebate."""
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


def get_sample_answers(sample):
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
    return agent_answers, judge_answer


files = {
    'nodebate': 'Cul/data/normad_mas_inference_nodebate_20260514_102211.jsonl',
    '1debate': 'Cul/data/normad_mas_inference_1debate_20260514_101819.jsonl',
    '2debate': 'Cul/data/normad_mas_inference_2debate_20260514_102337.jsonl',
}

data = {}
for name, filepath in files.items():
    with open(filepath, 'r') as f:
        data[name] = [json.loads(line) for line in f]

print("="*80)
print("DETAILED PER-SAMPLE COMPARISON")
print("="*80)

for i in range(len(data['nodebate'])):
    gt = data['nodebate'][i]['gt']
    print(f"\n--- Sample {i+1} (GT={gt}) ---")

    for name in ['nodebate', '1debate', '2debate']:
        agent_ans, judge_ans = get_sample_answers(data[name][i])
        agent_correct = sum(1 for a in agent_ans if a == gt)
        status = "✓" if judge_ans == gt else "✗"
        print(f"  {name:10s}: agents={agent_ans} ({agent_correct}/5 correct)  judge={judge_ans} {status}")

print("\n\n" + "="*80)
print("CONVERGENCE ANALYSIS: Do agents converge to WRONG answers after debate?")
print("="*80)

for i in range(len(data['nodebate'])):
    gt = data['nodebate'][i]['gt']
    nd_agents, nd_judge = get_sample_answers(data['nodebate'][i])
    d1_agents, d1_judge = get_sample_answers(data['1debate'][i])
    d2_agents, d2_judge = get_sample_answers(data['2debate'][i])

    # Check if debate made things worse
    nd_agent_correct = sum(1 for a in nd_agents if a == gt)
    d2_agent_correct = sum(1 for a in d2_agents if a == gt)

    if d2_agent_correct < nd_agent_correct:
        print(f"\n  Sample {i+1}: Debate DEGRADED agent accuracy")
        print(f"    GT: {gt}")
        print(f"    nodebate agents: {nd_agents} ({nd_agent_correct}/5)")
        print(f"    1debate  agents: {d1_agents} ({sum(1 for a in d1_agents if a == gt)}/5)")
        print(f"    2debate  agents: {d2_agents} ({d2_agent_correct}/5)")
        print(f"    → Agent answers converged toward WRONG answer after debate rounds")

print("\n\n" + "="*80)
print("REASONING LENGTH ANALYSIS")
print("="*80)

for name in ['nodebate', '1debate', '2debate']:
    lengths = []
    for sample in data[name]:
        response = sample['response']
        solutions = response.split('===== Solution ')
        for sol in solutions[1:]:
            sol_num = sol.split(' =====')[0]
            if sol_num != '6':
                sol_text = sol.split('=====\n', 1)[1] if '=====\n' in sol else sol
                lengths.append(len(sol_text))
    avg_len = sum(lengths) / len(lengths)
    print(f"  {name:10s}: avg agent response length = {avg_len:.0f} chars")
