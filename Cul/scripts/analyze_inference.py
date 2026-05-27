"""Analyze HF-CAC inference data quality and accuracy."""
import json
import re
from collections import Counter

DATA_PATH = "Cul/data/normad_hf_cac_inference_20260525_101428.jsonl"

def extract_judge_answer(response_text):
    judge_match = re.search(r'===== Solution \d+ \[JUDGE.*?\] =====\n(.*?)$', response_text, re.DOTALL)
    if not judge_match:
        return None
    judge_text = judge_match.group(1)
    m = re.search(r'Answer\s*:\s*([1-4])', judge_text, re.IGNORECASE)
    if m:
        return m.group(1)
    digits = re.findall(r'\b([1-3])\b', judge_text)
    return digits[-1] if digits else None

def extract_guardian_answer(response_text):
    guardian_match = re.search(r'===== Solution \d+ \[GUARDIAN\] =====\n(.*?)(?=\n===== Solution)', response_text, re.DOTALL)
    if not guardian_match:
        return None
    guardian_text = guardian_match.group(1)
    m = re.search(r'Answer\s*:\s*([1-4])', guardian_text, re.IGNORECASE)
    if m:
        return m.group(1)
    digits = re.findall(r'\b([1-3])\b', guardian_text)
    return digits[-1] if digits else None

def main():
    data = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"总样本数: {len(data)}")
    failed_count = sum(1 for d in data if d.get('guardian_failed', False))
    print(f"Guardian 失效样本数: {failed_count} ({failed_count/len(data)*100:.1f}%)")

    judge_correct = 0
    guardian_correct = 0
    judge_total = 0
    guardian_total = 0
    country_stats = {}

    for d in data:
        gt = d['gt'].strip()
        country = d.get('country', 'unknown')
        judge_ans = extract_judge_answer(d['response'])
        guardian_ans = extract_guardian_answer(d['response'])

        if judge_ans:
            judge_total += 1
            if judge_ans == gt:
                judge_correct += 1
        if guardian_ans:
            guardian_total += 1
            if guardian_ans == gt:
                guardian_correct += 1

        if country not in country_stats:
            country_stats[country] = {'total': 0, 'judge_correct': 0, 'guardian_correct': 0}
        country_stats[country]['total'] += 1
        if judge_ans == gt:
            country_stats[country]['judge_correct'] += 1
        if guardian_ans == gt:
            country_stats[country]['guardian_correct'] += 1

    print(f"\n--- 准确率统计 ---")
    print(f"Judge 准确率: {judge_correct}/{judge_total} = {judge_correct/judge_total*100:.2f}%")
    print(f"Guardian 准确率: {guardian_correct}/{guardian_total} = {guardian_correct/guardian_total*100:.2f}%")

    gt_dist = Counter(d['gt'].strip() for d in data)
    print(f"\nGT 标签分布: {dict(gt_dist)}")

    judge_ans_dist = Counter(extract_judge_answer(d['response']) for d in data)
    print(f"Judge 答案分布: {dict(judge_ans_dist)}")

    print(f"\n--- 按国家 Judge 准确率 (从低到高) ---")
    sorted_countries = sorted(country_stats.items(), key=lambda x: x[1]['judge_correct']/max(x[1]['total'],1))
    for country, stats in sorted_countries:
        acc = stats['judge_correct'] / stats['total'] * 100
        print(f"  {country}: {stats['judge_correct']}/{stats['total']} = {acc:.1f}%")

if __name__ == "__main__":
    main()
