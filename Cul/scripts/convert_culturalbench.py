"""
CulturalBench-Easy 数据集格式转换脚本
将 CSV 格式转换为指定的 JSON 格式
- 对每个问题随机打乱四个选项顺序，消除位置偏差
- 正确映射答案标签到打乱后的位置
- 统计打乱前后答案选项的分布
"""

import csv
import json
import os
import random
from collections import Counter

INSTRUCTION = "Please answer the following cultural knowledge question by selecting the correct option number."

# 设置随机种子以确保可复现
random.seed(42)


def convert():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    input_path = os.path.join(base_dir, "data", "CulturalBench-Easy.csv")
    output_path = os.path.join(base_dir, "data", "culturalBench_mas.json")

    results = []
    original_answer_dist = Counter()
    shuffled_answer_dist = Counter()

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row["prompt_question"].strip()
            options = [
                row["prompt_option_a"].strip(),
                row["prompt_option_b"].strip(),
                row["prompt_option_c"].strip(),
                row["prompt_option_d"].strip(),
            ]
            answer_letter = row["answer"].strip()  # A, B, C, D
            country = row["country"].strip()

            # 原始答案位置 (0-indexed)
            original_answer_idx = ord(answer_letter) - ord("A")
            correct_option_text = options[original_answer_idx]

            # 统计原始分布
            original_answer_dist[answer_letter] += 1

            # 随机打乱选项
            shuffled_options = options.copy()
            random.shuffle(shuffled_options)

            # 找到正确答案在打乱后的位置
            new_answer_idx = shuffled_options.index(correct_option_text)
            new_answer_number = str(new_answer_idx + 1)  # 1-indexed

            # 统计打乱后的分布
            shuffled_answer_dist[new_answer_number] += 1

            # 构建 input 文本
            input_text = f"{question}\n1. {shuffled_options[0]}\n2. {shuffled_options[1]}\n3. {shuffled_options[2]}\n4. {shuffled_options[3]}"

            results.append({
                "instruction": INSTRUCTION,
                "input": input_text,
                "output": new_answer_number,
                "country": country
            })

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印统计信息
    total = len(results)
    print(f"转换完成！共 {total} 条数据，保存至: {output_path}")
    print(f"\n{'='*50}")
    print("原始数据集答案分布 (打乱前):")
    print(f"{'='*50}")
    for letter in ["A", "B", "C", "D"]:
        count = original_answer_dist[letter]
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  选项 {letter}: {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\n{'='*50}")
    print("打乱后答案分布:")
    print(f"{'='*50}")
    for num in ["1", "2", "3", "4"]:
        count = shuffled_answer_dist[num]
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  选项 {num}: {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\n{'='*50}")
    print("验证: 原始总数 =", sum(original_answer_dist.values()),
          "| 打乱后总数 =", sum(shuffled_answer_dist.values()))


if __name__ == "__main__":
    convert()
