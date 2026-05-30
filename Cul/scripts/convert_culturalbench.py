"""
CulturalBench-Easy 数据集格式转换脚本
将 CSV 格式转换为指定的 JSON 格式
"""

import csv
import json
import os

ANSWER_MAP = {"A": "1", "B": "2", "C": "3", "D": "4"}

INSTRUCTION = "Please answer the following cultural knowledge question by selecting the correct option number."

def convert():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    input_path = os.path.join(base_dir, "data", "CulturalBench-Easy.csv")
    output_path = os.path.join(base_dir, "data", "culturalBench_mas.json")

    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row["prompt_question"].strip()
            option_a = row["prompt_option_a"].strip()
            option_b = row["prompt_option_b"].strip()
            option_c = row["prompt_option_c"].strip()
            option_d = row["prompt_option_d"].strip()
            answer = row["answer"].strip()
            country = row["country"].strip()

            input_text = f"{question}\n1. {option_a}\n2. {option_b}\n3. {option_c}\n4. {option_d}"
            output_text = ANSWER_MAP[answer]

            results.append({
                "instruction": INSTRUCTION,
                "input": input_text,
                "output": output_text,
                "country": country
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"转换完成！共 {len(results)} 条数据，保存至: {output_path}")

if __name__ == "__main__":
    convert()
