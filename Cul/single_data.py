"""
Single-model (base) evaluation on cultural alignment benchmarks.

使用单个基座模型（不经过任何多智能体协作）在三个文化数据集上评估：
  - CulturalBench (4-way 多选)   /autodl-fs/data/culturalBench_mas.json
  - NormAD        (3-way 可接受性) /autodl-fs/data/normad_mas.json
  - BLEnD         (4-way 多选)   /autodl-fs/data/blend_mas_after.json

两种方法（--method）：
  - base : zero-shot，模型仅根据数据集中的问题作答（不设 system prompt 的角色信息）。
  - role : 角色扮演，设定 "你是某国文化专家" 的 system prompt，依据该文化背景作答。

输出：
  - 结果文件   (--output_file)：JSONL，每行保存 query / country / gt / pred / response。
  - 指标文件   (output_file 同名 + .metrics.json)：准确率、各国别准确率、答案分布等。

Usage:
    # CulturalBench, Qwen, zero-shot
    python Cul/single_data.py \\
        --input_file /autodl-fs/data/culturalBench_mas.json \\
        --output_file /autodl-fs/data/culturalBench_qwen_base.json \\
        --model_name qwen \\
        --method base \\
        --tensor_parallel_size 2 --max_samples 0

    # NormAD, Llama, role-play
    python Cul/single_data.py \\
        --input_file /autodl-fs/data/normad_mas.json \\
        --output_file /autodl-fs/data/normad_llama_role.json \\
        --model_name llama \\
        --method role \\
        --tensor_parallel_size 2 --max_samples 0
"""

import os
import re
import sys
import json
import argparse
import threading
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import write_to_jsonl, reserve_unprocessed_queries


# ---------------------------------------------------------------------------
# Model aliases (与 evaluate.py / generate_hf_cac_data.py 保持一致)
# ---------------------------------------------------------------------------
MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}


# ---------------------------------------------------------------------------
# Data helpers (与 generate_hf_cac_data.py 复用同一套逻辑)
# ---------------------------------------------------------------------------

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())[0]
    return data


def detect_dataset_type(data: list) -> str:
    """自动判断数据集类型：culturalbench / normad / cultureatlas。

    CulturalBench 与 BLEnD 均为 4 选 1，统一按 culturalbench 处理（4-way）。
    """
    if not data:
        return "normad"

    sample = data[:min(10, len(data))]
    for item in sample:
        instr = item.get("instruction", "").lower()
        if "cultural knowledge question" in instr or "correct option number" in instr:
            return "culturalbench"
        if "more culturally specific" in instr:
            return "cultureatlas"
        if "acceptable" in instr or "unacceptable" in instr:
            return "normad"

    for item in sample:
        inp = item.get("input", "")
        if "\n1. " in inp and "\n2. " in inp and "\n3. " in inp and "\n4. " in inp:
            return "culturalbench"

    outputs = set(str(item.get("output", "")).strip()
                  for item in data[:min(100, len(data))])
    if "4" in outputs:
        return "culturalbench"
    if outputs and outputs <= {"1", "2"}:
        return "cultureatlas"
    return "normad"


def convert_sample(item):
    """将原始样本转换为 {query, gt, country} 内部格式。"""
    if "input" in item and item["input"] and "country" in item:
        query = item["input"].strip()
        country = item["country"].strip()
        gt = str(item.get("output", "")).strip()
        return {"query": query, "gt": gt, "country": country}

    instruction = item["instruction"]
    query = instruction.split("### Answer:")[0].strip()
    if "Country" in item and item["Country"]:
        country = item["Country"].strip()
    else:
        m = re.search(r"country or language that is (.+?)\.", instruction)
        country = m.group(1).strip() if m else ""
    gt = str(item.get("output", item.get("label", ""))).strip()
    return {"query": query, "gt": gt, "country": country}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def max_choice_of(task_type: str) -> int:
    if task_type == "cultureatlas":
        return 2
    if task_type == "culturalbench":
        return 4
    return 3


def build_system_prompt(method: str, task_type: str, country: str) -> str:
    """构造 system prompt。

    base : 通用助手，不注入文化角色信息（zero-shot）。
    role : 角色扮演——设定为目标国家/文化专家，依据该文化背景作答。
    """
    if method == "base":
        return (
            "You are a helpful assistant. Answer the question accurately and "
            "end your response with the chosen option number in the format "
            "'Answer: X'."
        )

    # role-play：依据数据集内容设计不同提示词，但整体逻辑一致
    target = country if country else "the target country"
    if task_type == "normad":
        return (
            f"You are a cultural expert specializing in the culture of {target}. "
            f"You have deep knowledge of {target}'s social norms, traditions, and "
            f"daily customs. Based on this cultural background, judge whether the "
            f"described behavior is socially acceptable in {target}. "
            f"End your response with 'Answer: X'."
        )
    if task_type == "cultureatlas":
        return (
            f"You are a cultural expert specializing in the culture of {target}. "
            f"Drawing on your deep, culture-specific knowledge of {target}, decide "
            f"which response demonstrates more culturally specific and insightful "
            f"knowledge. End your response with 'Answer: X'."
        )
    # culturalbench / blend
    return (
        f"You are a cultural expert specializing in the culture of {target}. "
        f"You possess authoritative knowledge of {target}'s cultural practices, "
        f"customs, and everyday life. Based on this cultural background, select the "
        f"correct option for the question. End your response with 'Answer: X'."
    )


def build_user_prompt(query: str, country: str, task_type: str) -> str:
    """构造 user prompt（题目本身 + 作答格式提示）。"""
    mc = max_choice_of(task_type)
    choices_hint = "/".join(str(i) for i in range(1, mc + 1))
    header = f"[{country}]\n" if country else ""
    return (
        f"{header}{query}\n\n"
        f"Respond with the correct option number ({choices_hint}) and finish with "
        f"'Answer: X'."
    )


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str, max_choice: int):
    """从模型回答中抽取答案数字。"""
    pattern = f"[1-{max_choice}]"
    m = re.search(rf"(?:Final\s+decision|Answer)\s*(?:is|[:\-])\s*({pattern})\b",
                  text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(rf"\b({pattern})\s*\.?\s*$", text.strip())
    if m:
        return m.group(1)
    m = re.search(rf"option\s*:?\s*({pattern})\b", text, re.IGNORECASE)
    if m:
        return m.group(1)
    digits = re.findall(rf"\b({pattern})\b", text)
    return digits[-1] if digits else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(output_file: str, task_type: str, model_name: str,
                    method: str) -> dict:
    """从结果 JSONL 计算准确率与各国别指标。"""
    max_choice = max_choice_of(task_type)
    data = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    correct = 0
    total = 0
    country_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for d in data:
        gt = str(d.get("gt", "")).strip()
        if not gt:
            continue
        pred = d.get("pred")
        country = d.get("country", "unknown")
        total += 1
        country_stats[country]["total"] += 1
        if pred == gt:
            correct += 1
            country_stats[country]["correct"] += 1

    per_country = {}
    for c in sorted(country_stats.keys()):
        st = country_stats[c]
        per_country[c] = {
            "accuracy": st["correct"] / st["total"] if st["total"] else 0.0,
            "correct": st["correct"],
            "total": st["total"],
        }

    gt_dist = dict(Counter(str(d.get("gt", "")).strip() for d in data if d.get("gt")))
    pred_dist = dict(Counter(d.get("pred") for d in data))

    return {
        "model_name": model_name,
        "method": method,
        "task_type": task_type,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "gt_distribution": gt_dist,
        "pred_distribution": pred_dist,
        "per_country": per_country,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-model (base) evaluation on cultural benchmarks"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="数据集 JSON 路径（culturalBench_mas.json / "
                             "normad_mas.json / blend_mas_after.json）")
    parser.add_argument("--output_file", type=str, required=True,
                        help="结果输出 JSONL 路径（会自动在文件名追加时间戳）")
    parser.add_argument("--model_name", type=str, required=True,
                        help="基座别名 qwen / llama，或完整本地路径")
    parser.add_argument("--method", type=str, required=True,
                        choices=["base", "role"],
                        help="base=zero-shot；role=角色扮演（设定文化专家 system prompt）")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度，默认 0.0（贪心，结果可复现）")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="vLLM 批处理大小（仅影响写盘节奏）")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="处理样本数，0 表示全部")
    args = parser.parse_args()

    # 模型别名解析
    model_path = MODEL_ALIASES.get(args.model_name.lower(), args.model_name)
    print(f"Model: {model_path}")
    print(f"Method: {args.method}")

    # 输出文件名追加时间戳，与现有脚本风格一致
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = Path(args.output_file)
    output_file = str(p.parent / f"{p.stem}_{timestamp}{p.suffix or '.json'}")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    print(f"Output file: {output_file}")

    # 加载并转换数据
    raw_data = load_dataset(args.input_file)
    dataset = [convert_sample(item) for item in raw_data]
    task_type = detect_dataset_type(raw_data)
    max_choice = max_choice_of(task_type)
    print(f"Loaded {len(dataset)} samples | task_type={task_type} "
          f"(max_choice={max_choice})")

    if args.max_samples > 0:
        dataset = dataset[: args.max_samples]
        print(f"Using first {args.max_samples} samples")

    # 断点续跑
    dataset = reserve_unprocessed_queries(output_file, dataset)
    print(f"After resume filter: {len(dataset)} samples remaining")
    if len(dataset) == 0:
        print("All samples already processed.")
        return

    # 初始化 vLLM
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>", "</s>"],
    )

    def build_prompt(sample):
        system = build_system_prompt(args.method, task_type, sample["country"])
        user = build_user_prompt(sample["query"], sample["country"], task_type)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # 批量推理
    lock = threading.Lock()
    for start in tqdm(range(0, len(dataset), args.batch_size), desc="Batches"):
        batch = dataset[start: start + args.batch_size]
        prompts = [build_prompt(s) for s in batch]
        outputs = llm.generate(prompts, sampling)
        for sample, out in zip(batch, outputs):
            response = out.outputs[0].text.strip()
            pred = extract_answer(response, max_choice)
            record = {
                "query": sample["query"],
                "country": sample["country"],
                "gt": sample["gt"],
                "pred": pred,
                "response": response,
                "model_name": args.model_name,
                "method": args.method,
                "task_type": task_type,
            }
            write_to_jsonl(lock, output_file, record)

    print(f"\nDone. Results saved to: {output_file}")

    # 计算并保存指标
    metrics = compute_metrics(output_file, task_type, args.model_name, args.method)
    metrics_file = str(Path(output_file).with_suffix(".metrics.json"))
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n--- Metrics ---")
    print(f"Task type: {metrics['task_type']} | Method: {metrics['method']}")
    print(f"Accuracy: {metrics['accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['total']})")
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
