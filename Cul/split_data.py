"""
CAMA-D 数据划分脚本：将 HFA-C²N 推理数据按 8:1:1 划分为训练集/验证集/测试集

输出一个 pkl 文件，包含三个 key:
  - "train": list[dict]  (80%)
  - "val":   list[dict]  (10%)
  - "test":  list[dict]  (10%)

每条数据保留原始 JSONL 中的所有字段（query, country, gt, response 等）。

Usage:
    python Cul/split_data.py \
        --input /autodl-fs/data/qwen/normad_hfa_c2n_inference.jsonl \
        --output /autodl-fs/data/qwen/normad_splits.pkl \
        --seed 42
"""

import json
import pickle
import random
import argparse
from pathlib import Path


def split_data(input_path: str, output_path: str, seed: int = 42,
               train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    按 train_ratio : val_ratio : (1 - train_ratio - val_ratio) 划分数据。
    默认 8:1:1。
    """
    # 加载全部数据
    samples = []
    for line in open(input_path, encoding="utf-8"):
        line = line.strip()
        if line:
            samples.append(json.loads(line))

    n_total = len(samples)
    if n_total == 0:
        raise ValueError(f"No data found in {input_path}")

    # 随机打乱
    random.seed(seed)
    random.shuffle(samples)

    # 计算划分点
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # 剩余的全部给测试集（避免浮点误差丢失样本）
    n_test = n_total - n_train - n_val

    train_data = samples[:n_train]
    val_data = samples[n_train:n_train + n_val]
    test_data = samples[n_train + n_val:]

    # 保存为 pkl
    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(splits, f)

    print(f"Data split complete (seed={seed}):")
    print(f"  Total:  {n_total}")
    print(f"  Train:  {n_train} ({n_train/n_total*100:.1f}%)")
    print(f"  Val:    {n_val} ({n_val/n_total*100:.1f}%)")
    print(f"  Test:   {n_test} ({n_test/n_total*100:.1f}%)")
    print(f"  Output: {output_path}")

    return splits


def main():
    parser = argparse.ArgumentParser(
        description="CAMA-D: Split HFA-C²N data into train/val/test (8:1:1)"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file (HFA-C²N inference data)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output pkl file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train set ratio (default: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Val set ratio (default: 0.1)")
    args = parser.parse_args()

    split_data(args.input, args.output, args.seed,
               args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()
