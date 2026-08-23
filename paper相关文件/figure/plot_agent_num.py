#!/usr/bin/env python3
"""
方案B：横坐标=智能体数量(2~6)，不同颜色=数据集
每个基座一张子图
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ──────────────────────────────────────────────
# 数据（保留两位小数）
# ──────────────────────────────────────────────

# Qwen: {agent_count: [NormAd, CulturalBench, BLEnD]}
qwen = {
    2: [66.84, 73.94, 66.61],
    3: [69.96, 74.18, 67.37],
    4: [66.35, 74.35, 65.62],
    5: [65.58, 75.49, 67.51],
    6: [66.81, 75.57, 68.45],
}

llama = {
    2: [64.41, 69.44, 65.13],
    3: [64.49, 69.60, 65.31],
    4: [64.87, 69.93, 65.53],
    5: [65.51, 70.33, 65.22],
    6: [65.86, 72.37, 67.25],
}

datasets = ["NormAd", "CulturalBench", "BLEnD"]
agent_counts = [2, 3, 4, 5, 6]

# ──────────────────────────────────────────────
# 莫兰迪配色（3 种，对应 3 个数据集）
# ──────────────────────────────────────────────
ds_colors = [
    "#7B9EAE",   # 灰蓝  — NormAd
    "#B07A84",   # 灰粉  — CulturalBench
    "#8FA880",   # 灰绿  — BLEnD
]

ds_markers = ["o", "s", "D"]


def plot_one(data, base_name, save_path):
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)

    x = np.array(agent_counts)

    for ds_idx, ds_name in enumerate(datasets):
        y = [data[n][ds_idx] for n in agent_counts]
        ax.plot(
            x, y,
            color=ds_colors[ds_idx],
            marker=ds_markers[ds_idx],
            markersize=8,
            linewidth=2.2,
            label=ds_name,
            zorder=3,
        )
        # 标注每个数据点
        for xi, yi in zip(x, y):
            ax.annotate(
                f"{yi:.2f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=7,
                fontweight="bold",
                color=ds_colors[ds_idx],
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in agent_counts], fontsize=11)
    ax.set_xlabel("Number of Agents", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"HF-CAC Agent Count Ablation ({base_name})",
        fontsize=13, fontweight="bold", pad=12,
    )

    all_vals = [data[n][i] for n in agent_counts for i in range(3)]
    y_min = min(all_vals) - 3.0
    y_max = max(all_vals) + 3.5
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(1.6, 6.4)

    ax.legend(
        loc="upper left",
        fontsize=9,
        framealpha=0.92,
        edgecolor="#bbbbbb",
        fancybox=True,
    )

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure")
os.makedirs(out_dir, exist_ok=True)

plot_one(qwen, "Qwen", os.path.join(out_dir, "figure3_1_agent_num_qwen_v2.png"))
plot_one(llama, "Llama", os.path.join(out_dir, "figure3_1_agent_num_llama_v2.png"))

print("Done.")
