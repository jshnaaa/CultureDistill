"""
Figure 4: 文化频率分组准确率柱状图
每个基座模型生成一张图，包含三个子图（三个数据集），
纵坐标=准确率，横坐标=方法，三组柱子=High-Freq / Rare / Overall。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
})

# ═══ Data from recomputed correct values ═══
with open("/Users/yzl/ownCode/AgentArk/figure4_data.json") as f:
    all_data = json.load(f)

methods = ["HF-CAC", "MAD", "MACD", "OG-MAR", "MD"]
datasets = ["NormAd", "CulturalBench", "BLEnD"]
groups = ["High-Freq", "Rare", "Overall"]

colors = {
    "High-Freq": "#2E86AB",
    "Rare":      "#E74C3C",
    "Overall":   "#7F8C8D",
}

def plot_figure(base_data, base_name, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    fig.subplots_adjust(wspace=0.28, left=0.06, right=0.97, top=0.88, bottom=0.18)
    fig.suptitle(f"Culture Frequency Analysis ({base_name})", fontsize=13, fontweight='bold', y=0.98)

    bar_width = 0.22
    x = np.arange(len(methods))

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        ds_data = base_data[ds]

        for g_idx, group in enumerate(groups):
            key = ["H_acc", "R_acc", "O_acc"][g_idx]
            vals = [ds_data[m][key] for m in methods]
            offset = (g_idx - 1) * bar_width
            bars = ax.bar(x + offset, vals, bar_width * 0.88,
                          label=group if idx == 0 else "",
                          color=colors[group], alpha=0.85,
                          edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=6.5,
                        color=colors[group], fontweight='bold')

        ax.set_title(ds, fontweight='bold', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha='right')
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")

        all_vals = [ds_data[m][k] for m in methods for k in ["H_acc", "R_acc", "O_acc"]]
        ymin = min(all_vals) - 6
        ymax = max(all_vals) + 5
        ax.set_ylim(ymin, ymax)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[g], alpha=0.85) for g in groups]
    axes[1].legend(handles, groups, loc='upper center', ncol=3, frameon=True,
                   fancybox=True, edgecolor='lightgray', framealpha=0.9,
                   bbox_to_anchor=(0.5, 1.22))

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Done: {output_path}")

# Generate both figures
plot_figure(all_data["qwen"], "Qwen", "/Users/yzl/ownCode/AgentArk/figure4_culture_freq_qwen.png")
plot_figure(all_data["llama"], "Llama", "/Users/yzl/ownCode/AgentArk/figure4_culture_freq_llama.png")
