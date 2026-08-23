"""
Figure 5: CFSE 第二层指标 —— 相对退化度 ΔΔDeg 柱状图
每个基座模型生成一张图，横坐标=数据集，不同颜色柱子=方法（HF-CAC/MAD/MACD/OG-MAR/MD），
纵坐标=ΔΔDeg（百分点），0 线以上表示退化加剧，0 线以下表示退化缓解。
数据直接写死，来自表 5（ΔΔDeg 汇总）。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
})

methods = ["HF-CAC", "MAD", "MACD", "OG-MAR", "MD"]
datasets = ["NormAd", "CulturalBench", "BLEnD"]

# ΔΔDeg = ΔDeg(method) - ΔDeg(Base)，数据来自正文表 5
ddeg = {
    "qwen": {
        "NormAd":        {"HF-CAC": 7.49,  "MAD": 10.76, "MACD": 6.67,  "OG-MAR": 6.20,  "MD": 5.56},
        "CulturalBench": {"HF-CAC": 0.43,  "MAD": 1.02,  "MACD": 0.83,  "OG-MAR": -1.87, "MD": 0.92},
        "BLEnD":         {"HF-CAC": -0.71, "MAD": 0.88,  "MACD": 0.05,  "OG-MAR": -0.83, "MD": -1.23},
    },
    "llama": {
        "NormAd":        {"HF-CAC": 5.54,  "MAD": 3.04,  "MACD": 4.83,  "OG-MAR": 7.20,  "MD": 4.51},
        "CulturalBench": {"HF-CAC": -1.88, "MAD": 0.84,  "MACD": 1.13,  "OG-MAR": 0.77,  "MD": 0.91},
        "BLEnD":         {"HF-CAC": -0.22, "MAD": 0.71,  "MACD": -2.27, "OG-MAR": -0.89, "MD": 0.87},
    },
}

method_colors = {
    "HF-CAC": "#2E86AB",
    "MAD":    "#E74C3C",
    "MACD":   "#F39C12",
    "OG-MAR": "#8E44AD",
    "MD":     "#7F8C8D",
}


def plot_figure(base_data, base_name, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    fig.subplots_adjust(wspace=0.32, left=0.06, right=0.97, top=0.85, bottom=0.15)
    fig.suptitle(f"CFSE Relative Degradation Δ\u0394Deg ({base_name})", fontsize=13, fontweight='bold', y=0.98)

    x = np.arange(len(methods))
    bar_width = 0.6

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        vals = [base_data[ds][m] for m in methods]
        colors = [method_colors[m] for m in methods]
        bars = ax.bar(x, vals, bar_width, color=colors, alpha=0.88, edgecolor="white", linewidth=0.6)

        for bar, val in zip(bars, vals):
            va = 'bottom' if val >= 0 else 'top'
            offset = 0.25 if val >= 0 else -0.25
            ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                     f'{val:+.2f}', ha='center', va=va, fontsize=7.5, fontweight='bold',
                     color=bar.get_facecolor())

        ax.axhline(0, color='black', linewidth=0.9, zorder=1)
        ax.set_title(ds, fontweight='bold', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha='right')
        ax.set_ylabel("ΔΔDeg (pp)" if idx == 0 else "")

        all_vals = vals
        ymin = min(all_vals) - 2.5
        ymax = max(all_vals) + 2.5
        ax.set_ylim(ymin, ymax)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    handles = [plt.Rectangle((0, 0), 1, 1, fc=method_colors[m], alpha=0.88) for m in methods]
    axes[1].legend(handles, methods, loc='upper center', ncol=5, frameon=True,
                   fancybox=True, edgecolor='lightgray', framealpha=0.9,
                   bbox_to_anchor=(0.5, 1.24), columnspacing=1.2)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Done: {output_path}")


plot_figure(ddeg["qwen"], "Qwen", "/Users/yzl/ownCode/AgentArk/figure/figure5_ddeg_qwen.png")
plot_figure(ddeg["llama"], "Llama", "/Users/yzl/ownCode/AgentArk/figure/figure5_ddeg_llama.png")
