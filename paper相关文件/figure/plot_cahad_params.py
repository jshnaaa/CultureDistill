# -*- coding: utf-8 -*-
"""
CAHAD Hyperparameter Analysis: β and λ_g sweep on NormAd × Qwen2.5-7B.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Data from Appendix C (NormAd × Qwen, 10% subsample) ──
beta_vals  = [0.1, 0.3, 0.5, 0.7]
beta_acc   = [65.15, 66.29, 65.91, 65.53]

lg_vals    = [0.1, 0.3, 0.5, 0.7]
lg_acc     = [64.77, 65.53, 66.29, 64.77]

# ── Plot styling ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# ── (a) β sweep ──
ax1.plot(beta_vals, beta_acc, 'o-', color='#2E75B6', linewidth=2.0,
         markersize=8, markerfacecolor='white', markeredgewidth=2.0,
         markeredgecolor='#2E75B6', zorder=5)
# Highlight default
best_idx = beta_acc.index(max(beta_acc))
ax1.plot(beta_vals[best_idx], beta_acc[best_idx], 's', color='#C0504D',
         markersize=10, markerfacecolor='#C0504D', zorder=6)
ax1.annotate(f'{beta_acc[best_idx]:.2f}',
             xy=(beta_vals[best_idx], beta_acc[best_idx]),
             xytext=(beta_vals[best_idx]+0.05, beta_acc[best_idx]+0.25),
             fontsize=11, fontweight='bold', color='#C0504D')

for i, (x, y) in enumerate(zip(beta_vals, beta_acc)):
    if i != best_idx:
        ax1.annotate(f'{y:.2f}', xy=(x, y),
                     xytext=(x+0.03, y-0.35),
                     fontsize=10, color='#555555')

ax1.set_xlabel(r'$\beta$ (SFT/RL loss balance coefficient)')
ax1.set_ylabel('Overall Accuracy (%)')
ax1.set_title(r'(a) $\beta$ sweep ($\lambda_g$=0.5)')
ax1.set_xticks(beta_vals)
ax1.set_xticklabels([str(v) if v != 0.3 else r'$\mathbf{0.3}$' for v in beta_vals])
ax1.set_ylim(64.2, 67.0)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── (b) λ_g sweep ──
ax2.plot(lg_vals, lg_acc, 's-', color='#4CAF50', linewidth=2.0,
         markersize=8, markerfacecolor='white', markeredgewidth=2.0,
         markeredgecolor='#4CAF50', zorder=5)
best_idx2 = lg_acc.index(max(lg_acc))
ax2.plot(lg_vals[best_idx2], lg_acc[best_idx2], 'D', color='#C0504D',
         markersize=10, markerfacecolor='#C0504D', zorder=6)
ax2.annotate(f'{lg_acc[best_idx2]:.2f}',
             xy=(lg_vals[best_idx2], lg_acc[best_idx2]),
             xytext=(lg_vals[best_idx2]+0.05, lg_acc[best_idx2]+0.25),
             fontsize=11, fontweight='bold', color='#C0504D')

for i, (x, y) in enumerate(zip(lg_vals, lg_acc)):
    if i != best_idx2:
        ax2.annotate(f'{y:.2f}', xy=(x, y),
                     xytext=(x+0.03, y-0.35),
                     fontsize=10, color='#555555')

ax2.set_xlabel(r'$\lambda_g$ (Guardian guidance strength)')
ax2.set_ylabel('Overall Accuracy (%)')
ax2.set_title(r'(b) $\lambda_g$ sweep ($\beta$=0.3)')
ax2.set_xticks(lg_vals)
ax2.set_xticklabels([str(v) if v != 0.5 else r'$\mathbf{0.5}$' for v in lg_vals])
ax2.set_ylim(64.2, 67.0)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout(pad=2.0)
plt.savefig('/Users/yzl/ownCode/AgentArk/figure/CAHAD_beta_lambda.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Figure saved: figure/CAHAD_beta_lambda.png')
