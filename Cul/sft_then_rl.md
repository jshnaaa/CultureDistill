# SFT 与 RL 混合训练：分类综述、最新进展与文化对齐场景的创新思路

> 本文档基于 Cul/混合训练调研.pdf 的系统调研，补充 2025 年底至 2026 年最新论文，并结合本项目 culturedistill.md 中的 CAMAD 文化对齐蒸馏框架，提出具有创新性且工程可行的 SFT+RL 混合训练思路。
>
> 阅读对象：CAMAD/AgentArk 训练管线开发者。
> 关键背景：CAMAD 当前采用经典的「Stage1 加权SFT → Stage2 PRM → Stage3 GRPO」三阶段串行范式，本文档的核心目标是论证为什么以及如何把 SFT 信号融入 GRPO 阶段。

---

## 一、问题的起点：为什么要讨论「混合」而不是「先后」

CAMAD 现状是一条标准的两阶段流水线：先用主场权威加权 SFT 把 Guardian 的确权推理模式注入单体模型，再用 PRM 提供的过程奖励做 GRPO 强化学习。这条路线易实现、好调试，但它继承了两阶段范式所有已知的结构性缺陷。

调研报告与最新研究普遍指出，先 SFT 后 RL 的串行结构存在四类根本性问题。第一是灾难性遗忘，GRPO 只盯着奖励信号优化，会逐渐冲刷掉 SFT 阶段学到的语言能力和文化知识，典型表现是输出变长、逻辑变乱、分布外样本崩溃（Mitigating Forgetting Between Supervised and Reinforcement Learning, ICLR 2026 观测到纯 RL 把平均输出从 1247 token 拉到 4421 token 且性能下降）。第二是解耦导致的低效探索，SFT 学到的策略和 RL 的最优策略可能差异很大，RL 几乎要从头探索，浪费算力重学已会的技能（BRIDGE, 2025）。第三是无法利用互补性，SFT Memorizes RL Generalizes（ICML 2025）从信息论角度证明 SFT 擅长记忆、RL 擅长泛化，串行范式无法在训练中动态平衡二者。第四是调参繁琐与切换时机不确定，过早切 RL 不稳定，过晚切又限制 RL 的优化空间，目前缺乏理论指导最优切换点。

对 CAMAD 而言，灾难性遗忘问题尤其致命：我们在 SFT 阶段费力注入的，恰恰是「主场文化确权」这种长尾、稀疏、易被西方语料覆盖的知识。一旦 GRPO 阶段把它遗忘，整个框架的核心卖点（Guardian 的知识精度）就被掏空了。这正是讨论混合训练的现实动机。

需要强调一个理论前提：On the Non-decoupling of SFT and RL（2026）证明 SFT 和 RL 在后训练中本质不可解耦——SFT 不只是 RL 的前置步骤，更是 RL 优化过程中不可缺少的组成部分，任何强行分离的尝试都会掉性能。既然不可解耦，那么把二者融合进同一个训练过程就是最自然的选择。

---

## 二、理论基础：SFT 与 RL 本就是同一件事的两种特例

混合训练能成立，靠的是「SFT 与 RL 在数学上统一」这一系列理论突破。理解这层统一性，是后面所有方法分类的基础。

最早的统一来自北京大学的 Intuitive Fine-Tuning（ACL 2025 Oral, arXiv:2405.11870），它在 token 级把语言生成建模为马尔可夫决策过程（MDP），证明 SFT 是 RLHF 的一个特例：SFT 只在目标答案的每个 token 位置做单步更新，RLHF 则在整个序列所有 token 位置做全序列更新，两者唯一区别是更新的时间步范围不同，优化本质都是最大化累积奖励。

清华大学的 HPT（Towards a Unified View of LLM Post-Training, arXiv:2509.04419）把这一观点工程化为统一策略梯度估计器（UPGE），将 SFT、RL、DPO、ORPO 等所有主流后训练方法的梯度统一写成一个通用表达式：梯度 = 期望[ 权重函数 w(x,y) 乘以 log 策略概率的梯度 ]。差异只在权重函数 w(x,y) 的形式。

各方法的权重函数对照：SFT 的权重恒为 1，表示对示范数据无差别拟合；RL（GRPO/PPO）的权重是优势函数 A(x,y)，按奖励高低加权；DPO 的权重是 sigmoid(beta 乘以 reward)，按偏好强度加权；混合训练的权重是 alpha 乘以 1 加 beta 乘以 A(x,y)，即 SFT 与 RL 信号线性叠加。

这个表达式是整篇文档的「元公式」。它告诉我们：所谓混合训练，本质就是设计一个合适的权重函数，让同一份梯度同时携带「模仿示范」和「最大化奖励」两种信号。后面六大类方法，无非是在回答两个问题——这个权重怎么设（固定/退火/动态），以及在什么粒度上设（token级/step级/样本级/阶段级）。

补充一个 2026 年的重要发现：GRPO is Secretly a Process Reward Model（arXiv:2509.21154）证明在组内轨迹共享 token 前缀的假设下，vanilla GRPO 即使只拿到结果级奖励，内部也隐式做了 step-level 的信用分配，等价于诱导出一个非平凡的 PRM。这对 CAMAD 有直接启发：我们显式训练的 Culture-Aware PRM，与 GRPO 内建的隐式 PRM 是可以协同的，而不是冗余的——显式 PRM 提供「文化合理性」的领域先验，隐式 PRM 提供「答案正确性」的过程信用，两者互补。

---

## 三、六大类混合方法的分类体系与代表工作

沿用调研报告基于「技术实现核心差异」的六类分类法，并对每类补充我的判断与 2026 最新代表工作。分类的核心区分维度是：融合发生在哪一层（损失/数据/时间/优化结构），以及任意时刻是否同时存在两种信号。

### 3.1 损失加权融合法（Loss-Weighting）

核心思想是在同一个损失函数里加权组合 SFT loss 和 RL loss，单阶段训练、同时使用两种信号。这是最直接的 UPGE 实例化——权重等于 alpha 乘以 1 加 beta 乘以 A。优点是实现简单、训练稳定、计算高效；缺点是当两种信号目标冲突时容易梯度抵消。权重可以是固定值、按训练步退火的值，或按模型状态动态调整的值，而「权重怎么定」恰恰是这一类方法的分水岭。

代表工作的演进很能说明问题。最早的 ORPO（arXiv:2403.07691）把 SFT 和偏好对比塞进单一损失，用对数几率比（odds ratio）做弱偏好惩罚，彻底去掉了参考模型，证明了单阶段可行。但固定权重无法适应训练动态，于是出现了动态加权分支。

SRFT（中科院+美团, ICLR 2026 Poster, arXiv:2506.19767）以熵作为核心调度指标。它发现 SFT 引起策略分布的粗粒度全局变化、RL 做细粒度选择性优化，于是设计熵感知加权：熵高时（模型还很迷茫）加大 SFT 权重快速学基本模式，熵低时（模型已较确定）加大 RL 权重做精细探索。在五个数学基准上比 zero-RL 高 9.0%，OOD 高 10.9%。

CHORD（阿里通义, arXiv:2508.11408）把 SFT 重新定义为 on-policy RL 过程中的动态加权辅助目标，提出双重控制：全局系数 mu 控制 SFT/RL 整体配比（前期 SFT 重、后期递减），token 级权重函数 phi 按当前策略对每个专家 token 的熵逐词加权——模型对某 token 越不确定就越从该 token 学 SFT，已掌握的低熵 token 则减少干预。总损失为 mu 乘以 SFT损失(phi) 加 (1 减 mu) 乘以 GRPO损失。

我的判断：CHORD 的 token 级不确定度加权是这一类里最适合 CAMAD 的机制。原因在后面第五节详述——文化知识的注入恰恰需要「只在模型不确定的文化 token 上强化监督，对已会的常识不打扰」这种精细控制。

### 3.2 数据混合法（Data-Mixing）

核心思想是在数据 batch 层面混合 SFT 示范数据和 RL 在线采样数据，但对所有数据点用同一个 RL 损失函数。它和损失加权法的本质区别是：融合发生在数据层而非损失层，损失函数保持统一纯粹，从而避免损失层面的目标冲突，代价是要处理两类数据分布不一致带来的梯度偏差。

LUFFY（上交等, arXiv:2504.14945）是这一类最有代表性的工作，也最值得 CAMAD 借鉴。它针对的痛点是：zero-RL 受限于模型自身能力，如果模型本身生成不出正确推理路径，纯 on-policy RL 根本无从学起。LUFFY 的做法是 Mixed-Policy GRPO——把强模型（如 DeepSeek-R1，对 CAMAD 来说就是 HF-CAC 多智能体系统）的 off-policy 推理轨迹，与模型自身的 on-policy rollout 混进同一次 advantage estimation。同时用正则化重要性采样做 policy shaping，强调那些「低概率但关键」的 token，避免模型只做表面僵硬模仿。

GTA（EMNLP 2025）走的是动态示范路线，用一个生成式教学助手实时诊断学生模型当前弱点、按需生成定制化中间推理步骤混入 RL。它证明针对性的 1 条动态数据约等于固定数据集里 10-20 条，收敛速度提升 3-5 倍。

我的判断：LUFFY 的 Mixed-Policy 思路对 CAMAD 几乎是量身定做。CAMAD 的学生模型（7B/8B）在长尾国家文化题上，on-policy rollout 大概率全军覆没（R_outcome 全 0、组内无梯度），这正是 LUFFY 要解决的「模型能力外」场景。把 HF-CAC 生成的 Guardian 确权轨迹作为 off-policy 示范混进 GRPO 组内，可以在 rollout 全错时仍然提供正向学习信号。

### 3.3 课程学习融合法（Curriculum）

核心思想是从完全 SFT 平滑过渡到完全 RL，SFT 比例随训练进程从 100% 单调降到 0%，模拟人类从模仿到自主探索的过程。单阶段、过渡平滑、最稳定，符合学习直觉；但过渡速度难控，仍存在过早/过晚切换的老问题，只是把硬切换变成了软退火。我的判断：它更像是损失加权法的一个特例（权重按时间退火），独立性不强，对 CAMAD 可作为辅助调度策略而非主框架。

### 3.4 交错 / 交替训练法（Interleaved）

核心思想是周期性地在 SFT 阶段和 RL 阶段之间切换，但两阶段比例相对稳定（不像课程学习那样单调过渡）。多阶段、时间维度分离但交替进行。它能有效解决灾难性遗忘（定期回注 SFT 把遗忘的知识捞回来），还能突破纯 RL 的能力边界；代价是流程复杂，要调阶段长度和切换频率。

ReLIFT（ICLR 2026, arXiv:2506.07527）是这一类的标杆。它的洞见是「学 RL 学不会的东西」：RL 擅长在模型已有能力范围内提升，但对超出能力边界的最难题束手无策。ReLIFT 以 RL 为主，当遇到 RL 持续失败的难题时，收集高质量示范解放进动态缓冲区，用 online SFT 针对这类难题微调，再恢复 RL。RL 巩固已知、SFT 突破未知，交替进行。

调研报告还提到一篇重要的警示性论文 SFT-then-RL Outperforms Mixed-Policy Methods（arXiv:2604.23747），它在严格控制变量下发现：很多复杂单阶段混合方法的性能优势其实来自有缺陷的基线实现或偷偷增加的训练步数，相同算力和超参下，简单的多轮 SFT-then-RL 交替反而打败了大多数花哨的单阶段方法。

我的判断：这篇警示论文必须认真对待。它的结论不是「混合无用」，而是「评估要公平、方法别为复杂而复杂」。对 CAMAD 的实践指导是——先把多轮 SFT 与 RL 交替（也就是 ReLIFT 的朴素版）做成一个强基线，任何更复杂的单阶段融合方法都必须在相同算力预算下打败它才算真正有效。这也是我在第五节会把交错回注作为 CAMAD 渐进式落地第一步的原因。

### 3.5 双层优化融合法（Bilevel）

核心思想是用元学习框架建立 SFT 与 RL 的深度协同，SFT 作内层、RL 作外层（或反之），SFT 的目标不再是单纯拟合示范，而是「最大化 RL 的最终性能」。嵌套优化结构，理论最深厚、协同最深，但实现复杂、算力高。

BRIDGE（微软亚研+港中文, arXiv:2509.06948）是代表：把 SFT 设为上层、RL 设为下层，上层 SFT 通过元学习去引导下层 RL 的优化过程，并做参数分离——base 参数由下层 RL 优化、LoRA 参数由上层 SFT 优化，避免两个目标互相覆盖。上层目标显式最大化「联合训练相对纯 RL 的增益」。在 Qwen2.5-3B 上比两阶段 cold-start 平均高 11.8%。

我的判断：理论很美但工程成本高，二阶梯度/penalty relaxation 对两卡 ZeRO-3 的 CAMAD 环境压力大，建议作为远期探索而非首选。不过它「LoRA 参数和 base 参数分别承载 SFT/RL 目标」的参数分离思想，对 CAMAD 用 LoRA 的现状有借鉴价值。

### 3.6 混合策略 / 条件训练法（Conditional Routing）

核心思想是根据输入条件或模型状态，动态二选一地使用 SFT 信号或 RL 信号（而非同时加权两者）。分支式流程，任意时刻只有一种信号，从而彻底避免梯度冲突；代价是要设计可靠的条件判断机制。

GHPO（华为诺亚+港城大, arXiv:2507.10628）做难度感知的硬切换：自动检测每题难度，对模型当前解不出的难题用 adaptive prompt refinement 把标准解题过程当 hint 喂进去（退化为模仿/SFT），对能力内的题启用探索式 GRPO，还能逐步增减 hint 长度形成平滑课程。

DYPO（ACL 2026 Findings, arXiv:2604.08926）做实例级三路路由：用 group rollout 准确率把样本分成 Hard（全错，多教师蒸馏 SFT 降偏差）、Mid（部分对，Group Alignment Loss 降方差）、Easy（大部分对，标准 GRPO）。难度分级直接靠 rollout 准确率，无需额外分类器。平均提升 4.8%，OOD 提升 13.3%。

UFT（MIT, NeurIPS 2025, arXiv:2505.16984）把条件路由做到了单步内：每题先让模型自主 rollout，失败就引入参考解的前缀作为 hint，模型基于 hint 续写并用 RL 信号优化，实现「在指导下探索」。

我的判断：DYPO 的「按 rollout 准确率做样本级路由」是工程性价比最高的范式，几乎零额外成本（rollout 本来就要做），且和 GRPO 天然兼容。对 CAMAD 而言，难度分级可以直接复用现有 GRPO 的 G=10 采样结果，无缝接入。

### 3.7 六大类横向对比

| 类别 | 融合层面 | 同时双信号 | 阶段数 | 抗遗忘 | 实现成本 | 代表工作 | 对 CAMAD 推荐度 |
|------|----------|------------|--------|--------|----------|----------|------------------|
| 损失加权 | 损失函数 | 是 | 单阶段 | 中 | 低 | ORPO/SRFT/CHORD | 强（CHORD token级加权） |
| 数据混合 | 数据 batch | 是 | 单阶段 | 中 | 中 | LUFFY/GTA | 强（LUFFY Mixed-Policy） |
| 课程学习 | 时间退火 | 是 | 单阶段 | 中 | 低 | UFT-curriculum | 弱（作辅助调度） |
| 交错交替 | 时间周期 | 否 | 多阶段 | 强 | 中 | ReLIFT | 较强（落地第一步） |
| 双层优化 | 优化结构 | 是 | 嵌套 | 强 | 高 | BRIDGE | 弱（远期探索） |
| 条件路由 | 分支选择 | 否 | 单阶段 | 中 | 中 | GHPO/DYPO/UFT | 强（DYPO 难度路由） |

### 3.8 任务偏好与一条重要纪律

调研报告总结了不同任务的方法偏好：数学/代码偏好 SFT 到 RL 或简单交替（需要精确知识，混合易梯度干扰）；对话/Agent 偏好交错/混合（要平衡流畅性和任务完成）；多模态偏好条件路由（模态差异大）。文化对齐任务的特点更接近「对话/Agent 加知识密集」的组合——既要文化事实精确（像数学），又要推理风格得体（像对话），还要处理长尾不对称知识。这意味着 CAMAD 不能照搬单一范式，需要组合方案。

一条必须遵守的纪律来自警示论文：任何混合方法的有效性，都必须在相同算力预算和公平超参下，与「多轮 SFT-then-RL 交替」基线对比验证。否则很容易把多训了几步误判为方法更优。

---

## 四、文化对齐场景的特殊性：为什么通用方法不能直接套

在提出 CAMAD 的混合思路前，必须先讲清楚文化对齐相比数学推理的三个本质差异，这些差异决定了哪些通用方法能用、要怎么改。

第一，奖励的双重性。数学题只有一个硬奖励——答案对不对。文化题有两层：答案对不对（R_outcome，硬指标）和推理是否体现主场文化合理性（R_process，软指标）。CAMAD 已经用 R_total = 0.6 乘以 R_outcome 加 0.4 乘以 Mean(R_process) 巧妙统一了量纲。这意味着 CAMAD 天然就是一个多奖励 GRPO，混合 SFT 信号时必须考虑它和这两层奖励的交互。

第二，知识的长尾不对称性。文化知识高度属地化，西方语料在预训练中占主导，导致小众非西方国家的文化题上，on-policy rollout 极易被西方视角带偏而全部答错。这恰好命中 LUFFY/ReLIFT/GHPO 所有「模型能力外难题」方法的核心场景。CAMAD 的 HF-CAC 主场守护者机制，本质上就是为这类长尾题提供高质量 off-policy 示范的源头。

第三，过程监督的领域信号已经现成。CAMAD 的 Culture-Aware PRM 把推理步骤标成主场确权步（0.9）、中立步（0.5）、文化混淆步（0.1）。这是一个比通用 PRM 更强的、带文化语义的过程信号。结合前面提到的 GRPO 隐式 PRM 理论，CAMAD 手里其实握着两个 PRM：显式的文化 PRM（管推理合理性）和 GRPO 隐式 PRM（管答案信用），这是通用方法没有的资源。

结论：CAMAD 不缺数据、不缺过程奖励，缺的是一个能在 GRPO 阶段持续防止文化知识遗忘、并在长尾题上注入 Guardian 示范的融合机制。下面的创新思路就围绕这个缺口设计。

---

## 五、面向 CAMAD 的创新混合思路

以下四个思路按工程可行性从高到低、创新性从渐进到激进排列。前两个建议作为近期可落地的主力方案，后两个作为论文创新点和中期探索。所有方案都复用 CAMAD 现有的 HF-CAC 数据、Culture-Aware PRM 和 GRPO 管线，改动集中在 Stage 3。

### 思路一（落地第一步）：文化锚点交错回注（Culture-Anchored Interleaved Replay）

这是 ReLIFT 在文化场景的特化版，也是必须先建立的强基线。

机制：GRPO 主训练每隔 K 轮（建议 K=5，对齐现有 eval_every），插入一个轻量 SFT 回注步。回注数据不是随机抽取，而是专门挑选那些当前 policy 在验证集上文化准确率掉得最多的国家对应的 HF-CAC Guardian 轨迹——也就是哪个文化圈被 RL 遗忘了，就专门回灌哪个文化圈的主场确权数据。回注时沿用 Stage 1 的 token 级加权（Guardian token alpha=2.0、Auditor 非最终轮掩码），保证回注的是纯净的确权信号。

为什么可行：完全复用现有 SFT 脚本，仅增加一个按国家分组的遗忘检测逻辑。是 ReLIFT 思想在文化场景的直接落地，风险最低、收益明确，应作为第一优先实现的方案。

### 创新思路二（主推方案）：文化难度感知的混合策略 GRPO（CGM-GRPO: Culture-Guided Mixed-Policy GRPO）

这是本文档的主推创新方案。核心思想不是简单的 Mixed-Policy（LUFFY 已做），而是 Culture-Difficulty-Aware Mixed-Policy——根据文化难度动态决定「什么时候相信专家（Guardian）、什么时候相信模型自己」。这是文化对齐任务独有的创新点：文化知识的难度拥有明确的结构化先验（数据频率 + 模型实时能力 + 文化迁移距离），而通用推理任务（如数学）的难度缺乏这样的外部先验。

机制概述：在 GRPO 的 advantage estimation 中，保持 RLOO 对 on-policy 轨迹的计算完全不变，额外叠加一个来自 HF-CAC Guardian 的引导信号作为 advantage 增强项。引导强度由三因子文化难度系数 w_culture 动态调制。Guardian 不参与 RLOO baseline 计算（保持理论合法性），也不参与 policy gradient 的梯度计算（不需要 importance sampling），它只通过自身的 reward 值影响 on-policy 轨迹被鼓励/抑制的程度。

核心公式：A_i = (R_i - R_on_bar) + lambda * w_culture * S_guardian

其中：第一项 (R_i - R_on_bar) 是标准 RLOO advantage，R_on_bar 是同一 prompt 下其他 on-policy 轨迹的 leave-one-out 均值。第二项是 Guardian 文化专家引导项，S_guardian = R_outcome_guardian * (R_guardian - R_on_bar_full)，即 Guardian 轨迹的 R_total 相对于 on-policy 全组均值的优势（乘以 Guardian 答案正确性作为质量门控）。w_culture 是三因子文化难度系数，根据当前 prompt 的文化学习难度动态调制引导强度。lambda 是全局引导强度超参（建议 0.5）。

公式的直觉：当模型在某文化题上全部答错（R_on_bar 趋近 0）而 Guardian 答对（R_guardian 较高）时，S_guardian 很大、w_culture 也很大（hit_rate=0），第二项把所有 on-policy 轨迹的 advantage 统一上调——即使它们本身的 reward 都很低，也会受到 Guardian 的正向引导。反之，模型自己答对率很高时，w_culture 趋近 0，第二项消失，完全回退为标准 GRPO。

为什么这比 LUFFY 更适合文化场景：LUFFY 对所有样本统一注入 off-policy 信号（且需要 importance sampling 修正），我们根据文化难度差异化地调控引导力度（且不需要 IS）。文化任务的独特之处在于：不同文化圈的「模型学习难度」差异极大且可被先验量化，这为差异化引导提供了理论和数据支撑。

---

#### 思路二详细实现方案（修订版：CGM-GRPO）

##### A. 核心算法设计与理论合法性

具体流程如下。对每个 prompt (question, country)：

第一步，当前 policy 采样 G 条推理路径（与现有逻辑完全一致，G=5）。

第二步，从 HF-CAC 数据中检索该 prompt 对应的 Guardian 推理轨迹（已存在于 pkl 数据的 response 字段中，格式为 "===== Solution 1 [GUARDIAN] ===== Reasoning: ... Answer: ..."），解析出 Guardian 的 reasoning 部分。对 Guardian 轨迹计算 R_guardian = alpha * R_outcome_guardian + (1-alpha) * Mean(R_process_guardian)，使用与 on-policy 轨迹完全相同的 reward 计算逻辑（规则验证答案 + PRM scoring）。

第三步，对 G 条 on-policy 轨迹计算标准 RLOO advantage：A_i_base = R_i - R_on_bar（R_on_bar 是 leave-one-out baseline）。这一步与现有 train_grpo_v3.py 的 rloo_advantages 函数完全一致。

第四步，计算文化难度系数 w_culture（三因子公式，详见 B 节）。

第五步，计算 Guardian 引导信号 S_guardian = R_outcome_guardian * (R_guardian - R_on_bar_full)，其中 R_on_bar_full 是 G 条 on-policy 轨迹的全组均值（注意这里不用 leave-one-out，因为 S_guardian 对所有 on-policy 轨迹是相同的增量）。

第六步，合成最终 advantage：A_i = A_i_base + lambda * w_culture * S_guardian。

第七步，正常计算 policy gradient：gradient = sum(A_i * grad(log pi_theta(y_i|x)))，只对 on-policy 轨迹求梯度。

理论合法性说明——为什么不需要 Importance Sampling：

之前版本的方案试图让 Guardian 轨迹直接参与 RLOO 的 advantage 计算和 policy gradient，这要求计算 rho = pi_current / pi_behavior。但 Guardian 轨迹的行为策略是 HF-CAC 多智能体系统（Agent+Judge+筛选），其生成概率根本没有保存，也无法精确计算。

CGM-GRPO 的解决方案：完全不需要 importance sampling。原因是——在我们的公式中，Guardian 轨迹不参与 policy gradient 的梯度计算。它只通过 S_guardian = R_guardian - R_on_bar 这个标量值影响 on-policy 轨迹的 advantage。梯度仍然只对 on-policy 轨迹 y_i ~ pi_theta 求导：gradient = A_i * grad(log pi_theta(y_i|x))。Guardian 轨迹的作用纯粹是「调整 on-policy 轨迹被鼓励/抑制的程度」，本身不需要对 Guardian 轨迹求梯度，因此不需要 IS 修正。这比 LUFFY 的处理更简洁——LUFFY 把 off-policy 轨迹也参与梯度更新所以需要 IS，而我们只用它的 reward 信号。

##### B. 文化难度系数 w_culture 的三因子设计

w_culture 的设计目标是：量化「当前 prompt 对模型的文化学习难度」，决定 Guardian 专家引导的强度。核心洞见是：文化难度不等于文化孤立度，而是数据频率、模型能力、文化迁移难度三者的综合。

三因子公式：w_culture = lambda_1 * (1 - hit_rate) + lambda_2 * rarity_i + lambda_3 * (1 - affinity_i)

推荐系数：lambda_1 = 0.6, lambda_2 = 0.3, lambda_3 = 0.1。设计逻辑如下：

第一因子 (1 - hit_rate)：动态模型能力信号，权重最大（0.6）。hit_rate 是当前 policy 对该 prompt 的 G 条 on-policy rollout 中答对的比例。这是最直接的难度度量——模型自己答不出来的题就是难题。它随训练实时变化，天然实现了「模型学会后自动减弱引导」的自适应效果。

第二因子 rarity_i：静态数据稀缺度信号，中等权重（0.3）。定义为 rarity_i = 1 - freq_i，其中 freq_i 是目标国家所属文化圈在训练集中的样本占比。以 NormAD 数据集为例（2633 样本，75 国家）：如果 Western 文化圈覆盖约 20 个国家、占比约 30%，则 rarity_western = 0.70；Sub-Saharan African 覆盖约 6 个国家、占比约 8%，则 rarity_africa = 0.92。这解决了 hit_rate 在训练初期（普遍低）时无法区分真长尾和假长尾的冷启动问题。

第三因子 (1 - affinity_i)：文化迁移难度的弱先验，最低权重（0.1）。affinity_i 取亲缘度矩阵第 i 行的非对角线均值。它反映的是「其他文化圈的知识能否迁移到目标文化」的难度。权重设为 0.1 是因为：(a) 它与数据频率存在相关性（高频文化通常也有更多跨文化交流），独立信息量有限；(b) 前面分析发现它单独使用时会产生反直觉结果（Western 反而权重高）。作为辅助修正信号即可。

关于 rarity_i 的计算：在训练开始前，对 pkl 数据做一次性统计，按 country -> culture_circle 映射后计算各文化圈样本占比，硬编码为一个 dict。这与亲缘度矩阵一样是确定性的、不随训练变化的静态量。

为什么这个设计合理：w_culture 的值域在 [0, 1] 范围内。当模型对某文化题全部答对（hit_rate=1）、该文化又是高频（rarity 低）、且与其他文化亲缘度高（affinity 高）时，w_culture 趋近于 0，Guardian 引导几乎消失——这正是我们想要的（模型已经会了，不需要专家帮忙）。反之，模型答不出来（hit=0）、数据稀缺（rarity 高）、文化孤立（affinity 低）时，w_culture 趋近于 1，Guardian 引导最大化——这也正确（模型最需要帮助的时候）。

##### C. 是否强依赖文化亲缘度矩阵

不强依赖。在三因子公式中，亲缘度矩阵仅贡献权重 0.1 的最弱信号。即使完全去掉第三因子（lambda_3=0），公式退化为 w_culture = 0.67*(1-hit_rate) + 0.33*rarity（重新归一化），仍然是一个完全合理的文化难度度量。

最小可行版本（MVP）：只用第一因子，w_culture = 1 - hit_rate。零额外依赖，核心机制完整。

标准版本：加入 rarity（需要一次性统计训练集的文化圈分布，约 10 行代码）。

完整版本：三因子全开（额外加载 hf_cac_config.yaml 中的亲缘度矩阵，约 15 行代码）。

消融实验设计：MVP -> 标准 -> 完整，逐步验证每个因子的边际贡献。

##### D. Guardian 引导信号的门控机制

一个重要的实现细节：并非所有 prompt 都应注入 Guardian 引导。需要两个门控条件：

门控一（质量门控）：只在 Guardian 答对时激活引导。如果 Guardian 轨迹的 R_outcome = 0（答错），令 S_guardian = 0，第二项自动消失。这可以通过在公式中乘以 R_outcome_guardian 来优雅实现：S_guardian = R_outcome_guardian * (R_guardian - R_on_bar)。

门控二（必要性门控）：如果 on-policy rollout 全部答对（hit_rate = 1），w_culture 的第一因子为 0，引导自然很弱。但更极端的做法是设一个阈值：当 hit_rate >= 0.8 时直接令第二项为 0（即模型已经够好了，不需要专家了）。这能进一步避免对已掌握文化的过度干预。

##### E. 现有代码需要的修改（新建文件方案）

根据实验设计需要保留 train_grpo_v3.py 作为 SFT+RL 基线，混合策略新建 Cul/grpo/train_grpo_mixed_policy.py。新文件可以大量复用 v3 的函数，核心差异如下：

第一，数据加载模块（新增约 40 行）。扩展 GRPOPromptDataset，额外解析 response 字段中的 Guardian 推理轨迹（正则匹配 "===== Solution N [GUARDIAN] =====" 段落），提取 Reasoning 部分。同时预计算 Guardian 的 R_outcome（与 gold answer 比对）用于质量门控。加载 guardian_idx 字段。

第二，文化难度计算模块（新增约 50 行）。包含：(a) 训练集文化圈频率统计（rarity_i 计算）；(b) 亲缘度矩阵加载（可选）；(c) country -> culture_circle 映射表构建；(d) w_culture 三因子计算函数。

第三，Guardian reward 计算（新增约 20 行）。对 Guardian 轨迹做 PRM scoring 得到 R_guardian（逻辑复用现有 PRM batch scoring）。

第四，CGM-GRPO advantage 计算（修改约 30 行）。替换原有的 rloo_advantages 调用，改为：先对 on-policy 的 G 条计算标准 RLOO advantage A_i = R_i - R_on_bar，然后叠加 Guardian 引导项 lambda * w_culture * S_guardian。

第五，policy gradient 计算（基本不变）。关键点：梯度只对 on-policy 轨迹求导，Guardian 轨迹不参与 backward。因此 Phase B 的循环范围仍然只遍历 on-policy 的 G 条 response，只是它们的 advantage 值被 Guardian 调整过了。不需要对 Guardian 轨迹计算 log-prob 或做 backward。这比之前的方案简单得多。

第六，新增命令行参数（约 10 行）：--lambda_guide（Guardian 引导强度，默认 0.5）、--affinity_config（可选，亲缘度矩阵配置路径）、--w_culture_mode（"hit_only" / "hit_rarity" / "full"，对应 MVP/标准/完整版）。

第七，日志与监控（新增约 15 行）。打印：平均 w_culture、S_guardian 均值、Guardian 答对率、各文化圈 hit_rate 分布。

总新增代码量预估：约 250-300 行（新文件），其中大部分是从 v3 复制过来的框架代码。核心新逻辑约 100 行。难度评估为中低——比之前的方案更简单，因为：(a) 不需要 importance sampling；(b) 不需要对 Guardian 轨迹计算 log-prob；(c) 不需要修改 Phase B 的梯度计算逻辑；(d) Guardian 信号只是一个标量加到 advantage 上。

##### F. 显卡资源需求与训练时长预估

当前配置：2 卡 48GB vGPU（policy on cuda:0, PRM on cuda:1）。

显存分析：CGM-GRPO 相比原版 GRPO 的额外显存开销极小。原因是 Guardian 轨迹不参与 policy 的 forward/backward（不需要计算 log-prob），只需要过一次 PRM 获取 R_guardian 分数。PRM scoring 在 cuda:1 上进行，每个 batch 多 8 条文本（8 个 prompt 各 1 条 Guardian 轨迹）的 PRM forward pass，相比现有的 8*5=40 条 on-policy 轨迹，增量仅 20%，且 PRM 本身已是 eval 模式、无梯度。结论：2x48GB 完全够用，不需要任何调整。

训练时长预估：主要额外开销是 Guardian 轨迹的 PRM scoring（每 batch 多 8 条，约增加 15% 的 PRM 时间）。on-policy 部分的 generate、Phase A、Phase B 完全不变。综合每轮增加约 5-10% 的时间，即每轮约 42-66 分钟（比之前方案的 15-20% 增量更小，因为不需要对 Guardian 做 log-prob 计算）。

收敛轮数预估：与之前分析一致，Guardian 引导提供了更强的学习方向信号（特别是在长尾文化的全错 batch 上），预估 10-15 轮收敛。总训练时长：10-15 轮 x 45-65 分钟/轮 = 7.5-16 小时，一天内可完成。

##### G. 潜在风险与应对

风险一：Guardian 本身在某些题上答错，导致错误引导。应对：质量门控（S_guardian 乘以 R_outcome_guardian），答错时引导项自动归零。在 NormAD 上 HF-CAC 的 Guardian 准确率约 70-80%，约 20-30% 的 prompt 会因为 Guardian 答错而退化为纯 on-policy GRPO，这是合理的安全网。

风险二：lambda 设置过大导致模型过度依赖 Guardian、丧失泛化能力。应对：(a) w_culture 随 hit_rate 提升自动衰减；(b) 可以设全局衰减 lambda_t = lambda * gamma^round（gamma=0.95），让引导随训练递减；(c) 消融实验搜索 lambda in {0.3, 0.5, 0.7}。

风险三：R_on_bar 在全错 batch 上为 0，此时 S_guardian = R_guardian - 0 = R_guardian，引导信号最大。这其实是好事——模型全答不对时最需要 Guardian 帮助。但需要确保此时引导方向正确（即 Guardian 确实答对），否则会产生很大的负面影响。门控一（质量门控）确保了这一点。

风险四：不同文化圈的 Guardian 质量差异大。比如 Western Guardian 准确率 90% 但 Sub-Saharan Guardian 可能只有 60%。应对：可以在训练前统计各文化圈的 Guardian 准确率，对低质量 Guardian 的样本降低 lambda 或直接跳过。但这属于进阶优化，MVP 阶段先用统一的质量门控即可。

##### H. CGM-GRPO 的方法论故事线

整个 CAMAD 框架通过 CGM-GRPO 实现了完美闭环：

第一环（数据生成）：HF-CAC 多智能体协作，利用主场 Guardian 机制生成高质量文化推理轨迹，亲缘度矩阵在 Guardian 失效时提供 fallback 仲裁。

第二环（过程监督）：Culture-Aware PRM 对推理步骤打文化合理性标签（0.9/0.5/0.1），R_total 统一结果正确性和文化合理性。

第三环（强化学习）：CGM-GRPO 在 GRPO 的 advantage 中注入 Guardian 引导信号，由文化难度系数 w_culture 动态调制引导强度，实现「文化难度感知的混合策略强化学习」。

核心创新叙事：不是简单的 Mixed-Policy（LUFFY 已做），而是 Culture-Difficulty-Aware Mixed-Policy。区别在于：LUFFY 对所有样本统一注入 off-policy 信号，我们根据文化难度动态决定「什么时候相信专家、什么时候相信模型自己」。这是文化对齐任务独有的——因为文化知识的难度有明确的结构化先验（数据频率 + 文化迁移距离），而通用推理任务（如数学）的难度缺乏这样的外部先验。

论文贡献可以概括为：(1) 提出 CGM-GRPO，首个文化难度感知的混合策略 GRPO 算法；(2) 设计三因子文化难度系数 w_culture，融合动态模型能力评估与静态文化先验；(3) 证明 Guardian 信号作为 advantage 增强项（而非参与 RLOO baseline）的理论合法性与实验有效性；(4) 在 CAMAD 框架下实现从数据生成到强化学习的完整闭环。

##### I. 与思路一的关系和实验设计

思路一（文化锚点交错回注）和 CGM-GRPO 解决的是不同问题：思路一防止已学知识在 RL 中遗忘（防守），CGM-GRPO 帮助模型在 RL 中首次学会长尾文化知识（进攻）。两者可以独立实现、也可以组合使用。

四组对比实验设计：(1) SFT-only：Cul/sft/train_sft_weighted.py 产出；(2) RL-only：train_grpo_v3.py 不传 --sft_adapter；(3) SFT+RL（串行两阶段）：train_grpo_v3.py 传 --sft_adapter；(4) SFT+RL+CGM（文化难度感知混合策略）：train_grpo_mixed_policy.py 传 --sft_adapter。四组使用相同的数据划分、PRM、base model，只有 Stage 3 策略不同。

可选的消融实验：(a) CGM-GRPO w_culture=hit_only vs hit+rarity vs full 三因子；(b) lambda 搜索 {0.3, 0.5, 0.7}；(c) 有/无质量门控；(d) CGM-GRPO + 思路一交错回注 vs 单独 CGM-GRPO。

### 创新思路三：PRM 信号驱动的 Token 级 SFT 动态加权（CHORD 乘以 Culture-PRM）

这是把 CHORD 的 token 级动态加权机制与 CAMAD 已训练好的 Culture-PRM 结合的方案，可作为单阶段融合的探索性尝试。

机制：在单阶段训练中，总损失为 mu 乘以 SFT损失(phi) 加 (1 减 mu) 乘以 GRPO损失。其中 SFT 部分使用 HF-CAC Guardian 轨迹作为 off-policy 专家数据，但 token 级权重函数 phi 不再用 CHORD 原版的熵来决定，而是用 Culture-PRM 对该 token 所在 step 的打分来决定：PRM 判定为主场确权步（0.9）的 token，加大 SFT 学习权重（这是最值得模仿的文化确权表达）；PRM 判定为文化混淆步（0.1）的 token，几乎不学（避免模仿到毒草）；中立步（0.5）给基准权重。全局系数 mu 随训练递减，从重模仿平滑过渡到重探索。

为什么创新：CHORD 原版用熵衡量模型对该 token 的不确定度，而文化场景下不确定不等于该学——模型可能对一个文化错误表达很确定。改用 Culture-PRM 打分做 token 加权，把该不该学的判断从信息论指标升级为文化语义指标，与 CAMAD 的过程监督理念一脉相承。同时复用了已训练的 PRM，无需额外标注。

风险提示：单阶段融合实现复杂度高于交错训练，且 SFT 与 GRPO 梯度可能冲突。建议在思路一（交错）跑通、确认强基线后，再作为进阶实验对比。

### 创新思路四：难度感知的三路分流（DYPO/GHPO 乘以 文化长尾）

机制：借鉴 DYPO/GHPO 的难度分级，按当前 policy 在每个 prompt 上 G 条 rollout 的答案正确率（R_outcome 命中率）将文化题目分三类，并差异化处理。Hard（G 条全错，多为冷门长尾文化）：模型自己探索不出来，切换为 SFT，直接用 HF-CAC Guardian 轨迹，或 GHPO 式的给出 Guardian 推理前缀作为 hint 让模型续写。Mid（部分对）：标准 GRPO，用 R_total = 0.6 乘以 R_outcome 加 0.4 乘以 Mean(R_process) 拉开组内 advantage，PRM 过程奖励在此发挥最大价值。Easy（基本全对，多为高频西方文化）：纯 GRPO 轻量优化，甚至降低采样数节省算力。

为什么贴合文化场景：文化对齐的难度天然与文化是否冷门长尾强相关，而这正是 CAMAD 主场机制要解决的西方语料主导型错误。难度分流让算力精准投放到长尾文化（Hard），与项目核心动机完全对齐。

### 四个思路的推荐落地顺序

建议按先稳妥强基线、后创新探索的顺序推进：第一步实现思路一（文化遗忘感知交错回注），它复用现有 SFT/GRPO 脚本、风险最低，先把 SFT+RL 防遗忘的强基线立住；第二步实现思路二（CGM-GRPO 文化难度感知混合策略），这是论文创新性最强、与 CAMAD 资产结合最深的方案，作为主推方法；思路三、思路四作为消融与进阶对比，验证 token 级 PRM 加权和难度分流的增量收益。所有方案都应与现有的 SFT-only、RL-only、朴素 SFT-then-RL 三个基线在相同算力预算下做严格公平对比（牢记 SFT-then-RL Outperforms 一文的警示）。

---

## 六、文化对齐方向的 2026 最新专项工作

除了通用混合方法，文化对齐方向自身也有可借鉴的最新进展。CultureRL 类工作把文化规范本身作为奖励信号（norm-driven RL），用一个文化规范判别器替代通用 reward model，与 CAMAD 的 Culture-PRM 思路同源，可借鉴其规范库构建方式。CARB（Cultural Affinity Reward Benchmark）类工作提供了文化奖励的评测基准，可用于校准 CAMAD 的 R_process 标注质量。C-3PO 类工作关注跨语言文化一致性（同一文化问题在不同语言下答案应一致），这对 CAMAD 的多语言鲁棒性评估是有益补充。这些工作整体印证了一个趋势：文化对齐正从单纯的 SFT 灌知识，转向用带文化语义的过程/规范奖励做 RL，而 CAMAD 的 PRM 加 GRPO 设计正处在这条主线上。

---

## 七、关键结论速览

第一，传统 SFT-then-RL 两阶段范式的核心痛点是灾难性遗忘、低效探索、信号互补性未被利用，这在文化对齐场景会放大为 RL 阶段遗忘长尾文化知识。第二，混合训练的理论基础是 SFT 与 RL 在统一策略梯度框架下只是权重函数不同的特例（HPT/UPGE），二者不可解耦（On the Non-decoupling）。第三，六大类混合方法各有适用边界，文化对齐这类需平衡知识精度与泛化、且知识高度长尾的任务，最适合交错训练（防遗忘）与混合策略/条件训练（难度分流）。第四，2026 年最新进展（CHORD、SRFT、ReLIFT、LUFFY、DYPO、GHPO、BRIDGE）共同指向动态、自适应、按样本/token 粒度决定 SFT 与 RL 配比的趋势。第五，必须警惕 SFT-then-RL Outperforms 一文的发现——评估混合方法务必在相同算力下做公平对比，避免把超参红利误认为方法创新。第六，针对 CAMAD 的推荐路线是：以文化遗忘感知交错回注立强基线，以 CGM-GRPO 文化难度感知混合策略作主推创新，把文化属地性从数据生成一路贯穿到 RL 优化，形成方法论闭环。

---

## 附录：核心参考文献

- Intuitive Fine-Tuning (IFT), ACL 2025 Oral, arXiv:2405.11870 — SFT 是 RLHF 的 token 级特例
- Towards a Unified View of LLM Post-Training (HPT/UPGE), arXiv:2509.04419 — 统一策略梯度估计器
- ORPO, arXiv:2403.07691 — 无参考模型的单阶段偏好优化
- SRFT, ICLR 2026, arXiv:2506.19767 — 熵感知的单阶段 SFT+RL 加权
- CHORD, arXiv:2508.11408 — SFT 作为 RL 的动态加权 token 级辅助目标
- GTA, EMNLP 2025 — 生成式教学助手动态示范
- LUFFY, arXiv:2504.14945 — Mixed-Policy GRPO 注入 off-policy 强示范
- ReLIFT, ICLR 2026, arXiv:2506.07527 — RL 与 online SFT 交替学难题
- BRIDGE, arXiv:2509.06948 — SFT/RL 双层优化与参数分离
- GHPO, arXiv:2507.10628 — 难度感知的 hint 注入硬切换
- DYPO, ACL 2026 Findings, arXiv:2604.08926 — 实例级三路难度路由
- UFT, NeurIPS 2025, arXiv:2505.16984 — 单步内在指导下探索
- GRPO is Secretly a PRM, arXiv:2509.21154 — GRPO 内建隐式过程奖励
- SFT Memorizes RL Generalizes, ICML 2025 — SFT 记忆/RL 泛化的信息论分析
- SFT-then-RL Outperforms Mixed-Policy Methods, arXiv:2604.23747 — 公平评估的警示
- On the Non-decoupling of SFT and RL, 2026 — SFT 与 RL 本质不可解耦
