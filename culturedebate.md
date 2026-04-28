# CultureDebate: Distilling Multi-Agent Cultural Intelligence into a Single LLM

## 1. 问题定义

### 1.1 核心目标

将多智能体系统在文化对齐任务上的集体推理能力蒸馏进单个语言模型的权重，使单体模型在文化条件下的决策质量接近多智能体系统，同时保持推理效率。

### 1.2 任务统一建模

所有文化对齐数据统一建模为：

```
y ~ P(y | x, c)，中间经过显式或隐式推理路径 r
```

- `x`：问题（scenario / question）
- `c`：文化标识（country-level，如 Egypt、Japan、Germany 等）
- `r`：推理路径（显式或隐式）
- `y`：答案（yes/no 或选项 1-4）

### 1.3 数据集说明

| 数据集 | 类型 | 推理结构 | 文化粒度 | 答案格式 |
|--------|------|----------|----------|----------|
| NormAD | etiquette 判断 | 显式（Value→Rule→Judgment） | 国家级（75国） | yes/no |
| CultureLLM | 价值观调查 | 隐式（偏好分布） | 国家级 | 选项 1-4 |

两类数据统一视角：文化条件下的决策生成问题。etiquette 任务学习显式规则推理，survey 任务学习隐式偏好建模，共享同一模型，训练时使用不同 prompt 格式和 loss 权重。

---

## 2. 数据集划分

数据规模：1000 questions × 20 cultures = 20000 个 (x, c) 样本。

所有数据按 **question 维度**（而非路径维度）划分，防止 PRM 训练与 GRPO 训练之间的数据泄露。同一 question 的所有路径只能出现在一个子集中。

```
原始数据集（NormAD + CultureLLM）
        ↓ 按 question 随机划分
├── 70% → PRM 训练集（用于训练 Reward 模型）
├── 20% → GRPO 训练集（用于强化学习微调）
└── 10% → 评估集（不参与任何训练）
```

---

## 3. 多智能体数据生成

### 3.1 Agent 设计

每个 Agent 代表一种具体文化（国家级），system prompt：

```
You are a cultural expert representing [COUNTRY].
When answering questions, reason strictly from the perspective of
[COUNTRY]'s cultural values, social norms, and behavioral tendencies.
Consider factors such as family structure, social hierarchy,
collectivism/individualism, religious influence, and historical context
specific to [COUNTRY].
```

文化粒度采用双层设计：生成阶段使用国家级 agent（细粒度，与数据集对齐）；训练输入使用国家级 culture token（`[Egypt]`、`[Japan]` 等）。

### 3.2 多路径采样策略

对每个样本 `(x, c)`，让同一文化 agent 生成 k=5 条推理路径：

```
"Generate a reasoning path that reflects [COUNTRY]'s cultural perspective.
Your reasoning should be internally consistent with this culture's values,
but may emphasize different aspects such as family bonds, social hierarchy,
religious principles, or community norms."
```

目的：保证 intra-cultural variation（路径多样但文化一致），为 GRPO 提供足够的对比信号。

对于 etiquette 数据（NormAD），推理路径格式为：
```
Cultural Value → Social Rule → Situational Judgment → Answer
```

对于 survey 数据（CultureLLM），推理路径格式为：
```
Cultural Background → Preference Tendency → Answer
```

### 3.3 多智能体方法选择

采用 LLM Debate 作为主要 MAS 方法：多个同文化 agent 独立生成初始答案，互相看到其他 agent 的推理后更新自己的答案，最终 aggregator 汇总生成最终答案。

每条样本最终生成数据格式：

```json
{
  "id": "normad_001",
  "dataset": "NormAD",
  "culture": "Egypt",
  "question": "...",
  "paths": [
    {"id": 1, "reasoning": "In Egyptian culture, family honor is paramount...", "answer": "yes"},
    {"id": 2, "reasoning": "Egyptian social norms emphasize respect for elders...", "answer": "yes"},
    {"id": 3, "reasoning": "From a religious perspective common in Egypt...", "answer": "no"}
  ],
  "gold": "yes",
  "labels": [true, true, false]
}
```

---

## 4. Reward 信号设计

```
R_total = α * R_ans + β * R_cultural
```

初始超参数：`α=1.0, β=0.3`。R_ans 作为主信号锚点，R_cultural 作为辅助信号，避免 PRM 误差主导梯度。

### 4.1 R_ans：答案正确性（可验证奖励）

直接规则计算，无需模型：

```
R_ans = 1  if predicted_answer == gold
R_ans = 0  otherwise
```

对 survey 类数据（CultureLLM），gold 是众数，放宽为：

```
R_ans = 1  if predicted_answer in top_2_answers（按频率排序）
R_ans = 0  otherwise
```

### 4.2 R_cultural：文化一致性（过程奖励，PRM 输出）

由训练好的 Process Reward Model 对推理路径打分，输出 0-1 之间的标量，并做 clip 处理：

```
R_cultural = clip(prm_score, 0.1, 0.9)
```

clip 的目的：避免极端分数主导梯度，减轻 PRM 误差对 GRPO 的影响。

---

## 5. Process Reward Model（PRM）训练

### 5.1 模型架构

文化任务的 reward 判断本质是语义分类问题，不需要复杂推理能力，使用小规模模型即可：

```
Base Model：Qwen3-0.6B（轻量，显存占用约 1.2GB/卡）
Reward Head：Linear(hidden_size → 1) + Sigmoid
输入：[Culture Token] + [Question] + [Reasoning Path] + [Answer]
输出：标量分数 s ∈ (0, 1)
```

训练时取最后一个 token 的 hidden state 经过 reward head 得到分数，不需要生成文字输出。

**关于 reward head 的结构说明**：reward head 和 PPO 中 critic model 的 value head 结构相同（均为 `Linear(hidden_size → 1)`），但语义完全不同。PPO critic 的 value head 输出的是当前状态的期望累积 reward V(s)，用于计算 advantage；reward model 的 reward head 输出的是对一条完整路径的质量评分 r，是训练信号而非状态估计。GRPO 不存在 critic model（用 RLOO baseline 替代了 advantage 估计），这里的 reward head 仅属于 reward model，两者不应混淆。

reward model 加 reward head 而非直接生成文字打分（LLM-as-a-Judge 风格）的原因：GRPO 在线采样阶段每轮需要对约 40000 条路径批量打分，生成式打分速度慢且输出不稳定；reward head 输出连续标量，推理快，与 Bradley-Terry loss 训练范式直接兼容。

### 5.2 训练数据构造

**来源 A：MAS 弱标签（全量，约 1500-2000 对）**

从 70% PRM 训练集的 MAS 生成数据中，构造 pairwise 排序对：

```
同一个 (question, culture) 下：
  chosen   = answer == gold 的路径
  rejected = answer != gold 的路径
```

若一个组内所有路径都正确或都错误，跳过该样本（无对比信号）。

**来源 B：细粒度监督标签（抽样，约 300-500 条路径）**

从来源 A 的数据中随机抽取约 300-500 条路径，对推理路径的文化一致性进行打分（0.1-0.9 之间的连续值）。打分来源有两种方式，可单独使用或结合使用：

**方式一：人类评审员标注**

由 5 位具有对应文化背景的评审员对每条路径独立打分，取加权平均作为最终分数。

优点：感知细粒度文化差异能力强（语气、隐含价值观等 LLM 难以捕捉的信号）；不存在 LLM 的西方文化偏见；标注结果可解释性强。

缺点：成本高、速度慢；需要每种文化都有对应母语背景的评审员，20 种文化难以全覆盖；评审员之间一致性差（Inter-Annotator Agreement 通常 < 0.7），需要多数投票或加权平均处理分歧。

适用场景：少量高质量标注（100-200 条），用于验证 LLM Judge 标注质量，或作为 PRM 验证集的参考标签。

**方式二：LLM-as-a-Judge**

使用 Qwen2.5-72B 或同等规模模型批量打分：

```
输入：(question, culture, reasoning_path)
输出：0.1-0.9 之间的连续分数

Prompt：
"You are evaluating whether the following reasoning reflects
the cultural values and norms of [COUNTRY].
Question: [question]
Reasoning: [reasoning path]
Rate the cultural consistency as a decimal between 0.1 and 0.9,
where 0.9 = perfectly reflects [COUNTRY]'s cultural values,
0.1 = does not reflect [COUNTRY]'s cultural values at all.
Respond with only a decimal number."
```

优点：成本低、速度快，可覆盖全量数据；一致性高；与 MAS 生成 pipeline 无缝衔接。

缺点：存在西方文化偏见（对非西方文化的判断准确性低于西方文化）；对细粒度文化差异感知能力弱于领域专家；评分分布可能集中于中间值。

适用场景：大量弱监督标注（300-500 条），作为 PRM 训练的主要强标签来源。

**两种方式的结合使用**

推荐 LLM Judge 覆盖全量，人类评审员抽样验证（50-100 条），计算两者的 Spearman 相关系数：
- 相关系数 > 0.6：LLM Judge 可信，继续使用
- 相关系数 < 0.5：增加人类标注比例或更换 Judge 模型

归一化到 0-1 后，同一题内 score 差值 > 0.2 的两条路径构成 pairwise 对（chosen = 高分，rejected = 低分）。

最终 PRM 训练数据：约 1800-2400 个 pairwise 对（弱标签为主，强标签补充质量）。

### 5.3 训练方式

Bradley-Terry ranking loss：

```
Loss = -log(σ(s_chosen - s_rejected))
```

### 5.4 训练配置（4卡 A100）

| 参数 | 值 |
|------|----|
| epochs | 3-5 |
| learning rate | 1e-5 |
| batch size | 32 pairs（每卡 8 对） |
| max sequence length | 1024 tokens |
| optimizer | AdamW |
| warmup ratio | 0.05 |
| bf16 | True |
| gradient checkpointing | True |

预计训练时间：约 15-30 分钟。

### 5.5 收敛标准与质量验证

**PRM 精度目标**：验证集 Pairwise Accuracy 在 65%-70% 即可为 GRPO 提供稳定对比信号，目标设为 68%。

PRM 的作用是辅助 reward 信号，不需要追求高精度：
- 精度过低（< 65%）：对比信号不可靠，GRPO 梯度方向错误
- 精度合理（65%-70%）：提供稳定辅助信号，配合 R_ans 主信号足够
- 训练集与验证集 Pairwise Accuracy 差距 > 10%：PRM 过拟合，对 GRPO 在线采样的新路径（分布外样本）打分不准，导致 reward hacking

**收敛标准**：验证集 Pairwise Accuracy > 68%，连续 2 个 epoch 不提升即停止。

**精度不足时的调整**：
- 增加 LLM Judge 强标签比例
- 调整训练超参数（降低 lr 至 5e-6，或调整 batch size）
- 检查文化样本分布，确保每个文化都有足够数据（某些文化样本极少会导致该文化的 reward 失准）

**Culture Sensitivity 验证**：同一 question，不同 culture 输入时，PRM 对各文化最高分路径的内容应有实质差异。

### 5.6 Reward 信号设计原理

**R_ans（主信号）** 完全依赖 gold label，是可验证奖励：预测答案与 gold 匹配则 R_ans=1，否则为 0。梯度可靠，训练方向明确，是 GRPO 的主导信号。

**R_cultural（辅助信号）** 依赖 PRM 对推理路径的评分，判断路径是否符合文化特征。PRM 的训练数据来自两类监督信号：answer correctness 弱标签（gold 路径 vs 非 gold 路径）和 LLM Judge 细粒度监督信号（文化一致性评分，非 ground truth，是另一模型的判断）。权重较小（β=0.3），避免 PRM 偏差主导训练。

```
R_total = α * R_ans + β * R_cultural
```

训练方向主要靠 ground truth 保证正确性，文化风格靠 PRM 细化差异。

### 5.7 Reward 风险与缓解策略

**风险1：PRM 误差累积**

PRM 是小模型预测的辅助 reward，本身存在偏差。在 GRPO 中，R_cultural 的误差信号会被梯度放大，可能引导模型偏离正确文化表达，尤其当 β 过大或 reward clip 设置不合理时。风险体现为不同文化下的 reasoning 路径趋同，或 val_accuracy 下降。

缓解策略：保持 α > β；clip(prm_score, 0.1, 0.9)；KL 惩罚约束 policy 偏离 reference。

**风险2：数据量小导致过拟合**

GRPO 训练数据仅 4000 个 prompt，文化任务收敛快，模型容易在少量样本上过拟合，val_accuracy 不再反映真实泛化能力。

缓解策略：每 5 轮评估验证集；train_acc - val_acc > 15% 时立即早停；保存所有 checkpoint，取 val_accuracy 最高点。

**风险3：Reward scale 不一致**

R_ans 是离散 0/1，R_cultural 是连续 0.1-0.9，梯度 scale 不一致。若不调节权重，policy 更新可能偏向辅助信号，导致 reward hacking 或训练不稳定。

缓解策略：α=1.0，β=0.3 的经验设置；reward normalization 或 clip。

**风险4：GRPO 收敛依赖 MAS 数据质量**

MAS 生成路径的多样性和文化一致性决定了 R_cultural 的有效性。若路径不够多样或文化倾向不明显，GRPO 无法学到细粒度文化差异。

缓解策略：k=5 条路径保证 intra-cultural variation；使用 aggregator 汇总提高路径 gold label 准确性。

---

## 6. GRPO 强化学习微调

### 6.1 Student 模型

分别在两个基座上训练并评测，与未蒸馏的 base 模型对比，验证蒸馏效果：

- Llama-3.1-8B
- Qwen2.5-7B

```
输入格式：[Culture Token] + [Question]
输出格式：[Reasoning Path] + [Answer]
```

### 6.2 在线采样（On-policy）

GRPO 采用在线采样，每轮迭代都用当前 policy 模型重新生成：

```
对每个 prompt (x, c)：
  → 当前 policy 模型生成 n=10 条路径
  → 计算每条路径的 R_total
  → 用 reward 计算 advantage（RLOO baseline）
  → 更新 policy 参数
  → 下一轮用更新后的模型重新采样
```

### 6.3 训练配置（4卡 A100）

显存分析：
- policy（8B，ZeRO-2）：约 35GB × 4 卡分片
- reference model（8B，冻结）：约 16GB，开启 cpu_offload
- PRM（0.6B，冻结）：约 1.2GB
- 4 卡 A100（320GB 总）可以承载，需开启 `cpu_offload` for reference model

| 参数 | 值 | 说明 |
|------|----|------|
| `n_samples_per_prompt` | 10 | group size |
| `advantage_estimator` | rloo | RLOO baseline |
| `reward_mode` | PRMVR | PRM + Verifiable Reward 组合 |
| `verifiable_reward_coef` | 1.0 | R_ans 权重 |
| `init_kl_coef` | 0.001 | KL 惩罚 |
| `temperature` | 0.7 | 采样温度 |
| `micro_rollout_batch_size` | 2 | 显存限制 |
| `micro_train_batch_size` | 2 | 显存限制 |
| `max sequence length` | 512 tokens | 文化推理路径较短 |
| `optimizer` | AdamW | |
| `actor_learning_rate` | 5e-7 | |
| `bf16` | True | |
| `gradient_checkpointing` | True | |
| `zero_stage` | 2 | |

### 6.4 两类数据的差异处理

etiquette 数据（NormAD）和 survey 数据（CultureLLM）混合训练，按 1:1 比例混合，在同一 batch 内可以混合出现：

- etiquette：prompt 要求显式输出 `Value → Rule → Judgment → Answer` 格式
- survey：prompt 要求输出 `Cultural Background → Tendency → Answer` 格式

---

## 7. 收敛分析

### 7.1 文化任务收敛快的原因

文化任务相比数学、医疗等推理任务收敛更快，原因如下：

**reward 稀疏性低**：文化任务答案空间有限（yes/no 或 1-4），随机猜测正确率 25-50%。数学任务中 n=10 条采样路径可能全错，group 内 reward 全为 0，梯度消失，白跑一轮；文化任务中几乎每组都有正有负，每轮都有有效梯度。

**预训练知识起点高**：预训练模型已见过大量文化相关文本，对各国文化价值观有隐式表示，GRPO 只需激活和强化已有知识，而非从头学习推理结构。

**数据规模小**：GRPO 训练数据仅 4000 个 prompt，模型很快遍历所有样本。

### 7.2 收敛轮次与时间估算

**PRM 训练**：约 186 steps，15-30 分钟。

**GRPO 训练**：

```
GRPO 训练数据：20000 × 20% = 4000 个 (x,c) prompt
每轮采样：4000 × 10 条路径 = 40000 条序列

单轮耗时（8B，A100，序列长 512）：
  采样阶段：约 20-30 分钟
  训练阶段：约 10-15 分钟
  单轮合计：约 30-45 分钟

预计收敛轮次：10-20 轮
总训练时间：约 5-15 小时
```

### 7.3 收敛指标

**训练过程监控指标**：

| 指标 | 含义 | 收敛信号 |
|------|------|----------|
| `reward/mean` | 每轮平均 reward | 上升后趋于平稳 |
| `reward/std` | reward 方差 | 下降（group 内差异缩小） |
| `kl_divergence` | 与初始模型的偏离 | 稳定在 0.05-0.2，不持续增大 |
| `policy_loss` | policy 梯度损失 | 震荡收窄 |
| `response_length` | 生成长度均值 | 稳定，不持续增长 |

**关键警报信号**：

```
reward/std → 0          # group 内 reward 无差异，训练退化
kl_divergence > 0.5     # 模型偏离过远，降低 learning rate
reward/mean 不上升       # reward 设计有问题或 lr 过小
response_length 暴增     # 模型用长度刷 reward，加长度惩罚
train_acc - val_acc > 15% # 过拟合，立即早停
```

**验证集指标（每 5 轮评估一次，文化任务收敛快，评估间隔应短于数学任务）**：

| 指标 | 说明 |
|------|------|
| `val_accuracy` | 评估集答案准确率，最终目标指标 |
| `culture_sensitivity` | 同一 question 不同 culture 下输出是否有实质差异 |

**GRPO 收敛判断**：val_accuracy 连续 3 次评估（15 轮）不再提升即停止。

**PRM 收敛判断**：验证集 Pairwise Accuracy > 68%，连续 2 个 epoch 不提升即停止。

---

## 8. 完整 Pipeline

```
Step 1: 数据准备
  NormAD + CultureLLM（1000 questions × 20 cultures = 20000 样本）
  → 按 question 划分：PRM 训练集(70%) / GRPO 训练集(20%) / 评估集(10%)

Step 2: MAS 数据生成
  对 PRM 训练集 + GRPO 训练集的所有样本：
  → 每个样本用 LLM Debate（国家级 agent）生成 k=5 条推理路径
  → 自动打 answer correctness 标签（R_ans）
  → 输出：带多路径和标签的 jsonl 文件

Step 3: LLM Judge 打分
  从 PRM 训练集的 MAS 数据中抽取约 300-500 条路径
  → 用 Qwen2.5-72B 打文化一致性分（1-5）
  → 归一化后与 Step 2 数据合并，构造 pairwise 对

Step 4: PRM 训练
  输入：Step 2+3 的 pairwise 数据（约 1800-2400 对）
  模型：Qwen3-0.6B + value head
  损失：Bradley-Terry ranking loss
  时间：约 15-30 分钟
  输出：文化一致性 reward 模型

Step 5: GRPO 训练
  输入：GRPO 训练集（4000 个 prompt）
  Policy：Llama-3.1-8B 或 Qwen2.5-7B（分别训练评测）
  Reward：R_ans（规则，α=1.0）+ R_cultural（PRM 冻结，β=0.3）
  时间：约 5-15 小时（预计 10-20 轮收敛）
  输出：文化对齐单体模型

Step 6: 评估
  在评估集上对比 base 模型与蒸馏模型
```

---

## 9. 评估设计

### 9.1 基础指标

Accuracy：预测答案与 gold label 的匹配率，分数据集、分文化报告。

### 9.2 文化敏感性（核心指标）

测试同一 question，不同 culture 输入时，模型输出是否有实质差异：

```
Culture Sensitivity Score = 不同文化下答案分布的 KL 散度均值
```

目标：同一问题在不同文化下产生不同答案分布，而不是趋同。

### 9.3 推理质量

- Reasoning Coherence：推理路径与最终答案是否一致（LLM Judge 评估）
- Cultural Grounding：推理路径中是否出现该文化的具体价值观关键词

### 9.4 消融实验

| 实验组 | 说明 |
|--------|------|
| Base | 基础模型直接推理，无文化 conditioning |
| + Culture Prompt | 加 culture token，无训练 |
| + SFT only | 只用正确路径做 SFT |
| + GRPO (R_ans only) | 只用答案正确性做 reward |
| + GRPO (R_ans + R_cultural) | 完整方案 |
| MAS Oracle | 多智能体系统直接推理的上界 |

---

## 10. 目录结构规划

```
AgentArk/
├── culture/                          # 新增模块，不修改原有文件
│   ├── __init__.py
│   ├── data/
│   │   ├── normad/                   # NormAD 原始数据
│   │   ├── culturellm/               # CultureLLM 原始数据
│   │   └── splits/                   # 划分后的训练/评估集
│   ├── generate/
│   │   ├── culture_inference.py      # MAS 生成文化推理路径
│   │   ├── culture_label.py          # answer correctness 自动标注
│   │   └── llm_judge.py              # LLM Judge 文化一致性打分
│   ├── prm/
│   │   ├── build_prm_data.py         # 构造 pairwise 训练数据
│   │   └── train_prm.py              # PRM 训练
│   ├── grpo/
│   │   └── train_culture_grpo.py     # GRPO 训练入口
│   ├── eval/
│   │   ├── accuracy.py               # 答案准确率
│   │   ├── culture_sensitivity.py    # 文化敏感性评估
│   │   └── reasoning_quality.py      # 推理质量评估
│   └── methods/
│       └── culture_debate.py         # 文化版 LLM Debate agent
├── culturedebate.md                  # 本文档
└── ...                               # 原有文件不变
```

---

## 11. 实战工程细节

### 11.1 答案格式解析

模型生成的答案格式不可控（如输出"Yes"而非"yes"，或"Option A"而非"1"），导致 R_ans 全为 0，GRPO 无梯度，训练静默失败。

处理方式：在 reward 计算前加鲁棒的答案抽取逻辑，用正则匹配多种格式变体；对无法解析的输出记录日志并赋予 R_ans=0 而非跳过，保留样本参与训练。

### 11.2 文化样本分布不均衡

20 种文化在数据集中样本数差异可能较大，导致 PRM 对低频文化打分不准，GRPO 对低频文化的梯度信号弱，最终模型在低频文化上表现差。

处理方式：PRM 训练时按文化做 weighted sampling，确保每种文化在每个 epoch 内都有足够样本；GRPO 训练时同样做文化均衡采样；评估时分文化报告 accuracy，不只看总体均值。

### 11.3 推理路径长度差异

不同文化、不同问题的推理路径长度差异大，统一 padding 到最长序列会浪费显存，降低有效 batch size。

处理方式：按序列长度分桶（length bucketing），同一 batch 内的序列长度相近；设置 max_sequence_length=512，超长路径截断并记录截断率，截断率 > 10% 时需要增大 max_sequence_length。

### 11.4 Reference Model 显存管理

GRPO 需要同时保留 policy model 和 reference model，reference 冻结但仍占显存。4 卡 A100 下 8B policy + 8B reference 共约 64GB 权重，加上优化器和激活值会超出预算。

处理方式：reference model 开启 `cpu_offload`，在需要计算 KL 时才加载到 GPU；或使用 ZeRO-3 分片 reference model（但会增加通信开销）。

### 11.5 Checkpoint 管理与最优模型选取

文化任务容易过拟合，训练后期 reward/mean 持续上升但 val_accuracy 不再提升甚至下降，若只保存最后一个 checkpoint 会丢失最优模型。

处理方式：保存所有 checkpoint（或至少每 5 轮保存一次）；最终取 val_accuracy 最高的 checkpoint 作为最终模型；同时记录对应的训练轮次，便于分析过拟合发生的时间点。

### 11.6 MAS 生成阶段的工程问题

20000 个样本 × 5 条路径 = 100000 次 LLM 调用，若使用外部 API 会面临限速和中断风险。

处理方式：实现断点续传（记录已完成的样本 id，重启时跳过）；做并发控制（参考 agentark 的 `reserve_unprocessed_queries` 逻辑）；对每次调用加重试机制（最多 3 次，失败后记录 error 而非丢弃样本）。

### 11.7 PRM 打分的批量推理效率

GRPO 在线采样阶段，每轮生成 40000 条路径后需要用 PRM 批量打分，若逐条调用 PRM 会成为瓶颈。

处理方式：PRM 使用 vLLM 或直接批量前向推理（0.6B 模型批量推理极快）；PRM 推理时关闭梯度（`torch.no_grad()`）；将 PRM 常驻 GPU，不在每轮重新加载。

### 11.8 多轮训练的随机性控制

文化任务数据量小，不同随机种子下收敛轮次可能差异较大（如一次 10 轮、一次 30 轮），单次实验结论不可靠。

处理方式：至少跑 3 个不同随机种子；报告均值和标准差；若标准差过大（> 5% accuracy），说明训练不稳定，需要调整 learning rate 或 KL 系数。

---

## 12. 关键设计决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Agent 粒度 | 国家级 | 与数据集标注粒度对齐，避免信息损失 |
| MAS 方法 | LLM Debate | 生成多样路径同时保持文化内一致性 |
| Student 模型 | Llama-3.1-8B / Qwen2.5-7B | 分别评测，与 base 对比验证蒸馏效果 |
| PRM 规模 | Qwen3-0.6B | 文化 reward 判断是语义分类，不需要大模型；节省显存 |
| Reward 设计 | R_ans(α=1.0) + R_cultural(β=0.3) | R_ans 作锚点，PRM 误差不主导梯度 |
| Reward clip | clip(0.1, 0.9) | 避免极端分数，减轻误差叠加 |
| PRM 训练数据 | MAS 弱标签 + 人类/LLM Judge 强标签 | 弱标签保证数量，强标签补充质量；LLM Judge 为主，人类标注验证可信度 |
| 数据分割 | 按 question 分割 | 防止 PRM 和 GRPO 数据泄露 |
| 图结构 | 不引入 | 无实质计算作用，增加实现复杂度 |
| 两类数据 | 混合训练，不同 prompt 格式 | 共享文化知识，保留任务差异 |
| 评估间隔 | 每 5 轮（非 10 轮） | 文化任务收敛快，需要更密集的早停检测 |
