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
Value Head：Linear(hidden_size → 1) + Sigmoid
输入：[Culture Token] + [Question] + [Reasoning Path] + [Answer]
输出：标量分数 s ∈ (0, 1)
```

训练时取最后一个 token 的 hidden state 经过 value head 得到分数，不需要生成文字输出。

### 5.2 训练数据构造

**来源 A：MAS 弱标签（全量，约 1500-2000 对）**

从 70% PRM 训练集的 MAS 生成数据中，构造 pairwise 排序对：

```
同一个 (question, culture) 下：
  chosen   = answer == gold 的路径
  rejected = answer != gold 的路径
```

若一个组内所有路径都正确或都错误，跳过该样本（无对比信号）。

**来源 B：LLM Judge 强标签（抽样，约 300-400 对）**

从来源 A 的数据中随机抽取约 300-500 条路径，使用 Qwen2.5-72B 打文化一致性分：

```
输入：(question, culture, reasoning_path)
输出：1-5 的整数分

Prompt：
"You are evaluating whether the following reasoning reflects
the cultural values and norms of [COUNTRY].
Question: [question]
Reasoning: [reasoning path]
Rate the cultural consistency on a scale of 1-5.
Respond with only a number from 1 to 5."
```

归一化到 0-1 后，同一题内 score 差值 > 1 的两条路径构成 pairwise 对（chosen = 高分，rejected = 低分）。

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

**收敛标准**：验证集 Pairwise Accuracy > 68%，连续 2 个 epoch 不提升即停止。

**精度不足时的调整**：
- 增加 LLM Judge 强标签比例
- 降低 learning rate（1e-5 → 5e-6）
- 检查文化分布是否均匀（某些文化样本极少会导致该文化的 reward 失准）

**Culture Sensitivity 验证**：同一 question，不同 culture 输入时，PRM 对各文化最高分路径的内容应有实质差异。

### 5.6 Reward 误差控制

PRM 本身存在误差，GRPO 阶段误差会进一步叠加，采用以下机制控制：

1. **R_ans 作为锚点**：可验证奖励占主导（α=1.0），PRM 仅作辅助（β=0.3），主信号不依赖 PRM
2. **Reward clip**：`clip(prm_score, 0.1, 0.9)`，避免极端分数主导梯度
3. **KL 惩罚**：`init_kl_coef` 约束 policy 偏离 reference 的上界，间接限制 reward hacking
4. **验证集监控**：每 5 轮评估 val_accuracy，PRM 误差导致 policy 跑偏时 val_accuracy 会下降，及时早停

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

## 11. 关键设计决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Agent 粒度 | 国家级 | 与数据集标注粒度对齐，避免信息损失 |
| MAS 方法 | LLM Debate | 生成多样路径同时保持文化内一致性 |
| Student 模型 | Llama-3.1-8B / Qwen2.5-7B | 分别评测，与 base 对比验证蒸馏效果 |
| PRM 规模 | Qwen3-0.6B | 文化 reward 判断是语义分类，不需要大模型；节省显存 |
| Reward 设计 | R_ans(α=1.0) + R_cultural(β=0.3) | R_ans 作锚点，PRM 误差不主导梯度 |
| Reward clip | clip(0.1, 0.9) | 避免极端分数，减轻误差叠加 |
| PRM 训练数据 | MAS 弱标签 + LLM Judge 强标签 | 弱标签保证数量，强标签保证质量 |
| 数据分割 | 按 question 分割 | 防止 PRM 和 GRPO 数据泄露 |
| 图结构 | 不引入 | 无实质计算作用，增加实现复杂度 |
| 两类数据 | 混合训练，不同 prompt 格式 | 共享文化知识，保留任务差异 |
| 评估间隔 | 每 5 轮（非 10 轮） | 文化任务收敛快，需要更密集的早停检测 |
