# CultureDebate: Distilling Multi-Agent Cultural Intelligence into a Single LLM

## 1. 问题定义

### 1.1 核心目标


将多智能体系统在文化对齐任务上的集体推理能力，蒸馏进单个语言模型的权重，使单体模型在文化条件下的决策质量接近多智能体系统，同时保持推理效率。

### 1.2 任务统一建模

所有文化对齐数据统一建模为：

```
y ~ P(y | x, c)，中间经过显式或隐式推理路径 r
```

其中：
- `x`：问题（scenario / question）
- `c`：文化标识（country-level，如 Egypt、Japan、Germany 等）
- `r`：推理路径（显式或隐式）
- `y`：答案（yes/no 或选项 1-4）

### 1.3 数据集说明

| 数据集 | 类型 | 推理结构 | 文化粒度 | 答案格式 |
|--------|------|----------|----------|----------|
| NormAD | etiquette 判断 | 显式（Value→Rule→Judgment） | 国家级（75国） | yes/no |
| CultureLLM | 价值观调查 | 隐式（偏好分布） | 国家级 | 选项 1-4 |

两类数据统一视角：**文化条件下的决策生成问题**。etiquette 任务学习显式规则推理，survey 任务学习隐式偏好建模，共享同一模型但训练时使用不同 prompt 格式和 loss 权重。

---

## 2. 数据集划分

所有数据按 **question 维度**（而非路径维度）进行划分，防止 PRM 训练与 GRPO 训练之间的数据泄露。

```
原始数据集（NormAD + CultureLLM）
        ↓ 按 question 随机划分
├── 70% → PRM 训练集（用于训练 Reward 模型）
├── 20% → GRPO 训练集（用于强化学习微调）
└── 10% → 评估集（不参与任何训练）
```

**关键约束**：同一个 question 的所有路径只能出现在一个子集中，不允许跨集出现。

---

## 3. 多智能体数据生成

### 3.1 Agent 设计

每个 Agent 代表一种具体文化（国家级），使用如下 system prompt：

```
You are a cultural expert representing [COUNTRY].
When answering questions, reason strictly from the perspective of 
[COUNTRY]'s cultural values, social norms, and behavioral tendencies.
Consider factors such as family structure, social hierarchy, 
collectivism/individualism, religious influence, and historical context 
specific to [COUNTRY].
```

文化粒度采用**双层设计**：
- 生成阶段：使用国家级 agent（细粒度，与数据集对齐）
- 训练输入：culture token 使用国家级标识（`[Egypt]`、`[Japan]` 等）

### 3.2 多路径采样策略

对每个样本 `(x, c)`，让同一文化 agent 生成 **k=5 条**推理路径：

```
Prompt 约束：
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

采用 **LLM Debate** 作为主要 MAS 方法：
- 多个同文化 agent 独立生成初始答案
- 互相看到其他 agent 的推理后更新自己的答案
- 最终 aggregator 汇总生成最终答案

每条样本最终生成数据格式：

```json
{
  "id": "normad_001",
  "dataset": "NormAD",
  "culture": "Egypt",
  "question": "...",
  "paths": [
    {
      "id": 1,
      "reasoning": "In Egyptian culture, family honor is paramount...",
      "answer": "yes"
    },
    {
      "id": 2,
      "reasoning": "Egyptian social norms emphasize respect for elders...",
      "answer": "yes"
    },
    {
      "id": 3,
      "reasoning": "From a religious perspective common in Egypt...",
      "answer": "no"
    }
  ],
  "gold": "yes",
  "labels": [true, true, false]
}
```

---

## 4. Reward 信号设计

文化对齐任务的 reward 由两个信号组合：

```
R_total = α * R_ans + β * R_cultural
```

初始超参数：`α=1.0, β=0.5`，后续通过验证集调整。

### 4.1 R_ans：答案正确性（可验证奖励）

直接规则计算，无需模型：

```
R_ans = 1  if predicted_answer == gold
R_ans = 0  otherwise
```

对 survey 类数据（CultureLLM），gold 是众数，可放宽：

```
R_ans = 1  if predicted_answer in top_2_answers（按频率排序）
R_ans = 0  otherwise
```

### 4.2 R_cultural：文化一致性（过程奖励，PRM 输出）

由训练好的 Process Reward Model 对推理路径打分，输出 0-1 之间的标量。

训练数据来源见第 5 节。

---

## 5. Process Reward Model（PRM）训练

### 5.1 训练数据构造

**来源 A：MAS 生成轨迹 + answer correctness 弱标签（大量）**

从 70% PRM 训练集的 MAS 生成数据中，构造 pairwise 排序对：

```
对每个 (question, culture) 组：
  chosen  = answer 正确的推理路径
  rejected = answer 错误的推理路径
```

若一个组内所有路径都正确或都错误，则跳过该样本（无对比信号）。

**来源 B：LLM Judge 文化一致性打分（少量强标签）**

从来源 A 的数据中随机抽取约 **2000-3000 条**路径，使用强模型（Qwen2.5-72B-Instruct）打文化一致性分：

```
Prompt：
"You are evaluating whether the following reasoning reflects 
the cultural values and norms of [COUNTRY].

Question: [question]
Reasoning: [reasoning path]

Rate the cultural consistency on a scale of 1-5:
5 = Perfectly reflects [COUNTRY]'s cultural values
3 = Partially reflects, some aspects missing
1 = Does not reflect [COUNTRY]'s cultural values at all

Respond with only a number from 1 to 5."
```

将 1-5 分归一化到 0-1，与来源 A 的 binary label 混合训练。

### 5.2 PRM 模型架构

```
Base Model：Qwen2.5-7B-Instruct（与后续 GRPO 训练的 policy 同规模）
Value Head：Linear(hidden_size → 1) + Sigmoid
输入：[Culture Token] + [Question] + [Reasoning Path]
输出：标量分数 ∈ (0, 1)
```

### 5.3 训练方式

使用 Bradley-Terry 排序损失：

```
Loss = -log(σ(r_chosen - r_rejected))
```

训练超参数参考 agentark 的 `prm/finetune2.py` 配置：
- epochs：3
- learning rate：1e-4
- batch size：64
- bf16：True
- gradient checkpointing：True

### 5.4 PRM 质量验证

在评估集上验证 PRM 的两个指标：
1. **Pairwise Accuracy**：PRM 对 chosen/rejected 对的排序准确率（目标 > 70%）
2. **Culture Sensitivity**：同一 question，不同 culture 的 PRM 分数是否有显著差异（目标：不同文化的高分路径内容有实质差别）

---

## 6. GRPO 强化学习微调

### 6.1 Policy 模型

```
Base Model：Qwen2.5-7B-Instruct（或更小的 0.6B/1.7B 用于消融）
输入格式：[Culture Token] + [Question]
输出格式：[Reasoning Path] + [Answer]
```

### 6.2 训练数据

使用 20% GRPO 训练集，对每个样本用 policy 模型在线采样 **n=8 条**路径，构成 group：

```json
{
  "question": "...",
  "culture": "Egypt",
  "sampled_paths": ["path_1", "path_2", ..., "path_8"],
  "rewards": [R_total_1, R_total_2, ..., R_total_8]
}
```

其中 `R_total = R_ans + 0.5 * R_cultural(PRM)`，PRM 权重冻结。

### 6.3 GRPO 配置

关键参数参考 agentark 的 GRPO 实现：

| 参数 | 值 | 说明 |
|------|----|------|
| `n_samples_per_prompt` | 8 | group size，保证足够对比信号 |
| `advantage_estimator` | rloo | RLOO baseline |
| `reward_mode` | PRMVR | PRM + Verifiable Reward 组合 |
| `verifiable_reward_coef` | 1.0 | R_ans 权重 |
| `init_kl_coef` | 0.001 | KL 惩罚，防止偏离太远 |
| `temperature` | 0.7 | 采样温度（文化任务比数学任务更需要多样性） |

### 6.4 两类数据的差异处理

etiquette 数据（NormAD）和 survey 数据（CultureLLM）混合训练时：

- etiquette：prompt 要求显式输出 `Value → Rule → Judgment → Answer` 格式
- survey：prompt 要求输出 `Cultural Background → Tendency → Answer` 格式
- 两类数据按 1:1 比例混合，在同一 batch 内可以混合出现

---

## 7. 完整 Pipeline

```
Step 1: 数据准备
  NormAD + CultureLLM
  → 按 question 划分为 PRM训练集(70%) / GRPO训练集(20%) / 评估集(10%)

Step 2: MAS 数据生成
  对 PRM训练集 + GRPO训练集 的所有样本：
  → 每个样本用 LLM Debate（国家级 agent）生成 k=5 条推理路径
  → 自动打 answer correctness 标签（R_ans）
  → 输出：带多路径和标签的 jsonl 文件

Step 3: LLM Judge 打分（可选强标签）
  从 PRM训练集 的 MAS 数据中抽取 ~3000 条路径
  → 用 Qwen2.5-72B 打文化一致性分（1-5）
  → 归一化后与 Step 2 数据合并

Step 4: PRM 训练
  输入：Step 2+3 的 PRM训练集数据（pairwise 格式）
  模型：Qwen2.5-7B + value head
  损失：Bradley-Terry ranking loss
  输出：文化一致性 reward 模型

Step 5: GRPO 训练
  输入：GRPO训练集（question + culture）
  Policy：Qwen2.5-7B（或更小）
  Reward：R_ans（规则） + R_cultural（PRM，冻结）
  输出：文化对齐的单体模型

Step 6: 评估
  在评估集上评估最终模型
```

---

## 8. 评估设计

### 8.1 基础指标

**Accuracy**：预测答案与 gold label 的匹配率，分数据集、分文化报告。

### 8.2 文化敏感性（核心指标）

测试同一 question，不同 culture 输入时，模型输出是否有实质差异：

```
Culture Sensitivity Score = 
  不同文化下答案分布的 KL 散度均值
```

目标：同一问题在不同文化下应该产生不同答案分布，而不是趋同。

### 8.3 推理质量

- **Reasoning Coherence**：推理路径与最终答案是否一致（用 LLM Judge 评估）
- **Cultural Grounding**：推理路径中是否出现了该文化的具体价值观关键词

### 8.4 多样性

- **Intra-cultural Diversity**：同一文化 agent 生成的多条路径之间的 BLEU 差异（越大越好，说明探索充分）

### 8.5 消融实验

| 实验组 | 说明 |
|--------|------|
| Base | 直接用基础模型推理，无文化 conditioning |
| + Culture Prompt | 加 culture token，无训练 |
| + SFT only | 只用正确路径做 SFT |
| + SFT + GRPO (R_ans only) | 只用答案正确性做 reward |
| + SFT + GRPO (R_ans + R_cultural) | 完整方案 |
| MAS Oracle | 多智能体系统直接推理的上界 |

---

## 9. 目录结构规划

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
│   │   └── train_prm.py              # PRM 训练（复用 agentark prm 逻辑）
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

## 10. 关键设计决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Agent 粒度 | 国家级（细粒度） | 与数据集标注粒度对齐，避免信息损失 |
| MAS 方法 | LLM Debate | 生成多样路径同时保持文化内一致性 |
| Reward 设计 | R_ans + R_cultural | 单独用 accuracy 无法区分推理质量 |
| PRM 训练数据 | MAS 弱标签 + LLM Judge 强标签 | 弱标签保证数量，强标签保证质量 |
| 数据分割 | 按 question 分割 | 防止 PRM 和 GRPO 数据泄露 |
| 图结构 | 不引入 | 无实质计算作用，增加实现复杂度 |
| 两类数据 | 混合训练，不同 prompt 格式 | 共享文化知识，但保留任务差异 |
