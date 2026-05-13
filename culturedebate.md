# CultureDebate: Distilling Multi-Agent Cultural Intelligence into a Single LLM

## 1. 问题定义

### 1.1 核心目标

将多智能体系统在文化对齐任务上的集体推理能力蒸馏进单个语言模型的权重，使单体模型在文化条件下的决策质量接近多智能体系统，同时保持推理效率。

### 1.2 任务统一建模

```
y ~ P(y | x, c)，中间经过显式或隐式推理路径 r
```

- `x`：问题（scenario / question）
- `c`：文化标识（country-level，如 Egypt、Japan、Germany 等）
- `r`：推理路径（显式或隐式）
- `y`：答案（yes/no 或选项 1-4）

### 1.3 数据集

使用 **CulturalBench**（`Cul/data/CulturalBench_mas.json`），共 1227 条样本，涵盖多国文化，答案格式为选项 1-4。

---

## 2. 创新点

相比 AgentArk（面向数学/代码等客观任务的同质多智能体蒸馏），本工作针对文化对齐任务提出两个关键创新：

### 2.1 异质多智能体蒸馏框架

AgentArk 使用同质 agent（同一模型的多个副本），多样性来自随机采样，且 reward 仅依赖答案正确性（R_ans），无法区分推理过程的文化合理性。本工作提出 **CultureDebate**，一个面向文化对齐任务的异质多智能体蒸馏框架，包含两个核心设计：

**异质文化 agent**：5个 agent 分别代表不同地理文化区域，多样性来自文化知识差异而非随机采样，天然产生跨文化的对比推理路径。

**文化一致性过程奖励（R_cultural）**：一条答案正确但推理不符合目标文化的路径（如用西方个人主义逻辑解释了阿拉伯集体主义场景的正确答案）对文化蒸馏没有价值。引入 R_cultural 由 PRM 对推理路径的文化一致性打分，与 R_ans 组合作为 GRPO 的 reward：

```
R_total = α * R_ans + β * R_cultural
```

使蒸馏后的单体模型不仅能给出正确答案，还能生成符合目标文化逻辑的推理路径。

**消融验证**：
- 异质 agent vs 同质 agent（去掉文化 system prompt）：推理路径多样性（embedding 距离）、Culture Sensitivity Score
- R_ans only vs R_ans + R_cultural：val_accuracy、Culture Sensitivity Score

### 2.2 Judge 综合推理轨迹的优先蒸馏（Judge-anchored Distillation）

AgentArk 蒸馏单个 agent 的推理路径，Judge 仅在 inference 阶段用于选答案，其推理过程被完全丢弃。

然而，Judge 在见过所有文化 agent 的分析后所做的综合裁决——"综合亚洲集体主义视角与欧洲世俗化视角，目标文化埃及更接近..."——代表了一种**跨文化综合推理能力**，是任何单个 agent 独立无法产生的，是多智能体交互涌现的高价值轨迹，也是单体模型蒸馏后应达到的能力上界。

本工作将 Judge 的完整推理轨迹（Solution 6）显式纳入蒸馏目标：

- **SFT 阶段**：Judge 轨迹作为优先级最高的监督数据，单独构成 SFT 训练集的核心子集
- **GRPO 阶段**：对 Judge 轨迹赋予额外 reward 权重，显式区分 Judge 推理与普通 agent 推理的蒸馏价值：

```
R_total = α * R_ans + β * R_cultural + γ * I(path == Judge_path)
```

**消融验证**：
- 蒸馏 Solution 1-5（agent only）vs Solution 1-6（含 Judge）vs Solution 6 only
- 指标：val_accuracy、Culture Sensitivity Score、Reasoning Coherence

---

## 3. 多智能体数据生成

### 3.1 框架：RECONCILE

采用 RECONCILE 框架，核心是**异质 agent**——5个 agent 分别代表不同文化区域，通过 system prompt 注入文化身份，独立生成推理路径后由 Judge 裁决。与 AgentArk 的同质 LLM Debate 不同，这里的多样性来自文化知识差异而非随机采样。

### 3.2 Agent 设计

| Agent | 文化角色 | 核心文化倾向 |
|-------|---------|-------------|
| Agent 0 | Asian Culture | 集体主义、孝道、等级尊重、社会和谐 |
| Agent 1 | European Culture | 个人主义、理性主义、社会福利、政教分离 |
| Agent 2 | North American Culture | 强个人主义、实用主义、竞争意识、直接表达 |
| Agent 3 | Latin American Culture | 家族主义、天主教影响、人际温情、集体与个人混合 |
| Agent 4 | African Culture | Ubuntu 哲学、社区主义、尊重长者、宗教多元 |

**Judge Agent**：唯一的答案决策者。读取所有 agent 的最终回答，基于题目中的 target culture 独立裁决。Judge 失效时 fallback 到多数投票。

### 3.3 生成流程

```
输入：(question, country)

Round 0：5 个 agent 各自独立生成（无辩论，保持多样性）
  输出：Reasoning: ... \n Answer: [1/2/3/4]

Judge：读取 5 条回答，独立给出最终答案
  输出：Reasoning: ... \n Answer: [1/2/3/4]

输出：Solution 1-5（各 agent）+ Solution 6（Judge）
```

**不设置辩论轮次的原因**：异质 agent 互看答案后会趋同，破坏多样性，减少 PRM 的 chosen/rejected 对。AgentArk 的辩论有效是因为同质 agent 可以互相纠错；异质 agent 独立生成天然具有多样性，辩论只会降低质量。

### 3.4 批量推理优化

每轮内将所有 sample × agent 的 prompt 合并为单次 vLLM batch：

```
Round 0：1227 × 5 = 6135 次
Judge：  1227 × 1 = 1227 次
总计：约 7362 次 LLM 调用
```

### 3.5 输入输出格式

**输入**：
```json
{"instruction": "### Question: ...", "input": "", "output": "1"}
```

**内部转换**：
```json
{"query": "### Question: ...", "gt": "1", "country": "Netherlands"}
```

**输出**（与 AgentArk 格式一致，可直接复用 `label.py`）：
```json
{
  "query": "...", "gt": "1", "country": "Netherlands",
  "response": "===== Solution 1 =====\nReasoning: ...\nAnswer: 3\n...\n===== Solution 6 =====\nReasoning: ...\nAnswer: 1\n"
}
```

---

## 4. Reward 信号设计

```
R_total = α * R_ans + (1-α) * R_cultural
```

只有一个超参数 α，初始值设为 **0.7**：R_ans 占主导（70%），R_cultural 作辅助（30%）。α=0.7 的理由：R_ans 是规则可验证的可靠信号；R_cultural 来自 0.6B 小模型，存在误差，不宜占比过高。

### 4.1 R_ans：答案正确性（可验证奖励）

直接规则计算，无需模型：

```
R_ans = 1  if predicted_answer == gold
R_ans = 0  otherwise
```

### 4.2 R_cultural：文化一致性（过程奖励，PRM 输出）

由 PRM 对推理路径打分，输出经 clip 后的连续标量：

```
R_cultural = clip(prm_score, 0.1, 0.9)
```

clip 避免极端分数主导梯度。R_cultural 的来源是训练好的 Qwen3-0.6B PRM，其训练数据标签来自 LLM-as-a-Judge 方法（见第5节）。

**与 R_ans 的分工**：R_ans 保证训练方向正确（答案对），R_cultural 细化推理质量（推理符合文化逻辑）。仅用 R_ans 无法区分"答对但推理文化错误"和"答对且推理文化正确"的路径。

---

## 5. Process Reward Model（PRM）训练

### 5.1 整体流程概览

```
MAS 推理数据（1227条 × 6 Solutions）
        ↓
Step 1: 构造 PRM 训练数据集（pairwise 格式）
        ↓
Step 2: 基座模型改造（添加 reward head）
        ↓
Step 3: Preference Learning 训练（Bradley-Terry loss）
        ↓
Step 4: 验证与质量评估
        ↓
冻结 PRM，供 GRPO 阶段使用
```

**注意**：PRM 训练不是标准 SFT（监督微调），而是 **Preference Learning（偏好学习）**。SFT 的目标是让模型生成正确的文字输出；PRM 的目标是让模型对 pairwise 路径打出正确的相对顺序分数。两者的数据格式、损失函数、输出层完全不同。

---

### 5.2 Step 1：构造 PRM 训练数据集

#### 5.2.1 数据来源与划分

基于已生成的 MAS 推理数据（`CulturalBench_mas_inference_20260510_192023.jsonl`，1227条），按 question 维度划分（5:2:3）：

```
1227 条样本
├── 50%（614条）→ PRM 训练集
├── 20%（245条）→ PRM 验证集（同时作为 GRPO 训练的分布内测试集）
└── 30%（368条）→ GRPO 训练集（不参与 PRM 训练）
```

**PRM 验证集复用说明**：GRPO 训练本身没有验证集（on-policy 训练），PRM 验证集（245条）未参与任何训练，可作为分布内测试集，在 GRPO 每 5 轮后推理评估 accuracy，用于早停和最优 checkpoint 选取。

#### 5.2.2 标签来源：LLM-as-a-Judge 打分

PRM 的训练标签完全来自 LLM-as-a-Judge 方法，使用与文化 agent 同规模的模型（Llama-3.1-8B 或 Qwen2.5-7B）对 MAS 推理数据中每条推理路径的文化一致性进行打分。

**打分流程：**

```
对 PRM 训练集（614条）中每条样本的每条推理路径（Solution 1-5）：
  输入：(question, country, reasoning_path)
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

**注意事项**：使用与 agent 同规模的模型打分存在**自我评分偏差**（self-evaluation bias）——模型倾向于给与自己风格相似的路径打高分，可能导致 PRM 学到的是"模型偏好的推理风格"而非"真正符合文化的推理质量"。此局限在资源受限条件下可接受，后续有条件时建议改用更大规模模型（如 72B）打分。

#### 5.2.3 Pairwise 对的构造

基于 LLM Judge 打出的分数，在同一 (question, country) 内按分数差构造 pairwise 对：

**构造规则：**
- 同一题内，score 差值 > 0.2 的两条路径构成一对
- 高分路径为 chosen，低分路径为 rejected
- 若所有路径分数相近（差值均 < 0.2），跳过该样本

**数据格式示例：**

```json
{
  "question": "### Question: Give me the answer from 1 to 4: What do Vietnamese grandparents usually gift their grandchildren for the traditional Lunar New Year celebration? 1. monetary gifts 2. Educational Materials 3. Cards with best wishes 4. Traditional Foods and Snacks. This question is for a country or language that is Vietnam. You can only choose one option.",
  "country": "Vietnam",
  "chosen": {
    "reasoning": "In Vietnamese culture, the Lunar New Year (Tết Nguyên Đán) is a time for family reunion and gift-giving. A central tradition is the giving of 'lì xì' (red envelopes with money) from elders to children and grandchildren, symbolizing good luck, prosperity, and blessings for the new year. This monetary gift tradition is deeply rooted in Vietnamese Confucian values of respect for elders and the importance of family harmony.",
    "answer": "1",
    "score": 0.85
  },
  "rejected": {
    "reasoning": "Vietnamese culture values family and tradition. During Lunar New Year, grandparents show love to grandchildren. Traditional foods and snacks are important in Vietnamese celebrations as they represent unity and good wishes.",
    "answer": "4",
    "score": 0.45
  }
}
```

chosen 和 rejected 的区分依据：chosen 包含具体的文化事实（lì xì、Confucian values），推理过程与越南文化高度匹配；rejected 的推理泛泛，缺乏文化特异性。

#### 5.2.4 最终数据集规模

```
PRM 训练集（614条样本）：
  每条样本 5 条路径，每条路径得到 LLM Judge 分数
  同题内 score 差值 > 0.2 的两两组合构成 pairwise 对
  预计有效 pairwise 对：约 800-1200 对

PRM 验证集（245条样本）：
  按同样方式构造
  预计 pairwise 对：约 300-500 对
```

---

### 5.3 Step 2：基座模型改造

#### 5.3.1 基座选择

```
Base Model：Qwen3-0.6B
理由：文化 reward 判断是语义分类问题，不需要复杂推理；
     0.6B 参数量轻量，显存仅 ~1.2GB，不占 GRPO 的显存预算。
```

#### 5.3.2 添加 Reward Head

在基座 LLM 之上添加一个线性层作为 reward head：

```python
class CultureRewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model          # Qwen3-0.6B，保留原始权重
        hidden_size = base_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)  # 新增，随机初始化

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # 取最后一个 token 的 hidden state
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        score = self.reward_head(last_hidden).squeeze(-1)  # (batch,)
        return torch.sigmoid(score)      # 映射到 (0, 1)
```

**为什么取最后一个 token**：自回归模型的最后一个 token 的 hidden state 对整个序列有最完整的上下文表示，是序列级打分的标准做法（RLHF reward model 的通用实现）。

**为什么加 Sigmoid**：将原始 logit 压缩到 (0,1)，与 clip(0.1, 0.9) 配合使用，避免极端分数主导梯度。

**Reward head vs 生成式打分**：GRPO 在线采样阶段需要对约 12000 条路径批量打分，生成式打分（让模型输出"0.8"这样的文字）速度慢且不稳定；reward head 输出连续标量，一次前向传播即得分数，与 Bradley-Terry loss 直接兼容。

#### 5.3.3 参数冻结策略

训练时三种策略的显存对比（单卡 RTX 4090 48GB）：

| 策略 | 显存估算 | 可行性 | 说明 |
|------|---------|--------|------|
| 全参微调（base + reward head） | ~11 GB | 单卡可行 | 模型小（0.6B），全参也只需约 11GB |
| 冻结 base，只训 reward head | ~6 GB | 单卡可行 | 最省显存，但表示能力受限 |
| LoRA 微调（base LoRA + reward head 全参） | ~8 GB | 单卡可行 | 平衡两者 |

```
全参微调显存明细（Qwen3-0.6B，bf16）：
  模型权重：0.6B × 2B = 1.2 GB
  梯度：1.2 GB
  AdamW 优化器状态：2 × 1.2 GB = 2.4 GB
  Reward head（Linear）：< 0.01 GB
  激活值（batch=32，seq=1024）：约 4-6 GB
  合计：约 9-11 GB（单卡 48GB 完全够用，不需要双卡）
```

**推荐：优先全参微调**（数据量小约 1000 对，0.6B 模型不易过拟合，全参微调表示能力更强）。若 Pairwise Accuracy 仍 < 65%，再改用 LoRA（rank=16）。

---

### 5.4 Step 3：Preference Learning 训练

#### 5.4.1 训练目标

PRM 训练不是 SFT（不优化 token 预测 loss），而是 **Preference Learning**——让模型学会对 chosen 路径打出比 rejected 路径更高的分数。

#### 5.4.2 损失函数：Bradley-Terry Ranking Loss

```python
def bradley_terry_loss(score_chosen, score_rejected):
    # score_chosen, score_rejected: (batch,) ∈ (0,1)
    return -torch.log(torch.sigmoid(score_chosen - score_rejected)).mean()
```

直觉：最大化 chosen 分数高于 rejected 分数的概率。当 score_chosen >> score_rejected 时，loss → 0；当两者相等时，loss = log(2) ≈ 0.693。

#### 5.4.3 数据加载格式

每个 batch 包含 N 个 pairwise 对，每对有 chosen 和 rejected 两条序列：

```python
# 输入序列构造
def build_input(question, country, reasoning_path):
    return f"[{country}]\n{question}\n{reasoning_path}"

# batch 格式
{
    "chosen_input_ids":   (N, seq_len),
    "chosen_attention_mask": (N, seq_len),
    "rejected_input_ids": (N, seq_len),
    "rejected_attention_mask": (N, seq_len),
}

# 前向传播
score_chosen  = model(chosen_input_ids, chosen_attention_mask)   # (N,)
score_rejected = model(rejected_input_ids, rejected_attention_mask)  # (N,)
loss = bradley_terry_loss(score_chosen, score_rejected)
```

#### 5.4.4 训练配置（单卡 RTX 4090 48GB）

| 参数 | 值 | 说明 |
|------|----|------|
| base model | Qwen3-0.6B | |
| 微调方式 | 全参微调 | 单卡约 11GB，48GB 完全够用 |
| epochs | 3-5 | |
| learning rate | 1e-5 | 全参微调用较小 lr |
| batch size | 32 pairs | |
| max sequence length | 1024 tokens | |
| optimizer | AdamW（weight_decay=0.01） | |
| warmup ratio | 0.05 | |
| bf16 | True | |
| gradient checkpointing | True | |

预计训练时间：约 10-20 分钟（数据量约 1000 对，收敛快）。

---

### 5.5 Step 4：验证与质量评估

#### 5.5.1 核心指标：Pairwise Accuracy

```
Pairwise Accuracy = 
  PRM 对 chosen 打分高于 rejected 的比例

目标：验证集 Pairwise Accuracy > 68%
```

- < 65%：对比信号不可靠，需要调整
- 65%-70%：稳定区间，足够为 GRPO 提供辅助信号
- train/val 差距 > 10%：过拟合，reward head 对新路径打分不准

#### 5.5.2 精度不足时的调整

| 问题 | 调整方案 |
|------|---------|
| Pairwise Acc < 65% | 降低 lr 至 5e-6；检查 LLM Judge 打分质量 |
| 过拟合（train-val > 10%） | 改用 LoRA 微调（rank=16）；增加 weight decay |
| 文化分布不均 | 按文化 weighted sampling，确保低频文化有足够样本 |

---

### 5.6 Reward 风险与缓解策略

| 风险 | 缓解策略 |
|------|---------|
| PRM 误差累积：R_cultural 误差被梯度放大 | α=0.7 使 R_ans 主导；clip(0.1, 0.9)；KL 惩罚 |
| 自我评分偏差（同规模模型打标签） | 记录局限性，后续条件允许时升级打分模型 |
| 数据量小导致过拟合 | 全参微调（0.6B 数据量小时不易过拟合）；每训练 1 epoch 评估一次 |
| Reward scale 不一致（R_ans 离散，R_cultural 连续） | α=0.7，(1-α)=0.3；clip(0.1, 0.9) 压缩连续值范围 |
| MAS 路径多样性不足，无有效 pairwise 对 | 0轮辩论独立生成，保证 agent 答案多样性 |

---

## 6. GRPO 强化学习微调

### 6.1 Student 模型

分别在两个基座上训练并评测，与未蒸馏的 base 模型对比：

- Llama-3.1-8B
- Qwen2.5-7B

```
输入格式：[Culture Token] + [Question]
输出格式：[Reasoning Path] + [Answer]
```

### 6.2 训练数据

GRPO 使用**多智能体推理数据原始数据集**（`CulturalBench_mas_inference.jsonl`）中的 GRPO 训练集部分（368条），而非 pairwise 对。原因：GRPO 是 on-policy 训练，每轮用当前 policy 在线采样新路径，prompt 来自原始 (question, country) 对，PRM 对新路径打分得到 R_cultural。pairwise 对仅用于 PRM 训练，不参与 GRPO。

验证：与 PRM 验证集相同（245条），每 5 轮在验证集上推理评估 accuracy，用于早停和最优 checkpoint 选取。

### 6.3 在线采样（On-policy）

```
对每个 prompt (x, c)：
  → 当前 policy 生成 n=10 条路径
  → 计算 R_total = 0.7 * R_ans + 0.3 * R_cultural（PRM 打分）
  → RLOO baseline 计算 advantage
  → 更新 policy 参数
  → 下一轮用更新后模型重新采样
```

### 6.4 训练配置与显存分析（2卡 RTX 4090，每卡 48GB）

**显存估算：**

```
Policy model（Llama-3.1-8B，bf16，ZeRO-3）：
  模型权重分片（ZeRO-3）：16GB ÷ 2卡 = 8GB/卡
  梯度 + 优化器分片（ZeRO-3）：约 32GB ÷ 2卡 = 16GB/卡
  激活值（micro_batch=2，seq=512）：约 4GB/卡
  合计：约 28GB/卡 ✓（48GB 安全）

Reference model（8B，冻结，cpu_offload）：
  卸载到 CPU，GPU 占用 ~0

PRM（Qwen3-0.6B，冻结）：约 1.2GB ÷ 2卡 = 0.6GB/卡

总计：约 28-30GB/卡，2卡 48GB 够用
```

**注意**：必须使用 ZeRO-3（而非 ZeRO-2），ZeRO-2 下单卡约 44GB，非常紧张；ZeRO-3 全分片后单卡降至约 28GB，安全。

| 参数 | 值 | 说明 |
|------|----|------|
| `n_samples_per_prompt` | 10 | group size |
| `advantage_estimator` | rloo | RLOO baseline |
| `reward_mode` | PRMVR | PRM + Verifiable Reward |
| `verifiable_reward_coef` | 0.7 | α，R_ans 权重 |
| `init_kl_coef` | 0.001 | KL 惩罚 |
| `temperature` | 0.7 | 采样温度 |
| `micro_rollout_batch_size` | 2 | 显存限制 |
| `micro_train_batch_size` | 2 | 显存限制 |
| `max sequence length` | 512 tokens | |
| `actor_learning_rate` | 5e-7 | |
| `bf16` | True | |
| `zero_stage` | 3 | 必须用 ZeRO-3 |

---

## 7. 收敛分析

### 7.1 文化任务收敛快的原因

- **reward 稀疏性低**：答案空间有限（1-4），随机猜测正确率 25%，几乎每组采样都有正有负，每轮有有效梯度
- **预训练知识起点高**：模型已有文化隐式表示，GRPO 只需激活强化
- **数据规模小**：GRPO 训练数据约 4000 prompt，模型快速遍历

### 7.2 收敛轮次与时间估算

```
单轮耗时（8B，A100，序列长 512）：
  采样阶段：约 20-30 分钟
  训练阶段：约 10-15 分钟
  合计：约 30-45 分钟

预计收敛：10-20 轮
总时间：约 5-15 小时
```

### 7.3 收敛指标

| 指标 | 收敛信号 |
|------|---------|
| `reward/mean` | 上升后趋于平稳 |
| `reward/std` | 下降（group 内差异缩小） |
| `kl_divergence` | 稳定在 0.05-0.2 |
| `policy_loss` | 震荡收窄 |
| `response_length` | 稳定，不持续增长 |

**警报信号**：
```
reward/std → 0            # 训练退化
kl_divergence > 0.5       # 降低 learning rate
train_acc - val_acc > 15% # 过拟合，立即早停
response_length 暴增       # 加长度惩罚
```

**验证集指标**（每 5 轮评估）：

| 指标 | 说明 |
|------|------|
| `val_accuracy` | 最终目标指标 |
| `culture_sensitivity` | 同 question 不同 culture 下输出是否有实质差异 |

**GRPO 收敛判断**：val_accuracy 连续 3 次评估（15 轮）不再提升即停止。

---

## 8. 完整 Pipeline

```
Step 1: 数据准备
  CulturalBench（1227 条）
  → 按 question 划分（5:2:3）：
    PRM 训练集(50%, 614条) / PRM 验证集兼分布内测试集(20%, 245条) / GRPO 训练集(30%, 368条)

Step 2: MAS 数据生成（异质 RECONCILE）
  5个文化 agent 独立生成推理路径（0轮辩论）+ Judge 裁决
  → 输出 jsonl（Solution 1-5: agent，Solution 6: Judge）

Step 3: 数据集划分
  split_dataset.py → prm_train.jsonl / prm_val.jsonl / grpo_train.jsonl

Step 4: LLM-as-a-Judge 打分 + 构造 pairwise 对
  label_data.py（Llama-3.1-8B 或 Qwen2.5-7B 打分）
  → prm_train_pairs.jsonl / prm_val_pairs.jsonl

Step 5: PRM 训练
  train_prm.py（Qwen3-0.6B 全参微调，Bradley-Terry loss）
  → /autodl-fs/models/prm_qwen3_0.6b/best/

Step 6: SFT 预训练（Judge 轨迹优先）
  训练数据：Judge 轨迹（Solution 6）作为核心 SFT 数据

Step 7: GRPO 训练
  Policy：Llama-3.1-8B 或 Qwen2.5-7B
  Reward：R_total = 0.7 * R_ans + 0.3 * R_cultural（PRM 冻结）
  时间：约 5-15 小时

Step 8: 评估
  base 模型 vs SFT 模型 vs GRPO 蒸馏模型 vs MAS Oracle
```

### 8.1 PRM 代码结构

```
Cul/prm/
├── split_dataset.py   # 按 5:2:3 划分 MAS 推理数据
├── label_data.py      # LLM-as-a-Judge 打分 + 构造 pairwise 对
└── train_prm.py       # Qwen3-0.6B 全参微调，Bradley-Terry loss
```

### 8.2 PRM 运行命令

```bash
# Step 1: 划分数据集
python Cul/prm/split_dataset.py \
    --input_file /autodl-fs/data/CulturalBench_mas_inference_20260510_192023.jsonl \
    --output_dir /autodl-fs/data/splits \
    --seed 42

# Step 2a: 对 PRM 训练集打分并构造 pairwise 对（用 llama 打分）
python Cul/prm/label_data.py \
    --input_file  /autodl-fs/data/splits/prm_train.jsonl \
    --output_file /autodl-fs/data/prm/prm_train_pairs.jsonl \
    --model_name  llama \
    --batch_size  32

# Step 2b: 对 PRM 验证集打分并构造 pairwise 对
python Cul/prm/label_data.py \
    --input_file  /autodl-fs/data/splits/prm_val.jsonl \
    --output_file /autodl-fs/data/prm/prm_val_pairs.jsonl \
    --model_name  llama \
    --batch_size  32

# Step 3: 训练 PRM
python Cul/prm/train_prm.py \
    --train_file /autodl-fs/data/prm/prm_train_pairs.jsonl \
    --val_file   /autodl-fs/data/prm/prm_val_pairs.jsonl \
    --output_dir /autodl-fs/models/prm_qwen3_0.6b \
    --epochs     5 \
    --batch_size 16 \
    --lr         1e-5
```

`--model_name` 支持 `llama`（Llama-3.1-8B-Instruct）或 `qwen`（Qwen2.5-7B-Instruct）或完整路径。

---

## 9. 评估设计

### 9.1 准确率

预测答案与 gold label 的匹配率，分文化报告。

### 9.2 文化敏感性（核心指标）

```
Culture Sensitivity Score = 不同文化下答案分布的 KL 散度均值
```

同一问题在不同文化下应产生不同答案分布。这一指标直接衡量蒸馏是否真正学到了文化差异，而非只是提高了准确率。

### 9.3 推理质量

- Reasoning Coherence：推理路径与最终答案是否一致（LLM Judge 评估）
- Cultural Grounding：推理路径中是否出现目标文化的具体价值观关键词

### 9.4 消融实验

| 实验组 | 说明 |
|--------|------|
| Base | 基础模型，无文化 conditioning |
| + Culture Prompt | 加 culture token，无训练 |
| + SFT only（agent paths） | 只用 agent 正确路径做 SFT，不含 Judge 轨迹 |
| + SFT only（Judge path） | 只用 Judge 轨迹做 SFT，验证创新点2的独立价值 |
| + GRPO (R_ans only) | 只用答案正确性做 reward |
| + GRPO (R_ans + R_cultural) | 加入文化一致性过程奖励，验证 R_cultural 增量价值 |
| + GRPO (full, w/ Judge reward) | 完整方案，含 Judge 路径额外权重 γ |
| Homogeneous MAS | 同质 agent（去掉文化 system prompt）蒸馏，验证异质设计的价值 |
| MAS Oracle | 多智能体系统直接推理的上界 |

