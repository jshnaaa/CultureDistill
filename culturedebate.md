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
R_total = α * R_ans + β * R_cultural
```

初始超参数：`α=1.0, β=0.3`。R_ans 作为主信号锚点，R_cultural 作为辅助信号。

### 4.1 R_ans：答案正确性（可验证奖励）

直接规则计算，无需模型：

```
R_ans = 1  if predicted_answer == gold
R_ans = 0  otherwise
```

### 4.2 R_cultural：文化一致性（过程奖励，PRM 输出）

由 PRM 对推理路径打分，判断路径是否与目标文化的价值观一致，输出经 clip 后的标量：

```
R_cultural = clip(prm_score, 0.1, 0.9)
```

**与 R_ans 的分工**：R_ans 保证训练方向正确（答案对），R_cultural 细化推理质量（推理符合文化逻辑）。两者互补：仅用 R_ans 无法区分"答对但推理错误"和"答对且推理正确"的路径。

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

基于已生成的 MAS 推理数据（`CulturalBench_mas_inference_20260510_192023.jsonl`，1227条），按 question 维度划分：

```
1227 条样本
├── 70%（858条）→ PRM 训练集
├── 15%（184条）→ PRM 验证集
└── 15%（185条）→ 留作 GRPO 训练集（不参与 PRM 训练）
```

#### 5.2.2 来源 A：Answer Correctness 弱标签（全量）

对 PRM 训练集的每条样本，从 Solution 1-5（agent 路径）中构造 pairwise 对：

```
同一 (question, country) 下：
  chosen   = answer == gold 的推理路径
  rejected = answer != gold 的推理路径
```

跳过条件：所有 agent 路径全部答对或全部答错（无对比信号）。

根据全量数据统计：
- 有效样本（有 PRM 对）：786/1227 = 64.1%
- 平均每样本 3.10 对
- 预计总 pairwise 对：约 2436 对（858条 × 64.1% × 3.10）

pairwise 数据格式：
```json
{
  "question": "### Question: ...",
  "country": "Vietnam",
  "chosen": "Reasoning: In Vietnamese culture...\nAnswer: 3",
  "rejected": "Reasoning: In Vietnamese culture...\nAnswer: 1"
}
```

#### 5.2.3 来源 B：文化一致性细粒度标签（抽样强标签）

从来源 A 的 chosen 路径中随机抽取约 300-500 条，使用 LLM Judge（Qwen2.5-72B）对推理路径的文化一致性打分（0.1-0.9 连续值）：

```
Prompt：
"You are evaluating whether the following reasoning reflects
the cultural values and norms of [COUNTRY].
Question: [question]
Reasoning: [reasoning path]
Rate the cultural consistency as a decimal between 0.1 and 0.9.
Respond with only a decimal number."
```

构造规则：同一题内 score 差值 > 0.2 的两条路径构成 pairwise 对。额外贡献约 300-400 个高质量对。

两种强标签方式对比：

| 方式 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| 人类评审员（5位） | 文化感知细腻，无西方偏见 | 成本高，IAA < 0.7，难覆盖全部文化 | 抽样验证（50-100条） |
| LLM Judge（Qwen2.5-72B） | 低成本，速度快，一致性高 | 西方文化偏见，评分分布集中 | 主要强标签来源（300-500条） |

推荐 LLM Judge 覆盖全量，人类评审员抽样验证，计算两者 Spearman 相关系数（> 0.6 则 LLM Judge 可信）。

#### 5.2.4 最终数据集规模

```
弱标签 pairwise 对：约 2436 对
强标签 pairwise 对：约 300-400 对
─────────────────────────────
PRM 训练集总计：约 2700-2800 对
PRM 验证集：按同比例构造，约 580-600 对
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

训练时有两种策略：

| 策略 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| 全参数微调 | base model + reward head 均参与训练 | 表示能力强 | 显存大，容易过拟合（数据量小） |
| 冻结 base，只训 reward head | 仅 reward head 参与训练 | 显存小，防过拟合 | 表示能力受限 |
| LoRA 微调 | base model 用 LoRA，reward head 全参 | 平衡两者 | 实现略复杂 |

**推荐**：先用"冻结 base + 只训 reward head"快速验证，若 Pairwise Accuracy < 65% 再改用 LoRA。

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

#### 5.4.4 训练配置（单卡 RTX 4090 / 4卡 A100）

| 参数 | 值 | 说明 |
|------|----|------|
| base model | Qwen3-0.6B | |
| 参数冻结 | 冻结 base，仅训 reward head | 防过拟合 |
| epochs | 3-5 | |
| learning rate | 1e-4（reward head）/ 0（base frozen） | reward head 随机初始化，需较大 lr |
| batch size | 32 pairs（4卡：每卡 8 对） | |
| max sequence length | 1024 tokens | |
| optimizer | AdamW（weight_decay=0.01） | |
| warmup ratio | 0.05 | |
| bf16 | True（A100）/ fp16（V100） | |
| gradient checkpointing | True | |

预计训练时间：约 10-20 分钟（数据量小，收敛快）。

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

#### 5.5.2 文化敏感性验证

同一 question，输入不同 country token，PRM 对各文化最高分路径内容应有实质差异。若不同文化的最高分路径内容高度相似，说明 PRM 没有学到文化区分能力。

#### 5.5.3 精度不足时的调整

| 问题 | 调整方案 |
|------|---------|
| Pairwise Acc < 65% | 增加 LLM Judge 强标签比例；降低 lr 至 5e-6 |
| 过拟合（train-val > 10%） | 改用 LoRA 微调替代全参；增加 weight decay |
| 文化分布不均 | 按文化 weighted sampling，确保低频文化有足够样本 |
| 不同文化分数无差异 | 检查 country token 是否正确注入输入序列 |

---

### 5.6 Reward 风险与缓解策略

| 风险 | 缓解策略 |
|------|---------|
| PRM 误差累积：R_cultural 误差被梯度放大 | α > β（1.0 vs 0.3）；clip(0.1, 0.9)；KL 惩罚 |
| 数据量小导致过拟合 | 冻结 base；每 5 轮评估；train-val gap > 15% 早停 |
| Reward scale 不一致（R_ans 离散，R_cultural 连续） | α=1.0, β=0.3；reward normalization |
| MAS 路径多样性不足 | 0轮辩论独立生成，保证 agent 答案多样性 |

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

### 6.2 在线采样（On-policy）

```
对每个 prompt (x, c)：
  → 当前 policy 生成 n=10 条路径
  → 计算 R_total = R_ans + 0.3 * R_cultural
  → RLOO baseline 计算 advantage
  → 更新 policy 参数
  → 下一轮用更新后模型重新采样
```

### 6.3 训练配置（4卡 A100）

显存分析：policy（8B，ZeRO-2）约 35GB × 4卡分片；reference（8B，冻结）约 16GB 需 cpu_offload；PRM（0.6B，冻结）约 1.2GB。

| 参数 | 值 |
|------|----|
| `n_samples_per_prompt` | 10 |
| `advantage_estimator` | rloo |
| `reward_mode` | PRMVR |
| `verifiable_reward_coef` | 1.0 |
| `init_kl_coef` | 0.001 |
| `temperature` | 0.7 |
| `micro_rollout_batch_size` | 2 |
| `micro_train_batch_size` | 2 |
| `max sequence length` | 512 tokens |
| `actor_learning_rate` | 5e-7 |
| `bf16` | True |
| `zero_stage` | 2 |

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
  → 按 question 划分：PRM 训练集(70%) / GRPO 训练集(20%) / 评估集(10%)

Step 2: MAS 数据生成（异质 RECONCILE）
  5个文化 agent 独立生成推理路径（0轮辩论）+ Judge 裁决
  → answer correctness 自动标注
  → 输出 jsonl（Solution 1-5: agent，Solution 6: Judge）

Step 3: LLM Judge 打分（R_cultural 标签）
  从 PRM 训练集抽取 300-500 条路径
  → Qwen2.5-72B 打文化一致性分（0.1-0.9）
  → 构造 pairwise 对

Step 4: PRM 训练
  输入：弱标签（来源A）+ 强标签（来源B），约 1800-2400 对
  模型：Qwen3-0.6B + reward head
  损失：Bradley-Terry ranking loss
  时间：约 15-30 分钟

Step 5: SFT 预训练（Judge 轨迹优先）
  训练数据：Judge 轨迹（Solution 6）作为核心 SFT 数据
  目标：让模型先学会跨文化综合推理的基本格式和能力上界

Step 6: GRPO 训练
  Policy：Llama-3.1-8B 或 Qwen2.5-7B
  Reward：R_ans（α=1.0）+ R_cultural（β=0.3）+ Judge 路径额外权重（γ）
  时间：约 5-15 小时

Step 7: 评估
  base 模型 vs SFT 模型 vs GRPO 蒸馏模型 vs MAS Oracle
```

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

