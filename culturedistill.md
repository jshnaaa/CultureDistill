# CAMA-D: Culture-Aware Multi-Agent Distillation Framework

## 1. 框架概览

### 1.1 核心目标

将 HFA-C²N 多智能体系统生成的结构化跨文化推理数据，通过三阶段蒸馏管线注入单体语言模型，使其具备：
- 主场文化确权能力（Guardian 的知识精度）
- 跨文化边界感知能力（Auditor 的对比视角）
- 文化一致性的自我过程监督能力（PRM 引导的推理路径优化）

### 1.2 三阶段总览

```
Stage 1: Home-Field Authority-Weighted SFT（主场权威加权监督微调）
  → 单体模型学习 Guardian 的确权推理模式，掩码 Auditor 早期混淆 Token

Stage 2: Open-Book Step Labeling（开卷式步骤标注）
  → 审计器在 Ground Truth 先验下，对推理步骤打离散标签 {-0.5, 0.0, +1.0}

Stage 3: Culture-Aware PRM → GRPO（文化感知过程奖励 → 强化学习）
  → 类别加权 MSE 训练 PRM；GRPO 使用加权平均 R_total 优化推理路径
```

### 1.3 与旧方案（culturedebate.md）的本质区别

| 维度 | 旧方案 (CultureDebate) | 新方案 (CAMA-D) |
|------|----------------------|----------------|
| SFT 数据处理 | 样本级过滤（选正确路径） | Token 级加权 + 掩码（精细控制学习信号） |
| PRM 标注方式 | LLM-as-a-Judge 打连续分 → pairwise | 开卷式审计器打离散步骤标签 → 逐步回归 |
| PRM 训练目标 | Bradley-Terry ranking loss | 类别加权 MSE 回归 loss |
| PRM 输出粒度 | 序列级单一分数 | 步骤级分数（每个推理步独立打分） |
| GRPO Reward | R_total = R_outcome + beta*Sum(R_process)（加法累加，长度不公平） | R_total = 0.6*R_outcome + 0.4*Mean(R_process)（归一化加权平均） |
| 数据来源 | RECONCILE（平等 Agent） | HFA-C²N（Guardian/Auditor 不对称角色） |

---

## 2. Stage 1：主场权威加权 SFT

### 2.1 设计动机

HFA-C²N 生成的多智能体对话数据中，包含了 Guardian（主场守护者）和 Auditor（客场审视者）两种角色的完整推理轨迹。Auditor 在辩论早期可能输出带有文化混淆、偏见或引导错误的内容。如果使用传统 SFT（对所有 Token 平等计算交叉熵），单体模型会在自回归预测中拟合这些"毒草 Token"，在内部种下文化混淆的种子。

### 2.2 核心策略：Token 级加权与掩码

**原则**：

- Guardian 的确权和纠偏 Token → 保留，loss 权重乘以 α（放大学习信号）
- Auditor 最终轮之前的对抗性输出（质疑、混淆、偏离目标文化的内容）→ labels 填充 -100（完全掩码，不参与梯度计算）
- Auditor 最终轮中被 Guardian 说服后的正确表态 → 保留，loss 权重 = 1.0（不放大，但允许学习"认知转换模式"）

**实现方式**：

由于 HFA-C²N 数据中已带有 `[GUARDIAN]` 和 `[AUDITOR]` 角色标签，在构建 PyTorch DataLoader 时：

```python
# 伪代码：构造 Token 级 loss mask 和 weight
def build_token_weights(input_ids, role_spans, alpha=2.0):
    """
    role_spans: list of (start_idx, end_idx, role, round_idx, is_final_round)
    """
    loss_mask = torch.ones_like(input_ids, dtype=torch.float)
    loss_weight = torch.ones_like(input_ids, dtype=torch.float)

    for start, end, role, round_idx, is_final in role_spans:
        if role == "GUARDIAN":
            # Guardian 所有 Token 保留，权重放大
            loss_weight[start:end] = alpha
        elif role == "AUDITOR":
            if not is_final:
                # Auditor 非最终轮：掩码（不学习对抗性混淆内容）
                loss_mask[start:end] = 0  # 等价于 labels=-100
            else:
                # Auditor 最终轮（被说服后的表态）：保留，权重=1.0
                loss_weight[start:end] = 1.0

    return loss_mask, loss_weight


# 在 loss 计算中
ce_loss = F.cross_entropy(logits, labels, reduction='none')  # (seq_len,)
weighted_loss = (ce_loss * loss_mask * loss_weight).sum() / loss_mask.sum()
```

### 2.3 超参数

| 参数 | 值 | 说明 |
|------|----|------|
| alpha (Guardian 权重) | 2.0 | Guardian Token 的 loss 放大系数 |
| Auditor 掩码范围 | 非最终轮全部 Token | 最终轮表态保留 |
| 学习率 | 2e-5 | 全参微调 |
| Epochs | 3 | 早停（val_acc 2 epoch 不提升） |

### 2.4 训练数据构造

数据来源：HFA-C²N 生成的完整多智能体对话（含 Guardian 确权 + Auditor 审视 + 多轮协商）。

**输入格式**：
```
[{country}]\n{question}
```

**输出格式**（完整对话，含角色标签）：
```
[GUARDIAN] Reasoning: 作为东亚文化守护者，我确认选项1在中国春节... Answer: 1
[AUDITOR-1] Reasoning: 从欧洲视角看... 我同意 Guardian 的判断。Answer: 1
[AUDITOR-2] Reasoning: 从北美视角看... Answer: 1
...
```

**数据规模估算**：
```
HFA-C²N 数据（如 NormAD 2633 条）中的 GRPO 训练集部分（约 791 条）
每条样本包含完整的 Guardian + Auditors 对话
→ 约 791 条 SFT 样本（每条含多个角色的完整推理轨迹）
```

### 2.5 设计收益

单体模型通过此阶段学到：
- **Guardian 的确权模式**："作为 X 文化的权威，我确认..."（高权重学习）
- **认知转换模式**："从 Y 文化视角看，可能是 Z，但结合目标文化，我同意..."（正常权重学习）
- **不学习**：Auditor 早期的文化混淆和引导错误内容（完全掩码）

---

## 3. Stage 2：开卷式步骤标注（Open-Book Step Labeling）

### 3.1 设计动机

传统 PRM 标注面临两个困境：
1. **闭卷式标注（无参考答案）**：要求标注模型在没有 Ground Truth 的情况下判断中间步骤的正确性，导致 self-evaluation bias（自信心膨胀，对自己的错误步骤也打高分）
2. **连续分数标注**：0.1-0.9 的连续值缺乏明确语义锚点，不同标注实例间一致性差

CAMA-D 提出"开卷式"标注：将 Ground Truth 答案作为外部先验输入给审计器，将标注任务从"开放式推理质量评判"降维为"局部语义关联匹配"——审计器只需判断当前步骤是"支持了正确选项"还是"指向了混淆项"。

### 3.2 步骤切分策略：启发式规则

**为什么不让审计器同时完成"切步+打标"**：8B 模型在长文本中同时做两件高度抽象的任务（逻辑切分 + 打分），输出 JSON 容易格式崩溃或打标尺度变形，增加不必要的工程调试成本。

**解耦策略**：先用确定性规则切分，再让审计器只做最简单的封闭式打标。

**切分规则（Python 脚本实现）**：

```python
import re

def split_reasoning_into_steps(reasoning_text: str, max_sentences_per_step: int = 3) -> list[str]:
    """
    启发式规则切分推理文本为语义单元。

    规则：
    1. 首选以换行符 \\n\\n 或 \\n 作为主切分点
    2. 若某段落过长（超过 max_sentences_per_step 句），
       检测强逻辑转折词，在转折词前进行二次切分
    3. 切分后打上显式前缀 [Step 1], [Step 2], ...
    """
    # 转折/因果词列表（跨语言可扩展）
    TRANSITION_MARKERS = [
        r'\bHowever,?\b', r'\bBut,?\b', r'\bTherefore,?\b',
        r'\bOn the contrary,?\b', r'\bNevertheless,?\b',
        r'\bIn contrast,?\b', r'\bConsequently,?\b',
        r'\bThus,?\b', r'\bMeanwhile,?\b', r'\bInstead,?\b',
    ]
    transition_pattern = '|'.join(TRANSITION_MARKERS)

    # Step 1: 按换行符主切分
    raw_segments = re.split(r'\n\n|\n', reasoning_text.strip())
    raw_segments = [s.strip() for s in raw_segments if s.strip()]

    # Step 2: 对过长段落进行二次切分
    steps = []
    for segment in raw_segments:
        sentences = re.split(r'(?<=[.!?])\s+', segment)
        if len(sentences) > max_sentences_per_step:
            # 检测转折词位置，进行二次切分
            current_chunk = []
            for sent in sentences:
                if re.search(transition_pattern, sent, re.IGNORECASE) and current_chunk:
                    steps.append(' '.join(current_chunk))
                    current_chunk = [sent]
                else:
                    current_chunk.append(sent)
            if current_chunk:
                steps.append(' '.join(current_chunk))
        else:
            steps.append(segment)

    # Step 3: 打上显式前缀
    labeled_steps = [f"[Step {i+1}] {step}" for i, step in enumerate(steps)]
    return labeled_steps
```

**切分示例**：

输入（Guardian 推理）：
```
In Vietnamese culture, the Lunar New Year (Tet Nguyen Dan) is the most
important holiday. A central tradition is the giving of 'li xi' (red
envelopes with money) from elders to children.

However, educational materials are more associated with the mid-autumn
festival. Traditional foods are important but as shared meals, not
individual gifts from grandparents specifically.

Therefore, monetary gifts (option 1) represent the most culturally
accurate answer for Vietnam's Lunar New Year grandparent-to-grandchild
gift-giving tradition.
```

输出：
```
[Step 1] In Vietnamese culture, the Lunar New Year (Tet Nguyen Dan) is the most important holiday. A central tradition is the giving of 'li xi' (red envelopes with money) from elders to children.
[Step 2] However, educational materials are more associated with the mid-autumn festival. Traditional foods are important but as shared meals, not individual gifts from grandparents specifically.
[Step 3] Therefore, monetary gifts (option 1) represent the most culturally accurate answer for Vietnam's Lunar New Year grandparent-to-grandchild gift-giving tradition.
```

### 3.3 审计器标注：封闭式三选一打标

**审计器模型**：与 MAS 数据生成同规模的模型（Qwen2.5-7B-Instruct 或 Llama-3.1-8B-Instruct）。

**为什么同规模模型足够**：
- 任务本质已从"开放式推理评判"降维为"局部语义关联匹配"
- 审计器接收了 Ground Truth 作为确定性先验，只需判断当前步骤是否支持正确答案
- 这是一个封闭式三选一分类任务，对模型能力要求远低于开放式生成

**Prompt 模板**：

```
You are evaluating a single reasoning step for cultural alignment.

Context:
- Question: {question}
- Target Culture: {country}
- Correct Answer: {ground_truth_answer}

Reasoning Step to Evaluate:
{step_text}

Task: Does this step support the correct answer within the target culture's context?

Label definitions:
- +1.0: This step provides culturally specific evidence that directly supports
        the correct answer (e.g., cites specific customs, traditions, values
        unique to the target culture)
- 0.0:  This step is neutral — it provides generic reasoning, format
        transitions, or universal logic that neither supports nor contradicts
        the correct answer in a culturally meaningful way
- -0.5: This step introduces cultural confusion — it points toward a wrong
        option, applies values from a different culture, or contains
        misconceptions about the target culture

Respond with ONLY one of: +1.0, 0.0, -0.5
```

**标签语义**：

| 标签 | 语义 | 示例 |
|------|------|------|
| +1.0（确权步） | 提供了目标文化的具体证据，直接支持正确答案 | "在越南，'li xi'（红包）是长辈给晚辈的传统..." |
| 0.0（中立步） | 格式转换、通用逻辑过渡、同义词复述 | "Let me analyze the options one by one..." |
| -0.5（混淆步） | 引入文化混淆，指向错误选项或应用了错误文化的价值观 | "在西方文化中，贺卡是最常见的节日礼物，所以选3..." |

### 3.4 标注质量保障

**批量化处理**：对每条推理路径的所有 Step 逐一独立打标（每个 Step 一次 LLM 调用），而非一次性打所有 Step。这确保审计器的注意力完全集中在单个 Step 上。

**一致性校验**：
- 对 10% 的样本进行重复标注（不同随机种子），计算标注一致率
- 目标：一致率 > 85%（三选一分类任务的合理期望）

**标注分布预期**：
```
中立步 (0.0):  ~55-65%（格式、过渡、通用逻辑居多）
确权步 (+1.0): ~20-30%（文化特异性证据）
混淆步 (-0.5): ~10-20%（文化混淆或错误引导）
```

### 3.5 输出数据格式

```json
{
  "question": "...",
  "country": "Vietnam",
  "gt": "1",
  "reasoning_source": "guardian",
  "steps": [
    {"step_idx": 1, "text": "[Step 1] In Vietnamese culture...", "label": 1.0},
    {"step_idx": 2, "text": "[Step 2] However, educational...", "label": 0.0},
    {"step_idx": 3, "text": "[Step 3] Therefore, monetary...", "label": 1.0}
  ]
}
```

---

## 4. Stage 3：Culture-Aware PRM 训练

### 4.1 PRM 架构

**基座模型**：Stage 1 SFT 训练完成的 student model（非独立小模型）。

**设计思路**：使用 SFT 后的模型作为 PRM 基座，因为该模型已经通过 Stage 1 学习了文化推理的语义表示，对文化相关 Token 具有更好的隐层表征。在此基础上添加线性回归头，以最小参数增量获得步骤级打分能力。

**架构**：

```python
class CulturePRM(nn.Module):
    def __init__(self, sft_model):
        super().__init__()
        self.backbone = sft_model  # Stage 1 SFT 后的模型
        hidden_size = sft_model.config.hidden_size
        self.score_head = nn.Linear(hidden_size, 1)  # 线性回归头

    def forward(self, input_ids, attention_mask, step_end_positions):
        """
        step_end_positions: (batch, max_steps) — 每个 step 终止符的位置索引
        输出：每个 step 位置的预测分数
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden)

        # 在每个 step 终止符位置提取 hidden state
        step_scores = []
        for b in range(hidden_states.size(0)):
            for pos in step_end_positions[b]:
                if pos == -1:  # padding
                    break
                h = hidden_states[b, pos, :]  # (hidden,)
                score = self.score_head(h).squeeze(-1)  # scalar
                step_scores.append(score)

        return torch.stack(step_scores)  # (total_steps,)
```

**Step 终止符定义**：每个 `[Step N]` 前缀对应的最后一个 Token 位置。在 tokenize 时，通过搜索 `[Step` 的 token pattern 确定每个 step 的边界。

### 4.2 训练目标：类别加权 MSE Loss

**为什么用 MSE 而非 Bradley-Terry**：
- Stage 2 产出的是每个 step 的绝对标签（+1.0, 0.0, -0.5），而非 pairwise 偏好对
- MSE 直接拟合绝对分数，训练更简单、标签利用更充分
- 步骤级标签天然比序列级标签数量多（一条路径 3-8 个 step），数据效率更高

**类别加权的必要性**：在自然生成的推理文本中，"中立步（0.0）"在统计学上占据绝大多数（长尾分布），"确权步（+1.0）"和"混淆步（-0.5）"属于高价值的边缘特征信号。如果不做损失加权，MSE Loss 会被海量中立步主导，导致 PRM "偷懒"——对任何步骤都倾向于输出接近 0 的预测值，失去对文化边界的敏感性。

**损失函数**：

```python
def class_weighted_mse_loss(pred_scores, true_labels, loss_mask):
    """
    pred_scores: (N,) — PRM 预测的步骤分数
    true_labels: (N,) — 真实标签 ∈ {-0.5, 0.0, +1.0}
    loss_mask:   (N,) — 有效步骤掩码（padding 位置为 0）
    """
    # 类别关联权重映射
    # 确权步 W(+1.0) = 2.5，混淆步 W(-0.5) = 2.0，中立步 W(0.0) = 1.0
    weights = torch.where(
        true_labels > 0.5, torch.tensor(2.5),   # +1.0 → W=2.5
        torch.where(
            true_labels < -0.25, torch.tensor(2.0),  # -0.5 → W=2.0
            torch.tensor(1.0)                         # 0.0  → W=1.0
        )
    )

    mse = (pred_scores - true_labels) ** 2  # (N,)
    weighted_mse = mse * weights * loss_mask

    return weighted_mse.sum() / loss_mask.sum()
```

**权重设定理据**：

| 类别 | 权重 W | 理由 |
|------|--------|------|
| 确权步 (+1.0) | 2.5 | 最高价值信号，模型需精确识别文化特异性证据 |
| 混淆步 (-0.5) | 2.0 | 次高价值，模型需识别文化偏差和跨文化混淆 |
| 中立步 (0.0) | 1.0 | 基准权重，数量多但信息密度低 |

### 4.3 训练配置

| 参数 | 值 | 说明 |
|------|----|------|
| 基座模型 | Stage 1 SFT model (7B/8B) | 已有文化语义表征 |
| 训练方式 | LoRA (rank=16) + score_head 全参 | 避免破坏 SFT 学到的生成能力 |
| 学习率 | 5e-5 (score_head), 1e-4 (LoRA) | |
| Epochs | 5 | |
| Batch size | 8 | |
| 优化器 | AdamW (weight_decay=0.01) | |
| bf16 | True | |

**为什么用 LoRA 而非全参微调**：PRM 的基座将在 Stage 3 GRPO 中作为打分器使用，如果全参微调可能破坏其作为生成模型的能力（后续若需要复用为 actor 模型）。LoRA 以最小扰动添加打分能力。

### 4.4 验证指标

| 指标 | 目标 | 说明 |
|------|------|------|
| 三分类准确率 | > 70% | 将预测分数离散化后与真实标签对比 |
| 确权步召回率 | > 75% | PRM 能识别大部分文化特异性步骤 |
| 混淆步召回率 | > 65% | PRM 能检出大部分文化偏差步骤 |
| Spearman 相关系数 | > 0.6 | 预测分数与真实标签的排序一致性 |

**离散化规则（验证用）**：
```
pred > 0.5   → 预测为 +1.0（确权步）
pred ∈ [-0.25, 0.5] → 预测为 0.0（中立步）
pred < -0.25 → 预测为 -0.5（混淆步）
```

---

## 5. Stage 3（续）：GRPO 强化学习

### 5.1 Reward 设计：加权平均形式

**旧方案的问题**：`R_total = R_outcome + beta * R_process`（加法形式）在长文本推理中会遭遇"路径长度惩罚/红利"不均的问题——通过堆砌大量平庸中立步（每步 PRM 给 ~0.2 分），累加效应会使长路径获得比精简纠偏路径更高的分数，破坏 GRPO 的优化方向。

**新方案**：

```
R_total = alpha * R_outcome + (1 - alpha) * Mean(R_process)
```

其中：
- `R_outcome ∈ {0, 1}`：答案正确性（规则可验证）
- `Mean(R_process)`：当前推理链中所有被激活步骤的 PRM 得分算术平均值，理论值域 ∈ [-0.5, 1.0]
- `alpha = 0.6`：结果奖励占主导

**超参数 alpha=0.6 的逻辑支撑**：

在文化对齐任务中，"答对（事实正确）"是硬指标，底线不能丢，因此 R_outcome 必须占大头（0.6）。而"推理路径的文化合理性"（R_process）作为软约束，负责从多组全部答对的采样中，选出表现得最像主场 Guardian、最优雅的那条路径。0.4 的权重足以在组内拉开相对 Advantage 的差距，促使 GRPO 向主场思辨方向演化。

**量纲安全性分析**：`R_outcome` 值域 {0,1}，`Mean(R_process)` 值域 [-0.5, 1.0]，两者量纲不完全对齐。但在 GRPO 中，最终起作用的是组内相对 Advantage（减去组内均值），绝对量纲差异会被归一化消除，设计安全。

### 5.2 GRPO 在线采样流程

```
对每个 prompt (question, country)：
  1. 当前 policy 采样 G=10 条推理路径
  2. 对每条路径：
     a. 规则验证答案 → R_outcome ∈ {0, 1}
     b. 启发式切分推理步骤 → [Step 1], [Step 2], ...
     c. PRM 对每个 Step 终止符位置打分 → scores[]
     d. Mean(R_process) = mean(scores)
     e. R_total = 0.6 * R_outcome + 0.4 * Mean(R_process)
  3. 组内计算 Advantage（RLOO baseline）
  4. 策略梯度更新 policy 参数
  5. 下一轮用更新后模型重新采样
```

### 5.3 训练配置

| 参数 | 值 | 说明 |
|------|----|------|
| Student model | Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct | |
| Group size (G) | 10 | 每 prompt 采样数 |
| Advantage estimator | RLOO | |
| alpha (R_outcome 权重) | 0.6 | |
| KL penalty | 0.05 | 防止 policy 漂移 |
| Temperature | 0.7 | 采样温度 |
| 学习率 | 5e-7 (RL-only), 1e-7 (SFT+RL) | |
| Max rounds | 30 (RL-only), 20 (SFT+RL) | |
| Eval every | 5 rounds | |
| bf16 | True | |
| ZeRO stage | 3 | 2卡必须 |

### 5.4 PRM 推理效率

GRPO 每轮需对 `prompt_count × G × avg_steps` 个 Step 打分。优化策略：

- PRM 冻结参数，纯推理模式（`torch.no_grad()`）
- 所有 Step 拼接为一个大 batch，单次前向传播完成
- PRM 使用 LoRA adapter，推理时合并权重（`merge_and_unload()`），无额外开销

---

## 6. 三种训练模式

### 6.1 模式 1：SFT-Only

```
Base Model → Stage 1 SFT（主场权威加权）→ 输出 SFT Model
```

仅做 Token 级加权 SFT，不做 RL。作为 baseline 验证 SFT 单独的价值。

### 6.2 模式 2：RL-Only

```
Base Model → Stage 3 GRPO（PRM 引导）→ 输出 RL Model
```

从 base 直接做 GRPO。PRM 仍需 Stage 2 标注数据训练，但 student 不经过 SFT。验证 RL 在无 SFT 初始化时的下限。

### 6.3 模式 3：SFT + RL（推荐）

```
Base Model → Stage 1 SFT → Stage 3 GRPO → 输出 SFT+RL Model
```

- SFT 让模型学会输出格式、Guardian 确权模式、认知转换模式
- GRPO 在此基础上进一步优化推理路径的文化质量
- 理论预期：SFT+RL >= RL-only >= SFT-only

### 6.4 PRM 训练流程（三种模式共用）

无论哪种 student 训练模式，Stage 2 标注 + PRM 训练（Stage 3 前半部分）都是必须的前置步骤：

```
HFA-C²N 数据 → 启发式切分 → 审计器开卷式打标 → PRM 训练（类别加权 MSE）
```

---

## 7. 完整 Pipeline

```
Phase 0: HFA-C²N 多智能体数据生成
  → 带 [GUARDIAN]/[AUDITOR] 标签的结构化推理数据

Phase 1 [Stage 1]: 主场权威加权 SFT
  → Token 级 -100 掩码 + Guardian alpha 加权
  → 输出: SFT Model (用于 PRM 基座 + SFT+RL 的 actor 初始化)

Phase 2 [Stage 2]: 开卷式步骤标注
  a. 启发式规则切分推理步骤
  b. 审计器（7B/8B）在 GT 先验下对每步打标 {-0.5, 0.0, +1.0}
  → 输出: step_labels.jsonl

Phase 3 [Stage 3-PRM]: PRM 训练（类别加权 MSE）
  → 基座: SFT Model + LoRA + score_head
  → 输出: 冻结的 PRM (供 GRPO 使用)

Phase 4 [Stage 3-GRPO]: GRPO 强化学习
  → R_total = 0.6 * R_outcome + 0.4 * Mean(R_process)
  → 输出: 最终 Student Model

Phase 5: 评估
  → val_accuracy, Cultural Sensitivity Score, Reasoning Coherence
```

---

## 8. 代码结构（规划）

```
Cul/
├── sft/
│   ├── train_sft_weighted.py      # Stage 1: Token 级加权 SFT
│   └── build_token_weights.py     # 构造 loss_mask 和 loss_weight
├── step_label/
│   ├── split_steps.py             # 启发式规则切分推理步骤
│   ├── label_steps.py             # 审计器开卷式打标（batch LLM 调用）
│   └── validate_labels.py         # 标注一致性校验
├── prm/
│   ├── train_prm_mse.py           # 类别加权 MSE 训练 PRM
│   └── eval_prm.py                # PRM 验证（三分类准确率、Spearman）
└── grpo/
    └── train_grpo_v3.py           # GRPO with Mean(R_process) reward
```

---

## 9. 消融实验设计

### 9.1 核心蒸馏方案对比

| 实验组 | 训练方式 | 预期排序 |
|--------|---------|---------|
| Base | 无训练 | 最低 |
| SFT-only (equal weight) | 传统 SFT（无 Token 加权） | 中低 |
| SFT-only (CAMA-D Stage 1) | Token 级加权 SFT | 中 |
| RL-only | GRPO from base | 中高 |
| SFT + RL (CAMA-D full) | Stage 1 → Stage 3 | 最高 |
| MAS Oracle | 多智能体系统直接推理 | 上界 |

### 9.2 模块贡献消融

| 消融项 | 对比 | 验证目标 |
|--------|------|---------|
| Token 加权 vs 样本级加权 | Stage 1 w/ vs w/o mask | 验证掩码 Auditor 混淆 Token 的价值 |
| 开卷式标注 vs 闭卷式标注 | Stage 2 w/ vs w/o GT prior | 验证 GT 先验消除 self-evaluation bias |
| 类别加权 MSE vs 均匀 MSE | PRM w/ vs w/o class weights | 验证加权对稀疏信号的保护 |
| Mean(R_process) vs Sum(R_process) | GRPO reward 形式 | 验证加权平均消除长度偏差 |
| alpha=0.6 vs alpha=0.8 vs alpha=0.4 | GRPO alpha 敏感性 | 找最优 R_outcome/R_process 平衡 |

### 9.3 评估指标

| 指标 | 说明 |
|------|------|
| val_accuracy | 预测答案与 gold label 匹配率 |
| Cultural Sensitivity Score | 同一问题不同文化下答案分布 KL 散度均值 |
| Reasoning Coherence | LLM Judge 评估推理路径与答案的一致性 |
| Cultural Grounding | 推理路径中目标文化具体价值观关键词出现率 |
| Cultural Boundary Awareness | 模型是否能正确区分相邻文化（如越南 vs 中国） |

---

## 10. 与旧管线的兼容性

CAMA-D 与现有代码（`culturedebate.md` 描述的旧管线）的关系：

- **数据生成**：复用 HFA-C²N（`generate_hfa_c2n_data.py`），无需修改
- **SFT**：需新建 `train_sft_weighted.py`，旧 `train_sft.py` 保留作为 baseline
- **PRM**：需新建 `train_prm_mse.py`，旧 `train_prm.py`（Bradley-Terry）保留作为对比
- **GRPO**：需修改 reward 计算逻辑（`train_grpo.py` → `train_grpo_v3.py`），旧版保留
- **评估**：评估指标和脚本完全复用

---

## 11. 风险与缓解

| 风险 | 缓解策略 |
|------|---------|
| 审计器标注噪声（7B 模型三选一也可能出错） | 一致性校验 > 85%；类别加权 MSE 容忍少量噪声 |
| PRM 在 GRPO 中被 reward hacking | alpha=0.6 使 R_outcome 主导；KL penalty 防漂移 |
| Token 加权 SFT 收敛不稳定 | 监控 Guardian/Auditor 分组 loss 曲线，确保 Guardian loss 下降更快 |
| 步骤切分粒度不一致影响 PRM 泛化 | 统一切分规则 + 限制每步最大长度；PRM 验证集覆盖不同长度 |
| Stage 1 SFT 作为 PRM 基座可能有偏 | 对比实验：用 base model 作 PRM 基座 vs SFT model 作基座 |

---

## 12. 待确认事项（实验启动前）

1. **HFA-C²N 数据是否已生成完毕**：Stage 1 SFT 和 Stage 2 标注都依赖 HFA-C²N 数据
2. **计算资源分配**：Stage 1 SFT (1 GPU) → Stage 2 标注 (1 GPU, batch inference) → PRM (1 GPU) → GRPO (2 GPU)
3. **优先跑哪个 student model**：Qwen2.5-7B 或 Llama-3.1-8B（建议先跑 Qwen，与 PRM 同系列）
4. **是否保留旧管线作为 baseline**：建议保留 `culturedebate.md` 方案作为 "Naive Distillation" baseline
