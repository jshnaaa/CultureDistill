# CAMAD：文化感知自适应混合蒸馏框架（创新点二）

> **创新点二**：提出文化感知自适应蒸馏方法 CAMAD，通过将文化专家产生的文化一致性信号引入策略优化过程，并根据模型文化掌握程度动态调节专家干预强度，实现多智能体文化推理能力向单体模型的高效迁移。

**CAMAD（Culture-Aware Adaptive Mixed Distillation，文化感知自适应混合蒸馏）** 是基于 HF-CAC 多智能体系统生成的结构化推理数据构建的蒸馏框架，目标是将多智能体系统的跨文化推理能力高效迁移到单体语言模型，使其在保留文化知识（尤其是稀有/复杂文化）的同时，仍具备自主探索与推理能力。

CAMAD 的核心是一种**联合 SFT+RL 的混合蒸馏训练方法**（见第 6.2 节）：在同一优化目标中，一方面将 HF-CAC 的最终决策（Judge）作为监督蒸馏目标以"保知识、防遗忘"，另一方面用 GRPO 进行在线探索，并将文化专家（Guardian）产生的文化一致性信号作为优势函数增强项注入 RL 过程；同时根据模型对每条样本的文化掌握程度（在线命中率）动态调节监督强度与专家干预强度，实现"越不会越强监督、已掌握则少干预"的自适应蒸馏。

CAMAD 在数据侧复用 HF-CAC 三类角色的结构化轨迹：

```
Judge（裁决者）   → 提供多智能体最终答案，作为联合训练中 SFT 监督蒸馏的目标（"老师给出标准答案"）

Guardian（守护者）→ 判断模型生成答案的文化推理方向是否正确，返回 0/1 文化一致性信号，注入 RL 优势函数（"向导给出文化正确方向"）

Auditor（审视者）→ 提供跨文化对比视角，用于加权 SFT 的 Token 级权威加权与混淆掩码（见第 3 节）
```

为支撑联合训练中的过程级奖励，CAMAD 在 RL 前先完成 Token 级加权 SFT 的数据准备（第 3 节）、开卷式步骤标注（第 4 节）与 Culture-Aware PRM 训练（第 5 节）；这些组件为第 6.2 节的联合 SFT+RL 训练提供监督数据、过程奖励模型与文化专家信号。

## 3. 主场权威加权 SFT

### 3.1 动机

HF-CAC 生成的多智能体对话数据中，包含了 Guardian（主场守护者）和 Auditor（客场审视者）两种角色的完整推理轨迹。Auditor 在辩论早期可能输出带有文化混淆、偏见或引导错误的内容。如果使用传统 SFT（对所有 Token 平等计算交叉熵），单体模型会在自回归预测中拟合这些"毒草 Token"。

### 3.2 核心策略：Token 级加权与掩码

**原则**：

- Guardian 的确权和纠偏 Token → 保留，loss 权重乘以 α（放大学习信号）
- Auditor 最终轮之前的对抗性输出（质疑、混淆、偏离目标文化的内容）→ labels 填充 -100（完全掩码，不参与梯度计算）
- Auditor 最终轮中被 Guardian 说服后的正确表态 → 保留，loss 权重 = 1.0（不放大，但允许学习"认知转换模式"）

### 3.3 运行命令

双卡 DDP 并行训练（推荐）：

```bash
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
accelerate launch --num_processes 2 Cul/sft/train_sft_weighted.py \
    --model_name llama \
    --data_pkl /autodl-fs/data/llama/normad_splits.pkl \
    --output_dir /root/autodl-tmp/model/llama/normad_camad_sft \
    --alpha 2.0 \
    --epochs 5 \
    --batch_size 4 \
    --lr 2e-4 \
    --lora_r 32 \
    --eval_every_n_epochs 1

accelerate launch --num_processes 2 Cul/sft/train_sft_weighted.py \
    --model_name llama \
    --data_pkl /autodl-fs/data/llama/culturalBench_splits.pkl \
    --output_dir /root/autodl-tmp/model/llama/culturalBench_camad_sft \
    --alpha 2.0 \
    --epochs 5 \
    --batch_size 4 \
    --lr 2e-4 \
    --lora_r 32 \
    --eval_every_n_epochs 1

accelerate launch --num_processes 2 Cul/sft/train_sft_weighted.py \
    --model_name llama \
    --data_pkl /autodl-fs/data/qwen/culturalBench_splits.pkl \
    --output_dir /root/autodl-tmp/model/qwen/culturalBench_camad_sft \
    --alpha 2.0 \
    --epochs 5 \
    --batch_size 4 \
    --lr 2e-4 \
    --lora_r 32 \
    --eval_every_n_epochs 1

accelerate launch --num_processes 2 Cul/sft/train_sft_weighted.py \
    --model_name qwen \
    --data_pkl /autodl-fs/data/qwen/normad_splits.pkl \
    --output_dir /root/autodl-tmp/model/qwen/normad_camad_sft \
    --alpha 2.0 \
    --epochs 5 \
    --batch_size 4 \
    --lr 2e-4 \
    --lora_r 32 \
    --eval_every_n_epochs 1 \
    --max_samples 0
```

| 参数 | 含义 |
|------|------|
| `--data_pkl` | split_data.py 生成的 pkl 文件（包含 train/val/test 划分）|
| `--alpha` | Guardian Token 的 loss 权重放大系数（默认 2.0）|
| `--lora_r` | LoRA rank（默认 32，保证文化知识充分学习）|
| `--lr` | 学习率（LoRA 默认 2e-4，高于全参微调）|
| `--eval_every_n_epochs` | 每 N 个 epoch 在验证集上评估一次（默认 1）|
| `--batch_size` | 每张卡的 batch size（默认 4，双卡时全局有效 batch size = 4×2 = 8）|
| `--grad_accum_steps` | 梯度累积步数（默认 1，可增大以模拟更大 batch）|


单卡训练：

```bash
python Cul/sft/train_sft_weighted.py \
    --model_name qwen \
    --data_pkl /autodl-fs/data/qwen/normad_splits.pkl \
    --output_dir /root/autodl-tmp/model/qwen/normad_camad_sft \
    --alpha 2.0 \
    --epochs 5 \
    --batch_size 4 \
    --lr 2e-4 \
    --lora_r 32 \
    --eval_every_n_epochs 1 \
    --max_samples 0
```

评估：

```bash
python Cul/evaluate.py \
    --mode sft \
    --model_name qwen \
    --data_pkl /autodl-fs/data/qwen/normad_splits.pkl \
    --sft_adapter /root/autodl-tmp/model/qwen/normad_camad_sft/best \
    --output_json /autodl-fs/data/model/qwen/eval_sft_only.json
```

---

## 4. 开卷式步骤标注

### 4.1 动机

传统 PRM 标注面临两个困境：
1. **闭卷式标注（无参考答案）**：要求标注模型在没有 Ground Truth 的情况下判断中间步骤的正确性，导致 self-evaluation bias（自信心膨胀，对自己的错误步骤也打高分）
2. **连续分数标注**：0.1-0.9 的连续值缺乏明确语义锚点，不同标注实例间一致性差

CAMAD 提出"开卷式"标注：将 Ground Truth 答案作为外部先验输入给审计器，将标注任务从"开放式推理质量评判"降维为"局部语义关联匹配"——审计器只需判断当前步骤是"支持了正确选项"还是"指向了混淆项"。

### 4.2 步骤切分策略：启发式规则

**为什么不让审计器同时完成"切步+打标"**：8/7B 模型在长文本中同时做两件高度抽象的任务（逻辑切分 + 打分），输出 JSON 容易格式崩溃或打标尺度变形，增加不必要的工程调试成本。

**解耦策略**：先用确定性规则切分，再让审计器只做最简单的封闭式打标。

**切分规则**：

采用三层级启发式规则将推理文本切分为语义单元：

1. **主切分（换行符）**：以换行符（`\n\n` 或 `\n`）作为首选切分点，将推理文本分割为初始段落。
2. **二次切分（逻辑转折词）**：若某段落过长（超过 3 个句子），则在强逻辑转折词（如 However、But、Therefore、On the contrary、Nevertheless、In contrast、Consequently、Thus、Meanwhile、Instead 等）出现的位置进行二次切分，在转折词前断开。
3. **标签化**：对切分后的每个步骤打上显式前缀 `[Step 1]`、`[Step 2]`、... 形成有序步骤序列。

**切分示例**：

输入（Guardian 推理）：
```
In Vietnamese culture, the Lunar New Year (Tet Nguyen Dan) is the most important holiday. A central tradition is the giving of 'li xi' (red envelopes with money) from elders to children.

However, educational materials are more associated with the mid-autumn festival. Traditional foods are important but as shared meals, not individual gifts from grandparents specifically.

Therefore, monetary gifts (option 1) represent the most culturally accurate answer for Vietnam's Lunar New Year grandparent-to-grandchild gift-giving tradition.
```

输出：
```
[Step 1] In Vietnamese culture, the Lunar New Year (Tet Nguyen Dan) is the most important holiday. A central tradition is the giving of 'li xi' (red envelopes with money) from elders to children.
[Step 2] However, educational materials are more associated with the mid-autumn festival. Traditional foods are important but as shared meals, not individual gifts from grandparents specifically.
[Step 3] Therefore, monetary gifts (option 1) represent the most culturally accurate answer for Vietnam's Lunar New Year grandparent-to-grandchild gift-giving tradition.
```

### 4.3 审计器标注：封闭式三选一打标

**审计器模型**：与 MAS 数据生成同规模的模型（Qwen2.5-7B-Instruct 或 Llama-3.1-8B-Instruct）。

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
- 0.9: This step provides culturally specific evidence that directly supports the correct answer (e.g., cites specific customs, traditions, values unique to the target culture). The model strongly endorses this step.
- 0.5: This step is neutral — it provides generic reasoning, format transitions, or universal logic that neither supports nor contradicts the correct answer in a culturally meaningful way. Neither reward nor penalty.
- 0.1: This step introduces cultural confusion — it points toward a wrong option, applies values from a different culture, or contains misconceptions about the target culture. The model strongly rejects this step.

Respond with ONLY one of: 0.9, 0.5, 0.1
```

**标签语义**：

| 标签 | 语义 | PRM 目标 | 示例 |
|------|------|---------|------|
| 0.9（主场确权步） | 提供了目标文化的具体证据，直接支持正确答案 | Sigmoid → 0.9 | "在越南，'li xi'（红包）是长辈给晚辈的传统..." |
| 0.5（中立讨论步） | 格式转换、通用逻辑过渡、同义词复述 | Sigmoid → 0.5 | "Let me analyze the options one by one..." |
| 0.1（文化混淆步） | 引入文化混淆，指向错误选项或应用了错误文化的价值观 | Sigmoid → 0.1 | "在西方文化中，贺卡是最常见的节日礼物，所以选3..." |

**为什么使用全正值标签 {0.1, 0.5, 0.9} 而非 {-0.5, 0.0, +1.0}**：
在大模型对齐的工业实践中，Reward Model 的最后一层通常使用 Sigmoid 激活函数，其输出区间严格锁定在 (0, 1)。

### 4.4 标注质量保障

**批量化处理**：对每条推理路径的所有 Step 逐一独立打标（每个 Step 一次 LLM 调用），而非一次性打所有 Step。这确保审计器的注意力完全集中在单个 Step 上。

**一致性校验**：
- 对 10% 的样本进行重复标注（不同随机种子），计算标注一致率
- 目标：一致率 > 85%（三选一分类任务的合理期望）

**标注分布预期**：
```
中立讨论步 (0.5):  ~55-65%（格式、过渡、通用逻辑居多）
主场确权步 (0.9): ~20-30%（文化特异性证据）
文化混淆步 (0.1): ~10-20%（文化混淆或错误引导）
```

### 4.5 运行命令

**1: 启发式步骤切分**
```bash
python Cul/step_label/split_steps.py \
    --input_file /autodl-fs/data/qwen/normad_hf_cac_inference.jsonl \
    --output_file /autodl-fs/data/qwen/normad_steps_split.jsonl \
    --max_sentences_per_step 3 \
    --sources guardian
```

| 参数 | 含义 |
|------|------|
| `--max_sentences_per_step` | 每步最大句数，超过则触发二次切分（默认 3）|
| `--sources` | 使用哪些 Agent 的推理路径（默认仅 guardian）|

**2: 开卷式审计器打标**
```bash
python Cul/step_label/label_steps.py \
    --input_file /autodl-fs/data/qwen/normad_steps_split.jsonl \
    --output_file /autodl-fs/data/qwen/normad_step_labels.jsonl \
    --model_name qwen \
    --batch_size 64 \
    --tensor_parallel_size 2 \
    --validate_consistency
```

| 参数 | 含义 |
|------|------|
| `--batch_size` | vLLM 批次大小 |
| `--validate_consistency` | 是否进行 10% 重复标注一致性校验 |

**3: 标注验证报告**
```bash
python Cul/step_label/validate_labels.py \
    --input_file /autodl-fs/data/qwen/normad_step_labels.jsonl \
    --report
```

### 4.6 输出数据格式

```json
{
  "question": "...",
  "country": "Vietnam",
  "gt": "1",
  "reasoning_source": "guardian",
  "steps": [
    {"step_idx": 1, "text": "[Step 1] In Vietnamese culture...", "label": 0.9},
    {"step_idx": 2, "text": "[Step 2] However, educational...", "label": 0.5},
    {"step_idx": 3, "text": "[Step 3] Therefore, monetary...", "label": 0.9}
  ]
}
```

---

## 5. Culture-Aware PRM 训练

### 5.1 PRM 架构

**基座模型**：student model（或 SFT 后的模型）。

**架构**：
在基座之上添加一个线性回归头（hidden_size → 1）和 Sigmoid 激活函数。前向推理时，将完整输入（含所有 Step）送入基座模型获取最后一层 hidden states，然后在每个 Step 终止符的位置提取对应的 hidden state 向量，经线性头映射为标量 logit，再通过 Sigmoid 压缩到 (0, 1) 区间，作为该 Step 的预测分数。最终输出为一组步骤级分数，每个分数对应一个 Step 的质量评估。

**为什么保留 Sigmoid 激活函数**：
这是大模型对齐中 Reward Model 的工业级标准实践。Sigmoid(x) = 1/(1+e^(-x)) 将原始 logit 严格压缩到 (0, 1)，带来三个关键好处：
1. PRM 输出与标签空间 {0.1, 0.5, 0.9} 天然对齐，无需额外 clip 或归一化
2. 后续 GRPO 中 Mean(R_process) 的值域被死死锁定在 (0, 1)，与 R_outcome ∈ {0,1} 量纲完美统一
3. 数值稳定——不会因输出值过大/过小导致梯度爆炸

**Step 终止符定义**：每个 `[Step N]` 前缀对应的最后一个 Token 位置。在 tokenize 时，通过搜索 `[Step` 的 token pattern 确定每个 step 的边界。

### 5.2 训练目标：类别加权 MSE Loss

**为什么用 MSE 而非 Bradley-Terry**：
- Stage 2 产出的是每个 step 的绝对标签（0.9, 0.5, 0.1），而非 pairwise 偏好对
- MSE 直接拟合绝对分数，训练更简单、标签利用更充分
- 步骤级标签天然比序列级标签数量多（一条路径 3-8 个 step），数据效率更高

**类别加权的必要性**：在自然生成的推理文本中，"中立讨论步（0.5）"在统计学上占据绝大多数（长尾分布），"主场确权步（0.9）"和"文化混淆步（0.1）"属于高价值的边缘特征信号。如果不做损失加权，MSE Loss 会被海量中立步主导，导致 PRM "偷懒"——对任何步骤都倾向于输出接近 0.5 的预测值，失去对文化边界的敏感性。

**损失函数**：

对每个有效步骤计算预测分数与真实标签之间的均方误差，然后根据标签类别施加不同权重：主场确权步（标签 0.9）权重为 2.5，文化混淆步（标签 0.1）权重为 2.0，中立讨论步（标签 0.5）权重为 1.0。将加权后的 MSE 在所有有效步骤上求和，再除以有效步骤总数得到最终损失值。padding 位置通过掩码排除，不参与损失计算。

**权重设定理据**：

| 类别 | 权重 W | 理由 |
|------|--------|------|
| 主场确权步 (0.9) | 2.5 | 最高价值信号，模型需精确识别文化特异性证据 |
| 文化混淆步 (0.1) | 2.0 | 次高价值，模型需识别文化偏差和跨文化混淆 |
| 中立讨论步 (0.5) | 1.0 | 基准权重，数量多但信息密度低 |

### 5.3 验证指标

| 指标 | 目标 | 说明 |
|------|------|------|
| 三分类准确率 | > 70% | 将预测分数离散化后与真实标签对比 |
| 确权步召回率 | > 75% | PRM 能识别大部分文化特异性步骤 |
| 混淆步召回率 | > 65% | PRM 能检出大部分文化偏差步骤 |
| Spearman 相关系数 | > 0.6 | 预测分数与真实标签的排序一致性 |

**离散化规则（验证用）**：
```
pred > 0.7   → 预测为 0.9（主场确权步）
pred ∈ [0.3, 0.7] → 预测为 0.5（中立讨论步）
pred < 0.3   → 预测为 0.1（文化混淆步）
```

### 5.4 运行命令

**切分标注数据为 train/val（PRM 训练需要）**
```bash
python Cul/step_label/split_step_labels.py \
    --input_file /autodl-fs/data/qwen/normad_step_labels.jsonl \
    --output_dir /autodl-fs/data/qwen \
    --val_ratio 0.2 \
    --seed 42
```

**PRM 训练（LoRA）（SFT）**
```bash
python Cul/prm/train_prm_mse.py \
    --base_model_path /root/autodl-tmp/base/Qwen2.5-7B-Instruct \
    --sft_adapter_path /autodl-fs/data/model/qwen/normad_camad_sft/best \
    --train_file /autodl-fs/data/qwen/normad_step_labels_train.jsonl \
    --val_file /autodl-fs/data/qwen/normad_step_labels_val.jsonl \
    --output_dir /autodl-fs/data/model/qwen/normad_camad_prm \
    --epochs 5 \
    --batch_size 8 \
    --lr_head 5e-5 \
    --lr_lora 1e-4 \
    --lora_r 16 \
    --eval_every_n_epochs 1
```

| 参数 | 含义 |
|------|------|
| `--base_model_path` | 基座模型路径（Qwen2.5-7B 或 Llama-3.1-8B）|
| `--sft_adapter_path` | Stage 1 SFT LoRA adapter 路径（会 merge 到 base 中作为 PRM 基座）|
| `--lr_head` | score_head 学习率（默认 5e-5）|
| `--lr_lora` | PRM LoRA 参数学习率（默认 1e-4）|
| `--lora_r` | PRM LoRA rank（默认 16）|
| `--eval_every_n_epochs` | 每 N 个 epoch 在验证集上评估一次（默认 1）|

**PRM 训练（无 SFT adapter，直接基于 base model）**
```bash
python Cul/prm/train_prm_mse.py \
    --base_model_path /root/autodl-tmp/base/Qwen2.5-7B-Instruct \
    --train_file /autodl-fs/data/qwen/normad_step_labels_train.jsonl \
    --val_file /autodl-fs/data/qwen/normad_step_labels_val.jsonl \
    --output_dir /autodl-fs/data/model/qwen/normad_camad_prm_rl_only \
    --epochs 5 \
    --batch_size 8 \
    --lr_head 5e-5 \
    --lr_lora 1e-4 \
    --lora_r 16 \
    --eval_every_n_epochs 1
```

**PRM 评估**
```bash
python Cul/prm/eval_prm.py \
    --prm_path /autodl-fs/data/model/qwen/normad_camad_prm/best \
    --sft_path /autodl-fs/data/model/qwen/normad_camad_sft/best \
    --val_file /autodl-fs/data/qwen/normad_step_labels_val.jsonl

python Cul/prm/eval_prm.py \
    --prm_path /autodl-fs/data/model/qwen/normad_camad_prm_rl_only/best \
    --val_file /autodl-fs/data/qwen/normad_step_labels_val.jsonl
```
---

## 6. GRPO

### 6.1 GRPO（不混合）

#### 6.1.1 Reward：加权平均形式

```
R_total = alpha * R_outcome + (1 - alpha) * Mean(R_process)
```

其中：
- `R_outcome ∈ {0, 1}`：答案正确性（规则可验证，答错为 0，答对为 1）
- `Mean(R_process) ∈ [0.1, 0.9]`：当前推理链中所有步骤的 PRM 得分（经 Sigmoid）的算术平均值。中间全走偏为 ~0.1，全中立为 ~0.5，完美主场确权为 ~0.9
- `alpha = 0.6`：结果奖励占主导

#### 6.1.2 运行命令

**GRPO （SFT+RL 模式，LoRA，无 DeepSpeed）**
```bash
python Cul/grpo/train_grpo_v3.py \
    --model_name qwen \
    --sft_adapter /autodl-fs/data/model/qwen/normad_camad_sft/best \
    --data_pkl /autodl-fs/data/qwen/normad_splits.pkl \
    --prm_path /autodl-fs/data/model/qwen/normad_camad_prm/best \
    --prm_backbone /root/autodl-tmp/base/Qwen2.5-7B-Instruct \
    --output_dir /autodl-fs/data/model/qwen/normad_camad_grpo \
    --alpha 0.6 \
    --n_samples 10 \
    --max_rounds 20 \
    --eval_every 5 \
    --lr 2e-5 \
    --lora_r 16
```

| 参数 | 含义 |
|------|------|
| `--data_pkl` | split_data.py 生成的 pkl 文件（GRPO 使用 train 作为 prompt 来源，val 做验证）|
| `--sft_adapter` | SFT LoRA adapter 路径（RL-only 模式不传此参数）|
| `--prm_path` | PRM checkpoint（含 LoRA adapter + score_head.pt）|
| `--prm_backbone` | PRM 基座模型路径（原始 base model）|
| `--alpha` | R_total 中 R_outcome 的权重（默认 0.6）|
| `--n_samples` | 每 prompt 每轮采样数 G（默认 10）|
| `--max_rounds` | 最大训练轮数（SFT+RL 建议 20，RL-only 建议 30）|
| `--eval_every` | 每 N 轮在验证集上评估一次（默认 5）|
| `--lr` | GRPO LoRA 学习率（SFT+RL 用 2e-5，RL-only 用 5e-5）|
| `--lora_r` | GRPO LoRA rank（默认 16）|


**GRPO（无 SFT adapter，lr=5e-5，max_rounds=30）**
```bash
python Cul/grpo/train_grpo_v3.py \
    --model_name qwen \
    --data_pkl /autodl-fs/data/qwen/normad_splits.pkl \
    --prm_path /autodl-fs/data/model/qwen/normad_camad_prm_rl_only/best \
    --prm_backbone /root/autodl-tmp/base/Qwen2.5-7B-Instruct \
    --output_dir /autodl-fs/data/model/qwen/normad_camad_grpo_rl_only \
    --alpha 0.6 \
    --n_samples 10 \
    --max_rounds 30 \
    --eval_every 5 \
    --lr 5e-5 \
    --lora_r 16
```
与 SFT+RL 模式的关键差异：不传 `--sft_adapter`（从 base model 出发），学习率 5e-5（高于 SFT+RL 的 2e-5），最大轮数 30（多于 SFT+RL 的 20）。

**备选: GRPO（DeepSpeed ZeRO-3 版，train_grpo.py）**
```bash
deepspeed --num_gpus 2 Cul/grpo/train_grpo.py \
    --model_name     qwen \
    --grpo_data      /autodl-fs/data/qwen/normad_splits/grpo_train.jsonl \
    --val_data       /autodl-fs/data/qwen/normad_splits/prm_val.jsonl \
    --prm_path       /autodl-fs/data/model/qwen/normad_camad_prm_rl_only/best \
    --prm_base_path  /root/autodl-tmp/base/Qwen2.5-7B-Instruct \
    --output_dir     /autodl-fs/data/model/qwen/grpo_qwen_culture \
    --n_samples      10 \
    --max_rounds     30 \
    --eval_every     5
```

| 参数 | 含义 |
|------|------|
| `--grpo_data` | GRPO 训练数据（prompt 来源）|
| `--val_data` | 验证数据 |
| `--prm_path` | PRM checkpoint 路径（含 LoRA adapter + score_head.pt）|
| `--prm_base_path` | PRM 基座模型路径（Qwen2.5-7B-Instruct）|
| `--output_dir` | 输出目录 |
| `--n_samples` | 每 prompt 采样数 G |
| `--max_rounds` | 最大训练轮数 |
| `--eval_every` | 每 N 轮评估一次 |

与 `train_grpo_v3.py` 的区别：使用 DeepSpeed ZeRO-3 进行多卡并行（显存效率更高），R_total = 0.7×R_ans + 0.3×R_cultural，PRM 使用 step-level scoring（与 `train_prm_mse.py` 训练的 PRM 完全适配）。

#### 6.1.3 评估的运行命令

```bash
# 评估 RL-only 模型
python Cul/evaluate.py \
    --mode rl \
    --model_name qwen \
    --data_pkl /autodl-fs/data/qwen/normad_splits.pkl \
    --grpo_adapter /autodl-fs/data/model/qwen/normad_camad_grpo/best \
    --output_json /autodl-fs/data/model/qwen/eval_rl.json

# 评估 SFT+RL 模型
python Cul/evaluate.py \
    --mode sft_rl \
    --model_name qwen \
    --data_pkl /autodl-fs/data/qwen/normad_splits.pkl \
    --sft_adapter /autodl-fs/data/model/qwen/normad_camad_sft/best \
    --grpo_adapter /autodl-fs/data/model/qwen/normad_camad_grpo/best \
    --output_json /autodl-fs/data/model/qwen/eval_sft_rl.json
```

| 参数 | 含义 |
|------|------|
| `--mode` | 评估模式：`sft`、`rl`、`sft_rl` |
| `--data_pkl` | pkl 文件路径（使用其中的 test 集）|
| `--sft_adapter` | SFT LoRA adapter 路径（sft 和 sft_rl 模式需要）|
| `--grpo_adapter` | GRPO LoRA adapter 路径（rl 和 sft_rl 模式需要）|
| `--output_json` | 可选，保存详细结果（含每条样本的预测和按国家分组准确率）|

### 6.2 CAMAD 联合 SFT+RL 混合蒸馏方法（核心方法）

CAMAD 的核心训练算法是一种**联合 SFT+RL 的混合蒸馏方法**。它在同一个优化目标中同时完成两件事：用 HF-CAC 的 **Judge 最终决策**做监督蒸馏以保留文化知识（防止 RL 遗忘稀有/复杂文化），用 **GRPO** 进行在线探索并以 HF-CAC 的 **Guardian 文化一致性信号**增强优势函数；并通过模型对每条样本的**文化掌握程度**动态调节监督强度与专家干预强度。

#### 6.2.1 动机

HF-CAC 多智能体跨文化推理虽然准确，但推理成本高、难以直接部署到线上单体模型。若仅用 RL（如纯 GRPO）训练单体模型，模型会在探索过程中**遗忘已有的文化知识**，对稀有文化和复杂文化尤其明显。CAMAD 的目标是得到一个**同时保留文化知识与探索能力**的单体模型：既能像 Judge 一样给出正确的文化判断，又能像 Guardian 一样把握文化推理的正确方向，还能保留 RL 带来的泛化与自我纠错能力。

#### 6.2.2 核心思想与角色类比

CAMAD 把 HF-CAC 的两类专家信号映射为两种互补的学习方式：

- **Judge = 老师（Teacher）**：提供该问题的最终答案，作为**监督学习（SFT）目标**，让模型直接记住"标准答案"，承担"保知识、防遗忘"的职责。
- **Guardian = 向导（Guide）**：判断模型自己生成的答案在**文化推理方向**上是否正确，返回 0/1 信号，作为 **RL 奖励增强**，承担"指方向、促文化对齐"的职责。

二者在同一步训练中并行作用：SFT 项把模型拉向 Judge 的知识，RL 项让模型在探索中被 Guardian 引导向文化正确的方向，难度权重 `w` 同时调度两者的强度。

#### 6.2.3 每条样本的输入

| 输入 | 来源 | 作用 |
|------|------|------|
| Prompt + 目标文化 | 原始数据集 | 模型生成与训练的输入 |
| HF-CAC 输出（Judge 最终答案 $y_{judge}$）| HF-CAC 多智能体推理 | 作为 SFT 监督蒸馏的目标 |
| Guardian 判断 | HF-CAC Guardian | 判断模型生成答案是否符合目标文化，返回 0/1 文化一致性信号 $S_{guardian}$ |
| 模型在线表现（命中率 $hitrate$）| 当前模型对该 prompt 的采样正确率 | 衡量学习难度，驱动动态权重 |

#### 6.2.4 文化难度权重

文化难度由模型对该 prompt 的掌握程度 $hitrate$ 驱动。RL 引导项与 SFT 监督项共享同一 $hitrate$，但权重略有差异：

$$w = 1 - hitrate \qquad w_{sft} = \max(1 - hitrate,\ w_{min})$$

- $w = 1 - hitrate$：RL 阶段 Guardian 引导权重，可降到 0（已掌握时不再外部引导，回归纯探索）
- $w_{sft} = \max(1-hitrate,\ w_{min})$：SFT 阶段监督权重，带**地板值** $w_{min}$（默认 0.1），即使 $hitrate=1$ 也保留弱监督锚定，防止长期训练遗忘

含义：**模型越不会（$hitrate$ 低，权重大）→ 监督越强、文化引导越强；模型已掌握（$hitrate$ 高，权重小）→ 减少干预**，使训练对每条样本的干预强度随掌握程度自适应。

**$hitrate$ 的计算**：取当前模型对该 prompt 的 on-policy 采样正确率，并用 EMA 滑动平均平滑（动量默认 0.9）：

$$hitrate \leftarrow m \cdot hitrate_{prev} + (1-m) \cdot acc_{cur}$$

其中 $acc_{cur}$ 为当轮 $G$ 个 rollout 的正确率。EMA 用于降低小采样数（如 $G=5$，正确率只有 6 个离散取值）下的噪声；也可关闭 EMA 直接使用当轮值。

#### 6.2.5 RL 阶段：文化引导的优势函数

标准 GRPO 的优势函数为组内 baseline 归一：

$$A_i^{base} = R_i - \bar{R}$$

其中 $R_i$ 为第 $i$ 个 rollout 的奖励（$R_{total} = \alpha \cdot R_{outcome} + (1-\alpha)\cdot \text{Mean}(R_{process})$，见 6.1.1），$\bar{R}$ 为同一 prompt 下所有 rollout 的奖励均值。

CAMAD 在此基础上叠加文化专家引导项（带全局强度系数 $\lambda_g$）：

$$A_i = A_i^{base} + \lambda_g \cdot w \cdot S_{guardian}$$

其中：

- $S_{guardian} \in \{0, 1\}$ 为 **per-rollout** 信号（对每个 rollout 单独判断），定义为该 rollout 的**文化推理方向**是否与 Guardian 判断一致：
  - $S_{guardian} = 1$：该 rollout 的文化推理方向正确（即使最终选项答错，只要文化方向对也给正信号）
  - $S_{guardian} = 0$：文化推理方向错误
- $\lambda_g$：Guardian 引导全局强度系数（默认 **0.5**），用于平衡原始 reward 与 Guardian 信号，防止二值信号完全主导 advantage 的符号

采用 per-rollout（而非对同一 prompt 统一偏移）能充分利用组内差异：在同一 prompt 的 $G$ 个 rollout 中，文化方向正确的轨迹被相对鼓励、错误的被相对抑制。

直观含义：在文化掌握程度低（$w$ 大）的样本上，文化推理方向正确的 rollout 会获得更高的优势、被更强地鼓励；模型已掌握（$w$ 小）时，引导项自动衰减，回归标准 GRPO 探索。

#### 6.2.6 SFT 阶段：Judge 输出的监督蒸馏

将 Judge 的最终答案 $y_{judge}$ 作为监督目标，标准交叉熵：

$$L_{SFT} = -\log P(y_{judge} \mid x)$$

按文化难度加权（SFT 使用带地板值的权重 $w_{sft}$，见 6.2.4）：

$$L_{SFT}^{weighted} = w_{sft} \cdot L_{SFT}$$

模型越不会的样本，监督信号越强，从而优先把 Judge 的文化知识"补"进单体模型；地板值 $w_{min}$ 保证即使 $hitrate=1$ 也保留弱监督锚定，防止长期训练遗忘。

#### 6.2.7 最终联合损失

$$L = L_{GRPO}(A_i) + \beta \cdot L_{SFT}^{weighted} = L_{GRPO}(A_i) + \beta \cdot w_{sft} \cdot L_{SFT}$$

其中：

- $L_{GRPO}(A_i)$ 使用 6.2.5 中文化引导后的优势 $A_i$（含 KL 惩罚项，KL 系数默认 0.05）
- $\beta$：两项 loss 的**平衡系数**（默认区间 **0.1~0.5**，可实验搜索）。由于 $L_{GRPO}$（on-policy，量级较小）与 $L_{SFT}$（交叉熵，量级较大）量纲不同，必须用 $\beta$ 缩放 SFT 项，否则 SFT 会碾压 RL、联合训练退化为加权 SFT

RL 部分负责**探索 + 文化引导并行**，SFT 部分负责**以难度自适应的方式保留 Judge 知识**。两项在同一步反向传播中联合优化，使单体模型在探索的同时持续锚定多智能体的文化决策。

#### 6.2.8 已确认的最终设计与默认超参

以下设计决策已确认，作为编码实现的基准：

| 项 | 决策 | 默认值 | 说明 |
|----|------|--------|------|
| 平衡系数 $\beta$ | 必须引入 | 0.1~0.5（搜索）| 缩放 SFT 项，防止 SFT 碾压 RL，确保联合训练有效 |
| Guardian 全局强度 $\lambda_g$ | 保留 | 0.5 | 平衡原始 reward 与 Guardian 信号，避免二值信号主导 advantage |
| $S_{guardian}$ 粒度 | per-rollout | — | 对每个 rollout 单独判断，利用组内差异精准引导 |
| $S_{guardian}$ 语义 | 文化推理方向正确性 | — | 判断文化方向（非最终选项对错），与 $R_{outcome}$ 互补 |
| $hitrate$ 统计 | 当轮 on-policy 正确率 + EMA | EMA 动量 0.9 | EMA 平滑降低小采样数（$G=5$）下的离散噪声，可退化为当轮值 |
| SFT 权重地板 $w_{min}$ | 引入 | 0.1 | $w_{sft}=\max(1-hitrate,\ w_{min})$，防长期遗忘 |

说明：RL 引导权重 $w = 1 - hitrate$（可降到 0），SFT 权重 $w_{sft} = \max(1-hitrate,\ w_{min})$（保留地板）。二者共享同一 $hitrate$，但 SFT 额外带地板以维持持续的知识锚定。

#### 6.2.9 显存布局（2×48GB vGPU）

联合训练相比纯 GRPO 仅增加一次对 $y_{judge}$ 的前向 + 交叉熵反传，显存占用基本不变：

- cuda:0（Policy）：base model bf16 ~15GB + LoRA ~0.2GB + KV cache ~4.5GB + 梯度/激活（gradient checkpointing）~8-12GB → 峰值 ~32-38GB
- cuda:1（PRM）：base model bf16 ~15GB + score_head ~0.01GB + 推理激活 ~2-3GB → 峰值 ~18-20GB
- Judge 监督目标为离线数据，Guardian 信号为查表 + 标量运算，二者均不增加 GPU 开销

#### 6.2.10 运行命令

CAMAD 联合 SFT+RL 训练入口为 `Cul/grpo/train_grpo_mixed_policy.py`（在原 CGM-GRPO 脚本基础上改造）。其中 `--guardian_data` 为 Stage 0 由 `Cul/generate_hf_cac_data.py` 产出的 HF-CAC 推理 JSONL，脚本会从中同时解析 Guardian 的判断（per-rollout 文化方向信号 $S_{guardian}$）与 Judge 的最终答案（SFT 蒸馏目标 $y_{judge}$）。

**冒烟测试（10 条样本，验证流程跑通）**
```bash
python Cul/grpo/train_grpo_mixed_policy.py \
--model_name qwen \
--sft_adapter /autodl-fs/data/model/qwen/normad_camad_sft/best \
--data_pkl /autodl-fs/data/qwen/normad_splits.pkl \
--prm_path /autodl-fs/data/model/qwen/normad_camad_prm/best \
--guardian_data /autodl-fs/data/qwen/normad_hf_cac_inference.jsonl \
--output_dir /autodl-fs/data/model/qwen/normad_camad \
--max_train_samples 10 \
--max_rounds 3 \
--n_samples 5
```

**完整 CAMAD 联合 SFT+RL 训练**
```bash
python Cul/grpo/train_grpo_mixed_policy.py \
--model_name qwen \
--sft_adapter /autodl-fs/data/model/qwen/normad_camad_sft/best \
--data_pkl /autodl-fs/data/qwen/normad_splits.pkl \
--prm_path /autodl-fs/data/model/qwen/normad_camad_prm/best \
--guardian_data /autodl-fs/data/qwen/normad_hf_cac_inference.jsonl \
--output_dir /autodl-fs/data/model/qwen/normad_camad \
--beta 0.3 \
--lambda_g 0.5 \
--w_min 0.1 \
--ema_momentum 0.9 \
--n_samples 5 \
--max_rounds 20 \
--batches_per_round 130
```

核心超参对应关系：

| 参数 | 含义 | 默认值 | 公式位置 |
| --- | --- | --- | --- |
| `--beta` | SFT 项平衡系数 $\beta$ | 0.3 | $L = L_{GRPO} + \beta \cdot w_{sft} \cdot L_{SFT}$ |
| `--lambda_g` | Guardian 全局强度 $\lambda_g$ | 0.5 | $A_i = A_i^{base} + \lambda_g \cdot w \cdot S_{guardian}$ |
| `--w_min` | SFT 权重地板 $w_{min}$ | 0.1 | $w_{sft} = \max(1-hitrate,\ w_{min})$ |
| `--ema_momentum` | hitrate EMA 动量 $m$ | 0.9 | $hitrate \leftarrow m\cdot prev + (1-m)\cdot acc_{cur}$ |
| `--use_ema` / `--no_ema` | 是否启用 EMA（否则用当轮命中率） | 启用 | 见 6.2.4 |
| `--alpha` | PRM 过程奖励权重 $\alpha$ | 0.6 | $R = R_{outcome} + \alpha \cdot R_{process}$ |

说明：$w = 1 - hitrate$ 可降到 0（RL 引导随掌握度退场），而 $w_{sft}$ 受 $w_{min}$ 约束始终 $\geq 0.1$（维持 Judge 知识锚定，防遗忘）。$S_{guardian}\in\{0,1\}$ 为 per-rollout 信号：当前 rollout 的文化推理方向（以解析出的选项为代理）与 Guardian 判断一致时取 1，从而放大组内"方向正确"的 rollout 优势。

---

## 8. 消融实验设计

### 8.1 主实验

主实验仅对比 **CAMAD 与其他 Baseline**，验证 CAMAD 相对于单 teacher 蒸馏、多 teacher 蒸馏与多智能体协作方法的整体优势。

| 实验组 | 方法 | NormAd |
|--------|---------|--|
| Base | zero-shot |  |
| 单teacher蒸馏 | SFT |  |
| 多智能体协作 | HF-CAC |  |
| 多teacher蒸馏 | MAGDi |  |
| 多teacher蒸馏 | AgentArk |  |
| Ours | CAMAD |  |

### 8.2 消融实验：蒸馏训练范式对比

本消融在均使用 HF-CAC 数据的前提下，对比不同蒸馏训练范式，验证 CAMAD 联合 SFT+RL 混合蒸馏相对于纯 SFT、纯 RL、以及分阶段 SFT→RL 的有效性。

| 实验组 | 训练范式 | 说明 | NormAd |
|--------|---------|------|--|
| SFT-only | 仅监督蒸馏 | 仅用 Judge 输出做加权 SFT |  |
| RL-only | 仅强化学习 | 从 base model 出发的 GRPO |  |
| SFT→RL（分阶段） | 先 SFT 再 RL | RL 从 SFT 后的模型出发，两阶段串联 |  |
| CAMAD | 联合 SFT+RL | 同一目标内联合优化（$L = L_{GRPO} + w\cdot L_{SFT}$）|  |

### 8.3 蒸馏方案对比（都使用HF-CAC的情况下）

| 实验组 | NormAd | CulturalBench |
|--------|---------|---------|
| HF-CAC |  |  |
| MAGDi(HF-CAC) | |  |
| AgentArk(HF-CAC) | |  |

### 8.4 多智能体协作方法对比

| 实验组 | NormAd | CulturalBench |
|--------|--------|---------|
| Vanilla RECONCILE |        |  |
| MAD |        |  |
| MACD |        |  |
| OG-MAR |        |  |
| HF-CAC |        |  |

### 8.5 分析实验一：HF-CAC中的智能体的数量

| 智能体数量 | NormAd | CulturalBench |
|--------|---------|---------|
| 6 |  |  |
| 5 | |  |
| 4 | |  |
| 3 | |  |
| 2 | |  |

### 8.6 分析实验二：HF-CAC中的辩论的轮次

| 辩论轮次 | NormAd | CulturalBench |
|--------|---------|---------|
| 0 |  |  |
| 1 | |  |
| 2 | |  |
| 3 | |  |

## 9. 代码结构

### 9.1 目录树

```
Cul/
├── run_camad_pipeline.py           # ★ 完整 Pipeline 入口脚本（一键运行全流程）
├── split_data.py                   # ★ 数据划分脚本（8:1:1 → pkl）
├── evaluate.py                     # ★ 评估脚本（支持 sft/rl/sft_rl 三种模式）
├── generate_hf_cac_data.py        # Phase 0: HF-CAC 多智能体数据生成（自动检测数据集类型）
├── resume_hf_cac.py               # Phase 0: HF-CAC 断点续跑工具
├── hf_cac_mas.py                  # HF-CAC 多智能体系统核心实现（支持 NormAD/CultureAtlas/CulturalBench）
├── scripts/
│   ├── convert_normad.py           # ★ 数据格式转换：normad.jsonl → normad_mas.json
│   ├── convert_culturalbench.py    # ★ 数据格式转换：CulturalBench CSV → culturalBench_mas.json
│   └── analyze_inference.py        # 推理结果分析工具（按国家/区域统计准确率）
├── configs/
│   ├── hf_cac_config.yaml         # HF-CAC 配置 — NormAD（三分类：可接受/不可接受/中立）
│   ├── hf_cac_config_cultureatlas.yaml  # HF-CAC 配置 — CultureAtlas（二分类比较）
│   ├── hf_cac_config_culturalbench.yaml # HF-CAC 配置 — CulturalBench
│   └── reconcile_config.yaml       # RECONCILE Agent 提示词配置（baseline 对比）
├── sft/
│   └── train_sft_weighted.py       # ★ Stage 1: Token 级加权 SFT（Guardian 权威加权）
├── step_label/
│   ├── split_steps.py              # ★ Stage 2a: 启发式规则切分推理步骤
│   ├── label_steps.py              # ★ Stage 2b: 审计器开卷式打标（vLLM batch）
│   ├── split_step_labels.py        # ★ Stage 2: 步骤标签数据划分（train/val）
│   └── validate_labels.py          # ★ Stage 2c: 标注一致性校验与分布报告
├── prm/
│   ├── train_prm_mse.py            # ★ Stage 3-PRM: 类别加权 MSE 训练
│   └── eval_prm.py                 # ★ PRM 验证（三分类准确率、Spearman）
├── grpo/
│   ├── train_grpo_v3.py            # ★ GRPO: Mean(R_process) reward + LoRA（基础 RL，6.1 节）
│   └── train_grpo_mixed_policy.py  # ★ CAMAD 联合 SFT+RL 训练脚本（核心方法 6.2 节）
└── data/                           # 数据存放目录
    ├── normad.jsonl                # 原始 NormAD 数据集
    ├── normad_mas.json             # NormAD 转换后（instruction/input/output/country）
    ├── cultureAtlas.json           # 原始 CultureAtlas 数据集
    ├── cultureAtlas_mas.json       # CultureAtlas 转换后
    ├── culturalBench_mas.json      # CulturalBench 转换后
    └── CulturalBench-Easy.csv      # 原始 CulturalBench 数据集
```

### 9.2 Pipeline 入口与工具

| 文件 | 功能 |
|------|------|
| `run_camad_pipeline.py` | 一键运行 CAMAD 全流程，支持 `full`、`sft_only`、`rl_only`、`sft_rl` 四种模式，自动串联 Phase 0-5 |
| `split_data.py` | 将 HF-CAC 推理数据按 8:1:1 划分训练集/验证集/测试集，输出 pkl 文件供所有训练和评估脚本使用 |
| `evaluate.py` | 在 pkl 测试集上评估最佳模型，支持 `sft`/`rl`/`sft_rl` 三种模式，输出整体准确率和按国家分组准确率 |
| `scripts/convert_normad.py` | 将原始 NormAD 数据集（JSONL）转换为 HF-CAC MAS 输入格式（JSON 数组），执行标签映射 yes→1/no→2/neutral→3，构建 instruction/input/output/country 四字段结构 |


```bash
python Cul/run_camad_pipeline.py \
    --mode sft_rl \
    --model_name qwen \
    --hf_cac_data /autodl-fs/data/qwen/normad_hf_cac_inference.jsonl \
    --output_root /autodl-fs/data/model/qwen/normad_sftrl_camad_outputs
```

```bash
python Cul/run_camad_pipeline.py \
    --mode sft_only \
    --model_name qwen \
    --hf_cac_data /autodl-fs/data/qwen/normad_hf_cac_inference.jsonl \
    --output_root /autodl-fs/data/model/qwen/normad_sft_camad_outputs
```

```bash
python Cul/run_camad_pipeline.py \
    --mode rl_only \
    --model_name qwen \
    --hf_cac_data /autodl-fs/data/qwen/normad_hf_cac_inference.jsonl \
    --output_root /autodl-fs/data/model/qwen/normad_rl_camad_outputs
```

参数说明：

| 参数 | 含义 |
|------|------|
| `--mode` | 训练模式：`full`（含数据生成）、`sft_only`、`rl_only`、`sft_rl`（推荐）|
| `--model_name` | Student 模型：`qwen`（Qwen2.5-7B）或 `llama`（Llama-3.1-8B）|
| `--hf_cac_data` | HF-CAC 推理数据 JSONL（pipeline 内部自动调用 split_data.py 生成 pkl）|
| `--data_pkl` | 可选，直接提供已切分的 pkl 文件（跳过数据划分步骤）|
| `--output_root` | 输出根目录，自动创建 `data/` 和 `models/` 子目录 |
| `--num_gpus` | GPU 数量（仅用于 vLLM 推理阶段，训练阶段使用模型放置）|
