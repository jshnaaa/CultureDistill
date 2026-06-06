# CAMAD：基于 HF-CAC 的文化感知蒸馏框架（创新点二）

CAMAD 是基于 HF-CAC 生成的结构化推理数据构建的三阶段蒸馏框架，目标是将多智能体系统的跨文化推理能力注入单体语言模型，使其具备主场文化确权能力（Guardian 的知识精度）、跨文化边界感知能力（Auditor 的对比视角）、以及文化一致性的自我过程监督能力（PRM 引导的推理路径优化）。三阶段如下：

```
Stage 1: 主场权威加权SFT → 单体模型学习 Guardian 的确权推理模式，掩码 Auditor 早期混淆 Token

Stage 2: 开卷式步骤标注 + PRM 训练 → 审计器在 Ground Truth 先验下，对推理步骤打离散标签 {0.1, 0.5, 0.9}

Stage 3: 文化感知过程奖励 → GRPO 强化学习 → 使用加权平均 R_total 优化推理路径（量纲统一于 [0,1]）
```

融合策略

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

### 6.2 CGM-GRPO

CGM-GRPO（Culture-Guided Mixed-Policy GRPO）是 CAMAD 框架的核心创新训练算法，在标准 GRPO 的 advantage estimation 中注入来自 HF-CAC Guardian 的文化专家引导信号，实现「文化难度感知的混合策略强化学习」。

**核心思想**：保持 RLOO 对 on-policy 轨迹的计算完全不变，额外叠加一个 Guardian 引导项作为 advantage 增强（uniform bonus，对同一 prompt 的所有 rollout 施加相同偏移）。引导强度由三因子文化难度系数 $w_{culture}$ 动态调制。Guardian 不参与 policy gradient 的梯度计算（不需要 importance sampling），也不参与 RLOO baseline 计算，只通过自身的 reward 值影响 on-policy 轨迹被鼓励/抑制的程度。

**核心公式**：

$$A_i = A_i^{base} + \lambda \cdot w_{culture} \cdot S_{guardian}$$

其中：

- $A_i^{base} = R_i - \bar{R}_{on}$：标准 RLOO advantage（leave-one-out baseline，仅在 on-policy 轨迹之间计算）
- $S_{guardian} = R_{outcome}^{guardian} \cdot (R_{guardian} - \bar{R}_{on}^{full})$：质量门控的 Guardian 信号
- $R_{guardian} = 1.0$（Guardian 已通过质量门控过滤，只有答对时才参与计算）
- $\bar{R}_{on}^{full}$：当前 prompt 所有 on-policy rollout 的 $R_{total}$ 均值
- $\lambda$：全局引导强度超参（默认 0.5，建议搜索 {0.3, 0.5, 0.7}）
- bonus 对同一 prompt 的所有 $n\_samples$ 个 rollout 统一施加，起到整体抬升/压低 advantage 的效果

**三因子文化难度系数**（支持三种模式）：

$$w_{culture} = \lambda_1 \cdot (1 - hit\_rate) + \lambda_2 \cdot rarity_i + \lambda_3 \cdot isolation_i$$

- `hit_only` 模式（MVP）：$w = 1 - hit\_rate$
- `hit_rarity` 模式（标准推荐）：$w = 0.67 \cdot (1 - hit\_rate) + 0.33 \cdot rarity_i$
- `full` 模式（三因子）：$w = 0.6 \cdot (1 - hit\_rate) + 0.3 \cdot rarity_i + 0.1 \cdot isolation_i$

其中 $hit\_rate$ 为当前 prompt 的 on-policy 正确率，$rarity_i = 1 - freq_i$ 为文化圈在训练集中的稀缺度，$isolation_i = 1 - avg\_affinity_i$ 为文化圈的孤立度（从 `hf_cac_config.yaml` 的 6×6 亲缘矩阵计算）。

**门控机制**：

- 质量门控：Guardian 答错时 $R_{outcome}^{guardian}=0$，整个 $S_{guardian}=0$，引导项自动消失
- 必要性门控：$hit\_rate \geq 0.8$ 时强制 $w_{culture}=0$（模型对该 prompt 已足够好，无需外部引导）

**显存布局**（2×48GB vGPU）：

- cuda:0（Policy）：base model bf16 ~15GB + LoRA ~0.2GB + KV cache（20 seq × 640 tok）~4.5GB + 梯度/激活（gradient checkpointing）~8-12GB → 峰值 ~32-38GB
- cuda:1（PRM）：base model bf16 ~15GB + score_head ~0.01GB + 推理激活 ~2-3GB → 峰值 ~18-20GB
- Guardian 引导不增加任何 GPU 开销（纯 CPU 查表 + 标量运算）

**与标准 GRPO 的关键区别**：Guardian 轨迹不参与 RLOO baseline 计算，不参与 policy gradient 的 backward，不需要 importance sampling，不需要 Sim 相似度调制。它只是一个标量 bonus 统一叠加到 on-policy 轨迹的 advantage 上。

**训练命令**（双卡，SFT+CGM-GRPO 模式，少数样本快速验证）：

```bash
python Cul/grpo/train_grpo_mixed_policy.py \
    --model_name     qwen \
    --sft_adapter    /autodl-fs/data/model/qwen/normad_camad_sft/best \
    --data_pkl       /autodl-fs/data/qwen/normad_splits.pkl \
    --prm_path       /autodl-fs/data/model/qwen/normad_camad_prm/best \
    --guardian_data  /autodl-fs/data/qwen/normad_hf_cac_inference.jsonl \
    --affinity_config Cul/configs/hf_cac_config.yaml \
    --output_dir     /autodl-fs/data/model/qwen/normad_camad_cgm_grpo \
    --max_train_samples 10 \
    --max_rounds     3 \
    --n_samples      5 \
    --prompt_batch   4 \
    --guardian_lambda 0.5 \
    --w_culture_mode hit_only \
    --gen_mini_batch 4 \
    --logprob_mini_batch 8 \
    --logprob_mini_batch_grad 4
```

**训练命令**（双卡，SFT+CGM-GRPO 模式，全量训练）：

```bash
python Cul/grpo/train_grpo_mixed_policy.py \
    --model_name     qwen \
    --sft_adapter    /autodl-fs/data/model/qwen/normad_camad_sft/best \
    --data_pkl       /autodl-fs/data/qwen/normad_splits.pkl \
    --prm_path       /autodl-fs/data/model/qwen/normad_camad_prm/best \
    --guardian_data  /autodl-fs/data/qwen/normad_hf_cac_inference.jsonl \
    --affinity_config Cul/configs/hf_cac_config.yaml \
    --output_dir     /autodl-fs/data/model/qwen/normad_camad_cgm_grpo \
    --max_train_samples 0 \
    --max_rounds     20 \
    --n_samples      5 \
    --prompt_batch   8 \
    --guardian_lambda 0.5 \
    --w_culture_mode hit_rarity \
    --alpha          0.6 \
    --lr             2e-5 \
    --lora_r         16 \
    --batches_per_round 130 \
    --eval_every     5 \
    --gen_mini_batch 4 \
    --logprob_mini_batch 8 \
    --logprob_mini_batch_grad 4
```

**RL-only 模式**（不使用 SFT adapter，从 base model 出发）：

```bash
python Cul/grpo/train_grpo_mixed_policy.py \
    --model_name     qwen \
    --data_pkl       /autodl-fs/data/qwen/normad_splits.pkl \
    --prm_path       /autodl-fs/data/model/qwen/normad_camad_prm_rl_only/best \
    --guardian_data  /autodl-fs/data/qwen/normad_hf_cac_inference.jsonl \
    --affinity_config Cul/configs/hf_cac_config.yaml \
    --output_dir     /autodl-fs/data/model/qwen/normad_camad_cgm_grpo_rl_only \
    --max_train_samples 0 \
    --max_rounds     30 \
    --n_samples      5 \
    --prompt_batch   8 \
    --guardian_lambda 0.5 \
    --w_culture_mode hit_rarity \
    --alpha          0.6 \
    --lr             5e-5 \
    --lora_r         16 \
    --batches_per_round 130 \
    --eval_every     5
```

**评估命令**：

```bash
python Cul/evaluate.py \
    --mode sft_rl \
    --model_name qwen \
    --data_pkl /autodl-fs/data/qwen/normad_splits.pkl \
    --sft_adapter /autodl-fs/data/model/qwen/normad_camad_sft/best \
    --grpo_adapter /autodl-fs/data/model/qwen/normad_camad_cgm_grpo/best \
    --output_json /autodl-fs/data/results/cgm_grpo_eval.json
```

**关键参数说明**：

| 参数 | 说明 |
|------|------|
| `--guardian_data` | HF-CAC 推理 JSONL 文件路径（包含 Guardian 响应，必需）|
| `--guardian_lambda` | Guardian 引导强度（默认 0.5，建议搜索 {0.3, 0.5, 0.7}）|
| `--w_culture_mode` | 文化难度系数模式：`hit_only`（MVP）/ `hit_rarity`（推荐）/ `full`（三因子）|
| `--affinity_config` | 亲缘度矩阵配置路径（`hit_rarity` 和 `full` 模式需要，指向 hf_cac_config.yaml）|
| `--max_train_samples` | 训练样本数限制：0=全部，N=只用前 N 条（调试用）|
| `--gen_mini_batch` | Batch Generate 每批 prompt 数（默认 4，OOM 时降为 2）|
| `--logprob_mini_batch` | Phase A ref log-prob 每批样本数（默认 8，无梯度）|
| `--logprob_mini_batch_grad` | Phase B policy log-prob 每批样本数（默认 4，有梯度，OOM 时降为 2）|
| `--no_prm` | 禁用 PRM 评分（R_total = R_outcome，单卡即可运行）|

---

## 7. CAMAD 的 baseline

### 7.1 MAGDi

**论文**：MAGDi: Structured Distillation of Multi-Agent Interaction Graphs Improves Reasoning in Smaller Language Models (ICML 2024, UNC Chapel Hill)

**核心思想**：MAGDi 将多个大型教师模型之间的多轮讨论交互建模为有向无环图（Multi-Agent Interaction Graph, MAG），然后通过结构化蒸馏（Next-Token Prediction + Margin Ranking + GCN Node Classification 三目标联合优化）将交互中蕴含的推理知识注入小型学生模型，使其在推理时无需多智能体协作即可获得接近多智能体系统的推理能力。

**文化对齐适配方案**：

将 MAGDi 迁移到文化对齐任务时，核心改动是多智能体数据来源和图结构的适配。支持两种数据源模式，通过 `--data_source` 参数切换：

1. **MAGDi + RECONCILE**（主实验对比）：使用 RECONCILE 对称多智能体系统（5 个平等文化专家 + 1 个 Judge）生成讨论数据，图结构为全对称（所有 Agent → Judge）。这代表"通用多智能体蒸馏方法直接应用于文化任务"，与完整 CAMAD pipeline 做方法级对比。
2. **MAGDi + HF-CAC**（消融实验）：使用 CAMAD 的 HF-CAC 非对称多智能体系统（6 个 Agent，含 Guardian/Auditor 角色 + Judge）生成的数据，图结构为非对称（Guardian → 所有 Auditor，所有 Agent → Judge）。这与 CAMAD 的加权 SFT 蒸馏在相同数据上对比，隔离蒸馏方法本身的差异。

**实验设置**：

数据集使用与 CAMAD 完全相同的 train/test 划分（由 `split_data.py` 生成的 pkl 文件）。
学生模型通过 `--model_name` 参数指定，支持 `llama`和 `qwen`两种基座。
训练 10 个 epoch，损失权重 α=1.0, β=1.0, γ=0.1。评估指标为 overall accuracy。

**预估运行时长**（2 × 48GB vGPU 并行，7-8B 模型 fp16）：

| Step | 内容 | CultureBench (1227 samples) | NormAD (2633 samples) |
|------|------|----------------------------|----------------------|
| Step 0 | RECONCILE 推理数据生成（vLLM, tp=2） | ~15-20 min | ~30-40 min |
| Step 1 | MAG 图格式转换（纯 CPU） | < 1 min | < 1 min |
| Step 2 | 节点嵌入提取（推理，batch=32） | ~3-5 min | ~8-12 min |
| Step 3 | MAGDi 训练（10 epochs, batch=4, grad_accum=4） | ~20-30 min | ~45-60 min |
| Step 4 | 评估（推理，batch=16） | ~2-3 min | ~5-8 min |
| **合计** | | **~40-60 min** | **~90-120 min** |

说明：Step 0 使用 vLLM tensor_parallel_size=2 将模型分布在两卡上加速推理（每条样本需 6 次 Agent 推理 + 1 次 Judge 推理）；Step 2-4 通过 accelerate 自动设备映射实现双卡模型并行；Step 3 已启用 gradient checkpointing，每个 batch 需做两次 LLM 前向（正/负样本 margin ranking），是耗时最长的步骤。如果已有 HF-CAC 推理数据（跳过 Step 0），则 CultureBench 约 25-40 min，NormAD 约 60-80 min。

**Pipeline 与运行命令**：

代码位于 `MAGDi/` 目录，完整 pipeline 包含 4 步（RECONCILE 模式额外有 Step 0 自动生成推理数据）：

```bash
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
cd MAGDi
# Step 0（仅 RECONCILE 模式，自动触发）：生成对称多智能体推理数据
python generate_reconcile_data.py \
    --input_file /autodl-fs/data/normad_mas.json \
    --output_file /autodl-fs/data/qwen/normad_reconcile_inference.jsonl \
    --config_file ../Cul/configs/reconcile_config.yaml \
    --model_name qwen --use_vllm --tensor_parallel_size 2
    
python generate_reconcile_data.py \
    --input_file /autodl-fs/data/culturalBench_mas.json \
    --output_file /autodl-fs/data/qwen/culturalbench_reconcile_inference.jsonl \
    --config_file ../Cul/configs/reconcile_config.yaml \
    --model_name qwen --use_vllm --tensor_parallel_size 2

# Step 1：将推理数据转换为 MAG 图格式
python generate_mag_data.py \
    --data_source reconcile \
    --input_file /autodl-fs/data/qwen/normad_reconcile_inference.jsonl \
    --dataset normad \
    --output_file /autodl-fs/data/MAGDi/MAG/qwen/normad_reconcile.json

python generate_mag_data.py \
    --data_source reconcile \
    --input_file /autodl-fs/data/qwen/culturalbench_reconcile_inference.jsonl \
    --dataset culturalbench \
    --output_file /autodl-fs/data/MAGDi/MAG/qwen/culturalbench_reconcile.json
    
# Step 2：提取节点嵌入（加权平均池化 last hidden states）
python get_node_emb_culture.py \
    --mag_file /autodl-fs/data/MAGDi/MAG/qwen/normad_reconcile.json \
    --model_name qwen \
    --output_file /autodl-fs/data/MAGDi/MAG/qwen/normad_reconcile_node_emb.pkl \
    --data_source reconcile
    
python get_node_emb_culture.py \
    --mag_file /autodl-fs/data/MAGDi/MAG/qwen/culturalbench_reconcile.json \
    --model_name qwen \
    --output_file /autodl-fs/data/MAGDi/MAG/qwen/culturalbench_reconcile_node_emb.pkl \
    --data_source reconcile

# Step 3：训练 MAGDi（NTP + Margin Ranking + GCN 三目标）
python train_culture.py \
    --dataset normad --data_source reconcile \
    --mag_file /autodl-fs/data/MAGDi/MAG/qwen/normad_reconcile.json \
    --node_emb_file /autodl-fs/data/MAGDi/MAG/qwen/normad_reconcile_node_emb.pkl \
    --model_name qwen \
    --output_dir /autodl-fs/data/MAGDi/model/MAGDi_normad_reconcile_qwen \
    --num_epochs 10 --lr 5e-6 --alpha 1.0 --beta 1.0 --gamma 0.1
    
python train_culture.py \
    --dataset culturalbench --data_source reconcile \
    --mag_file /autodl-fs/data/MAGDi/MAG/qwen/culturalbench_reconcile.json \
    --node_emb_file /autodl-fs/data/MAGDi/MAG/qwen/culturalbench_reconcile_node_emb.pkl \
    --model_name qwen \
    --output_dir /autodl-fs/data/MAGDi/model/MAGDi_culturalbench_reconcile_qwen \
    --num_epochs 10 --lr 5e-6 --alpha 1.0 --beta 1.0 --gamma 0.1

# Step 4：评估（使用与 CAMAD 相同的 test split）
python test_culture.py \
    --dataset normad --data_source reconcile \
    --data_pkl /autodl-fs/data/qwen/normad_splits.pkl \
    --base_model qwen \
    --lora_model /autodl-fs/data/MAGDi/model/MAGDi_normad_reconcile_qwen \
    --output_json /autodl-fs/data/MAGDi/results/magdi_normad_reconcile_qwen.json
    
python test_culture.py \
    --dataset culturalbench --data_source reconcile \
    --data_pkl /autodl-fs/data/qwen/culturalbench_splits.pkl \
    --base_model qwen \
    --lora_model /autodl-fs/data/MAGDi/model/MAGDi_culturalbench_reconcile_qwen \
    --output_json /autodl-fs/data/MAGDi/results/magdi_culturalbench_reconcile_qwen.json
```

### 7.2 AgentArk

**论文**：AgentArk: Distilling Multi-Agent Intelligence into a Single LLM Agent (Luo et al., 2026, arXiv:2602.03955)

**核心思想**：AgentArk 提出将多智能体辩论系统（Multi-Agent Debate）的集体推理能力蒸馏进单个 LLM 的权重中，从而在保持单模型推理效率的同时获得接近多智能体系统的推理性能。其核心洞见是：将推理开销从推理时（test-time）转移到训练时（training-time），让显式的多智能体交互转化为隐式的模型能力。

**三层层次化蒸馏策略**：

1. **Reasoning-Enhanced SFT（R-SFT）**：使用教师模型（如 Qwen3-32B）的多智能体系统（LLM Debate、DyLAN、MAV 等）生成高质量推理轨迹，对学生模型进行监督微调。多个 Agent 经过多轮辩论产生的最终聚合答案作为训练目标，使学生模型内化多智能体的推理深度。

2. **Reasoning Trajectory-based Data Augmentation（DA）**：不仅使用最终答案，还保留完整的多轮辩论轨迹（包括各 Agent 的中间推理、互相批评与修正过程）作为增强数据。学生模型学习的不只是"正确答案"，而是"如何通过自我审视和修正到达正确答案"的过程。

3. **Process-Aware Distillation（PAD）**：训练一个过程奖励模型（Process Reward Model, PRM），对推理步骤进行细粒度质量评估，然后使用 GRPO（Group Relative Policy Optimization）强化学习优化学生模型的推理路径。PRM 对多智能体生成的解题方案进行正确性标注（通过独立的标注模型判断每个 solution 是否正确），据此训练步骤级奖励模型，再以 RLOO 优势估计和 token/step 级奖励基线指导策略优化。

**技术实现细节**：

- **多智能体推理阶段**：支持 14 种多智能体方法（LLM Debate、AgentVerse、DyLAN、MAD、MAV、Self-Consistency 等），通过 vLLM 批量推理高效生成约 342K 问题 / 2M 条推理轨迹。LLM Debate 为核心方法，典型配置为 3-5 个 Agent 进行 2 轮辩论后聚合。
- **方案标注阶段**：使用强教师模型（Qwen2.5-72B-Instruct）对每个解题方案进行二值正确性判定（true/false），通过 guided decoding 约束输出，构建 PRM 训练所需的正负样本对。
- **PRM 训练**：基于 TRL 的 PRMTrainer，在标注后的多方案数据上训练步骤级奖励模型，学习判断推理链中每一步的质量。
- **GRPO 强化学习**：使用训练好的 PRM 作为奖励信号，通过 RLOO 优势估计对学生模型进行在线策略优化，支持 PRM 奖励、可验证奖励（VR）及混合模式（PRMVR）。

**实现方案概述**：

在 `ark/culture/` 目录实现了 AgentArk 应用于文化对齐任务的完整 pipeline，共 5 个阶段：

- **Stage 0 — 同质多智能体辩论数据生成**（`generate_debate_data.py`）：部署 5 个相同的 LLM Agent（默认 Qwen2.5-7B-Instruct），对文化选择题进行 2 轮辩论。所有 Agent 使用统一 system prompt，不赋予任何文化身份标签。辩论结束后通过多数投票聚合最终答案，将完整辩论轨迹和最终回答作为 SFT 训练数据输出。通过 vLLM 进行批量推理加速。
- **Stage 1 — 标准监督微调**（`train_sft.py`）：使用辩论生成的数据对学生模型进行 LoRA SFT。采用均匀交叉熵损失（uniform CE loss），对所有 token 等权处理，不做任何文化权威加权或 token 掩码。与 CAMA-D 的 α=2.0 token-weighted SFT 形成对照。训练使用 Accelerate DDP 多卡并行。
- **Stage 2 — 过程奖励模型训练**（`train_prm.py`）：基于辩论数据中的正确性标注训练步骤级 PRM。使用标准 MSE 损失，不对正负样本做类别加权，与 CAMA-D 的 class-weighted MSE（权重 3:1）形成对照。模型架构复用 SFT 模型，在末端添加线性回归头。
- **Stage 3 — GRPO 强化学习**（`train_grpo.py`）：以训练好的 PRM 作为奖励信号，通过 RLOO 优势估计对 SFT 模型进行策略优化。支持 KL 散度惩罚（β=0.05）以防止策略漂移。采样时不对 prompt 添加国家/文化前缀，保持 AgentArk 原始设计的通用性。
- **Stage 4 — 评估**（`evaluate.py`）：在测试集上进行推理评估，输出整体准确率和按国家分组的细粒度准确率。评估时同样不使用 `[country]` 前缀，确保与训练一致。

**实验设置**：

实验覆盖两个数据集（NormAd 三分类、CultureBench 四分类）和两种数据来源（通过 `--data_source reconcile|hf_cac` 参数切换）。基座模型限定为两种：Qwen2.5-7B-Instruct（`--model_name qwen`）和 LLaMA-3.1-8B-Instruct（`--model_name llama`），不支持其他模型。数据以 pkl 格式存储，结构为 `{"train": [...], "val": [...], "test": [...]}`。SFT 和 GRPO 各训练 3 epoch，学习率 2e-5，LoRA rank=64。PRM 训练 5 epoch，学习率 1e-5。GRPO 每条样本生成 4 条候选（group_size=4）。

**Pipeline 与运行命令**：
位于 `ark/culture/` 目录，完整 pipeline 包含 5 步（以 CulturalBench + RECONCILE + Qwen 为例）：

```bash
# Step 0：同质多智能体辩论数据生成
python ark/culture/generate_debate_data.py \
    --input_file /autodl-fs/data/culturalBench_mas.json \
    --output_file /autodl-fs/data/qwen/culturalbench_agentark_debate.jsonl \
    --model_name qwen \
    --num_agents 5 --num_rounds 2 \
    --use_vllm --tensor_parallel_size 2

# NormAD 版本：
python ark/culture/generate_debate_data.py \
    --input_file /autodl-fs/data/normad_mas.json \
    --output_file /autodl-fs/data/qwen/normad_agentark_debate.jsonl \
    --model_name qwen \
    --num_agents 5 --num_rounds 2 \
    --use_vllm --tensor_parallel_size 2

# Step 0.5：数据划分（8:1:1 train/val/test）
python Cul/split_data.py \
    --input /autodl-fs/data/qwen/culturalbench_agentark_debate.jsonl \
    --output /autodl-fs/data/qwen/culturalbench_agentark_reconcile_splits.pkl \
    --seed 42

# Step 1：标准 SFT（均匀 CE Loss，无 token 加权）
python ark/culture/train_sft.py \
    --model_name qwen \
    --data_pkl /autodl-fs/data/qwen/culturalbench_agentark_reconcile_splits.pkl \
    --output_dir /autodl-fs/data/model/agentark/sft_culturalbench_reconcile_qwen \
    --epochs 3 --batch_size 4 --lr 2e-4 --lora_r 32

# Step 2a：推理步骤切分
python Cul/step_label/split_steps.py \
    --input_file /autodl-fs/data/qwen/culturalbench_agentark_debate.jsonl \
    --output_file /autodl-fs/data/qwen/culturalbench_agentark_steps.jsonl \
    --max_sentences_per_step 3 --sources guardian judge

# Step 2b：步骤标注（LLM open-book labeling）
python Cul/step_label/label_steps.py \
    --input_file /autodl-fs/data/qwen/culturalbench_agentark_steps.jsonl \
    --output_file /autodl-fs/data/qwen/culturalbench_agentark_step_labels.jsonl \
    --model_name qwen --batch_size 64 --tensor_parallel_size 2

# Step 2c：PRM 训练（标准 MSE，无类别加权）
python ark/culture/train_prm.py \
    --base_model_path /root/autodl-tmp/base/Qwen2.5-7B-Instruct \
    --sft_adapter_path /autodl-fs/data/model/agentark/sft_culturalbench_reconcile_qwen/best \
    --train_file /autodl-fs/data/qwen/culturalbench_agentark_step_labels.jsonl \
    --val_file /autodl-fs/data/qwen/culturalbench_agentark_step_labels_val.jsonl \
    --output_dir /autodl-fs/data/model/agentark/prm_culturalbench_reconcile_qwen \
    --epochs 5 --batch_size 8

# Step 3：GRPO 强化学习
python ark/culture/train_grpo.py \
    --model_name qwen \
    --data_source reconcile \
    --sft_adapter /autodl-fs/data/model/agentark/sft_culturalbench_reconcile_qwen/best \
    --data_pkl /autodl-fs/data/qwen/culturalbench_agentark_reconcile_splits.pkl \
    --prm_path /autodl-fs/data/model/agentark/prm_culturalbench_reconcile_qwen/best \
    --output_dir /autodl-fs/data/model/agentark/grpo_culturalbench_reconcile_qwen \
    --alpha 0.6 --n_samples 5 --max_rounds 30

# Step 4：评估（使用与 CAMAD 相同的 test split）
python ark/culture/evaluate.py \
    --mode sft_rl \
    --model_name qwen \
    --data_source reconcile \
    --data_pkl /autodl-fs/data/qwen/culturalbench_agentark_reconcile_splits.pkl \
    --sft_adapter /autodl-fs/data/model/agentark/sft_culturalbench_reconcile_qwen/best \
    --grpo_adapter /autodl-fs/data/model/agentark/grpo_culturalbench_reconcile_qwen/best \
    --output_json results/agentark_culturalbench_reconcile_qwen.json
```

**`--no_prm` 模式运行命令**（跳过 Step 2，GRPO 仅使用 outcome reward）：

```bash
# Step 0：同质多智能体辩论数据生成（同正常模式）
python ark/culture/generate_debate_data.py \
    --input_file /autodl-fs/data/culturalBench_mas.json \
    --output_file /autodl-fs/data/qwen/culturalbench_agentark_debate.jsonl \
    --model_name qwen \
    --num_agents 5 --num_rounds 2 \
    --use_vllm --tensor_parallel_size 2

# Step 0.5：数据划分（同正常模式）
python Cul/split_data.py \
    --input /autodl-fs/data/qwen/culturalbench_agentark_debate.jsonl \
    --output /autodl-fs/data/qwen/culturalbench_agentark_reconcile_splits.pkl \
    --seed 42

# Step 1：标准 SFT（同正常模式）
python ark/culture/train_sft.py \
    --model_name qwen \
    --data_pkl /autodl-fs/data/qwen/culturalbench_agentark_reconcile_splits.pkl \
    --output_dir /autodl-fs/data/model/agentark/sft_culturalbench_reconcile_qwen \
    --epochs 3 --batch_size 4 --lr 2e-4 --lora_r 32

# Step 2：跳过（--no_prm 模式无需 PRM 训练）

# Step 3：GRPO 强化学习（--no_prm，仅使用 binary outcome reward）
python ark/culture/train_grpo.py \
    --model_name qwen \
    --data_source reconcile \
    --sft_adapter /autodl-fs/data/model/agentark/sft_culturalbench_reconcile_qwen/best \
    --data_pkl /autodl-fs/data/qwen/culturalbench_agentark_reconcile_splits.pkl \
    --output_dir /autodl-fs/data/model/agentark/grpo_culturalbench_reconcile_qwen_noprm \
    --alpha 0.6 --n_samples 5 --max_rounds 30 \
    --no_prm

# Step 4：评估
python ark/culture/evaluate.py \
    --mode sft_rl \
    --model_name qwen \
    --data_source reconcile \
    --data_pkl /autodl-fs/data/qwen/culturalbench_agentark_reconcile_splits.pkl \
    --sft_adapter /autodl-fs/data/model/agentark/sft_culturalbench_reconcile_qwen/best \
    --grpo_adapter /autodl-fs/data/model/agentark/grpo_culturalbench_reconcile_qwen_noprm/best \
    --output_json results/agentark_culturalbench_reconcile_qwen_noprm.json
```

**预估运行时长**（2×48GB vGPU，以 ~3000 样本数据集为基准）：

正常模式（完整 PRM + GRPO pipeline）：

| 阶段 | 显存分配 | 预估时长 | 说明 |
|------|----------|----------|------|
| Step 0: 辩论推理 | 双卡 TP=2 | ~2-3h | 5 agents × 2 rounds, vLLM tensor parallel 加速 |
| Step 0.5: 数据划分 | CPU | <1min | 纯 CPU 操作 |
| Step 1: SFT | 双卡 DDP | ~0.5-1h | 3 epochs, LoRA r=32, Accelerate DDP |
| Step 2a: 步骤切分 | CPU | ~5min | 正则匹配，纯 CPU |
| Step 2b: 步骤标注 | 双卡 TP=2 | ~1-1.5h | vLLM 推理标注 |
| Step 2c: PRM 训练 | 单卡 | ~1-1.5h | 5 epochs, LoRA r=16 |
| Step 3: GRPO | 双卡（policy cuda:0 + PRM cuda:1） | ~15-25h | 130 batches/round, early stop ~10-15 rounds |
| Step 4: 评估 | 单卡 | ~30-45min | greedy decode test set |
| **总计** | — | **~21-33h** | |

`--no_prm` 模式（跳过 PRM，仅使用 outcome reward）：

| 阶段 | 显存分配 | 预估时长 | 说明 |
|------|----------|----------|------|
| Step 0: 辩论推理 | 双卡 TP=2 | ~2-3h | 同上 |
| Step 0.5: 数据划分 | CPU | <1min | 同上 |
| Step 1: SFT | 双卡 DDP | ~0.5-1h | 同上 |
| Step 2a/2b/2c: PRM 相关 | — | **跳过** | `--no_prm` 模式无需 PRM |
| Step 3: GRPO (no_prm) | 单卡即可（~30-36GB） | ~8-15h | 无 PRM 评分开销，每 batch 快约 30% |
| Step 4: 评估 | 单卡 | ~30-45min | 同上 |
| **总计** | — | **~11-20h** | |

说明：`--no_prm` 模式下 Step 3 仅使用 binary outcome reward（答对=1，答错=0），不加载 PRM 模型，因此单卡 48GB 即可运行，且每 batch 节省 PRM scoring 时间（约 10-15s/batch）。代价是丢失过程奖励信号，GRPO 对齐效果预期下降。正常模式下 GRPO 将 policy 放 cuda:0（峰值 ~30-36GB）、PRM 放 cuda:1（峰值 ~18-20GB），双卡各有余量。

**与 CAMA-D 的关键差异总结**：AgentArk baseline 实现中刻意保持了"通用蒸馏方法原样迁移"的设计哲学——同质 Agent（无文化身份）、均匀损失（无 token 加权）、标准 MSE PRM（无类别加权）、无国家前缀。这些设计选择使其与 CAMA-D 的文化感知机制形成清晰的消融对照，从而验证文化特异性设计带来的增益。

## 8. 消融实验设计

### 8.1 主实验

| 实验组 | 方法 | NormAd |
|--------|---------|--|
| Base | zero-shot |  |
| 单teacher蒸馏 | SFT |  |
| 多智能体协作 | HF-CAC |  |
| 多teacher蒸馏 | MAGDi |  |
| 多teacher蒸馏 | AgentArk |  |
| Ours | CAMAD(SFT-only) |  |
| Ours | CAMAD(RL-only) |  |
| Ours | CAMAD(SFT+RL) |  |
| Ours | CAMAD(CGM-RL) |  |

### 8.2 蒸馏方案对比（都使用HF-CAC的情况下）

| 实验组 | NormAd | CulturalBench |
|--------|---------|---------|
| HF-CAC |  |  |
| MAGDi(HF-CAC) | |  |
| AgentArk(HF-CAC) | |  |

### 8.3 多智能体协作方法对比

| 实验组 | NormAd | CulturalBench |
|--------|--------|---------|
| Vanilla RECONCILE |        |  |
| MAD |        |  |
| MACD |        |  |
| OG-MAR |        |  |
| HF-CAC |        |  |

### 8.4 分析实验一：HF-CAC中的智能体的数量

| 智能体数量 | NormAd | CulturalBench |
|--------|---------|---------|
| 6 |  |  |
| 5 | |  |
| 4 | |  |
| 3 | |  |
| 2 | |  |

### 8.5 分析实验二：HF-CAC中的辩论的轮次

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
│   └── train_grpo_v3.py            # ★ Stage 3-GRPO: Mean(R_process) reward + LoRA
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
| `run_camad_pipeline.py` | 一键运行 CAMA-D 全流程，支持 `full`、`sft_only`、`rl_only`、`sft_rl` 四种模式，自动串联 Phase 0-5 |
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
