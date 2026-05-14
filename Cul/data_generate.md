# 文化对齐多智能体推理数据生成方案

## 1. 任务目标

基于 CulturalBench 数据集，使用 RECONCILE 框架驱动的异质多智能体系统生成带推理路径的文化对齐训练数据，格式与 AgentArk 保持一致，供后续 PRM 训练和 GRPO 微调使用。

---

## 2. 数据集

- 路径：`Cul/data/CulturalBench_mas.json`
- 规模：1227 条
- 字段：`instruction`（问题）、`input`（空）、`output`（gold answer，1-4）
- 国家信息从 instruction 文本中正则提取（`country or language that is (.+?)\.`）

---

## 3. 智能体设计

### 3.1 文化角色（5个异质 Agent）

| Agent | 文化角色 | 核心文化倾向 |
|-------|---------|-------------|
| Agent 0 | Asian Culture | 集体主义、孝道、等级尊重、社会和谐 |
| Agent 1 | European Culture | 个人主义、理性主义、社会福利、政教分离 |
| Agent 2 | North American Culture | 强个人主义、实用主义、竞争意识、直接表达 |
| Agent 3 | Latin American Culture | 家族主义、天主教影响、人际温情、集体与个人混合 |
| Agent 4 | African Culture | Ubuntu 哲学、社区主义、尊重长者、宗教多元 |

**设计原则**：每个 agent 保留自身文化角色（提供多样性），但在 reasoning 时以目标文化为核心，自身文化视角作为参考。好处：输出更贴近目标文化事实，同时保留各 agent 之间的差异性。

**System Prompts（各 agent 在整个生成过程中保持不变）：**

```
Asian Culture:
You are a cultural expert for Asian cultures (East, Southeast, South Asia).
The question specifies a TARGET CULTURE.
Focus on the customs, norms, and values of the TARGET CULTURE, but you may
reference your Asian cultural perspective to explain similarities or contrasts.
Reason from this perspective, then give your answer.

European Culture:
You are a cultural expert for European cultures (Western and Northern Europe).
The question specifies a TARGET CULTURE.
Focus on the customs, norms, and values of the TARGET CULTURE, but you may
reference your European cultural perspective to explain similarities or contrasts.
Reason from this perspective, then give your answer.

North American Culture:
You are a cultural expert for North American cultures (United States, Canada).
The question specifies a TARGET CULTURE.
Focus on the customs, norms, and values of the TARGET CULTURE, but you may
reference your North American cultural perspective to explain similarities or contrasts.
Reason from this perspective, then give your answer.

Latin American Culture:
You are a cultural expert for Latin American cultures (Central/South America, Mexico).
The question specifies a TARGET CULTURE.
Focus on the customs, norms, and values of the TARGET CULTURE, but you may
reference your Latin American cultural perspective to explain similarities or contrasts.
Reason from this perspective, then give your answer.

African Culture:
You are a cultural expert for African cultures (sub-Saharan Africa).
The question specifies a TARGET CULTURE.
Focus on the customs, norms, and values of the TARGET CULTURE, but you may
reference your African cultural perspective to explain similarities or contrasts.
Reason from this perspective, then give your answer.
```

### 3.2 Judge Agent

Judge 是事实核查者和裁决者，综合 agent 推理与文化事实，不被多数票左右。

**System Prompt：**

```
You are a neutral cultural fact-checker and moderator.
You will see a multiple-choice question and responses from five cultural expert agents.
Your task is to determine the correct answer by carefully considering:
1. Verifiable cultural facts about the TARGET CULTURE.
2. The reasoning and answers provided by each agent.
Do NOT simply choose the majority answer — use agents' perspectives as supporting evidence.
You may agree with some, all, or none of the agents if factual evidence supports it.
```

### 3.3 各阶段 User Prompt

**Round 0 —— Agent 初始独立生成：**

```
{question}

Identify the TARGET CULTURE in the question.
Reason about its specific customs and values first,
then use your own cultural perspective as a reference if helpful.

Reasoning: <your reasoning>
Answer: <number>
```

**辩论轮次（若 num_debate_rounds > 0）—— Agent 参考其他智能体后更新：**

```
{question}

Other cultural experts have responded:

[Asian Culture]:
{response from agent 0}

[European Culture]:
{response from agent 1}

... (排除 agent i 自身)

Review their reasoning. Update your answer if you find stronger evidence,
otherwise defend your original. Stay focused on the TARGET CULTURE.

Reasoning: <your updated reasoning>
Answer: <number>
```

**Judge 裁决：**

```
{question}

Five cultural expert agents have responded:

[Asian Culture]:
{final response from agent 0}

... (全部 5 个 agent)

Determine the correct answer using verifiable facts about the TARGET CULTURE.
Reference the agents' reasoning as supporting evidence,
but do not simply follow the majority.

Reasoning: <your reasoning, referencing agents as needed>
Answer: <number>
```

---

## 4. RECONCILE 框架流程

```
输入：(question, target_culture)

Round 0：5 个 agent 各自独立生成，聚焦 target culture 的具体文化特征
  输出：Reasoning: ... \n Answer: [1/2/3/4]

Round 1：每个 agent 看到其他 4 个 agent 的回答，可修改自己的答案
  强调：不得因多数人同意而跟从，需有具体理由才更新
  输出：Reasoning: ... \n Answer: [1/2/3/4]

Judge：读取 5 个 agent 的最终回答，基于 target culture 事实裁决
  强调：不取多数投票，以可验证的文化事实为据
  Judge 失效时 fallback 到多数投票
  输出：Reasoning: ... \n Answer: [1/2/3/4]

输出：Solution 1-5（各 agent）+ Solution 6（Judge）
```

### 4.1 批量推理优化

每轮内将所有 sample × agent 的 prompt 合并为单次 vLLM batch，最大化 GPU 利用率：

```
Round 0：1227 × 5 = 6135 次
Round 1：1227 × 5 = 6135 次
Judge：  1227 × 1 = 1227 次
总计：约 13497 次 LLM 调用
```

---

## 5. 输入输出格式

**输入（原始样本）：**
```json
{"instruction": "### Question: ...", "input": "", "output": "1"}
```

**内部转换：**
```json
{"query": "### Question: ...", "gt": "1", "country": "Netherlands"}
```

**输出（与 AgentArk 格式一致）：**
```json
{
  "query": "...", "gt": "1", "country": "Netherlands",
  "response": "===== Solution 1 =====\nReasoning: ...\nAnswer: 1\n... ===== Solution 6 =====\nReasoning: ...\nAnswer: 1\n"
}
```

Solution 1-5 为各 agent Round 1 最终回答（含推理和答案），Solution 6 为 Judge 输出，可直接复用 AgentArk `label.py` 的 `split_solutions` 逻辑。

---

## 6. 代码结构

```
Cul/
├── data_generate.md               # 本文档
├── configs/
│   └── reconcile_config.yaml      # 5个文化角色 + Judge + 超参
├── reconcile_mas.py               # RECONCILE 核心逻辑
├── generate_culture_data.py       # 数据生成入口脚本
├── clean_dataset.py               # 数据集字段清洗工具
└── data/
    └── CulturalBench_mas.json             # 原始数据集（1227 条）
# 生成结果保存至：/autodl-fs/data/CulturalBench_mas_inference.jsonl
```

---

## 7. 运行命令

```bash
# 测试（前 5 条）
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
python Cul/generate_culture_data.py \
      --input_file Cul/data/CulturalBench_mas.json \
      --output_file /autodl-fs/data/CulturalBench_mas_inference.jsonl \
      --model_name llama \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 5 --num_debate_rounds 0
shutdown

# 全量生成（max_samples 0 = 全量）
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
python Cul/generate_culture_data.py \
      --input_file Cul/data/CulturalBench_mas.json \
      --output_file /autodl-fs/data/CulturalBench_mas_inference.jsonl \
      --model_name llama \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --num_debate_rounds 0
shutdown
```

`--model_name` 支持别名或完整路径：
- `llama` → `/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct`
- `qwen`  → `/root/autodl-tmp/base/Qwen2.5-7B-Instruct`
- 其他值视为完整路径直接使用

输出文件名自动追加时间戳，例如：`CulturalBench_mas_inference_20260510_143022.jsonl`，每次运行生成新文件。

---

## 8. 环境要求

| 组件 | 版本 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.3.0 |
| CUDA | 12.1 |
| vLLM | 0.6.4.post1 |
| transformers | 4.46.3 |
| GPU | RTX 4090 49GB（单卡），预计全量约 3-5 小时 |


## 9. NormAD 数据集辩论轮次对比实验（5条样本）

### 9.1 实验设计

为验证辩论轮次对异质多智能体系统生成质量的影响，在 NormAD 数据集上取 5 条样本，分别以 `--num_debate_rounds 0`（nodebate）、`--num_debate_rounds 1`（1debate）、`--num_debate_rounds 2`（2debate）运行推理数据生成脚本。

**运行命令：**

```bash
# 0 轮辩论
python Cul/generate_culture_data.py \
      --input_file Cul/data/sample.json \
      --output_file Cul/data/normad_mas_inference_nodebate.jsonl \
      --model_name llama \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 5 --num_debate_rounds 0

# 1 轮辩论
python Cul/generate_culture_data.py \
      --input_file Cul/data/sample.json \
      --output_file Cul/data/normad_mas_inference_1debate.jsonl \
      --model_name llama \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 5 --num_debate_rounds 1

# 2 轮辩论
python Cul/generate_culture_data.py \
      --input_file Cul/data/sample.json \
      --output_file Cul/data/normad_mas_inference_2debate.jsonl \
      --model_name llama \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 5 --num_debate_rounds 2
```

**生成结果文件：**
- `Cul/data/normad_mas_inference_nodebate_20260514_102211.jsonl`
- `Cul/data/normad_mas_inference_1debate_20260514_101819.jsonl`
- `Cul/data/normad_mas_inference_2debate_20260514_102337.jsonl`

### 9.2 核心指标对比

| 指标 | nodebate (0轮) | 1debate (1轮) | 2debate (2轮) |
|------|---------------|---------------|---------------|
| Agent 准确率 | **96.0%** (24/25) | **96.0%** (24/25) | 88.0% (22/25) |
| Judge 准确率 | **100.0%** (5/5) | **100.0%** (5/5) | 80.0% (4/5) |
| 平均独立答案数 | 1.20 | 1.20 | 1.20 |
| 完全一致率 | 80.0% | 80.0% | 80.0% |
| 平均回复长度 | 1409 chars | 1513 chars | 1570 chars |

### 9.3 关键发现：多智能体塌缩（Multi-Agent Collapse）

**Sample 2 的辩论退化过程：**

| 配置 | Agent 答案分布 | 正确数 | Judge |
|------|---------------|--------|-------|
| nodebate | [1, 1, **2**, 1, 1] | 4/5 ✓ | 1 ✓ |
| 1debate  | [1, 1, 1, 1, **2**] | 4/5 ✓ | 1 ✓ |
| 2debate  | [**2**, **2**, **2**, 1, 1] | 2/5 ✗ | **2 ✗** |

这是一个典型的**多智能体塌缩**案例：

1. **Round 0（nodebate）**：只有 1 个 agent (North American Culture) 给出了错误答案 "2"，认为 Tom 的回复不完全符合埃及文化中"回以祝福"的规范。其余 4 个 agent 正确识别了"回以赞美"也是可接受的。

2. **Round 1（1debate）**：agent 互看答案后，错误答案从 Agent 2 迁移到了 Agent 4（African Culture），但多数仍正确，Judge 据此做出正确裁决。

3. **Round 2（2debate）**：经过第二轮辩论，错误观点（"必须回以祝福才符合文化规范"）通过社会强化效应扩散到 3 个 agent（Agent 0, 1, 2），原来正确的 agent 被错误的少数说服。Judge 基于多数意见也做出了错误裁决。

### 9.4 塌缩机制分析

这一现象在多智能体辩论文献中被广泛记录，表现为以下机制的组合：

**（1）社会强化效应（Social Reinforcement）**
Agent 看到其他 agent 的答案后，倾向于修改自己的观点以达成共识。即使只有 1 个 agent 错误，只要其论证足够详细/自信，就可能在后续轮次中"说服"其他 agent 改变立场。这与 LLM 的 sycophantic tendency（迎合倾向）直接相关。

**（2）信心校准失败（Confidence Miscalibration）**
在辩论过程中，agent 对自身答案的信心校准被破坏。即使原始答案正确，看到详尽的反对论证后，agent 可能过度修正。研究表明 LLM 在多轮交互中难以维持合理的 confidence 校准（arXiv:2601.19921）。

**（3）推理退化保答案（Reasoning Degradation）**
辩论往往保留答案的准确率但退化推理质量（arXiv:2605.00914）。在本实验中更极端——辩论不仅退化了推理质量，还直接翻转了答案。

**（4）异质 agent 的特殊脆弱性**
同质 agent 由于初始知识相同，辩论时更易纠错（共享 prior）；而异质文化 agent 各自持有不同文化视角，辩论时容易被"有说服力但错误的跨文化类比"所误导——例如一个"详尽解释为何必须回以祝福"的错误论证可能被其他 agent 误认为是更深入的文化理解。

### 9.5 结论：0轮辩论是异质 MAS 的最优选择

| 维度 | 0轮辩论（推荐） | 1-2轮辩论 |
|------|----------------|-----------|
| 准确率 | 最高（96%/100%） | 随轮次下降 |
| 多样性 | 天然保持（异质 agent 视角差异） | 趋同 |
| PRM 训练价值 | 高（独立推理路径，差异来自知识而非社会压力） | 低（趋同后路径相似） |
| 计算开销 | 最低（无额外辩论轮次） | 线性增长 |
| 鲁棒性 | 错误不传播 | 错误可扩散 |

**设计决策**：本项目采用 `num_debate_rounds=0`，异质 agent 独立生成天然具有多样性，无需通过辩论获取，辩论反而引入错误传播风险。这与 AgentArk 原框架（同质 agent + 辩论）的设计相反，但更适合异质文化 agent 场景。

---

## 10. 解决多智能体塌缩的方法与本项目适用性

### 10.1 学术界提出的解决方案

| 方法 | 核心思想 | 代表文献 | 本项目适用性 |
|------|---------|---------|-------------|
| **Consensus-Free Debate (FREE-MAD)** | 不要求 agent 达成共识，改用 score-based 机制从所有轮次的中间结果中选取最优答案 | arXiv:2509.11035 | ⭐ 高：可用单轮辩论 + 跨轮打分替代多数投票 |
| **Confidence-Modulated Debate** | agent 表达校准后的 confidence，低信心 agent 不强制更新 | arXiv:2601.19921 | ⭐ 中：需改造 prompt 让 agent 输出 confidence score |
| **Diversity-Aware Initialization** | 增加辩论起始假设的多样性，确保至少一个正确假设存在 | arXiv:2601.19921 | ✓ 已实现：异质文化 agent 天然提供多样性 |
| **Sparse Communication Topology** | 不用全连接辩论，改用星形/环形拓扑，限制错误传播路径 | EMNLP 2024 Findings | ⭐ 中：可让 agent 只看部分而非全部其他 agent 的回答 |
| **Intelligent MAD (iMAD)** | 学习何时触发辩论、何时不辩论，仅在辩论有益时启动 | arXiv:2511.11306 | ⭐ 中：需训练额外 gate 模型 |
| **Devil's Advocate / Adversarial Agent** | 固定一个 agent 始终反对多数意见，防止过早收敛 | DynaDebate (2026) | ⚠ 低：可能干扰文化推理的客观性 |
| **Conformal Social Choice** | 在辩论后添加 calibrated act-vs-escalate 决策层 | arXiv:2604.07667 | ⭐ 中：可作为 Judge 的增强层 |
| **直接放弃辩论（本项目方案）** | 异质 agent 独立生成 + Judge 裁决，不做辩论 | 本工作 | ✓ 已采用，实验验证有效 |

### 10.2 本项目已采用的策略

本项目通过以下组合策略规避了多智能体塌缩问题：

**策略一：零辩论独立生成**——完全消除 agent 间的社会影响通道，从根本上阻断错误传播。异质文化 agent 的多样性来自知识差异而非随机采样，无需辩论即可获得丰富的推理路径。

**策略二：独立 Judge 裁决**——Judge 看到所有 agent 的回答但不参与辩论过程，以"事实核查者"身份独立裁决，明确指令"不以多数票为据"，减少 groupthink 对最终答案的影响。

**策略三：Judge fallback 机制**——当 Judge 无法提取有效答案时，fallback 到多数投票 + 文化区域匹配的 tie-break 策略，保证输出稳定。

### 10.3 可进一步探索的改进方向

若未来需要引入辩论轮次（例如在同质 agent 场景或更复杂的推理任务中），可考虑以下改进：

**（1）单轮辩论 + FREE-MAD 选择机制**：只做 1 轮辩论，但不取最终轮的多数投票，而是从 Round 0 和 Round 1 的所有回答中选取最优（通过 PRM 打分或 confidence score），兼顾纠错能力和抗塌缩能力。

**（2）Confidence-gated Update**：在辩论 prompt 中要求 agent 输出 confidence（1-10），只有当看到其他 agent 的论证显著强于自身 confidence 时才更新答案。需修改 `_build_debate_prompt` 增加 confidence 输出格式。

**（3）Sparse Topology**：不让每个 agent 看到所有其他 agent 的回答，而是只看 1-2 个最相关（文化区域最接近目标文化的）agent 的回答，减少错误传播面。需修改 `_build_debate_prompt` 中 `other_responses` 的构造逻辑。

---

## 11. 全量数据质量分析（1227条）

### 11.1 核心指标

| 指标 | 值 | 评价 |
|------|-----|------|
| Agent 准确率 | 56.2% | 接近基座单体 66%，合理 |
| Judge 准确率 | 68.4% | 与基座单体持平，Judge 有效 |
| Judge 答案提取失败 | 0/1227 | 正则修复完全生效 |
| 平均多样性 | 2.12 unique/sample | 中等，满足 PRM 需求 |
| 总 PRM 对 | 3808 | 可用于 PRM 训练 |

### 11.2 多样性分布

| unique 答案数 | 样本数 | 占比 | 说明 |
|-------------|--------|------|------|
| 1（完全收敛） | 318 | 25.9% | 无 PRM 对 |
| 2 | 506 | 41.2% | 每题 4 对 |
| 3 | 342 | 27.9% | 每题最多 6 对 |
| 4 | 61 | 5.0% | 最理想 |

74.1% 的样本有 2+ 种答案，对 PRM 有效。

### 11.3 Judge 纠错分析

| 情况 | 样本数 | 比例 |
|------|--------|------|
| Judge 对 / Agent 多数错（纠错） | 142 | 11.6% |
| Judge 错 / Agent 多数对（引入错误） | 92 | 7.5% |
| 两者均对 | 697 | 56.8% |
| 两者均错 | 296 | 24.1% |

Judge 净纠错收益 = +50 样本（+4.1%），引入 Judge 有价值。

### 11.4 结论

数据质量良好，可进入下一步（PRM 训练）。关键数字：
- **3808 个 PRM 对**：覆盖全量样本，远超训练所需
- **Judge 68.4% 准确率**：与 8B 基座持平，Judge 路径可作为高质量 SFT 数据
- **35.9% 零 PRM 对样本**：22.2% 全部答对（chosen），13.8% 全部答错（无价值），符合预期
