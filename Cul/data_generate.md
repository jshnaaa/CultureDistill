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
# 测试（前 5 条，含 Judge）
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
python Cul/generate_culture_data.py \
      --input_file Cul/data/CulturalBench_mas.json \
      --output_file /autodl-fs/data/CulturalBench_mas_inference.jsonl \
      --model_name llama \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 5 --num_debate_rounds 0 \
      --include_judge true
shutdown

# 全量生成（含 Judge，max_samples 0 = 全量）
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
python Cul/generate_culture_data.py \
      --input_file Cul/data/CulturalBench_mas.json \
      --output_file /autodl-fs/data/CulturalBench_mas_inference.jsonl \
      --model_name llama \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --num_debate_rounds 0 \
      --include_judge true
shutdown

# 全量生成（不含 Judge，仅 Agent 路径用于蒸馏）
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
python Cul/generate_culture_data.py \
      --input_file Cul/data/CulturalBench_mas.json \
      --output_file /autodl-fs/data/CulturalBench_mas_inference_nojudge.jsonl \
      --model_name llama \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --num_debate_rounds 0 \
      --include_judge false
shutdown
```

**参数说明：**

`--model_name` 支持别名或完整路径：
- `llama` → `/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct`
- `qwen`  → `/root/autodl-tmp/base/Qwen2.5-7B-Instruct`
- 其他值视为完整路径直接使用

`--include_judge`（默认 `true`）：
- `true`：生成 Solution 1-5（Agent）+ Solution 6（Judge），Judge 综合推理轨迹参与后续 SFT/GRPO 蒸馏
- `false`：仅生成 Solution 1-5（Agent），不执行 Judge 推理。用于消融实验（验证 Judge 路径对蒸馏的增量价值）

输出文件名自动追加时间戳，例如：`CulturalBench_mas_inference_20260510_143022.jsonl`，每次运行生成新文件。

**`--include_judge` 的蒸馏影响：**

| include_judge | SFT 蒸馏数据 | GRPO 行为 | 适用场景 |
|---------------|-------------|-----------|---------|
| true（默认） | Agent 正确路径 + Judge 正确路径 | 无影响（GRPO 在线生成，不使用 MAS 数据） | 完整方案，验证创新点2（Judge-anchored Distillation） |
| false | 仅 Agent 正确路径 | 无影响 | 消融对比：去掉 Judge 后 SFT 质量是否下降 |

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

---

## 12. HFA-C²N：基于主场权威激活的跨文化动态协商范式

### 12.1 动机与核心洞察

传统 RECONCILE 框架中，所有 Agent 无论讨论什么国家的题目，地位都是平等的。这在科学/逻辑推理任务中合理，但在文化对齐任务中存在根本性缺陷——**文化知识具有强烈的"属地性"和"不对称性"**。

例如：关于中国春节的知识，东亚文化 Agent 的话语权天然应该高于欧洲文化 Agent；关于巴西狂欢节的知识，拉美文化 Agent 比北美 Agent 更具权威性。然而在传统 RECONCILE 中，一个对目标文化一知半解的客场 Agent 与一个对目标文化了如指掌的主场 Agent 享有相同的投票权和影响力，这会导致"西方语料主导型错误"——在小众、非西方国家的题目上，被训练数据中占主导地位的西方视角带偏。

### 12.2 方法论定位

HFA-C²N（Home-Field Authority-Activated Cross-Cultural Negotiation）是针对文化对齐任务量身定制的算法架构创新，核心思想是：**根据目标国家动态调整 Agent 的权威度**，引入"主场/客场"不对称机制，使多智能体系统在文化题目上产生更高质量的推理数据。

**与"简单搬用 RECONCILE"的本质区别**：
- RECONCILE：所有 Agent 平等 → 多数投票 → 均质推理路径
- HFA-C²N：动态权威激活 → 主场确权 + 客场审视 → 结构化对比推理路径

这一改进直接回应了"仅是应用型论文，无方法论创新"的审稿质疑。

### 12.3 核心机制设计

#### 12.3.1 主场权威激活（Home-Field Authority Activation）

系统解析输入数据中的目标国家（如 China），自动将对应文化背景的 Agent（如 East-Asian Agent）标记为 **"主场文化守护者"（Host-Culture Guardian）**，其余 Agent 标记为 **"跨文化审视者"（Cross-Cultural Auditors）**。

**匹配规则**：基于 config 中每个 Agent 的 `region_keywords` 列表进行模糊匹配。例如 target_country="Vietnam" 匹配到 East Asian Culture Agent 的 keyword "vietnam"。

#### 12.3.2 话语权不对称设计

| 维度 | Host-Culture Guardian | Cross-Cultural Auditors |
|------|----------------------|------------------------|
| 生成顺序 | Phase 1（优先生成） | Phase 2（看到 Guardian 后生成） |
| 采样温度 | 0.5（低温精确） | 0.9（高温多样） |
| System Prompt | 权威确认/纠偏 | 对比分析/承认不确定性 |
| Judge 权重 | 高权重 + 一票否决权 | 辅助参考 |
| 推理角色 | "我确认选项 X 在目标文化中正确，因为..." | "从我的文化视角看，可能是 Y，但对目标文化不确定..." |

#### 12.3.3 两阶段结构化协商（Structured Negotiation）

```
输入：(question, target_country)

Step 1: 主场识别 — detect_guardian(target_country) → Agent_i

Step 2: Phase 1 — Guardian 独立生成（低温，权威分析）
  输出：确认具体文化事实，解释为何选该选项，纠正潜在误解

Step 3: Phase 2 — Auditors 生成（看到 Guardian 的分析后）
  输出：从各自文化视角提供对比/审视，同意则解释跨文化相似性，
       不同意则给出具体反驳证据（同时承认 Guardian 的主场权威）

Step 4: Judge — 带权威权重裁决
  规则：Guardian 有一票否决权（当 Guardian 提供具体证据时，
       即使其他 4 个 Auditor 持不同意见，仍优先采信 Guardian）

输出：Solution 1-5 [GUARDIAN/AUDITOR] + Solution 6 [JUDGE]
```

#### 12.3.4 Guardian 一票否决权（Veto Power）机制

在 Judge 裁决和 fallback 投票中：
- 如果 Guardian 的答案与多数不同，但 Guardian 提供了具体文化证据 → 采信 Guardian
- 如果 Guardian 的答案与多数相同 → 直接确认
- 如果 Guardian 未能给出有效答案 → 退化为标准多数投票

### 12.4 推理路径的蒸馏价值提升

传统 RECONCILE 生成的 CoT 数据是"各 Agent 各自站队"的扁平推理。HFA-C²N 生成的推理数据具有**结构化对比信息**，蒸馏价值显著更高：

**客场 Auditor 的推理路径（增加蒸馏数据熵）**：
```
"从欧洲文化的视角看，庆祝活动常有饮酒习俗（选项2），但结合中国过年的上下文，
作为非主场文化观察者，我不确定这是否为最传统做法。Host-Culture Guardian 确认了
选项1（舞狮），这与我对亚洲集体性庆典的印象一致。我同意 Guardian 的判断。"
```

**主场 Guardian 的推理路径（提供精准对齐信号）**：
```
"作为东亚文化守护者，我确认选项1（舞狮）在中国春节具有普适性的传统意义。
选项4（绿包/green packet）属于东南亚部分伊斯兰文化圈（如马来西亚/新加坡穆斯林社区）
的 Hari Raya 节日习俗，在中国文化中并不存在。两者存在明确的文化地理边界差异。"
```

**蒸馏收益**：单体模型通过 SFT 学习这种结构化对比数据后，不仅学会"中国过年选1"，还能学会"为什么不能选4"——即学到了**文化逻辑的边界感**（Cultural Boundary Awareness）。

### 12.5 代码结构

```
Cul/
├── configs/
│   ├── reconcile_config.yaml        # 原 RECONCILE 配置（保留）
│   └── hfa_c2n_config.yaml          # HFA-C²N 配置（新增）
│                                     #   - 每个 Agent 含 guardian_prompt + auditor_prompt
│                                     #   - region_keywords 用于主场匹配
│                                     #   - Judge system prompt 含权威权重说明
├── hfa_c2n_mas.py                   # HFA-C²N 核心推理引擎（新增）
│                                     #   - HFA_C2N_MAS 类
│                                     #   - detect_guardian(): 主场识别
│                                     #   - 两阶段 batch inference
│                                     #   - Guardian veto fallback
├── generate_hfa_c2n_data.py         # HFA-C²N 数据生成入口（新增）
│                                     #   - 参数：--negotiation_rounds, --include_judge
│                                     #   - 兼容现有 convert_sample / resume 逻辑
├── reconcile_mas.py                 # 原 RECONCILE 引擎（保留不动）
└── generate_culture_data.py         # 原 RECONCILE 入口（保留不动）
```

### 12.6 运行命令

```bash
# 测试（5 条样本，标准协商模式：Auditor 看到 Guardian）
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
python Cul/generate_hfa_c2n_data.py \
      --input_file Cul/data/CulturalBench_mas.json \
      --output_file /autodl-fs/data/hfa_c2n_inference.jsonl \
      --model_name llama \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 5 --negotiation_rounds 1 \
      --include_judge true
shutdown

# 全量生成（NormAD 数据集）
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
python Cul/generate_hfa_c2n_data.py \
      --input_file Cul/data/sample.json \
      --output_file /autodl-fs/data/qwen/normad_hfa_c2n_inference.jsonl \
      --model_name qwen \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --negotiation_rounds 1 \
      --include_judge true
shutdown

# 消融：无协商（Guardian 和 Auditor 独立生成，不互看）
python Cul/generate_hfa_c2n_data.py \
      --input_file Cul/data/sample.json \
      --output_file /autodl-fs/data/hfa_c2n_independent.jsonl \
      --model_name qwen \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --negotiation_rounds 0 \
      --include_judge true

# 消融：无 Judge（仅 Agent 路径）
python Cul/generate_hfa_c2n_data.py \
      --input_file Cul/data/sample.json \
      --output_file /autodl-fs/data/hfa_c2n_nojudge.jsonl \
      --model_name qwen \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --negotiation_rounds 1 \
      --include_judge false
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--negotiation_rounds` | 1 | 协商轮次。0=独立生成（Auditor 不看 Guardian），1=标准协商（Auditor 看到 Guardian 分析后生成） |
| `--include_judge` | true | 是否包含 Judge 裁决。false 时仅输出 Solution 1-5 |
| `--model_name` | — | `llama`/`qwen`/完整路径 |
| `--max_samples` | 0 | 0=全量 |

### 12.7 输出格式

```json
{
  "query": "### Question: ...",
  "gt": "1",
  "country": "China",
  "guardian_idx": 0,
  "guardian_name": "East Asian Culture",
  "response": "===== Solution 1 [GUARDIAN] =====\nReasoning: ...\nAnswer: 1\n===== Solution 2 [AUDITOR] =====\n...\n===== Solution 6 [JUDGE] =====\n..."
}
```

**与原 RECONCILE 格式的区别**：
- Solution 标题包含 `[GUARDIAN]`/`[AUDITOR]`/`[JUDGE]` 角色标签
- 额外输出 `guardian_idx` 和 `guardian_name` 字段，便于下游分析
- Guardian 的推理路径包含权威确认语言，Auditor 包含对比/不确定性表达

### 12.8 与 SFT/GRPO 管道的衔接

HFA-C²N 生成的数据完全兼容现有的蒸馏管道：

**SFT 阶段**：`train_sft.py` 的 `split_solutions()` 使用正则 `===== Solution \d+ =====` 切分，`[GUARDIAN]`/`[AUDITOR]` 标签不影响切分。Guardian 路径和 Judge 路径作为高优先级监督数据。

**PRM 阶段**：`label_data.py` 对每条路径打分构造 pairwise 对。Guardian 路径由于文化精度更高，预计会获得更高的 R_cultural 分数，自然形成更多 chosen 样本。

**GRPO 阶段**：不受影响——GRPO 使用 prompt 池在线生成，不依赖 MAS 数据内容。

### 12.9 实验设计：消融验证

#### 12.9.1 多智能体层面消融（验证数据生成质量）

| 实验组 | 方法 | 预期效果 |
|--------|------|---------|
| Baseline（RECONCILE） | 原 `generate_culture_data.py`，平等 Agent | 全球平均准确率尚可，小众国家偏低 |
| HFA-C²N（negotiation=0） | 主场/客场区分，但独立生成 | 小众国家准确率提升（Guardian 不被客场带偏） |
| HFA-C²N（negotiation=1） | 完整协商（Auditor 看到 Guardian） | 准确率最高 + 推理路径含对比结构 |

**核心指标**：按国家分组的准确率。预期发现——HFA-C²N 在非西方、小众国家（如越南、肯尼亚、埃及）题目上准确率暴涨，因为这些题目不再被西方语料主导的 Agent 带偏。

#### 12.9.2 单体蒸馏后消融（验证单体学得更好）

| 实验组 | SFT 数据来源 | 验证指标 |
|--------|-------------|---------|
| SFT (RECONCILE data) | 原 RECONCILE 推理数据 | val_accuracy, Cultural Sensitivity |
| SFT (HFA-C²N data) | HFA-C²N 推理数据 | val_accuracy, Cultural Sensitivity |

**预期结论**：学习了"主客场思辨/确权路径"的单体模型，在未见过的文化测试集上（Zero-shot OOD）表现出：
- 更高的准确率（尤其小众国家）
- 更强的文化一致性（推理路径引用具体文化事实）
- 更低的文化混淆率（不会用西方逻辑解释东方文化）

### 12.10 LLM 调用量估算

```
设 N = 样本数（如 NormAD 2633 条）

Phase 1（Guardian）：N × 1 = N 次
Phase 2（Auditor）：N × 4 = 4N 次
Phase 3（Judge）：  N × 1 = N 次
总计：6N 次（与原 RECONCILE 的 6N 相同，无额外开销）

注意：Phase 1 和 Phase 2 是串行的（Phase 2 依赖 Phase 1 的输出），
但 Phase 1 内部和 Phase 2 内部都是全并行 batch 推理。
总计算量与原 RECONCILE 基本一致。
```
