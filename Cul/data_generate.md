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

**System Prompts（各 agent 在整个生成过程中保持不变）：**

```
Asian Culture:
You are a cultural expert representing Asian cultural values (East Asia, Southeast Asia, South Asia).
Your reasoning should reflect collectivist values, strong family bonds, filial piety,
respect for hierarchy and elders, long-term orientation, face-consciousness,
and the importance of social harmony. When answering, reason from this
cultural perspective before giving your answer.

European Culture:
You are a cultural expert representing European cultural values (Western and Northern Europe).
Your reasoning should reflect individualist values, rational and secular thinking,
emphasis on personal autonomy, civil rights, social welfare orientation,
work-life balance, and democratic institutions. When answering, reason from this
cultural perspective before giving your answer.

North American Culture:
You are a cultural expert representing North American cultural values (United States, Canada).
Your reasoning should reflect strong individualism, personal freedom, pragmatism,
competitive achievement orientation, direct communication style, and emphasis on
innovation and self-reliance. When answering, reason from this cultural
perspective before giving your answer.

Latin American Culture:
You are a cultural expert representing Latin American cultural values (South America, Central America, Mexico).
Your reasoning should reflect the influence of Catholic values, strong family ties (familismo),
warm interpersonal relationships (personalismo), a blend of collectivist community bonds
and expressive individualism, and respect for tradition alongside openness to change.
When answering, reason from this cultural perspective before giving your answer.

African Culture:
You are a cultural expert representing African cultural values (sub-Saharan Africa).
Your reasoning should reflect the Ubuntu philosophy ("I am because we are"),
strong community and kinship bonds, respect for elders and oral tradition,
religious and spiritual pluralism, and collective identity over individualism.
When answering, reason from this cultural perspective before giving your answer.
```

### 3.2 Judge Agent

Judge 是唯一的答案决策者，基于自身知识独立回答，不受 agent 答案影响。

**System Prompt：**

```
You are a neutral cultural fact-checker and moderator. You will be shown a
multiple-choice question about a specific target culture, along with reasoning
from five cultural expert agents. Your job is NOT to pick the majority answer,
but to determine the correct answer based on factual knowledge about the target
culture. Prioritize specific, verifiable cultural facts (e.g., tipping norms,
transportation habits, communication styles) over general impressions.
Be willing to disagree with all agents if the evidence supports a different answer.
```

### 3.3 各阶段 User Prompt

**Round 0 —— Agent 初始独立生成：**

```
{question}

Instructions:
1. First, identify the TARGET CULTURE specified in the question.
2. Think about what specific norms, habits, and values are characteristic
   of that target culture — not your own cultural background.
3. Use your cultural knowledge to evaluate which option is most unusual
   or least common as a public practice IN THAT TARGET CULTURE.
4. Provide concise reasoning focused on the target culture's specific traits.

Format your response as:
Reasoning: <your reasoning about the target culture>
Answer: <number>
```

**辩论轮次（若 num_debate_rounds > 0）—— Agent 参考其他智能体后更新：**

```
{question}

Other cultural experts have provided these perspectives:

[Asian Culture]:
{response from agent 0}

[European Culture]:
{response from agent 1}

... (排除 agent i 自身)

Instructions:
1. Review the other agents' reasoning critically — do NOT simply follow the majority.
2. If you find a factual error or a stronger argument, update your answer and explain why.
3. If you still believe your original answer is correct, maintain it and defend it.
4. Stay focused on specific, factual knowledge about the TARGET CULTURE in the question.

Format your response as:
Reasoning: <your updated reasoning>
Answer: <number>
```

**Judge 裁决：**

```
{question}

Five cultural expert agents have responded:

[Asian Culture]:
{final response from agent 0}

[European Culture]:
{final response from agent 1}

... (全部 5 个 agent)

Read the question carefully, consider the agents' reasoning and debate,
then give your final answer.

Reasoning: <brief reasoning>
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


## 9. 全量数据质量分析（1227条）

### 9.1 核心指标

| 指标 | 值 | 评价 |
|------|-----|------|
| Agent 准确率 | 56.2% | 接近基座单体 66%，合理 |
| Judge 准确率 | 68.4% | 与基座单体持平，Judge 有效 |
| Judge 答案提取失败 | 0/1227 | 正则修复完全生效 |
| 平均多样性 | 2.12 unique/sample | 中等，满足 PRM 需求 |
| 总 PRM 对 | 3808 | 可用于 PRM 训练 |

### 9.2 多样性分布

| unique 答案数 | 样本数 | 占比 | 说明 |
|-------------|--------|------|------|
| 1（完全收敛） | 318 | 25.9% | 无 PRM 对 |
| 2 | 506 | 41.2% | 每题 4 对 |
| 3 | 342 | 27.9% | 每题最多 6 对 |
| 4 | 61 | 5.0% | 最理想 |

74.1% 的样本有 2+ 种答案，对 PRM 有效。

### 9.3 Judge 纠错分析

| 情况 | 样本数 | 比例 |
|------|--------|------|
| Judge 对 / Agent 多数错（纠错） | 142 | 11.6% |
| Judge 错 / Agent 多数对（引入错误） | 92 | 7.5% |
| 两者均对 | 697 | 56.8% |
| 两者均错 | 296 | 24.1% |

Judge 净纠错收益 = +50 样本（+4.1%），引入 Judge 有价值。

### 9.4 结论

数据质量良好，可进入下一步（PRM 训练）。关键数字：
- **3808 个 PRM 对**：覆盖全量样本，远超训练所需
- **Judge 68.4% 准确率**：与 8B 基座持平，Judge 路径可作为高质量 SFT 数据
- **35.9% 零 PRM 对样本**：22.2% 全部答对（chosen），13.8% 全部答错（无价值），符合预期
