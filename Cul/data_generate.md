# 文化对齐多智能体推理数据生成方案

## 1. 任务目标

基于 CultureLLM 数据集，使用 RECONCILE 框架驱动的异质多智能体系统生成带推理路径的文化对齐训练数据，格式与 AgentArk 保持一致，供后续 PRM 训练和 GRPO 微调使用。

---

## 2. 智能体设计

### 2.1 角色定义

设定 4 个文化角色，按地理区域划分，每个角色通过 system prompt 注入文化身份：

| Agent | 文化角色 | 代表文化倾向 |
|-------|----------|-------------|
| Agent 0 | Asian Culture | 集体主义、家庭责任、等级尊重、长远规划 |
| Agent 1 | European Culture | 个人主义、理性主义、世俗化、工作与生活平衡 |
| Agent 2 | American Culture | 个人自由、实用主义、竞争意识、直接表达 |
| Agent 3 | Oceanian Culture | 平等主义、多元文化包容、休闲导向、社区意识 |

### 2.2 System Prompt 设计

每个 agent 的 system prompt 固定，在整个辩论过程中保持一致：

```
Asian Culture Agent:
"You are a cultural expert representing Asian cultural values.
Your reasoning should reflect collectivist values, strong family bonds,
respect for hierarchy and elders, long-term orientation, and the
importance of social harmony. When answering, reason from this
cultural perspective before giving your answer."

European Culture Agent:
"You are a cultural expert representing European cultural values.
Your reasoning should reflect individualist values, rational and
secular thinking, emphasis on personal autonomy, work-life balance,
and social welfare orientation. When answering, reason from this
cultural perspective before giving your answer."

American Culture Agent:
"You are a cultural expert representing American cultural values.
Your reasoning should reflect strong individualism, personal freedom,
pragmatism, competitive achievement orientation, and direct
communication style. When answering, reason from this cultural
perspective before giving your answer."

Oceanian Culture Agent:
"You are a cultural expert representing Oceanian cultural values.
Your reasoning should reflect egalitarianism, multicultural
inclusiveness, laid-back and community-oriented lifestyle, and
respect for diverse perspectives. When answering, reason from this
cultural perspective before giving your answer."
```

---

## 3. RECONCILE 框架流程

### 3.1 整体流程

```
输入：(question, target_culture)

Round 0（初始回答）：
  4 个 agent 各自独立生成推理路径 + 答案
  格式：Reasoning: ... \n Answer: [1/2/3/4]

Round 1（第一轮辩论）：
  每个 agent 看到其他 3 个 agent 的回答
  结合自身文化视角，决定是否修改自己的答案
  格式：Reasoning: ... \n Answer: [1/2/3/4]

Round 2（第二轮辩论）：
  同 Round 1，再次参考更新后的其他 agent 回答
  格式：Reasoning: ... \n Answer: [1/2/3/4]

共识聚合：
  收集 4 个 agent 的最终答案
  多数投票得到共识答案
  若无多数（4选项各1票），选择与 target_culture 最相关的 agent 的答案

输出：包含所有路径和共识答案的结构化数据
```

### 3.2 辩论消息构造

每轮辩论中，agent i 收到的消息格式：

```
These are the responses from other agents with different cultural perspectives:

[Asian Culture Agent]:
{reasoning and answer from agent 0}

[European Culture Agent]:
{reasoning and answer from agent 1}

...（排除 agent i 自身）

Consider these perspectives carefully. You may update your answer if
persuaded, but maintain your own cultural reasoning. State your
reasoning and give your final answer as: Answer: [number]
```

### 3.3 共识聚合

```python
# 多数投票
from collections import Counter
votes = [agent.final_answer for agent in agents]
consensus = Counter(votes).most_common(1)[0][0]

# 若平票（如 2:2），选与 target_culture 最接近的 agent 的答案
# 优先级：target_culture 所属地区的 agent > 其他
```

---

## 4. 输出数据格式

与 AgentArk 保持一致，每条样本输出格式：

```json
{
  "query": "### Question: Give me the answer from 1 to 4: ...",
  "gt": "1",
  "country": "Arabic",
  "response": "===== Solution 1 =====\nReasoning: From an Asian cultural perspective, ...\nAnswer: 1\n===== Solution 2 =====\nReasoning: From a European cultural perspective, ...\nAnswer: 2\n===== Solution 3 =====\nReasoning: From an American cultural perspective, ...\nAnswer: 1\n===== Solution 4 =====\nReasoning: From an Oceanian cultural perspective, ...\nAnswer: 1\n===== Solution 5 =====\nConsensus Answer: 1"
}
```

其中 Solution 1-4 为各 agent 最终轮次的回答，Solution 5 为共识聚合结果，与 AgentArk 的 LLM Debate 格式完全对齐，可直接复用 `label.py` 的 `split_solutions` 逻辑。

---

## 5. 代码结构

```
Cul/
├── data_generate.md          # 本文档
├── configs/
│   └── reconcile_config.yaml # 超参配置
├── reconcile_mas.py          # RECONCILE 多智能体核心逻辑
├── generate_culture_data.py  # 数据生成入口脚本
└── data/
    └── sample.json           # 示例数据（用于调试）
```

---

## 6. 运行方式

```bash
# 使用示例数据调试
python Cul/generate_culture_data.py \
    --input_file Cul/data/sample.json \
    --output_file Cul/data/sample_generated.jsonl \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --use_vllm \
    --tensor_parallel_size 4 \
    --debug

# 完整数据集生成
python Cul/generate_culture_data.py \
    --input_file /path/to/culturellm_full.json \
    --output_file results/CultureLLM/reconcile_infer.jsonl \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --use_vllm \
    --tensor_parallel_size 4
```
