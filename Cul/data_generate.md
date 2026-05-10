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

### 3.2 Judge Agent

辩论结束后，Judge 读取所有 agent 的最终回答及题目中的 target culture，输出带推理的最终答案。相比纯多数投票，Judge 能根据 target culture 做有偏向的裁决，并在平票时给出有据可查的理由。Judge 失效（无法提取有效答案）时自动 fallback 到多数投票。

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

Solution 1-5 为各 agent Round 2 最终回答，Solution 6 为 Judge 输出，可直接复用 AgentArk `label.py` 的 `split_solutions` 逻辑。

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
sh git.sh
python Cul/generate_culture_data.py \
      --input_file Cul/data/CulturalBench_mas.json \
      --output_file /autodl-fs/data/CulturalBench_mas_inference.jsonl \
      --model_name /root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct \
      --use_vllm --tensor_parallel_size 2 --max_samples 5

# 全量生成（max_samples 0 = 全量）
cd autodl-tmp/distill
sh git.sh
python Cul/generate_culture_data.py \
      --input_file Cul/data/CulturalBench_mas.json \
      --output_file /autodl-fs/data/CulturalBench_mas_inference.jsonl \
      --model_name /root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct \
      --use_vllm --tensor_parallel_size 2 --max_samples 0
```

支持断点续传：中断后直接重跑，已处理的样本自动跳过。

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
