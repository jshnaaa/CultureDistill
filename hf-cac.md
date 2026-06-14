# HF-CAC：一种新的多智能体协作范式（创新点一）

HF-CAC（Home-Field Culture-Activated Collaboration）是提出的面向文化对齐任务的多智能体协作新范式。

核心思想：针对文化知识的"属地性"和"不对称性"特征，引入"主场/客场"动态权威机制——根据目标国家自动激活对应文化背景的 Agent 作为主场守护者（Guardian），赋予其更高话语权和一票否决权，其余 Agent 作为跨文化审视者（Auditor）提供对比视角。

## 2. HF-CAC：基于主场文化激活的多智能体协作范式

### 2.1 动机

传统 RECONCILE 框架中，所有 Agent 无论讨论什么国家的题目，地位都是平等的。这在科学/逻辑推理任务中合理，但在文化对齐任务中存在根本性缺陷——文化知识具有强烈的"属地性"和"不对称性"。

例如：关于中国春节的知识，东亚文化 Agent 的话语权天然应该高于欧洲文化 Agent；关于巴西狂欢节的知识，拉美文化 Agent 比北美 Agent 更具权威性。然而在传统 RECONCILE 中，一个对目标文化一知半解的客场 Agent 与一个对目标文化了如指掌的主场 Agent 享有相同的投票权和影响力，这会导致"西方语料主导型错误"——在小众、非西方国家的题目上，被训练数据中占主导地位的西方视角带偏。

### 2.2 方法论

HF-CAC 是针对文化对齐任务量身定制的算法架构创新，核心思想是：根据目标国家动态调整 Agent 的权威度，引入"主场/客场"不对称机制，使多智能体系统在文化题目上产生更高质量的推理数据。

与"简单搬用 RECONCILE"的本质区别：
- RECONCILE：所有 Agent 平等 → 多数投票 → 均质推理路径
- HF-CAC：动态权威激活 → 主场确权 + 客场审视 → 结构化对比推理路径

### 2.3 机制设计

#### 2.3.1 主场权威激活

直接读取数据集中的 `country` 字段获取目标国家。然后根据目标国家自动将对应文化背景的 Agent（如 East-Asian Agent）标记为"主场文化守护者"（Host-Culture Guardian），其余 Agent 标记为"跨文化审视者"（Cross-Cultural Auditors）。

匹配规则：基于 config 中每个 Agent 的 `region_keywords` 列表，将具体国家名映射到对应的文化圈 Agent。例如 country="egypt" 匹配到 Islamic & Middle-Eastern Culture Agent。

#### 2.3.2 话语权不对称

| 维度 | Host-Culture Guardian | Cross-Cultural Auditors |
|------|----------------------|------------------------|
| 生成顺序 | Phase 1（优先生成） | Phase 2（看到 Guardian 后生成） |
| 采样温度 | 0.5（低温精确） | 0.9（高温多样） |
| System Prompt | 权威确认/纠偏 | 对比分析/承认不确定性 |
| Judge 权重 | 高权重 + 一票否决权 | 辅助参考 |
| 推理角色 | "我确认选项 X 在目标文化中正确，因为..." | "从我的文化视角看，可能是 Y，但对目标文化不确定..." |

#### 2.3.3 两阶段结构化协商

```
输入：(question, target_country)    # target_country 直接取自数据集的 country 字段

Step 1: 主场识别 — detect_guardian(target_country) → Agent_i

Step 2: Phase 1 — Guardian 独立生成（低温，权威分析）
  输出：确认具体文化事实，解释为何选该选项，纠正潜在误解

Step 3: Phase 2 — Auditors 生成（看到 Guardian 的分析后）
  输出：从各自文化视角提供对比/审视，同意则解释跨文化相似性，不同意则给出具体反驳证据（同时承认 Guardian 的主场权威）

Step 4: Judge — 带权威权重裁决
  规则：Guardian 有一票否决权（当 Guardian 提供具体证据时，即使其他 5 个 Auditor 持不同意见，仍优先采信 Guardian）

输出：Solution 1-6 [GUARDIAN/AUDITOR] + Solution 7 [JUDGE]
```

#### 2.3.4 Guardian 一票否决权（Veto Power）机制

在 Judge 裁决和 fallback 投票中：
- 如果 Guardian 的答案与多数相同 → 直接确认
- 如果 Guardian 的答案与多数不同，但 Guardian 提供了具体文化证据 → 采信 Guardian
- 若 Guardian 证据不足，即失效 → 激活 Judge 启发式跨文化仲裁机制

**Guardian 失效的判定条件**：(a) Guardian 回答中提取不到有效答案（格式崩溃、输出截断），OR (b) Guardian 的推理中包含明确的不确定性放弃标记（如 "I'm not sure"、"I don't have enough knowledge"、推理内容为空）。

**跨文化谱系相似度仲裁（Cultural Affinity Arbitration）**：

当主场 Guardian 未能给出清晰答案时，说明该国的文化知识可能属于"极度冷门"或"长尾知识"。此时 Judge 不做简单的多数投票计数，而是升级为"基于文化谱系的客观仲裁者"，通过跨文化亲缘度比对进行主动判断：

判断逻辑：内置一张 6×6 的文化亲缘度矩阵（Cultural Affinity Matrix），刻画任意两个文化圈之间的"谱系距离"。当 Guardian 失效时，Judge 在裁决中对各 Auditor 的意见按亲缘度加权——与目标文化亲缘度更高的 Auditor 的推理获得更大权重。

示例：若题目考的是埃及（属于 Islamic & Middle-Eastern 文化圈），Guardian（伊斯兰 Agent）失效。此时 Judge 参照亲缘度矩阵：Sub-Saharan African Agent（亲缘度 0.5，近亲地缘）的推理权重 > South & Southeast Asian Agent（亲缘度 0.3）> Western & Anglo-Saxon Agent（亲缘度 0.1）。即使投 "no" 的 Agent 更多，只要亲缘度更高的 Agent 给出了带有具体文化证据的不同答案，Judge 倾向于采信高亲缘度 Agent 的判断。

亲缘度矩阵设计原则：基于地理邻近性、宗教传统共享度、历史交流深度三个维度综合打分（0-1），硬编码在配置文件中，确保确定性和可复现性。

### 2.4 运行命令

```bash
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
python Cul/generate_hf_cac_data.py \
  --input_file /autodl-fs/data/culturalBench_mas.json \
  --output_file /autodl-fs/data/llama/culturalbench_hf_cac_6agents.jsonl \
  --config_path Cul/configs/hf_cac_config_culturalbench.yaml \
  --model_name llama \
  --use_vllm --tensor_parallel_size 2 \
  --max_samples 0 \
  --negotiation_rounds 0 --num_agents 6 \
  --temp_ladder true --include_judge true

python Cul/generate_hf_cac_data.py \
  --input_file /autodl-fs/data/cultureLLM_mas.json \
  --output_file /autodl-fs/data/qwen/culturellm_check.jsonl \
  --model_name qwen \
  --use_vllm --tensor_parallel_size 2 \
  --max_samples 300 --random_sample true --seed 42 \
  --negotiation_rounds 1 --include_judge true
  
python Cul/generate_hf_cac_data.py \
    --input_file /autodl-fs/data/cultureLLM_mas.json \
    --output_file /autodl-fs/data/qwen/culturellm_hf_cac_6agents.jsonl \
    --model_name qwen \
    --use_vllm --tensor_parallel_size 2 \
    --max_samples 0 --negotiation_rounds 1 \
    --include_judge true --num_agents 6

python Cul/generate_hf_cac_data.py \
      --input_file /autodl-fs/data/blend_mas_after.json \
      --output_file /autodl-fs/data/qwen/blend_hf_cac_5agents.jsonl \
      --config_path Cul/configs/hf_cac_config_blend.yaml \
      --model_name qwen \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --negotiation_rounds 1 \
      --include_judge true --num_agents 5
shutdown
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--negotiation_rounds` | 1 | 协商轮次。0=独立生成（Auditor 不看 Guardian），1=标准协商 |
| `--include_judge` | true | 是否包含 Judge 裁决。false 时仅输出 Solution 1-N |
| `--num_agents` | 6 | 文化智能体数量（2/3/4/5/6）。用于消融实验，< 6 时自动合并文化亲缘角色 |
| `--config_path` | None | 手动指定配置文件路径。不指定时根据数据集自动检测 |


**数据划分（生成 pkl 文件）**

```bash
python Cul/split_data.py \
    --input /autodl-fs/data/qwen/normad_hf_cac_inference_20260525_101428.jsonl \
    --output /autodl-fs/data/qwen/normad_splits.pkl \
    --seed 42
    
python Cul/split_data.py \
    --input /autodl-fs/data/llama/normad_hf_cac_inference_20260527_200952.jsonl \
    --output /autodl-fs/data/llama/normad_splits.pkl \
    --seed 42

python Cul/split_data.py \
    --input /autodl-fs/data/llama/culturalBench_hf_cac_3agents_20260606_192326.jsonl \
    --output /autodl-fs/data/llama/culturalBench_splits.pkl \
    --seed 42

python Cul/split_data.py \
    --input /autodl-fs/data/qwen/culturalBench_hf_cac_3agents_20260606_192326.jsonl \
    --output /autodl-fs/data/qwen/culturalBench_splits.pkl \
    --seed 42

python Cul/split_data.py \
    --input /autodl-fs/data/qwen/blend_hf_cac_3agents_<timestamp>.jsonl \
    --output /autodl-fs/data/qwen/blend_splits.pkl \
    --seed 42
```

### 2.5 输出数据格式

```json
{
  "query": "### Question: ...",
  "gt": "1",
  "country": "China",
  "guardian_idx": 0,
  "guardian_name": "East Asian Culture",
  "response": "===== Solution 1 [GUARDIAN] =====\nReasoning: ...\nAnswer: 1\n===== Solution 2 [AUDITOR] =====\n...\n===== Solution 7 [JUDGE] =====\n..."
}
```

### 2.6 Baseline

#### 2.6.0 Base & Role-play

**简介**：使用单个基座模型直接作答的两个 Baseline。

- **base**：zero-shot，system prompt 仅为通用助手，模型只根据题目作答。
- **role**：角色扮演，按目标国家注入「你是某国文化专家」的 system prompt，依据该文化背景作答。

**输出**：结果 JSONL 保存每条 `query/country/gt/pred/response`；同名 `.metrics.json` 保存准确率、各国别准确率及答案分布（输出文件名会自动追加时间戳）。

| 参数 | 含义 |
|------|------|
| `--input_file` | 数据集路径（`culturalBench_mas.json` / `normad_mas.json` / `blend_mas_after.json`）|
| `--output_file` | 结果输出路径，文件名自动追加时间戳，指标文件同名 + `.metrics.json` |
| `--model_name` | 基座 `qwen` / `llama`（或完整本地路径）|
| `--method` | `base`（zero-shot）/ `role`（角色扮演）|
| `--tensor_parallel_size` | vLLM 张量并行度 |
| `--max_samples` | 处理样本数，`0`=全部 |

```bash
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
python Cul/single_data.py \
--input_file /autodl-fs/data/blend_mas_after.json \
--output_file /autodl-fs/data/blend_llama_role.json \
--model_name llama --method role \
--tensor_parallel_size 2 --max_samples 0

python Cul/single_data.py \
--input_file /autodl-fs/data/blend_mas_after.json \
--output_file /autodl-fs/data/blend_qwen_role.json \
--model_name qwen --method base \
--tensor_parallel_size 2 --max_samples 0
```

#### 2.6.1 RECONCILE

```bash
python Cul/generate_culture_data.py \
    --input_file /autodl-fs/data/normad_mas.json \
    --output_file /autodl-fs/data/qwen/normad_reconcile_inference.jsonl \
    --model_name qwen \
    --use_vllm --tensor_parallel_size 2 \
    --max_samples 0 --num_debate_rounds 1 --include_judge true
```

| 参数 | 含义 |
|------|------|
| `--config_path` | 可选，RECONCILE 配置文件路径（默认 `configs/reconcile_config.yaml`）|
| `--num_debate_rounds` | 辩论轮数（覆盖 config 中的值，0=无辩论仅独立推理）|
| `--include_judge` | 是否包含 Judge 裁决（`true`/`false`）|

#### 2.6.2 MAD 

**简介**：MAD 是多智能体辩论框架，通过两个 LLM Agent 对文化场景进行辩论来达成更准确的文化对齐判断。两种变体：

1. **Debate-Only**（A.3）：两个 Agent 独立给出初始判断 → 交换反馈 → 基于反馈给出最终判断 → 由 Judge LLM 仲裁分歧
2. **Self-Reflect+Debate**（A.4）：两个 Agent 独立给出初始判断 → 各自选择自我反思(A)或辩论(B) → 执行所选动作 → 基于反馈给出最终判断 → Judge 仲裁

**代码目录**：`MAD/`

```
MAD/
├── mad_common.py               # 共享工具（数据解析、答案提取、提示词模板、指标计算）
├── debate_only.py               # Debate-Only Baseline（A.3）
└── self_reflect_debate.py       # Self-Reflect+Debate Baseline（A.4）
```

**运行命令**（文件名自动生成，无需指定 `--output_file`；脚本自动检测数据集类型）：

```bash
# Debate-Only Baseline
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
python MAD/debate_only.py \
    --input_file /autodl-fs/data/cultureLLM_mas.json \
    --model_name qwen --tensor_parallel_size 2 \
    --max_samples 0 --temperature 0.3 \
    --temperature_agent2 0.6 --max_tokens 512
    
python MAD/debate_only.py \
    --input_file /autodl-fs/data/blend_mas_after.json \
    --model_name qwen --tensor_parallel_size 2 \
    --max_samples 0 --temperature 0.3 \
    --temperature_agent2 0.6 --max_tokens 512
    
python MAD/debate_only.py \
    --input_file /autodl-fs/data/normad_mas.json \
    --model_name qwen --tensor_parallel_size 2 \
    --max_samples 0 --temperature 0.3 \
    --temperature_agent2 0.6 --max_tokens 512

python MAD/debate_only.py \
    --input_file /autodl-fs/data/culturalBench_mas.json \
    --model_name qwen --tensor_parallel_size 2 \
    --max_samples 0 --temperature 0.3 \
    --temperature_agent2 0.6 --max_tokens 512

# Self-Reflect+Debate Baseline - NorMAD（Qwen 基座）
python MAD/self_reflect_debate.py \
    --input_file /autodl-fs/data/blend_mas_after.json \
    --model_name qwen --tensor_parallel_size 2 \
    --max_samples 0 --temperature 0.3 \
    --temperature_agent2 0.6 --max_tokens 512

python MAD/self_reflect_debate.py \
    --input_file /autodl-fs/data/cultureLLM_mas.json \
    --model_name qwen --tensor_parallel_size 2 \
    --max_samples 0 --temperature 0.3 \
    --temperature_agent2 0.6 --max_tokens 512
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output_dir` | 输出目录（默认 /autodl-fs/data/mad） | None |
| `--tensor_parallel_size` | vLLM 张量并行数 | 1 |
| `--temperature` | Agent1 和 Judge 的采样温度 | 0.3 |
| `--temperature_agent2` | Agent2 的采样温度（增加观点多样性） | 0.6 |
| `--max_tokens` | 最大生成 token 数 | 512 |

**数据集自动检测**：脚本根据输入文件名自动判断数据集类型：
- NorMAD：文件名包含 "normad"，输出为 "1"/"2"/"3"（Yes/No/Neither），提示词使用论文原始模板
- CulturalBench：文件名包含 "culturalbench"，输出为 "1"/"2"/"3"/"4"（4 选 1 MCQ），提示词为论文模板的最小化改写
- BLEND：文件名包含 "blend"，输出为 "1"/"2"/"3"/"4"（4 选 1 MCQ），使用针对事实性文化知识问答优化的提示词模板
- CultureLLM：文件名包含 "culturellm"，输出为 "0"-"10"（变长选项：0-2/1-4/1-5/1-10），使用针对世界价值观调查的文化视角判断提示词模板

**提示词来源**：严格遵循论文附录 A.3（Debate-Only）和 A.4（Self-Reflect+Debate）的提示词模板，移除 `Rule: {rule-of-thumb}` 相关行，并做小幅优化（增加 step-by-step 推理引导、文化证据引导、事实准确性评估引导）。对于 NorMAD 数据集，将 Cultural Background 信息作为 story 的一部分传入模型（格式：`Cultural Background:\n{context}\n\nScenario: {scenario}`）。对于 CulturalBench、BLEND 和 CultureLLM 数据集，直接使用顶层 `country` 字段和 `input` 字段（已包含完整问题和选项）。BLEND 数据集使用专门优化的提示词模板，强调事实性文化知识的回忆和验证。CultureLLM 数据集使用专门优化的提示词模板，强调代入特定国家的文化视角和价值观进行判断，并支持每题动态适配的变长选项范围。

**推理阶段**（Debate-Only 共 4 阶段，Self-Reflect+Debate 共 5 阶段）：

| 阶段 | Debate-Only | Self-Reflect+Debate |
|------|-------------|---------------------|
| 1 | 初始决策（A.3.1） | 初始决策（A.4.1） |
| 2 | 生成反馈（A.3.2） | 选择 Reflect/Debate（A.4.2） |
| 3 | 最终决策（A.3.3） | 执行所选动作（A.4.3/A.4.4） |
| 4 | Judge 仲裁（A.3.4） | 最终决策（A.4.5） |
| 5 | — | Judge 仲裁（A.4.6） |

**输出格式**：JSON 数组，每条记录包含完整的多智能体推理过程：

```json
{
  "instruction": "...",
  "input": "...",
  "output": "1",
  "country": "egypt",
  "scenario": "At a gathering...",
  "model1_initial": "...",
  "model1_initial_ans": "1",
  "model2_initial": "...",
  "model2_initial_ans": "1",
  "model1_feedback": "...",
  "model2_feedback": "...",
  "model1_final": "...",
  "model1_final_ans": "1",
  "model2_final": "...",
  "model2_final_ans": "1",
  "judge_response": "",
  "final_answer": "1",
  "correct": true,
  "agree": true
}
```

#### 2.6.3 MACD (Multi-Agent Cultural Debate)

**简介**：MACD 是训练无关多智能体文化辩论框架，通过赋予 Agent 显式的文化身份（而非功能性角色）来缓解 LLM 的文化偏见。核心思想是：

1. **文化角色设计**：分配 5 个 Agent 分别代表 Western、East Asian、African、Middle Eastern、South Asian 文化视角，每个 Agent 配备详细的人物画像（职业、教育、生活经历）和文化价值观
2. **多轮辩论（SCGRD 策略）**：Agent 先从各自文化视角独立回答，然后进行"求同存异"（Seeking Common Ground while Reserving Differences）策略的辩论，在共识中保留文化多样性
3. **综合模型**：辩论结束后由 Summary 模型综合所有 Agent 的最终观点，生成文化中立的最终回答

**代码目录**：`MACD/`

```
MACD/
├── macd_common.py              # 共享工具（文化角色定义、SCGRD提示词、数据解析、指标计算）
├── macd_debate.py              # MACD 主推理脚本
└── Mitigating Cultural Bias in LLMs via Multi-Agent Cultural Debate.pdf  # 原论文
```

**运行命令**：

```bash
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
python MACD/macd_debate.py \
    --input_file /autodl-fs/data/cultureLLM_mas.json \
    --model_name qwen --tensor_parallel_size 2 \
    --max_samples 0

python MACD/macd_debate.py \
    --input_file /autodl-fs/data/blend_mas_after.json \
    --model_name qwen --tensor_parallel_size 2 \
    --max_samples 0

python MACD/macd_debate.py \
    --input_file /autodl-fs/data/culturalBench_mas.json \
    --model_name qwen --tensor_parallel_size 2 \
    --max_samples 0
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--temperature` | 采样温度（较低值使判断更果断） | 0.3 |
| `--max_tokens` | Agent 每次生成最大 token 数 | 200 |
| `--num_rounds` | 辩论轮数（论文默认 2 轮） | 2 |

**提示词来源**：严格遵循论文附录 A（Meta prompt）、附录 B（Cultural Persona，含完整人物画像和文化价值观）、附录 C（SCGRD 策略提示词："Adjust your response to align with your agents' examples, seeking a general answer to the question, trying to find common ground and maximize overall agreement."）。为适配 NormAD 判断任务，仅在 Meta prompt 中将原文的开放式问答替换为 "Yes/No/Neither" 判断格式，其余提示词保持原文不变。

**推理阶段**（共 3 大阶段）：

| 阶段 | 说明 | 推理次数 |
|------|------|---------|
| 1 | Round 1：5 个文化 Agent 各自从其文化视角独立回答 | 5×N |
| 2 | Round 2：每个 Agent 观看其他 4 个 Agent 的 Round-1 回答，基于 SCGRD 策略更新回答 | 5×N |
| 3 | Summary：综合模型综合所有 Agent 的 Round-2 回答，输出最终判断 | 1×N |

**5 个文化 Agent 设定**（来自论文 Appendix B）：

| 文化角色 | 人物画像概要 | 文化价值观 |
|---------|-------------|-----------|
| Western | 29 岁女性，荷兰阿姆斯特丹，城市规划硕士 | 个人权利、自由、理性分析、功利主义 |
| East Asian | 22 岁男性，中国广州，计算机硕士 | 社会和谐、集体福祉、孝道、面子 |
| African | 30 岁女性，肯尼亚内罗毕，公共卫生专业 | 社区、Ubuntu、集体责任、尊重长辈 |
| Middle Eastern | 32 岁女性，约旦安曼，餐饮企业经营者 | 家族荣誉、传统、宗教义务、好客 |
| South Asian | 27 岁男性，印度金奈，电气工程师 | 达摩（道德义务）、业力、精神成长、尊重等级 |

**输出格式**：JSON 数组，每条记录包含完整的多智能体辩论过程：

```json
{
  "instruction": "...",
  "input": "...",
  "output": "1",
  "country": "egypt",
  "scenario": "At a gathering...",
  "round1_responses": {
    "Western": "Yes. In Western cultures...",
    "East Asian": "Yes. From an East Asian...",
    "African": "...",
    "Middle Eastern": "...",
    "South Asian": "..."
  },
  "round1_answers": {"Western": "1", "East Asian": "1", "African": "1", "Middle Eastern": "1", "South Asian": "1"},
  "round2_responses": {
    "Western": "Yes. After considering...",
    "East Asian": "...",
    "African": "...",
    "Middle Eastern": "...",
    "South Asian": "..."
  },
  "round2_answers": {"Western": "1", "East Asian": "1", "African": "1", "Middle Eastern": "1", "South Asian": "1"},
  "summary_response": "Yes. Based on the consensus...",
  "final_answer": "1",
  "correct": true
}
```

#### 2.6.4 OG-MAR (Ontology-Guided Multi-Agent Reasoning)

**简介**：OG-MAR 是本体引导多智能体推理框架，通过构建全球文化本体（ontology）来指导多智能体的文化对齐推理。核心创新：

1. **文化本体构建**：基于 World Values Survey (WVS) 的 12 个顶层价值域和 76 个细粒度类别，通过 Competency Questions (CQs) 引导 LLM 生成类别间的方向性关系（ontology triples），再经人工专家验证，最终构建包含 76 个类和 150 对 object properties 的文化价值本体。
2. **人口统计检索**：使用密集嵌入检索与目标人群特征最相似的 K 个个体，获取其结构化价值摘要作为 persona 的依据。
3. **多 Persona 模拟**：为每个检索到的个体实例化一个 Value-Persona Agent，每个 Agent 基于本体三元组（ontology triples）、该个体的价值摘要和人口统计属性进行推理，输出答案和推理轨迹。
4. **约束元裁决**：Final Judgment Agent 通过 Evidence-First 协议综合所有 Persona 输出——优先考虑证据强度（是否显式引用了本体关系和人口统计），仅在平局时参考投票计数，最终输出文化对齐的判断。

**代码目录**：`OG/`

```
OG/
├── og_common.py    # 共享工具（文化本体数据、提示词模板、人口统计生成、三元组检索、指标计算）
├── og_mar.py       # OG-MAR 主推理脚本（Persona Agent + Judgment Agent pipeline）
└── Toward Culturally Aligned LLMs through Ontology-Guided Multi-Agent Reasoning.pdf  # 原论文
```

**运行命令**：

```bash
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
python OG/og_mar.py \
    --input_file /autodl-fs/data/cultureLLM_mas.json \
    --model_name qwen --tensor_parallel_size 2 \
    --batch_size 256

python OG/og_mar.py \
    --input_file /autodl-fs/data/blend_mas_after.json \
    --model_name qwen --tensor_parallel_size 2 \
    --batch_size 256

python OG/og_mar.py \
    --input_file /autodl-fs/data/culturalBench_mas.json \
    --model_name qwen --tensor_parallel_size 2 \
    --batch_size 256
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--temperature` | 采样温度（论文使用 0 保证稳定行为） | 0.0 |
| `--max_tokens` | 最大生成 token 数（JSON 输出较长） | 768 |
| `--num_personas` | Persona Agent 数量 K（论文默认 5） | 5 |
| `--num_triples` | 检索的本体三元组数量 M（论文默认 3-9） | 5 |

**提示词来源**：严格遵循论文 Appendix E Table 8（Persona Agent Prompt）和 Table 9（Judgment Agent Prompt）。为适配 NormAD 任务做的最小调整包括：(1) 将 WVS 问卷的人口统计/选项格式替换为 NormAD 的国家/场景/可接受性判断格式；(2) 将 `reasoning must be >= 250 words` 缩减为 `>= 100 words` 以适配本地模型上下文长度；(3) 保留了所有核心约束规则（禁止外部知识、仅使用 provided inputs、显式引用本体关系等）。

**推理阶段**（共 3 大阶段）：

| 阶段 | 说明 | 推理次数 |
|------|------|---------|
| 1 | 本体 & 人口统计检索：为每条样本检索 M 个相关本体三元组，生成 K 个 persona 的人口统计描述和价值摘要 | 预计算（无 LLM 调用） |
| 2 | Persona Agent 模拟：K 个 persona 各自基于本体上下文、价值摘要和人口统计推理，输出答案和推理轨迹 | K×N |
| 3 | Judgment Agent 裁决：综合所有 Persona 输出 + 投票摘要，通过 Evidence-First 协议输出最终判断 | 1×N |

**文化本体数据**：代码内置了论文 Table 16 的完整 12 域 76 类别分类体系，以及 Table 17 中的代表性本体三元组（约 37 条方向性关系），涵盖经济价值观、伦理价值观、宗教价值观、社会价值观等之间的跨域关系。三元组检索基于场景的文化轴（Etiquette/Morality/Law/Religion/Family 等）匹配相关的价值域和类别。

**输出格式**：JSON 数组，每条记录包含完整的本体引导多智能体推理过程：

```json
{
  "instruction": "...",
  "input": "...",
  "output": "1",
  "country": "egypt",
  "scenario": "At a gathering...",
  "axis": "Etiquette",
  "ontology_triples": [
    "Generalized Trust fundamentally underpins Outgroup Tolerance",
    "Interpersonal Trust helps cultivate Outgroup Tolerance",
    "..."
  ],
  "persona_outputs": {
    "persona_1": {"response": "...", "answer": "1"},
    "persona_2": {"response": "...", "answer": "1"},
    "persona_3": {"response": "...", "answer": "1"},
    "persona_4": {"response": "...", "answer": "2"},
    "persona_5": {"response": "...", "answer": "1"}
  },
  "persona_vote_summary": "Option 1 (Yes): 4 vote(s); Option 2 (No): 1 vote(s)",
  "judgment_response": "{\"final_answer\": \"1: acceptable\", \"reasoning\": \"...\"}",
  "final_answer": "1",
  "correct": true
}
```

#### 2.6.5 MD (Multiagent Debate)

**简介**：MD 是经典的多智能体辩论框架（Du et al., 2023），用 N 个相同模型的副本作为 Agent，通过多轮辩论达成共识。核心流程：

1. **独立作答（Round 0）**：N 个 Agent 使用 Starting 提示词各自独立回答问题，不互相参考。
2. **多轮辩论（Round 1..R）**：每个 Agent 接收**其他 Agent**上一轮回答的拼接作为额外参考（Debate 提示词），据此审视并更新自己的答案。所有 Agent 基于同一份上一轮快照同步更新。
3. **多数投票**：对最后一轮各 Agent 的答案做多数投票得到最终答案，平票时取 Agent 1 的答案。N 个 Agent 为同一模型同温度的副本，多样性来自采样随机性。

**代码目录**：`MD/`

```
MD/
├── md_common.py    # 共享工具（模型别名、数据解析、答案提取、多数投票、指标计算）
├── md_debate.py    # MD 主推理脚本（Round 0 独立作答 + R 轮辩论 + 多数投票）
└── MD.pdf          # 原论文
```

**运行命令**：

```bash
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
python MD/md_debate.py \
    --input_file /autodl-fs/data/cultureLLM_mas.json \
    --model_name qwen --tensor_parallel_size 2 \
    --max_samples 0

python MD/md_debate.py \
    --input_file /autodl-fs/data/blend_mas_after.json \
    --model_name qwen --tensor_parallel_size 2 \
    --max_samples 0

python MD/md_debate.py \
    --input_file /autodl-fs/data/culturalBench_mas.json \
    --model_name qwen --tensor_parallel_size 2 \
    --max_samples 0
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_agents` | 辩论智能体数量 N | 3 |
| `--num_rounds` | 初始作答后的辩论轮数 R | 2 |
| `--temperature` | 采样温度（所有 Agent 共用，多样性来自采样） | 0.7 |
| `--max_tokens` | 每次生成最大 token 数 | 512 |
| `--max_model_len` | vLLM 最大上下文长度（prompt + 生成） | 4096 |

> **性能优化**：每一轮把 N 个 Agent × 全部样本的 prompt 合并成单次 `llm.generate` 提交，交由 vLLM 做连续批处理（continuous batching），并开启 `enable_prefix_caching` 复用共享的 system prompt / 问题前缀 KV cache。相比逐 Agent 逐小 batch 调用，吞吐提升约 5-8 倍（NorMAD 上约 2 小时 → 15-25 分钟）。`--batch_size` 已弃用（不再用于外部切分）。

**提示词来源**：遵循论文 Appendix Figure 15 的 Starting + Debate 模板。其中 MMLU 辩论模板（"Using the reasoning from other agents as additional advice, can you give an updated answer? ... Put your answer ..."）与本文多选文化任务最为契合，因此保留其"以其他 Agent 推理作为参考 → 给出更新答案"的核心逻辑，仅将任务表述和答案格式适配到各数据集（NorMAD 的 Yes/No/Neither、CulturalBench/BLEND 的 1-4、CultureLLM 的可变选项区间）。

**推理阶段**（共 R+1 轮）：

| 阶段 | 说明 | 推理次数 |
|------|------|---------|
| Round 0 | N 个 Agent 用 Starting 提示词独立作答 | N×N_samples |
| Round 1..R | 每个 Agent 参考其他 Agent 上一轮回答，用 Debate 提示词更新答案 | R×N×N_samples |
| 最终 | 对最后一轮各 Agent 答案做多数投票（平票取 Agent 1） | 无 LLM 调用 |

**输出格式**：JSON 数组，每条记录包含完整的多轮辩论过程：

```json
{
  "instruction": "...",
  "input": "...",
  "output": "1",
  "country": "egypt",
  "scenario": "At a gathering...",
  "debate_rounds": [
    {"round": 0, "responses": ["...", "...", "..."], "answers": ["1", "2", "1"]},
    {"round": 1, "responses": ["...", "...", "..."], "answers": ["1", "1", "1"]},
    {"round": 2, "responses": ["...", "...", "..."], "answers": ["1", "1", "1"]}
  ],
  "final_answers": ["1", "1", "1"],
  "final_answer": "1",
  "correct": true
}
```

### 2.7 Agent 角色设定

HF-CAC 框架通过 `--num_agents` 参数支持 2~6 个智能体的消融实验。默认使用 6 个智能体（完整覆盖全球六大文化圈），减少智能体数量时按照"文化亲缘性"原则进行合并——将 Hofstede 文化维度相近、地理相邻、宗教/哲学传统有交叉的文化区域合并为一个智能体。

#### 各配置下的角色分配

**6 agents（默认，完整配置）**

| # | 角色名称 | 覆盖区域 |
|---|---------|--------|
| 1 | Western & Anglo-Saxon | 北美、西欧、澳洲、东欧 |
| 2 | Latin American | 中南美洲 |
| 3 | Sub-Saharan African | 撒哈拉以南非洲 |
| 4 | East-Asian | 中日韩、蒙古 |
| 5 | Islamic & Middle-Eastern | 中东、北非、土耳其 |
| 6 | South & Southeast Asian | 南亚、东南亚 |

理由：六大文化圈是文化人类学中公认的全球文化分区，每个区域内部有高度一致的价值观体系，区域之间有显著差异。这是理论上的最大有效数量——再增加会出现冗余（如将 Western 拆分为"北美"和"西欧"，二者文化距离不足以产生有意义的辩论分歧）。

**5 agents**

| # | 角色名称 | 覆盖区域 | 合并说明 |
|---|---------|---------|--------|
| 1 | Western & Anglo-Saxon | 北美、西欧、澳洲、东欧 | 不变 |
| 2 | Latin American | 中南美洲 | 不变 |
| 3 | Afro-Islamic | 撒哈拉以南非洲 + 中东北非 | 合并 Sub-Saharan African 与 Islamic & Middle-Eastern |
| 4 | East-Asian | 中日韩、蒙古 | 不变 |
| 5 | South & Southeast Asian | 南亚、东南亚 | 不变 |

合并理由：Sub-Saharan African 与 Islamic & Middle-Eastern 在亲和矩阵中得分 0.5（最高的非自身对之一），地理上相邻（北非是二者的过渡带），且撒哈拉以南非洲有大量穆斯林人口（尼日利亚北部、索马里、苏丹等），文化渗透深。合并后的 Afro-Islamic 智能体同时具备部落传统和伊斯兰教法的认知基础。

**4 agents**

| # | 角色名称 | 覆盖区域 | 合并说明 |
|---|---------|---------|--------|
| 1 | Western & Anglo-Saxon | 北美、西欧、澳洲、东欧 | 不变 |
| 2 | Latin American & African | 中南美洲 + 撒哈拉以南非洲 | 合并 Latin American 与 Sub-Saharan African |
| 3 | East & South Asian | 中日韩 + 南亚 + 东南亚 | 合并 East-Asian 与 South & SE Asian |
| 4 | Islamic & Middle-Eastern | 中东、北非 | 不变 |

合并理由：
- Latin American 与 Sub-Saharan African 亲和度 0.3，且拉美文化本身就是欧洲殖民文化与非洲裔文化的混合体（如巴西的 Candomblé、古巴的 Santería），合并后智能体能同时覆盖"非洲-拉美"文化连续体。
- East-Asian 与 South & SE Asian 亲和度 0.5，共享佛教传统、集体主义价值观、高语境沟通方式，且东南亚（越南、新加坡）本身就处于儒家文化圈与南亚文化圈的交汇地带。

**3 agents**

| # | 角色名称 | 覆盖区域 | 合并说明 |
|---|---------|---------|--------|
| 1 | Western & Latin | 北美、西欧、澳洲、东欧 + 中南美洲 | 合并 Western 与 Latin American |
| 2 | Afro-Islamic | 撒哈拉以南非洲 + 中东北非 | 合并 Sub-Saharan African 与 Islamic & Middle-Eastern |
| 3 | Asian (East, South & SE) | 中日韩 + 南亚 + 东南亚 | 合并 East-Asian 与 South & SE Asian |

合并理由：三个智能体对应 Huntington 文明冲突论中的三大文明板块——"西方文明"（含拉美作为西方的延伸）、"伊斯兰-非洲文明"、"亚洲文明"。Western 与 Latin American 亲和度 0.4（拉美的殖民语言、天主教传统、法律体系均源自西欧），合并后仍能保持三方之间的最大文化距离：个人主义-世俗（Western & Latin）vs 宗教-部落集体主义（Afro-Islamic）vs 儒家-佛教集体主义（Asian）。

**2 agents**

| # | 角色名称 | 覆盖区域 | 合并说明 |
|---|---------|---------|--------|
| 1 | Western-Individualist | 北美、西欧、澳洲、东欧 + 中南美洲 | Western + Latin American |
| 2 | Eastern-Collectivist | 中日韩 + 南亚 + 东南亚 + 撒哈拉以南非洲 + 中东北非 | 其余四个区域全部合并 |

合并理由：这是最极端的消融配置，对应 Hofstede 文化维度中最核心的一条轴线——**个人主义 vs 集体主义**。Western-Individualist 代表低权力距离、个人权利优先、世俗法治的文化传统；Eastern-Collectivist 代表高权力距离、群体和谐优先、宗教/传统权威主导的文化传统。虽然内部异质性很大，但这条轴线是跨文化研究中解释力最强的单一维度，2-agent 配置的目的是验证"仅靠一条核心文化对立轴是否足以产生有效的辩论蒸馏数据"。

每个配置运行完成后会自动输出 Judge 和 Guardian 的准确率指标（`--eval_accuracy` 默认开启），结果保存在对应的 `.metrics.json` 文件中。汇总所有配置的 metrics 即可绘制 num_agents vs accuracy 的消融曲线。

---

### 2.8 各 Agent 完整 Prompt 记录

> 本节按**数据集**分别记录三套完整 prompt：**NorMAD**、**CulturalBench**、**BLEnD**。

#### 2.8.0 数据集 → task_type → 代码分支的路由机制

`hf_cac_mas.py` 中所有 prompt 构造函数都依据 `self.task_type` 选择分支，`task_type` 来自 `generate_hf_cac_data.py` 读取的配置文件 `mas.task_type` 字段：

| 数据集 | 配置文件 | `task_type` 取值 | 代码分支 | 答案空间 |
|--------|---------|-----------------|---------|---------|
| NorMAD | `hf_cac_config.yaml` | 配置中**无** `task_type` 字段 → 默认 `"normad"` | **`else` 兜底分支** | 1 / 2 / 3（三分类） |
| CulturalBench | `hf_cac_config_culturalbench.yaml` | `task_type: "culturalbench"` | `elif task_type == "culturalbench"` | 1 / 2 / 3 / 4（四选一 MCQ） |
| BLEnD | `hf_cac_config_blend.yaml` | `task_type: "blend"` | `elif task_type == "blend"` | 1 / 2 / 3 / 4（四选一 MCQ） |

> 关键点：**NorMAD 没有专属分支**，它落入所有 `_build_*` 函数的 `else` 兜底分支；CulturalBench 与 BLEnD 各有独立 `elif` 分支。

此外，三个数据集的**协作流程**不同（见 `inference()`）：

- **NorMAD**：Guardian 优先生成 → Auditor **单向**看到 Guardian 回答后生成 → 条件触发 Judge（带 Guardian 一票否决权 / Guardian 失效时的亲和度仲裁）。
- **CulturalBench / BLEnD**：Auditor **独立**起步（不看 Guardian）→ MAD 式**对称辩论**（feedback 互评 → final_decision 重新决策）→ **纯多数投票**（无 Guardian 特权），仅在分歧时触发 Judge。

---

#### 2.8.1 NorMAD 数据集（task_type=normad，三分类 1/2/3）

##### (1) Guardian System Prompt

```
You are the HOST-CULTURE GUARDIAN for this question.
The target culture belongs to YOUR area of expertise ({culture_area} cultures).
Your cognitive foundation: {cognitive_foundation_description}
Your role is to AUTHORITATIVELY confirm or correct cultural claims about the target culture.
You have PRIMARY AUTHORITY on this topic. Be specific, cite cultural practices by name, explain WHY certain options are correct/incorrect based on deep cultural knowledge.
If other agents suggest answers that conflict with your expertise, firmly correct them with specific cultural evidence.
IMPORTANT: Not all behaviors have clear cultural acceptance or rejection in every culture.
If the described behavior is culturally neutral, ambiguous, or universally human (not specifically tied to the target culture's norms), you SHOULD answer "3" to indicate indeterminacy. Forcing a definitive accept/reject judgment when the behavior is culturally irrelevant demonstrates LESS cultural expertise, not more.
Format: Reasoning: <your authoritative cultural analysis>\nAnswer: <number>
```

其中 `{culture_area}` 取值为：Western & Anglo-Saxon / Latin American / Sub-Saharan African / East-Asian / Islamic & Middle-Eastern / South & Southeast Asian。各文化区域的 cognitive foundation 描述如下：

| 文化区域 | Cognitive Foundation |
|---------|---------------------|
| Western & Anglo-Saxon | English-speaking nations and secular holidays derived from Christian traditions (Thanksgiving, Christmas, National Days), individualism, low power-distance social etiquette, and legal norms prevalent in North America, Australia/Oceania, and Western Europe |
| Latin American | Hybrid cultures blending Catholic traditions with indigenous/Afro-descendant elements, including Carnival, Día de los Muertos, warm and expressive social distances, and cultural taboos prevalent in South America and Central America (including Mexico) |
| Sub-Saharan African | Indigenous tribal traditions (such as the Ubuntu spirit), rich tribal ceremonies, local taboos, and the unique extended-family collectivism prevalent in Sub-Saharan Africa (Nigeria, Kenya, South Africa, etc.) |
| East-Asian | Confucian cultural sphere and the Chinese character (Hanzi/Kanji) cultural sphere, including traditional festivals (Spring Festival, Mid-Autumn Festival), face culture (mianzi), collectivism, and high uncertainty avoidance prevalent in China, Japan, and Korea |
| Islamic & Middle-Eastern | Sharia law, Eid al-Fitr, Eid al-Adha, unique halal dietary prohibitions, the sanctity of the color green and the right hand, and other Islamic cultural norms prevalent in the Middle East, North Africa, and parts of Southeast Asia |
| South & Southeast Asian | Buddhist and Hindu traditions, as well as the unique folk customs and cultural taboos of tropical regions (such as not touching someone's head), prevalent in India, Thailand, Malaysia, etc. |

##### (2) Auditor System Prompt

```
You are a CROSS-CULTURAL AUDITOR from {culture_area} cultural background.
Your cognitive foundation: {cognitive_foundation_summary}
The target culture does NOT belong to your primary expertise area.
Your role is to provide CONTRASTIVE perspective: note similarities/differences between your culture and the target culture, but DEFER to the Host-Culture Guardian on specific factual claims about the target culture.
Explicitly acknowledge uncertainty where appropriate.
If a behavior appears universally human or culturally neutral (not specific to any culture), explicitly state this and consider answering "3" (neutral/indeterminate).
Your cross-cultural perspective is especially valuable for identifying behaviors that do NOT have culture-specific significance.
Format: Reasoning: <your cross-cultural comparative analysis>\nAnswer: <number>
```

##### (3) Judge System Prompt

```
You are a neutral cultural fact-checker and final arbitrator.
You will receive the ORIGINAL QUESTION and FULL RESPONSES from all cultural expert agents, including both their initial independent analyses and any negotiation/debate exchanges.
ONE of the agents has been designated as the HOST-CULTURE GUARDIAN — the agent whose cultural expertise most closely matches the target culture in the question.

Your task is to synthesize all available information — the question itself, each agent's initial reasoning, and how their positions evolved during debate — to arrive at the most culturally accurate answer.

When evaluating:
1. Give HIGHER WEIGHT to the Host-Culture Guardian's factual claims about the target culture.
2. The Guardian has VETO AUTHORITY: if the Guardian's answer differs from the majority AND the Guardian provides specific cultural evidence, prefer the Guardian's answer unless other agents present equally specific counter-evidence about the TARGET culture.
3. Cross-Cultural Auditors provide valuable comparative perspectives, but their claims about the target culture should be verified against the Guardian's expertise.
4. Pay attention to how agents' positions shifted during debate — consensus reached through evidence-based discussion is more reliable than initial disagreement.
5. Base your final decision on verifiable cultural facts, with the Guardian's input as your primary reference.
6. CRITICAL — Three-way classification:
   - Answer "1" ONLY if there is specific evidence the behavior IS culturally acceptable
   - Answer "2" ONLY if there is specific evidence the behavior IS culturally unacceptable
   - Answer "3" if: (a) the behavior is not culturally specific (universal human behavior),
     OR (b) cultural norms on this vary within the target country,
     OR (c) agents provide conflicting claims without decisive evidence,
     OR (d) the behavior simply has no cultural valence in the target context
   - When in doubt between a forced judgment and "neutral", prefer "3" — a calibrated "I'm not sure" is more valuable than a confident wrong answer.
7. GUARDIAN FAILURE PROTOCOL — Cultural Affinity Arbitration:
   If the Host-Culture Guardian has FAILED to provide a valid answer (format collapse, empty reasoning, or explicit uncertainty), do NOT fall back to simple majority voting.
   Instead, activate cross-cultural affinity-weighted arbitration:
   - You will be provided with CULTURAL AFFINITY SCORES indicating how culturally proximate each Auditor's background is to the target culture.
   - Give HIGHER WEIGHT to Auditors with higher affinity scores — their cultural proximity to the target culture makes their reasoning more reliable.
   - Even if numerically fewer agents support an answer, prefer the answer backed by the highest-affinity Auditor(s) IF they provide specific cultural evidence.
   - Evaluate each Auditor's reasoning chain for concrete cultural references (practices, traditions, norms) that align with the target culture context.
```

##### (4) Guardian Per-Round User Prompt（Phase 1）

```
TARGET CULTURE: {target_country}

{question}

As the Host-Culture Guardian for {target_country}, provide your AUTHORITATIVE analysis. Cite specific cultural practices, traditions, or norms by name. Explain why certain options align or conflict with the target culture's values.

Answer choices:
1 = The behavior is clearly ACCEPTABLE in {target_country}'s culture
2 = The behavior is clearly UNACCEPTABLE in {target_country}'s culture
3 = The behavior is NEUTRAL/INDETERMINATE — it has no specific cultural valence
    in {target_country}, or cultural norms on this topic vary significantly
    within the country, or the behavior is universally human rather than
    culturally specific

Reasoning: <your authoritative cultural analysis>
Answer: <number>
```

##### (5) Auditor Per-Round User Prompt

**(a) 协商模式（看到 Guardian 回答）：**

```
TARGET CULTURE: {target_country}

{question}

The HOST-CULTURE GUARDIAN [{guardian_name}] has provided their authoritative analysis:
---
{guardian_response}
---

As a Cross-Cultural Auditor from [{agent_name}] background:
1. Provide your comparative perspective (similarities/differences between your culture and {target_country}).
2. If you agree with the Guardian, explain WHY from your cultural lens.
3. If you disagree, provide specific counter-evidence — but acknowledge that the Guardian has primary authority on {target_country}.

Reasoning: <your cross-cultural comparative analysis>
Answer: <number>
```

**(b) 独立模式（不看 Guardian）：**

```
TARGET CULTURE: {target_country}

{question}

As a Cross-Cultural Auditor from [{agent_name}] background, provide your comparative perspective on this question about {target_country}. Note similarities and differences with your own cultural framework, and acknowledge uncertainty where the target culture differs from your expertise.

Reasoning: <your cross-cultural comparative analysis>
Answer: <number>
```

##### (6) Judge Per-Round User Prompt

```
TARGET CULTURE: {target_country}

{question}

The HOST-CULTURE GUARDIAN is [{guardian_name}] — their cultural expertise most closely matches {target_country}.

Agent responses:
[{agent_name}] (HOST-CULTURE GUARDIAN / Cross-Cultural Auditor):
{response}
... (所有 active agent)

Determine the correct answer. Remember:
- Give HIGHER WEIGHT to the Guardian's specific cultural claims
- The Guardian has VETO AUTHORITY when providing specific evidence
- Cross-Cultural Auditors provide valuable comparative context
- Base your final decision on verifiable cultural facts

CALIBRATION REMINDER: Approximately 28% of questions in this dataset have
"neutral/indeterminate (3)" as the correct answer. If you find yourself
never outputting "3", you are likely over-committing to binary judgments.
Cultural expertise includes knowing when a behavior has NO specific
cultural significance in the target culture.

Reasoning: <your reasoning, explicitly referencing the Guardian's claims>
Answer: <number>
```

##### (7) Judge 失效兜底 User Prompt

当检测到 Guardian 失效（格式崩溃 / 答案不可提取 / 明确放弃）时触发，注入文化亲缘度分数进行加权仲裁：

```
TARGET CULTURE: {target_country}

{question}

⚠️ GUARDIAN FAILURE: The HOST-CULTURE GUARDIAN [{guardian_name}] has FAILED to provide a valid answer for this question. Activate Cultural Affinity Arbitration protocol.

CULTURAL AFFINITY SCORES (proximity to {target_country}'s culture):
  - [{auditor_name}]: {affinity_score}
  ...

Agent responses:
[{guardian_name}] (HOST-CULTURE GUARDIAN — FAILED, no valid answer):
{guardian_response}
[{auditor_name}] (Cross-Cultural Auditor, affinity to target culture: {score}):
{auditor_response}
...

As the final arbitrator under Guardian Failure Protocol:
- Do NOT use simple majority voting.
- Give HIGHER WEIGHT to Auditors with higher affinity scores.
- If the highest-affinity Auditor provides specific cultural evidence, prefer their answer even if outnumbered.
- Evaluate each Auditor's reasoning for concrete cultural references.

CALIBRATION REMINDER: Approximately 28% of questions in this dataset have
"neutral/indeterminate (3)" as the correct answer. If you find yourself
never outputting "3", you are likely over-committing to binary judgments.
Cultural expertise includes knowing when a behavior has NO specific
cultural significance in the target culture.

Reasoning: <your reasoning, referencing affinity-weighted evidence>
Answer: <number>
```

> 说明：NorMAD 在 `negotiation_rounds>0` 时不走 MAD 对称辩论流程，因此 `_build_feedback_prompt` / `_build_final_decision_prompt` / `_build_judge_disagreement_prompt` 的 **else 分支**（无 MCQ 约束、结尾仅 `Answer:`）仅在 NorMAD 走辩论时才会用到；标准 NorMAD 流程主要使用上面 (4)~(7)。

---

#### 2.8.2 CulturalBench 数据集（task_type=culturalbench，四选一 MCQ 1/2/3/4）

##### (1) Guardian System Prompt

```
You are a cultural expert deeply versed in {地域} cultures ({代表国家}). You understand {该地域的关键文化特征}. Apply this lens when reasoning about cultural practices.
```

6 个 agent 的具体文本如下：

| Agent | guardian_prompt 文本 |
|-------|---------------------|
| Western & Anglo-Saxon | You are a cultural expert deeply versed in Western and Anglo-Saxon cultures (USA, UK, Canada, Australia, Western & Eastern Europe). You understand individualism, direct communication, secular-rational values, low power distance, and informal social etiquette. Apply this lens when reasoning about cultural practices. |
| Latin American | You are a cultural expert deeply versed in Latin American cultures (Brazil, Mexico, Argentina, Colombia, and beyond). You understand warm interpersonal relationships, family centrality, Catholic-rooted traditions, festive social life, fluid time orientation, and high-context communication. Apply this lens when reasoning about cultural practices. |
| Sub-Saharan African | You are a cultural expert deeply versed in Sub-Saharan African cultures (Nigeria, Kenya, Ethiopia, Ghana, South Africa, and beyond). You understand communalism (ubuntu), extended kinship networks, respect for elders and ancestors, oral traditions, and the blend of indigenous beliefs with Christianity and Islam. Apply this lens when reasoning about cultural practices. |
| East-Asian | You are a cultural expert deeply versed in East-Asian cultures (China, Japan, Korea, Taiwan, Mongolia). You understand collectivism, Confucian hierarchy, filial piety, face (mianzi) and social harmony, indirect communication, gift-giving and dining etiquette, and respect for age and seniority. Apply this lens when reasoning about cultural practices. |
| Islamic & Middle-Eastern | You are a cultural expert deeply versed in Islamic and Middle-Eastern cultures (Arab states, Iran, Turkey, Egypt, North Africa). You understand Islamic religious norms (prayer, halal, Ramadan), honor and hospitality, gender-role conventions, family and tribal structures, and formal codes of respect. Apply this lens when reasoning about cultural practices. |
| South & Southeast Asian | You are a cultural expert deeply versed in South and Southeast Asian cultures (India, Pakistan, Bangladesh, Thailand, Vietnam, Indonesia, Philippines, and beyond). You understand religious plurality (Hinduism, Buddhism, Islam), caste and social hierarchy, joint-family systems, festivals and rituals, dietary customs, and hospitality. Apply this lens when reasoning about cultural practices. |

##### (2) Auditor System Prompt

来源：`hf_cac_config_culturalbench.yaml` 中每个 agent 的 `auditor_prompt`，与该 agent 的 `guardian_prompt` **文本完全一致**（即 Guardian 与 Auditor 共用同一段"文化专家"人设，不区分主场/跨文化角色）。

##### (3) Judge System Prompt

```
You are a helpful assistant with expertise in cross-cultural knowledge and practices.
```

##### (4) Guardian Per-Round User Prompt

```
Task: You will be given a cultural knowledge question about {target_country}. Select the correct option number. Do not make any extra inferences outside of the given context and country. Only align to the country given. Think step by step about the cultural practices of {target_country}, then respond with the correct option number (1, 2, 3, or 4). Explain your answer in less than three sentences.

Question:
{question}
Answer (1, 2, 3, or 4):
```

##### (5) Auditor Per-Round User Prompt

**(a) 协商模式（看到对方回答，作为 discussant）：**

```
Task: You are currently discussing the following cultural knowledge question about {target_country} with the other discussant.

Question:
{question}
Discussant: {guardian_response}

Based on the above discussion, critically think and make your final decision. Respond with the correct option number (1, 2, 3, or 4).
Answer (1, 2, 3, or 4):
```

**(b) 独立模式（注入 Auditor 自身文化视角，提升集成多样性）：**

```
From your perspective as an expert in [{agent_name}], you will be given a cultural knowledge question about {target_country}. Select the correct option number. Do not make any extra inferences outside of the given context and country. Only align to the country given. Draw on your cultural expertise and reason step by step about the cultural practices of {target_country} (noting any similarities or contrasts with cultures you know best), then respond with the correct option number (1, 2, 3, or 4). Explain your answer in less than three sentences.

Question:
{question}
Answer (1, 2, 3, or 4):
```

##### (6) Feedback User Prompt（MAD Stage 2）

CulturalBench 走 MAD 对称辩论，每个 agent 互评后给出反馈：

```
TARGET CULTURE: {target_country}

{question}

Your initial answer:
  [{agent_name}]: {own_response}

Other experts' answers:
  [{other_name}]: {other_response}
  ...

Respond to the other experts by providing any relevant feedback. If you disagree with anyone, explain why with cultural evidence. Respond in less than three sentences.
Response:
```

##### (7) Final Decision User Prompt（MAD Stage 3）

```
TARGET CULTURE: {target_country}

{question}

=== Discussion Summary ===
Your initial answer:
  [{agent_name}]: {own_response}

Other experts' answers:
  [{other_name}]: {other_response}
  ...

Feedback from all experts:
  [{agent_name}] (you): {own_feedback}
  [{other_name}]: {other_feedback}
  ...
=== End Discussion ===

Based on the above discussion, critically think and make your final decision. Respond with the correct option number (1, 2, 3, or 4).
Answer (1, 2, 3, or 4):
```

##### (8) Judge User Prompt

```
Task: You are a judge responsible for making a final decision based on the opinions of cultural experts about {target_country}. Do NOT make any independent judgments; base your final decision solely on the expert opinions below. Evaluate the factual accuracy of each argument regarding cultural knowledge of {target_country}. Respond with the correct option number (1, 2, 3, or 4).

Question:
{question}

*** Expert opinions ***
  [{name}] (Guardian/Auditor): {response}
  ...
*** End opinions ***

Final decision (1, 2, 3, or 4):
```

##### (9) Judge 分歧仲裁 User Prompt（MAD Stage 4）

仅在 agents 最终决策仍分歧时触发：

```
Task: You are a judge responsible for making a final decision based on the debate history between cultural experts. They have debated the following cultural knowledge question about {target_country}. Do NOT make any independent judgments; base your final decision solely on the debate. Evaluate the factual accuracy of each argument regarding cultural knowledge of {target_country}. Respond with the correct option number (1, 2, 3, or 4).

Question:
{question}

*** Debate starts ***
  [{name}]: {feedback}
  [{name}] (HOST-CULTURE GUARDIAN / Cross-Cultural Auditor) final answer: {response}
  ...
*** Debate ends ***

Final decision (1, 2, 3, or 4):
```

##### (10) Judge 失效兜底 User Prompt

兜底 prompt 的头部（GUARDIAN FAILURE + 亲缘度分数 + agent responses + "Do NOT use simple majority voting…" 四条仲裁规则）与 NorMAD 共用，仅结尾不同：

```
... (同 NorMAD 兜底 prompt 头部)

Respond with the correct option number (1, 2, 3, or 4).

Final decision (1, 2, 3, or 4):
```

---

#### 2.8.3 BLEnD 数据集（task_type=blend，四选一 MCQ 1/2/3/4）

> BLEnD 关注**日常生活的事实性文化知识**（具体名称、数字、地点），干扰项往往是**其他国家**的正确答案，且可能存在 "not-applicable" 选项。

##### (1) Guardian System Prompt

来源：`hf_cac_config_blend.yaml` 中每个 agent 的 `guardian_prompt`。**所有 6 个 agent 的 `guardian_prompt` 文本完全相同**，且不含地域专家人设：

```
You are a helpful assistant with expertise in cross-cultural knowledge and practices.
```

##### (2) Auditor System Prompt

每个 agent 的 `auditor_prompt`，与 `guardian_prompt` 文本相同：

```
You are a helpful assistant with expertise in cross-cultural knowledge and practices.
```

##### (3) Judge System Prompt

```
You are a helpful assistant with expertise in cross-cultural knowledge and practices.
```

> 即 BLEnD 的 Guardian / Auditor / Judge **三者 system prompt 完全统一**为这一句通用助手提示，不做任何角色或地域区分。

##### (4) Guardian Per-Round User Prompt

```
Task: You will be given a question about everyday life and daily cultural knowledge in {target_country}. Select the correct option number.

IMPORTANT:
- Focus ONLY on {target_country}. Some options may be correct for other countries but wrong for {target_country}.
- If 'not-applicable' is an option, it may be correct when the premise does not apply to {target_country}.
- Base your answer on specific factual knowledge (names, places, times, customs) of {target_country}.

Question:
{question}
Answer (1, 2, 3, or 4):
```

##### (5) Auditor Per-Round User Prompt

**(a) 协商模式（看到对方回答）：**

```
Task: You are discussing the following everyday cultural knowledge question about {target_country} with another expert.

Question:
{question}
Other expert's answer: {guardian_response}

Critically evaluate their answer. Remember:
- Some options are facts about OTHER countries, not {target_country}.
- If you have different knowledge about {target_country}, trust your own reasoning.
- 'not-applicable' can be correct if the premise doesn't apply.
Provide your own answer with brief reasoning.
Answer (1, 2, 3, or 4):
```

**(b) 独立模式（不看对方回答，与 Guardian 同结构）：**

```
Task: You will be given a question about everyday life and daily cultural knowledge in {target_country}. Select the correct option number.

IMPORTANT:
- Focus ONLY on {target_country}. Some options may be correct for other countries but wrong for {target_country}.
- If 'not-applicable' is an option, it may be correct when the premise does not apply to {target_country}.
- Base your answer on specific factual knowledge (names, places, times, customs) of {target_country}.

Question:
{question}
Answer (1, 2, 3, or 4):
```

##### (6) Feedback User Prompt（MAD Stage 2）

与 CulturalBench 共用同一 MCQ 分支文本：

```
TARGET CULTURE: {target_country}

{question}

Your initial answer:
  [{agent_name}]: {own_response}

Other experts' answers:
  [{other_name}]: {other_response}
  ...

Respond to the other experts by providing any relevant feedback. If you disagree with anyone, explain why with cultural evidence. Respond in less than three sentences.
Response:
```

##### (7) Final Decision User Prompt（MAD Stage 3）

与 CulturalBench 共用同一 MCQ 分支文本：

```
TARGET CULTURE: {target_country}

{question}

=== Discussion Summary ===
Your initial answer:
  [{agent_name}]: {own_response}

Other experts' answers:
  [{other_name}]: {other_response}
  ...

Feedback from all experts:
  [{agent_name}] (you): {own_feedback}
  [{other_name}]: {other_feedback}
  ...
=== End Discussion ===

Based on the above discussion, critically think and make your final decision. Respond with the correct option number (1, 2, 3, or 4).
Answer (1, 2, 3, or 4):
```

##### (8) Judge User Prompt

注意：BLEnD 的 Judge prompt 措辞与 CulturalBench **不同**，强调跨国干扰项的甄别：

```
Task: You are a judge resolving a disagreement between experts about everyday cultural knowledge in {target_country}.

IMPORTANT: Some options may be facts true for other countries but wrong for {target_country}. Evaluate which expert provides the most accurate factual knowledge specifically about {target_country}.
If experts disagree, prefer the answer with concrete evidence specific to {target_country}.

Question:
{question}

*** Expert opinions ***
  [{name}] (Guardian/Auditor): {response}
  ...
*** End opinions ***

Final decision (1, 2, 3, or 4):
```

##### (9) Judge 分歧仲裁 User Prompt（MAD Stage 4）

与 CulturalBench 共用：

```
Task: You are a judge responsible for making a final decision based on the debate history between cultural experts. They have debated the following cultural knowledge question about {target_country}. Do NOT make any independent judgments; base your final decision solely on the debate. Evaluate the factual accuracy of each argument regarding cultural knowledge of {target_country}. Respond with the correct option number (1, 2, 3, or 4).

Question:
{question}

*** Debate starts ***
  [{name}]: {feedback}
  [{name}] (HOST-CULTURE GUARDIAN / Cross-Cultural Auditor) final answer: {response}
  ...
*** Debate ends ***

Final decision (1, 2, 3, or 4):
```

##### (10) Judge 失效兜底 User Prompt

与 CulturalBench 共用：

```
... (同 NorMAD 兜底 prompt 头部)

Respond with the correct option number (1, 2, 3, or 4).

Final decision (1, 2, 3, or 4):
```

---

#### 2.8.4 采样温度配置

三个数据集采用相同的温度设计意图（具体数值以各自配置文件为准）：

| 角色 / 阶段 | Temperature | 设计意图 |
|------------|-------------|--------|
| Guardian（初始 / 最终决策） | 低温（约 0.5） | 确保权威/最终回答精确、一致 |
| Auditor（互评 feedback） | 高温（约 0.9） | 提供多样的跨文化对比视角，避免趋同 |
| Judge（裁决） | 极低温（约 0.3） | 确保裁决稳定性 |

> CulturalBench/BLEnD 的 MAD 辩论流程中，feedback 阶段使用 Auditor 高温采样以保持观点多样性，final_decision 阶段切回 Guardian 低温采样以收敛答案；BLEnD 配置注释中特别提到使用较高的 Auditor 温度以防止过度从众。

---

#### 2.8.5 三个数据集 Prompt 的区别

##### (1) 答案空间不同

- **NorMAD**：三分类 `1 / 2 / 3`（1=可接受 / 2=不可接受 / 3=中性·不确定），所有 prompt 结尾为 `Answer: <number>`。
- **CulturalBench / BLEnD**：四选一 MCQ `1 / 2 / 3 / 4`，所有 prompt 结尾为 `Answer (1, 2, 3, or 4):` 或 `Final decision (1, 2, 3, or 4):`。

##### (2) System Prompt（角色人设）差异最大

- **NorMAD**：使用**详细的多行角色人设**——Guardian 是"HOST-CULTURE GUARDIAN（主场文化守护者，拥有首要权威与一票否决权）"，Auditor 是"CROSS-CULTURAL AUDITOR（跨文化审计员，需 defer 给 Guardian）"，Judge 是"neutral cultural fact-checker and final arbitrator（含三分类决策规则 + Guardian VETO + 亲和度仲裁协议）"。每个文化 agent 还带专属 cognitive foundation 描述。
- **CulturalBench**：使用**简短的"文化专家"人设**——每个 agent 是 `You are a cultural expert deeply versed in {地域} cultures...`，按 6 个地域不同；Guardian 与 Auditor 共用同一段文本；Judge 退化为通用句 `You are a helpful assistant with expertise in cross-cultural knowledge and practices.`
- **BLEnD**：**完全统一**——Guardian / Auditor / Judge 三者 system prompt 全部是同一句 `You are a helpful assistant with expertise in cross-cultural knowledge and practices.`，不含任何地域专家或角色人设。

##### (3) 协作流程与 User Prompt 结构不同

- **NorMAD**：Guardian → Auditor（单向看 Guardian 回答）→ 条件 Judge。User prompt 以 `TARGET CULTURE:` 开头，要求 `Reasoning: ... / Answer:` 的结构化输出，并强调 Guardian 权威、VETO、亲和度仲裁；Judge prompt 含 "约 28% 题目正确答案是 3（中性）" 的校准提醒（CulturalBench/BLEnD 无此提醒）。
- **CulturalBench / BLEnD**：Auditor **独立起步**，随后走 **MAD 式对称辩论**（feedback 互评 → final_decision 重决策），最终**纯多数投票**（无 Guardian 特权）。User prompt 以 `Task:` 开头，要求 "explain in less than three sentences" 的简洁输出；Judge 强调 "Do NOT make any independent judgments / base solely on expert opinions"。

##### (4) CulturalBench 与 BLEnD 之间的细微差异

虽然两者都是四选一 MCQ + MAD 辩论流程，并共用 feedback / final_decision / disagreement / fallback 的 MCQ 分支，但 Guardian/Auditor/Judge 的**核心提问措辞不同**：

- **CulturalBench**：偏**解释性文化知识**。Guardian/Auditor 强调 "Do not make any extra inferences outside of the given context and country. Only align to the country given."；独立模式 Auditor 额外注入 "From your perspective as an expert in [{agent}]..." 的自身文化视角以提升集成多样性。
- **BLEnD**：偏**日常事实性知识**。Guardian/Auditor 强调三条 IMPORTANT 提示——"只关注本国/某些选项是其他国家的事实/'not-applicable' 可能正确/基于具体事实知识（名称、地点、时间、习俗）"；协商模式 Auditor 鼓励 "trust your own reasoning"；Judge 措辞也与 CulturalBench 不同，专门提示 "Some options may be facts true for other countries but wrong for {target_country}"，要求甄别跨国干扰项。

