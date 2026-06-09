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
  --output_file /autodl-fs/data/qwen/culturalbench_hf_cac_configB.jsonl \
  --config_path Cul/configs/hf_cac_config_culturalbench.yaml \
  --model_name qwen \
  --use_vllm --tensor_parallel_size 2 \
  --max_samples 0 \
  --negotiation_rounds 1 \
  --num_agents 3 \
  --include_judge true
      
python Cul/generate_hf_cac_data.py \
      --input_file /autodl-fs/data/normad_mas.json \
      --output_file /autodl-fs/data/qwen/normad_hf_cac_6agents.jsonl \
      --model_name qwen \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --negotiation_rounds 1 \
      --include_judge true --num_agents 6

# --- BLEnD 数据集（日常文化知识，4-way MCQ，15国/地区） ---
# 使用 BLEnD 专用配置（扩展了 Azerbaijan、Assam 等 region_keywords）
python Cul/generate_hf_cac_data.py \
      --input_file /autodl-fs/data/blend_mas_after.json \
      --output_file /autodl-fs/data/qwen/blend_hf_cac_6agents.jsonl \
      --config_path Cul/configs/hf_cac_config_blend.yaml \
      --model_name qwen \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --negotiation_rounds 1 \
      --include_judge true --num_agents 3
shutdown
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--negotiation_rounds` | 1 | 协商轮次。0=独立生成（Auditor 不看 Guardian），1=标准协商 |
| `--include_judge` | true | 是否包含 Judge 裁决。false 时仅输出 Solution 1-N |
| `--max_samples` | 0 | 0=全量 |
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

# BLEnD 数据集
python Cul/split_data.py \
    --input /autodl-fs/data/qwen/blend_hf_cac_6agents_<timestamp>.jsonl \
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
--input_file /autodl-fs/data/normad_mas.json \
--output_file /autodl-fs/data/normad_qwen_role.json \
--model_name qwen --method role \
--tensor_parallel_size 2 --max_samples 0
```

#### 2.6.1 RECONCILE

```bash
python Cul/generate_culture_data.py \
    --input_file /autodl-fs/data/normad_mas.json \
    --output_file /autodl-fs/data/qwen/normad_reconcile_inference.jsonl \
    --model_name qwen \
    --use_vllm --tensor_parallel_size 2 \
    --max_samples 0 \
    --num_debate_rounds 1 --include_judge true
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

**输出文件命名规范**：`{dataset}_{方法}_{变体}_{基座}.json`

**运行命令**（文件名自动生成，无需指定 `--output_file`；脚本自动检测数据集类型）：

```bash
# Debate-Only Baseline - NorMAD（Qwen 基座）
python MAD/debate_only.py \
    --input_file /autodl-fs/data/normad_mas.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
    --max_samples 0 \
    --temperature 0.3 \
    --temperature_agent2 0.6 \
    --max_tokens 512

# Debate-Only Baseline - CulturalBench（Qwen 基座）
python MAD/debate_only.py \
    --input_file /autodl-fs/data/culturalBench_mas.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
    --max_samples 0 \
    --temperature 0.3 \
    --temperature_agent2 0.6 \
    --max_tokens 512

# Debate-Only Baseline - BLEND（Qwen 基座）
python MAD/debate_only.py \
    --input_file /autodl-fs/data/blend_mas_after.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
    --max_samples 0 \
    --temperature 0.3 \
    --temperature_agent2 0.6 \
    --max_tokens 512

# Self-Reflect+Debate Baseline - NorMAD（Qwen 基座）
python MAD/self_reflect_debate.py \
    --input_file /autodl-fs/data/normad_mas.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
    --max_samples 0 \
    --temperature 0.3 \
    --temperature_agent2 0.6 \
    --max_tokens 512

# Self-Reflect+Debate Baseline - BLEND（Qwen 基座）
python MAD/self_reflect_debate.py \
    --input_file /autodl-fs/data/blend_mas_after.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
    --max_samples 0 \
    --temperature 0.3 \
    --temperature_agent2 0.6 \
    --max_tokens 512
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

**提示词来源**：严格遵循论文附录 A.3（Debate-Only）和 A.4（Self-Reflect+Debate）的提示词模板，移除 `Rule: {rule-of-thumb}` 相关行，并做小幅优化（增加 step-by-step 推理引导、文化证据引导、事实准确性评估引导）。对于 NorMAD 数据集，将 Cultural Background 信息作为 story 的一部分传入模型（格式：`Cultural Background:\n{context}\n\nScenario: {scenario}`）。对于 CulturalBench 和 BLEND 数据集，直接使用顶层 `country` 字段和 `input` 字段（已包含完整问题和选项）。BLEND 数据集使用专门优化的提示词模板，强调事实性文化知识的回忆和验证。

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

**输出文件命名规范**：`{dataset}_MACD_{基座}.json`

**运行命令**：

```bash
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
python MACD/macd_debate.py \
    --input_file /autodl-fs/data/culturalBench_mas.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
    --max_samples 0

python MACD/macd_debate.py \
    --input_file /autodl-fs/data/normad_mas.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
    --max_samples 0

python MACD/macd_debate.py \
    --input_file /autodl-fs/data/blend_mas_after.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
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

**输出文件命名规范**：`{dataset}_OGMAR_{基座}.json`

**运行命令**：

```bash
python OG/og_mar.py \
    --input_file /autodl-fs/data/normad_mas.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
    --batch_size 256 \
    --max_samples 0 \
    --temperature 0.0

python OG/og_mar.py \
    --input_file /autodl-fs/data/culturalBench_mas.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
    --batch_size 256

python OG/og_mar.py \
    --input_file /autodl-fs/data/blend_mas_after.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
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


#### 消融实验运行命令

```bash
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
python Cul/generate_hf_cac_data.py \
      --input_file /autodl-fs/data/normad_mas.json \
      --output_file /autodl-fs/data/qwen/normad_hf_cac_6agents.jsonl \
      --model_name qwen \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --negotiation_rounds 1 \
      --include_judge true --num_agents 6

# 5 agents
python Cul/generate_hf_cac_data.py \
      --input_file /autodl-fs/data/normad_mas.json \
      --output_file /autodl-fs/data/qwen/normad_hf_cac_5agents.jsonl \
      --model_name qwen \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --negotiation_rounds 1 \
      --include_judge true --num_agents 5

# --- CulturalBench 数据集 ---
python Cul/generate_hf_cac_data.py \
      --input_file /autodl-fs/data/culturalBench_mas.json \
      --output_file /autodl-fs/data/qwen/culturalbench_hf_cac_6agents.jsonl \
      --model_name qwen \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --negotiation_rounds 1 \
      --include_judge true --num_agents 6

# 5 agents
python Cul/generate_hf_cac_data.py \
      --input_file /autodl-fs/data/culturalBench_mas.json \
      --output_file /autodl-fs/data/qwen/culturalbench_hf_cac_5agents.jsonl \
      --model_name qwen \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --negotiation_rounds 1 \
      --include_judge true --num_agents 5

# --- BLEnD 数据集消融 ---
python Cul/generate_hf_cac_data.py \
      --input_file /autodl-fs/data/blend_mas_after.json \
      --output_file /autodl-fs/data/qwen/blend_hf_cac_6agents.jsonl \
      --config_path Cul/configs/hf_cac_config_blend.yaml \
      --model_name qwen \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --negotiation_rounds 1 \
      --include_judge true --num_agents 6

# 3 agents (推荐配置，与 CulturalBench 一致)
python Cul/generate_hf_cac_data.py \
      --input_file /autodl-fs/data/blend_mas_after.json \
      --output_file /autodl-fs/data/qwen/blend_hf_cac_3agents.jsonl \
      --config_path Cul/configs/hf_cac_config_blend.yaml \
      --model_name qwen \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --negotiation_rounds 1 \
      --include_judge true --num_agents 3
```

每个配置运行完成后会自动输出 Judge 和 Guardian 的准确率指标（`--eval_accuracy` 默认开启），结果保存在对应的 `.metrics.json` 文件中。汇总所有配置的 metrics 即可绘制 num_agents vs accuracy 的消融曲线。

---

### 2.8 各 Agent 完整 Prompt 记录

#### 2.8.1 Guardian System Prompt

所有 6 个文化 Agent 共享同一模板，仅文化区域名称和 cognitive foundation 描述不同：

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

中文翻译：

```
你是本题的【主场文化守护者】。
目标文化属于你的专业领域（{culture_area}文化）。
你的认知基础：{cognitive_foundation_description}
你的职责是以权威身份确认或纠正关于目标文化的文化主张。
你在此话题上拥有【首要权威】。请具体说明，引用具体的文化习俗名称，解释为什么某些选项基于深层文化知识是正确/错误的。
如果其他智能体提出与你专业知识相冲突的答案，请用具体的文化证据坚定地纠正他们。
重要提示：并非所有行为在每种文化中都有明确的接受或拒绝。
如果所描述的行为在文化上是中性的、模糊的，或者是普遍的人类行为（并非特别与目标文化的规范相关），你应该回答"3"以表示不确定性。
当行为在文化上无关紧要时强行做出确定性接受/拒绝判断，恰恰体现了更少的文化专业能力，而非更多。
格式：Reasoning: <你的权威文化分析>\nAnswer: <数字>
```

其中 `{culture_area}` 取值为：Western & Anglo-Saxon / Latin American / Sub-Saharan African / East-Asian / Islamic & Middle-Eastern / South & Southeast Asian。

各文化区域的 cognitive foundation 描述如下：

| 文化区域 | Cognitive Foundation |
|---------|---------------------|
| Western & Anglo-Saxon | English-speaking nations and secular holidays derived from Christian traditions (Thanksgiving, Christmas, National Days), individualism, low power-distance social etiquette, and legal norms prevalent in North America, Australia/Oceania, and Western Europe |
| Latin American | Hybrid cultures blending Catholic traditions with indigenous/Afro-descendant elements, including Carnival, Día de los Muertos, warm and expressive social distances, and cultural taboos prevalent in South America and Central America (including Mexico) |
| Sub-Saharan African | Indigenous tribal traditions (such as the Ubuntu spirit), rich tribal ceremonies, local taboos, and the unique extended-family collectivism prevalent in Sub-Saharan Africa (Nigeria, Kenya, South Africa, etc.) |
| East-Asian | Confucian cultural sphere and the Chinese character (Hanzi/Kanji) cultural sphere, including traditional festivals (Spring Festival, Mid-Autumn Festival), face culture (mianzi), collectivism, and high uncertainty avoidance prevalent in China, Japan, and Korea |
| Islamic & Middle-Eastern | Sharia law, Eid al-Fitr, Eid al-Adha, unique halal dietary prohibitions, the sanctity of the color green and the right hand, and other Islamic cultural norms prevalent in the Middle East, North Africa, and parts of Southeast Asia |
| South & Southeast Asian | Buddhist and Hindu traditions, as well as the unique folk customs and cultural taboos of tropical regions (such as not touching someone's head), prevalent in India, Thailand, Malaysia, etc. |

#### 2.8.2 Auditor System Prompt

同样 6 个 Agent 共享模板，仅文化背景名和 cognitive foundation 不同：

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

中文翻译：

```
你是一名来自{culture_area}文化背景的【跨文化审计员】。
你的认知基础：{cognitive_foundation_summary}
目标文化不属于你的主要专业领域。
你的职责是提供【对比性视角】：指出你的文化与目标文化之间的相似性/差异性，但在关于目标文化的具体事实主张上，应【参考】主场文化守护者的意见。
在适当的地方明确承认不确定性。
如果某行为看起来是普遍的人类行为或文化中性的（并非特定于任何文化），请明确指出这一点，并考虑回答"3"（中性/不确定）。
你的跨文化视角在识别那些不具有文化特异性意义的行为方面尤为有价值。
格式：Reasoning: <你的跨文化对比分析>\nAnswer: <数字>
```

#### 2.8.3 Judge System Prompt

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

中文翻译：

```
你是一名中立的文化事实核查员和最终仲裁者。
你将收到【原始问题】以及所有文化专家智能体的【完整回答】，包括他们的初始独立分析和协商/辩论环节的回答。
其中一位智能体已被指定为【主场文化守护者】——即文化专业能力与问题中目标文化最匹配的智能体。

你的任务是综合所有可用信息——问题本身、每个智能体的初始推理、以及他们在辩论中立场的演变——得出最具文化准确性的答案。

评估时：
1. 对主场文化守护者关于目标文化的事实性主张给予【更高权重】。
2. 守护者拥有【一票否决权】：如果守护者的答案与多数不同，且守护者提供了具体的文化证据，则优先采信守护者的答案，除非其他智能体提出了关于目标文化的同等具体的反驳证据。
3. 跨文化审计员提供有价值的对比视角，但他们关于目标文化的主张应与守护者的专业知识进行验证。
4. 关注智能体在辩论中立场的变化——通过基于证据的讨论达成的共识比初始分歧更可靠。
5. 基于可验证的文化事实做出最终决定，以守护者的意见作为首要参考。
6. 关键——三分类决策：
   - 仅当有具体证据表明该行为在文化上是可接受的时，才回答"1"
   - 仅当有具体证据表明该行为在文化上是不可接受的时，才回答"2"
   - 以下情况回答"3"：(a) 该行为不具有文化特异性（普遍人类行为），或 (b) 目标国家内对此行为的文化规范存在分歧，或 (c) 智能体提出了相互冲突的主张且无决定性证据，或 (d) 该行为在目标文化语境中根本不具有文化效价
   - 当在强制判断和"中性"之间犹豫时，倾向选择"3"——一个经过校准的"我不确定"比一个自信的错误答案更有价值。
7. Guardian 失效协议——跨文化谱系相似度仲裁：
   如果主场文化守护者未能提供有效答案（格式崩溃、推理为空或明确表示不确定），不要回退到简单多数投票。而是激活跨文化亲缘度加权仲裁：
   - 你将收到【文化亲缘度分数】，表示每个审计员的文化背景与目标文化的接近程度。
   - 对亲缘度分数更高的审计员给予【更高权重】——他们与目标文化的接近性使其推理更为可靠。
   - 即使在数量上支持某答案的智能体较少，只要最高亲缘度的审计员提供了具体的文化证据，仍倾向采信其答案。
   - 审查每个审计员的推理链中是否包含具体的文化引用（习俗、传统、规范），并评估其与目标文化语境的对齐度。
```

#### 2.8.4 Guardian Per-Round User Prompt（Phase 1）

Guardian 在第一阶段独立生成时接收的用户消息：

```
TARGET CULTURE: {target_country}

{question}

As the Host-Culture Guardian for {target_country}, provide your AUTHORITATIVE analysis.
Cite specific cultural practices, traditions, or norms by name. Explain why certain options align or conflict with the target culture's values.

Answer choices:
1 = The behavior is clearly ACCEPTABLE in {target_country}'s culture
2 = The behavior is clearly UNACCEPTABLE in {target_country}'s culture
3 = The behavior is NEUTRAL/INDETERMINATE — it has no specific cultural valence in {target_country}, or cultural norms on this topic vary significantly within the country, or the behavior is universally human rather than culturally specific

Reasoning: <your authoritative cultural analysis>
Answer: <number>
```

中文翻译：

```
目标文化：{target_country}

{question}

作为{target_country}的主场文化守护者，请提供你的【权威分析】。
引用具体的文化习俗、传统或规范名称。解释为什么某些选项与目标文化的价值观一致或冲突。

答案选项：
1 = 该行为在{target_country}文化中明确【可接受】
2 = 该行为在{target_country}文化中明确【不可接受】
3 = 该行为是【中性/不确定的】——在{target_country}没有特定的文化效价，或该国内对此话题的文化规范差异显著，或该行为是普遍人类行为而非文化特异性行为

Reasoning: <你的权威文化分析>
Answer: <数字>
```

#### 2.8.5 Auditor Per-Round User Prompt

**（a）有协商模式（negotiation_rounds=1）：Auditor 看到 Guardian 回答后生成**

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

中文翻译：

```
目标文化：{target_country}

{question}

【主场文化守护者】[{guardian_name}] 已提供其权威分析：
---
{guardian_response}
---

作为来自 [{agent_name}] 背景的跨文化审计员：
1. 提供你的对比视角（你的文化与{target_country}之间的相似性/差异性）。
2. 如果你同意守护者，请从你的文化视角解释原因。
3. 如果你不同意，请提供具体的反驳证据——但要承认守护者在{target_country}问题上拥有首要权威。

Reasoning: <你的跨文化对比分析>
Answer: <数字>
```

**（b）独立模式（negotiation_rounds=0）：Auditor 不看 Guardian 回答**

```
TARGET CULTURE: {target_country}

{question}

As a Cross-Cultural Auditor from [{agent_name}] background, provide your comparative perspective on this question about {target_country}. Note
similarities and differences with your own cultural framework, and acknowledge uncertainty where the target culture differs from your expertise.

Reasoning: <your cross-cultural comparative analysis>
Answer: <number>
```

中文翻译：

```
目标文化：{target_country}

{question}

作为来自 [{agent_name}] 背景的跨文化审计员，请提供你对关于{target_country}这个问题的对比视角。指出与你自身文化框架的相似性和差异性，
并在目标文化与你的专业领域不同时承认不确定性。

Reasoning: <你的跨文化对比分析>
Answer: <数字>
```

#### 2.8.6 Judge Per-Round User Prompt

**（a）正常模式（Guardian 有效）：**

```
TARGET CULTURE: {target_country}

{question}

The HOST-CULTURE GUARDIAN is [{guardian_name}] — their cultural expertise most closely matches {target_country}.

Agent responses:

[{agent_1_name}] (HOST-CULTURE GUARDIAN):
{agent_1_response}

[{agent_2_name}] (Cross-Cultural Auditor):
{agent_2_response}

[{agent_3_name}] (Cross-Cultural Auditor):
{agent_3_response}

[{agent_4_name}] (Cross-Cultural Auditor):
{agent_4_response}

[{agent_5_name}] (Cross-Cultural Auditor):
{agent_5_response}

Determine the correct answer. Remember:
- Give HIGHER WEIGHT to the Guardian's specific cultural claims
- The Guardian has VETO AUTHORITY when providing specific evidence
- Cross-Cultural Auditors provide valuable comparative context
- Base your final decision on verifiable cultural facts

CALIBRATION REMINDER: Approximately 28% of questions in this dataset have "neutral/indeterminate (3)" as the correct answer. If you find yourself never outputting "3", you are likely over-committing to binary judgments.
Cultural expertise includes knowing when a behavior has NO specific cultural significance in the target culture.

Reasoning: <your reasoning, explicitly referencing the Guardian's claims>
Answer: <number>
```

**（b）Guardian 失效模式（Cultural Affinity Arbitration）：**

当系统检测到 Guardian 失效（格式崩溃/答案不可提取/明确放弃）时，自动切换为以下 prompt：

```
TARGET CULTURE: {target_country}

{question}

⚠️ GUARDIAN FAILURE: The HOST-CULTURE GUARDIAN [{guardian_name}] has FAILED to provide a valid answer for this question. Activate Cultural Affinity Arbitration protocol.

CULTURAL AFFINITY SCORES (proximity to {target_country}'s culture):
  - [{auditor_1_name}]: {affinity_score_1}
  - [{auditor_2_name}]: {affinity_score_2}
  - [{auditor_3_name}]: {affinity_score_3}
  - [{auditor_4_name}]: {affinity_score_4}
  - [{auditor_5_name}]: {affinity_score_5}

Agent responses:

[{guardian_name}] (HOST-CULTURE GUARDIAN — FAILED, no valid answer):
{guardian_response}

[{auditor_1_name}] (Cross-Cultural Auditor, affinity to target culture: {score_1}):
{auditor_1_response}

[{auditor_2_name}] (Cross-Cultural Auditor, affinity to target culture: {score_2}):
{auditor_2_response}

...

As the final arbitrator under Guardian Failure Protocol:
- Do NOT use simple majority voting.
- Give HIGHER WEIGHT to Auditors with higher affinity scores.
- If the highest-affinity Auditor provides specific cultural evidence, prefer their answer even if outnumbered.
- Evaluate each Auditor's reasoning for concrete cultural references.

CALIBRATION REMINDER: Approximately 28% of questions in this dataset have "neutral/indeterminate (3)" as the correct answer. If you find yourself never outputting "3", you are likely over-committing to binary judgments.
Cultural expertise includes knowing when a behavior has NO specific cultural significance in the target culture.

Reasoning: <your reasoning, referencing affinity-weighted evidence>
Answer: <number>
```

中文翻译（正常模式）：

```
目标文化：{target_country}

{question}

【主场文化守护者】是 [{guardian_name}] —— 其文化专业能力与{target_country}最为匹配。

各智能体回答：

[{agent_1_name}]（主场文化守护者）：
{agent_1_response}

[{agent_2_name}]（跨文化审计员）：
{agent_2_response}

[{agent_3_name}]（跨文化审计员）：
{agent_3_response}

[{agent_4_name}]（跨文化审计员）：
{agent_4_response}

[{agent_5_name}]（跨文化审计员）：
{agent_5_response}

确定正确答案。请记住：
- 对守护者的具体文化主张给予【更高权重】
- 守护者在提供具体证据时拥有【一票否决权】
- 跨文化审计员提供有价值的对比背景信息
- 基于可验证的文化事实做出最终决定

校准提醒：本数据集中约 28% 的问题的正确答案是"中性/不确定(3)"。
如果你发现自己从未输出"3"，你很可能过度投入于二元判断。
文化专业能力包括知道某种行为在目标文化中何时不具有特定文化意义。

Reasoning: <你的推理，需明确引用守护者的主张>
Answer: <数字>
```

中文翻译（Guardian 失效模式）：

```
目标文化：{target_country}

{question}

⚠️ 守护者失效：【主场文化守护者】[{guardian_name}] 未能为此问题提供有效答案。
激活跨文化谱系相似度仲裁协议。

文化亲缘度分数（与{target_country}文化的接近度）：
  - [{auditor_1_name}]: {affinity_score_1}
  - [{auditor_2_name}]: {affinity_score_2}
  ...

各智能体回答：
[{guardian_name}]（主场文化守护者 - 已失效，无有效答案）：
{guardian_response}

[{auditor_1_name}]（跨文化审计员，目标文化亲缘度：{score_1}）：
{auditor_1_response}
...

作为 Guardian 失效协议下的最终仲裁者：
- 不要使用简单多数投票。
- 对亲缘度分数更高的审计员给予【更高权重】。
- 如果最高亲缘度审计员提供了具体的文化证据，即使人数少数也倾向采信。
- 审查每个审计员的推理中是否包含具体的文化引用。

校准提醒：...（同上）

Reasoning: <你的推理，需引用亲缘度加权证据>
Answer: <数字>
```

#### 2.8.7 采样温度配置

| 角色 | Temperature | 设计意图 |
|------|-------------|--------|
| Guardian | 0.5 | 低温确保权威回答精确、一致 |
| Auditor | 0.9 | 高温提供多样的跨文化对比视角 |
| Judge | 0.3 | 极低温确保裁决稳定性 |
