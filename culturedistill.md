# CAMAD: Culture-Aware Multi-Agent Distillation Framework

## 1. 概览

贡献：

### 1.1 HF-CAC：一种新的多智能体协作范式（创新点一）

HF-CAC（Home-Field Culture-Activated Collaboration）是我们提出的面向文化对齐任务的多智能体协作新范式。核心思想是：针对文化知识的"属地性"和"不对称性"特征，引入"主场/客场"动态权威机制——根据目标国家自动激活对应文化背景的 Agent 作为主场守护者（Guardian），赋予其更高话语权和一票否决权，其余 Agent 作为跨文化审视者（Auditor）提供对比视角。

### 1.2 CAMAD：基于 HF-CAC 推理数据的文化感知蒸馏框架

CAMAD 是基于 HF-CAC 生成的结构化推理数据构建的三阶段蒸馏框架，目标是将多智能体系统的跨文化推理能力注入单体语言模型，使其具备主场文化确权能力（Guardian 的知识精度）、跨文化边界感知能力（Auditor 的对比视角）、以及文化一致性的自我过程监督能力（PRM 引导的推理路径优化）。三阶段如下：

```
Stage 1: 主场权威加权SFT → 单体模型学习 Guardian 的确权推理模式，掩码 Auditor 早期混淆 Token

Stage 2: 开卷式步骤标注 → 审计器在 Ground Truth 先验下，对推理步骤打全正值离散标签 {0.1, 0.5, 0.9}

Stage 3: 文化感知过程奖励 → GRPO强化学习 → PRM 保留 Sigmoid 激活 + 类别加权 MSE 训练；GRPO 使用加权平均 R_total 优化推理路径（量纲完美统一于 [0,1]）
```

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
- 如果 Guardian 的答案与多数不同，但 Guardian 提供了具体文化证据 → 采信 Guardian
- 如果 Guardian 的答案与多数相同 → 直接确认
- 若 Guardian 失效 → 激活 Judge 启发式跨文化仲裁机制

**Guardian 失效的判定条件**：(a) Guardian 回答中提取不到有效答案（格式崩溃、输出截断），OR (b) Guardian 的推理中包含明确的不确定性放弃标记（如 "I'm not sure"、"I don't have enough knowledge"、推理内容为空）。

**跨文化谱系相似度仲裁（Cultural Affinity Arbitration）**：

当主场 Guardian 未能给出清晰答案时，说明该国的文化知识可能属于"极度冷门"或"长尾知识"。此时 Judge 不做简单的多数投票计数，而是升级为"基于文化谱系的客观仲裁者"，通过跨文化亲缘度比对进行主动判断：

判断逻辑：内置一张 6×6 的文化亲缘度矩阵（Cultural Affinity Matrix），刻画任意两个文化圈之间的"谱系距离"。当 Guardian 失效时，Judge 在裁决中对各 Auditor 的意见按亲缘度加权——与目标文化亲缘度更高的 Auditor 的推理获得更大权重。

示例：若题目考的是埃及（属于 Islamic & Middle-Eastern 文化圈），Guardian（伊斯兰 Agent）失效。此时 Judge 参照亲缘度矩阵：Sub-Saharan African Agent（亲缘度 0.5，近亲地缘）的推理权重 > South & Southeast Asian Agent（亲缘度 0.3）> Western & Anglo-Saxon Agent（亲缘度 0.1）。即使投 "no" 的 Agent 更多，只要亲缘度更高的 Agent 给出了带有具体文化证据的不同答案，Judge 倾向于采信高亲缘度 Agent 的判断。

亲缘度矩阵设计原则：基于地理邻近性、宗教传统共享度、历史交流深度三个维度综合打分（0-1），硬编码在配置文件中，确保确定性和可复现性。

### 2.4 运行命令

```bash
# 全量生成（NormAD 数据集，Qwen）— 自动检测为 normad 类型，使用 hf_cac_config.yaml
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
python Cul/generate_hf_cac_data.py \
      --input_file /autodl-fs/data/normad_mas.json \
      --output_file /autodl-fs/data/qwen/normad_hf_cac_inference.jsonl \
      --model_name qwen \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --negotiation_rounds 1 \
      --include_judge true
shutdown
```

```bash
python Cul/generate_hf_cac_data.py \
      --input_file /autodl-fs/data/culturalBench_mas.json \
      --output_file /autodl-fs/data/qwen/culturalbench_hf_cac_inference.jsonl \
      --model_name qwen \
      --use_vllm --tensor_parallel_size 2 \
      --max_samples 0 --negotiation_rounds 1 \
      --include_judge true
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--negotiation_rounds` | 1 | 协商轮次。0=独立生成（Auditor 不看 Guardian），1=标准协商 |
| `--include_judge` | true | 是否包含 Judge 裁决。false 时仅输出 Solution 1-6 |
| `--model_name` | — | `llama`/`qwen`/完整路径 |
| `--max_samples` | 0 | 0=全量 |
| `--config_path` | None | 手动指定配置文件路径。不指定时根据数据集自动检测 |


**数据划分（生成 pkl 文件）**

```bash
python Cul/split_data.py \
    --input /autodl-fs/data/qwen/normad_hf_cac_inference.jsonl \
    --output /autodl-fs/data/qwen/normad_splits.pkl \
    --seed 42
```

| 参数 | 含义 |
|------|------|
| `--input` | HF-CAC 推理数据 JSONL |
| `--output` | 输出 pkl 文件路径（包含 train/val/test 三个 key）|
| `--seed` | 随机种子（默认 42，确保可复现）|


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

与原 RECONCILE 格式的区别：
- Solution 标题包含 `[GUARDIAN]`/`[AUDITOR]`/`[JUDGE]` 角色标签
- 额外输出 `guardian_idx` 和 `guardian_name` 字段，便于下游蒸馏管线使用
- Guardian 的推理路径包含权威确认语言，Auditor 包含对比/不确定性表达

### 2.6 Baseline

#### 2.6.0 RECONCILE

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
| `--max_samples` | 处理样本数（0=全量，>0=取前 N 条用于快速测试）|
| `--num_debate_rounds` | 辩论轮数（覆盖 config 中的值，0=无辩论仅独立推理）|
| `--include_judge` | 是否包含 Judge 裁决（`true`/`false`）|
| `--batch_size` | vLLM 批次大小（默认 8）|


#### 2.6.1 MAD 

**方法简介**：MAD 是 Ki et al. (2024) 提出的多智能体辩论框架，通过两个 LLM Agent 对文化场景进行辩论来达成更准确的文化对齐判断。论文提出了两种变体：

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
    --temperature 0.7 \
    --max_tokens 512

# Debate-Only Baseline - CulturalBench（Qwen 基座）
python MAD/debate_only.py \
    --input_file /autodl-fs/data/culturalBench_mas.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
    --max_samples 0 \
    --temperature 0.7 \
    --max_tokens 512

# Self-Reflect+Debate Baseline - NorMAD（Qwen 基座）
python MAD/self_reflect_debate.py \
    --input_file /autodl-fs/data/normad_mas.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
    --max_samples 0 \
    --temperature 0.7 \
    --max_tokens 512

# Self-Reflect+Debate Baseline - CulturalBench（Qwen 基座）
python MAD/self_reflect_debate.py \
    --input_file /autodl-fs/data/culturalBench_mas.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
    --max_samples 0 \
    --temperature 0.7 \
    --max_tokens 512
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_file` | 数据集路径（自动检测 normad/culturalBench） | 必填 |
| `--output_dir` | 输出目录（默认 /autodl-fs/data/mad） | None |
| `--tensor_parallel_size` | vLLM 张量并行数 | 1 |
| `--batch_size` | 每批处理样本数 | 8 |
| `--max_samples` | 最大处理样本数（0=全部） | 0 |
| `--temperature` | 采样温度 | 0.7 |
| `--max_tokens` | 最大生成 token 数 | 512 |

**数据集自动检测**：脚本根据数据样本的 `output` 字段值范围自动判断数据集类型：
- NorMAD：输出为 "1"/"2"/"3"（Yes/No/Neither），提示词使用论文原始模板
- CulturalBench：输出为 "1"/"2"/"3"/"4"（4 选 1 MCQ），提示词为论文模板的最小化改写（将 "story" → "question"，"Yes, No or Neither" → "1, 2, 3, or 4"）

**提示词来源**：严格遵循论文附录 A.3（Debate-Only）和 A.4（Self-Reflect+Debate）的提示词模板，移除 `Rule: {rule-of-thumb}` 相关行。对于 NorMAD 数据集，将 Cultural Background 信息作为 story 的一部分传入模型（格式：`Cultural Background:\n{context}\n\nScenario: {scenario}`）。对于 CulturalBench 数据集，直接使用顶层 `country` 字段和 `input` 字段（已包含完整问题和选项）。提示词模板本身仅做最小格式适配。

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

#### 2.6.2 MACD (Multi-Agent Cultural Debate)

**方法简介**：MACD 是 Tan et al. (2026) 提出的训练无关多智能体文化辩论框架，通过赋予 Agent 显式的文化身份（而非功能性角色）来缓解 LLM 的文化偏见。核心思想是：

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
# MACD Baseline（Qwen 基座，完整数据集，双卡并行）
python MACD/macd_debate.py \
    --input_file /autodl-fs/data/normad_mas.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
    --max_samples 0

# 快速测试（5 条样本）
python MACD/macd_debate.py \
    --input_file /autodl-fs/data/normad_mas.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
    --max_samples 5
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--tensor_parallel_size` | vLLM 张量并行数（推荐双卡设为 2） | 2 |
| `--max_samples` | 最大处理样本数（0=全部） | 0 |
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

#### 2.6.3 OG-MAR (Ontology-Guided Multi-Agent Reasoning)

**方法简介**：OG-MAR 是 Seo et al. (2026) 提出的本体引导多智能体推理框架，通过构建全球文化本体（ontology）来指导多智能体的文化对齐推理。核心创新：

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
# OG-MAR Baseline（Qwen 基座，完整数据集，论文默认参数）
python OG/og_mar.py \
    --input_file /autodl-fs/data/normad_mas.json \
    --model_name qwen \
    --tensor_parallel_size 2 \
    --batch_size 256 \
    --max_samples 0 \
    --temperature 0.0

# CulturalBench
python OG/og_mar.py 
    --input_file /autodl-fs/data/culturalBench_mas.json 
    --model_name qwen 
    --tensor_parallel_size 2 
    --batch_size 256
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--tensor_parallel_size` | vLLM 张量并行数 | 1 |
| `--batch_size` | 每批处理样本数 | 8 |
| `--max_samples` | 最大处理样本数（0=全部） | 0 |
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

### 2.7 代码结构

```
Cul/
├── scripts/
│   ├── convert_normad.py              # 数据格式转换：normad.jsonl → normad_mas.json
│   ├── convert_culturalbench.py       # 数据格式转换：CulturalBench-Easy.csv → culturalBench_mas.json
│   └── analyze_inference.py           # 推理结果分析脚本
├── configs/
│   ├── hf_cac_config.yaml             # HF-CAC 配置（NormAD：行为可接受性 1/2/3）
│   ├── hf_cac_config_cultureatlas.yaml  # HF-CAC 配置（CultureAtlas：文化深度比较 1/2）
│   ├── hf_cac_config_culturalbench.yaml # HF-CAC 配置（CulturalBench：文化知识四选一 1/2/3/4）
│   └── reconcile_config.yaml          # 原 RECONCILE 配置（baseline）
├── hf_cac_mas.py                      # HF-CAC 核心推理引擎（主场识别、两阶段 batch inference、Affinity Arbitration）
├── generate_hf_cac_data.py            # HF-CAC 数据生成入口（自动检测 normad/cultureatlas/culturalbench）
├── resume_hf_cac.py                   # HF-CAC 断点续跑工具
├── run_camad_pipeline.py              # CAMA-D 全流程 Pipeline 入口（一键串联 Phase 0-5）
├── split_data.py                      # 数据划分（8:1:1 → pkl）
├── evaluate.py                        # 评估脚本（支持 sft/rl/sft_rl 模式）
├── sft/
│   └── train_sft_weighted.py          # Stage 1: 主场权威加权 SFT 训练
├── step_label/
│   ├── split_steps.py                 # 推理步骤切分
│   ├── label_steps.py                 # Stage 2: 开卷式步骤标注（{0.1, 0.5, 0.9}）
│   ├── split_step_labels.py           # 步骤标签数据划分
│   └── validate_labels.py             # 标签验证
├── prm/
│   ├── train_prm_mse.py               # Stage 3a: 过程奖励模型训练（Sigmoid + 加权 MSE）
│   └── eval_prm.py                    # PRM 评估
└── grpo/
    └── train_grpo_v3.py               # Stage 3b: GRPO 强化学习训练
```

### 2.8 各 Agent 完整 Prompt 记录

#### 2.11.1 Guardian System Prompt

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

#### 2.11.2 Auditor System Prompt

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

#### 2.11.3 Judge System Prompt

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

#### 2.11.4 Guardian Per-Round User Prompt（Phase 1）

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

#### 2.11.5 Auditor Per-Round User Prompt

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

#### 2.11.6 Judge Per-Round User Prompt

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

#### 2.11.7 采样温度配置

| 角色 | Temperature | 设计意图 |
|------|-------------|---------|
| Guardian | 0.5 | 低温确保权威回答精确、一致 |
| Auditor | 0.9 | 高温提供多样的跨文化对比视角 |
| Judge | 0.3 | 极低温确保裁决稳定性 |

---


## 3. 主场权威加权 SFT

### 3.1 动机

HF-CAC 生成的多智能体对话数据中，包含了 Guardian（主场守护者）和 Auditor（客场审视者）两种角色的完整推理轨迹。Auditor 在辩论早期可能输出带有文化混淆、偏见或引导错误的内容。如果使用传统 SFT（对所有 Token 平等计算交叉熵），单体模型会在自回归预测中拟合这些"毒草 Token"，在内部种下文化混淆的种子。

### 3.2 核心策略：Token 级加权与掩码

**原则**：

- Guardian 的确权和纠偏 Token → 保留，loss 权重乘以 α（放大学习信号）
- Auditor 最终轮之前的对抗性输出（质疑、混淆、偏离目标文化的内容）→ labels 填充 -100（完全掩码，不参与梯度计算）
- Auditor 最终轮中被 Guardian 说服后的正确表态 → 保留，loss 权重 = 1.0（不放大，但允许学习"认知转换模式"）


### 3.3 超参数

| 参数 | 值 | 说明 |
|------|----|------|
| alpha (Guardian 权重) | 2.0 | Guardian Token 的 loss 放大系数 |
| Auditor 掩码范围 | 非最终轮全部 Token | 最终轮表态保留 |
| 学习率 | 2e-5 | 全参微调 |
| Epochs | 3 | 早停（val_acc 2 epoch 不提升） |

### 3.4 运行命令

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
    --eval_every_n_epochs 1
```

| 参数 | 含义 |
|------|------|
| `--data_pkl` | split_data.py 生成的 pkl 文件（包含 train/val/test 划分）|
| `--alpha` | Guardian Token 的 loss 权重放大系数（默认 2.0）|
| `--lora_r` | LoRA rank（默认 32，保证文化知识充分学习）|
| `--lr` | 学习率（LoRA 默认 2e-4，高于全参微调）|
| `--eval_every_n_epochs` | 每 N 个 epoch 在验证集上评估一次（默认 1）|

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
| 0.9（主场确权步） | 提供了目标文化的具体证据，直接支持正确答案 | Sigmoid → 0.9 | “在越南，‘li xi’（红包）是长辈给晚辈的传统...” |
| 0.5（中立讨论步） | 格式转换、通用逻辑过渡、同义词复述 | Sigmoid → 0.5 | "Let me analyze the options one by one..." |
| 0.1（文化混淆步） | 引入文化混淆，指向错误选项或应用了错误文化的价值观 | Sigmoid → 0.1 | “在西方文化中，贺卡是最常见的节日礼物，所以选3...” |

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

**基座模型**：Stage 1 SFT 训练完成的 student model（非独立小模型）。

**设计思路**：使用 SFT 后的模型作为 PRM 基座，因为该模型已经通过 Stage 1 学习了文化推理的语义表示，对文化相关 Token 具有更好的隐层表征。在此基础上添加线性回归头，以最小参数增量获得步骤级打分能力。

**架构**：

在 SFT 模型基座之上添加一个线性回归头（hidden_size → 1）和 Sigmoid 激活函数。前向推理时，将完整输入（含所有 Step）送入基座模型获取最后一层 hidden states，然后在每个 Step 终止符的位置提取对应的 hidden state 向量，经线性头映射为标量 logit，再通过 Sigmoid 压缩到 (0, 1) 区间，作为该 Step 的预测分数。最终输出为一组步骤级分数，每个分数对应一个 Step 的质量评估。

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

### 5.3 训练配置

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

### 5.4 验证指标

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

### 5.5 运行命令

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

注意：RL-only 模式不传 `--sft_adapter_path`，PRM 直接在原始 base model 上训练。

**PRM 评估**
```bash
python Cul/prm/eval_prm.py \
    --prm_path /autodl-fs/data/model/qwen/normad_camad_prm/best \
    --sft_path /autodl-fs/data/model/qwen/normad_camad_sft/best \
    --val_file /autodl-fs/data/qwen/normad_step_labels_val.jsonl
```
---

## 6. GRPO 强化学习

### 6.1 Reward 设计：加权平均形式

```
R_total = alpha * R_outcome + (1 - alpha) * Mean(R_process)
```

其中：
- `R_outcome ∈ {0, 1}`：答案正确性（规则可验证，答错为 0，答对为 1）
- `Mean(R_process) ∈ [0.1, 0.9]`：当前推理链中所有步骤的 PRM 得分（经 Sigmoid）的算术平均值。中间全走偏为 ~0.1，全中立为 ~0.5，完美主场确权为 ~0.9
- `alpha = 0.6`：结果奖励占主导

**具体数值示例**：
- 模型答对 + 推理全是文化混淆步：`R_total = 0.6 × 1 + 0.4 × 0.1 = 0.64`
- 模型答对 + 推理展现完美主场确权：`R_total = 0.6 × 1 + 0.4 × 0.9 = 0.96`
- 模型答错 + 推理全是确权步：`R_total = 0.6 × 0 + 0.4 × 0.9 = 0.36`
- 模型答错 + 推理全是混淆步：`R_total = 0.6 × 0 + 0.4 × 0.1 = 0.04`

**超参数 alpha=0.6 的逻辑支撑**：

在文化对齐任务中，"答对（事实正确）"是硬指标，底线不能丢，因此 R_outcome 必须占大头（0.6）。而"推理路径的文化合理性"（R_process）作为软约束，负责从多组全部答对的采样中，选出表现得最像主场 Guardian、最优雅的那条路径。0.4 的权重足以在组内拉开相对 Advantage 的差距，促使 GRPO 向主场思辨方向演化。

### 6.2 GRPO 在线采样流程

```
对每个 prompt (question, country)：
  1. 当前 policy 采样 G=10 条推理路径
  2. 对每条路径：
     a. 规则验证答案 → R_outcome ∈ {0, 1}
     b. 启发式切分推理步骤 → [Step 1], [Step 2], ...
     c. PRM 对每个 Step 终止符位置打分（Sigmoid 输出 ∈ (0,1)）→ scores[]
     d. Mean(R_process) = mean(scores)  // ∈ [0.1, 0.9]
     e. R_total = 0.6 * R_outcome + 0.4 * Mean(R_process)
  3. 组内计算 Advantage（RLOO baseline）
  4. 策略梯度更新 policy 参数
  5. 下一轮用更新后模型重新采样
```

### 6.3 训练配置

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

### 6.4 PRM 推理效率

GRPO 每轮需对 `prompt_count × G × avg_steps` 个 Step 打分。优化策略：

- PRM 冻结参数，纯推理模式（`torch.no_grad()`）
- 所有 Step 拼接为一个大 batch，单次前向传播完成
- PRM 使用 LoRA adapter，推理时合并权重（`merge_and_unload()`），无额外开销

### 6.5 运行命令

**GRPO 强化学习（SFT+RL 模式，LoRA，无 DeepSpeed）**
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


**GRPO 强化学习（无 SFT adapter，lr=5e-5，max_rounds=30）**
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

**备选: GRPO 强化学习（DeepSpeed ZeRO-3 版，train_grpo.py）**
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

### 6.6 测试 运行命令

```bash
# 评估 SFT 模型
python Cul/evaluate.py \
    --mode sft \
    --model_name qwen \
    --data_pkl /autodl-fs/data/qwen/normad_splits.pkl \
    --sft_adapter /autodl-fs/data/model/qwen/normad_camad_sft/best \
    --output_json /autodl-fs/data/model/qwen/eval_sft.json

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

---


## 7. 代码结构

### 7.1 目录树

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

标注 ★ 的文件为 CAMAD 管线核心代码。

### 7.2 Pipeline 入口与工具

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


## 8. 消融实验设计

### 8.1 核心蒸馏方案对比

| 实验组 | 训练方式 | 预期排序 |
|--------|---------|---------|
| Base | 无训练 | 最低 |
| SFT-only (equal weight) | 传统 SFT（无 Token 加权） | 中低 |
| SFT-only (CAMA-D Stage 1) | Token 级加权 SFT | 中 |
| RL-only | GRPO from base | 中高 |
| SFT + RL (CAMA-D full) | Stage 1 → Stage 3 | 最高 |
| MAS Oracle | 多智能体系统直接推理 | 上界 |

### 8.2 评估指标

| 指标 | 说明 |
|------|------|
| val_accuracy | 预测答案与 gold label 匹配率 |
| Cultural Sensitivity Score | 同一问题不同文化下答案分布 KL 散度均值 |
| Reasoning Coherence | LLM Judge 评估推理路径与答案的一致性 |
| Cultural Grounding | 推理路径中目标文化具体价值观关键词出现率 |
| Cultural Boundary Awareness | 模型是否能正确区分相邻文化（如越南 vs 中国） |

