# CAMA-D: Culture-Aware Multi-Agent Distillation Framework

## 1. 框架概览

### 1.1 核心目标

将 HFA-C²N 多智能体系统生成的结构化跨文化推理数据，通过三阶段蒸馏pipeline注入单体语言模型，使其具备：
- 主场文化确权能力（Guardian 的知识精度）
- 跨文化边界感知能力（Auditor 的对比视角）
- 文化一致性的自我过程监督能力（PRM 引导的推理路径优化）

### 1.2 三阶段总览

```
Stage 1: Home-Field Authority-Weighted SFT（主场权威加权监督微调）
  → 单体模型学习 Guardian 的确权推理模式，掩码 Auditor 早期混淆 Token

Stage 2: Open-Book Step Labeling（开卷式步骤标注）
  → 审计器在 Ground Truth 先验下，对推理步骤打全正值离散标签 {0.1, 0.5, 0.9}

Stage 3: Culture-Aware PRM → GRPO（文化感知过程奖励 → 强化学习）
  → PRM 保留 Sigmoid 激活 + 类别加权 MSE 训练；GRPO 使用加权平均 R_total 优化推理路径（量纲完美统一于 [0,1]）
```

## 2. HFA-C²N：基于主场权威激活的跨文化动态协商数据生成

### 2.1 动机与核心洞察

传统 RECONCILE 框架中，所有 Agent 无论讨论什么国家的题目，地位都是平等的。这在科学/逻辑推理任务中合理，但在文化对齐任务中存在根本性缺陷——文化知识具有强烈的"属地性"和"不对称性"。

例如：关于中国春节的知识，东亚文化 Agent 的话语权天然应该高于欧洲文化 Agent；关于巴西狂欢节的知识，拉美文化 Agent 比北美 Agent 更具权威性。然而在传统 RECONCILE 中，一个对目标文化一知半解的客场 Agent 与一个对目标文化了如指掌的主场 Agent 享有相同的投票权和影响力，这会导致"西方语料主导型错误"——在小众、非西方国家的题目上，被训练数据中占主导地位的西方视角带偏。

### 2.2 方法论定位

HFA-C²N（Home-Field Authority-Activated Cross-Cultural Negotiation）是针对文化对齐任务量身定制的算法架构创新，核心思想是：根据目标国家动态调整 Agent 的权威度，引入"主场/客场"不对称机制，使多智能体系统在文化题目上产生更高质量的推理数据。

与"简单搬用 RECONCILE"的本质区别：
- RECONCILE：所有 Agent 平等 → 多数投票 → 均质推理路径
- HFA-C²N：动态权威激活 → 主场确权 + 客场审视 → 结构化对比推理路径

这一改进直接回应了"仅是应用型论文，无方法论创新"的审稿质疑。

### 2.3 核心机制设计

#### 2.3.1 主场权威激活（Home-Field Authority Activation）

系统解析输入数据中的目标国家（如 China），自动将对应文化背景的 Agent（如 East-Asian Agent）标记为"主场文化守护者"（Host-Culture Guardian），其余 Agent 标记为"跨文化审视者"（Cross-Cultural Auditors）。

匹配规则：基于 config 中每个 Agent 的 `region_keywords` 列表进行模糊匹配。例如 target_country="Vietnam" 匹配到 East Asian Culture Agent 的 keyword "vietnam"。

#### 2.3.2 话语权不对称设计

| 维度 | Host-Culture Guardian | Cross-Cultural Auditors |
|------|----------------------|------------------------|
| 生成顺序 | Phase 1（优先生成） | Phase 2（看到 Guardian 后生成） |
| 采样温度 | 0.5（低温精确） | 0.9（高温多样） |
| System Prompt | 权威确认/纠偏 | 对比分析/承认不确定性 |
| Judge 权重 | 高权重 + 一票否决权 | 辅助参考 |
| 推理角色 | "我确认选项 X 在目标文化中正确，因为..." | "从我的文化视角看，可能是 Y，但对目标文化不确定..." |

#### 2.3.3 两阶段结构化协商（Structured Negotiation）

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

#### 2.3.4 Guardian 一票否决权（Veto Power）机制

在 Judge 裁决和 fallback 投票中：
- 如果 Guardian 的答案与多数不同，但 Guardian 提供了具体文化证据 → 采信 Guardian
- 如果 Guardian 的答案与多数相同 → 直接确认
- 如果 Guardian 未能给出有效答案 → 退化为标准多数投票

### 2.4 推理路径的蒸馏价值

传统 RECONCILE 生成的 CoT 数据是"各 Agent 各自站队"的扁平推理。HFA-C²N 生成的推理数据具有结构化对比信息，蒸馏价值显著更高：

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

**蒸馏收益**：单体模型通过 SFT 学习这种结构化对比数据后，不仅学会"中国过年选1"，还能学会"为什么不能选4"——即学到了文化逻辑的边界感（Cultural Boundary Awareness）。

### 2.5 输出数据格式

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

与原 RECONCILE 格式的区别：
- Solution 标题包含 `[GUARDIAN]`/`[AUDITOR]`/`[JUDGE]` 角色标签
- 额外输出 `guardian_idx` 和 `guardian_name` 字段，便于下游蒸馏管线使用
- Guardian 的推理路径包含权威确认语言，Auditor 包含对比/不确定性表达

### 2.6 与蒸馏管线的衔接

HFA-C²N 生成的数据直接服务于后续三阶段蒸馏：

**Stage 1（主场权威加权 SFT）**：利用 `[GUARDIAN]`/`[AUDITOR]` 角色标签，对 Guardian Token 加权、对 Auditor 早期混淆 Token 掩码。数据中的角色标签是 Token 级加权的直接依据。

**Stage 2（开卷式步骤标注）**：对 Guardian 和 Auditor 的推理路径分别进行步骤切分和打标。Guardian 路径预期获得更多 0.9（确权步）标签，Auditor 路径中可能包含更多 0.1（混淆步）标签。

**Stage 3（GRPO）**：不直接依赖 MAS 数据内容——GRPO 使用 prompt 池在线生成。但 PRM 的训练数据来源于 Stage 2 对 HFA-C²N 数据的标注。

### 2.7 LLM 调用量估算

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

### 2.8 运行命令

```bash
# 全量生成（NormAD 数据集，Qwen）
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
| `--negotiation_rounds` | 1 | 协商轮次。0=独立生成（Auditor 不看 Guardian），1=标准协商 |
| `--include_judge` | true | 是否包含 Judge 裁决。false 时仅输出 Solution 1-5 |
| `--model_name` | — | `llama`/`qwen`/完整路径 |
| `--max_samples` | 0 | 0=全量 |

### 2.9 代码结构

```
Cul/
├── configs/
│   ├── reconcile_config.yaml        # 原 RECONCILE 配置（保留）
│   └── hfa_c2n_config.yaml          # HFA-C²N 配置
│                                     #   - 每个 Agent 含 guardian_prompt + auditor_prompt
│                                     #   - region_keywords 用于主场匹配
│                                     #   - Judge system prompt 含权威权重说明
├── hfa_c2n_mas.py                   # HFA-C²N 核心推理引擎
│                                     #   - HFA_C2N_MAS 类
│                                     #   - detect_guardian(): 主场识别
│                                     #   - 两阶段 batch inference
│                                     #   - Guardian veto fallback
├── generate_hfa_c2n_data.py         # HFA-C²N 数据生成入口
│                                     #   - 参数：--negotiation_rounds, --include_judge
│                                     #   - 兼容现有 convert_sample / resume 逻辑
├── reconcile_mas.py                 # 原 RECONCILE 引擎（保留作为 baseline）
└── generate_culture_data.py         # 原 RECONCILE 入口（保留作为 baseline）
```

### 2.10 各 Agent 完整 Prompt 记录

本节记录 HFA-C²N 系统中各角色的完整 System Prompt 和 Per-Round User Prompt，供后续优化和蒸馏参考。

#### 2.10.1 Guardian System Prompt

所有 5 个文化 Agent 共享同一模板，仅文化区域名称（如 "Asian cultures"、"European cultures"）不同：

```
You are the HOST-CULTURE GUARDIAN for this question.
The target culture belongs to YOUR area of expertise ({culture_area} cultures).
Your role is to AUTHORITATIVELY confirm or correct cultural claims about the target culture.
You have PRIMARY AUTHORITY on this topic. Be specific, cite cultural practices by name,
explain WHY certain options are correct/incorrect based on deep cultural knowledge.
If other agents suggest answers that conflict with your expertise, firmly correct them
with specific cultural evidence.
Format: Reasoning: <your authoritative cultural analysis>\nAnswer: <number>
```

中文翻译：

```
你是本题的【主场文化守护者】。
目标文化属于你的专业领域（{culture_area}文化）。
你的职责是以权威身份确认或纠正关于目标文化的文化主张。
你在此话题上拥有【首要权威】。请具体说明，引用具体的文化习俗名称，
解释为什么某些选项基于深层文化知识是正确/错误的。
如果其他智能体提出与你专业知识相冲突的答案，请用具体的文化证据坚定地纠正他们。
格式：Reasoning: <你的权威文化分析>\nAnswer: <数字>
```

其中 `{culture_area}` 取值为：Asian / European / North American / Latin American / African。

#### 2.10.2 Auditor System Prompt

同样 5 个 Agent 共享模板，仅文化背景名不同：

```
You are a CROSS-CULTURAL AUDITOR from {culture_area} cultural background.
The target culture does NOT belong to your primary expertise area.
Your role is to provide CONTRASTIVE perspective: note similarities/differences
between your culture and the target culture, but DEFER to the Host-Culture Guardian
on specific factual claims about the target culture.
Explicitly acknowledge uncertainty where appropriate.
Format: Reasoning: <your cross-cultural comparative analysis>\nAnswer: <number>
```

中文翻译：

```
你是一名来自{culture_area}文化背景的【跨文化审计员】。
目标文化不属于你的主要专业领域。
你的职责是提供【对比性视角】：指出你的文化与目标文化之间的相似性/差异性，
但在关于目标文化的具体事实主张上，应【参考】主场文化守护者的意见。
在适当的地方明确承认不确定性。
格式：Reasoning: <你的跨文化对比分析>\nAnswer: <数字>
```

#### 2.10.3 Judge System Prompt

```
You are a neutral cultural fact-checker and final arbitrator.
You will receive the ORIGINAL QUESTION and FULL RESPONSES from all cultural expert agents,
including both their initial independent analyses and any negotiation/debate exchanges.
ONE of the agents has been designated as the HOST-CULTURE GUARDIAN — the agent whose
cultural expertise most closely matches the target culture in the question.

Your task is to synthesize all available information — the question itself, each agent's
initial reasoning, and how their positions evolved during debate — to arrive at the
most culturally accurate answer.

When evaluating:
1. Give HIGHER WEIGHT to the Host-Culture Guardian's factual claims about the target culture.
2. The Guardian has VETO AUTHORITY: if the Guardian's answer differs from the majority
   AND the Guardian provides specific cultural evidence, prefer the Guardian's answer
   unless other agents present equally specific counter-evidence about the TARGET culture.
3. Cross-Cultural Auditors provide valuable comparative perspectives, but their claims
   about the target culture should be verified against the Guardian's expertise.
4. Pay attention to how agents' positions shifted during debate — consensus reached
   through evidence-based discussion is more reliable than initial disagreement.
5. Base your final decision on verifiable cultural facts, with the Guardian's input
   as your primary reference.
```

中文翻译：

```
你是一名中立的文化事实核查员和最终仲裁者。
你将收到【原始问题】以及所有文化专家智能体的【完整回答】，
包括他们的初始独立分析和协商/辩论环节的回答。
其中一位智能体已被指定为【主场文化守护者】——即文化专业能力与问题中目标文化最匹配的智能体。

你的任务是综合所有可用信息——问题本身、每个智能体的初始推理、
以及他们在辩论中立场的演变——得出最具文化准确性的答案。

评估时：
1. 对主场文化守护者关于目标文化的事实性主张给予【更高权重】。
2. 守护者拥有【一票否决权】：如果守护者的答案与多数不同，
   且守护者提供了具体的文化证据，则优先采信守护者的答案，
   除非其他智能体提出了关于目标文化的同等具体的反驳证据。
3. 跨文化审计员提供有价值的对比视角，但他们关于目标文化的主张
   应与守护者的专业知识进行验证。
4. 关注智能体在辩论中立场的变化——通过基于证据的讨论达成的共识
   比初始分歧更可靠。
5. 基于可验证的文化事实做出最终决定，以守护者的意见作为首要参考。
```

#### 2.10.4 Guardian Per-Round User Prompt（Phase 1）

Guardian 在第一阶段独立生成时接收的用户消息：

```
TARGET CULTURE: {target_country}

{question}

As the Host-Culture Guardian for {target_country}, provide your AUTHORITATIVE analysis.
Cite specific cultural practices, traditions, or norms by name. Explain why certain
options align or conflict with the target culture's values.

Reasoning: <your authoritative cultural analysis>
Answer: <number>
```

中文翻译：

```
目标文化：{target_country}

{question}

作为{target_country}的主场文化守护者，请提供你的【权威分析】。
引用具体的文化习俗、传统或规范名称。解释为什么某些选项与目标文化的
价值观一致或冲突。

Reasoning: <你的权威文化分析>
Answer: <数字>
```

#### 2.10.5 Auditor Per-Round User Prompt

**（a）有协商模式（negotiation_rounds=1）：Auditor 看到 Guardian 回答后生成**

```
TARGET CULTURE: {target_country}

{question}

The HOST-CULTURE GUARDIAN [{guardian_name}] has provided their authoritative analysis:
---
{guardian_response}
---

As a Cross-Cultural Auditor from [{agent_name}] background:
1. Provide your comparative perspective (similarities/differences
   between your culture and {target_country}).
2. If you agree with the Guardian, explain WHY from your cultural lens.
3. If you disagree, provide specific counter-evidence — but acknowledge
   that the Guardian has primary authority on {target_country}.

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
3. 如果你不同意，请提供具体的反驳证据——但要承认
   守护者在{target_country}问题上拥有首要权威。

Reasoning: <你的跨文化对比分析>
Answer: <数字>
```

**（b）独立模式（negotiation_rounds=0）：Auditor 不看 Guardian 回答**

```
TARGET CULTURE: {target_country}

{question}

As a Cross-Cultural Auditor from [{agent_name}] background, provide your
comparative perspective on this question about {target_country}. Note
similarities and differences with your own cultural framework, and
acknowledge uncertainty where the target culture differs from your expertise.

Reasoning: <your cross-cultural comparative analysis>
Answer: <number>
```

中文翻译：

```
目标文化：{target_country}

{question}

作为来自 [{agent_name}] 背景的跨文化审计员，请提供你对关于{target_country}
这个问题的对比视角。指出与你自身文化框架的相似性和差异性，
并在目标文化与你的专业领域不同时承认不确定性。

Reasoning: <你的跨文化对比分析>
Answer: <数字>
```

#### 2.10.6 Judge Per-Round User Prompt

```
TARGET CULTURE: {target_country}

{question}

The HOST-CULTURE GUARDIAN is [{guardian_name}] — their cultural expertise
most closely matches {target_country}.

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

Reasoning: <your reasoning, explicitly referencing the Guardian's claims>
Answer: <number>
```

中文翻译：

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

Reasoning: <你的推理，需明确引用守护者的主张>
Answer: <数字>
```

#### 2.10.7 采样温度配置

| 角色 | Temperature | 设计意图 |
|------|-------------|---------|
| Guardian | 0.5 | 低温确保权威回答精确、一致 |
| Auditor | 0.9 | 高温提供多样的跨文化对比视角 |
| Judge | 0.3 | 极低温确保裁决稳定性 |

### 2.11 Prompt 优化方案：解决 "Neutral 盲区" 问题

#### 2.11.1 问题诊断

基于 NormAD 2633 条数据的准确率分析，HFA-C²N 系统暴露出严重的"Neutral 盲区"：

| Ground Truth | 样本数 | Judge 准确率 | 问题 |
|------|--------|------------|------|
| 1（acceptable）| 958 | ~98% | 正常 |
| 2（unacceptable）| 946 | ~98% | 正常 |
| 3（neutral/indeterminate）| 729 | **~1.4%** | 灾难性失败 |

**根因分析**：

当前 Prompt 体系存在三个系统性缺陷导致模型几乎永远不会输出 "3"：

1. **极性引导偏差（Polarity Bias）**：Guardian Prompt 要求 "AUTHORITATIVELY confirm or correct"，强烈暗示必须给出确定性判断，将"不确定/中立"视为无能表现。模型被激励去选择极性选项（1 或 2）。

2. **二元框架效应（Binary Framing）**：所有 Prompt 中仅提供"align/conflict"、"correct/incorrect"、"agree/disagree"等二元表述，没有给"中立/无法确定/文化间不通用"留下语义空间。

3. **Judge 的确认偏差（Confirmation Bias）**：Judge Prompt 要求 "Base your final decision on verifiable cultural facts"，当文化事实模糊或行为跨文化时，Judge 被迫在不足证据下做出二元裁决，而非承认 "this behavior has no clear cultural valence in the target culture"。

#### 2.11.2 优化策略

**策略 1：在 Guardian System Prompt 中显式引入"中立"选项的合法性**

在 Guardian System Prompt 末尾添加：

```diff
+ IMPORTANT: Not all behaviors have clear cultural acceptance or rejection in every culture.
+ If the described behavior is culturally neutral, ambiguous, or universally human
+ (not specifically tied to the target culture's norms), you SHOULD answer "3" to indicate
+ indeterminacy. Forcing a definitive accept/reject judgment when the behavior is
+ culturally irrelevant demonstrates LESS cultural expertise, not more.
```

**策略 2：在 Guardian User Prompt 中提供选项 3 的语义锚**

在 Guardian Per-Round Prompt 的格式说明前添加：

```diff
+ Answer choices:
+ 1 = The behavior is clearly ACCEPTABLE in {target_country}'s culture
+ 2 = The behavior is clearly UNACCEPTABLE in {target_country}'s culture
+ 3 = The behavior is NEUTRAL/INDETERMINATE — it has no specific cultural valence
+     in {target_country}, or cultural norms on this topic vary significantly
+     within the country, or the behavior is universally human rather than
+     culturally specific
+
  Reasoning: <your authoritative cultural analysis>
  Answer: <number>
```

**策略 3：在 Auditor Prompt 中鼓励 "不确定性" 表达**

在 Auditor System Prompt 中强化：

```diff
+ If a behavior appears universally human or culturally neutral (not specific to any
+ culture), explicitly state this and consider answering "3" (neutral/indeterminate).
+ Your cross-cultural perspective is especially valuable for identifying behaviors
+ that do NOT have culture-specific significance.
```

**策略 4：重构 Judge Prompt，加入三分类决策框架**

在 Judge System Prompt 中添加决策逻辑：

```diff
+ 5. CRITICAL — Three-way classification:
+    - Answer "1" ONLY if there is specific evidence the behavior IS culturally acceptable
+    - Answer "2" ONLY if there is specific evidence the behavior IS culturally unacceptable
+    - Answer "3" if: (a) the behavior is not culturally specific (universal human behavior),
+      OR (b) cultural norms on this vary within the target country,
+      OR (c) agents provide conflicting claims without decisive evidence,
+      OR (d) the behavior simply has no cultural valence in the target context
+    - When in doubt between a forced judgment and "neutral", prefer "3" —
+      a calibrated "I'm not sure" is more valuable than a confident wrong answer.
```

**策略 5：在 Judge User Prompt 中引入 Calibration Reminder**

在 Judge 用户消息的 "Determine the correct answer" 部分之后添加：

```diff
+ CALIBRATION REMINDER: Approximately 28% of questions in this dataset have
+ "neutral/indeterminate (3)" as the correct answer. If you find yourself
+ never outputting "3", you are likely over-committing to binary judgments.
+ Cultural expertise includes knowing when a behavior has NO specific
+ cultural significance in the target culture.
```
---

## 3. Stage 1：主场权威加权 SFT

### 3.1 设计动机

HFA-C²N 生成的多智能体对话数据中，包含了 Guardian（主场守护者）和 Auditor（客场审视者）两种角色的完整推理轨迹。Auditor 在辩论早期可能输出带有文化混淆、偏见或引导错误的内容。如果使用传统 SFT（对所有 Token 平等计算交叉熵），单体模型会在自回归预测中拟合这些"毒草 Token"，在内部种下文化混淆的种子。

### 3.2 核心策略：Token 级加权与掩码

**原则**：

- Guardian 的确权和纠偏 Token → 保留，loss 权重乘以 α（放大学习信号）
- Auditor 最终轮之前的对抗性输出（质疑、混淆、偏离目标文化的内容）→ labels 填充 -100（完全掩码，不参与梯度计算）
- Auditor 最终轮中被 Guardian 说服后的正确表态 → 保留，loss 权重 = 1.0（不放大，但允许学习"认知转换模式"）

**实现方式**：

由于 HFA-C²N 数据中已带有 `[GUARDIAN]` 和 `[AUDITOR]` 角色标签，在构建 PyTorch DataLoader 时：

```python
# 伪代码：构造 Token 级 loss mask 和 weight
def build_token_weights(input_ids, role_spans, alpha=2.0):
    """
    role_spans: list of (start_idx, end_idx, role, round_idx, is_final_round)
    """
    loss_mask = torch.ones_like(input_ids, dtype=torch.float)
    loss_weight = torch.ones_like(input_ids, dtype=torch.float)

    for start, end, role, round_idx, is_final in role_spans:
        if role == "GUARDIAN":
            # Guardian 所有 Token 保留，权重放大
            loss_weight[start:end] = alpha
        elif role == "AUDITOR":
            if not is_final:
                # Auditor 非最终轮：掩码（不学习对抗性混淆内容）
                loss_mask[start:end] = 0  # 等价于 labels=-100
            else:
                # Auditor 最终轮（被说服后的表态）：保留，权重=1.0
                loss_weight[start:end] = 1.0

    return loss_mask, loss_weight


# 在 loss 计算中
ce_loss = F.cross_entropy(logits, labels, reduction='none')  # (seq_len,)
weighted_loss = (ce_loss * loss_mask * loss_weight).sum() / loss_mask.sum()
```

### 3.3 超参数

| 参数 | 值 | 说明 |
|------|----|------|
| alpha (Guardian 权重) | 2.0 | Guardian Token 的 loss 放大系数 |
| Auditor 掩码范围 | 非最终轮全部 Token | 最终轮表态保留 |
| 学习率 | 2e-5 | 全参微调 |
| Epochs | 3 | 早停（val_acc 2 epoch 不提升） |

### 3.4 训练数据构造

数据来源：HFA-C²N 生成的完整多智能体对话（含 Guardian 确权 + Auditor 审视 + 多轮协商）。

**输入格式**：
```
[{country}]\n{question}
```

**输出格式**（完整对话，含角色标签）：
```
[GUARDIAN] Reasoning: 作为东亚文化守护者，我确认选项1在中国春节... Answer: 1
[AUDITOR-1] Reasoning: 从欧洲视角看... 我同意 Guardian 的判断。Answer: 1
[AUDITOR-2] Reasoning: 从北美视角看... Answer: 1
...
```

**数据规模估算**：
```
HFA-C²N 数据（如 NormAD 2633 条）中的 GRPO 训练集部分（约 791 条）
每条样本包含完整的 Guardian + Auditors 对话
→ 约 791 条 SFT 样本（每条含多个角色的完整推理轨迹）
```

### 3.5 设计收益

单体模型通过此阶段学到：
- **Guardian 的确权模式**："作为 X 文化的权威，我确认..."（高权重学习）
- **认知转换模式**："从 Y 文化视角看，可能是 Z，但结合目标文化，我同意..."（正常权重学习）
- **不学习**：Auditor 早期的文化混淆和引导错误内容（完全掩码）

---

## 4. Stage 2：开卷式步骤标注（Open-Book Step Labeling）

### 4.1 设计动机

传统 PRM 标注面临两个困境：
1. **闭卷式标注（无参考答案）**：要求标注模型在没有 Ground Truth 的情况下判断中间步骤的正确性，导致 self-evaluation bias（自信心膨胀，对自己的错误步骤也打高分）
2. **连续分数标注**：0.1-0.9 的连续值缺乏明确语义锚点，不同标注实例间一致性差

CAMA-D 提出"开卷式"标注：将 Ground Truth 答案作为外部先验输入给审计器，将标注任务从"开放式推理质量评判"降维为"局部语义关联匹配"——审计器只需判断当前步骤是"支持了正确选项"还是"指向了混淆项"。

### 4.2 步骤切分策略：启发式规则

**为什么不让审计器同时完成"切步+打标"**：8B 模型在长文本中同时做两件高度抽象的任务（逻辑切分 + 打分），输出 JSON 容易格式崩溃或打标尺度变形，增加不必要的工程调试成本。

**解耦策略**：先用确定性规则切分，再让审计器只做最简单的封闭式打标。

**切分规则（Python 脚本实现）**：

```python
import re

def split_reasoning_into_steps(reasoning_text: str, max_sentences_per_step: int = 3) -> list[str]:
    """
    启发式规则切分推理文本为语义单元。

    规则：
    1. 首选以换行符 \\n\\n 或 \\n 作为主切分点
    2. 若某段落过长（超过 max_sentences_per_step 句），
       检测强逻辑转折词，在转折词前进行二次切分
    3. 切分后打上显式前缀 [Step 1], [Step 2], ...
    """
    # 转折/因果词列表（跨语言可扩展）
    TRANSITION_MARKERS = [
        r'\bHowever,?\b', r'\bBut,?\b', r'\bTherefore,?\b',
        r'\bOn the contrary,?\b', r'\bNevertheless,?\b',
        r'\bIn contrast,?\b', r'\bConsequently,?\b',
        r'\bThus,?\b', r'\bMeanwhile,?\b', r'\bInstead,?\b',
    ]
    transition_pattern = '|'.join(TRANSITION_MARKERS)

    # Step 1: 按换行符主切分
    raw_segments = re.split(r'\n\n|\n', reasoning_text.strip())
    raw_segments = [s.strip() for s in raw_segments if s.strip()]

    # Step 2: 对过长段落进行二次切分
    steps = []
    for segment in raw_segments:
        sentences = re.split(r'(?<=[.!?])\s+', segment)
        if len(sentences) > max_sentences_per_step:
            # 检测转折词位置，进行二次切分
            current_chunk = []
            for sent in sentences:
                if re.search(transition_pattern, sent, re.IGNORECASE) and current_chunk:
                    steps.append(' '.join(current_chunk))
                    current_chunk = [sent]
                else:
                    current_chunk.append(sent)
            if current_chunk:
                steps.append(' '.join(current_chunk))
        else:
            steps.append(segment)

    # Step 3: 打上显式前缀
    labeled_steps = [f"[Step {i+1}] {step}" for i, step in enumerate(steps)]
    return labeled_steps
```

**切分示例**：

输入（Guardian 推理）：
```
In Vietnamese culture, the Lunar New Year (Tet Nguyen Dan) is the most
important holiday. A central tradition is the giving of 'li xi' (red
envelopes with money) from elders to children.

However, educational materials are more associated with the mid-autumn
festival. Traditional foods are important but as shared meals, not
individual gifts from grandparents specifically.

Therefore, monetary gifts (option 1) represent the most culturally
accurate answer for Vietnam's Lunar New Year grandparent-to-grandchild
gift-giving tradition.
```

输出：
```
[Step 1] In Vietnamese culture, the Lunar New Year (Tet Nguyen Dan) is the most important holiday. A central tradition is the giving of 'li xi' (red envelopes with money) from elders to children.
[Step 2] However, educational materials are more associated with the mid-autumn festival. Traditional foods are important but as shared meals, not individual gifts from grandparents specifically.
[Step 3] Therefore, monetary gifts (option 1) represent the most culturally accurate answer for Vietnam's Lunar New Year grandparent-to-grandchild gift-giving tradition.
```

### 4.3 审计器标注：封闭式三选一打标

**审计器模型**：与 MAS 数据生成同规模的模型（Qwen2.5-7B-Instruct 或 Llama-3.1-8B-Instruct）。

**为什么同规模模型足够**：
- 任务本质已从"开放式推理评判"降维为"局部语义关联匹配"
- 审计器接收了 Ground Truth 作为确定性先验，只需判断当前步骤是否支持正确答案
- 这是一个封闭式三选一分类任务，对模型能力要求远低于开放式生成

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
- 0.9: This step provides culturally specific evidence that directly supports
       the correct answer (e.g., cites specific customs, traditions, values
       unique to the target culture). The model strongly endorses this step.
- 0.5: This step is neutral — it provides generic reasoning, format
       transitions, or universal logic that neither supports nor contradicts
       the correct answer in a culturally meaningful way. Neither reward nor penalty.
- 0.1: This step introduces cultural confusion — it points toward a wrong
       option, applies values from a different culture, or contains
       misconceptions about the target culture. The model strongly rejects this step.

Respond with ONLY one of: 0.9, 0.5, 0.1
```

**标签语义**：

| 标签 | 语义 | PRM 目标 | 示例 |
|------|------|---------|------|
| 0.9（主场确权步） | 提供了目标文化的具体证据，直接支持正确答案 | Sigmoid → 0.9 | “在越南，‘li xi’（红包）是长辈给晚辈的传统...” |
| 0.5（中立讨论步） | 格式转换、通用逻辑过渡、同义词复述 | Sigmoid → 0.5 | "Let me analyze the options one by one..." |
| 0.1（文化混淆步） | 引入文化混淆，指向错误选项或应用了错误文化的价值观 | Sigmoid → 0.1 | “在西方文化中，贺卡是最常见的节日礼物，所以选3...” |

**为什么使用全正值标签 {0.1, 0.5, 0.9} 而非 {-0.5, 0.0, +1.0}**：

在大模型对齐的工业实践中，Reward Model 的最后一层通常使用 Sigmoid 激活函数，其输出区间严格锁定在 (0, 1)。将标签设计为全正值并落在 [0, 1] 区间内，可以：

1. **数值稳定性**：Reward 在 0~1 之间，后续计算 KL 散度惩罚或 GRPO 总奖励复合时，不会因量纲过大导致梯度爆炸
2. **量纲统一**：Mean(R_process) 的值域也严格落在 [0.1, 0.9]，与 R_outcome ∈ {0, 1} 完美对齐
3. **与 Sigmoid 激活函数天然匹配**：PRM 的输出直接就是标签空间的值，无需额外映射或 clip

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

### 4.5 输出数据格式

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

## 5. Stage 3：Culture-Aware PRM 训练

### 5.1 PRM 架构

**基座模型**：Stage 1 SFT 训练完成的 student model（非独立小模型）。

**设计思路**：使用 SFT 后的模型作为 PRM 基座，因为该模型已经通过 Stage 1 学习了文化推理的语义表示，对文化相关 Token 具有更好的隐层表征。在此基础上添加线性回归头，以最小参数增量获得步骤级打分能力。

**架构**：

```python
class CulturePRM(nn.Module):
    def __init__(self, sft_model):
        super().__init__()
        self.backbone = sft_model  # Stage 1 SFT 后的模型
        hidden_size = sft_model.config.hidden_size
        self.score_head = nn.Linear(hidden_size, 1)  # 线性层
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活：输出严格锁定在 (0, 1)

    def forward(self, input_ids, attention_mask, step_end_positions):
        """
        step_end_positions: (batch, max_steps) — 每个 step 终止符的位置索引
        输出：每个 step 位置的预测分数，经 Sigmoid 后 ∈ (0, 1)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden)

        # 在每个 step 终止符位置提取 hidden state
        step_scores = []
        for b in range(hidden_states.size(0)):
            for pos in step_end_positions[b]:
                if pos == -1:  # padding
                    break
                h = hidden_states[b, pos, :]  # (hidden,)
                logit = self.score_head(h).squeeze(-1)  # raw logit
                score = self.sigmoid(logit)  # → (0, 1)
                step_scores.append(score)

        return torch.stack(step_scores)  # (total_steps,) ∈ (0, 1)
```

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

```python
def class_weighted_mse_loss(pred_scores, true_labels, loss_mask):
    """
    pred_scores: (N,) — PRM 预测的步骤分数（经 Sigmoid，∈ (0,1)）
    true_labels: (N,) — 真实标签 ∈ {0.1, 0.5, 0.9}
    loss_mask:   (N,) — 有效步骤掩码（padding 位置为 0）
    """
    # 类别关联权重映射
    # 主场确权步 W(0.9) = 2.5，文化混淆步 W(0.1) = 2.0，中立讨论步 W(0.5) = 1.0
    weights = torch.where(
        true_labels > 0.7, torch.tensor(2.5),   # 0.9 → W=2.5
        torch.where(
            true_labels < 0.3, torch.tensor(2.0),  # 0.1 → W=2.0
            torch.tensor(1.0)                       # 0.5 → W=1.0
        )
    )

    mse = (pred_scores - true_labels) ** 2  # (N,)
    weighted_mse = mse * weights * loss_mask

    return weighted_mse.sum() / loss_mask.sum()
```

**训练直觉**：MSE Loss 逼迫 Sigmoid 的输出向标签靠拢——看到"主场确权步"，就逼迫 Sigmoid 输出向 0.9 靠拢；看到"文化混淆步"，就逼迫 Sigmoid 输出向 0.1 靠拢。类别加权确保模型不会偷懒地对所有步骤输出 0.5。

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

---

## 6. Stage 3（续）：GRPO 强化学习

### 6.1 Reward 设计：加权平均形式

**旧方案的问题**：`R_total = R_outcome + beta * R_process`（加法形式）在长文本推理中会遭遇"路径长度惩罚/红利"不均的问题——通过堆砌大量平庸中立步（每步 PRM 给 ~0.2 分），累加效应会使长路径获得比精简纠偏路径更高的分数，破坏 GRPO 的优化方向。

**新方案**：

```
R_total = alpha * R_outcome + (1 - alpha) * Mean(R_process)
```

其中：
- `R_outcome ∈ {0, 1}`：答案正确性（规则可验证，答错为 0，答对为 1）
- `Mean(R_process) ∈ [0.1, 0.9]`：当前推理链中所有步骤的 PRM 得分（经 Sigmoid）的算术平均值。中间全走偏为 ~0.1，全中立为 ~0.5，完美主场确权为 ~0.9
- `alpha = 0.6`：结果奖励占主导

**量纲完美统一**：由于 PRM 使用 Sigmoid 激活且标签为全正值 {0.1, 0.5, 0.9}，Mean(R_process) 的值域严格锁定在 [0, 1]，与 R_outcome ∈ {0, 1} 完美对齐。两者量纲完全统一，R_total 的计算无比丝滑。

**具体数值示例**：
- 模型答对 + 推理全是文化混淆步：`R_total = 0.6 × 1 + 0.4 × 0.1 = 0.64`
- 模型答对 + 推理展现完美主场确权：`R_total = 0.6 × 1 + 0.4 × 0.9 = 0.96`
- 模型答错 + 推理全是确权步：`R_total = 0.6 × 0 + 0.4 × 0.9 = 0.36`
- 模型答错 + 推理全是混淆步：`R_total = 0.6 × 0 + 0.4 × 0.1 = 0.04`

在 GRPO 组内进行相对比较时，0.96 相比于 0.64 具有绝对优势，单体模型就会疯狂向"完美主场确权"的方向收敛。

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

---

## 7. 三种训练模式

### 7.1 模式 1：SFT-Only

```
Base Model → Stage 1 SFT（主场权威加权）→ 输出 SFT Model
```

仅做 Token 级加权 SFT，不做 RL。作为 baseline 验证 SFT 单独的价值。

### 7.2 模式 2：RL-Only

```
Base Model → Stage 3 GRPO（PRM 引导）→ 输出 RL Model
```

从 base 直接做 GRPO。PRM 仍需 Stage 2 标注数据训练，但 student 不经过 SFT。验证 RL 在无 SFT 初始化时的下限。

### 7.3 模式 3：SFT + RL（推荐）

```
Base Model → Stage 1 SFT → Stage 3 GRPO → 输出 SFT+RL Model
```

- SFT 让模型学会输出格式、Guardian 确权模式、认知转换模式
- GRPO 在此基础上进一步优化推理路径的文化质量
- 理论预期：SFT+RL >= RL-only >= SFT-only

### 7.4 PRM 训练流程（三种模式共用）

无论哪种 student 训练模式，Stage 2 标注 + PRM 训练（Stage 3 前半部分）都是必须的前置步骤：

```
HFA-C²N 数据 → 启发式切分 → 审计器开卷式打标 → PRM 训练（类别加权 MSE）
```

---

## 8. 完整 Pipeline

```
Phase 0: HFA-C²N 多智能体数据生成
  → 带 [GUARDIAN]/[AUDITOR] 标签的结构化推理数据

Phase 1 [Stage 1]: 主场权威加权 SFT
  → Token 级 -100 掩码 + Guardian alpha 加权
  → 输出: SFT Model (用于 PRM 基座 + SFT+RL 的 actor 初始化)

Phase 2 [Stage 2]: 开卷式步骤标注
  a. 启发式规则切分推理步骤
  b. 审计器（7B/8B）在 GT 先验下对每步打标 {0.1, 0.5, 0.9}
  → 输出: step_labels.jsonl

Phase 3 [Stage 3-PRM]: PRM 训练（类别加权 MSE）
  → 基座: SFT Model + LoRA + score_head
  → 输出: 冻结的 PRM (供 GRPO 使用)

Phase 4 [Stage 3-GRPO]: GRPO 强化学习
  → R_total = 0.6 * R_outcome + 0.4 * Mean(R_process)
  → 输出: 最终 Student Model

Phase 5: 评估
  → val_accuracy, Cultural Sensitivity Score, Reasoning Coherence
```

---

## 9. 代码结构

### 9.1 目录树

```
Cul/
├── run_camad_pipeline.py           # ★ 完整 Pipeline 入口脚本（一键运行全流程）
├── generate_hfa_c2n_data.py        # Phase 0: HFA-C²N 多智能体数据生成
├── hfa_c2n_mas.py                  # HFA-C²N 多智能体系统核心实现
├── configs/
│   └── hfa_c2n_config.yaml         # HFA-C²N Agent 提示词配置
├── sft/
│   ├── train_sft_weighted.py       # ★ Stage 1: Token 级加权 SFT（CAMA-D 新）
│   └── train_sft.py                # 旧管线: 传统 SFT（baseline 对比用）
├── step_label/
│   ├── split_steps.py              # ★ Stage 2a: 启发式规则切分推理步骤
│   ├── label_steps.py              # ★ Stage 2b: 审计器开卷式打标（vLLM batch）
│   └── validate_labels.py          # ★ Stage 2c: 标注一致性校验与分布报告
├── prm/
│   ├── train_prm_mse.py            # ★ Stage 3-PRM: 类别加权 MSE 训练
│   ├── eval_prm.py                 # ★ PRM 验证（三分类准确率、Spearman）
│   ├── train_prm.py                # 旧管线: Bradley-Terry PRM（baseline）
│   ├── label_data.py               # 旧管线: 构建 pairwise 偏好对
│   └── split_dataset.py            # 数据集切分（PRM train/val/GRPO train）
├── grpo/
│   ├── train_grpo_v3.py            # ★ Stage 3-GRPO: Mean(R_process) reward
│   └── train_grpo.py               # 旧管线: 旧 GRPO（baseline 对比用）
└── data/                           # 数据存放目录
    ├── splits/                     # 数据集切分结果
    └── prm/                        # PRM 训练数据
```

标注 ★ 的文件为 CAMA-D 新管线代码，无标注的为旧管线保留的 baseline。

### 9.2 各文件功能说明

**Pipeline 入口**

| 文件 | 功能 |
|------|------|
| `run_camad_pipeline.py` | 一键运行 CAMA-D 全流程，支持 `full`、`sft_only`、`rl_only`、`sft_rl` 四种模式，自动串联 Phase 0-4 |

**Phase 0: 数据生成**

| 文件 | 功能 |
|------|------|
| `generate_hfa_c2n_data.py` | 调用 HFA-C²N 多智能体系统生成带角色标签的结构化推理数据 |
| `hfa_c2n_mas.py` | HFA-C²N 核心逻辑：Guardian/Auditor/Judge 三类智能体的 prompt 构建、vLLM batch 推理、多轮协商 |
| `configs/hfa_c2n_config.yaml` | 5 个 Guardian prompt（按文化区域）+ 5 个 Auditor prompt + Judge prompt，已集成 Neutral 优化策略 |

**Stage 1: Token 级加权 SFT**

| 文件 | 功能 |
|------|------|
| `sft/train_sft_weighted.py` | 从 HFA-C²N 数据中提取角色标签，构造 Token 级 loss_mask（Auditor 非最终轮掩码）和 loss_weight（Guardian×α 放大），LoRA 微调 student model（rank=32，仅保存 adapter）|

**Stage 2: 开卷式步骤标注**

| 文件 | 功能 |
|------|------|
| `step_label/split_steps.py` | 启发式规则切分：按段落边界主切分 → 超长段落按转折词二次切分 → 打上 `[Step N]` 前缀 |
| `step_label/label_steps.py` | 开卷式审计器标注：将 Ground Truth 作为先验输入，对每个 Step 独立打标 {0.9, 0.5, 0.1}，使用 vLLM batch 推理 |
| `step_label/validate_labels.py` | 标注质量校验：计算标签分布、10% 重复标注一致率（目标 >85%）、分布健康度检查 |

**Stage 3-PRM: Culture-Aware PRM**

| 文件 | 功能 |
|------|------|
| `prm/train_prm_mse.py` | 以 base model + SFT-LoRA 合并 为基座 + 新 PRM-LoRA + Linear score_head + Sigmoid，用类别加权 MSE 在步骤标签上训练。加权：0.9→W=2.5, 0.1→W=2.0, 0.5→W=1.0 |
| `prm/eval_prm.py` | PRM 综合评估：三分类准确率（目标>70%）、确权步召回率（>75%）、混淆步召回率（>65%）、Spearman（>0.6）|

**Stage 3-GRPO: 强化学习**

| 文件 | 功能 |
|------|------|
| `grpo/train_grpo_v3.py` | GRPO 在线采样 → 启发式切步 → PRM 逐步打分 → Mean(R_process) → R_total = 0.6×R_outcome + 0.4×Mean(R_process) → RLOO Advantage → 策略梯度更新。LoRA Policy + `disable_adapter()` Reference，无 DeepSpeed |

### 9.3 运行命令

#### 一键运行（推荐：SFT+RL 全流程）

```bash
python Cul/run_camad_pipeline.py \
    --mode sft_rl \
    --model_name qwen \
    --hfa_c2n_data /autodl-fs/data/qwen/normad_hfa_c2n_inference.jsonl \
    --output_root /autodl-fs/data/model/qwen/normad_sftrl_camad_outputs
```

```bash
python Cul/run_camad_pipeline.py \
    --mode sft_only \
    --model_name qwen \
    --hfa_c2n_data /autodl-fs/data/qwen/normad_hfa_c2n_inference.jsonl \
    --output_root /autodl-fs/data/model/qwen/normad_sft_camad_outputs
```

```bash
python Cul/run_camad_pipeline.py \
    --mode rl_only \
    --model_name qwen \
    --hfa_c2n_data /autodl-fs/data/qwen/normad_hfa_c2n_inference.jsonl \
    --output_root /autodl-fs/data/model/qwen/normad_rl_camad_outputs
```

参数说明：

| 参数 | 含义 |
|------|------|
| `--mode` | 训练模式：`full`（含数据生成）、`sft_only`、`rl_only`、`sft_rl`（推荐）|
| `--model_name` | Student 模型：`qwen`（Qwen2.5-7B）或 `llama`（Llama-3.1-8B）|
| `--hfa_c2n_data` | HFA-C²N 推理数据 JSONL（自动按 90%/10% 划分为训练集和分布内验证集）|
| `--val_file` | 可选，外部提供验证集。未指定时自动从 `--hfa_c2n_data` 中切分 10% 作为验证集 |
| `--output_root` | 输出根目录，自动创建 `data/` 和 `models/` 子目录 |
| `--num_gpus` | GPU 数量（仅用于 vLLM 推理阶段，训练阶段使用模型放置）|

#### 分步运行

**Phase 0: HFA-C²N 数据生成**
```bash
python Cul/generate_hfa_c2n_data.py \
    --input_file /autodl-fs/data/normad_merge_gen.json \
    --output_file /autodl-fs/data/qwen/normad_hfa_c2n_inference.jsonl \
    --model_name qwen \
    --use_vllm --tensor_parallel_size 2 \
    --negotiation_rounds 1 --include_judge true
```

| 参数 | 含义 |
|------|------|
| `--input_file` | 原始数据集 JSON（CulturalBench/NormAD 格式）|
| `--model_name` | 推理模型（Agent 共用同一模型）|
| `--negotiation_rounds` | 协商轮数（0=独立推理，1=标准协商）|
| `--include_judge` | 是否包含 Judge 裁决环节 |

**Phase 1: Stage 1 Token 级加权 SFT（LoRA）**
```bash
python Cul/sft/train_sft_weighted.py \
    --model_name qwen \
    --data_file /autodl-fs/data/qwen/normad_hfa_c2n_inference.jsonl \
    --output_dir /autodl-fs/data/model/qwen/normad_camad_sft \
    --alpha 2.0 \
    --epochs 3 \
    --batch_size 4 \
    --lr 2e-4 \
    --lora_r 32
```

| 参数 | 含义 |
|------|------|
| `--alpha` | Guardian Token 的 loss 权重放大系数（默认 2.0）|
| `--data_file` | HFA-C²N 推理数据（含 [GUARDIAN]/[AUDITOR] 角色标签）|
| `--lora_r` | LoRA rank（默认 32，保证文化知识充分学习）|
| `--lr` | 学习率（LoRA 默认 2e-4，高于全参微调）|

**Phase 2a: 启发式步骤切分**
```bash
python Cul/step_label/split_steps.py \
    --input_file /autodl-fs/data/qwen/normad_hfa_c2n_inference.jsonl \
    --output_file /autodl-fs/data/qwen/normad_steps_split.jsonl \
    --max_sentences_per_step 3 \
    --sources guardian
```

| 参数 | 含义 |
|------|------|
| `--max_sentences_per_step` | 每步最大句数，超过则触发二次切分（默认 3）|
| `--sources` | 使用哪些 Agent 的推理路径（默认仅 guardian）|

**Phase 2b: 开卷式审计器打标**
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
| `--model_name` | 审计器模型（与 MAS 同规模即可，7B/8B）|
| `--batch_size` | vLLM 批次大小 |
| `--validate_consistency` | 是否进行 10% 重复标注一致性校验 |

**Phase 2c: 标注验证报告**
```bash
python Cul/step_label/validate_labels.py \
    --input_file /autodl-fs/data/qwen/normad_step_labels.jsonl \
    --report
```

**Phase 3: Culture-Aware PRM 训练（LoRA）**
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
    --lora_r 16
```

| 参数 | 含义 |
|------|------|
| `--base_model_path` | 基座模型路径（Qwen2.5-7B 或 Llama-3.1-8B）|
| `--sft_adapter_path` | Stage 1 SFT LoRA adapter 路径（会 merge 到 base 中作为 PRM 基座）|
| `--lr_head` | score_head 学习率（默认 5e-5）|
| `--lr_lora` | PRM LoRA 参数学习率（默认 1e-4）|
| `--lora_r` | PRM LoRA rank（默认 16）|

**Phase 3: PRM 评估**
```bash
python Cul/prm/eval_prm.py \
    --prm_path /autodl-fs/data/model/qwen/normad_camad_prm/best \
    --sft_path /autodl-fs/data/model/qwen/normad_camad_sft/best \
    --val_file /autodl-fs/data/qwen/normad_step_labels_val.jsonl
```

**Phase 4: GRPO 强化学习（SFT+RL 模式，LoRA，无 DeepSpeed）**
```bash
python Cul/grpo/train_grpo_v3.py \
    --model_name qwen \
    --sft_adapter /autodl-fs/data/model/qwen/normad_camad_sft/best \
    --grpo_data /autodl-fs/data/qwen/normad_hfa_c2n_inference.jsonl \
    --val_data /autodl-fs/data/qwen/normad_hfa_c2n_inference.jsonl \
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
| `--sft_adapter` | SFT LoRA adapter 路径（RL-only 模式不传此参数）|
| `--prm_path` | PRM checkpoint（含 LoRA adapter + score_head.pt）|
| `--prm_backbone` | PRM 基座模型路径（原始 base model）|
| `--alpha` | R_total 中 R_outcome 的权重（默认 0.6）|
| `--n_samples` | 每 prompt 每轮采样数 G（默认 10）|
| `--max_rounds` | 最大训练轮数（SFT+RL 建议 20，RL-only 建议 30）|
| `--lr` | GRPO LoRA 学习率（SFT+RL 用 2e-5，RL-only 用 5e-5）|
| `--lora_r` | GRPO LoRA rank（默认 16）|

### 9.4 计算资源需求

**硬件要求：2×vGPU-48GB（总计 96GB 显存）**

所有训练阶段均使用 LoRA + 梯度检查点，无需 DeepSpeed。单卡 48GB 可容纳 7B/8B 模型的 LoRA 微调。

| 阶段 | GPU 需求 | 显存估算 | 预估时间 | 说明 |
|------|---------|---------|---------|------|
| Phase 0 数据生成 | 2 GPU | ~30GB (vLLM) | ~2h (2633 samples) | vLLM tensor parallel |
| Phase 1 SFT | 1 GPU | ~22GB | ~40min (3 epochs) | LoRA rank=32 + gradient ckpt |
| Phase 2 标注 | 1-2 GPU | ~30GB (vLLM) | ~30min | vLLM batch inference |
| Phase 3 PRM | 1 GPU | ~24GB | ~30min (5 epochs) | base+SFT合并 + PRM LoRA rank=16 |
| Phase 4 GRPO | 2 GPU | cuda:0 ~28GB, cuda:1 ~16GB | ~4h (20 rounds) | Policy LoRA on cuda:0, PRM on cuda:1 |

**存储估算（仅保存 LoRA adapter，不保存全量模型）：**

| 产物 | 体积 | 说明 |
|------|------|------|
| SFT LoRA adapter | ~200MB | rank=32, 7 target modules |
| PRM LoRA adapter + score_head.pt | ~80MB | rank=16, 4 target modules + Linear |
| GRPO LoRA adapter | ~80MB | rank=16, 7 target modules |
| **总计** | **~360MB** | 相比全参保存 ~28GB 减少 99% |

**GRPO 显存分布明细：**

- cuda:0（Policy）：base model bf16 ~14GB + LoRA ~80MB + optimizer states ~400MB + gradient ckpt ~8GB + generation KV cache ~5GB ≈ **~28GB**
- cuda:1（PRM）：base model bf16 ~14GB + PRM LoRA ~40MB + score_head ~1MB ≈ **~16GB**
- Reference model：与 Policy 共享同一模型，通过 `disable_adapter()` 实现，**零额外显存**

---

## 10. 消融实验设计

### 10.1 核心蒸馏方案对比

| 实验组 | 训练方式 | 预期排序 |
|--------|---------|---------|
| Base | 无训练 | 最低 |
| SFT-only (equal weight) | 传统 SFT（无 Token 加权） | 中低 |
| SFT-only (CAMA-D Stage 1) | Token 级加权 SFT | 中 |
| RL-only | GRPO from base | 中高 |
| SFT + RL (CAMA-D full) | Stage 1 → Stage 3 | 最高 |
| MAS Oracle | 多智能体系统直接推理 | 上界 |

### 10.2 模块贡献消融

| 消融项 | 对比 | 验证目标 |
|--------|------|---------|
| Token 加权 vs 样本级加权 | Stage 1 w/ vs w/o mask | 验证掩码 Auditor 混淆 Token 的价值 |
| 开卷式标注 vs 闭卷式标注 | Stage 2 w/ vs w/o GT prior | 验证 GT 先验消除 self-evaluation bias |
| 类别加权 MSE vs 均匀 MSE | PRM w/ vs w/o class weights | 验证加权对稀疏信号的保护 |
| Mean(R_process) vs Sum(R_process) | GRPO reward 形式 | 验证加权平均消除长度偏差 |
| alpha=0.6 vs alpha=0.8 vs alpha=0.4 | GRPO alpha 敏感性 | 找最优 R_outcome/R_process 平衡 |

### 10.3 评估指标

| 指标 | 说明 |
|------|------|
| val_accuracy | 预测答案与 gold label 匹配率 |
| Cultural Sensitivity Score | 同一问题不同文化下答案分布 KL 散度均值 |
| Reasoning Coherence | LLM Judge 评估推理路径与答案的一致性 |
| Cultural Grounding | 推理路径中目标文化具体价值观关键词出现率 |
| Cultural Boundary Awareness | 模型是否能正确区分相邻文化（如越南 vs 中国） |

---

## 11. 与旧管线的兼容性

CAMA-D 与现有代码（`culturedebate.md` 描述的旧管线）的关系：

- **数据生成**：复用 HFA-C²N（`generate_hfa_c2n_data.py`），无需修改
- **SFT**：需新建 `train_sft_weighted.py`，旧 `train_sft.py` 保留作为 baseline
- **PRM**：需新建 `train_prm_mse.py`，旧 `train_prm.py`（Bradley-Terry）保留作为对比
- **GRPO**：需修改 reward 计算逻辑（`train_grpo.py` → `train_grpo_v3.py`），旧版保留
- **评估**：评估指标和脚本完全复用

---

## 12. 风险与缓解

| 风险 | 缓解策略 |
|------|---------|
| 审计器标注噪声（7B 模型三选一也可能出错） | 一致性校验 > 85%；类别加权 MSE 容忍少量噪声 |
| PRM 在 GRPO 中被 reward hacking | alpha=0.6 使 R_outcome 主导；KL penalty 防漂移 |
| Token 加权 SFT 收敛不稳定 | 监控 Guardian/Auditor 分组 loss 曲线，确保 Guardian loss 下降更快 |
| 步骤切分粒度不一致影响 PRM 泛化 | 统一切分规则 + 限制每步最大长度；PRM 验证集覆盖不同长度 |
| Stage 1 SFT 作为 PRM 基座可能有偏 | 对比实验：用 base model 作 PRM 基座 vs SFT model 作基座 |

---

## 13. 待确认事项（实验启动前）

1. **HFA-C²N 数据是否已生成完毕**：Stage 1 SFT 和 Stage 2 标注都依赖 HFA-C²N 数据
2. **计算资源分配**：Stage 1 SFT (1 GPU) → Stage 2 标注 (1 GPU, batch inference) → PRM (1 GPU) → GRPO (2 GPU)
3. **优先跑哪个 student model**：Qwen2.5-7B 或 Llama-3.1-8B（建议先跑 Qwen，与 PRM 同系列）
4. **是否保留旧管线作为 baseline**：建议保留 `culturedebate.md` 方案作为 "Naive Distillation" baseline
