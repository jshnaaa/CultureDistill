## 7. CAMAD 的 baseline

### 7.1 MAGDi

**论文**：MAGDi: Structured Distillation of Multi-Agent Interaction Graphs Improves Reasoning in Smaller Language Models (ICML 2024, UNC Chapel Hill)

**核心思想**：MAGDi 将多个大型教师模型之间的多轮讨论交互建模为有向无环图（Multi-Agent Interaction Graph, MAG），然后通过结构化蒸馏（Next-Token Prediction + Margin Ranking + GCN Node Classification 三目标联合优化）将交互中蕴含的推理知识注入小型学生模型，使其在推理时无需多智能体协作即可获得接近多智能体系统的推理能力。

**文化对齐适配**：
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

**Pipeline 与运行命令**：代码位于 `MAGDi/` 目录，完整 pipeline 包含 4 步（RECONCILE 模式额外有 Step 0 自动生成推理数据）：

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

**实验设置**： 两个数据集（NormAd、CultureBench）和两种数据来源（通过 `--data_source reconcile|hf_cac` 参数切换）。
基座模型限定为两种：Qwen2.5-7B-Instruct（`--model_name qwen`）和 LLaMA-3.1-8B-Instruct（`--model_name llama`）。
数据以 pkl 格式存储，结构为 `{"train": [...], "val": [...], "test": [...]}`。
SFT 和 GRPO 各训练 3 epoch，学习率 2e-5，LoRA rank=64。PRM 训练 5 epoch，学习率 1e-5。GRPO 每条样本生成 4 条候选（group_size=4）。

**Pipeline 与运行命令**：位于 `ark/culture/` 目录，完整 pipeline 包含 5 步（以 CulturalBench + RECONCILE + Qwen 为例）：

```bash
cd autodl-tmp/distill
source /etc/network_turbo
sh git.sh
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
