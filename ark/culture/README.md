# AgentArk Baseline for Cultural Alignment

This directory implements the **AgentArk baseline** for cultural alignment distillation,
serving as a comparison baseline for CAMA-D (Culture-Aware Multi-Agent Distillation).

## Pipeline Overview

```
Stage 0: Multi-Agent Debate (homogeneous agents) → inference JSONL
  └── split_data.py → train/val/test pkl splits
Stage 1: Standard SFT (uniform loss, no token weighting) → SFT LoRA adapter
Stage 2: Step labeling + PRM training (standard MSE) → PRM checkpoint
Stage 3: GRPO reinforcement learning (--data_source param) → GRPO LoRA adapter
Stage 4: Evaluation on test set → per-country accuracy
```

## Key Differences from CAMA-D

| Aspect | AgentArk (this) | CAMA-D |
|--------|-----------------|--------|
| Agent type | Homogeneous (same model, same prompt) | Heterogeneous (cultural agents) |
| Authority | Symmetric (equal debate) | Asymmetric (Guardian/Auditor) |
| SFT weighting | Uniform (all tokens equal) | Token-level (Guardian α=2.0) |
| PRM loss | Standard MSE (uniform) | Class-weighted MSE (2.5x/2.0x/1.0x) |
| Prompt format | Raw question (no country) | `[country]\nquestion` |
| GRPO reward | R_outcome + R_process | Culture-aware R_total |

## Data Source Parameter

Use `--data_source` to switch between:
- `reconcile`: Uses RECONCILE-style multi-agent debate data (symmetric homogeneous agents)
- `hf_cac`: Uses HF-CAC data (from CAMA-D's heterogeneous agents, but trains WITHOUT culture-aware weighting)

Both produce AgentArk-style training (no cultural authority mechanism).
This enables ablation: same data → different training method.

## Files

| File | Description |
|------|-------------|
| `generate_debate_data.py` | Stage 0: Homogeneous LLM debate inference |
| `train_sft.py` | Stage 1: Standard SFT with uniform CE loss |
| `train_prm.py` | Stage 2: PRM training with standard MSE |
| `train_grpo.py` | Stage 3: GRPO with `--data_source` parameter |
| `evaluate.py` | Stage 4: Test set evaluation |
| `run_pipeline.sh` | Unified pipeline entry point |

## Supported Base Models

Only two base models are supported (specified via `--model_name` parameter):

| Alias | Model | Path |
|-------|-------|------|
| `qwen` | Qwen2.5-7B-Instruct | `/root/autodl-tmp/base/Qwen2.5-7B-Instruct` |
| `llama` | Meta-Llama-3.1-8B-Instruct | `/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct` |

No other models are accepted — argparse will reject any value other than `qwen` or `llama`.

## Quick Start

```bash
# Full pipeline: RECONCILE data + NormAD + Qwen
bash ark/culture/run_pipeline.sh reconcile normad qwen

# Full pipeline: HF-CAC data + CulturalBench + LLaMA
bash ark/culture/run_pipeline.sh hf_cac culturalbench llama

# Run specific stage only (e.g., only SFT)
START_STAGE=1 END_STAGE=1 bash ark/culture/run_pipeline.sh reconcile normad qwen
```

## Datasets

- **NormAD** (`normad_mas.json`): 3-way classification (acceptable/unacceptable/neutral)
- **CulturalBench** (`culturalBench_mas.json`): 4-way multiple-choice cultural QA
