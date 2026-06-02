"""
AgentArk Baseline for Cultural Alignment Tasks

This module implements the AgentArk distillation pipeline adapted for
NormAD and CulturalBench datasets. It serves as the baseline comparison
against CAMA-D (Culture-Aware Multi-Agent Distillation).

Pipeline stages:
  0. generate_debate_data.py  — Homogeneous multi-agent debate inference
  1. train_sft.py             — Standard SFT (uniform loss, no token weighting)
  2. train_prm.py             — PRM training (standard MSE, no class weights)
  3. train_grpo.py            — GRPO training (supports reconcile/hf_cac data)
  4. evaluate.py              — Test set evaluation (per-country accuracy)

Key differences from CAMA-D:
  - Homogeneous agents (no cultural identity / no Guardian/Auditor hierarchy)
  - Uniform SFT loss (no role-based token weighting)
  - Standard PRM (no class-weighted MSE, no cultural authority in labels)
  - Culture-agnostic prompts (no [country] prefix)

Data sources (controlled by --data_source parameter):
  - "reconcile": RECONCILE-style symmetric debate (AgentArk native)
  - "hf_cac":   HF-CAC data from CAMA-D (cross-comparison, same data different method)

Usage:
  bash ark/culture/run_pipeline.sh reconcile normad qwen
  bash ark/culture/run_pipeline.sh hf_cac culturalbench qwen
"""
