"""
CAMA-D Complete Pipeline Runner (完整管线运行入口)

Orchestrates the full Culture-Aware Multi-Agent Distillation pipeline:

  Phase 0: HFA-C²N Data Generation (multi-agent inference)
  Phase 1: Stage 1 — Token-level Weighted SFT
  Phase 2: Stage 2 — Open-Book Step Labeling (split + label)
  Phase 3: Stage 3-PRM — Culture-Aware PRM Training (class-weighted MSE)
  Phase 4: Stage 3-GRPO — GRPO with Mean(R_process)
  Phase 5: Evaluation

Usage (full pipeline):
    python Cul/run_camad_pipeline.py \\
        --mode full \\
        --model_name qwen \\
        --hfa_c2n_data /path/to/hfa_c2n_inference.jsonl \\
        --output_root /path/to/camad_outputs

Usage (individual stages):
    python Cul/run_camad_pipeline.py --mode sft_only --model_name qwen ...
    python Cul/run_camad_pipeline.py --mode rl_only --model_name qwen ...
    python Cul/run_camad_pipeline.py --mode sft_rl --model_name qwen ...

Three training modes:
  - sft_only:  Base → Stage 1 SFT → output (baseline)
  - rl_only:   Base → Stage 3 GRPO → output (RL without SFT init)
  - sft_rl:    Base → Stage 1 SFT → Stage 3 GRPO → output (recommended)
  - full:      All phases including data generation
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}

# Default hyperparameters
DEFAULTS = {
    # Stage 1: Weighted SFT (LoRA)
    "sft_alpha": 2.0,
    "sft_epochs": 3,
    "sft_batch_size": 4,
    "sft_lr": 2e-4,
    "sft_lora_r": 32,
    # Stage 2: Step labeling
    "max_sentences_per_step": 3,
    "label_batch_size": 64,
    # Stage 3-PRM
    "prm_epochs": 5,
    "prm_batch_size": 8,
    "prm_lr_head": 5e-5,
    "prm_lr_lora": 1e-4,
    "prm_lora_r": 16,
    # Stage 3-GRPO (LoRA)
    "grpo_alpha": 0.6,
    "grpo_n_samples": 10,
    "grpo_max_rounds": 30,
    "grpo_eval_every": 5,
    "grpo_lr_rl_only": 5e-5,
    "grpo_lr_sft_rl": 2e-5,
    "grpo_temperature": 0.7,
    "grpo_lora_r": 16,
}


def run_cmd(cmd: list, description: str) -> int:
    """Run a subprocess command with logging."""
    print(f"\n{'='*60}")
    print(f"[CAMA-D] {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with return code {result.returncode}")
    return result.returncode


def run_phase0_data_generation(args, output_root: Path) -> str:
    """Phase 0: Generate HFA-C²N data."""
    if args.hfa_c2n_data and Path(args.hfa_c2n_data).exists():
        print(f"[Phase 0] Using existing HFA-C²N data: {args.hfa_c2n_data}")
        return args.hfa_c2n_data

    output_file = str(output_root / "data" / "hfa_c2n_inference.jsonl")
    cmd = [
        sys.executable, "Cul/generate_hfa_c2n_data.py",
        "--input_file", args.input_dataset,
        "--output_file", output_file,
        "--model_name", args.model_name,
        "--use_vllm",
        "--tensor_parallel_size", str(args.num_gpus),
        "--negotiation_rounds", "1",
        "--include_judge", "true",
    ]
    rc = run_cmd(cmd, "Phase 0: HFA-C²N Data Generation")
    if rc != 0:
        raise RuntimeError("Phase 0 failed")
    return output_file


def run_phase1_sft(args, hfa_data: str, val_file: str, output_root: Path) -> str:
    """Phase 1: Stage 1 — Token-level Weighted SFT (LoRA)."""
    sft_output = str(output_root / "models" / "camad_sft")
    cmd = [
        sys.executable, "Cul/sft/train_sft_weighted.py",
        "--model_name", args.model_name,
        "--data_file", hfa_data,
        "--val_file", val_file,
        "--output_dir", sft_output,
        "--alpha", str(DEFAULTS["sft_alpha"]),
        "--epochs", str(DEFAULTS["sft_epochs"]),
        "--batch_size", str(DEFAULTS["sft_batch_size"]),
        "--lr", str(DEFAULTS["sft_lr"]),
        "--lora_r", str(DEFAULTS["sft_lora_r"]),
    ]
    rc = run_cmd(cmd, "Phase 1: Token-level Weighted SFT — LoRA (Stage 1)")
    if rc != 0:
        raise RuntimeError("Phase 1 (SFT) failed")
    return str(Path(sft_output) / "best")


def run_phase2_step_labeling(args, hfa_data: str, output_root: Path) -> tuple:
    """Phase 2: Stage 2 — Split steps + Open-book labeling."""
    data_dir = output_root / "data"

    # 2a: Split steps
    split_output = str(data_dir / "steps_split.jsonl")
    cmd = [
        sys.executable, "Cul/step_label/split_steps.py",
        "--input_file", hfa_data,
        "--output_file", split_output,
        "--max_sentences_per_step", str(DEFAULTS["max_sentences_per_step"]),
        "--sources", "guardian",
    ]
    rc = run_cmd(cmd, "Phase 2a: Heuristic Step Splitting")
    if rc != 0:
        raise RuntimeError("Phase 2a (step splitting) failed")

    # 2b: Label steps
    label_output = str(data_dir / "step_labels.jsonl")
    cmd = [
        sys.executable, "Cul/step_label/label_steps.py",
        "--input_file", split_output,
        "--output_file", label_output,
        "--model_name", args.model_name,
        "--batch_size", str(DEFAULTS["label_batch_size"]),
        "--tensor_parallel_size", str(args.num_gpus),
        "--validate_consistency",
    ]
    rc = run_cmd(cmd, "Phase 2b: Open-Book Step Labeling")
    if rc != 0:
        raise RuntimeError("Phase 2b (step labeling) failed")

    # 2c: Validate
    cmd = [
        sys.executable, "Cul/step_label/validate_labels.py",
        "--input_file", label_output,
        "--report",
    ]
    run_cmd(cmd, "Phase 2c: Label Validation Report")

    # Split into train/val for PRM
    train_file, val_file = split_jsonl_data(
        label_output, data_dir,
        "step_labels_train.jsonl", "step_labels_val.jsonl",
        val_ratio=0.2,
    )
    return train_file, val_file


def split_jsonl_data(input_file: str, data_dir: Path,
                     train_name: str, val_name: str,
                     val_ratio: float = 0.2) -> tuple:
    """
    Split a JSONL file into train and val sets.

    Used for:
      - HFA-C²N data → SFT train / SFT+GRPO val
      - Step label data → PRM train / PRM val
    """
    import random
    random.seed(42)

    samples = [json.loads(l) for l in open(input_file, encoding="utf-8")]
    random.shuffle(samples)

    n_val = max(1, int(len(samples) * val_ratio))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    train_file = str(data_dir / train_name)
    val_file = str(data_dir / val_name)

    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"  Data split: train={len(train_samples)}, val={len(val_samples)} "
          f"({val_ratio*100:.0f}% val)")
    return train_file, val_file


def run_phase3_prm(args, sft_adapter_path: str, train_file: str,
                   val_file: str, output_root: Path) -> str:
    """
    Phase 3: Stage 3-PRM — Culture-Aware PRM Training (LoRA).

    Since SFT is LoRA, PRM needs:
      --base_model_path: the original base model
      --sft_adapter_path: SFT LoRA adapter to merge into backbone
    """
    prm_output = str(output_root / "models" / "camad_prm")
    base_model = MODEL_ALIASES.get(args.model_name, args.model_name)
    cmd = [
        sys.executable, "Cul/prm/train_prm_mse.py",
        "--base_model_path", base_model,
        "--sft_adapter_path", sft_adapter_path,
        "--train_file", train_file,
        "--val_file", val_file,
        "--output_dir", prm_output,
        "--epochs", str(DEFAULTS["prm_epochs"]),
        "--batch_size", str(DEFAULTS["prm_batch_size"]),
        "--lr_head", str(DEFAULTS["prm_lr_head"]),
        "--lr_lora", str(DEFAULTS["prm_lr_lora"]),
        "--lora_r", str(DEFAULTS["prm_lora_r"]),
    ]
    rc = run_cmd(cmd, "Phase 3: Culture-Aware PRM Training — LoRA (Stage 3-PRM)")
    if rc != 0:
        raise RuntimeError("Phase 3 (PRM) failed")
    return str(Path(prm_output) / "best")


def run_phase4_grpo(args, sft_adapter_path: str, prm_path: str,
                    grpo_data: str, val_data: str,
                    output_root: Path, mode: str) -> str:
    """
    Phase 4: Stage 3-GRPO — GRPO with Mean(R_process) (LoRA, no DeepSpeed).

    Launches via plain Python (no deepspeed). Policy uses GRPO LoRA
    on merged SFT model; reference via disable_adapter.
    """
    grpo_output = str(output_root / "models" / f"camad_grpo_{mode}")
    base_model = MODEL_ALIASES.get(args.model_name, args.model_name)

    lr = DEFAULTS["grpo_lr_sft_rl"] if mode == "sft_rl" else DEFAULTS["grpo_lr_rl_only"]
    max_rounds = 20 if mode == "sft_rl" else DEFAULTS["grpo_max_rounds"]

    cmd = [
        sys.executable, "Cul/grpo/train_grpo_v3.py",
        "--model_name", args.model_name,
        "--grpo_data", grpo_data,
        "--val_data", val_data,
        "--prm_path", prm_path,
        "--prm_backbone", base_model,
        "--output_dir", grpo_output,
        "--alpha", str(DEFAULTS["grpo_alpha"]),
        "--n_samples", str(DEFAULTS["grpo_n_samples"]),
        "--max_rounds", str(max_rounds),
        "--eval_every", str(DEFAULTS["grpo_eval_every"]),
        "--temperature", str(DEFAULTS["grpo_temperature"]),
        "--lr", str(lr),
        "--lora_r", str(DEFAULTS["grpo_lora_r"]),
    ]

    if mode == "sft_rl":
        cmd.extend(["--sft_adapter", sft_adapter_path])

    rc = run_cmd(cmd, f"Phase 4: GRPO ({mode}) — LoRA with Mean(R_process)")
    if rc != 0:
        raise RuntimeError(f"Phase 4 (GRPO {mode}) failed")
    return str(Path(grpo_output) / "best")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CAMA-D Complete Pipeline Runner"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["full", "sft_only", "rl_only", "sft_rl"],
                        help="Training mode: full | sft_only | rl_only | sft_rl")
    parser.add_argument("--model_name", type=str, required=True,
                        help="'llama', 'qwen', or full model path")
    parser.add_argument("--hfa_c2n_data", type=str, default=None,
                        help="Pre-generated HFA-C²N inference JSONL "
                             "(skip Phase 0 if provided)")
    parser.add_argument("--input_dataset", type=str, default=None,
                        help="Raw dataset for HFA-C²N generation (Phase 0)")
    parser.add_argument("--val_file", type=str, default=None,
                        help="Validation JSONL for accuracy evaluation")
    parser.add_argument("--grpo_data", type=str, default=None,
                        help="GRPO training data (if different from hfa_c2n_data)")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory for all outputs")
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="Number of GPUs for vLLM inference (default: 2). "
                             "Training uses model placement, not data parallelism.")
    args = parser.parse_args()

    # Resolve model name
    args.model_name = MODEL_ALIASES.get(args.model_name, args.model_name)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "data").mkdir(exist_ok=True)
    (output_root / "models").mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'#'*60}")
    print(f"# CAMA-D Pipeline — {args.mode} mode")
    print(f"# Started: {timestamp}")
    print(f"# Model: {args.model_name}")
    print(f"# Output: {output_root}")
    print(f"{'#'*60}")

    # Determine val_file
    val_file = args.val_file
    grpo_data = args.grpo_data

    try:
        if args.mode == "full":
            # Phase 0: Data generation
            hfa_data = run_phase0_data_generation(args, output_root)
            if not val_file:
                # Auto-split: 90% train, 10% val (from HFA-C²N data)
                hfa_data, val_file = split_jsonl_data(
                    hfa_data, output_root / "data",
                    "hfa_c2n_train.jsonl", "hfa_c2n_val.jsonl",
                    val_ratio=0.1,
                )
            if not grpo_data:
                grpo_data = hfa_data

            # Phase 1: Weighted SFT
            sft_path = run_phase1_sft(args, hfa_data, val_file, output_root)

            # Phase 2: Step labeling
            prm_train, prm_val = run_phase2_step_labeling(
                args, hfa_data, output_root
            )

            # Phase 3: PRM training
            prm_path = run_phase3_prm(
                args, sft_path, prm_train, prm_val, output_root
            )

            # Phase 4: GRPO (SFT+RL mode)
            final_model = run_phase4_grpo(
                args, sft_path, prm_path, grpo_data, val_file,
                output_root, "sft_rl"
            )

        elif args.mode == "sft_only":
            hfa_data = args.hfa_c2n_data
            if not hfa_data:
                raise ValueError("--hfa_c2n_data required for sft_only mode")
            if not val_file:
                # Auto-split: 90% train, 10% val
                hfa_data, val_file = split_jsonl_data(
                    hfa_data, output_root / "data",
                    "hfa_c2n_train.jsonl", "hfa_c2n_val.jsonl",
                    val_ratio=0.1,
                )

            sft_path = run_phase1_sft(args, hfa_data, val_file, output_root)
            final_model = sft_path

        elif args.mode == "rl_only":
            hfa_data = args.hfa_c2n_data
            if not hfa_data:
                raise ValueError("--hfa_c2n_data required for rl_only mode")
            if not val_file:
                # Auto-split: 90% train, 10% val
                hfa_data, val_file = split_jsonl_data(
                    hfa_data, output_root / "data",
                    "hfa_c2n_train.jsonl", "hfa_c2n_val.jsonl",
                    val_ratio=0.1,
                )
            if not grpo_data:
                grpo_data = hfa_data

            # Still need PRM (requires SFT model as backbone)
            # Use base model as PRM backbone in RL-only mode
            base_model = args.model_name

            # Phase 2: Step labeling (needed for PRM)
            prm_train, prm_val = run_phase2_step_labeling(
                args, hfa_data, output_root
            )

            # Phase 3: PRM (use base model as backbone)
            prm_path = run_phase3_prm(
                args, base_model, prm_train, prm_val, output_root
            )

            # Phase 4: GRPO from base
            final_model = run_phase4_grpo(
                args, base_model, prm_path, grpo_data, val_file,
                output_root, "rl_only"
            )

        elif args.mode == "sft_rl":
            hfa_data = args.hfa_c2n_data
            if not hfa_data:
                raise ValueError("--hfa_c2n_data required for sft_rl mode")
            if not val_file:
                # Auto-split: 90% train, 10% val
                hfa_data, val_file = split_jsonl_data(
                    hfa_data, output_root / "data",
                    "hfa_c2n_train.jsonl", "hfa_c2n_val.jsonl",
                    val_ratio=0.1,
                )
            if not grpo_data:
                grpo_data = hfa_data

            # Phase 1: SFT
            sft_path = run_phase1_sft(args, hfa_data, val_file, output_root)

            # Phase 2: Step labeling
            prm_train, prm_val = run_phase2_step_labeling(
                args, hfa_data, output_root
            )

            # Phase 3: PRM
            prm_path = run_phase3_prm(
                args, sft_path, prm_train, prm_val, output_root
            )

            # Phase 4: GRPO from SFT
            final_model = run_phase4_grpo(
                args, sft_path, prm_path, grpo_data, val_file,
                output_root, "sft_rl"
            )

        print(f"\n{'#'*60}")
        print(f"# CAMA-D Pipeline Complete!")
        print(f"# Final model: {final_model}")
        print(f"{'#'*60}")

    except RuntimeError as e:
        print(f"\n[FATAL] Pipeline aborted: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
