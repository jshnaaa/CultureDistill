"""
CAMA-D PRM Evaluation Script (PRM 验证)

Evaluates the trained PRM on validation data with comprehensive metrics:
  - Three-class accuracy (discretized predictions)
  - Per-class recall (确权步/中立步/混淆步)
  - Spearman correlation between predictions and labels
  - Score distribution analysis

Usage:
    python Cul/prm/eval_prm.py \\
        --prm_path    /path/to/camad_prm/best \\
        --sft_path    /path/to/camad_sft_qwen/best \\
        --val_file    /path/to/step_labels_val.jsonl \\
        --batch_size  8
"""

import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from train_prm_mse import CulturePRM, StepLabelDataset


def load_trained_prm(prm_path: str, sft_path: str, device) -> CulturePRM:
    """Load a trained PRM from checkpoint."""
    # First, create the model structure
    base_model = AutoModelForCausalLM.from_pretrained(
        sft_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        output_hidden_states=True,
    )

    # Load LoRA adapter
    model_with_lora = PeftModel.from_pretrained(base_model, prm_path)

    # Build CulturePRM manually
    prm = CulturePRM.__new__(CulturePRM)
    nn.Module.__init__(prm)
    prm.backbone = model_with_lora
    hidden_size = base_model.config.hidden_size
    prm.score_head = nn.Linear(hidden_size, 1)
    prm.sigmoid = nn.Sigmoid()

    # Load score_head weights
    head_path = Path(prm_path) / "score_head.pt"
    state = torch.load(head_path, map_location="cpu")
    prm.score_head.load_state_dict(state)

    prm.to(device)
    prm.eval()
    print(f"Loaded PRM from {prm_path}")
    return prm


@torch.no_grad()
def evaluate_full(model, loader, device) -> dict:
    """Comprehensive PRM evaluation."""
    from scipy.stats import spearmanr
    import numpy as np

    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        step_positions = batch["step_end_positions"].to(device)
        labels = batch["labels"]

        scores = model(input_ids, attention_mask, step_positions)

        # Collect valid labels
        for b in range(labels.size(0)):
            for s in range(labels.size(1)):
                if labels[b, s] >= 0:
                    all_labels.append(labels[b, s].item())

        all_preds.extend(scores.cpu().tolist())

    n = min(len(all_preds), len(all_labels))
    all_preds = all_preds[:n]
    all_labels = all_labels[:n]

    # Discretize
    def discretize(val):
        if val > 0.7:
            return 0.9
        elif val < 0.3:
            return 0.1
        else:
            return 0.5

    pred_d = [discretize(p) for p in all_preds]
    label_d = [discretize(l) for l in all_labels]

    # Overall accuracy
    acc = sum(p == l for p, l in zip(pred_d, label_d)) / n if n > 0 else 0

    # Per-class metrics
    classes = [0.9, 0.5, 0.1]
    class_names = {0.9: "确权步", 0.5: "中立步", 0.1: "混淆步"}
    recalls = {}
    precisions = {}
    for cls in classes:
        tp = sum(1 for p, l in zip(pred_d, label_d) if p == cls and l == cls)
        fn = sum(1 for p, l in zip(pred_d, label_d) if p != cls and l == cls)
        fp = sum(1 for p, l in zip(pred_d, label_d) if p == cls and l != cls)
        recalls[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0
        precisions[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Spearman
    try:
        spearman_r, p_val = spearmanr(all_preds, all_labels)
    except Exception:
        spearman_r, p_val = 0.0, 1.0

    # Score distribution
    pred_arr = np.array(all_preds)

    return {
        "n_steps": n,
        "accuracy": acc,
        "spearman": spearman_r,
        "spearman_p": p_val,
        "recalls": recalls,
        "precisions": precisions,
        "pred_mean": float(pred_arr.mean()),
        "pred_std": float(pred_arr.std()),
        "pred_min": float(pred_arr.min()),
        "pred_max": float(pred_arr.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="CAMA-D PRM Evaluation")
    parser.add_argument("--prm_path", type=str, required=True,
                        help="PRM checkpoint directory")
    parser.add_argument("--sft_path", type=str, required=True,
                        help="SFT model path (PRM backbone base)")
    parser.add_argument("--val_file", type=str, required=True,
                        help="Validation step-labeled JSONL")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model and data
    model = load_trained_prm(args.prm_path, args.sft_path, device)
    val_ds = StepLabelDataset(args.val_file, tokenizer)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Evaluate
    metrics = evaluate_full(model, val_loader, device)

    # Report
    print("\n" + "=" * 60)
    print("CAMA-D PRM Evaluation Report")
    print("=" * 60)
    print(f"Total steps evaluated: {metrics['n_steps']}")
    print(f"\nOverall accuracy: {metrics['accuracy']:.4f} (target: >0.70)")
    print(f"Spearman correlation: {metrics['spearman']:.4f} (target: >0.60)")
    print(f"\nPer-class metrics:")
    print(f"  {'Class':<12} {'Recall':<10} {'Precision':<10} {'Target'}")
    print(f"  {'确权步(0.9)':<12} {metrics['recalls'][0.9]:<10.4f} "
          f"{metrics['precisions'][0.9]:<10.4f} >0.75")
    print(f"  {'中立步(0.5)':<12} {metrics['recalls'][0.5]:<10.4f} "
          f"{metrics['precisions'][0.5]:<10.4f} -")
    print(f"  {'混淆步(0.1)':<12} {metrics['recalls'][0.1]:<10.4f} "
          f"{metrics['precisions'][0.1]:<10.4f} >0.65")
    print(f"\nScore distribution:")
    print(f"  Mean={metrics['pred_mean']:.4f} Std={metrics['pred_std']:.4f} "
          f"Min={metrics['pred_min']:.4f} Max={metrics['pred_max']:.4f}")


if __name__ == "__main__":
    main()
