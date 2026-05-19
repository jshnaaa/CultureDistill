"""
CAMA-D Stage 3-PRM: Culture-Aware PRM Training (类别加权 MSE)

Architecture: SFT Model (Stage 1 output) + LoRA + Linear score_head + Sigmoid
Loss: Class-weighted MSE on step-level labels {0.1, 0.5, 0.9}
Input: Full reasoning path with [Step N] markers
Output: Per-step score ∈ (0, 1) via Sigmoid activation

Key differences from old PRM (Bradley-Terry):
  - Uses absolute step-level labels instead of pairwise preferences
  - Class-weighted MSE to prevent "lazy neutral" collapse
  - Based on SFT model (already has cultural representations)
  - Sigmoid output naturally aligns with [0, 1] label space

Usage:
    python Cul/prm/train_prm_mse.py \\
        --sft_model_path  /path/to/camad_sft_qwen/best \\
        --train_file      /path/to/step_labels.jsonl \\
        --val_file        /path/to/step_labels_val.jsonl \\
        --output_dir      /path/to/models/camad_prm \\
        --epochs 5 \\
        --batch_size 8 \\
        --lr_head 5e-5 \\
        --lr_lora 1e-4 \\
        --lora_r 16
"""

import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType


MAX_SEQ_LEN = 2048


# ---------------------------------------------------------------------------
# Model: CulturePRM (SFT backbone + LoRA + score_head + Sigmoid)
# ---------------------------------------------------------------------------

class CulturePRM(nn.Module):
    """
    Process Reward Model for culture-aware step scoring.

    Architecture:
      - Backbone: Stage 1 SFT model (already has cultural representations)
      - Adapter: LoRA (rank=16) to avoid destroying generation capability
      - Head: Linear(hidden_size, 1) + Sigmoid → score ∈ (0, 1)

    Scoring mechanism:
      - At each [Step N] terminator position, extract hidden state
      - Pass through score_head + Sigmoid to get step score
    """

    def __init__(self, sft_model_path: str, lora_r: int = 16,
                 lora_alpha: int = 32, lora_dropout: float = 0.05):
        super().__init__()

        # Load SFT backbone
        base_model = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            output_hidden_states=True,
        )

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        self.backbone = get_peft_model(base_model, lora_config)
        self.backbone.print_trainable_parameters()

        # Score head
        hidden_size = base_model.config.hidden_size
        self.score_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.score_head.weight, std=0.02)
        nn.init.zeros_(self.score_head.bias)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                step_end_positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute scores at step terminator positions.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            step_end_positions: (batch, max_steps) — token indices of step
                                terminators. Padded with -1.

        Returns:
            scores: (total_valid_steps,) scores ∈ (0, 1)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden)

        # Extract hidden states at step end positions
        step_scores = []
        for b in range(hidden_states.size(0)):
            for pos in step_end_positions[b]:
                if pos == -1:  # padding
                    break
                pos = pos.item()
                if pos >= hidden_states.size(1):
                    pos = hidden_states.size(1) - 1
                h = hidden_states[b, pos, :]  # (hidden,)
                logit = self.score_head(h.float()).squeeze(-1)
                score = self.sigmoid(logit)
                step_scores.append(score)

        if not step_scores:
            return torch.tensor([], device=hidden_states.device)

        return torch.stack(step_scores)


# ---------------------------------------------------------------------------
# Loss: Class-weighted MSE
# ---------------------------------------------------------------------------

def class_weighted_mse_loss(
    pred_scores: torch.Tensor,
    true_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Class-weighted MSE loss for step-level PRM training.

    Weights:
      - 0.9 (主场确权步) → W=2.5 (highest value signal)
      - 0.1 (文化混淆步) → W=2.0 (second highest)
      - 0.5 (中立讨论步) → W=1.0 (baseline, abundant)

    Args:
        pred_scores: (N,) PRM predicted scores ∈ (0, 1)
        true_labels: (N,) true labels ∈ {0.1, 0.5, 0.9}

    Returns:
        Weighted MSE loss scalar
    """
    # Class-dependent weights
    weights = torch.where(
        true_labels > 0.7, torch.tensor(2.5, device=pred_scores.device),
        torch.where(
            true_labels < 0.3, torch.tensor(2.0, device=pred_scores.device),
            torch.tensor(1.0, device=pred_scores.device),
        )
    )

    mse = (pred_scores - true_labels) ** 2
    weighted_mse = mse * weights

    return weighted_mse.mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StepLabelDataset(Dataset):
    """
    Dataset for PRM training on step-labeled data.

    Each sample contains a full reasoning path with [Step N] markers.
    We tokenize the full path and record positions of step terminators.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_len: int = MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        for line in open(jsonl_path, encoding="utf-8"):
            obj = json.loads(line)
            steps = obj["steps"]
            if not steps or not all("label" in s for s in steps):
                continue

            # Build full reasoning text with step markers
            question = obj["question"]
            country = obj.get("country", "")
            full_reasoning = "\n".join(s["text"] for s in steps)

            # Input format: [country]\nquestion\nstep1\nstep2\n...
            input_text = f"[{country}]\n{question}\n{full_reasoning}"
            labels = [s["label"] for s in steps]

            self.samples.append({
                "input_text": input_text,
                "labels": labels,
                "steps": steps,
            })

        print(f"Loaded {len(self.samples)} PRM training samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tok = self.tokenizer

        # Tokenize full input
        enc = tok(
            sample["input_text"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        offsets = enc["offset_mapping"].squeeze(0)

        # Find step terminator positions
        # Each [Step N] section ends at the last token before the next [Step N+1]
        # or at the end of the sequence
        step_end_positions = self._find_step_end_positions(
            sample["input_text"], input_ids, tok
        )

        # Pad step positions to fixed size
        max_steps = 20  # Maximum steps per sample
        labels_tensor = torch.full((max_steps,), -1.0)
        positions_tensor = torch.full((max_steps,), -1, dtype=torch.long)

        for i, (pos, label) in enumerate(
            zip(step_end_positions, sample["labels"])
        ):
            if i >= max_steps:
                break
            positions_tensor[i] = pos
            labels_tensor[i] = label

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "step_end_positions": positions_tensor,
            "labels": labels_tensor,
        }

    def _find_step_end_positions(
        self, text: str, input_ids: torch.Tensor, tokenizer
    ) -> list[int]:
        """
        Find the token position of the last token in each step.

        Strategy: Find [Step N] token patterns and mark the last token
        before the next [Step N+1] (or end of sequence) as the terminator.
        """
        import re

        # Find character positions of each [Step N] marker
        step_starts = []
        for m in re.finditer(r'\[Step \d+\]', text):
            step_starts.append(m.start())

        if not step_starts:
            return []

        # Tokenize to get offset mapping
        enc = tokenizer(
            text, max_length=self.max_len, truncation=True,
            return_offsets_mapping=True, add_special_tokens=True,
        )
        offsets = enc["offset_mapping"]

        # For each step, find the last token before the next step starts
        end_positions = []
        for i, start_char in enumerate(step_starts):
            if i + 1 < len(step_starts):
                end_char = step_starts[i + 1] - 1
            else:
                # Last step: end at last non-padding token
                end_char = len(text) - 1

            # Find the token that covers end_char
            end_tok_pos = 0
            for tok_idx, (s, e) in enumerate(offsets):
                if s <= end_char < e or (e > 0 and e <= end_char + 1):
                    end_tok_pos = tok_idx

            end_positions.append(min(end_tok_pos, input_ids.shape[0] - 1))

        return end_positions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: CulturePRM, loader: DataLoader, device) -> dict:
    """
    Evaluate PRM on validation set.

    Metrics:
      - Three-class accuracy (discretized predictions vs labels)
      - Spearman correlation
      - Per-class recall
    """
    from scipy.stats import spearmanr

    model.eval()
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

    if not all_preds or not all_labels:
        model.train()
        return {"acc": 0.0, "spearman": 0.0}

    # Ensure same length
    n = min(len(all_preds), len(all_labels))
    all_preds = all_preds[:n]
    all_labels = all_labels[:n]

    # Discretize predictions for accuracy
    def discretize(val):
        if val > 0.7:
            return 0.9
        elif val < 0.3:
            return 0.1
        else:
            return 0.5

    pred_discrete = [discretize(p) for p in all_preds]
    label_discrete = [discretize(l) for l in all_labels]

    # Overall accuracy
    correct = sum(p == l for p, l in zip(pred_discrete, label_discrete))
    acc = correct / n if n > 0 else 0.0

    # Per-class recall
    class_correct = {0.9: 0, 0.5: 0, 0.1: 0}
    class_total = {0.9: 0, 0.5: 0, 0.1: 0}
    for p, l in zip(pred_discrete, label_discrete):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    recalls = {}
    for cls in [0.9, 0.5, 0.1]:
        recalls[cls] = (class_correct[cls] / class_total[cls]
                        if class_total[cls] > 0 else 0.0)

    # Spearman correlation
    try:
        spearman_r, _ = spearmanr(all_preds, all_labels)
    except Exception:
        spearman_r = 0.0

    model.train()
    return {
        "acc": acc,
        "spearman": spearman_r if spearman_r == spearman_r else 0.0,  # NaN check
        "recall_0.9": recalls[0.9],
        "recall_0.5": recalls[0.5],
        "recall_0.1": recalls[0.1],
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"SFT model (PRM backbone): {args.sft_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.sft_model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build model
    model = CulturePRM(
        sft_model_path=args.sft_model_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device)

    # Datasets
    train_ds = StepLabelDataset(args.train_file, tokenizer)
    val_ds = StepLabelDataset(args.val_file, tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Optimizer with different LRs for LoRA vs score_head
    param_groups = [
        {"params": [p for n, p in model.backbone.named_parameters()
                    if p.requires_grad],
         "lr": args.lr_lora},
        {"params": model.score_head.parameters(),
         "lr": args.lr_head},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            step_positions = batch["step_end_positions"].to(device)
            labels = batch["labels"].to(device)

            # Forward: get scores at step positions
            pred_scores = model(input_ids, attention_mask, step_positions)

            # Collect valid labels (non-padded)
            valid_labels = []
            for b in range(labels.size(0)):
                for s in range(labels.size(1)):
                    if labels[b, s] >= 0:
                        valid_labels.append(labels[b, s])

            if not valid_labels or len(pred_scores) == 0:
                continue

            true_labels = torch.stack(valid_labels[:len(pred_scores)])

            # Compute class-weighted MSE loss
            loss = class_weighted_mse_loss(pred_scores, true_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 20 == 0:
                print(f"  Epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={loss.item():.4f}")

        avg_loss = total_loss / max(len(train_loader), 1)
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | avg_loss={avg_loss:.4f} | "
              f"acc={metrics['acc']:.4f} | spearman={metrics['spearman']:.4f}")
        print(f"  Recall: 0.9={metrics['recall_0.9']:.3f} "
              f"0.5={metrics['recall_0.5']:.3f} "
              f"0.1={metrics['recall_0.1']:.3f}")

        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            ckpt_dir = Path(args.output_dir) / "best"
            ckpt_dir.mkdir(exist_ok=True)
            # Save LoRA adapter + score_head
            model.backbone.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save(model.score_head.state_dict(), ckpt_dir / "score_head.pt")
            print(f"  ✓ Saved best (acc={best_acc:.4f}) → {ckpt_dir}")

    print(f"\nTraining complete. Best accuracy: {best_acc:.4f}")
    print(f"Best checkpoint: {args.output_dir}/best")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CAMA-D Stage 3-PRM: Culture-Aware PRM Training"
    )
    parser.add_argument("--sft_model_path", type=str, required=True,
                        help="Path to Stage 1 SFT model (PRM backbone)")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Step-labeled JSONL from label_steps.py")
    parser.add_argument("--val_file", type=str, required=True,
                        help="Validation step-labeled JSONL")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for PRM checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_head", type=float, default=5e-5,
                        help="Learning rate for score_head (default: 5e-5)")
    parser.add_argument("--lr_lora", type=float, default=1e-4,
                        help="Learning rate for LoRA params (default: 1e-4)")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (default: 32)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
