"""
AgentArk Baseline — Stage 2: Process Reward Model Training

Key differences from CAMA-D PRM:
  - NO cultural-specific label semantics (no 主场确权/文化混淆 distinction)
  - Standard step labels: {0.9=correct_step, 0.5=neutral, 0.1=wrong_step}
  - Same architecture: Base Model + SFT LoRA (merged) + PRM LoRA + score_head

Pipeline:
  1. split_steps: Split debate reasoning into steps with [Step N] markers
  2. label_steps: Use LLM to label each step (open-book, correct answer given)
  3. train_prm: Train PRM on step-labeled data (this script)

Usage:
    # Single GPU
    python ark/culture/train_prm.py \\
        --base_model_path /path/to/Qwen2.5-7B-Instruct \\
        --sft_adapter_path /path/to/agentark_sft_qwen/best \\
        --train_file      /path/to/agentark_step_labels_train.jsonl \\
        --val_file        /path/to/agentark_step_labels_val.jsonl \\
        --output_dir      /path/to/models/agentark_prm \\
        --epochs 5 \\
        --batch_size 8

    # Multi-GPU (DDP via torchrun)
    torchrun --nproc_per_node=2 ark/culture/train_prm.py \\
        --base_model_path /path/to/Qwen2.5-7B-Instruct \\
        --sft_adapter_path /path/to/agentark_sft_qwen/best \\
        --train_file      /path/to/agentark_step_labels_train.jsonl \\
        --val_file        /path/to/agentark_step_labels_val.jsonl \\
        --output_dir      /path/to/models/agentark_prm

Step label format (JSONL):
    {
      "question": "...",
      "country": "Vietnam",     # retained for data compatibility but NOT used in PRM
      "gt": "1",
      "steps": [
        {"step_idx": 1, "text": "[Step 1] ...", "label": 0.9},
        {"step_idx": 2, "text": "[Step 2] ...", "label": 0.5},
        ...
      ]
    }
"""

import os
import re
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize DDP if launched via torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def is_main_process(rank: int) -> bool:
    return rank == 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


MAX_SEQ_LEN = 1024


# ---------------------------------------------------------------------------
# Model: AgentArk PRM (Base + SFT LoRA merged + PRM LoRA + score_head)
# ---------------------------------------------------------------------------

class AgentArkPRM(nn.Module):
    """
    Process Reward Model for AgentArk baseline.

    Architecture (same as CAMA-D PRM — fair comparison):
      - Backbone: Base model + SFT LoRA (merged)
      - Adapter: New PRM LoRA trained on step labels
      - Head: Linear(hidden_size, 1) + Sigmoid → score ∈ (0, 1)

    The key difference is in the training data:
      - AgentArk labels are based on generic step quality (correct/neutral/wrong)
      - CAMA-D labels encode cultural authority (主场确权/文化混淆)
    """

    def __init__(self, base_model_path: str, sft_adapter_path: str = None,
                 lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05):
        super().__init__()

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            output_hidden_states=True,
        )

        # Merge SFT LoRA (if provided)
        if sft_adapter_path:
            print(f"  [PRM] Merging SFT LoRA adapter: {sft_adapter_path}")
            base_model = PeftModel.from_pretrained(base_model, sft_adapter_path)
            base_model = base_model.merge_and_unload()
            print(f"  [PRM] SFT adapter merged into backbone")

        # Apply PRM-specific LoRA
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

        # Enable gradient checkpointing
        self.backbone.enable_input_require_grads()
        self.backbone.gradient_checkpointing_enable()

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
            step_end_positions: (batch, max_steps) token indices, padded with -1

        Returns:
            scores: (total_valid_steps,) scores ∈ (0, 1)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden)

        step_scores = []
        for b in range(hidden_states.size(0)):
            for pos in step_end_positions[b]:
                if pos == -1:
                    break
                pos = pos.item()
                if pos >= hidden_states.size(1):
                    pos = hidden_states.size(1) - 1
                h = hidden_states[b, pos, :]
                logit = self.score_head(h.float()).squeeze(-1)
                score = self.sigmoid(logit)
                step_scores.append(score)

        if not step_scores:
            return torch.tensor([], device=hidden_states.device)

        return torch.stack(step_scores)


# ---------------------------------------------------------------------------
# Loss: Standard MSE (no class weighting — AgentArk baseline)
# ---------------------------------------------------------------------------

def standard_mse_loss(
    pred_scores: torch.Tensor,
    true_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Standard MSE loss (uniform weighting across all step classes).

    This is the AgentArk baseline — no class-dependent weighting.
    CAMA-D uses weighted MSE (2.5x for 0.9, 2.0x for 0.1).
    AgentArk treats all steps equally.

    Args:
        pred_scores: (N,) PRM predicted scores ∈ (0, 1)
        true_labels: (N,) true labels ∈ {0.1, 0.5, 0.9}

    Returns:
        MSE loss scalar
    """
    mse = (pred_scores - true_labels) ** 2
    return mse.mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StepLabelDataset(Dataset):
    """
    Dataset for PRM training on step-labeled data.

    Pre-tokenizes all samples in __init__ for fork-safe DataLoader usage.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_len: int = MAX_SEQ_LEN):
        self.max_len = max_len
        self.processed_samples = []
        max_steps = 20

        raw_count = 0
        for line in open(jsonl_path, encoding="utf-8"):
            obj = json.loads(line)
            steps = obj["steps"]
            if not steps or not all("label" in s for s in steps):
                continue

            raw_count += 1
            # Build full reasoning text with step markers
            # AgentArk: no country prefix (culture-agnostic PRM)
            question = obj["question"]
            full_reasoning = "\n".join(s["text"] for s in steps)
            input_text = f"{question}\n{full_reasoning}"
            labels = [s["label"] for s in steps]

            # Tokenize with offset mapping
            enc = tokenizer(
                input_text,
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            offsets = enc["offset_mapping"].squeeze(0).tolist()

            # Find step end positions
            step_starts_char = [m.start() for m in re.finditer(r'\[Step \d+\]', input_text)]
            if not step_starts_char:
                continue

            end_positions = []
            for i, start_char in enumerate(step_starts_char):
                if i + 1 < len(step_starts_char):
                    end_char = step_starts_char[i + 1] - 1
                else:
                    end_char = len(input_text) - 1

                end_tok_pos = 0
                for tok_idx, (s, e) in enumerate(offsets):
                    if s <= end_char < e or (e > 0 and e <= end_char + 1):
                        end_tok_pos = tok_idx

                end_positions.append(min(end_tok_pos, input_ids.shape[0] - 1))

            # Pad to fixed size
            labels_tensor = torch.full((max_steps,), -1.0)
            positions_tensor = torch.full((max_steps,), -1, dtype=torch.long)

            for i, (pos, label) in enumerate(zip(end_positions, labels)):
                if i >= max_steps:
                    break
                positions_tensor[i] = pos
                labels_tensor[i] = label

            self.processed_samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "step_end_positions": positions_tensor,
                "labels": labels_tensor,
            })

        print(f"Loaded {len(self.processed_samples)} PRM training samples "
              f"from {jsonl_path} (filtered from {raw_count})")

    def __len__(self):
        return len(self.processed_samples)

    def __getitem__(self, idx):
        return self.processed_samples[idx]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: AgentArkPRM, loader: DataLoader, device) -> dict:
    """Evaluate PRM on validation set."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        step_positions = batch["step_end_positions"].to(device)
        labels = batch["labels"]

        scores = model(input_ids, attention_mask, step_positions)

        for b in range(labels.size(0)):
            for s in range(labels.size(1)):
                if labels[b, s] >= 0:
                    all_labels.append(labels[b, s].item())

        all_preds.extend(scores.cpu().tolist())

    if not all_preds or not all_labels:
        model.train()
        return {"acc": 0.0, "spearman": 0.0}

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

    correct = sum(p == l for p, l in zip(pred_discrete, label_discrete))
    acc = correct / n if n > 0 else 0.0

    # Spearman correlation
    try:
        from scipy.stats import spearmanr
        spearman_r, _ = spearmanr(all_preds, all_labels)
    except Exception:
        spearman_r = 0.0

    model.train()
    return {
        "acc": acc,
        "spearman": spearman_r if spearman_r == spearman_r else 0.0,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    use_ddp = world_size > 1

    if is_main_process(rank):
        print(f"{'='*60}")
        print(f"AgentArk Baseline — Stage 2: PRM Training (Standard MSE)")
        print(f"{'='*60}")
        print(f"Device: {device} | World size: {world_size}")
        print(f"Base model: {args.base_model_path}")
        print(f"SFT adapter: {args.sft_adapter_path or 'None (using base directly)'}")
        print(f"Loss: Standard MSE (uniform weighting, no class weights)")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build model
    model = AgentArkPRM(
        base_model_path=args.base_model_path,
        sft_adapter_path=args.sft_adapter_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Datasets
    if is_main_process(rank):
        print("[INFO] Loading datasets...")
    train_ds = StepLabelDataset(args.train_file, tokenizer)
    val_ds = StepLabelDataset(args.val_file, tokenizer)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=0, pin_memory=False,
    )
    if is_main_process(rank):
        print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Optimizer
    raw_model = model.module if use_ddp else model
    param_groups = [
        {"params": [p for n, p in raw_model.backbone.named_parameters()
                    if p.requires_grad],
         "lr": args.lr_lora},
        {"params": raw_model.score_head.parameters(),
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
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            step_positions = batch["step_end_positions"].to(device)
            labels = batch["labels"].to(device)

            pred_scores = model(input_ids, attention_mask, step_positions)

            # Collect valid labels
            valid_labels = []
            for b in range(labels.size(0)):
                for s in range(labels.size(1)):
                    if labels[b, s] >= 0:
                        valid_labels.append(labels[b, s])

            if not valid_labels or len(pred_scores) == 0:
                continue

            true_labels = torch.stack(valid_labels[:len(pred_scores)])

            # Standard MSE loss (AgentArk baseline: no class weighting)
            loss = standard_mse_loss(pred_scores, true_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 20 == 0 and is_main_process(rank):
                print(f"  Epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={loss.item():.4f}")

        avg_loss = total_loss / max(len(train_loader), 1)
        if is_main_process(rank):
            print(f"Epoch {epoch}/{args.epochs} | avg_loss={avg_loss:.4f}")

        # Evaluate
        if epoch % args.eval_every_n_epochs == 0 and is_main_process(rank):
            eval_model = raw_model
            metrics = evaluate(eval_model, val_loader, device)
            print(f"  [Eval] Epoch {epoch} | "
                  f"acc={metrics['acc']:.4f} | spearman={metrics['spearman']:.4f}")

            if metrics["acc"] > best_acc:
                best_acc = metrics["acc"]
                ckpt_dir = Path(args.output_dir) / "best"
                ckpt_dir.mkdir(exist_ok=True)
                raw_model.backbone.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                torch.save(raw_model.score_head.state_dict(),
                           ckpt_dir / "score_head.pt")
                print(f"    ✓ Saved best (acc={best_acc:.4f}) → {ckpt_dir}")
            else:
                print(f"    No improvement (best={best_acc:.4f})")

        if use_ddp:
            dist.barrier()

    if best_acc == 0.0 and is_main_process(rank):
        ckpt_dir = Path(args.output_dir) / "best"
        ckpt_dir.mkdir(exist_ok=True)
        raw_model.backbone.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        torch.save(raw_model.score_head.state_dict(), ckpt_dir / "score_head.pt")
        print(f"  Saved final model → {ckpt_dir}")

    if is_main_process(rank):
        print(f"\nTraining complete. Best accuracy: {best_acc:.4f}")
        print(f"Best checkpoint: {args.output_dir}/best")

    cleanup_distributed()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AgentArk Baseline — Stage 2: PRM Training (Standard MSE)"
    )
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model (e.g., Qwen2.5-7B-Instruct)")
    parser.add_argument("--sft_adapter_path", type=str, default=None,
                        help="Path to SFT LoRA adapter (merged into base)")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Step-labeled JSONL training file")
    parser.add_argument("--val_file", type=str, required=True,
                        help="Step-labeled JSONL validation file")
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
    parser.add_argument("--eval_every_n_epochs", type=int, default=1,
                        help="Evaluate every N epochs (default: 1)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
