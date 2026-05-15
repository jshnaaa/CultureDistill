"""
Step 2: Train the Process Reward Model (PRM) on pairwise cultural consistency data.

Architecture: Qwen3-0.6B base + LoRA + Linear reward head
Loss: Bradley-Terry ranking loss
Input: [country] question reasoning_path [ANSWER: X]
Output: scalar score ∈ (0, 1)

Usage:
    python Cul/prm/train_prm.py \
        --train_file /autodl-fs/data/prm/prm_train_pairs.jsonl \
        --val_file   /autodl-fs/data/prm/prm_val_pairs.jsonl \
        --output_dir /autodl-fs/models/prm_qwen3_0.6b \
        --epochs 5 \
        --batch_size 16 \
        --lr 1e-5

Improvements over v1:
    1. LoRA (r=16) instead of full fine-tuning → regularization for small datasets
    2. Explicit [ANSWER: X] appended to input → clear discriminative signal
    3. Mean-pool over non-padding tokens → robust sequence representation
    4. MAX_SEQ_LEN raised to 2048 → avoid truncating answer at tail
"""

# ---------------------------------------------------------------------------
# Qwen3 requires transformers >= 4.51.0. Auto-upgrade if needed.
# ---------------------------------------------------------------------------
import importlib.metadata as _meta
import os, subprocess, sys

_MIN_TRANSFORMERS = "4.51.0"

def _version_tuple(v: str):
    return tuple(int(x) for x in v.split(".")[:3])

_cur_ver = _meta.version("transformers")
if _version_tuple(_cur_ver) < _version_tuple(_MIN_TRANSFORMERS):
    print(f"[auto-fix] transformers {_cur_ver} < {_MIN_TRANSFORMERS}, "
          f"upgrading to support Qwen3 architecture ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        f"transformers>={_MIN_TRANSFORMERS}",
    ])
    print("[auto-fix] transformers upgraded — restarting script ...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Ensure peft is installed for LoRA
try:
    import peft as _peft_check  # noqa: F401
except ImportError:
    print("[auto-fix] Installing peft for LoRA support ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "peft>=0.11",
    ])

import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType


PRM_BASE = "/root/autodl-tmp/base/Qwen3-0.6B-Base"
MAX_SEQ_LEN = 2048      # v1 was 1024, often truncated answers


# ---------------------------------------------------------------------------
# Model — LoRA + mean-pooling reward head
# ---------------------------------------------------------------------------

class CultureRewardModel(nn.Module):
    def __init__(self, model_path: str, use_lora: bool = True,
                 lora_r: int = 16, lora_alpha: int = 32):
        super().__init__()
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            output_hidden_states=True,
        )

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            self.model = get_peft_model(base_model, lora_config)
            self.model.print_trainable_parameters()
        else:
            self.model = base_model

        hidden_size = base_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    def _pool(self, hidden_states: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool over non-padding tokens (much more robust than last-token
        for base models with right-padding)."""
        mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
        summed = (hidden_states * mask).sum(dim=1)           # (B, H)
        lengths = mask.sum(dim=1).clamp(min=1)               # (B, 1)
        return summed / lengths                              # (B, H)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]              # (B, L, H)
        pooled = self._pool(last_hidden, attention_mask)     # (B, H)
        score = self.reward_head(pooled.float()).squeeze(-1)  # (B,)
        return torch.sigmoid(score)


# ---------------------------------------------------------------------------
# Dataset — now includes explicit answer in input text
# ---------------------------------------------------------------------------

class PairwiseDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_len: int = MAX_SEQ_LEN):
        self.pairs = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _encode(self, question: str, country: str,
                reasoning: str, answer: str) -> dict:
        """Build input text with explicit answer for clear signal."""
        text = f"[{country}]\n{question}\n{reasoning}\n[ANSWER: {answer}]"
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        question = pair["question"]
        country  = pair["country"]
        chosen_enc = self._encode(
            question, country,
            pair["chosen"]["reasoning"], pair["chosen"].get("answer", ""))
        rejected_enc = self._encode(
            question, country,
            pair["rejected"]["reasoning"], pair["rejected"].get("answer", ""))
        return chosen_enc, rejected_enc


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def bradley_terry_loss(score_chosen: torch.Tensor,
                       score_rejected: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(score_chosen - score_rejected) + 1e-8).mean()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Return pairwise accuracy and mean margin (chosen_score - rejected_score)."""
    model.eval()
    correct, total = 0, 0
    margins = []
    for chosen_enc, rejected_enc in loader:
        chosen_ids  = chosen_enc["input_ids"].to(device)
        chosen_mask = chosen_enc["attention_mask"].to(device)
        rej_ids     = rejected_enc["input_ids"].to(device)
        rej_mask    = rejected_enc["attention_mask"].to(device)

        sc = model(chosen_ids, chosen_mask)
        sr = model(rej_ids, rej_mask)
        correct += (sc > sr).sum().item()
        total   += sc.size(0)
        margins.append((sc - sr).cpu())

    model.train()
    all_margins = torch.cat(margins)
    return {
        "acc": correct / total if total > 0 else 0.0,
        "margin_mean": all_margins.mean().item(),
        "margin_std":  all_margins.std().item(),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPUs available: {n_gpus}")

    tokenizer = AutoTokenizer.from_pretrained(PRM_BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = CultureRewardModel(
        PRM_BASE,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device)

    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel on {n_gpus} GPUs")

    base_model = model.module if n_gpus > 1 else model
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"Trainable params: {trainable/1e6:.1f}M / {total_params/1e6:.1f}M "
          f"({100*trainable/total_params:.2f}%)")

    train_ds = PairwiseDataset(args.train_file, tokenizer, args.max_seq_len)
    val_ds   = PairwiseDataset(args.val_file,   tokenizer, args.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2)
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")
    print(f"Max seq len: {args.max_seq_len}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.05,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, (chosen_enc, rejected_enc) in enumerate(train_loader, 1):
            chosen_ids  = chosen_enc["input_ids"].to(device)
            chosen_mask = chosen_enc["attention_mask"].to(device)
            rej_ids     = rejected_enc["input_ids"].to(device)
            rej_mask    = rejected_enc["attention_mask"].to(device)

            sc = model(chosen_ids, chosen_mask)
            sr = model(rej_ids,    rej_mask)
            loss = bradley_terry_loss(sc, sr)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 20 == 0:
                print(f"  Epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        metrics  = evaluate(model, val_loader, device)
        val_acc  = metrics["acc"]
        print(f"Epoch {epoch}/{args.epochs} | avg_loss={avg_loss:.4f} | "
              f"val_acc={val_acc:.4f} | "
              f"margin={metrics['margin_mean']:.4f}±{metrics['margin_std']:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_dir = Path(args.output_dir) / "best"
            ckpt_dir.mkdir(exist_ok=True)
            # Save LoRA adapter (or full model) + reward head
            base_model.model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save(base_model.reward_head.state_dict(),
                       ckpt_dir / "reward_head.pt")
            print(f"  ✓ Saved best checkpoint (val_acc={best_acc:.4f})")

    print(f"\nTraining complete. Best val Pairwise Accuracy: {best_acc:.4f}")
    print(f"Best checkpoint: {args.output_dir}/best")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file",   type=str, required=True)
    parser.add_argument("--val_file",     type=str, required=True)
    parser.add_argument("--output_dir",   type=str, required=True)
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--max_seq_len",  type=int,   default=MAX_SEQ_LEN,
                        help="Maximum sequence length (default: 2048)")
    parser.add_argument("--no_lora",      action="store_true",
                        help="Disable LoRA and use full fine-tuning")
    parser.add_argument("--lora_r",       type=int,   default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha",   type=int,   default=32,
                        help="LoRA alpha (default: 32)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
