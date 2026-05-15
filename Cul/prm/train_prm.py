"""
Step 2: Train the Process Reward Model (PRM) on pairwise cultural consistency data.

Architecture: Qwen3-0.6B base + Linear reward head (full fine-tuning)
Loss: Bradley-Terry ranking loss
Input: [country] question reasoning_path
Output: scalar score ∈ (0, 1)

Usage:
    python Cul/prm/train_prm.py \
        --train_file /autodl-fs/data/prm/prm_train_pairs.jsonl \
        --val_file   /autodl-fs/data/prm/prm_val_pairs.jsonl \
        --output_dir /autodl-fs/models/prm_qwen3_0.6b \
        --epochs 5 \
        --batch_size 16 \
        --lr 1e-5
"""

# ---------------------------------------------------------------------------
# Qwen3 requires transformers >= 4.51.0. Auto-upgrade if the installed
# version is too old, so the script works out-of-the-box on fresh servers.
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
    # Re-exec this script so the new transformers is fully loaded
    os.execv(sys.executable, [sys.executable] + sys.argv)

import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup


PRM_BASE = "/root/autodl-tmp/base/Qwen3-0.6B-Base"
MAX_SEQ_LEN = 1024


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CultureRewardModel(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        # Use AutoModelForCausalLM instead of AutoModel to bypass the
        # CONFIG_MAPPING check that fails for new architectures (e.g. qwen3)
        # on older transformers versions. trust_remote_code loads the model
        # class from the checkpoint's modeling_*.py directly.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            output_hidden_states=True,
        )
        hidden_size = self.model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # AutoModelForCausalLM returns hidden_states tuple; take last layer
        # last_hidden_state is not available on CausalLM outputs
        last_hidden = outputs.hidden_states[-1][:, -1, :]   # (B, hidden)
        score = self.reward_head(last_hidden.float()).squeeze(-1)
        return torch.sigmoid(score)     # (batch,) ∈ (0, 1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairwiseDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_len: int = MAX_SEQ_LEN):
        self.pairs = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _encode(self, question: str, country: str, reasoning: str) -> dict:
        text = f"[{country}]\n{question}\n{reasoning}"
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
        chosen_enc  = self._encode(question, country, pair["chosen"]["reasoning"])
        rejected_enc = self._encode(question, country, pair["rejected"]["reasoning"])
        return chosen_enc, rejected_enc


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def bradley_terry_loss(score_chosen: torch.Tensor, score_rejected: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(score_chosen - score_rejected) + 1e-8).mean()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for chosen_enc, rejected_enc in loader:
        chosen_ids  = chosen_enc["input_ids"].to(device)
        chosen_mask = chosen_enc["attention_mask"].to(device)
        rej_ids     = rejected_enc["input_ids"].to(device)
        rej_mask    = rejected_enc["attention_mask"].to(device)

        sc = model(chosen_ids, chosen_mask)
        sr = model(rej_ids, rej_mask)
        correct += (sc > sr).sum().item()
        total   += sc.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    # Multi-GPU: use all available GPUs via DataParallel (simple 2-card setup)
    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPUs available: {n_gpus}")

    tokenizer = AutoTokenizer.from_pretrained(PRM_BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = CultureRewardModel(PRM_BASE).to(device)

    # Wrap with DataParallel when multiple GPUs are available
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel on {n_gpus} GPUs")

    base_model = model.module if n_gpus > 1 else model
    print(f"Model params: {sum(p.numel() for p in base_model.parameters()) / 1e6:.1f}M")

    train_ds = PairwiseDataset(args.train_file, tokenizer)
    val_ds   = PairwiseDataset(args.val_file,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.05))
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 20 == 0:
                print(f"  Epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | avg_loss={avg_loss:.4f} | "
              f"val_pairwise_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_dir = Path(args.output_dir) / "best"
            ckpt_dir.mkdir(exist_ok=True)
            # Access underlying model through .module when using DataParallel
            base_model.model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save(base_model.reward_head.state_dict(),
                       ckpt_dir / "reward_head.pt")
            print(f"  ✓ Saved best checkpoint (val_acc={best_acc:.4f})")

        if val_acc < 0.55 and epoch >= 2:
            print("  Warning: val_acc < 55% — consider switching to LoRA or more data")

    print(f"\nTraining complete. Best val Pairwise Accuracy: {best_acc:.4f}")
    print(f"Best checkpoint: {args.output_dir}/best")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file",  type=str, required=True)
    parser.add_argument("--val_file",    type=str, required=True)
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
