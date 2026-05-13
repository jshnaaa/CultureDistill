"""
SFT fine-tuning for cultural alignment distillation.

Training data: correct reasoning paths from MAS inference data (grpo_train.jsonl).
  - Solution 1-5 (agent paths) where answer == gold
  - Solution 6 (Judge path) where answer == gold (higher priority)

Input format:  [{country}]\n{question}
Output format: Reasoning: {reasoning}\nAnswer: {answer}

Usage:
    python Cul/sft/train_sft.py \
        --model_name  llama \
        --data_file   /autodl-fs/data/splits/grpo_train.jsonl \
        --val_file    /autodl-fs/data/splits/prm_val.jsonl \
        --output_dir  /autodl-fs/models/sft_llama \
        --epochs      3 \
        --batch_size  8 \
        --lr          1e-5

    python Cul/sft/train_sft.py \
        --model_name  qwen \
        --data_file   /autodl-fs/data/splits/grpo_train.jsonl \
        --val_file    /autodl-fs/data/splits/prm_val.jsonl \
        --output_dir  /autodl-fs/models/sft_qwen \
        --epochs      3 \
        --batch_size  8 \
        --lr          1e-5
"""

import re
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


MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}

MAX_SEQ_LEN = 1024
IGNORE_INDEX = -100     # standard label mask for padding / prompt tokens


# ---------------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------------

def split_solutions(response: str) -> list[str]:
    parts = re.split(r"===== Solution \d+ =====", response)
    return [p.strip() for p in parts if p.strip()]


def extract_reasoning_and_answer(text: str) -> tuple[str, str] | None:
    """Return (reasoning, answer) or None if either is missing."""
    r_match = re.search(r"Reasoning\s*:\s*(.*?)(?:\nAnswer|\Z)", text,
                        re.DOTALL | re.IGNORECASE)
    a_match = re.search(r"Answer\s*:\s*([1-4])", text, re.IGNORECASE)
    if not a_match:
        # Fallback answer extraction
        a_match = re.search(r"answer\s+is\s*:?\s*([1-4])\b", text, re.IGNORECASE)
    if r_match and a_match:
        reasoning = r_match.group(1).strip()
        answer    = a_match.group(1).strip()
        return reasoning, answer
    return None


def build_sft_samples(jsonl_path: str) -> list[dict]:
    """
    Extract (query, country, reasoning, answer) tuples from MAS inference data.
    Includes:
      - Agent paths (Solution 1-5) where answer == gold
      - Judge path  (Solution 6)   where answer == gold  [marked as is_judge=True]
    """
    samples = []
    for line in open(jsonl_path, encoding="utf-8"):
        obj     = json.loads(line)
        query   = obj["query"]
        country = obj.get("country", "")
        gold    = str(obj["gt"]).strip()
        sols    = split_solutions(obj["response"])

        agent_sols = sols[:5]
        judge_sols = sols[5:6]   # at most one

        for sol in agent_sols:
            parsed = extract_reasoning_and_answer(sol)
            if parsed and parsed[1] == gold:
                samples.append({
                    "query":    query,
                    "country":  country,
                    "reasoning": parsed[0],
                    "answer":   parsed[1],
                    "is_judge": False,
                })

        for sol in judge_sols:
            parsed = extract_reasoning_and_answer(sol)
            if parsed and parsed[1] == gold:
                samples.append({
                    "query":    query,
                    "country":  country,
                    "reasoning": parsed[0],
                    "answer":   parsed[1],
                    "is_judge": True,
                })

    n_agent = sum(1 for s in samples if not s["is_judge"])
    n_judge = sum(1 for s in samples if s["is_judge"])
    print(f"SFT samples: {len(samples)} total "
          f"(agent={n_agent}, judge={n_judge})")
    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, max_len: int = MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.records   = []

        for s in samples:
            input_text  = f"[{s['country']}]\n{s['query']}"
            target_text = (f"Reasoning: {s['reasoning']}\n"
                           f"Answer: {s['answer']}")
            messages = [
                {"role": "user",      "content": input_text},
                {"role": "assistant", "content": target_text},
            ]
            # Full sequence = prompt + response (no separate tokenization needed)
            full = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            # Prompt only (to compute where to start supervising)
            prompt_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                tokenize=False, add_generation_prompt=True
            )
            self.records.append((full, prompt_only, s["is_judge"]))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        full_text, prompt_text, is_judge = self.records[idx]
        tok = self.tokenizer

        full_enc   = tok(full_text,   max_length=self.max_len,
                         truncation=True, return_tensors="pt")
        prompt_enc = tok(prompt_text, max_length=self.max_len,
                         truncation=True, return_tensors="pt")

        input_ids = full_enc["input_ids"].squeeze(0)
        labels    = input_ids.clone()

        # Mask prompt tokens: only supervise on the response part
        prompt_len = prompt_enc["input_ids"].shape[1]
        labels[:prompt_len] = IGNORE_INDEX

        return {
            "input_ids":      input_ids,
            "attention_mask": full_enc["attention_mask"].squeeze(0),
            "labels":         labels,
            "is_judge":       torch.tensor(is_judge, dtype=torch.bool),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad to max length in batch."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_id  = 0

    input_ids_list, mask_list, label_list = [], [], []
    for b in batch:
        n = b["input_ids"].shape[0]
        pad = max_len - n
        input_ids_list.append(
            torch.cat([b["input_ids"], torch.full((pad,), pad_id)])
        )
        mask_list.append(
            torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)])
        )
        label_list.append(
            torch.cat([b["labels"], torch.full((pad,), IGNORE_INDEX)])
        )

    return {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(mask_list),
        "labels":         torch.stack(label_list),
        "is_judge":       torch.stack([b["is_judge"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Validation accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, tokenizer, val_jsonl: str, device, max_samples: int = 245) -> float:
    model.eval()
    correct, total = 0, 0
    for line in open(val_jsonl, encoding="utf-8"):
        if total >= max_samples:
            break
        obj     = json.loads(line)
        query   = obj["query"]
        country = obj.get("country", "")
        gold    = str(obj["gt"]).strip()

        input_text = f"[{country}]\n{query}"
        messages   = [{"role": "user", "content": input_text}]
        prompt     = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt, return_tensors="pt",
                        max_length=MAX_SEQ_LEN, truncation=True).to(device)
        outs = model.generate(
            **enc,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        response   = tokenizer.decode(outs[0][prompt_len:], skip_special_tokens=True)

        pred = None
        m = re.search(r"Answer\s*:\s*([1-4])", response, re.IGNORECASE)
        if m:
            pred = m.group(1)
        else:
            digits = re.findall(r"\b([1-4])\b", response)
            if digits:
                pred = digits[-1]

        if pred == gold:
            correct += 1
        total += 1

    model.train()
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = MODEL_ALIASES.get(args.model_name, args.model_name)
    print(f"Base model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Build SFT samples from MAS inference data
    raw_samples = build_sft_samples(args.data_file)
    if len(raw_samples) == 0:
        raise ValueError("No SFT samples found. Check data_file path and format.")

    train_ds = SFTDataset(raw_samples, tokenizer)
    loader   = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    print(f"Training batches per epoch: {len(loader)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    total_steps  = len(loader) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.05))
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    no_improve   = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(loader, 1):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 20 == 0:
                print(f"  Epoch {epoch} step {step}/{len(loader)} "
                      f"loss={loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        val_acc  = validate(model, tokenizer, args.val_file, device)
        print(f"Epoch {epoch}/{args.epochs} | "
              f"avg_loss={avg_loss:.4f} | val_accuracy={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            ckpt = Path(args.output_dir) / "best"
            model.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            print(f"  ✓ Saved best checkpoint (val_acc={best_val_acc:.4f}) → {ckpt}")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/2)")
            if no_improve >= 2:
                print("Early stopping.")
                break

    print(f"\nTraining complete. Best val_accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {args.output_dir}/best")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  type=str,   required=True,
                        help="'llama', 'qwen', or full path")
    parser.add_argument("--data_file",   type=str,   required=True,
                        help="grpo_train.jsonl (MAS inference data)")
    parser.add_argument("--val_file",    type=str,   required=True,
                        help="prm_val.jsonl for validation accuracy")
    parser.add_argument("--output_dir",  type=str,   required=True)
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-5)
    args = parser.parse_args()

    args.model_name = MODEL_ALIASES.get(args.model_name, args.model_name)
    train(args)


if __name__ == "__main__":
    main()
