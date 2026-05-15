"""
GRPO fine-tuning for cultural alignment distillation.

R_total = 0.7 * R_ans + 0.3 * R_cultural
  R_ans     : verifiable reward (answer == gold → 1, else 0)
  R_cultural: PRM score (Qwen3-0.6B + LoRA + mean-pool reward head), clipped to [0.1, 0.9]

Training data: grpo_train.jsonl  (prompts, online sampling)
Validation:    prm_val.jsonl     (samples, every eval_every rounds)

Usage:
    # RL-only (from base model)
    deepspeed --num_gpus 2 Cul/grpo/train_grpo.py \
        --model_name     qwen \
        --grpo_data      /autodl-fs/data/qwen/normad_splits/grpo_train.jsonl \
        --val_data       /autodl-fs/data/qwen/normad_splits/prm_val.jsonl \
        --prm_path       /autodl-tmp/models/normad_prm_qwen3_0.6b_v2/best \
        --output_dir     /autodl-tmp/models/grpo_qwen_culture \
        --n_samples      10 \
        --max_rounds     30 \
        --eval_every     5

    # SFT+RL (from SFT checkpoint)
    deepspeed --num_gpus 2 Cul/grpo/train_grpo.py \
        --model_name     qwen \
        --pretrain_path  /autodl-tmp/models/sft_qwen/best \
        --grpo_data      /autodl-fs/data/qwen/normad_splits/grpo_train.jsonl \
        --val_data       /autodl-fs/data/qwen/normad_splits/prm_val.jsonl \
        --prm_path       /autodl-tmp/models/normad_prm_qwen3_0.6b_v2/best \
        --output_dir     /autodl-tmp/models/grpo_sft_qwen_culture \
        --n_samples      10 \
        --max_rounds     30 \
        --eval_every     5
"""

import re
import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}
PRM_BASE = "/root/autodl-tmp/base/Qwen3-0.6B-Base"

ALPHA = 0.7          # R_ans weight
MAX_GEN_LEN  = 512
MAX_PROMPT_LEN = 512
KL_COEF = 0.001


# ---------------------------------------------------------------------------
# PRM (frozen, for R_cultural scoring) — matches train_prm.py v2 architecture
# ---------------------------------------------------------------------------

class CultureRewardModel(nn.Module):
    """
    Must match the architecture used during PRM training (v2):
      - AutoModelForCausalLM (not AutoModel)
      - LoRA adapter loaded on top
      - Mean-pooling over non-padding tokens (not last-token)
      - Input format: [country]\nquestion\nreasoning\n[ANSWER: X]
    """
    def __init__(self, prm_checkpoint_dir: str, base_path: str = PRM_BASE):
        super().__init__()
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            output_hidden_states=True,
        )

        # Load LoRA adapter if present
        adapter_path = Path(prm_checkpoint_dir) / "adapter_model.safetensors"
        adapter_path_bin = Path(prm_checkpoint_dir) / "adapter_model.bin"
        if adapter_path.exists() or adapter_path_bin.exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base_model, prm_checkpoint_dir)
            print(f"  [PRM] Loaded LoRA adapter from {prm_checkpoint_dir}")
        else:
            # Fallback: full model checkpoint (no LoRA)
            self.model = AutoModelForCausalLM.from_pretrained(
                prm_checkpoint_dir, torch_dtype=torch.bfloat16,
                trust_remote_code=True, output_hidden_states=True,
            )
            print(f"  [PRM] Loaded full model from {prm_checkpoint_dir}")

        # Load reward head
        hidden_size = base_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)
        head_path = Path(prm_checkpoint_dir) / "reward_head.pt"
        state = torch.load(head_path, map_location="cpu")
        self.reward_head.load_state_dict(state)
        print(f"  [PRM] Loaded reward head from {head_path}")

    def _pool(self, hidden_states: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool over non-padding tokens (matches train_prm.py v2)."""
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        return summed / lengths

    @torch.no_grad()
    def score(self, input_ids: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]
        pooled = self._pool(last_hidden, attention_mask)
        s = torch.sigmoid(self.reward_head(pooled.float()).squeeze(-1))
        return s.clamp(0.1, 0.9)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class GRPOPromptDataset(Dataset):
    """Each item is a (query, country, gold_answer) tuple."""
    def __init__(self, jsonl_path: str):
        self.items = []
        for line in open(jsonl_path, encoding="utf-8"):
            obj = json.loads(line)
            self.items.append({
                "query":   obj["query"],
                "country": obj.get("country", ""),
                "gt":      str(obj["gt"]).strip(),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class ValDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt(query: str, country: str, tokenizer) -> str:
    """Build culture-conditioned prompt."""
    content = f"[{country}]\n{query}\n\nReasoning: "
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_answer(text: str) -> str | None:
    m = re.search(r"Answer\s*:\s*([1-4])", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"answer\s+is\s*:?\s*([1-4])\b", text, re.IGNORECASE)
    if m:
        return m.group(1)
    digits = re.findall(r"\b([1-4])\b", text)
    return digits[-1] if digits else None


def compute_r_ans(predicted: str | None, gold: str) -> float:
    return 1.0 if predicted == gold else 0.0


def build_prm_input(country: str, query: str, reasoning: str,
                    answer: str | None) -> str:
    """
    Build PRM input text matching the format used during PRM training:
    [country]\nquestion\nreasoning\n[ANSWER: X]
    """
    text = f"[{country}]\n{query}\n{reasoning}"
    if answer:
        text += f"\n[ANSWER: {answer}]"
    return text


def rloo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    rewards: (n_prompts, n_samples)
    RLOO baseline: subtract mean of other samples in the same group.
    """
    n = rewards.size(1)
    group_sum = rewards.sum(dim=1, keepdim=True)
    baseline  = (group_sum - rewards) / max(n - 1, 1)
    return rewards - baseline


# ---------------------------------------------------------------------------
# Policy generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_responses(
    model, tokenizer, prompts: list[str], n_samples: int,
    max_new_tokens: int, temperature: float, device
) -> list[list[str]]:
    """
    For each prompt generate n_samples responses.
    Returns list of length len(prompts), each is list of n_samples strings.
    """
    all_responses = []
    for prompt in prompts:
        enc = tokenizer(
            prompt, return_tensors="pt",
            max_length=MAX_PROMPT_LEN, truncation=True
        ).to(device)
        outs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n_samples,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        responses = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
            for o in outs
        ]
        all_responses.append(responses)
    return all_responses


# ---------------------------------------------------------------------------
# Log-prob computation for policy gradient
# ---------------------------------------------------------------------------

def compute_logprobs(
    model, tokenizer, prompt: str, response: str, device
) -> torch.Tensor:
    """Return mean log-prob of response tokens given prompt."""
    full_text = prompt + response
    enc = tokenizer(
        full_text, return_tensors="pt",
        max_length=MAX_PROMPT_LEN + MAX_GEN_LEN, truncation=True
    ).to(device)
    prompt_enc = tokenizer(
        prompt, return_tensors="pt",
        max_length=MAX_PROMPT_LEN, truncation=True
    )
    prompt_len = prompt_enc["input_ids"].shape[1]

    with torch.no_grad() if not model.training else torch.enable_grad():
        logits = model(**enc).logits  # (1, seq, vocab)

    log_probs = F.log_softmax(logits[0], dim=-1)
    target_ids = enc["input_ids"][0, prompt_len:]
    response_logprobs = log_probs[prompt_len - 1: -1]

    if response_logprobs.shape[0] == 0 or target_ids.shape[0] == 0:
        return torch.tensor(0.0, device=device)

    n = min(response_logprobs.shape[0], target_ids.shape[0])
    gathered = response_logprobs[:n].gather(1, target_ids[:n].unsqueeze(1)).squeeze(1)
    return gathered.mean()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, tokenizer, val_dataset, device, temperature=0.0) -> float:
    model.eval()
    correct, total = 0, 0
    for item in val_dataset:
        prompt = build_prompt(item["query"], item.get("country", ""), tokenizer)
        enc = tokenizer(prompt, return_tensors="pt",
                        max_length=MAX_PROMPT_LEN, truncation=True).to(device)
        outs = model.generate(
            **enc, max_new_tokens=MAX_GEN_LEN,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        response = tokenizer.decode(outs[0][prompt_len:], skip_special_tokens=True)
        pred = extract_answer(response)
        if pred == str(item["gt"]).strip():
            correct += 1
        total += 1
    model.train()
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# DeepSpeed ZeRO-3 config
# ---------------------------------------------------------------------------

def get_ds_config(batch_size: int, micro_batch: int, lr: float) -> dict:
    return {
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": max(1, batch_size // micro_batch // int(os.environ.get("WORLD_SIZE", 1))),
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": lr, "weight_decay": 0.0, "betas": [0.9, 0.95]}
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {"warmup_min_lr": 0, "warmup_max_lr": lr,
                       "warmup_num_steps": 10, "total_num_steps": 10000}
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "none"},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
        },
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    model_path = MODEL_ALIASES.get(args.model_name, args.model_name)
    load_path = args.pretrain_path if args.pretrain_path else model_path
    if is_main:
        print(f"Base model:   {model_path}")
        print(f"Load weights: {load_path} "
              f"({'SFT checkpoint' if args.pretrain_path else 'base model'})")
        print(f"PRM path:     {args.prm_path}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ---- Policy (to be trained) ----
    policy = AutoModelForCausalLM.from_pretrained(
        load_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    # ---- Reference model (frozen, CPU-offloaded) ----
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cpu()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ---- PRM (frozen, on GPU) — v2 architecture with LoRA + mean-pool ----
    prm = CultureRewardModel(
        prm_checkpoint_dir=args.prm_path,
        base_path=PRM_BASE,
    ).to(device)
    prm_tokenizer = AutoTokenizer.from_pretrained(PRM_BASE, trust_remote_code=True)
    if prm_tokenizer.pad_token is None:
        prm_tokenizer.pad_token = prm_tokenizer.eos_token
    for p in prm.parameters():
        p.requires_grad_(False)
    prm.eval()

    # ---- DeepSpeed init ----
    ds_config = get_ds_config(
        batch_size=args.train_batch_size,
        micro_batch=args.micro_batch,
        lr=args.lr,
    )
    policy_engine, optimizer, _, _ = deepspeed.initialize(
        model=policy, config=ds_config
    )

    # ---- Datasets ----
    grpo_ds  = GRPOPromptDataset(args.grpo_data)
    val_ds   = ValDataset(args.val_data)
    loader   = DataLoader(grpo_ds, batch_size=args.prompt_batch, shuffle=True)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    no_improve   = 0

    if is_main:
        print(f"GRPO prompts: {len(grpo_ds)} | Val samples: {len(val_ds)}")
        print(f"Rounds: {args.max_rounds} | n_samples: {args.n_samples}")

    # ---- Training rounds ----
    for rnd in range(1, args.max_rounds + 1):
        round_loss = 0.0
        round_steps = 0

        for batch in loader:
            queries   = batch["query"]
            countries = batch["country"]
            golds     = batch["gt"]
            n_prompts = len(queries)

            # 1. Build prompts
            prompts = [build_prompt(q, c, tokenizer)
                       for q, c in zip(queries, countries)]

            # 2. Generate n_samples responses per prompt (policy)
            policy_engine.module.eval()
            all_responses = generate_responses(
                policy_engine.module, tokenizer, prompts,
                n_samples=args.n_samples,
                max_new_tokens=MAX_GEN_LEN,
                temperature=args.temperature,
                device=device,
            )
            policy_engine.module.train()

            # 3. Compute rewards for each (prompt, response) pair
            rewards = torch.zeros(n_prompts, args.n_samples, device=device)

            for pi, (responses, gold, query, country) in enumerate(
                zip(all_responses, golds, queries, countries)
            ):
                for si, resp in enumerate(responses):
                    # R_ans
                    pred = extract_answer(resp)
                    r_ans = compute_r_ans(pred, str(gold).strip())

                    # R_cultural via PRM (v2 format with [ANSWER: X])
                    prm_text = build_prm_input(
                        country, query, resp, pred
                    )
                    prm_enc = prm_tokenizer(
                        prm_text, return_tensors="pt",
                        max_length=2048, truncation=True,
                        padding="max_length",
                    ).to(device)
                    r_cultural = prm.score(
                        prm_enc["input_ids"], prm_enc["attention_mask"]
                    ).item()

                    rewards[pi, si] = ALPHA * r_ans + (1 - ALPHA) * r_cultural

            # 4. RLOO advantages
            advantages = rloo_advantages(rewards)

            # 5. Policy gradient loss with KL penalty
            total_loss = torch.tensor(0.0, device=device, requires_grad=False)
            loss_count = 0

            for pi, (prompt, responses) in enumerate(zip(prompts, all_responses)):
                for si, resp in enumerate(responses):
                    adv = advantages[pi, si].item()

                    # Policy log-prob
                    lp_policy = compute_logprobs(
                        policy_engine.module, tokenizer, prompt, resp, device
                    )

                    # Reference log-prob
                    ref_model_device = next(ref_model.parameters()).device
                    lp_ref = compute_logprobs(
                        ref_model, tokenizer, prompt, resp, ref_model_device
                    ).to(device)

                    kl  = (lp_policy - lp_ref).clamp(min=-10, max=10)
                    pg_loss = -(adv * lp_policy - KL_COEF * kl)

                    if loss_count == 0:
                        total_loss = pg_loss
                    else:
                        total_loss = total_loss + pg_loss
                    loss_count += 1

            if loss_count > 0:
                total_loss = total_loss / loss_count
                policy_engine.backward(total_loss)
                policy_engine.step()
                round_loss += total_loss.item()
                round_steps += 1

        avg_loss = round_loss / max(round_steps, 1)
        avg_reward = rewards.mean().item()

        if is_main:
            print(f"Round {rnd}/{args.max_rounds} | "
                  f"loss={avg_loss:.4f} | avg_reward={avg_reward:.4f}")

        # 6. Validation
        if rnd % args.eval_every == 0 and is_main:
            val_acc = validate(policy_engine.module, tokenizer, val_ds, device)
            print(f"  [Eval] Round {rnd} | val_accuracy={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                ckpt = Path(args.output_dir) / "best"
                policy_engine.module.save_pretrained(str(ckpt))
                tokenizer.save_pretrained(str(ckpt))
                print(f"  ✓ Saved best (val_acc={best_val_acc:.4f}) → {ckpt}")
            else:
                no_improve += 1
                print(f"  No improvement ({no_improve}/3)")
                if no_improve >= 3:
                    print("Early stopping: val_accuracy not improving.")
                    break

    if is_main:
        print(f"\nTraining complete. Best val_accuracy: {best_val_acc:.4f}")
        print(f"Best checkpoint: {args.output_dir}/best")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",       type=str,   required=True,
                        help="'llama', 'qwen', or full path")
    parser.add_argument("--grpo_data",        type=str,   required=True,
                        help="grpo_train.jsonl (prompts for online sampling)")
    parser.add_argument("--val_data",         type=str,   required=True,
                        help="prm_val.jsonl (validation samples)")
    parser.add_argument("--prm_path",         type=str,   required=True,
                        help="PRM v2 checkpoint dir (LoRA adapter + reward_head.pt)")
    parser.add_argument("--output_dir",       type=str,   required=True)
    parser.add_argument("--n_samples",        type=int,   default=10,
                        help="Responses per prompt per round")
    parser.add_argument("--temperature",      type=float, default=0.7)
    parser.add_argument("--max_rounds",       type=int,   default=30)
    parser.add_argument("--eval_every",       type=int,   default=5)
    parser.add_argument("--prompt_batch",     type=int,   default=4,
                        help="Prompts per optimizer step")
    parser.add_argument("--train_batch_size", type=int,   default=8)
    parser.add_argument("--micro_batch",      type=int,   default=2)
    parser.add_argument("--lr",               type=float, default=5e-7)
    parser.add_argument("--pretrain_path",    type=str,   default=None,
                        help="SFT checkpoint dir for SFT+RL mode. "
                             "Leave empty for RL-only (starts from base model).")
    # DeepSpeed launcher injects --local_rank automatically
    parser.add_argument("--local_rank",       type=int,   default=0)
    args = parser.parse_args()

    args.model_name = MODEL_ALIASES.get(args.model_name, args.model_name)
    train(args)


if __name__ == "__main__":
    main()
