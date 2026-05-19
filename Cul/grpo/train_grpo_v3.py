"""
CAMA-D Stage 3-GRPO: GRPO with Mean(R_process) Reward

Key innovations over old GRPO (train_grpo.py):
  - R_total = alpha * R_outcome + (1-alpha) * Mean(R_process)
  - PRM uses step-level scoring (Sigmoid at each [Step N] terminator)
  - Mean(R_process) ∈ [0.1, 0.9] — perfectly aligned with R_outcome ∈ {0, 1}
  - Heuristic step splitting before PRM scoring (same rules as Stage 2)
  - alpha=0.6 (outcome-dominant, process as soft constraint)

Usage:
    # RL-only (from base model)
    deepspeed --num_gpus 2 Cul/grpo/train_grpo_v3.py \\
        --model_name     qwen \\
        --grpo_data      /path/to/grpo_train.jsonl \\
        --val_data       /path/to/prm_val.jsonl \\
        --prm_path       /path/to/camad_prm/best \\
        --prm_backbone   /path/to/camad_sft_qwen/best \\
        --output_dir     /path/to/models/camad_grpo_qwen \\
        --alpha          0.6 \\
        --n_samples      10 \\
        --max_rounds     30 \\
        --eval_every     5

    # SFT+RL (from Stage 1 SFT checkpoint — recommended)
    deepspeed --num_gpus 2 Cul/grpo/train_grpo_v3.py \\
        --model_name     qwen \\
        --pretrain_path  /path/to/camad_sft_qwen/best \\
        --grpo_data      /path/to/grpo_train.jsonl \\
        --val_data       /path/to/prm_val.jsonl \\
        --prm_path       /path/to/camad_prm/best \\
        --prm_backbone   /path/to/camad_sft_qwen/best \\
        --output_dir     /path/to/models/camad_grpo_sft_qwen \\
        --alpha          0.6 \\
        --n_samples      10 \\
        --max_rounds     20 \\
        --eval_every     5
"""

import re
import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed

# Import step splitting utility
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step_label.split_steps import split_reasoning_into_steps


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}

MAX_GEN_LEN = 512
MAX_PROMPT_LEN = 512
KL_COEF = 0.05       # KL penalty coefficient (higher than old version)
DEFAULT_ALPHA = 0.6   # R_outcome weight


# ---------------------------------------------------------------------------
# PRM v3: Step-level scoring with Sigmoid (matches train_prm_mse.py)
# ---------------------------------------------------------------------------

class CulturePRM_v3(nn.Module):
    """
    Process Reward Model for GRPO scoring.

    Loads the trained PRM (SFT backbone + LoRA + score_head) and scores
    each step in a reasoning path. Returns Mean(scores) as R_process.

    Architecture matches train_prm_mse.py:
      - SFT backbone with LoRA adapter
      - score_head: Linear(hidden, 1) + Sigmoid → (0, 1)
      - Scoring at [Step N] terminator positions
    """

    def __init__(self, prm_checkpoint_dir: str, sft_backbone_path: str):
        super().__init__()
        from peft import PeftModel

        # Load SFT backbone
        base_model = AutoModelForCausalLM.from_pretrained(
            sft_backbone_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            output_hidden_states=True,
        )

        # Load LoRA adapter
        adapter_path = Path(prm_checkpoint_dir) / "adapter_model.safetensors"
        adapter_path_bin = Path(prm_checkpoint_dir) / "adapter_model.bin"
        if adapter_path.exists() or adapter_path_bin.exists():
            self.model = PeftModel.from_pretrained(base_model, prm_checkpoint_dir)
            print(f"  [PRM-v3] Loaded LoRA adapter from {prm_checkpoint_dir}")
        else:
            self.model = base_model
            print(f"  [PRM-v3] No adapter found, using backbone directly")

        # Load score_head
        hidden_size = base_model.config.hidden_size
        self.score_head = nn.Linear(hidden_size, 1)
        head_path = Path(prm_checkpoint_dir) / "score_head.pt"
        if head_path.exists():
            state = torch.load(head_path, map_location="cpu")
            self.score_head.load_state_dict(state)
            print(f"  [PRM-v3] Loaded score_head from {head_path}")
        else:
            print(f"  [PRM-v3] WARNING: score_head.pt not found at {head_path}")

        self.sigmoid = nn.Sigmoid()

    @torch.no_grad()
    def score_reasoning(
        self,
        input_text: str,
        tokenizer,
        device,
        max_len: int = 2048,
    ) -> float:
        """
        Score a full reasoning path and return Mean(R_process).

        Steps:
          1. Split reasoning into [Step N] segments using heuristic rules
          2. Tokenize the full text with step markers
          3. Find step terminator positions
          4. Score each step via backbone + score_head + Sigmoid
          5. Return mean of all step scores

        Args:
            input_text: Full formatted text "[country]\\nquestion\\n[Step 1]...\\n[Step 2]..."
            tokenizer: Tokenizer for the PRM backbone
            device: torch device

        Returns:
            Mean step score ∈ [0.1, 0.9] (clamped)
        """
        enc = tokenizer(
            input_text,
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Find [Step N] terminator positions
        step_positions = self._find_step_positions(input_text, tokenizer, max_len)

        if not step_positions:
            # No steps found — return neutral score
            return 0.5

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (1, seq_len, hidden)

        scores = []
        seq_len = hidden_states.size(1)
        for pos in step_positions:
            if pos >= seq_len:
                pos = seq_len - 1
            h = hidden_states[0, pos, :]
            logit = self.score_head(h.float()).squeeze(-1)
            score = self.sigmoid(logit).item()
            scores.append(score)

        mean_score = sum(scores) / len(scores)
        # Clamp to [0.1, 0.9] as designed
        return max(0.1, min(0.9, mean_score))

    def _find_step_positions(self, text: str, tokenizer, max_len: int) -> list[int]:
        """Find token positions of each step's last token."""
        import re as _re

        step_starts = [m.start() for m in _re.finditer(r'\[Step \d+\]', text)]
        if not step_starts:
            return []

        enc = tokenizer(
            text, max_length=max_len, truncation=True,
            return_offsets_mapping=True, add_special_tokens=True,
        )
        offsets = enc["offset_mapping"]

        end_positions = []
        for i, start_char in enumerate(step_starts):
            if i + 1 < len(step_starts):
                end_char = step_starts[i + 1] - 1
            else:
                end_char = len(text) - 1

            end_tok_pos = 0
            for tok_idx, (s, e) in enumerate(offsets):
                if e > 0 and s <= end_char:
                    end_tok_pos = tok_idx

            end_positions.append(end_tok_pos)

        return end_positions


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
                "query": obj["query"],
                "country": obj.get("country", ""),
                "gt": str(obj["gt"]).strip(),
            })

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


def extract_answer(text: str):
    """Extract answer from generated response."""
    m = re.search(r"Answer\s*:\s*([1-4])", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"answer\s+is\s*:?\s*([1-4])\b", text, re.IGNORECASE)
    if m:
        return m.group(1)
    digits = re.findall(r"\b([1-4])\b", text)
    return digits[-1] if digits else None


def compute_r_outcome(predicted, gold: str) -> float:
    """Binary outcome reward."""
    return 1.0 if predicted == gold else 0.0


def build_prm_input(country: str, query: str, reasoning: str) -> str:
    """
    Build PRM input text with step markers.

    Steps:
      1. Split reasoning into steps using heuristic rules
      2. Format as: [country]\\nquestion\\n[Step 1]...\\n[Step 2]...
    """
    steps = split_reasoning_into_steps(reasoning)
    if not steps:
        # If splitting fails, wrap entire reasoning as single step
        steps = [f"[Step 1] {reasoning}"]

    step_text = "\n".join(steps)
    return f"[{country}]\n{query}\n{step_text}"


def rloo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    RLOO (Reinforce Leave-One-Out) advantage estimation.

    rewards: (n_prompts, n_samples)
    baseline = mean of other samples in the group.
    """
    n = rewards.size(1)
    group_sum = rewards.sum(dim=1, keepdim=True)
    baseline = (group_sum - rewards) / max(n - 1, 1)
    return rewards - baseline


# ---------------------------------------------------------------------------
# Policy generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_responses(
    model, tokenizer, prompts: list, n_samples: int,
    max_new_tokens: int, temperature: float, device
) -> list:
    """Generate n_samples responses per prompt."""
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
# Log-prob computation
# ---------------------------------------------------------------------------

def compute_logprobs(model, tokenizer, prompt: str, response: str, device):
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
        logits = model(**enc).logits

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
def validate(model, tokenizer, val_path: str, device, max_samples: int = 200) -> float:
    """Compute validation accuracy."""
    model.eval()
    correct, total = 0, 0
    for line in open(val_path, encoding="utf-8"):
        if total >= max_samples:
            break
        obj = json.loads(line)
        prompt = build_prompt(obj["query"], obj.get("country", ""), tokenizer)
        enc = tokenizer(
            prompt, return_tensors="pt",
            max_length=MAX_PROMPT_LEN, truncation=True
        ).to(device)
        outs = model.generate(
            **enc, max_new_tokens=MAX_GEN_LEN,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        response = tokenizer.decode(outs[0][prompt_len:], skip_special_tokens=True)
        pred = extract_answer(response)
        if pred == str(obj["gt"]).strip():
            correct += 1
        total += 1
    model.train()
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# DeepSpeed config
# ---------------------------------------------------------------------------

def get_ds_config(batch_size: int, micro_batch: int, lr: float) -> dict:
    return {
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": max(
            1, batch_size // micro_batch // int(os.environ.get("WORLD_SIZE", 1))
        ),
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": lr, "weight_decay": 0.0, "betas": [0.9, 0.95]}
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0, "warmup_max_lr": lr,
                "warmup_num_steps": 10, "total_num_steps": 10000
            }
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
        print(f"=" * 60)
        print(f"CAMA-D GRPO v3: Mean(R_process) Reward")
        print(f"=" * 60)
        print(f"Base model:   {model_path}")
        print(f"Load weights: {load_path} "
              f"({'SFT checkpoint' if args.pretrain_path else 'base model'})")
        print(f"PRM path:     {args.prm_path}")
        print(f"PRM backbone: {args.prm_backbone}")
        print(f"Alpha:        {args.alpha} "
              f"(R_total = {args.alpha}*R_outcome + {1-args.alpha:.1f}*Mean(R_process))")
        print(f"KL coef:      {KL_COEF}")

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

    # ---- PRM v3 (frozen, on GPU) ----
    prm = CulturePRM_v3(
        prm_checkpoint_dir=args.prm_path,
        sft_backbone_path=args.prm_backbone,
    ).to(device)
    prm_tokenizer = AutoTokenizer.from_pretrained(
        args.prm_backbone, trust_remote_code=True
    )
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

    # ---- Dataset ----
    grpo_ds = GRPOPromptDataset(args.grpo_data)
    loader = DataLoader(grpo_ds, batch_size=args.prompt_batch, shuffle=True)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    no_improve = 0

    if is_main:
        print(f"GRPO prompts: {len(grpo_ds)} | "
              f"Rounds: {args.max_rounds} | n_samples: {args.n_samples}")

    # ---- Training rounds ----
    for rnd in range(1, args.max_rounds + 1):
        round_loss = 0.0
        round_steps = 0
        round_r_outcome = 0.0
        round_r_process = 0.0
        round_r_total = 0.0
        round_n = 0

        for batch in loader:
            queries = batch["query"]
            countries = batch["country"]
            golds = batch["gt"]
            n_prompts = len(queries)

            # 1. Build prompts
            prompts = [build_prompt(q, c, tokenizer)
                       for q, c in zip(queries, countries)]

            # 2. Generate n_samples responses per prompt
            policy_engine.module.eval()
            all_responses = generate_responses(
                policy_engine.module, tokenizer, prompts,
                n_samples=args.n_samples,
                max_new_tokens=MAX_GEN_LEN,
                temperature=args.temperature,
                device=device,
            )
            policy_engine.module.train()

            # 3. Compute rewards
            rewards = torch.zeros(n_prompts, args.n_samples, device=device)

            for pi, (responses, gold, query, country) in enumerate(
                zip(all_responses, golds, queries, countries)
            ):
                for si, resp in enumerate(responses):
                    # R_outcome: binary correctness
                    pred = extract_answer(resp)
                    r_outcome = compute_r_outcome(pred, str(gold).strip())

                    # R_process: Mean(PRM step scores)
                    prm_input = build_prm_input(country, query, resp)
                    r_process = prm.score_reasoning(
                        prm_input, prm_tokenizer, device
                    )

                    # R_total = alpha * R_outcome + (1-alpha) * Mean(R_process)
                    r_total = args.alpha * r_outcome + (1 - args.alpha) * r_process
                    rewards[pi, si] = r_total

                    # Tracking
                    round_r_outcome += r_outcome
                    round_r_process += r_process
                    round_r_total += r_total
                    round_n += 1

            # 4. RLOO advantages
            advantages = rloo_advantages(rewards)

            # 5. Policy gradient with KL penalty
            total_loss = torch.tensor(0.0, device=device, requires_grad=False)
            loss_count = 0

            for pi, (prompt, responses) in enumerate(zip(prompts, all_responses)):
                for si, resp in enumerate(responses):
                    adv = advantages[pi, si].item()

                    # Policy log-prob
                    lp_policy = compute_logprobs(
                        policy_engine.module, tokenizer, prompt, resp, device
                    )

                    # Reference log-prob (on CPU)
                    ref_device = next(ref_model.parameters()).device
                    lp_ref = compute_logprobs(
                        ref_model, tokenizer, prompt, resp, ref_device
                    ).to(device)

                    # KL penalty
                    kl = (lp_policy - lp_ref).clamp(min=-10, max=10)
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

        # Round summary
        avg_loss = round_loss / max(round_steps, 1)
        avg_r_outcome = round_r_outcome / max(round_n, 1)
        avg_r_process = round_r_process / max(round_n, 1)
        avg_r_total = round_r_total / max(round_n, 1)

        if is_main:
            print(f"Round {rnd}/{args.max_rounds} | "
                  f"loss={avg_loss:.4f} | "
                  f"R_outcome={avg_r_outcome:.3f} | "
                  f"R_process={avg_r_process:.3f} | "
                  f"R_total={avg_r_total:.3f}")

        # 6. Validation
        if rnd % args.eval_every == 0 and is_main:
            val_acc = validate(
                policy_engine.module, tokenizer, args.val_data, device
            )
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
    parser = argparse.ArgumentParser(
        description="CAMA-D Stage 3-GRPO: GRPO with Mean(R_process)"
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="'llama', 'qwen', or full model path")
    parser.add_argument("--grpo_data", type=str, required=True,
                        help="GRPO training JSONL (prompts for online sampling)")
    parser.add_argument("--val_data", type=str, required=True,
                        help="Validation JSONL for accuracy evaluation")
    parser.add_argument("--prm_path", type=str, required=True,
                        help="CAMA-D PRM checkpoint dir (LoRA + score_head.pt)")
    parser.add_argument("--prm_backbone", type=str, required=True,
                        help="PRM backbone model path (Stage 1 SFT model)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model checkpoints")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="R_outcome weight in R_total (default: 0.6)")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Responses per prompt per round (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max_rounds", type=int, default=30,
                        help="Maximum training rounds (default: 30)")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate every N rounds (default: 5)")
    parser.add_argument("--prompt_batch", type=int, default=4,
                        help="Prompts per optimizer step (default: 4)")
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="DeepSpeed train batch size (default: 8)")
    parser.add_argument("--micro_batch", type=int, default=2,
                        help="Micro batch per GPU (default: 2)")
    parser.add_argument("--lr", type=float, default=5e-7,
                        help="Learning rate (default: 5e-7 for RL-only, "
                             "use 1e-7 for SFT+RL)")
    parser.add_argument("--pretrain_path", type=str, default=None,
                        help="SFT checkpoint for SFT+RL mode. "
                             "Leave empty for RL-only.")
    # DeepSpeed injects --local_rank
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    args.model_name = MODEL_ALIASES.get(args.model_name, args.model_name)
    train(args)


if __name__ == "__main__":
    main()
