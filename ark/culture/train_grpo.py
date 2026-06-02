"""
AgentArk Baseline — Stage 3: GRPO Training

Supports two data sources via --data_source parameter:
  - "reconcile": Use RECONCILE multi-agent debate data (homogeneous agents)
  - "hf_cac":   Use HF-CAC data (heterogeneous agents, for cross-comparison)

Key differences from CAMA-D GRPO:
  - R_total = alpha * R_outcome + (1-alpha) * Mean(R_process)
  - BUT no cultural authority mechanism in PRM scoring
  - Standard (non-weighted) step splitting and labeling
  - Homogeneous agent debate → SFT → PRM → GRPO pipeline

Architecture (LoRA, no DeepSpeed):
  - Policy: base model + SFT-LoRA merged + new GRPO-LoRA (trainable)
  - Reference: same model with adapter disabled (zero extra memory)
  - PRM: loaded separately for scoring (optional, can run without PRM)
  - Gradient checkpointing enabled

Hardware requirement: 2×vGPU-48GB recommended (policy on cuda:0, PRM on cuda:1)

Usage:
    # With RECONCILE data source (AgentArk default)
    python ark/culture/train_grpo.py \\
        --model_name     qwen \\
        --data_source    reconcile \\
        --sft_adapter    /path/to/agentark_sft_qwen/best \\
        --data_pkl       /path/to/normad_agentark_splits.pkl \\
        --prm_path       /path/to/agentark_prm/best \\
        --output_dir     /path/to/models/agentark_grpo_qwen \\
        --alpha          0.6 \\
        --n_samples      5

    # With HF-CAC data source (cross-comparison baseline)
    python ark/culture/train_grpo.py \\
        --model_name     qwen \\
        --data_source    hf_cac \\
        --sft_adapter    /path/to/agentark_sft_qwen/best \\
        --data_pkl       /path/to/normad_hfcac_splits.pkl \\
        --prm_path       /path/to/agentark_prm/best \\
        --output_dir     /path/to/models/agentark_grpo_hfcac_qwen \\
        --alpha          0.6

    # Fast validation (no PRM, outcome-only reward)
    python ark/culture/train_grpo.py \\
        --model_name     qwen \\
        --data_source    reconcile \\
        --sft_adapter    /path/to/agentark_sft_qwen/best \\
        --data_pkl       /path/to/normad_agentark_splits.pkl \\
        --output_dir     /path/to/models/agentark_grpo_qwen \\
        --no_prm \\
        --max_rounds     10
"""

import re
import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# Import step splitting utility (same as CAMA-D)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'Cul'))

try:
    from step_label.split_steps import split_reasoning_into_steps
except ImportError:
    # Fallback: inline simple splitting
    def split_reasoning_into_steps(text, max_sentences_per_step=3):
        if not text or not text.strip():
            return []
        segments = [s.strip() for s in re.split(r'\n\n|\n', text.strip()) if s.strip()]
        return [f"[Step {i+1}] {seg}" for i, seg in enumerate(segments)]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}

MAX_GEN_LEN = 128
MAX_PROMPT_LEN = 512
KL_COEF = 0.05
DEFAULT_ALPHA = 0.6


# ---------------------------------------------------------------------------
# PRM: AgentArk PRM for GRPO scoring (same architecture as train_prm.py)
# ---------------------------------------------------------------------------

class AgentArkPRM_Scorer(nn.Module):
    """
    PRM scorer for GRPO training.
    Loads trained PRM checkpoint and scores reasoning paths.
    """

    def __init__(self, prm_checkpoint_dir: str, backbone_path: str,
                 sft_adapter_path: str = None):
        super().__init__()

        base_model = AutoModelForCausalLM.from_pretrained(
            backbone_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            output_hidden_states=True,
        )

        if sft_adapter_path:
            base_model = PeftModel.from_pretrained(base_model, sft_adapter_path)
            base_model = base_model.merge_and_unload()

        # Load PRM LoRA adapter
        adapter_path = Path(prm_checkpoint_dir) / "adapter_model.safetensors"
        adapter_path_bin = Path(prm_checkpoint_dir) / "adapter_model.bin"
        if adapter_path.exists() or adapter_path_bin.exists():
            self.model = PeftModel.from_pretrained(base_model, prm_checkpoint_dir)
            print(f"  [PRM] Loaded PRM LoRA from {prm_checkpoint_dir}")
        else:
            self.model = base_model
            print(f"  [PRM] No adapter found, using backbone directly")

        # Load score head
        hidden_size = base_model.config.hidden_size
        self.score_head = nn.Linear(hidden_size, 1)
        head_path = Path(prm_checkpoint_dir) / "score_head.pt"
        if head_path.exists():
            state = torch.load(head_path, map_location="cpu")
            self.score_head.load_state_dict(state)
            print(f"  [PRM] Loaded score_head from {head_path}")
        else:
            print(f"  [PRM] WARNING: score_head.pt not found at {head_path}")

        self.sigmoid = nn.Sigmoid()

    @torch.no_grad()
    def score_batch(
        self,
        input_texts: list[str],
        tokenizer,
        device,
        max_len: int = 512,
    ) -> list[float]:
        """Batch-score reasoning paths. Returns Mean(step_scores) per input."""
        if not input_texts:
            return []

        all_step_positions = []
        valid_indices = []
        for i, text in enumerate(input_texts):
            positions = self._find_step_positions(text, tokenizer, max_len)
            all_step_positions.append(positions)
            if positions:
                valid_indices.append(i)

        if not valid_indices:
            return [0.5] * len(input_texts)

        valid_texts = [input_texts[i] for i in valid_indices]
        enc = tokenizer(
            valid_texts,
            max_length=max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]

        results = [0.5] * len(input_texts)
        seq_len = hidden_states.size(1)

        for batch_pos, orig_idx in enumerate(valid_indices):
            step_positions = all_step_positions[orig_idx]
            scores = []
            for pos in step_positions:
                if pos >= seq_len:
                    pos = seq_len - 1
                h = hidden_states[batch_pos, pos, :]
                logit = self.score_head(h.float()).squeeze(-1)
                score = self.sigmoid(logit).item()
                scores.append(score)
            if scores:
                mean_score = sum(scores) / len(scores)
                results[orig_idx] = max(0.1, min(0.9, mean_score))

        return results

    def _find_step_positions(self, text: str, tokenizer, max_len: int) -> list[int]:
        """Find token positions of each step's last token."""
        step_starts = [m.start() for m in re.finditer(r'\[Step \d+\]', text)]
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
    def __init__(self, samples: list[dict]):
        self.items = []
        for obj in samples:
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
    """
    Build prompt for generation.

    AgentArk baseline: NO country prefix (culture-agnostic policy).
    The question is given without cultural conditioning.
    """
    content = f"{query}\n\nReasoning: "
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


def build_prm_input(query: str, reasoning: str) -> str:
    """
    Build PRM input text with step markers.

    AgentArk: no country prefix (culture-agnostic scoring).
    """
    steps = split_reasoning_into_steps(reasoning)
    if not steps:
        steps = [f"[Step 1] {reasoning}"]
    step_text = "\n".join(steps)
    return f"{query}\n{step_text}"


def rloo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """RLOO (Reinforce Leave-One-Out) advantage estimation."""
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

def compute_logprobs(model, tokenizer, prompt: str, response: str, device,
                     require_grad: bool = False):
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

    if require_grad:
        logits = model(**enc).logits
    else:
        with torch.no_grad():
            logits = model(**enc).logits

    target_ids = enc["input_ids"][0, prompt_len:]
    resp_logits = logits[0, prompt_len - 1: -1, :]

    if resp_logits.shape[0] == 0 or target_ids.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=require_grad)

    n = min(resp_logits.shape[0], target_ids.shape[0])
    resp_logits = resp_logits[:n]
    target_ids = target_ids[:n]

    log_probs = F.log_softmax(resp_logits, dim=-1)
    gathered = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return gathered.mean()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, tokenizer, val_samples: list[dict], device,
             max_samples: int = 200) -> float:
    """Compute validation accuracy."""
    model.eval()
    correct, total = 0, 0
    for obj in val_samples:
        if total >= max_samples:
            break
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
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    if not args.no_prm and args.prm_path is None:
        raise ValueError("--prm_path required unless --no_prm is set.")

    policy_device = torch.device("cuda:0")
    prm_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

    model_path = MODEL_ALIASES.get(args.model_name, args.model_name)

    print(f"{'='*60}")
    print(f"AgentArk GRPO — data_source={args.data_source}")
    print(f"{'='*60}")
    print(f"Base model:    {model_path}")
    print(f"SFT adapter:   {args.sft_adapter or 'None (RL-only)'}")
    print(f"Data source:   {args.data_source}")
    print(f"PRM path:      {args.prm_path}")
    print(f"Alpha:         {args.alpha} "
          f"(R_total = {args.alpha}*R_outcome + {1-args.alpha:.1f}*Mean(R_process))")
    print(f"KL coef:       {KL_COEF}")
    print(f"LoRA rank:     {args.lora_r}")

    # Load data
    print(f"Loading data splits from: {args.data_pkl}")
    with open(args.data_pkl, "rb") as f:
        splits = pickle.load(f)
    train_data = splits["train"]
    val_data = splits["val"]
    print(f"  Data ({args.data_source}): train={len(train_data)}, "
          f"val={len(val_data)}, test={len(splits['test'])}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Build policy
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    if args.sft_adapter:
        print(f"  Merging SFT LoRA adapter: {args.sft_adapter}")
        base_model = PeftModel.from_pretrained(base_model, args.sft_adapter)
        base_model = base_model.merge_and_unload()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    policy = get_peft_model(base_model, lora_config).to(policy_device)
    policy.print_trainable_parameters()
    policy.enable_input_require_grads()
    policy.gradient_checkpointing_enable()

    # PRM
    if not args.no_prm:
        prm_backbone = args.prm_backbone if args.prm_backbone else model_path
        prm = AgentArkPRM_Scorer(
            prm_checkpoint_dir=args.prm_path,
            backbone_path=prm_backbone,
            sft_adapter_path=args.sft_adapter,
        ).to(prm_device)
        prm_tokenizer = AutoTokenizer.from_pretrained(
            prm_backbone, trust_remote_code=True
        )
        if prm_tokenizer.pad_token is None:
            prm_tokenizer.pad_token = prm_tokenizer.eos_token
        for p in prm.parameters():
            p.requires_grad_(False)
        prm.eval()
    else:
        prm = None
        prm_tokenizer = None
        print("  [PRM] Skipped (--no_prm mode)")

    # Optimizer
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95)
    )

    # Dataset
    grpo_ds = GRPOPromptDataset(train_data)
    loader = DataLoader(grpo_ds, batch_size=args.prompt_batch, shuffle=True)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    no_improve = 0

    effective_batches = (len(loader) if args.batches_per_round < 0
                         else min(args.batches_per_round, len(loader)))
    print(f"GRPO prompts: {len(grpo_ds)} | Rounds: {args.max_rounds} | "
          f"n_samples: {args.n_samples}")
    print(f"Batches per round: {effective_batches}/{len(loader)}")
    print(f"PRM scoring: {'DISABLED' if args.no_prm else 'ENABLED'}")
    print(f"Starting training...\n")
    sys.stdout.flush()

    # Training rounds
    for rnd in range(1, args.max_rounds + 1):
        round_loss = 0.0
        round_steps = 0
        round_r_outcome = 0.0
        round_r_process = 0.0
        round_r_total = 0.0
        round_n = 0

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= effective_batches:
                break

            queries = batch["query"]
            countries = batch["country"]
            golds = batch["gt"]
            n_prompts = len(queries)

            if batch_idx % 5 == 0:
                print(f"  Round {rnd} | Batch {batch_idx+1}/{effective_batches} | "
                      f"Generating...", flush=True)

            # Build prompts
            prompts = [build_prompt(q, c, tokenizer)
                       for q, c in zip(queries, countries)]

            # Generate responses
            policy.eval()
            policy.gradient_checkpointing_disable()
            all_responses = generate_responses(
                policy, tokenizer, prompts,
                n_samples=args.n_samples,
                max_new_tokens=MAX_GEN_LEN,
                temperature=args.temperature,
                device=policy_device,
            )
            policy.gradient_checkpointing_enable()
            policy.train()
            torch.cuda.empty_cache()

            # Compute rewards
            rewards = torch.zeros(n_prompts, args.n_samples, device=policy_device)

            r_outcomes = []
            for pi, (responses, gold) in enumerate(zip(all_responses, golds)):
                for si, resp in enumerate(responses):
                    pred = extract_answer(resp)
                    r_outcomes.append(compute_r_outcome(pred, str(gold).strip()))

            # PRM scoring
            if args.no_prm:
                r_processes = [0.0] * (n_prompts * args.n_samples)
            else:
                prm_inputs = []
                for pi, (responses, query) in enumerate(
                    zip(all_responses, queries)
                ):
                    for si, resp in enumerate(responses):
                        prm_inputs.append(build_prm_input(query, resp))

                PRM_BATCH_SIZE = 20
                r_processes = []
                for i in range(0, len(prm_inputs), PRM_BATCH_SIZE):
                    batch_texts = prm_inputs[i:i + PRM_BATCH_SIZE]
                    batch_scores = prm.score_batch(
                        batch_texts, prm_tokenizer, prm_device, max_len=512
                    )
                    r_processes.extend(batch_scores)

            # Combine rewards
            idx = 0
            for pi in range(n_prompts):
                for si in range(args.n_samples):
                    r_outcome = r_outcomes[idx]
                    r_process = r_processes[idx]
                    if args.no_prm:
                        r_total = r_outcome
                    else:
                        r_total = args.alpha * r_outcome + (1 - args.alpha) * r_process
                    rewards[pi, si] = r_total
                    round_r_outcome += r_outcome
                    round_r_process += r_process
                    round_r_total += r_total
                    round_n += 1
                    idx += 1

            # RLOO advantages
            advantages = rloo_advantages(rewards)

            # Phase A: Reference log-probs
            policy.eval()
            policy.gradient_checkpointing_disable()
            ref_logprobs = {}
            with torch.no_grad():
                with policy.disable_adapter():
                    for pi, (prompt, responses) in enumerate(zip(prompts, all_responses)):
                        for si, resp in enumerate(responses):
                            adv = advantages[pi, si].item()
                            if abs(adv) < 1e-8:
                                continue
                            lp_ref = compute_logprobs(
                                policy, tokenizer, prompt, resp, policy_device,
                                require_grad=False,
                            )
                            ref_logprobs[(pi, si)] = lp_ref.item()

            # Phase B: Policy gradient
            policy.gradient_checkpointing_enable()
            policy.train()
            optimizer.zero_grad()
            loss_count = 0
            accumulated_loss = 0.0
            total_samples = n_prompts * args.n_samples

            for pi, (prompt, responses) in enumerate(zip(prompts, all_responses)):
                for si, resp in enumerate(responses):
                    if (pi, si) not in ref_logprobs:
                        continue

                    adv = advantages[pi, si].item()
                    lp_ref_val = ref_logprobs[(pi, si)]

                    lp_policy = compute_logprobs(
                        policy, tokenizer, prompt, resp, policy_device,
                        require_grad=True,
                    )

                    lp_ref_t = torch.tensor(lp_ref_val, device=policy_device)
                    kl = (lp_policy - lp_ref_t).clamp(min=-10, max=10)
                    pg_loss = -(adv * lp_policy - KL_COEF * kl) / total_samples

                    pg_loss.backward()
                    accumulated_loss += pg_loss.item()
                    loss_count += 1

            if loss_count > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                round_loss += accumulated_loss
                round_steps += 1

        # Round summary
        avg_loss = round_loss / max(round_steps, 1)
        avg_r_outcome = round_r_outcome / max(round_n, 1)
        avg_r_process = round_r_process / max(round_n, 1)
        avg_r_total = round_r_total / max(round_n, 1)

        print(f"Round {rnd}/{args.max_rounds} | "
              f"loss={avg_loss:.4f} | "
              f"R_outcome={avg_r_outcome:.3f} | "
              f"R_process={avg_r_process:.3f} | "
              f"R_total={avg_r_total:.3f}")

        # Validation
        if rnd % args.eval_every == 0:
            val_acc = validate(policy, tokenizer, val_data, policy_device)
            print(f"  [Eval] Round {rnd} | val_accuracy={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                ckpt = Path(args.output_dir) / "best"
                policy.save_pretrained(str(ckpt))
                tokenizer.save_pretrained(str(ckpt))
                print(f"  Saved best GRPO LoRA (val_acc={best_val_acc:.4f}) -> {ckpt}")
            else:
                no_improve += 1
                print(f"  No improvement ({no_improve}/3)")
                if no_improve >= 3:
                    print("Early stopping.")
                    break

    if best_val_acc == 0.0:
        ckpt = Path(args.output_dir) / "best"
        policy.save_pretrained(str(ckpt))
        tokenizer.save_pretrained(str(ckpt))
        print(f"  Saved final GRPO LoRA -> {ckpt}")

    print(f"\nTraining complete. Best val_accuracy: {best_val_acc:.4f}")
    print(f"Data source: {args.data_source}")
    print(f"Best GRPO LoRA: {args.output_dir}/best")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AgentArk Baseline — Stage 3: GRPO Training"
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="'llama', 'qwen', or full model path")
    parser.add_argument("--data_source", type=str, required=True,
                        choices=["reconcile", "hf_cac"],
                        help="Data source: 'reconcile' (homogeneous debate) or "
                             "'hf_cac' (heterogeneous HF-CAC data)")
    parser.add_argument("--sft_adapter", type=str, default=None,
                        help="SFT LoRA adapter path")
    parser.add_argument("--data_pkl", type=str, required=True,
                        help="Path to splits pkl file")
    parser.add_argument("--prm_path", type=str, default=None,
                        help="PRM checkpoint dir")
    parser.add_argument("--prm_backbone", type=str, default=None,
                        help="PRM backbone model path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="R_outcome weight (default: 0.6)")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Responses per prompt (default: 5)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_rounds", type=int, default=30)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--prompt_batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--batches_per_round", type=int, default=130)
    parser.add_argument("--no_prm", action="store_true",
                        help="Skip PRM, use outcome-only reward")
    args = parser.parse_args()

    args.model_name = MODEL_ALIASES.get(args.model_name, args.model_name)
    train(args)


if __name__ == "__main__":
    main()
