"""
Single-Teacher Distillation SFT (单 teacher 蒸馏 SFT) — LoRA 版本

核心思想（与普通 SFT 的本质区别）：
  - 监督信号 = 大模型 role-play 方法生成的输出（response），而非数据集原始 label（gt）。
  - 即把"角色扮演大模型"当作 single teacher，将其 role-play 产出的完整回答
    （包含推理过程 + 'Answer: X'）作为目标序列，蒸馏到基座学生模型上。
  - 这样学生模型不仅学会答案，还学会 teacher 的文化推理风格。

数据划分（train / val / test）：
  统一复用 Cul/split_data.py 产出的 pkl（按 8:1:1 划分，保留所有原始字段）：
    python Cul/split_data.py \\
        --input  /autodl-fs/data/blend_llama_role_20260610_112253.json \\
        --output /autodl-fs/data/blend_llama_splits.pkl \\
        --seed 42
  - train : 用于蒸馏训练（以 teacher 的 response 作为目标）
  - val   : 每个 epoch 结束后做生成式评估，按数据集原始 gt 计算准确率，
            据此保存最优 LoRA adapter 并触发早停。
  - test  : 本脚本不使用；最终测试由 Cul/evaluate.py 在 pkl 的 test 上完成。

监督目标的构造：
  input  (user)      : [{country}]\n{query}
  output (assistant) : teacher 的 response（原样作为蒸馏目标）

验证 / 早停：
  - 每个 epoch 结束在 val 上生成回答，抽取答案并与原始 gt 比对得到 accuracy。
  - 验证准确率创新高 → 保存到 {output_dir}/best（仅 LoRA adapter）。
  - 连续 --patience（默认 2）个 epoch 验证准确率不再提升 → 早停。

样本过滤（可选，仅作用于训练集）：
  --filter_correct   仅保留 teacher 答对（pred == gt）的样本做蒸馏（rejection sampling）。
                     默认 False —— 严格按"用 role-play 输出作为 label"全量蒸馏。
  --drop_no_pred     丢弃无法抽取答案（pred 为 null/空）的样本。默认 True。

LoRA 说明：
  - 本脚本为 LoRA 微调，只训练并保存 LoRA adapter 参数（adapter_model.safetensors +
    adapter_config.json，约几十~几百 MB），不保存 14GB 基座权重。
  - 评估 / 推理时由 evaluate.py 用 "基座 + PeftModel.from_pretrained(adapter)" 还原。

Usage (single GPU):
    python Cul/sft/train_single_teacher_distill.py \\
        --model_name llama \\
        --data_pkl   /autodl-fs/data/blend_llama_splits.pkl \\
        --output_dir /root/autodl-tmp/models/distill_single_llama \\
        --epochs 5 --batch_size 4 --lr 2e-4 --lora_r 32 --patience 2

Usage (multi-GPU DDP via accelerate):
    accelerate launch --num_processes 2 Cul/sft/train_single_teacher_distill.py \\
        --model_name llama \\
        --data_pkl   /autodl-fs/data/blend_llama_splits.pkl \\
        --output_dir /root/autodl-tmp/models/distill_single_llama \\
        --epochs 5 --batch_size 4 --lr 2e-4 --lora_r 32 --patience 2
"""

import re
import pickle
import argparse
from pathlib import Path
from functools import partial
from datetime import timedelta

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed, broadcast


MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}

MAX_SEQ_LEN = 2048
IGNORE_INDEX = -100

# 与 single_data.py / evaluate.py 中保持一致的系统提示，保证训练 / 推理格式统一。
SYSTEM_PROMPT = (
    "You are a helpful assistant with expertise in cross-cultural knowledge. "
    "When given a cultural question with multiple choices, reason step by step "
    "about the cultural context, then provide your answer in the format: Answer: X"
)


# ---------------------------------------------------------------------------
# 答案抽取（与 single_data.py / evaluate.py 保持一致）
# ---------------------------------------------------------------------------

def _max_choice_of(task_type: str) -> int:
    if task_type == "cultureatlas":
        return 2
    if task_type == "culturalbench":
        return 4
    return 3


def extract_answer(text: str, max_choice: int = 4):
    """从模型/teacher 回答中抽取答案数字。"""
    pattern = f"[1-{max_choice}]"
    m = re.search(rf"(?:Final\s+decision|Answer)\s*(?:is|[:\-])\s*({pattern})\b",
                  text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(rf"\b({pattern})\s*\.?\s*$", text.strip())
    if m:
        return m.group(1)
    m = re.search(rf"option\s*:?\s*({pattern})\b", text, re.IGNORECASE)
    if m:
        return m.group(1)
    digits = re.findall(rf"\b({pattern})\b", text)
    return digits[-1] if digits else None


# ---------------------------------------------------------------------------
# 从 pkl 的 train split 构造蒸馏样本（teacher response 作为 label）
# ---------------------------------------------------------------------------

def build_distill_samples(records: list[dict],
                          filter_correct: bool = False,
                          drop_no_pred: bool = True) -> list[dict]:
    """从 teacher role-play 记录（pkl 的 train split）构造蒸馏 SFT 样本。

    监督目标（target）= teacher 的 response 字段（role-play 输出），
    *不是* 数据集原始 label（gt）。

    Args:
        records:        role-play 输出记录（含 query/country/gt/pred/response）
        filter_correct: 仅保留 teacher 答对（pred == gt）的样本
        drop_no_pred:   丢弃无法抽取答案（pred 为空）的样本
    Returns:
        list of {query, country, target, gt, pred, task_type}
    """
    samples = []
    skip_empty_resp = 0
    skip_no_pred = 0
    skip_wrong = 0

    for obj in records:
        query = (obj.get("query") or "").strip()
        country = (obj.get("country") or "").strip()
        gt = str(obj.get("gt", "")).strip()
        response = (obj.get("response") or "").strip()
        task_type = obj.get("task_type", "culturalbench")

        # teacher 输出为空 —— 无监督信号，跳过
        if not response or not query:
            skip_empty_resp += 1
            continue

        # 抽取 teacher 答案（优先用文件里的 pred，缺失则现场抽取）
        pred = obj.get("pred")
        if pred is None:
            pred = extract_answer(response, _max_choice_of(task_type))

        if drop_no_pred and (pred is None or str(pred).strip() == ""):
            skip_no_pred += 1
            continue

        # 可选：拒绝采样，仅蒸馏 teacher 答对的样本
        if filter_correct and gt and str(pred).strip() != gt:
            skip_wrong += 1
            continue

        samples.append({
            "query": query,
            "country": country,
            "target": response,      # ★ 蒸馏目标 = teacher role-play 输出
            "gt": gt,
            "pred": str(pred).strip() if pred is not None else None,
            "task_type": task_type,
        })

    print(f"Distillation (train) samples: {len(samples)} kept | "
          f"skipped: empty_response={skip_empty_resp}, "
          f"no_pred={skip_no_pred}, wrong(filtered)={skip_wrong}")
    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DistillSFTDataset(Dataset):
    """标准 SFT 数据集：prompt 部分 mask，仅在 teacher response 上计算 loss。"""

    def __init__(self, samples: list[dict], tokenizer, max_len: int = MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.records = []

        for s in samples:
            input_text = f"[{s['country']}]\n{s['query']}" if s["country"] else s["query"]
            target_text = s["target"]

            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": target_text},
                ],
                tokenize=False, add_generation_prompt=False,
            )
            prompt_only = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": input_text},
                ],
                tokenize=False, add_generation_prompt=True,
            )
            self.records.append((full_text, prompt_only))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        full_text, prompt_text = self.records[idx]
        tok = self.tokenizer

        full_enc = tok(full_text, max_length=self.max_len,
                       truncation=True, return_tensors="pt")
        prompt_enc = tok(prompt_text, max_length=self.max_len,
                         truncation=True, return_tensors="pt")

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX  # 仅监督 teacher 输出部分

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict:
    """右侧 padding 到 batch 内最大长度。"""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids_list, mask_list, label_list = [], [], []

    for b in batch:
        n = b["input_ids"].shape[0]
        pad = max_len - n
        input_ids_list.append(
            torch.cat([b["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)])
        )
        mask_list.append(
            torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)])
        )
        label_list.append(
            torch.cat([b["labels"], torch.full((pad,), IGNORE_INDEX, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(mask_list),
        "labels": torch.stack(label_list),
    }


# ---------------------------------------------------------------------------
# Loss（标准交叉熵，prompt 部分由 IGNORE_INDEX 屏蔽）
# ---------------------------------------------------------------------------

def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )
    return loss


# ---------------------------------------------------------------------------
# 验证：在 val split 上生成式评估，按数据集原始 gt 计算准确率
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, tokenizer, val_samples: list[dict], accelerator: Accelerator,
             max_samples: int = 300) -> float:
    """在验证集上计算学生模型的准确率（vs. 原始 gt）。仅在主进程执行。

    注意：验证口径与 evaluate.py 在 test 上完全一致 —— 生成回答后抽取答案，
    与数据集原始 gt 比对。蒸馏目标虽然是 teacher 输出，但我们关心的是学生
    在真实任务上的正确率，因此验证 / 测试都用 gt。
    """
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()

    # 关闭 gradient checkpointing 并恢复 use_cache，否则 generate() 会因
    # use_cache=False 而极慢甚至卡住。
    if hasattr(unwrapped_model, "base_model"):
        underlying = (unwrapped_model.base_model.model
                      if hasattr(unwrapped_model.base_model, "model")
                      else unwrapped_model.base_model)
    else:
        underlying = unwrapped_model
    underlying.gradient_checkpointing_disable()
    underlying.config.use_cache = True

    device = accelerator.device
    correct, total = 0, 0

    for obj in val_samples:
        if total >= max_samples:
            break
        query = obj["query"]
        country = obj.get("country", "")
        gold = str(obj.get("gt", "")).strip()
        if not gold:
            continue
        task_type = obj.get("task_type", "culturalbench")
        max_choice = _max_choice_of(task_type)

        input_text = f"[{country}]\n{query}" if country else query
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input_text},
            ],
            tokenize=False, add_generation_prompt=True,
        )
        enc = tokenizer(prompt, return_tensors="pt",
                        max_length=MAX_SEQ_LEN, truncation=True).to(device)

        outs = unwrapped_model.generate(
            **enc,
            max_new_tokens=512,
            do_sample=False,
            top_k=None, top_p=None, temperature=None,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        prompt_len = enc["input_ids"].shape[1]
        response = tokenizer.decode(outs[0][prompt_len:], skip_special_tokens=True)

        pred = extract_answer(response, max_choice)
        if pred == gold:
            correct += 1
        total += 1

        if total % 50 == 0:
            accelerator.print(f"    Eval progress: {total} "
                              f"(acc so far: {correct/total:.4f})")

    # 恢复训练态：先关 use_cache，再重新开启 gradient checkpointing
    underlying.config.use_cache = False
    underlying.gradient_checkpointing_enable()
    unwrapped_model.train()

    acc = correct / total if total > 0 else 0.0
    accelerator.print(f"    [Eval] val_accuracy={acc:.4f} ({correct}/{total})")
    return acc


# ---------------------------------------------------------------------------
# Training (LoRA + Accelerate DDP)
# ---------------------------------------------------------------------------

def train(args):
    ddp_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(42)

    accelerator.print(f"Number of processes: {accelerator.num_processes}")
    accelerator.print(f"Device: {accelerator.device}")
    accelerator.print(f"LoRA rank: {args.lora_r}, LoRA alpha: {args.lora_alpha}")
    accelerator.print(f"filter_correct={args.filter_correct}, "
                      f"drop_no_pred={args.drop_no_pred}")
    accelerator.print(f"Early stopping patience: {args.patience}")

    model_path = MODEL_ALIASES.get(args.model_name.lower(), args.model_name)
    accelerator.print(f"Base (student) model: {model_path}")

    # 加载 pkl 划分（train 训练 / val 验证 / test 不在此处使用）
    accelerator.print(f"Loading data splits from: {args.data_pkl}")
    with open(args.data_pkl, "rb") as f:
        splits = pickle.load(f)
    train_raw = splits["train"]
    val_raw = splits["val"]
    accelerator.print(f"  Splits: train={len(train_raw)}, val={len(val_raw)}, "
                      f"test={len(splits['test'])} (test 由 evaluate.py 使用)")

    # 构造蒸馏训练样本（teacher response 作为 label）
    samples = build_distill_samples(
        train_raw,
        filter_correct=args.filter_correct,
        drop_no_pred=args.drop_no_pred,
    )
    if len(samples) == 0:
        raise ValueError("No valid distillation samples found in train split.")

    if args.max_samples > 0:
        samples = samples[: args.max_samples]
        accelerator.print(f"Using first {args.max_samples} train samples")
    else:
        accelerator.print(f"Using all {len(samples)} train samples")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    # LoRA：只训练 adapter 参数，基座冻结
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    train_ds = DistillSFTDataset(samples, tokenizer)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
        num_workers=2,
        pin_memory=True,
    )
    accelerator.print(f"Training batches per epoch (per device): {len(loader)}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    total_steps = (len(loader) * args.epochs) // args.grad_accum_steps
    warmup_steps = max(1, int(total_steps * 0.05))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    no_improve = 0  # 连续多少个 epoch 验证集准确率未提升

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, num_steps = 0.0, 0

        for step, batch in enumerate(loader, 1):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = compute_loss(outputs.logits, batch["labels"])

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            num_steps += 1
            if step % 20 == 0:
                accelerator.print(f"  Epoch {epoch} step {step}/{len(loader)} "
                                  f"loss={loss.item():.4f}")

        avg_loss = total_loss / max(num_steps, 1)
        accelerator.print(f"Epoch {epoch}/{args.epochs} | avg_loss={avg_loss:.4f}")

        # ---- 每个 epoch 在验证集上评估（仅主进程 generate）----
        # 用一个张量作为跨进程的"早停信号"，确保所有 rank 同步退出训练循环。
        accelerator.wait_for_everyone()
        stop_signal = torch.zeros(1, device=accelerator.device)

        if accelerator.is_main_process:
            accelerator.print(f"  [Eval] Validating epoch {epoch} on val split "
                              f"(max {args.eval_max_samples} samples)...")
            val_acc = validate(model, tokenizer, val_raw, accelerator,
                               max_samples=args.eval_max_samples)
            accelerator.print(f"  [Eval] Epoch {epoch} | val_accuracy={val_acc:.4f} "
                              f"| previous best={best_val_acc:.4f}")

            improved = val_acc > best_val_acc + args.min_delta
            if improved:
                best_val_acc = val_acc
                no_improve = 0
                ckpt = Path(args.output_dir) / "best"
                unwrapped = accelerator.unwrap_model(model)
                # 仅保存 LoRA adapter（adapter_model.safetensors + adapter_config.json）
                unwrapped.save_pretrained(str(ckpt))
                tokenizer.save_pretrained(str(ckpt))
                accelerator.print(f"  ✓ New best (val_acc={best_val_acc:.4f}) "
                                  f"→ saved LoRA adapter to {ckpt}")
            else:
                no_improve += 1
                accelerator.print(f"  No improvement ({no_improve}/{args.patience}) "
                                  f"| best={best_val_acc:.4f}")
                if no_improve >= args.patience:
                    accelerator.print(f"  ✋ Early stopping triggered at epoch {epoch} "
                                      f"(no improvement for {args.patience} epochs).")
                    stop_signal += 1.0

        # 广播早停信号到所有进程，保证 DDP 下同步退出
        accelerator.wait_for_everyone()
        stop_signal = broadcast(stop_signal, from_process=0)
        if stop_signal.item() > 0:
            break

    # 若整个训练过程从未保存（极端情况：val 始终为 0），兜底保存最终 adapter
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        best_dir = Path(args.output_dir) / "best"
        if not best_dir.exists():
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            accelerator.print(f"  Saved final LoRA adapter (no val improvement "
                              f"observed) → {best_dir}")
        accelerator.print(f"\nDistillation complete. "
                          f"Best val_accuracy: {best_val_acc:.4f}")
        accelerator.print(f"Best LoRA adapter: {args.output_dir}/best")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-Teacher Distillation SFT (LoRA) — Accelerate DDP. "
                    "在 pkl 的 train 上用 role-play 输出（response）作为监督信号蒸馏基座；"
                    "每个 epoch 在 val 上评估准确率，保存最优 adapter 并支持早停。"
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="学生基座别名 'llama' / 'qwen'，或完整本地路径")
    parser.add_argument("--data_pkl", type=str, required=True,
                        help="split_data.py 产出的 pkl（含 train/val/test）；"
                             "输入应为 single_data.py --method role 的 role-play 输出")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="保存 LoRA adapter 的目录（最优模型存到 {output_dir}/best）")
    parser.add_argument("--filter_correct", action="store_true",
                        help="仅蒸馏 teacher 答对（pred==gt）的样本（拒绝采样）。"
                             "默认 False，按 role-play 输出全量蒸馏。")
    parser.add_argument("--drop_no_pred", action="store_true", default=True,
                        help="丢弃无法抽取答案的样本（默认 True）")
    parser.add_argument("--keep_no_pred", dest="drop_no_pred", action="store_false",
                        help="保留无法抽取答案的样本")
    parser.add_argument("--epochs", type=int, default=5,
                        help="最大训练轮数（早停可能提前结束）")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--patience", type=int, default=2,
                        help="连续 N 个 epoch 验证集准确率不提升则早停（默认 2）")
    parser.add_argument("--min_delta", type=float, default=0.0,
                        help="判定'提升'所需的最小准确率增量（默认 0.0）")
    parser.add_argument("--eval_max_samples", type=int, default=300,
                        help="每轮验证最多评估的样本数（默认 300，加速验证）")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="最大训练样本数，0=全部")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
