"""
Single-Teacher Distillation SFT (单 teacher 蒸馏 SFT) — LoRA 版本

核心思想（与普通 SFT 的本质区别）：
  - 监督信号 = 大模型 role-play 方法生成的输出（response），而非数据集原始 label（gt）。
  - 即把"角色扮演大模型"当作 single teacher，将其 role-play 产出的完整回答
    （包含推理过程 + 'Answer: X'）作为目标序列，蒸馏到基座学生模型上。
  - 这样学生模型不仅学会答案，还学会 teacher 的文化推理风格。

输入数据：
  single_data.py 在 --method role 下产出的 JSONL（每行一条），字段示例：
    {
      "query":   "...",            # 题目
      "country": "Japan",          # 国家/文化
      "gt":      "1",              # 数据集原始 label（蒸馏时默认 *不用* 作为目标）
      "pred":    "1",              # 从 teacher response 抽取出的答案
      "response":"... Answer: 1",  # ★ teacher 的 role-play 完整输出，作为蒸馏目标
      "model_name": "llama",
      "method":  "role",
      "task_type": "culturalbench"
    }
  参考文件：Cul/data/blend_llama_role_20260610_112253.json

监督目标的构造：
  input  (user)      : [{country}]\n{query}
  output (assistant) : teacher 的 response（原样作为 label）

样本过滤（可选）：
  --filter_correct   仅保留 teacher 答对（pred == gt）的样本做蒸馏（rejection sampling）。
                     默认 False —— 严格按"用 role-play 输出作为 label"全量蒸馏。
  --drop_no_pred     丢弃无法抽取答案（pred 为 null/空）的样本。默认 True。

Usage (single GPU):
    python Cul/sft/train_single_teacher_distill.py \\
        --model_name   llama \\
        --teacher_files Cul/data/blend_llama_role_20260610_112253.json \\
        --output_dir   /root/autodl-tmp/models/distill_single_llama \\
        --epochs 3 --batch_size 4 --lr 2e-4 --lora_r 32

Usage (multi-GPU DDP via accelerate):
    accelerate launch --num_processes 2 Cul/sft/train_single_teacher_distill.py \\
        --model_name   llama \\
        --teacher_files Cul/data/blend_llama_role_xxx.json,Cul/data/normad_llama_role_xxx.json \\
        --output_dir   /root/autodl-tmp/models/distill_single_llama \\
        --epochs 3 --batch_size 4 --lr 2e-4 --lora_r 32
"""

import re
import json
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
from accelerate.utils import set_seed


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
# 数据加载：读取 role-play 输出 JSONL，并构造蒸馏样本
# ---------------------------------------------------------------------------

def _max_choice_of(task_type: str) -> int:
    if task_type == "cultureatlas":
        return 2
    if task_type == "culturalbench":
        return 4
    return 3


def _extract_answer(text: str, max_choice: int):
    """从 teacher response 中抽取答案数字（与 single_data.py 抽取逻辑保持一致）。"""
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


def load_teacher_records(files: list[str]) -> list[dict]:
    """加载一个或多个 role-play 输出 JSONL 文件，合并为记录列表。"""
    records = []
    for fp in files:
        path = Path(fp.strip())
        if not path.exists():
            raise FileNotFoundError(f"Teacher file not found: {path}")
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                n += 1
        print(f"  Loaded {n} records from {path}")
    return records


def build_distill_samples(records: list[dict],
                          filter_correct: bool = False,
                          drop_no_pred: bool = True) -> list[dict]:
    """从 teacher role-play 记录构造蒸馏 SFT 样本。

    监督目标（target）= teacher 的 response 字段（role-play 输出），
    *不是* 数据集原始 label（gt）。

    Args:
        records:        role-play 输出记录
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
            pred = _extract_answer(response, _max_choice_of(task_type))

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

    print(f"Distillation samples: {len(samples)} kept | "
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

    model_path = MODEL_ALIASES.get(args.model_name.lower(), args.model_name)
    accelerator.print(f"Base (student) model: {model_path}")

    # 加载 teacher role-play 输出并构造蒸馏样本
    teacher_files = [f for f in args.teacher_files.split(",") if f.strip()]
    accelerator.print(f"Teacher role-play files ({len(teacher_files)}):")
    records = load_teacher_records(teacher_files)
    samples = build_distill_samples(
        records,
        filter_correct=args.filter_correct,
        drop_no_pred=args.drop_no_pred,
    )
    if len(samples) == 0:
        raise ValueError("No valid distillation samples found. Check teacher files.")

    if args.max_samples > 0:
        samples = samples[: args.max_samples]
        accelerator.print(f"Using first {args.max_samples} distillation samples")
    else:
        accelerator.print(f"Using all {len(samples)} distillation samples")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

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

        # 每个 epoch 结束保存一次 LoRA adapter（覆盖到 epoch{e} 子目录）
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            ckpt = Path(args.output_dir) / f"epoch{epoch}"
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            accelerator.print(f"  ✓ Saved LoRA adapter → {ckpt}")
        accelerator.wait_for_everyone()

    # 额外保存最终 adapter 到 final 目录，方便统一引用
    if accelerator.is_main_process:
        ckpt = Path(args.output_dir) / "final"
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(str(ckpt))
        tokenizer.save_pretrained(str(ckpt))
        accelerator.print(f"\nDistillation complete. Final LoRA adapter → {ckpt}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-Teacher Distillation SFT (LoRA) — Accelerate DDP. "
                    "用 role-play 输出（response）作为监督信号蒸馏基座模型。"
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="学生基座别名 'llama' / 'qwen'，或完整本地路径")
    parser.add_argument("--teacher_files", type=str, required=True,
                        help="teacher role-play 输出 JSONL 路径，逗号分隔可传多个 "
                             "（如 single_data.py --method role 的产出）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="保存 LoRA adapter 的目录")
    parser.add_argument("--filter_correct", action="store_true",
                        help="仅蒸馏 teacher 答对（pred==gt）的样本（拒绝采样）。"
                             "默认 False，按 role-play 输出全量蒸馏。")
    parser.add_argument("--drop_no_pred", action="store_true", default=True,
                        help="丢弃无法抽取答案的样本（默认 True）")
    parser.add_argument("--keep_no_pred", dest="drop_no_pred", action="store_false",
                        help="保留无法抽取答案的样本")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="最大蒸馏样本数，0=全部")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
