"""
MAGDi training script adapted for cultural alignment tasks (NormAD & CultureBench).

This script trains a student model using the MAGDi structured distillation method
on cultural alignment data. It supports both HF-CAC and RECONCILE data sources
via a single --data_source parameter.

Key differences from original MAGDi train.py:
  1. Supports variable number of agents (6 for HF-CAC, 5 for RECONCILE)
  2. Adapted graph structure for cultural negotiation (1-round + judge)
  3. Uses cultural MAG data format (with country, role fields)
  4. Supports both NormAD (3-class) and CultureBench (4-class) datasets

Usage:
    # Train on CultureBench with HF-CAC data
    python train_culture.py \
        --dataset culturalbench \
        --data_source hf_cac \
        --mag_file MAG/culturalbench_hf_cac.json \
        --node_emb_file node_emb/culturalbench_hf_cac_node_emb.pkl \
        --num_epochs 10 --lr 5e-6

    # Train on NormAD with RECONCILE data
    python train_culture.py \
        --dataset normad \
        --data_source reconcile \
        --mag_file MAG/normad_reconcile.json \
        --node_emb_file node_emb/normad_reconcile_node_emb.pkl \
        --num_epochs 10 --lr 5e-6

    # Train on CultureBench with HF-CAC data (for ablation: same data, different method)
    python train_culture.py \
        --dataset culturalbench \
        --data_source hf_cac \
        --mag_file MAG/culturalbench_hf_cac.json \
        --node_emb_file node_emb/culturalbench_hf_cac_node_emb.pkl \
        --output_dir checkpoints/MAGDi_culturalbench_hf_cac \
        --num_epochs 10 --lr 5e-6
"""

import json
import torch
import pickle
import random
import numpy as np
import argparse
import transformers

from torch_geometric.data import Data
from itertools import cycle

from peft import (
    LoraConfig,
    get_peft_model,
)
from model import MAGDi, MAGDiTrainer
from data_utils import MAGDiDataCollator
from transformers import AutoTokenizer, AutoModelForCausalLM

np.random.seed(42)


# ---------------------------------------------------------------------------
# Model aliases
# ---------------------------------------------------------------------------

MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}


# ---------------------------------------------------------------------------
# Culture-specific utility functions
# ---------------------------------------------------------------------------

class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def generate_ordered_list_culture(all_result: list, data_source: str):
    """
    Generate ordered list of reasoning chains and correctness labels.
    Same logic as get_node_emb_culture.py for consistency.
    """
    ordered_list = []
    labels = []
    
    if data_source == 'hf_cac':
        agent_names = ['agent0', 'agent1', 'agent2', 'agent3', 'agent4', 'agent5']
    else:
        agent_names = ['agent0', 'agent1', 'agent2', 'agent3', 'agent4']
    
    for result in all_result:
        q = result['question']
        gold = str(result['gold_answer']).strip()
        
        # Agent nodes
        for agent_name in agent_names:
            key = f"{agent_name}_output_0"
            if result.get(key):
                node = result[key]
                reasoning = node.get('reasoning', '')
                answer = str(node.get('answer', '')).strip()
                
                if answer == gold:
                    labels.append(1)
                else:
                    labels.append(0)
                
                full_sent = f"[INST] ### Question: {q}[/INST] ### Answer: {reasoning} So the answer is {answer}"
                ordered_list.append(full_sent)
            else:
                labels.append(2)
                ordered_list.append("[INST] ### Question: None[/INST] ### Answer: None")
        
        # Judge node
        if result.get('judge_output'):
            judge = result['judge_output']
            reasoning = judge.get('reasoning', '')
            answer = str(judge.get('answer', '')).strip()
            
            if answer == gold:
                labels.append(1)
            else:
                labels.append(0)
            
            full_sent = f"[INST] ### Question: {q}[/INST] ### Answer: {reasoning} So the answer is {answer}"
            ordered_list.append(full_sent)
        else:
            labels.append(2)
            ordered_list.append("[INST] ### Question: None[/INST] ### Answer: None")
    
    return ordered_list, labels


def construct_graph_hf_cac(guardian_idx: int = 0):
    """
    Construct graph structure for HF-CAC MAG (6 agents + 1 judge = 7 nodes).
    
    Edge structure:
      - Guardian (node at guardian_idx) → all Auditor nodes (information flow)
      - All agent nodes (0-5) → Judge node (6)
    
    This captures the asymmetric information flow in HF-CAC:
      Phase 1: Guardian generates first
      Phase 2: Auditors see Guardian's output
      Judge: Sees all agents' outputs
    """
    edges = []
    
    # Guardian → all Auditors
    for i in range(6):
        if i != guardian_idx:
            edges.append([guardian_idx, i])
    
    # All agents → Judge (node 6)
    for i in range(6):
        edges.append([i, 6])
    
    edge_index = torch.tensor(edges, dtype=torch.long)
    data = Data(edge_index=edge_index.t().contiguous())
    return data


def construct_graph_reconcile():
    """
    Construct graph structure for RECONCILE MAG (5 agents + 1 judge = 6 nodes).
    
    Edge structure (symmetric):
      - All agents are independent (no inter-agent edges)
      - All agent nodes (0-4) → Judge node (5)
    
    This captures RECONCILE's symmetric structure where all agents
    have equal authority and the judge synthesizes.
    """
    edges = []
    
    # All agents → Judge (node 5)
    for i in range(5):
        edges.append([i, 5])
    
    edge_index = torch.tensor(edges, dtype=torch.long)
    data = Data(edge_index=edge_index.t().contiguous())
    return data


def construct_graphs_culture(all_result, embeddings, num_train_samples, max_node_num, data_source):
    """
    Construct list of graphs with node embeddings and labels for cultural MAGs.
    """
    _, labels = generate_ordered_list_culture(all_result, data_source)
    
    # Build graphs with appropriate structure
    graphs = []
    for i, result in enumerate(all_result):
        if data_source == 'hf_cac':
            guardian_idx = result.get('guardian_idx', 0)
            g = construct_graph_hf_cac(guardian_idx)
        else:
            g = construct_graph_reconcile()
        graphs.append(g)
    
    labels = torch.tensor(labels, dtype=torch.long)
    labels = labels.reshape(num_train_samples, max_node_num)
    
    for g, emb, y in zip(graphs, embeddings, labels):
        g.x = emb
        g.y = y
    
    return graphs


def prepare_contrastive_samples(samples, labels):
    """
    Prepare contrastive learning samples (same as original MAGDi).
    """
    if len(samples) != len(labels):
        raise ValueError("Samples and labels must be of the same length.")
    
    positive_samples = [s for s, l in zip(samples, labels) if l == 1]
    negative_samples = [s for s, l in zip(samples, labels) if l == 0]
    
    if len(negative_samples) == 0:
        negative_samples = ["NA"]
    
    if len(positive_samples) == 0:
        return None
    
    if len(positive_samples) > len(negative_samples):
        negative_samples = (negative_samples * ((len(positive_samples) // len(negative_samples)) + 1))[:len(positive_samples)]
    elif len(negative_samples) > len(positive_samples):
        positive_samples = (positive_samples * ((len(negative_samples) // len(positive_samples)) + 1))[:len(negative_samples)]
    
    return positive_samples, negative_samples


def prepare_batch_culture(tokenizer, all_result, num_train_samples, max_node_num, data_source, max_length=256):
    """
    Prepare training batch with positive/negative pairs for cultural MAGs.
    """
    ordered_list, labels = generate_ordered_list_culture(all_result, data_source)
    ordered_list = np.array(ordered_list).reshape(num_train_samples, max_node_num)
    labels = np.array(labels).reshape(num_train_samples, max_node_num)
    
    result = []
    for sentence, label in zip(ordered_list, labels):
        pairs = prepare_contrastive_samples(sentence, label)
        if pairs:
            positive_samples, negative_samples = pairs
            pos_enc = tokenizer(positive_samples, truncation=True, max_length=max_length)
            neg_enc = tokenizer(negative_samples, truncation=True, max_length=max_length)
            for pi, pa, ni, na in zip(
                pos_enc.input_ids, pos_enc.attention_mask,
                neg_enc.input_ids, neg_enc.attention_mask
            ):
                result.append({
                    'pos_input_ids': pi,
                    'pos_attention_mask': pa,
                    'pos_labels': pi,
                    'neg_input_ids': ni,
                    'neg_attention_mask': na,
                    'neg_labels': ni
                })
    return result


def pad_graphs(training_batch, graphs):
    """Pad graphs to match training batch size (cycle if needed)."""
    pool = cycle(graphs)
    graphs = [next(pool) for _ in range(len(training_batch))]
    for tb, g in zip(training_batch, graphs):
        tb['graph'] = g
    return training_batch, graphs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="MAGDi training for cultural alignment tasks"
    )
    # Dataset and data source
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['normad', 'culturalbench'],
                        help="Dataset type")
    parser.add_argument('--data_source', type=str, required=True,
                        choices=['hf_cac', 'reconcile'],
                        help="Data source: hf_cac or reconcile")
    
    # File paths
    parser.add_argument('--mag_file', type=str, required=True,
                        help="Path to MAG JSON file")
    parser.add_argument('--node_emb_file', type=str, required=True,
                        help="Path to node embedding pickle file")
    parser.add_argument('--output_dir', type=str, default='',
                        help="Output directory for checkpoints (auto-generated if empty)")
    
    # Model
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['llama', 'qwen'],
                        help="Base student model alias (llama or qwen)")
    parser.add_argument('--cache_dir', type=str, default='',
                        help="Model cache directory")
    
    # GCN hyperparameters
    parser.add_argument('--gcn_in_channels', type=int, default=4096,
                        help="GCN input dimension (must match model hidden size)")
    parser.add_argument('--gcn_hidden_channels', type=int, default=512,
                        help="GCN hidden dimension")
    parser.add_argument('--gcn_out_channels', type=int, default=2,
                        help="GCN output dimension (2 for binary: correct/incorrect)")
    
    # Loss weights
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Weight for NLL loss (positive reasoning)")
    parser.add_argument('--beta', type=float, default=1.0,
                        help="Weight for graph node classification loss")
    parser.add_argument('--gamma', type=float, default=0.1,
                        help="Weight for margin ranking loss (contrastive)")
    
    # Training
    parser.add_argument('--max_samples', '--num_train_samples', type=int, default=0,
                        dest='num_train_samples',
                        help="Max number of MAG samples for training (0 = use all). "
                             "E.g. --max_samples 10 for quick debugging.")
    parser.add_argument('--num_epochs', type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Per-device training batch size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument('--warmup_steps', type=int, default=50,
                        help="Warmup steps")
    parser.add_argument('--max_length', type=int, default=192,
                        help="Max sequence length for tokenization")
    
    # LoRA
    parser.add_argument('--lora_r', type=int, default=16,
                        help="LoRA rank")
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help="LoRA dropout")
    
    args = parser.parse_args()
    
    # Resolve model alias to full path
    model_path = MODEL_ALIASES.get(args.model_name, args.model_name)
    
    # Determine max_node_num based on data source
    if args.data_source == 'hf_cac':
        max_node_num = 7  # 6 agents + 1 judge
    else:
        max_node_num = 6  # 5 agents + 1 judge
    
    # Auto-generate output directory
    if not args.output_dir:
        args.output_dir = f"checkpoints/MAGDi_{args.dataset}_{args.data_source}"
    
    print(f"=" * 60)
    print(f"MAGDi Cultural Alignment Training")
    print(f"=" * 60)
    print(f"  Dataset: {args.dataset}")
    print(f"  Data source: {args.data_source}")
    print(f"  Max nodes per MAG: {max_node_num}")
    print(f"  Model: {args.model_name} -> {model_path}")
    print(f"  Output: {args.output_dir}")
    print(f"  Loss weights: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")
    print(f"=" * 60)
    
    # Load node embeddings
    print("\nLoading node embeddings...")
    with open(args.node_emb_file, "rb") as f:
        node_embeddings = pickle.load(f)
    print(f"  Shape: {node_embeddings.shape}")
    
    # Load MAG data
    print("Loading MAG data...")
    with open(args.mag_file, "r", encoding='utf-8') as f:
        all_result = json.load(f)
    
    num_train_samples = len(all_result)
    if args.num_train_samples > 0:
        num_train_samples = min(args.num_train_samples, len(all_result))
        all_result = all_result[:num_train_samples]
    print(f"  Training samples: {num_train_samples}")
    
    # Initialize model
    # Auto-detect hidden_size from model config to set GCN input dimension
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    gcn_in_channels = model_config.hidden_size
    print(f"\nInitializing MAGDi model (hidden_size={gcn_in_channels})...")
    
    # Create MAGDi shell (GCN + MLPs on cuda:0, decoder on cuda:1)
    # Split across 2 GPUs: decoder on cuda:1 (≈14GB), GCN/MLP on cuda:0
    aux_device = "cuda:0"
    decoder_device = "cuda:1"
    
    model = MAGDi(
        model_name=model_path,
        gcn_in_channels=gcn_in_channels,
        gcn_hidden_channels=args.gcn_hidden_channels,
        gcn_out_channels=args.gcn_out_channels,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        aux_device=aux_device,
        decoder_device=decoder_device
    )
    
    # Load decoder on cuda:1 (7B fp16 ≈ 14GB, fits in one 48GB card)
    # No device_map="auto" — direct .to() preserves gradient flow
    print(f"Loading decoder on {decoder_device}...")
    decoder = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(decoder_device)
    
    # Reshape node embeddings
    node_embeddings = node_embeddings.reshape(
        num_train_samples, max_node_num, gcn_in_channels
    )
    node_embeddings = torch.tensor(node_embeddings)
    node_embeddings = node_embeddings[:num_train_samples, :, :]
    print(f"  Node embeddings reshaped: {node_embeddings.size()}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side='left',
        add_eos_token=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Apply LoRA to decoder BEFORE assigning to MAGDi
    print("Applying LoRA...")
    decoder.gradient_checkpointing_enable()
    decoder.enable_input_require_grads()
    
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    decoder = get_peft_model(decoder, config)
    decoder.print_trainable_parameters()
    
    # Assign decoder to MAGDi model and initialize MLPs
    model.set_decoder(decoder, gcn_in_channels)
    print(f"  Decoder device_map: {decoder.hf_device_map if hasattr(decoder, 'hf_device_map') else 'N/A'}")
    
    # Prepare training data
    print(f"\nPreparing training batch (max_length={args.max_length})...")
    training_batch = prepare_batch_culture(
        tokenizer, all_result, num_train_samples, max_node_num, args.data_source,
        max_length=args.max_length
    )
    print(f"  Training pairs: {len(training_batch)}")
    
    # Construct graphs
    print("Constructing graphs...")
    graphs = construct_graphs_culture(
        all_result, node_embeddings, num_train_samples, max_node_num, args.data_source
    )
    print(f"  Graphs: {len(graphs)}")
    
    # Pad graphs to match training batch
    training_batch, graphs = pad_graphs(training_batch, graphs)
    print(f"  After padding: {len(training_batch)} batches, {len(graphs)} graphs")
    
    # Split into train/eval (90%/10%)
    random.seed(42)
    indices = list(range(len(training_batch)))
    random.shuffle(indices)
    split_idx = int(len(indices) * 0.9)
    train_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]
    train_dataset = [training_batch[i] for i in train_indices]
    eval_dataset = [training_batch[i] for i in eval_indices]
    print(f"  Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    # Train
    print(f"\nStarting training for {args.num_epochs} epochs...")
    
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        fp16=False,
        bf16=True,  # BF16 mixed precision: faster, no GradScaler needed, wider dynamic range
        logging_steps=10,
        output_dir=args.output_dir,
        remove_unused_columns=False,
        # Evaluate & save at end of each epoch
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.num_epochs,
        # Best model tracking
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
    )
    
    # Early stopping: stop if eval_loss doesn't improve for 3 consecutive epochs
    from transformers import EarlyStoppingCallback
    early_stop_callback = EarlyStoppingCallback(early_stopping_patience=3)
    
    trainer = MAGDiTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=MAGDiDataCollator(tokenizer),
        callbacks=[early_stop_callback]
    )
    
    import os
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted by user.")
        print("[INFO] Checkpoints already saved in output_dir. Finding best one...")
    
    # Save best model to final directory
    # If training completed normally, load_best_model_at_end already loaded the best weights.
    # If interrupted, we find the best checkpoint from saved ones.
    final_dir = os.path.join(args.output_dir, "best")
    os.makedirs(final_dir, exist_ok=True)
    
    # Check if trainer has best_model_checkpoint info
    best_ckpt = getattr(trainer.state, 'best_model_checkpoint', None)
    if best_ckpt and os.path.exists(best_ckpt):
        print(f"\nBest checkpoint: {best_ckpt} (eval_loss={trainer.state.best_metric:.6f})")
        # Copy best checkpoint to final dir
        import shutil
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.copytree(best_ckpt, final_dir)
    else:
        # Fallback: save current model state (may be last epoch or interrupted state)
        print(f"\nSaving current model state to: {final_dir}")
        model.decoder.save_pretrained(final_dir)
        aux_state = {
            'gcn': model.gcn.state_dict(),
            'mlp1': model.mlp1.state_dict(),
            'mlp2': model.mlp2.state_dict(),
        }
        torch.save(aux_state, os.path.join(final_dir, "aux_modules.pt"))
    
    print(f"Best model saved to: {final_dir}")
    print("Training complete!")
