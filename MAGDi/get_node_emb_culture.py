"""
Extract node embeddings for cultural alignment MAGs.

This script generates initial node embeddings by running each reasoning chain
through the base student model and performing weighted average pooling over
the last hidden layer. These embeddings are used to initialize the GCN in MAGDi.

Supports both HF-CAC (6 agents + judge = 7 nodes) and RECONCILE (5 agents + judge = 6 nodes)
MAG structures for NormAD and CultureBench datasets.

Usage:
    python get_node_emb_culture.py \
        --mag_file MAG/culturalbench_hf_cac.json \
        --model_name mistralai/Mistral-7B-Instruct-v0.2 \
        --output_file node_emb/culturalbench_hf_cac_node_emb.pkl \
        --data_source hf_cac \
        --batch_size 32
"""

import os
import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Model aliases
# ---------------------------------------------------------------------------

MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}


def generate_ordered_list_culture(all_result: list, data_source: str):
    """
    Generate ordered list of reasoning chains and their correctness labels
    for cultural alignment MAGs.
    
    For HF-CAC (6 agents + judge = 7 nodes per sample):
        Node order: agent0_r0, agent1_r0, ..., agent5_r0, judge
    
    For RECONCILE (5 agents + judge = 6 nodes per sample):
        Node order: agent0_r0, agent1_r0, ..., agent4_r0, judge
    
    Labels: 1 = correct, 0 = incorrect, 2 = padding
    """
    ordered_list = []
    labels = []
    
    if data_source == 'hf_cac':
        num_agents = 6
        agent_names = ['agent0', 'agent1', 'agent2', 'agent3', 'agent4', 'agent5']
    else:
        num_agents = 5
        agent_names = ['agent0', 'agent1', 'agent2', 'agent3', 'agent4']
    
    for result in all_result:
        q = result['question']
        gold = str(result['gold_answer']).strip()
        
        # Agent nodes (Round 0)
        for agent_name in agent_names:
            key = f"{agent_name}_output_0"
            if result.get(key):
                node = result[key]
                reasoning = node.get('reasoning', '')
                answer = str(node.get('answer', '')).strip()
                
                # Determine correctness
                if answer == gold:
                    labels.append(1)
                else:
                    labels.append(0)
                
                # Format as instruction-following prompt
                full_sent = f"[INST] ### Question: {q}[/INST] ### Answer: {reasoning} So the answer is {answer}"
                ordered_list.append(full_sent)
            else:
                # Padding node
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract node embeddings for cultural MAGs"
    )
    parser.add_argument('--mag_file', type=str, required=True,
                        help="Path to MAG JSON file")
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['llama', 'qwen'],
                        help="Base model alias (llama or qwen)")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Output pickle file for node embeddings")
    parser.add_argument('--data_source', type=str, required=True,
                        choices=['hf_cac', 'reconcile'],
                        help="Data source type (determines node count)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for embedding extraction")
    parser.add_argument('--max_length', type=int, default=512,
                        help="Max token length for truncation")
    parser.add_argument('--cache_dir', type=str, default='',
                        help="Model cache directory")
    
    args = parser.parse_args()
    
    # Resolve model alias to full path
    model_path = MODEL_ALIASES.get(args.model_name, args.model_name)
    
    # Load MAG data
    print(f"Loading MAG data from: {args.mag_file}")
    with open(args.mag_file, 'r', encoding='utf-8') as f:
        all_result = json.load(f)
    print(f"  Loaded {len(all_result)} MAGs")
    
    # Determine node count
    if args.data_source == 'hf_cac':
        max_node_num = 7  # 6 agents + 1 judge
    else:
        max_node_num = 6  # 5 agents + 1 judge
    
    print(f"  Nodes per sample: {max_node_num}")
    
    # Generate ordered list
    ordered_list, labels = generate_ordered_list_culture(all_result, args.data_source)
    print(f"  Total nodes to embed: {len(ordered_list)}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name} -> {model_path}")
    cache_dir = args.cache_dir if args.cache_dir else None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side='left',
        add_eos_token=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Extract embeddings
    print("Extracting node embeddings...")
    node_embeddings = None
    
    for i in tqdm(range(0, len(ordered_list), args.batch_size)):
        batch = ordered_list[i:i + args.batch_size]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        
        last_hidden_state = outputs.hidden_states[-1]
        
        # Weighted average pooling (same as original MAGDi)
        weights_for_non_padding = tokens.attention_mask * torch.arange(
            start=1,
            end=last_hidden_state.shape[1] + 1
        ).to(tokens.attention_mask.device).unsqueeze(0)
        weights_for_non_padding = weights_for_non_padding.to(last_hidden_state.device)
        
        sum_node_embeddings = torch.sum(
            last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1
        )
        num_of_none_padding_tokens = torch.sum(
            weights_for_non_padding, dim=-1
        ).unsqueeze(-1)
        
        emb = (sum_node_embeddings / num_of_none_padding_tokens).detach().cpu().numpy()
        
        if node_embeddings is None:
            node_embeddings = emb
        else:
            node_embeddings = np.concatenate([node_embeddings, emb])
    
    print(f"  Embedding shape: {node_embeddings.shape}")
    expected_shape = (len(all_result) * max_node_num, model.config.hidden_size)
    print(f"  Expected shape: {expected_shape}")
    
    # Save
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    with open(args.output_file, "wb") as f:
        pickle.dump(node_embeddings, f)
    
    print(f"  Saved embeddings to: {args.output_file}")


if __name__ == '__main__':
    main()
