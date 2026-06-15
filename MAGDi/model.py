import torch
import logging
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, Linear
from torch.nn import CrossEntropyLoss, MarginRankingLoss
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer

class GCN(torch.nn.Module):

    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)
    
    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gcn2(x, edge_index)
        # Return raw logits (NOT log_softmax) — cross_entropy expects raw logits
        return x, x


class MAGDi(torch.nn.Module):
    """
    MAGDi model: combines a causal LM decoder with a GCN for graph-based
    distillation and MLPs for margin ranking.
    
    This version does NOT use accelerate dispatch_model. Instead:
      - The decoder uses device_map="auto" internally (handled externally via PEFT)
      - GCN and MLPs are placed on a specified auxiliary device
    """

    def __init__(self, model_name, gcn_in_channels, gcn_hidden_channels,
                 gcn_out_channels, alpha, beta, gamma, aux_device="cuda:0",
                 decoder_device="cuda:1"):
        super(MAGDi, self).__init__()
        self.aux_device = aux_device
        self.decoder_device = decoder_device
        self._gcn_in_channels = gcn_in_channels
        self._hidden_size = None  # set later via set_decoder
        
        # GCN on auxiliary device (registered as submodule)
        self.gcn = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels).to(aux_device)
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def set_decoder(self, decoder, hidden_size):
        """Set the decoder (after LoRA is applied) and initialize MLPs."""
        # Use add_module so PyTorch registers it for parameters() traversal
        self.add_module('decoder', decoder)
        self._hidden_size = hidden_size
        self.mlp1 = Linear(hidden_size, hidden_size).to(self.aux_device)
        self.mlp2 = Linear(hidden_size, 1).to(self.aux_device)
        # Enable gradient for GCN and MLPs
        for param in self.gcn.parameters():
            param.requires_grad = True
        for param in self.mlp1.parameters():
            param.requires_grad = True
        for param in self.mlp2.parameters():
            param.requires_grad = True
        # Signal to HF Trainer that this model is already distributed
        # Trainer checks hasattr(model, 'hf_device_map') to set is_model_parallel
        self.hf_device_map = {
            'decoder': self.decoder_device,
            'gcn': self.aux_device,
            'mlp1': self.aux_device,
            'mlp2': self.aux_device,
        }
        
    def _weighted_pool(self, hidden_states, attention_mask):
        """Weighted average pooling (recency-weighted)."""
        device = hidden_states.device
        # Use simple mean pooling over non-padding positions (more numerically stable)
        mask = attention_mask.to(device).unsqueeze(-1).float()  # (B, T, 1)
        sum_emb = torch.sum(hidden_states.float() * mask, dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)  # avoid division by zero
        return sum_emb / denom

    def _get_decoder_device(self):
        """Get decoder device (cached after first call for speed)."""
        if hasattr(self, '_cached_dec_device'):
            return self._cached_dec_device
        # Detect once from embedding layer
        base = self.decoder
        if hasattr(base, 'base_model'):
            base = base.base_model
        if hasattr(base, 'model'):
            base = base.model
        if hasattr(base, 'model') and hasattr(base.model, 'embed_tokens'):
            self._cached_dec_device = next(base.model.embed_tokens.parameters()).device
        elif hasattr(base, 'embed_tokens'):
            self._cached_dec_device = next(base.embed_tokens.parameters()).device
        else:
            self._cached_dec_device = self.decoder_device
        return self._cached_dec_device

    def forward(self, pos_input_ids, pos_attention_mask, pos_labels, neg_input_ids, neg_attention_mask, neg_labels, graph):
        
        # Batch graphs directly using PyG's Batch (avoids DataLoader overhead per step)
        graph_batch = Batch.from_data_list(list(graph)).to(self.aux_device)
        
        # Route decoder inputs to decoder's actual device (detect dynamically)
        dec_device = self._get_decoder_device()
        pos_input_ids = pos_input_ids.to(dec_device)
        pos_attention_mask = pos_attention_mask.to(dec_device)
        pos_labels = pos_labels.to(dec_device)
        neg_input_ids = neg_input_ids.to(dec_device)
        neg_attention_mask = neg_attention_mask.to(dec_device)
        
        # === Positive forward (with gradient for NLL loss) ===
        pos_output = self.decoder(input_ids=pos_input_ids,
                             attention_mask=pos_attention_mask,
                             labels=pos_labels,
                             output_hidden_states=True)
        nll_loss = pos_output.loss
        # Only keep last hidden state, discard the rest immediately
        pos_last_hidden = pos_output.hidden_states[-1]
        del pos_output
        
        pos_seq_emb = self._weighted_pool(pos_last_hidden, pos_attention_mask)
        del pos_last_hidden

        # === Negative forward (NO gradient — saves ~40% compute & memory) ===
        with torch.no_grad():
            neg_output = self.decoder(input_ids=neg_input_ids,
                                 attention_mask=neg_attention_mask,
                                 labels=None,
                                 output_hidden_states=True)
            neg_last_hidden = neg_output.hidden_states[-1]
            del neg_output
            neg_seq_emb = self._weighted_pool(neg_last_hidden, neg_attention_mask)
            del neg_last_hidden

        # Filter out padding-only negatives
        row_sums = neg_attention_mask.sum(dim=1)
        neg_mask = row_sums > 5
        if neg_mask.any():
            neg_mask = neg_mask.to(pos_seq_emb.device)
            pos_seq_emb = pos_seq_emb[neg_mask]
            neg_seq_emb = neg_seq_emb[neg_mask]
        
        # Move embeddings to auxiliary device for MLP computation
        pos_seq_emb = pos_seq_emb.to(self.aux_device).float()
        neg_seq_emb = neg_seq_emb.to(self.aux_device).float()
        # Normalize to prevent overflow in MLP (add eps to avoid 0-norm → nan)
        pos_seq_emb = F.normalize(pos_seq_emb + 1e-8, p=2, dim=-1)
        neg_seq_emb = F.normalize(neg_seq_emb.detach() + 1e-8, p=2, dim=-1)
            
        pos_h = torch.relu(self.mlp1(pos_seq_emb))
        pos_score = torch.tanh(self.mlp2(pos_h))
        
        neg_h = torch.relu(self.mlp1(neg_seq_emb))
        neg_score = torch.tanh(self.mlp2(neg_h))
        
        # Margin ranking loss (handle empty case)
        if pos_score.numel() == 0:
            mr_loss = torch.tensor(0.0, device=self.aux_device, requires_grad=True)
        else:
            mr_loss = F.margin_ranking_loss(
                pos_score, neg_score, torch.ones_like(pos_score), margin=1.0)
        
        # GCN node classification (normalize node embeddings to prevent overflow)
        node_x = graph_batch.x.float()
        # Replace any nan/inf in node embeddings with 0
        node_x = torch.nan_to_num(node_x, nan=0.0, posinf=1.0, neginf=-1.0)
        # Add small epsilon before normalize to avoid 0-norm → nan
        node_x = F.normalize(node_x + 1e-8, p=2, dim=-1)
        gcn_output, logits = self.gcn(node_x, graph_batch.edge_index)
        # logits are raw (no softmax applied) — F.cross_entropy handles softmax internally
        y = graph_batch.y.to(logits.device)
        valid_nodes = (y != 2)
        if valid_nodes.any():
            # Clamp logits to prevent extreme values causing nan
            logits_clamped = logits.clamp(-30, 30)
            node_loss = F.cross_entropy(logits_clamped[valid_nodes], y[valid_nodes])
        else:
            node_loss = torch.tensor(0.0, device=self.aux_device, requires_grad=True)
        
        # Combine losses on aux_device in float32
        # nll_loss is on decoder_device (cuda:1), move to aux_device (cuda:0)
        # .to() preserves gradient for scalar tensors
        nll_loss = nll_loss.to(self.aux_device).float()
        
        total_loss = self.alpha * nll_loss + self.beta * node_loss + self.gamma * mr_loss
        
        return total_loss


class MAGDiTrainer(Trainer):
    """
    Custom Trainer for MAGDi that handles multi-device model properly.
    Prevents Trainer from wrapping model in DataParallel and from
    moving our manually-placed multi-device model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Tell Trainer this model is already distributed across devices
        self.is_model_parallel = True
        self.place_model_on_device = False

    def _move_model_to_device(self, model, device):
        """Override to prevent Trainer from moving our multi-device model."""
        pass  # Model components are already on correct devices

    def _wrap_model(self, model, training=True, dataloader=None):
        """Override to prevent Trainer from wrapping model in DataParallel/DDP."""
        return model

    def _prepare_inputs(self, inputs):
        """Override to prevent Trainer from moving inputs to a single device.
        Our forward() handles device routing internally."""
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Use the underlying MAGDi model directly, bypassing accelerate's
        # forward wrapper which tries to move tensors to a single device.
        # When accelerate wraps the model, the real model is at model.module or model itself.
        raw_model = model.module if hasattr(model, 'module') else model
        
        loss = raw_model.forward(
            pos_input_ids=inputs["pos_input_ids"],
            pos_attention_mask=inputs["pos_attention_mask"],
            pos_labels=inputs["pos_labels"],
            neg_input_ids=inputs["neg_input_ids"],
            neg_attention_mask=inputs["neg_attention_mask"],
            neg_labels=inputs["neg_labels"],
            graph=inputs["graph"]
        )

        # When return_outputs=True (evaluation), Trainer expects (loss, outputs) tuple
        return (loss, {"loss": loss}) if return_outputs else loss

    def _get_raw_model(self):
        """Get the underlying MAGDi model, unwrapping accelerate if needed."""
        m = self.model
        if hasattr(m, 'module'):
            m = m.module
        return m

    def _save(self, output_dir=None, state_dict=None):
        """Custom save: save LoRA decoder + GCN/MLP weights separately."""
        import os
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        raw_model = self._get_raw_model()
        
        # Save LoRA adapter (decoder)
        raw_model.decoder.save_pretrained(output_dir)
        
        # Save GCN + MLP weights
        aux_state = {
            'gcn': raw_model.gcn.state_dict(),
            'mlp1': raw_model.mlp1.state_dict(),
            'mlp2': raw_model.mlp2.state_dict(),
        }
        torch.save(aux_state, os.path.join(output_dir, "aux_modules.pt"))

    def _load_best_model(self):
        """Load best checkpoint back into model for load_best_model_at_end."""
        import os
        
        best_model_path = self.state.best_model_checkpoint
        if best_model_path is None:
            return
        
        raw_model = self._get_raw_model()
        print(f"\nLoading best model from: {best_model_path}")
        
        # Load LoRA adapter weights
        adapter_path = os.path.join(best_model_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            adapter_path = os.path.join(best_model_path, "adapter_model.bin")
        
        if os.path.exists(adapter_path):
            from safetensors.torch import load_file
            if adapter_path.endswith(".safetensors"):
                adapter_state = load_file(adapter_path)
            else:
                adapter_state = torch.load(adapter_path, map_location="cpu")
            # Load into decoder (PEFT model)
            missing, unexpected = raw_model.decoder.load_state_dict(adapter_state, strict=False)
            if missing:
                print(f"  [Warning] Missing keys when loading adapter: {len(missing)}")
        
        # Load GCN + MLP weights
        aux_path = os.path.join(best_model_path, "aux_modules.pt")
        if os.path.exists(aux_path):
            aux_state = torch.load(aux_path, map_location="cpu")
            raw_model.gcn.load_state_dict(aux_state['gcn'])
            raw_model.mlp1.load_state_dict(aux_state['mlp1'])
            raw_model.mlp2.load_state_dict(aux_state['mlp2'])
            # Move back to correct device
            raw_model.gcn.to(raw_model.aux_device)
            raw_model.mlp1.to(raw_model.aux_device)
            raw_model.mlp2.to(raw_model.aux_device)
        
        print(f"  Best model loaded successfully.")