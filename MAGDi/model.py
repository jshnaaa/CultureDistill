import torch
import logging
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
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
        return x, F.log_softmax(x, dim=1)


class MAGDi(torch.nn.Module):
    """
    MAGDi model: combines a causal LM decoder with a GCN for graph-based
    distillation and MLPs for margin ranking.
    
    This version does NOT use accelerate dispatch_model. Instead:
      - The decoder uses device_map="auto" internally (handled externally via PEFT)
      - GCN and MLPs are placed on a specified auxiliary device
    """

    def __init__(self, model_name, gcn_in_channels, gcn_hidden_channels,
                 gcn_out_channels, alpha, beta, gamma, aux_device="cuda:0"):
        super(MAGDi, self).__init__()
        self.aux_device = aux_device
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
        
    def _weighted_pool(self, hidden_states, attention_mask):
        """Weighted average pooling (recency-weighted) — no gradient needed."""
        device = hidden_states.device
        weights = attention_mask.to(device) * torch.arange(
            1, hidden_states.shape[1] + 1, device=device).unsqueeze(0)
        sum_emb = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        denom = torch.sum(weights, dim=-1, keepdim=True)
        return sum_emb / denom

    def forward(self, pos_input_ids, pos_attention_mask, pos_labels, neg_input_ids, neg_attention_mask, neg_labels, graph):
        
        graph_loader = DataLoader(graph, batch_size=len(graph), shuffle=False, pin_memory=False, num_workers=0)
        graph_batch = next(iter(graph_loader))
        graph_batch = graph_batch.to(self.aux_device)
        
        # === Positive forward (with gradient for NLL loss) ===
        pos_output = self.decoder(input_ids=pos_input_ids,
                             attention_mask=pos_attention_mask,
                             labels=pos_labels,
                             output_hidden_states=True)
        nll_loss = pos_output.loss
        
        # Debug: print loss components on first few steps
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 3:
            print(f"\n[DEBUG step {self._debug_count}] nll_loss={nll_loss.item():.6f}, "
                  f"requires_grad={nll_loss.requires_grad}, "
                  f"pos_input_ids shape={pos_input_ids.shape}, "
                  f"pos_labels unique={pos_labels.unique()[:10].tolist()}, "
                  f"pos_labels -100 count={(pos_labels == -100).sum().item()}/{pos_labels.numel()}")
            self._debug_count += 1
        
        pos_seq_emb = self._weighted_pool(pos_output.hidden_states[-1], pos_attention_mask)

        # === Negative forward (NO gradient — saves ~40% compute) ===
        with torch.no_grad():
            neg_output = self.decoder(input_ids=neg_input_ids,
                                 attention_mask=neg_attention_mask,
                                 labels=None,
                                 output_hidden_states=True)
            neg_seq_emb = self._weighted_pool(neg_output.hidden_states[-1], neg_attention_mask)

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
        # Re-enable gradient for neg embeddings through MLP
        neg_seq_emb = neg_seq_emb.detach().requires_grad_(False)
            
        pos_h = torch.relu(self.mlp1(pos_seq_emb))
        pos_score = torch.tanh(self.mlp2(pos_h))
        
        neg_h = torch.relu(self.mlp1(neg_seq_emb))
        neg_score = torch.tanh(self.mlp2(neg_h))
        
        # Margin ranking loss
        mr_loss = F.margin_ranking_loss(
            pos_score, neg_score, torch.ones_like(pos_score), margin=1.0)
        
        # GCN node classification
        gcn_output, logits = self.gcn(graph_batch.x.float(), graph_batch.edge_index)
        node_loss = F.cross_entropy(logits, graph_batch.y.to(logits.device))
        
        # Combine losses on aux_device in float32
        nll_loss = nll_loss.float()
        
        total_loss = self.alpha * nll_loss + self.beta * node_loss + self.gamma * mr_loss
        
        if self._debug_count <= 3:
            print(f"[DEBUG] node_loss={node_loss.item():.6f}, mr_loss={mr_loss.item():.6f}, "
                  f"total_loss={total_loss.item():.6f}, requires_grad={total_loss.requires_grad}")
        
        return total_loss


class MAGDiTrainer(Trainer):
    """
    Custom Trainer for MAGDi that handles multi-device model properly.
    Prevents Trainer from wrapping model in DataParallel (we use device_map).
    """

    def _move_model_to_device(self, model, device):
        """Override to prevent Trainer from moving our multi-device model."""
        pass  # Model components are already on correct devices

    def _wrap_model(self, model, training=True, dataloader=None):
        """Override to prevent Trainer from wrapping model in DataParallel/DDP."""
        return model

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Move tensor inputs to the model's device
        device = model.aux_device
        
        loss = model(
            pos_input_ids=inputs["pos_input_ids"].to(device),
            pos_attention_mask=inputs["pos_attention_mask"].to(device),
            pos_labels=inputs["pos_labels"].to(device),
            neg_input_ids=inputs["neg_input_ids"].to(device),
            neg_attention_mask=inputs["neg_attention_mask"].to(device),
            neg_labels=inputs["neg_labels"].to(device),
            graph=inputs["graph"]
        )

        return loss