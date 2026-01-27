"""Mixture of Experts (MoE) implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Expert(nn.Module):
    """Single expert network in the MoE layer.
    
    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension (feed-forward)
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert network.
        
        Args:
            x: Input tensor of shape (*, d_model)
        
        Returns:
            Output tensor of shape (*, d_model)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with top-k gating.
    
    This implementation uses a gating network to select the top-k experts for each token,
    enabling conditional computation and increased model capacity.
    
    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension for each expert
        num_experts: Number of expert networks
        top_k: Number of experts to activate per token
        dropout: Dropout probability
        expert_capacity_factor: Factor to determine expert capacity (for load balancing)
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        expert_capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        # Note: expert_capacity_factor is reserved for future capacity-based load balancing
        # Currently using variance-based load balancing in aux_loss
        self.expert_capacity_factor = expert_capacity_factor
        
        # Gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Expert networks
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])
        
        # For load balancing loss
        self.register_buffer("expert_counts", torch.zeros(num_experts))
    
    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through MoE layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            return_aux_loss: Whether to return auxiliary load balancing loss
        
        Returns:
            Tuple of (output, aux_loss) where:
                - output: Tensor of shape (batch_size, seq_len, d_model)
                - aux_loss: Optional load balancing loss term
        """
        batch_size, seq_len, d_model = x.shape
        
        # Reshape for processing: (batch_size * seq_len, d_model)
        x_flat = x.view(-1, d_model)
        
        # Compute gating scores: (batch_size * seq_len, num_experts)
        gate_logits = self.gate(x_flat)
        gate_scores = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        
        # Normalize top-k scores
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process through selected experts
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_scores = top_k_scores[:, i].unsqueeze(-1)
            
            # Process each expert
            for expert_id in range(self.num_experts):
                # Find tokens assigned to this expert
                mask = expert_indices == expert_id
                if not mask.any():
                    continue
                
                # Get inputs for this expert
                expert_input = x_flat[mask]
                
                # Process through expert
                expert_output = self.experts[expert_id](expert_input)
                
                # Weight by gating score and add to output
                output[mask] += expert_output * expert_scores[mask]
                
                # Update expert usage count (for load balancing)
                if self.training:
                    self.expert_counts[expert_id] += mask.sum().item()
        
        # Reshape output back: (batch_size, seq_len, d_model)
        output = output.view(batch_size, seq_len, d_model)
        
        # Compute auxiliary load balancing loss
        aux_loss = None
        if return_aux_loss and self.training:
            # Encourage balanced expert usage
            # Loss is variance of expert assignment probabilities
            expert_probs = gate_scores.mean(dim=0)
            aux_loss = self.num_experts * (expert_probs ** 2).sum()
        
        return output, aux_loss
    
    def reset_expert_counts(self):
        """Reset expert usage counts."""
        self.expert_counts.zero_()
