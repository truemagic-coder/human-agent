import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.silu(gate)

class TransformerBlock(nn.Module):
    """Simplified Transformer block with modern improvements"""
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Multi-head attention
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Feed-forward network with GLU
        self.ff = nn.Sequential(
            GLU(dim),
            nn.Linear(dim, dim, bias=False)
        )
        
        # RMSNorm layers
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        
        # Multi-head attention
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -float('inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        
        x = residual + out
        
        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        
        return x

class RecurrentModule(nn.Module):
    """Base recurrent module for HRM"""
    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.transformer = TransformerBlock(dim, n_heads)
        self.dim = dim

    def forward(self, hidden_state, *inputs):
        """
        Update hidden state based on inputs
        Args:
            hidden_state: Current hidden state [B, T, dim]
            *inputs: Variable number of input tensors to combine
        """
        # Combine all inputs through element-wise addition
        combined_input = hidden_state
        for inp in inputs:
            if inp is not None:
                combined_input = combined_input + inp
        
        # Apply transformer block
        new_state = self.transformer(combined_input)
        return new_state

class HierarchicalReasoningModel(nn.Module):
    """
    Hierarchical Reasoning Model (HRM) implementation
    
    Based on the paper: "Hierarchical Reasoning Model: Learning to Achieve Turing-Complete
    Algorithmic Reasoning via Multi-Timescale Recurrent Dynamics"
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        n_heads: int = 8,
        max_seq_len: int = 1024,
        N: int = 4,  # Number of high-level cycles
        T: int = 8,  # Number of low-level steps per cycle
        use_act: bool = True,  # Adaptive Computation Time
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.N = N
        self.T = T
        self.use_act = use_act
        self.max_seq_len = max_seq_len
        
        # Input embedding
        self.input_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, dim) * 0.02)
        
        # Recurrent modules
        self.low_level_module = RecurrentModule(dim, n_heads)
        self.high_level_module = RecurrentModule(dim, n_heads)
        
        # Output head
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Q-head for Adaptive Computation Time
        if use_act:
            self.q_head = nn.Linear(dim, 2, bias=False)  # [halt, continue]
        
        # Initialize hidden states
        self.register_buffer('z_init_L', torch.randn(1, 1, dim) * 1.0)
        self.register_buffer('z_init_H', torch.randn(1, 1, dim) * 1.0)
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        max_segments: int = 4,
        min_segments: int = 1,
        epsilon: float = 0.1,
        training: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass of HRM
        
        Args:
            x: Input token sequence [B, seq_len]
            max_segments: Maximum number of segments for ACT
            min_segments: Minimum number of segments for ACT
            epsilon: Exploration probability for ACT
            training: Whether in training mode
            
        Returns:
            Dictionary containing outputs and intermediate states
        """
        B, seq_len = x.shape
        
        # Input embedding
        x_embed = self.input_embedding(x)
        if seq_len <= self.max_seq_len:
            pos_embed = self.pos_embedding[:seq_len].unsqueeze(0)
            x_embed = x_embed + pos_embed
        x_embed = self.dropout(x_embed)
        
        # Initialize hidden states
        z_L = self.z_init_L.expand(B, seq_len, -1).contiguous()
        z_H = self.z_init_H.expand(B, seq_len, -1).contiguous()
        
        outputs = []
        q_values = []
        segment_count = 0
        
        # Deep supervision loop
        while segment_count < max_segments:
            # Single forward pass of HRM
            z_H_new, z_L_new, output = self._hrm_forward_pass(z_L, z_H, x_embed)
            
            outputs.append(output)
            segment_count += 1
            
            # Adaptive Computation Time decision
            if self.use_act and training:
                q_vals = self.q_head(z_H_new.mean(dim=1))  # [B, 2]
                q_values.append(q_vals)
                
                # Determine minimum segments stochastically
                if torch.rand(1).item() < epsilon:
                    current_min = torch.randint(2, max_segments + 1, (1,)).item()
                else:
                    current_min = min_segments
                
                # Halting decision
                if segment_count >= current_min:
                    halt_probs = torch.softmax(q_vals, dim=-1)
                    should_halt = halt_probs[:, 0] > halt_probs[:, 1]
                    if should_halt.all() or segment_count >= max_segments:
                        break
            elif not training and self.use_act:
                # Simple greedy halting for inference
                q_vals = self.q_head(z_H_new.mean(dim=1))
                q_values.append(q_vals)
                halt_probs = torch.softmax(q_vals, dim=-1)
                if (halt_probs[:, 0] > halt_probs[:, 1]).all():
                    break
            else:
                # Fixed number of segments
                if segment_count >= max_segments:
                    break
            
            # Detach states for next segment (1-step gradient approximation)
            z_L = z_L_new.detach()
            z_H = z_H_new.detach()
        
        return {
            'outputs': outputs,
            'final_output': outputs[-1],
            'q_values': q_values,
            'num_segments': segment_count,
            'final_states': (z_H_new, z_L_new)
        }

    def _hrm_forward_pass(self, z_L, z_H, x_embed):
        """
        Single forward pass of HRM with hierarchical convergence
        
        Args:
            z_L: Low-level hidden state [B, seq_len, dim]
            z_H: High-level hidden state [B, seq_len, dim] 
            x_embed: Input embeddings [B, seq_len, dim]
            
        Returns:
            Tuple of (new_z_H, new_z_L, output)
        """
        # Hierarchical convergence: N cycles of T steps each
        for cycle in range(self.N):
            # Low-level updates (T steps with fixed high-level state)
            current_z_H = z_H  # Fixed during this cycle
            
            for step in range(self.T):
                z_L = self.low_level_module(z_L, current_z_H, x_embed)
            
            # High-level update (once per cycle)
            z_H = self.high_level_module(z_H, z_L)
        
        # Generate output from final high-level state
        output = self.output_head(z_H)
        
        return z_H, z_L, output

    def compute_loss(self, outputs, targets, q_values=None):
        """
        Compute loss with deep supervision and ACT
        
        Args:
            outputs: List of output logits from each segment
            targets: Target token sequence [B, seq_len]
            q_values: List of Q-values for ACT (optional)
            
        Returns:
            Total loss
        """
        total_loss = 0.0
        
        # Deep supervision: loss at each segment
        for i, output in enumerate(outputs):
            seq_loss = F.cross_entropy(
                output.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
            total_loss += seq_loss
        
        # ACT Q-learning loss
        if q_values and len(q_values) > 0:
            for i, q_vals in enumerate(q_values):
                # Compute Q-learning targets
                if i == len(outputs) - 1:  # Final segment
                    # Reward based on prediction correctness
                    predictions = outputs[i].argmax(dim=-1)
                    correct = (predictions == targets).float().mean(dim=-1)
                    target_halt = correct
                    target_continue = torch.zeros_like(correct)
                else:
                    # Intermediate segments
                    next_q = q_values[i + 1] if i + 1 < len(q_values) else q_vals
                    target_halt = torch.zeros(q_vals.size(0), device=q_vals.device)
                    target_continue = torch.max(next_q, dim=-1)[0]
                
                q_targets = torch.stack([target_halt, target_continue], dim=-1)
                act_loss = F.mse_loss(q_vals, q_targets.detach())
                total_loss += 0.1 * act_loss  # Weight the ACT loss
        
        return total_loss / len(outputs)

def create_hrm_model(vocab_size: int, **kwargs) -> HierarchicalReasoningModel:
    """
    Factory function to create an HRM model
    
    Args:
        vocab_size: Size of vocabulary
        **kwargs: Additional model parameters
        
    Returns:
        HRM model instance
    """
    default_config = {
        'dim': 512,
        'n_heads': 8,
        'max_seq_len': 1024,
        'N': 4,
        'T': 8,
        'use_act': True,
        'dropout': 0.1
    }
    
    config = {**default_config, **kwargs}
    return HierarchicalReasoningModel(vocab_size, **config)
