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

class SimpleTransformerBlock(nn.Module):
    """Simplified Transformer block"""
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Multi-head attention
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, nh, T, hd]
        
        # Attention
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
        
        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        
        return x

class SimpleRecurrentModule(nn.Module):
    """Simplified recurrent module for HRM"""
    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.transformer = SimpleTransformerBlock(dim, n_heads)
        self.combine = nn.Linear(dim * 3, dim)  # Combine 3 inputs
        self.dim = dim

    def forward(self, hidden_state, *inputs):
        """Update hidden state based on inputs"""
        # Combine all inputs
        all_inputs = [hidden_state]
        for inp in inputs:
            if inp is not None:
                all_inputs.append(inp)
        
        # Pad to 3 inputs if needed
        while len(all_inputs) < 3:
            all_inputs.append(torch.zeros_like(hidden_state))
        
        # Combine inputs
        combined = torch.cat(all_inputs[:3], dim=-1)  # [B, T, 3*dim]
        combined = self.combine(combined)  # [B, T, dim]
        
        # Apply transformer
        new_state = self.transformer(combined)
        return new_state

class SimpleHierarchicalReasoningModel(nn.Module):
    """
    Simplified HRM - keeps hierarchical reasoning but fixes training issues
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        n_heads: int = 8,
        max_seq_len: int = 1024,
        N: int = 2,  # REDUCED: 4 → 2 cycles
        T: int = 4,  # REDUCED: 8 → 4 steps  
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.N = N
        self.T = T
        self.max_seq_len = max_seq_len
        
        # Input embedding
        self.input_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, dim) * 0.02)
        
        # Simplified recurrent modules
        self.low_level_module = SimpleRecurrentModule(dim, n_heads)
        self.high_level_module = SimpleRecurrentModule(dim, n_heads)
        
        # Output head
        self.ln_f = nn.LayerNorm(dim)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        
        # SIMPLIFIED: No ACT, no Q-heads
        
        # Initialize hidden states  
        self.register_buffer('z_init_L', torch.zeros(1, 1, dim))
        self.register_buffer('z_init_H', torch.zeros(1, 1, dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Conservative initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Conservative initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        x: torch.Tensor, 
        max_segments: int = 2,  # SIMPLIFIED: Only 1-2 segments
        min_segments: int = 1,
        epsilon: float = 0.8,   # High epsilon for stability
        training: bool = True
    ) -> Dict[str, Any]:
        """
        Simplified forward pass - no complex ACT logic
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
        
        # FIXED: Single reasoning segment with gradient flow
        z_H, z_L, output = self._hrm_forward_pass(z_L, z_H, x_embed)
        
        # SIMPLIFIED: Return single output
        return {
            'outputs': output,  # Single output, not list
            'q_values': torch.ones(B, 1, device=x.device),  # Dummy q-values
            'num_segments': 1,
            'final_states': (z_H, z_L)
        }

    def _hrm_forward_pass(self, z_L, z_H, x_embed):
        """
        FIXED: Single forward pass with proper gradient flow
        """
        # Hierarchical reasoning: N cycles of T steps each
        for cycle in range(self.N):
            # Low-level updates (T steps with current high-level state)
            for step in range(self.T):
                z_L = self.low_level_module(z_L, z_H, x_embed)
            
            # High-level update (once per cycle, using updated low-level)
            z_H = self.high_level_module(z_H, z_L, x_embed)
        
        # Generate output from final high-level state
        z_H = self.ln_f(z_H)
        output = self.output_head(z_H)
        
        return z_H, z_L, output

    def compute_loss(self, outputs, targets, q_values=None):
        """
        SIMPLIFIED: Single cross-entropy loss
        """
        # Single output - simple cross-entropy
        if isinstance(outputs, list):
            outputs = outputs[-1]  # Take final output
            
        loss = F.cross_entropy(
            outputs.view(-1, self.vocab_size),
            targets.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        return loss

def create_hrm_model(vocab_size: int, **kwargs) -> SimpleHierarchicalReasoningModel:
    """
    Factory function for simplified HRM
    """
    default_config = {
        'dim': 512,
        'n_heads': 8,
        'max_seq_len': 1024,
        'N': 2,  # Reduced complexity
        'T': 4,  # Reduced complexity
        'dropout': 0.1
    }
    
    config = {**default_config, **kwargs}
    return SimpleHierarchicalReasoningModel(vocab_size, **config)