import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional

# --- Building Blocks from Reference ---

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or int(2 * (4 * dim) / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.shape[0]:
            raise ValueError("Sequence length exceeds cached rotary embedding size.")
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

# --- Hierarchical Model Implementation ---

class HRMBlock(nn.Module):
    """A single block in the reasoning hierarchy, using post-norm."""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.self_attn = Attention(dim, n_heads)
        self.mlp = SwiGLU(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor]):
        x = x + self.self_attn(self.norm1(x), cos, sin, mask)
        x = x + self.mlp(self.norm2(x))
        return x

class ReasoningModule(nn.Module):
    """A stack of HRMBlocks forming one level of the hierarchy."""
    def __init__(self, dim: int, n_heads: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([HRMBlock(dim, n_heads) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, injection: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor]):
        x = x + injection
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return x

class HierarchicalReasoningModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        n_heads: int = 8,
        H_layers: int = 4,
        L_layers: int = 4,
        H_cycles: int = 2,
        L_cycles: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
        **kwargs
    ):
        super().__init__()
        self.config = {
            'vocab_size': vocab_size,
            'dim': dim,
            'n_heads': n_heads,
            'H_layers': H_layers,
            'L_layers': L_layers,
            'H_cycles': H_cycles,
            'L_cycles': L_cycles,
            'max_seq_len': max_seq_len,
            'dropout': dropout,
            'rope_theta': rope_theta,
        }
        self.config.update(kwargs)

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.H_level = ReasoningModule(dim, n_heads, H_layers)
        self.L_level = ReasoningModule(dim, n_heads, L_layers)
        
        self.norm = RMSNorm(dim)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(dim // n_heads, max_seq_len, base=rope_theta)
        
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        # Initial states for reasoning levels
        self.H_init = nn.Parameter(torch.randn(1, 1, dim))

        mask = torch.full((1, 1, max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        bsz, seqlen = input_ids.shape
        
        # 1. Initial Setup
        x_embed = self.token_embedding(input_ids)
        x_embed = self.dropout(x_embed)
        
        cos, sin = self.rotary_emb(seqlen)
        mask = self.mask[:, :, :seqlen, :seqlen]
        
        # Initialize reasoning states
        z_H = self.H_init.expand(bsz, seqlen, -1)
        # Start the low-level state with the input embedding
        z_L = x_embed 

        # 2. Hierarchical Reasoning Loop
        # This revised loop provides a more stable training dynamic.
        for _ in range(self.H_cycles):
            # Inject high-level state into low-level state for refinement
            current_L_input = z_L + z_H
            
            # Process at the low level
            for _ in range(self.L_cycles):
                current_L_input = self.L_level(current_L_input, torch.zeros_like(current_L_input), cos, sin, mask)
            
            # Update the low-level state with the result of the cycles
            z_L = current_L_input

            # Update the high-level state based on the new low-level state
            z_H = self.H_level(z_H, z_L, cos, sin, mask)
            
        # 3. Final Output
        # Combine the final high-level state with the initial embedding
        # This residual connection is crucial for stable learning.
        h = self.norm(z_H + x_embed)
        logits = self.output_head(h)

        return {'logits': logits}

def create_hrm_model(vocab_size: int, **kwargs) -> HierarchicalReasoningModel:
    """Factory function to create the model."""
    return HierarchicalReasoningModel(vocab_size, **kwargs)
