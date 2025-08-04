import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional

# --- Standard Building Blocks (Corrected and Verified) ---

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
        hidden_dim = hidden_dim or 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        # The dimension for inv_freq should be halved because we are dealing
        # with pairs of values for the real and imaginary parts of a complex number.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # --- THIS IS THE FIX ---
        # We need to repeat the frequencies for the full head dimension to match
        # the query and key tensors.
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len: int):
        if seq_len > self.cos_cached.shape[0]:
             raise ValueError("Sequence length exceeds cached rotary embedding size.")
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    cos = cos.unsqueeze(1).unsqueeze(2)
    sin = sin.unsqueeze(1).unsqueeze(2)
    
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

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = SwiGLU(dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), cos, sin, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

# --- Main Model (Corrected and Simplified) ---

class HierarchicalReasoningModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        n_heads: int = 8,
        N: int = 4, # Number of layers
        max_seq_len: int = 256,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
        **kwargs # Absorb extra args from old configs
    ):
        super().__init__()
        
        self.config = {
            "vocab_size": vocab_size, "dim": dim, "n_heads": n_heads,
            "N": N, "max_seq_len": max_seq_len, "dropout": dropout,
            "rope_theta": rope_theta
        }
        
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads) for _ in range(N)])
        self.norm = RMSNorm(dim)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(dim // n_heads, max_seq_len, base=rope_theta)
        
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
        h = self.token_embedding(input_ids)
        h = self.dropout(h)
        
        cos, sin = self.rotary_emb(h, seqlen)
        
        for layer in self.layers:
            h = layer(h, cos, sin, self.mask[:, :, :seqlen, :seqlen])
            
        h = self.norm(h)
        logits = self.output_head(h)

        return {'outputs': logits}

def create_hrm_model(vocab_size: int, **kwargs) -> HierarchicalReasoningModel:
    """Factory function to create the model."""
    return HierarchicalReasoningModel(vocab_size, **kwargs)
