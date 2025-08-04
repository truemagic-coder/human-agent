import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float = 2.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.expansion_size = int(hidden_size * expansion)
        self.fc1 = nn.Linear(hidden_size, self.expansion_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, self.expansion_size, bias=False)
        self.fc_out = nn.Linear(self.expansion_size, hidden_size, bias=False)

    def forward(self, x):
        return self.fc_out(F.silu(self.fc1(x)) * self.fc2(x))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Pre-compute for efficiency
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def apply_rotary_pos_emb(self, q, k, seq_len):
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        
        def rotate_half(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.cat((-x2, x1), dim=-1)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, causal: bool = False):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, rotary_emb: Optional[RotaryEmbedding] = None):
        B, T, C = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if rotary_emb is not None:
            q, k = rotary_emb.apply_rotary_pos_emb(q, k, T)
        
        # Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if self.causal:
            mask = torch.tril(torch.ones(T, T, device=attn_scores.device, dtype=torch.bool))
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)

class HierarchicalReasoningBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, expansion: float = 2.0, eps: float = 1e-5):
        super().__init__()
        self.self_attn = Attention(dim, num_heads, causal=False)
        self.mlp = SwiGLU(dim, expansion)
        self.norm1 = RMSNorm(dim, eps)
        self.norm2 = RMSNorm(dim, eps)

    def forward(self, hidden_states: torch.Tensor, rotary_emb: Optional[RotaryEmbedding] = None):
        # Pre-norm architecture
        normed = self.norm1(hidden_states)
        attn_out = self.self_attn(normed, rotary_emb)
        hidden_states = hidden_states + attn_out
        
        normed = self.norm2(hidden_states)
        mlp_out = self.mlp(normed)
        hidden_states = hidden_states + mlp_out
        
        return hidden_states

class ReasoningModule(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_layers: int, expansion: float = 2.0):
        super().__init__()
        self.layers = nn.ModuleList([
            HierarchicalReasoningBlock(dim, num_heads, expansion) 
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, 
                rotary_emb: Optional[RotaryEmbedding] = None):
        # Input injection
        hidden_states = hidden_states + input_injection
        
        # Apply layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary_emb)
        
        return hidden_states

class HierarchicalReasoningModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        n_heads: int = 8,
        max_seq_len: int = 1024,
        N: int = 4,  # High-level cycles
        T: int = 8,  # Low-level steps per cycle
        dropout: float = 0.1,
        H_layers: int = 2,
        L_layers: int = 2,
        expansion: float = 2.0,
        eps: float = 1e-5,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        
        self.config = {
            "vocab_size": vocab_size,
            "dim": dim,
            "n_heads": n_heads,
            "N": N,
            "T": T,
            "dropout": dropout,
            "max_seq_len": max_seq_len,
        }

        self.vocab_size = vocab_size
        self.dim = dim
        self.n_heads = n_heads
        self.N = N
        self.T = T
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Position encoding
        self.rotary_emb = RotaryEmbedding(
            dim // n_heads, 
            max_position_embeddings=max_seq_len, 
            base=rope_theta
        )

        # Reasoning modules
        self.H_level = ReasoningModule(dim, n_heads, H_layers, expansion)
        self.L_level = ReasoningModule(dim, n_heads, L_layers, expansion)

        # Output
        self.output_norm = RMSNorm(dim, eps)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Token embedding
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        # Output head (tied to token embedding for efficiency)
        nn.init.normal_(self.output_head.weight, std=0.02)
        
        # Initialize all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> Dict[str, Any]:
        B, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        x_embed = self.token_embedding(input_ids)
        x_embed = self.dropout(x_embed)
        
        # Scale embeddings
        x_embed = x_embed * math.sqrt(self.dim)

        # Initialize reasoning states
        z_H = torch.zeros(B, seq_len, self.dim, device=device, dtype=x_embed.dtype)
        z_L = torch.zeros(B, seq_len, self.dim, device=device, dtype=x_embed.dtype)

        # Hierarchical reasoning cycles
        for _ in range(self.N):
            # Low-level reasoning steps
            for _ in range(self.T):
                z_L = self.L_level(z_L, z_H + x_embed, self.rotary_emb)
            
            # High-level reasoning step
            z_H = self.H_level(z_H, z_L, self.rotary_emb)

        # Output projection
        hidden_states = self.output_norm(z_H)
        logits = self.output_head(hidden_states)

        return {
            'outputs': logits,
            'final_states': (z_H, z_L),
            'num_cycles': self.N,
            'steps_per_cycle': self.T
        }

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, 
                     ignore_index: int = -100) -> torch.Tensor:
        """Compute cross-entropy loss"""
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=ignore_index
        )
        return loss

def create_hrm_model(vocab_size: int, **kwargs) -> HierarchicalReasoningModel:
    return HierarchicalReasoningModel(vocab_size, **kwargs)
