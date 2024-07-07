import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return self.scale * x / torch.sqrt(norm**2 + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.linear2 = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        return F.silu(self.linear1(x)) * self.linear2(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, seq_length, dim = x.size()
        q = self.query(x).view(
            batch_size, seq_length, self.n_heads, dim // self.n_heads
        )
        k = self.key(x).view(
            batch_size, seq_length, self.n_kv_heads, dim // self.n_kv_heads
        )
        v = self.value(x).view(
            batch_size, seq_length, self.n_kv_heads, dim // self.n_kv_heads
        )

        scores = torch.einsum("bhqd, bhkd -> bhqk", q, k) / math.sqrt(
            dim // self.n_heads
        )
        attn = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.einsum("bhqk, bhvd -> bhqd", attn, v)
        context = context.contiguous().view(batch_size, seq_length, dim)
        return self.out(context)


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn = GroupedQueryAttention(args.dim, args.n_heads, args.n_kv_heads)
        self.norm1 = RMSNorm(args.dim, args.norm_eps)
        self.norm2 = RMSNorm(args.dim, args.norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(args.dim, args.ffn_dim_multiplier * args.dim),
            SwiGLU(args.ffn_dim_multiplier * args.dim, args.dim),
        )
        self.rotary_emb = RotaryEmbedding(args.dim)

    def forward(self, x):
        seq_len, device = x.shape[1], x.device
        rotary_emb = self.rotary_emb(seq_len, device)  # Should match x.shape[2]
        x = x + rotary_emb

        attn_out = self.attn(self.norm1(x))
        x = x + attn_out

        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out

        return x


# the Transformer class
class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.token_emb = nn.Embedding(
            args.vocab_size, args.dim, padding_idx=args.pad_token_id
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.n_layers)]
        )
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, x):
        x = self.token_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits
