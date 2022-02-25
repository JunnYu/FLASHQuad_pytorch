import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN


def rope(x, dim):
    """RoPE position embedding."""
    shape = x.shape
    if isinstance(dim, int):
        dim = [dim]
    spatial_shape = [shape[i] for i in dim]
    total_len = 1
    for i in spatial_shape:
        total_len *= i
        position = torch.reshape(
            torch.arange(total_len, dtype=x.dtype,
                         device=x.device), spatial_shape
        )
    for i in range(dim[-1] + 1, len(shape) - 1, 1):
        position = position.unsqueeze(-1)
    half_size = shape[-1] // 2
    freq_seq = -torch.arange(half_size, dtype=x.dtype, device=x.device) / float(
        half_size
    )
    inv_freq = 10000 ** freq_seq
    sinusoid = torch.einsum("...,d->...d", position, inv_freq)
    sin = sinusoid.sin()
    cos = sinusoid.cos()
    x1, x2 = torch.chunk(x, 2, dim=-1)

    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class ScaleNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scala = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean_square = (x ** 2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps) * self.scala
        return x


class GAU(nn.Module):
    """GAU block.
    Input shape: batch size x sequence length x model size
    """

    def __init__(
        self,
        hidden_size=768,
        expansion_factor=2,
        s=128,
        norm_type="layer_norm",
        eps=1e-5,
        hidden_act="silu",
    ):
        super().__init__()
        self.s = s
        self.e = int(hidden_size * expansion_factor)
        self.uv = nn.Linear(hidden_size, 2 * self.e + self.s)
        self.weight = nn.Parameter(torch.randn(2, self.s))
        self.bias = nn.Parameter(torch.zeros(2, self.s))
        self.o = nn.Linear(self.e, hidden_size)
        self.LayerNorm = (
            nn.LayerNorm(hidden_size, eps=eps)
            if norm_type == "layer_norm"
            else ScaleNorm(eps=eps)
        )
        self.w = nn.Parameter(torch.randn(2 * 512 - 1))
        self.a = nn.Parameter(torch.randn(1, self.s))
        self.b = nn.Parameter(torch.randn(1, self.s))
        self.act_fn = ACT2FN[hidden_act]

        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.w, std=0.02)
        nn.init.normal_(self.a, std=0.02)
        nn.init.normal_(self.b, std=0.02)

    def rel_pos_bias(self, seq_len):
        """Relative position bias."""
        if seq_len <= 512:
            # Construct Toeplitz matrix directly when the sequence length is less than 512
            t = F.pad(self.w[: 2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        else:
            # Construct Toeplitz matrix using RoPE when the sequence length is over 512.
            a = rope(self.a.repeat(seq_len, 1), dim=0)
            b = rope(self.b.repeat(seq_len, 1), dim=0)
            t = torch.einsum("mk,nk ->mn", a, b)

        return t

    def forward(self, x, attention_mask=None, output_attentions=False, causal=False):
        seq_len = x.shape[1]
        shortcut, x = x, self.LayerNorm(x)
        uv = self.uv(x)
        u, v, base = torch.split(self.act_fn(
            uv), [self.e, self.e, self.s], dim=-1)
        # Generate Query (q) and Key (k) from base.
        base = torch.einsum("...r,hr->...hr", base, self.weight) + self.bias
        base = rope(base, dim=1)
        q, k = torch.unbind(base, dim=-2)
        # Calculate the quadratic attention.
        qk = torch.einsum("bnd,bmd->bnm", q, k)

        bias = self.rel_pos_bias(seq_len)
        kernel = torch.square(torch.relu(qk / seq_len + bias))
        # attention_mask
        if attention_mask is not None:
            assert attention_mask.ndim == 2
            attn_mask = (
                attention_mask[:, None, :] * attention_mask[:, :, None]
            ).type_as(x)
            kernel *= attn_mask

        if causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0)
            kernel *= causal_mask

        x = u * torch.einsum("bnm,bme->bne", kernel, v)
        x = self.o(x)
        if output_attentions:
            return x + shortcut, kernel
        return (x + shortcut,)
