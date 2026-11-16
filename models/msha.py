import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleBlock(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation, groups=dim)
        self.pw = nn.Conv1d(dim, dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        z = x.unsqueeze(-1)
        z = F.gelu(self.pw(self.dw(z))).squeeze(-1)
        return self.norm(z + x)

class MultiScaleHierarchicalAttention(nn.Module):
    def __init__(self, dim=512, heads=4, layers=6, out_dim=256):
        super().__init__()
        self.layers = nn.ModuleList(
            [ScaleBlock(dim, min(2 ** i, 8)) for i in range(layers)]
        )
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.proj = nn.Linear(dim * (layers + 1), out_dim)

    def forward(self, x):
        cur = x
        features = [cur]
        for block in self.layers:
            cur = block(cur)
            q = cur.unsqueeze(1)
            cur = self.attn(q, q, q, need_weights=False)[0].squeeze(1)
            features.append(cur)
        return F.normalize(self.proj(torch.cat(features, dim=-1)), dim=-1)
