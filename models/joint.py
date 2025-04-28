"""
中间连接层
"""
import torch
from torch import nn
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import math
class Joint(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=568, device='cuda'):
        super().__init__()
        self.device = device

        # 添加输入归一化
        self.input_norm = nn.LayerNorm(in_channels)

        # 使用残差连接
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1)  # 增加dropout
        )

        self.proj = nn.Linear(hidden_channels, out_channels)
        self.attn = Block(out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, mask=None):
        x = x.to(self.device)
        # 输入归一化
        x = self.input_norm(x)

        # 残差连接
        identity = x
        x = self.mlp(x)
        x = self.proj(x)

        # 确保值在合理范围内
        x = torch.clamp(x, -100, 100)

        x = self.attn(x, mask=mask)
        x = self.norm(x)

        return x



class Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):  # 增加dropout率
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 动态计算scale
        self.scale = nn.Parameter(torch.ones(1) / math.sqrt(out_channels))

        self.q_linear = nn.Linear(in_channels, out_channels)
        self.k_linear = nn.Linear(in_channels, out_channels)
        self.v_linear = nn.Linear(in_channels, out_channels)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(out_channels)
        self.modify = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

        # 增加中间维度，改善信息流动
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 4, out_channels)
        )
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(self, x, mask=None):
        identity = x

        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # 使用可学习的scale
        attn_scores = (q @ k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights)  # 对attention weights应用dropout
        attn_output = attn_weights @ v

        # 第一个残差连接
        x = identity + self.dropout(attn_output)
        x = self.norm1(x)

        # 第二个残差连接
        identity = x
        x = self.ffn(x)
        x = identity + self.dropout(x)
        x = self.norm2(x)

        return x

