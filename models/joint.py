"""
中间连接层
"""
import torch
from torch import nn
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Joint(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=568,device ='cuda'):
        super().__init__()
        self.device = device
        self.mlp = nn.Sequential(nn.Linear(in_features=in_channels, out_features = hidden_channels),
                                 nn.ReLU(),
                                 )
        self.attn = nn.Sequential(Block(in_channels=hidden_channels, out_channels=out_channels),
                                  nn.ReLU(),
                                  )

    def forward(self, x):
        x = x.to(self.device)
        x = self.mlp(x)
        return self.attn(x)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.q_linear = nn.Linear(in_channels, out_channels)
        self.k_linear = nn.Linear(in_channels, out_channels)
        self.v_linear = nn.Linear(in_channels, out_channels)
        self.softmax = nn.Softmax(dim=-1)

        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 加 LayerNorm，稳定训练
        self.norm1 = nn.LayerNorm(out_channels)

        # 残差的时候报错了，需要改改维度
        self.modify = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.norm2 = nn.LayerNorm(out_channels)
    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        attn_weights = self.softmax(q@k.transpose(-1, -2)/(torch.sqrt(torch.tensor(self.out_channels, dtype=torch.float32))))
        attn_output =  attn_weights @ v

        # print(x.shape)
        # print(attn_output.shape)
        x_modify = self.modify(x)       # 修改维度
        x = self.norm1(x_modify + self.dropout(attn_output))   # 残差连接
        fnn = self.ffn(x)
        x = self.norm2(x + self.dropout(fnn))   # 残差连接
        return x
