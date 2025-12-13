import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================
# ① Transformer 模块
# ======================
class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, T, D]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x


# ======================
# ② Mamba 模块 (极简实现)
# ======================
class SimpleMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.B_proj = nn.Linear(dim, dim)
        self.C_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        outs = []

        for t in range(T):
            u_t = x[:, t, :]  # [B, D]
            B_t = self.B_proj(u_t)
            C_t = self.C_proj(u_t)
            g_t = torch.sigmoid(self.gate(u_t))  # 选择性门控

            # 状态更新 (简化版离散 SSM)
            h = torch.tanh(h @ self.A.T + B_t * g_t)
            y_t = C_t * h
            outs.append(y_t.unsqueeze(1))

        y = torch.cat(outs, dim=1)  # [B, T, D]
        return self.norm(y)


# ======================
# ③ 对比模型封装
# ======================
class CompareModels(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.transformer = SimpleTransformerBlock(dim, num_heads)
        self.mamba = SimpleMambaBlock(dim)

    def forward(self, x):
        out_trans = self.transformer(x)
        out_mamba = self.mamba(x)
        return out_trans, out_mamba


# ======================
# ④ 测试代码
# ======================
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, D = 2, 5, 8  # batch=2, 序列长=5, 维度=8
    x = torch.randn(B, T, D)

    model = CompareModels(dim=D)
    out_t, out_m = model(x)

    print("输入 x:\n", x)
    print("\nTransformer 输出:\n", out_t)
    print("\nMamba 输出:\n", out_m)
