# varmaformer.py
# PyTorch implementation of "VARMA-Enhanced Transformer for Time Series Forecasting"
# faithful to the paper's formulas and architecture.
# Author: your_name

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import time
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import List


# ---------------------------
# utils: patchify / depatchify / mask_mse_loss
# ---------------------------

# 本代码将时间维(最后一维)用复制最后一个时间步的方式补充到multiple的整数倍
def replication_pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int]:
    """
    Replicate-pad x along time dim to make length a multiple of 'multiple'.
    x: (B, C, L)
    return: (x_pad, pad_len)
    """
    B, C, L = x.shape
    if L % multiple == 0:
        return x, 0
    
    # 需要补充的长度, 复制最后一个时间步
    need = multiple - (L % multiple)
    last = x[..., -1:]  # (B,C,1)
    pad = last.repeat(1, 1, need)
    return torch.cat([x, pad], dim=-1), need


# 本函数将(B, C, L)分块成N个长度P = patch_size的patch
# 并把每一个patch的(C, P)展平成一个特征向量, 得到(B, N, C * P)
def patchify(x: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, int, int]:
    """
    x: (B, C, L)
    return:
      patches: (B, N, C*P)  -- flatten (C,P) as feature for one patch
      N: number of patches
      P: patch size
    """
    x_pad, _ = replication_pad_to_multiple(x, patch_size)
    B, C, Lp = x_pad.shape
    N = Lp // patch_size
    # (B, C, N, P) -> (B, N, C, P)
    patches = x_pad.view(B, C, N, patch_size).permute(0, 2, 1, 3).contiguous()
    patches = patches.reshape(B, N, C * patch_size)  # (B,N,F), F=C*P
    return patches, N, patch_size

# 把未来的patch预测(B, H, C * P)还原成连续的时间序列(B, C, H * P)
# 并裁剪到目标长度T, 输出(B, C, T)
def depatchify(patches: torch.Tensor, C: int, P: int, T: int) -> torch.Tensor:
    """
    patches: (B, H, C*P), H patches for the future
    Return: (B, C, T) cropped to exact horizon T
    """
    B, H, FP = patches.shape
    assert FP == C * P
    y = patches.view(B, H, C, P).permute(0, 2, 1, 3).contiguous()  # (B,C,H,P)
    y = y.view(B, C, H * P)  # (B,C,H*P)
    return y[..., :T]  # crop to T


# ---------------------------
# Instance Normalization (per sample, per channel)
# ---------------------------

# 本函数实现归一化
def instance_norm(x: torch.Tensor, eps: float=1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x: (B, C, L)
    returns: x_norm, mu, sigma  where mu/sigma are (B,C,1)
    """
    mu = x.mean(dim=-1, keepdim=True)
    sigma = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mu) / sigma, mu, sigma



# ---------------------------
# VARMA-inspired Feature Extractor (VFE)
# ---------------------------

class VFE(nn.Module):
    """
    VFE模块:
        Implements equations (2)(3)(4)(5).
        Input x_patches: (B, N, F) with F = C * P (C是补丁数, P是每个补丁的长度, 捕捉局部依赖)
        Output Z_VARMA: (B, N, D)

    Args:
        in_feat: 每一个patch的特征维度, in_feat = C * P, C个通道, 每一个通道P个时间点, flatten成vector
        d_model: 隐藏维度
        p, q: AR(p), MA(q)
    """
    def __init__(self, in_feat: int, d_model: int, p: int = 2, q: int = 2):
        super().__init__()
        self.p = p
        self.q = q
        self.phi = nn.Parameter(torch.ones(p))    # AR weights φ_i (learnable scalars)
        self.theta = nn.Parameter(torch.ones(q))  # MA weights θ_j (learnable scalars)

        self.in_feat = in_feat

        # 每部分线性投影到d_model // 2, 然后拼接成d_model
        self.proj_ar = nn.Linear(p * in_feat, d_model // 2)
        self.proj_ma = nn.Linear(q * in_feat, d_model // 2)

        self.fuse = nn.Linear(d_model, d_model)   # W_fuse
    
    @staticmethod
    def _shift_right(x: torch.Tensor, steps: int, fill: torch.Tensor) -> torch.Tensor:
        """
        本函数实现向右移动steps位的操作
            Shift x along N-dim to the right by 'steps' with left fill.
            x: (B, N, F), 其中N是时间维度
            fill: (B, 1, F)
        """
        if steps == 0:
            return x
        B, N, F = x.shape
        left = fill.expand(B, steps, F)
        return torch.cat([left, x[:, :N - steps, :]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F = C * P
        B, N, F = x.shape

        # ---- AR part: concat {φ_i * x_{t-i}} for i=1..p ----
        ar_list = []
        # replicate the first patch for left fill (as paper uses replication padding spirit)
        first = x[:, :1, :].detach()

        # 文章中提到的: Z^t = Proj(φ_1 * x_{t-1}, φ_2 * x_{t-2}, ...)
        # 但是这个计算的只是t时刻的Z
        # 而Z^(t - 1) = Proj(φ_1 * x_{t-2}, φ_2 * x_{t-3}, ...)

        # 那么可以: 
        # 1.先整体移动steps(steps = 1, 2, 3, ..., p)
        # 2.加权φ_(steps)
        # 3.投影Proj_MA, Proj_AR

        # 计算AR部分
        for i in range(1, self.p + 1):
            xi = self._shift_right(x, i, first)     # (B,N,F) -> previous i
            ar_list.append(self.phi[i - 1] * xi)    # scalar * patch
        AR = torch.cat(ar_list, dim=-1)             # (B,N,p*F)
        Z_AR = self.proj_ar(AR)                     # (B,N,D/2)
        
        # 计算MA部分
        # ---- MA part: concat {θ_j * ε_{t-j}}, ε_k = x_k - x_{k-1} ----
        eps_all = x - self._shift_right(x, 1, first)          # (B,N,F), ε_0≈0
        zero = torch.zeros_like(eps_all[:, :1, :])
        ma_list = []
        for j in range(1, self.q + 1):
            eps_j = self._shift_right(eps_all, j, zero)       # ε_{t-j}
            ma_list.append(self.theta[j - 1] * eps_j)

        MA = torch.cat(ma_list, dim=-1)                       # (B,N,q*F)
        Z_MA = self.proj_ma(MA)                               # (B,N,D/2)

        # ---- Fuse ----woxia
        Z = torch.cat([Z_AR, Z_MA], dim=-1)                   # (B,N,D)
        Z = self.fuse(Z)                                      # (B,N,D)
        return Z

# ---------------------------
# VE-atten decoder layer (cross-attn only) §3.4
# ---------------------------

class VEAttnLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Gate: two-layer MLP + sigmoid, applied to mean(K) as in eq.(7)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        # β (learnable scalar, clamp via sigmoid to (0,1))
        self._beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor,
                key_padding_mask: torch.Tensor=None,
                ) -> torch.Tensor:
        """
        q: (B, H, D), H是未来的patch个数
        k/v: (B, N, D), N是历史的patch个数
        key_padding_mask: mask
        returns: (B, H, D)
        """
        B, H, D = q.shape

        # Global context gate from keys (mean over patch dim), G是全局上下文向量
        # 注意：如果 k 序列中包含无效值 (例如 Nan 或 0)，这里计算均值会受到污染。
        # 严格来说，计算 mean(k) 时需要排除被 mask 的位置，但这会使代码复杂化。
        # 假设：无效 Patch 被设置为 0，且数量较少，对均值影响有限，或者在前面已被妥善处理

        G = self.gate(k.mean(dim=1))           # (B, D)
        beta = torch.sigmoid(self._beta)       # (0,1)

        # Gate queries: Q' = β (Q ⊙ G) + (1-β) Q
        q_gated = beta * (q * G.unsqueeze(1)) + (1.0 - beta) * q  # (B,H,D)

        # Cross-attention (no self-attention)
        if key_padding_mask is not None:
            attn_out, _ = self.mha(q_gated, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        else:
            attn_out, _ = self.mha(q_gated, k, v, need_weights=False)

        x = self.norm1(q + self.dropout(attn_out))

        y = self.ff(x)
        y = self.norm2(x + y)
        return y


# ---------------------------
# VARMAformer (Decoder-only, Cross-Attention)
# ---------------------------

@dataclass
class VARMAConfig:
    # 变量个数
    c_in: int

    # 时间窗口长度
    lookback: int=32

    # 预测步数T
    horizon: int=4

    # 每一个patch块的长度, 即每一个patch块的时间长度
    patch_size: int = 16

    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 2048
    dropout: float = 0.1
    max_patches: int = 512   # for positional embeddings
    p: int = 2               # AR order
    q: int = 2               # MA order
    target_indices: List[int] = field(default_factory=list)


class VARMAformer(nn.Module):
    """
    Implements Algorithm 1 (paper §3.5) with modules defined in §3.2 and §3.4.
    """
    def __init__(self, cfg: VARMAConfig):
        super().__init__()
        self.cfg = cfg

        C, L = cfg.c_in, cfg.lookback
        P = cfg.patch_size
        self.P = P
        self.C = C

        self.out_patches = math.ceil(cfg.horizon / P)

        self.target_indices = cfg.target_indices
        self.num_targets = len(self.target_indices)

        
        self.op_type_embedding = nn.Embedding(2, cfg.d_model)

        # 每一个patch块的特征维度, 变量C个, 时间长度P
        self.in_feat = C * P

        self.out_feat = self.num_targets * P

        # VFE模块提取特征, Z_ARMA
        self.vfe = VFE(in_feat=self.in_feat, d_model=cfg.d_model, p=cfg.p, q=cfg.q)
        self.patch_embed = nn.Linear(self.in_feat, cfg.d_model)

        # α as learnable gate in [0,1]
        self._alpha = nn.Parameter(torch.tensor(0.3))

        # learnable positional encoding for keys/values (patch positions)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.max_patches, cfg.d_model) * 0.02)

        # decoder layers (VE-atten)
        self.layers = nn.ModuleList([
            VEAttnLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        # learnable query embeddings for "future" patches (H horizons in patch units)
        self.Q = nn.Parameter(torch.randn(self.out_patches, cfg.d_model) * (1.0 / math.sqrt(cfg.d_model)))

        # output projection from decoder states to patch vector (C*P)
        self.proj_out = nn.Linear(cfg.d_model, self.out_feat)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor=None, op_type=None) -> torch.Tensor:
        """
        x.permute(0, 2, 1): (B, C, L) past window, C是变量个数, L是look-back window长度
        mask: 掩码
        returns: (B, C, T) forecast
        """
        x = x.permute(0, 2, 1)
        B, C, L = x.shape
        assert C == self.C

        # 1) InstanceNorm & patchify  (Algorithm 1 lines 5-6)
        x_norm, mu, sigma = instance_norm(x)           # (B,C,L)
        patches, N, P = patchify(x_norm, self.P)       # (B,N,C*P)

        # 2) VFE & fuse into embeddings (lines 7-9)
        z_varma = self.vfe(patches)                    # (B,N,D)
        alpha = torch.sigmoid(self._alpha)             # constrain to [0,1]
        # positional encoding
        assert N <= self.cfg.max_patches, "Increase max_patches for longer lookback or smaller patch_size."

        # 可学习位置编码
        pe = self.pos_emb[:, :N, :]                    # (1,N,D)

        E = self.patch_embed(patches) + alpha * z_varma + pe  # (B,N,D)
        if op_type is not None:
            op_idx = op_type.view(-1).long() # Shape 变成 [1]
            op_emb = self.op_type_embedding(op_idx)
            # op_emb = self.op_type_embedding(op_type.squeeze(-1).long())
            E = E + op_emb.unsqueeze(1)

        K = V = E

        # 3) Decoder with VE-atten (lines 10-14)
        # Q: (H, D)
        Q = self.Q.unsqueeze(0).expand(B, -1, -1)      # (B,H,D)
        for layer in self.layers:
            Q = layer(Q, K, V, key_padding_mask)                         # (B,H,D)

        # 4) Project to patches & depatchify & de-normalize (lines 15-16)
        y_patches = self.proj_out(Q)                   # (B,H,C*P)
        # y = depatchify(y_patches, C=self.C, P=self.P, T=self.cfg.horizon)  # (B,C,T)
        y = depatchify(y_patches, C=self.num_targets, P=self.P, T=self.cfg.horizon)  # (B,C,T)
        target_mu = mu[:, self.target_indices, :]
        target_sigma = sigma[:, self.target_indices, :]
        # y = y * sigma + mu                             # denormalize
        y = y * target_sigma + target_mu
        y = y.permute(0, 2, 1)
        return y

    def encode_tokens(self, x, op_type=None):
        # x: (B, C, L) past window, C是变量个数, L是look-back window长度
        x = x.permute(0, 2, 1)
        B, C, L = x.shape
        x_norm, mu, sigma = instance_norm(x)
        patches, N, P = patchify(x_norm, self.P)
        
        # VAMRA融合特征
        z_varma = self.vfe(patches)                      # (B,N,D)
        pe = self.pos_emb[:, :N, :]
        alpha = torch.sigmoid(self._alpha)
        E = self.patch_embed(patches) + alpha * z_varma + pe

        if op_type is not None:
            op_idx = op_type.view(-1).long() # Shape 变成 [1]
            op_emb = self.op_type_embedding(op_idx)
            # op_emb = self.op_type_embedding(op_type.squeeze(-1).long())
            E = E + op_emb.unsqueeze(1)


        flat_e = E.reshape(E.shape[0], -1)
        
        flat_mu = mu.squeeze(-1)
        flat_sigma = sigma.squeeze(-1)
        flat_stats = torch.cat([flat_mu, flat_sigma], dim=-1)

        return flat_e, flat_stats                        # E: (B,N,D)


# VARMA特征提取分支
class VARMAFeatureBranch(nn.Module):
    def __init__(self, varma: VARMAformer, nQ=3):
        super().__init__()
        self.varma = varma
        D = varma.cfg.d_model

        # 少量查询向量
        self.Q = nn.Parameter(torch.randn(nQ, D) / math.sqrt(D))

        # 门控机制
        self.gate = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D), nn.Sigmoid())

        # beta参数
        self._beta = nn.Parameter(torch.tensor(0.5))
    
    # 套壳一下
    def forward(self, x: torch.Tensor, no_grad: bool=False):
        if no_grad:
            with torch.no_grad():
                return self.raw_forward(x)
        else:
            return self.raw_forward(x)

    # 返回最终得到的时间特征
    def raw_forward(self, x: torch.Tensor):
        # 输入的x的时间维度在前
        x = x.permute(0, 2, 1)
        E, mu, sigma, N = self.varma.encode_tokens(x)     # E: (B,N,D)


        # 向量门控让查询随体制自适应
        g = self.gate(E.mean(dim=1))                      # (B,D)
        Q_base = self.Q.unsqueeze(0).expand(E.size(0), -1, -1)  # (B,nQ,D)
        beta = torch.sigmoid(self._beta)
        
        Q = beta * (Q_base * g.unsqueeze(1)) + (1.0 - beta) * Q_base

        attn = torch.softmax(Q @ E.transpose(1,2) / math.sqrt(E.size(-1)), dim=-1)  # (B,nQ,N)
        O = attn @ E                                      # (B,nQ,D)
        S = O.reshape(E.size(0), -1)                      # (B, nQ*D)
        # 可把 mu/sigma 的通道统计拼进去
        mu_c = mu.mean(dim=1).squeeze(-1)                 # (B,)
        sig_c = sigma.mean(dim=1).squeeze(-1)             # (B,)

        return torch.cat([S, mu_c.unsqueeze(-1), sig_c.unsqueeze(-1)], dim=-1)




class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean



def train_one_epoch(model, loader, optim, device, grad_clip=1.0):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(device)  # (B,C,L)
        y = y.to(device)  # (B,C,T)
        pred = model(x)

        # 损失函数(损失掩码)
        loss = F.mse_loss(pred, y)

        optim.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

def evaluate(model, loader, device, loss_fn=None):
    model.eval()
    if loss_fn is None:
        loss_fn = F.mse_loss
    
    total_squared_error = 0.0      # 用于计算最终的 MSE
    total_absolute_error = 0.0     # 用于计算最终的 MAE
    total_reliable_points = 0      # 累加所有批次中的有效点总数
    
    with torch.no_grad():
        # 确保 loader 返回 (x, y, mask)
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x) # (B, T, C)


            # --- 修改位置 1: 计算 MSE ---
            # 使用 loss_fn (masked_mse_loss) 并设置 reduction='sum' 来获取总平方误差
            current_squared_error_sum = loss_fn(
                pred, 
                y, 
                reduction='sum'
            ).item()
            
            total_squared_error += current_squared_error_sum
            
            # --- 修改位置 2: 计算 MAE ---
            absolute_error = torch.abs(pred - y)
            masked_absolute_error = absolute_error
            total_absolute_error += masked_absolute_error.sum().item()
            
            total_reliable_points = absolute_error.shape[0]
            

    # 最终的 MSE 和 MAE 是总误差除以所有批次的有效点总数
    if total_reliable_points == 0:
        return 0.0, 0.0
        
    final_mse = total_squared_error / total_reliable_points
    final_mae = total_absolute_error / total_reliable_points
    
    return final_mse, final_mae

# ---------------------------
# Main Training Function (新增)
# ---------------------------

def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    test_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    loss_fn: nn.Module, # 实际上在 train_one_epoch 里是 MSELoss，这里只作签名占位
    device: torch.device, 
    epochs: int = 20, 
    grad_clip: float = 1.0,
    patience: int = 5,
    save_path='./miniQMT/DL/VARMA/VARMA_best_dummy_data.pth'
):
    """
    主训练函数，负责迭代 epoch 调用训练/评估工具，并实现早停和模型保存。
    
    :param model: 要训练的 PyTorch 模型
    :param train_loader: 训练集 DataLoader
    :param val_loader: 验证集 DataLoader (用于早停)
    :param test_loader: 测试集 DataLoader (用于最终评估)
    :param optimizer: 优化器
    :param loss_fn: 损失函数 (实际在 train_one_epoch 中使用了 F.mse_loss)
    :param device: 设备 (e.g., 'cuda', 'cpu')
    :param epochs: 最大训练轮数
    :param grad_clip: 梯度裁剪阈值
    :param patience: 早停容忍的 epoch 数
    :param save_path: 最佳模型保存路径
    """
    print(f"Total epochs: {epochs}, Early Stopping Patience: {patience}")

    best_val_mse = float('inf')
    epochs_no_improve = 0
    
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 1. 训练阶段
        epoch_start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip)
        
        # 2. 验证阶段
        val_mse, val_mae = evaluate(model, val_loader, device)
        
        epoch_duration = time.time() - epoch_start_time

        # 3. 打印/记录结果
        print(
            f"Epoch {epoch:02d} | Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f}", 
            end=''
        )

        # 4. 模型保存和早停逻辑 (基于验证集 MSE)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(" -> **Saving Best Model**")
        else:
            epochs_no_improve += 1
            print()
        
        # 5. 早停检查
        if epochs_no_improve >= patience:
            print(f"\n[Early Stop] Validation MSE did not improve for {patience} epochs.")
            break

    # 6. 最终测试阶段 (加载最佳模型并评估)
    print("\n--- Final Evaluation ---")
    try:
        model.load_state_dict(torch.load(save_path))
        model.to(device)
        test_mse, test_mae = evaluate(model, test_loader, device)
        print(f"Loaded best model from '{save_path}'.")
        print(f"Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f}")
    except Exception as e:
        print(f"Error loading best model or running test: {e}")
        # 如果加载失败，用当前模型评估一次
        test_mse, test_mae = evaluate(model, test_loader, device)
        print(f"Using last epoch model for test: Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f}")


    total_time = time.time() - start_time
    print(f"--- Total training finished in {total_time/60:.2f} minutes. ---")
    return best_val_mse # 返回最佳验证 MSE

if __name__ == "__main__":
    C_IN = 10
    LOOKBACK = 32
    HORIZON = 4
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    PATIENCE = 5
    MIN_RATIO = 0.8

    from preTrainDataGen import OptionTrainingDataGenerator
    generator = OptionTrainingDataGenerator(window_size=LOOKBACK, predict_horizon=HORIZON, min_ratio=MIN_RATIO)
    train_loader, valid_loader, test_loader = generator.get_data_loader()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    cfg = VARMAConfig(
        c_in=C_IN, lookback=LOOKBACK, horizon=HORIZON,
        patch_size=16, d_model=128, n_heads=8, n_layers=4,
        d_ff=2048, dropout=0.1, max_patches=512, p=2, q=2
    )

    model = VARMAformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss() # 实际在 train_one_epoch 中使用了 F.mse_loss

    train_model(
            model=model,
            train_loader=train_loader,
            val_loader=valid_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epochs=300,
            grad_clip=1.0,
            patience=5,
            save_path='./miniQMT/DL/preTrain/weights/VARMA_best_dummy_data.pth'
        )
    
    SAVE_PATH = './miniQMT/DL/preTrain/weights/VARMA_best_dummy_data.pth'
    state_dict = torch.load(SAVE_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    print(0 / 0)

