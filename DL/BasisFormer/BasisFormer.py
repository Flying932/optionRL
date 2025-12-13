# BasisFormer.py
# 可学习基底分支(BasisFormer)+ 强化学习特征分支封装
# 参考: VARMA.py / BasisFormer 论文 / 官方 model.py

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from torch.utils.data import DataLoader
import time


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    计算掩码后的 MSE 损失。
    :param reduction: 损失的聚合方式, 可选 'mean' (默认), 'sum', 或 'none'。
    """
    mask_float = mask.float()
    
    # 1. 计算平方误差
    squared_error = (pred - target)**2
    
    # 2. 应用掩码：将不可信位置的误差置为 0
    masked_squared_error = squared_error * mask_float
    
    # 3. 根据 reduction 参数进行聚合
    if reduction == 'none':
        return masked_squared_error
    
    total_error_sum = masked_squared_error.sum()
    
    if reduction == 'sum':
        return total_error_sum
    
    if reduction == 'mean':
        # 标准化：除以有效点的数量
        num_reliable_points = mask_float.sum().clamp_min(1.0) 
        return total_error_sum / num_reliable_points
        
    raise ValueError(f"Invalid reduction parameter: {reduction}. Must be 'none', 'sum', or 'mean'.")

# ===============================
# 配置
# ===============================

@dataclass
class BasisFormerConfig:
    c_in: int                 # 特征维度 C
    seq_len: int=32              # 输入序列长度 I
    pred_len: int=8             # 预测长度 O(如果只做 RL,可以设成一个小值,比如 8 或 16)
    d_model: int = 128        # 通道嵌入维度
    heads: int = 4            # basis 头数 k
    basis_nums: int = 8       # 可学习基底个数 N
    block_nums: int = 2       # BCAB 块数
    bottle: int = 4           # MLP_bottle 压缩比例
    map_bottleneck: int = 32  # map_MLP 的 bottleneck 维度
    tau: float = 0.07         # 对比学习温度
    is_MS: bool = False       # 是否开启 multi-series 模式(可以先关掉)
    input_channel: int = 0    # multi-series 对应的通道数(不开 MS 可以忽略)


# ===============================
# 工具模块: MLP_bottle
# ===============================

class MLPBottle(nn.Module):
    """
    模拟原实现里的 MLP_bottle:
    输入 (..., in_dim),输出 (..., out_dim),中间有 bottleneck 维度.
    """
    def __init__(self, in_dim: int, out_dim: int, bottleneck_dim: int, bias: bool = True):
        super().__init__()
        bottleneck_dim = max(1, bottleneck_dim)
        self.net = nn.Sequential(
            wn(nn.Linear(in_dim, bottleneck_dim, bias=bias)),
            nn.GELU(),
            wn(nn.Linear(bottleneck_dim, bottleneck_dim, bias=bias)),
            nn.GELU(),
            wn(nn.Linear(bottleneck_dim, out_dim, bias=bias)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim)
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        out = self.net(x_flat)
        out = out.view(*orig_shape, -1)
        return out


# ===============================
# BCAB / CoefNet: 系列-基底双向交互 + 系数计算
# ===============================

class CrossAttentionBlock(nn.Module):
    """
    一个标准 Cross-Attention + FFN block:
    query 来自 q,key/value 来自 kv.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout,
        )
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

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q:  (B, L_q, D)
        # kv: (B, L_kv, D)
        attn_out, attn_weights = self.attn(q, kv, kv, need_weights=True)
        x = self.norm1(q + self.dropout(attn_out))
        y = self.ff(x)
        y = self.norm2(x + y)
        return y, attn_weights


class BCAB(nn.Module):
    """
    Bidirectional Cross-Attention Block (BCAB):
    - series 作为 query,basis 作为 key/value
    - basis 作为 query,series 作为 key/value
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.series_to_basis = CrossAttentionBlock(d_model, n_heads, d_ff, dropout)
        self.basis_to_series = CrossAttentionBlock(d_model, n_heads, d_ff, dropout)

    def forward(
        self,
        series_tokens: torch.Tensor,  # (B, C, D)
        basis_tokens: torch.Tensor,   # (B, N, D)
    ):
        # basis->series
        series_new, attn_series = self.basis_to_series(series_tokens, basis_tokens)
        # series->basis
        basis_new, attn_basis = self.series_to_basis(basis_tokens, series_tokens)
        return series_new, basis_new, attn_series, attn_basis


class CoefNet(nn.Module):
    """
    将多层 BCAB 堆叠,然后对最终的 series_tokens / basis_tokens 做 head-wise 内积,
    得到 (B, k, C, N) 的系数矩阵 score.
    """
    def __init__(self, blocks: int, d_model: int, heads: int, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            BCAB(d_model, heads, d_ff=d_ff, dropout=dropout) for _ in range(blocks)
        ])
        self.heads = heads
        self.d_model = d_model

    def forward(
        self,
        basis_tokens: torch.Tensor,   # (B, N, D)
        series_tokens: torch.Tensor,  # (B, C, D)
    ):
        x = series_tokens
        b = basis_tokens
        attn_series_last = None
        attn_basis_last = None

        for layer in self.layers:
            x, b, attn_series_last, attn_basis_last = layer(x, b)

        B, C, D = x.shape
        _, N, D2 = b.shape
        assert D == D2, "d_model mismatch between series and basis"
        H = self.heads
        assert D % H == 0, "d_model must be divisible by heads"
        d_head = D // H

        # 拆成多头维度,然后计算 head-wise 内积
        x_h = x.view(B, C, H, d_head)       # (B, C, H, d_head)
        b_h = b.view(B, N, H, d_head)       # (B, N, H, d_head)

        # score_{B,h,C,N} = <x_h, b_h>
        score = torch.einsum("bchd,bnhd->bhcn", x_h, b_h) / math.sqrt(d_head)

        return score, attn_series_last, attn_basis_last, x, b


# ===============================
# BasisFormer 主体
# ===============================

class BasisFormer(nn.Module):
    """
    BasisFormer:
    - 使用 map_MLP 生成可学习基底 m(t),长度为 seq_len + pred_len,维度为 N.
    - 使用 CoefNet(BCAB 堆叠) 计算每条序列 / 每个基底的系数 score (B,k,C,N).
    - 通过 basis_y + score 生成预测序列.
    - 训练时还提供:
        * m 的平滑正则 l_smooth
        * 基底系数的 InfoNCE 对比损失 l_entropy
    - 这里同时提供一个 encode_coef() 用于强化学习特征抽取.
    """

    def __init__(self, cfg: BasisFormerConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.k = cfg.heads

        # 基的个数
        self.N = cfg.basis_nums

        # 输入序列的长度
        self.seq_len = cfg.seq_len

        # 预测长度
        self.pred_len = cfg.pred_len

        self.tau = cfg.tau
        self.eps = 1e-5
        self.is_MS = cfg.is_MS

        # 系数网络
        self.coefnet = CoefNet(
            blocks=cfg.block_nums,
            d_model=cfg.d_model,
            heads=cfg.heads,
        )

        # MLP_bottle 部分:做时间维度的重排
        self.MLP_x = MLPBottle(
            cfg.seq_len,
            cfg.heads * (cfg.seq_len // cfg.heads),
            max(1, cfg.seq_len // cfg.bottle),
        )
        self.MLP_y = MLPBottle(
            cfg.pred_len,
            cfg.heads * (cfg.pred_len // cfg.heads),
            max(1, cfg.pred_len // cfg.bottle),
        )
        self.MLP_sx = MLPBottle(
            cfg.heads * (cfg.seq_len // cfg.heads),
            cfg.seq_len,
            max(1, cfg.seq_len // cfg.bottle),
        )
        self.MLP_sy = MLPBottle(
            cfg.heads * (cfg.pred_len // cfg.heads),
            cfg.pred_len,
            max(1, cfg.pred_len // cfg.bottle),
        )

        # 将时间长度映射到 d_model
        self.project1 = wn(nn.Linear(cfg.seq_len, cfg.d_model))
        self.project2 = wn(nn.Linear(cfg.seq_len, cfg.d_model))
        self.project3 = wn(nn.Linear(cfg.pred_len, cfg.d_model))
        self.project4 = wn(nn.Linear(cfg.pred_len, cfg.d_model))

        # 时间戳 -> 基底向量 m
        self.map_MLP = MLPBottle(
            1,
            self.N * (self.seq_len + self.pred_len),
            cfg.map_bottleneck,
            bias=True,
        )

        # multi-series 相关(可先不启用)
        if self.is_MS and cfg.input_channel > 0:
            self.MLP_MS = wn(nn.Linear(cfg.input_channel, 1))
            self.mean_MS = wn(nn.Linear(cfg.input_channel, 1))
            self.std_MS = wn(nn.Linear(cfg.input_channel, 1))
        else:
            self.MLP_MS = None
            self.mean_MS = None
            self.std_MS = None

        # 平滑矩阵: 三点差分 [-1, 2, -1]
        arr = torch.zeros((self.seq_len + self.pred_len - 2, self.seq_len + self.pred_len))
        for i in range(self.seq_len + self.pred_len - 2):
            arr[i, i] = -1.0
            arr[i, i + 1] = 2.0
            arr[i, i + 2] = -1.0
        self.register_buffer("smooth_mat", arr)

    # ------------ 内部: 构造时间戳 ------------
    def _build_mark_for_basis(
        self,
        x: torch.Tensor,
        mark: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        原实现只用 mark[:, 0] 的时间戳,这里做一个兼容:
        - mark 为 None 时,使用全 0 占位(基底纯可学习,和具体时间无关)
        - mark 为 (B, L) 或 (B, L, D) 时,使用第一个时间步的第一个维度
        """
        B, L, _ = x.shape
        if mark is None:
            mark_in = torch.zeros(B, 1, 1, device=x.device)  # (B,1,1)
        else:
            if mark.dim() == 2:
                # (B, L)
                mark_in = mark[:, 0:1].unsqueeze(-1)         # (B,1,1)
            elif mark.dim() == 3:
                # (B, L, D)
                mark_in = mark[:, 0:1, 0:1]                  # (B,1,1)
            else:
                raise ValueError("mark must be (B, L) or (B, L, D)")
        return mark_in

    # ------------ 用于 RL 的特征编码接口 ------------
    def encode_coef(
        self,
        x: torch.Tensor,
        mark: Optional[torch.Tensor] = None,
    ):
        """
        只用历史序列 x 计算基底系数,用于 RL 特征提取.

        参数:
            x:    (B, L, C), L = seq_len
            mark: (B, L) 或 (B, L, D) 或 None

        返回:
            score:        (B, k, C, N)   系数矩阵
            basis_tokens: (B, N, d)      最终的基底表示
            series_tokens:(B, C, d)      最终的序列表示
            mean_x:       (B, 1, C)      时间维上的均值
            std_x:        (B, 1, C)      时间维上的标准差
        """
        B, L, C = x.shape
        assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got {L}"

        # 时间归一化 (沿时间维求均值/方差)
        mean_x = x.mean(dim=1, keepdim=True)          # (B,1,C)
        std_x = x.std(dim=1, keepdim=True)            # (B,1,C)
        feat = (x - mean_x) / (std_x + self.eps)      # (B,L,C)

        # (B,C,L) -> (B,C,d_model)
        feat = feat.permute(0, 2, 1)                  # (B,C,L)
        feat = self.project1(feat)                    # (B,C,d)

        # 基底 m: (B, L_total, N)
        mark_in = self._build_mark_for_basis(x, mark) # (B,1,1)
        m = self.map_MLP(mark_in).reshape(
            B, self.seq_len + self.pred_len, self.N
        )                                             # (B,L_total,N)
        m = m / torch.sqrt((m ** 2).sum(dim=1, keepdim=True) + self.eps)

        raw_m1 = m[:, :self.seq_len].permute(0, 2, 1) # (B,N,L)
        m1 = self.project2(raw_m1)                    # (B,N,d)

        score, attn_x, attn_b, series_tokens, basis_tokens = self.coefnet(m1, feat)

        return score, basis_tokens, series_tokens, mean_x, std_x
    
    # ------------ 预测 / 预训练前向 (可选用于预训练) ------------
    def forward(
        self,
        x: torch.Tensor,
        mark: Optional[torch.Tensor],
        y: Optional[torch.Tensor] = None,
        train: bool = True,
        y_mark: Optional[torch.Tensor] = None,
    ):
        """
        x: (B, seq_len, C)
        y: (B, pred_len, C) - 训练用真实未来
        mark, y_mark: 时间戳特征(这里只严格用到 mark 的第一个时间点)

        训练模式:
            返回: 
                output:   (B, pred_len, C)
                l_entropy: 对比损失(InfoNCE)
                l_smooth:  平滑正则
                attn_x1, attn_x2, attn_y1, attn_y2: 若想可视化注意力可以用
        推理模式 (train=False 或 y=None):
            返回:
                output: (B, pred_len, C)
                score:  (B, k, C, N)
                m_all:  (B, seq_len+pred_len, N)
        """
        B, L, C = x.shape
        assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got {L}"

        # ---- 历史部分的归一化 + 表示 ----
        mean_x = x.mean(dim=1, keepdim=True)        # (B,1,C)
        std_x = x.std(dim=1, keepdim=True)          # (B,1,C)
        feat_x = (x - mean_x) / (std_x + self.eps)  # (B,L,C)

        # 反转了时间维度和变量维度, 即变量的整个时间序列被编码成d维的token
        feat_x = feat_x.permute(0, 2, 1)            # (B,C,L)
        feat_x = self.project1(feat_x)              # (B,C,d)

        # ---- 基底 m 的构造 ----
        # map_MLP输入的是(B, 1, 1)的时间戳标量, 输出的是(B, L_total * N)
        mark_in = self._build_mark_for_basis(x, mark)  # (B,1,1)
        m = self.map_MLP(mark_in).reshape(
            B, self.seq_len + self.pred_len, self.N
        )                                              # (B,L_total,N)
        m = m / torch.sqrt((m ** 2).sum(dim=1, keepdim=True) + self.eps)

        raw_m1 = m[:, :self.seq_len].permute(0, 2, 1)  # (B,N,seq_len)
        raw_m2 = m[:, self.seq_len:].permute(0, 2, 1)  # (B,N,pred_len)

        m1 = self.project2(raw_m1)                     # (B,N,d)

        # ---- CoefNet 计算历史系数 ----
        score, attn_x1, attn_x2, series_tokens, basis_tokens = self.coefnet(m1, feat_x)
        # score: (B,k,C,N), k是head数

        # multi-series 可选门控
        if self.is_MS and self.MLP_MS is not None:
            # (B,k,C,N) -> (B,k,1,N) 再回到 (B,k,C,N) 形状
            tmp = self.MLP_MS(score.permute(0, 1, 3, 2))  # (B,k,N,1)
            score = tmp.permute(0, 1, 3, 2)               # (B,k,1,N)
        # 否则直接用 score

        # ---- 使用预测段基底 raw_m2 生成 basis_y ----
        base = self.MLP_y(raw_m2).reshape(
            B, self.N, self.k, -1
        ).permute(0, 2, 1, 3)                         # (B,k,N,L'/k)

        # 分解:score (B,k,C,N) * base (B,k,N,L'/k) -> (B,k,C,L'/k) -> (B,C,L')
        # 用score对base上的N个向量进行加权
        out = torch.matmul(score, base)               # (B,k,C,L'/k)
        out = out.permute(0, 2, 1, 3).reshape(B, C, -1)  # (B,C,L')
        out = self.MLP_sy(out).reshape(B, C, -1).permute(0, 2, 1)  # (B,L',C)

        # multi-series 情况下对 mean/std 做线性变换
        if self.is_MS and self.std_MS is not None and self.mean_MS is not None:
            std_x_ = self.std_MS(std_x)   # (B,1,1)
            mean_x_ = self.mean_MS(mean_x)
        else:
            std_x_ = std_x
            mean_x_ = mean_x

        output = out * (std_x_ + self.eps) + mean_x_   # (B,pred_len,C)

        # ---------------- 推理模式 ----------------
        if (not train) or (y is None):
            return output, score, m

        # ---------------- 训练: 平滑正则 + 对比损失 ----------------

        # 1) 平滑正则 (对 m 在时间维做三点差分), x = l - 2
        l_smooth = torch.einsum("xl,bln->xbn", self.smooth_mat, m)  # (L_total-2, B, N)
        l_smooth = l_smooth.abs().mean()

        # 2) 未来段的对比损失
        B2, Ly, Cy = y.shape
        assert B2 == B and Cy == C and Ly == self.pred_len

        mean_y = y.mean(dim=1, keepdim=True)
        std_y = y.std(dim=1, keepdim=True)
        feat_y = (y - mean_y) / (std_y + self.eps)   # (B,L',C)

        feat_y = feat_y.permute(0, 2, 1)             # (B,C,L')
        feat_y = self.project3(feat_y)               # (B,C,d)

        m2 = self.project4(raw_m2)                   # (B,N,d)

        score_y, attn_y1, attn_y2, series_y, basis_y = self.coefnet(m2, feat_y)
        # score_y: (B,k,C,N), 反映未来序列y下, 第n个基底在第h个head的权重

        # InfoNCE-like 对比:
        # logit_q, logit_k: (B,C,N,k)
        # logit_q[b, c]形状(N, k), 第i行是第i个基底在head上的系数向量, 长度为k
        # 历史x下, 第i个基底的表示向量q_i

        logit_q = score.permute(0, 2, 3, 1)

        # logit_k[b, c]的第j行是未来y下, 第j个基底的表示向量k_j
        logit_k = score_y.permute(0, 2, 3, 1)
        
        # bmm: 批量矩阵相乘
        # logit_q[b, c]的第i行和logit_k[b, c]的第j列相乘则反映其相似性
        # (B*C, N, k) @ (B*C, k, N) -> (B*C, N, N) -> (B*C*N, N)
        l_neg = torch.bmm(
            logit_q.reshape(-1, self.N, self.k),
            logit_k.reshape(-1, self.N, self.k).permute(0, 2, 1)
        ).reshape(-1, self.N)

        # l_neg已经变成了(B*C*N, N)
        labels = torch.arange(0, self.N, dtype=torch.long, device=x.device)

        # labels的维度是(B*C*N, )
        labels = labels.unsqueeze(0).repeat(B * C, 1).reshape(-1)
        
        # 需要实现的: l_neg的维度(B*C*N, N), 共N个样本, 正确标签是labels[i]
        # labels=[0,1,2..., N-1, 0,1,2,...,N-1, ...]
        # 对于给定的[b, c, i], 对应labels_flat的行索引r = bc*N + i
        # 而labels[bc*N + i]  = i

        # cross_entropy, 对某一行softmax, p_j = exp(z_j ) / sum(exp), 然后计算交叉熵损失
        # 最后全部加权求平均值
        l_entropy = F.cross_entropy(l_neg / self.tau, labels)

        return output, l_entropy, l_smooth, attn_x1, attn_x2, attn_y1, attn_y2


# ===============================
# 强化学习特征分支: BasisFormerFeatureBranch
# ===============================

class BasisFormerFeatureBranch(nn.Module):
    """
    给 PPO 用的“可学习基底分支”,接口类似 VARMAFeatureBranch:
    - 输入:  state 形状 (B, T, C),T = seq_len
    - 输出:  特征向量 (B, D_feat),可以拼接到 PPO 的全局 / 局部状态里

    思路:
      1. 调用 basisformer.encode_coef() 得到 score, basis_tokens, series_tokens, mean_x, std_x
      2. 用几个可学习 query 对 basis_tokens 做 attention,得到 (B, nQ, d_model) 的摘要
      3. 展平 + 拼上 mean_x/std_x 的统计量,作为最终特征
    """
    def __init__(self, basisformer: BasisFormer, nQ: int = 3):
        super().__init__()
        self.basisformer = basisformer
        self.nQ = nQ
        d = basisformer.d_model

        # 可学习 query(在 basis token 空间里)
        self.queries = nn.Parameter(
            torch.randn(nQ, d) / math.sqrt(d)
        )

        # 简单的门控,用 basis 的全局信息自适应调节 query
        self.gate = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
            nn.Sigmoid(),
        )

        # 控制门控强度的标量
        self._beta = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor, no_grad: bool=False, mark: Optional[torch.Tensor]=None):
        if no_grad:
            with torch.no_grad():
                return self.old_forward(x, mark)
        return self.old_forward(x, mark)

    def old_forward(
        self,
        x: torch.Tensor,                     # (B, T, C)
        mark: Optional[torch.Tensor] = None # (B, T) or (B, T, D),可选
    ) -> torch.Tensor:
        """
        返回: features, 形状 (B, nQ * d_model + 2)
        其中最后两个维度是 mean/std 的全局统计,方便 PPO 利用整体波动信息.
        """
        B, T, C = x.shape
        assert T == self.basisformer.seq_len, \
            f"BasisFormerFeatureBranch: seq_len mismatch, expect {self.basisformer.seq_len}, got {T}"

        # with torch.no_grad():
        score, basis_tokens, series_tokens, mean_x, std_x = self.basisformer.encode_coef(x, mark)
            # basis_tokens: (B, N, d)

        N = basis_tokens.size(1)
        d = basis_tokens.size(2)

        # 全局 basis 表示,做一个 gate
        g = basis_tokens.mean(dim=1)              # (B,d)
        g = self.gate(g)                          # (B,d)

        # broadcast queries
        Q_base = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B,nQ,d)
        beta = torch.sigmoid(self._beta)

        # gated query: Q = beta * (Q_base ⊙ g) + (1-beta) * Q_base
        Q = beta * (Q_base * g.unsqueeze(1)) + (1.0 - beta) * Q_base  # (B,nQ,d)

        # Attention 权重: (B,nQ,N)
        attn = torch.matmul(
            Q, basis_tokens.transpose(1, 2)
        ) / math.sqrt(d)
        attn = torch.softmax(attn, dim=-1)

        # 聚合后的摘要: (B,nQ,d)
        summary = torch.matmul(attn, basis_tokens)  # (B,nQ,d)

        # 展平
        summary_flat = summary.reshape(B, -1)       # (B, nQ*d)

        # 加上 mean/std 全局统计(对时间 + 通道做一个简单平均)
        mean_global = mean_x.mean(dim=(1, 2), keepdim=False)  # (B,1)
        std_global = std_x.mean(dim=(1, 2), keepdim=False)    # (B,1)

        mean_global = mean_global.unsqueeze(-1)
        std_global = std_global.unsqueeze(-1)

        features = torch.cat(
            [summary_flat, mean_global, std_global],
            dim=-1
        )                                           # (B, nQ*d + 2)

        return features

def train_one_epoch(model, loader, optim, device, grad_clip=1.0):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(device)  # (B,C,L)
        y = y.to(device)  # (B,C,T)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        optim.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

def train_one_epoch_basis(
    model,
    loader,
    optim,
    device,
    lambda_align=0.1,
    lambda_smooth=0.01,
    grad_clip=1.0,
):
    model.train()
    total_loss = 0.0
    total_pred = 0.0
    total_align = 0.0
    total_smooth = 0.0
    n_batch = 0

    for x, y in loader:
        # 假设 DataLoader 给的是 (B, seq_len, C) / (B, pred_len, C)
        x = x.to(device)
        y = y.to(device)

        # 调用 “训练模式” 的 forward，传入 y
        output, l_align, l_smooth, *_ = model(
            x,
            mark=None,   # 你暂时不用时间戳的话就传 None
            y=y,
            train=True,
            y_mark=None,
        )

        # 1) 预测误差
        L_pred = F.mse_loss(output, y)

        # 2) 总 loss = 预测 + 对比 + 平滑
        loss = L_pred + lambda_align * l_align + lambda_smooth * l_smooth

        optim.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_pred += L_pred.item() * bs
        total_align += l_align.item() * bs
        total_smooth += l_smooth.item() * bs
        n_batch += bs

    return {
        "loss": total_loss / n_batch,
        "pred": total_pred / n_batch,
        "align": total_align / n_batch,
        "smooth": total_smooth / n_batch,
    }


def evaluate_basis(model, loader, device):
    model.eval()
    mse, mae, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # 推理模式：不传 y，或 train=False
            output, _, _ = model(x, mark=None, y=None, train=False)

            mse += F.mse_loss(output, y, reduction='sum').item()
            mae += F.l1_loss(output, y, reduction='sum').item()
            n += y.numel()
    return mse / n, mae / n


def evaluate(model, loader, device):
    model.eval()
    mse, mae, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            mse += F.mse_loss(pred, y, reduction='sum').item()
            mae += F.l1_loss(pred, y, reduction='sum').item()
            n += y.numel()
    return mse / n, mae / n

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
    save_path: str = 'best_varmaformer.pth'
):
    """
    主训练函数，负责迭代 epoch，调用训练/评估工具，并实现早停和模型保存。
    
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
    print(f"--- Training VARMAformer on {device} ---")
    print(f"Total epochs: {epochs}, Early Stopping Patience: {patience}")

    best_val_mse = float('inf')
    epochs_no_improve = 0
    
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 1. 训练阶段
        epoch_start_time = time.time()
        train_loss_result = train_one_epoch_basis(model, train_loader, optimizer, device, grad_clip)
        train_loss = train_loss_result['loss']
        
        # 2. 验证阶段
        val_mse, val_mae = evaluate_basis(model, val_loader, device)
        
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
    from dataGen import OptionTrainingDataGenerator
    generator = OptionTrainingDataGenerator(window_size=32, predict_horizon=4)
    train_loader, valid_loader, test_loader = generator.get_data_loader()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    C_IN = 10
    LOOKBACK = 32
    HORIZON = 4
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 300

    cfg = BasisFormerConfig(
        c_in=C_IN,
        seq_len=LOOKBACK,
        pred_len=HORIZON,
        d_model=128,
        heads=4,
        basis_nums=8,
        block_nums=2,
        bottle=4,
        map_bottleneck=32,
        tau=0.07,
        is_MS=False,
        # MS开启才有效
        input_channel=C_IN,
    )

    model = BasisFormer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
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
            save_path='./miniQMT/DL/BasisFormer/basisformer_best_dummy_data.pth'
        )
    
    SAVE_PATH = './miniQMT/DL/VARMA/basisformer_dummy_data.pth'
    state_dict = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    
    

    # # quick sanity run
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # cfg = VARMAConfig(
    #     c_in=12, lookback=96, horizon=24,
    #     patch_size=16, d_model=512, n_heads=8, n_layers=4,
    #     d_ff=1024, dropout=0.1, max_patches=512, p=2, q=2
    # )
    # model = VARMAformer(cfg).to(device)


    # # 构造数据集
    # lookback = 96
    # horizon  = 24

    # root_path = "./miniQMT/DL/VARMA/datasets"
    # data_path = 'WTH.csv'
    # train_ds = Dataset_VARMA(
    #     root_path=root_path,
    #     flag='train',
    #     size=(lookback, lookback//2, horizon),  # label_len 占位，不使用
    #     features='M',                            # 多变量输入/输出
    #     data_path=data_path,
    #     target='OT',
    #     scale=False,                             # 模型内有 instance_norm 建议关掉
    # )

    # val_ds = Dataset_VARMA(root_path=root_path, flag='val',
    #                     size=(lookback, lookback//2, horizon),
    #                     features='M', data_path=data_path, target='OT', scale=False)

    # test_ds = Dataset_VARMA(root_path=root_path, flag='test',
    #                         size=(lookback, lookback//2, horizon),
    #                         features='M', data_path=data_path, target='OT', scale=False)
        

    # train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
    # val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)


    # # train_ds = DummyDataset(length=2000, c_in=cfg.c_in, lookback=cfg.lookback, horizon=cfg.horizon)
    # # val_ds = DummyDataset(length=256, c_in=cfg.c_in, lookback=cfg.lookback, horizon=cfg.horizon, seed=1)
    # # train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    # # val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # for epoch in range(5):
    #     tr_loss = train_one_epoch(model, train_loader, optim, device)
    #     mse, mae = evaluate(model, val_loader, device)
    #     print(f"Epoch {epoch+1}: train_loss={tr_loss:.6f}  val_MSE={mse:.6f}  val_MAE={mae:.6f}")

    # state = torch.rand(2, 32, 26)       # (组合数, 时间窗口, 变量个数)
    # config = VARMAConfig(c_in=26)
    # varma = VARMAformer(config)
    # extractor = VARMAFeatureBranch(varma, nQ=3)

    # res = extractor(state)      # (2, 1538)





# ===============================
# 简单自测 (可选)
# ===============================

if __name__ == "__main_":
    # 假设每个期权组合的局部状态窗口长度为 32,特征维度为 26
    B = 2
    T = 32
    C = 26

    cfg = BasisFormerConfig(
        c_in=C,
        seq_len=T,
        pred_len=16,
        d_model=128,
        heads=4,
        basis_nums=8,
        block_nums=2,
        bottle=4,
        map_bottleneck=32,
        tau=0.07,
        is_MS=False,
        input_channel=C,
    )

    model = BasisFormer(cfg)
    branch = BasisFormerFeatureBranch(model, nQ=3)

    x = torch.randn(B, T, C)
    feat = branch(x)  # (B, 3*128 + 2) = (B, 386)

    print("BasisFormerFeatureBranch output shape:", feat.shape)
