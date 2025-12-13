"""
    本模块实现的是按照变量的因果“时间注意力”选择滞后.

    [主要思想]: iTransformer中提到, 一个事件影响不同的时序变量可能是存在一个先后逻辑的
                同一时刻的变量并不来自于同一事件, 因此不能用同一时间的变量进行建模.
        在期权中, 有可能存在下面的逻辑关系:
            宏观变化 -> 影响波动率 -> 影响期权价格, 但时间并不是同时
        
        因此, 我们首先构建了一个可学习的滞后模块.
    
    [目标]: 在因果约束(只看过去)条件下, 为每个变量在最近T根K线的窗口中学习出一组滞后权重 alpha_(i, t)
    X_(i, t) = sum(k = 0, 1, ..., T - 1) [ alpha_(i, k) * x(i, t - k) ]

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LagAlign(nn.Module):
    """
    可学习滞后对齐(因果、按变量)
    输入:  X [B, V, T]  —— T 为窗口长度(仅过去)
    输出:  Y [B, V]     —— 每变量一个对齐后的标量
         W [B, V, T]   —— 可解释的滞后权重(可用于可视化)
    """
    def __init__(self, V, T, d=8, L_max=None, temperature=1.0, use_gumbel=False):
        super().__init__()
        # V, T是变量个数、窗口长度
        self.V, self.T = V, T

        # 注意力隐藏维度
        self.d = d

        # softmax温度, 值越大越尖锐, 等于1是普通的softmax
        self.temperature = temperature

        # 若为True, 则采用Gumbel-Softmax离散采样
        self.use_gumbel = use_gumbel

        # 允许选择的最大滞后
        self.L_max = T - 1 if L_max is None else L_max

        assert 0 <= self.L_max <= T-1

        # info: 下面的key和query都是可以改进的, 可以改成若干个局部 -> d维

        # 对每个变量的每个时间位置,提取一个 d 维 key
        self.key = nn.Linear(1, d)

        # 以“最新一根bar”的值为 query(也可改成最近几根的函数)
        self.query = nn.Linear(1, d)

        # 可选：给每个滞后位置一个可学习位置偏置(鼓励集中或平滑)
        self.pos_bias = nn.Parameter(torch.zeros(1, 1, T))

        # 生成因果 mask,只允许最后 L_max 根内的索引, 即最远只能从结束位置回看多少根K线
        mask = torch.ones(1, 1, T, dtype=torch.bool)  # True=keep
        # 只保留 [T-1-L_max, ..., T-1]
        if self.L_max < T - 1:
            mask[..., :T - 1-self.L_max] = False
        self.register_buffer("valid_mask", mask)

    def forward(self, X):
        # X: [B,V,T] -> 变形为 [B*V, T, 1]
        B, V, T = X.shape
        x = X.reshape(B*V, T, 1)

        # 每一个时间位置构建一个K, [B * V, T, 1] -> [B * V, T, d]
        K = self.key(x)

        # 最后一步的查询: [B * V, 1] -> [B * V, d] -> [B * V, 1, d]
        q = self.query(x[:, -1, :]).unsqueeze(1)  # [B * V, 1, d]

        # scaled dot-product over time -> logits: [B*V, 1, T]
        logits = (q * K).sum(-1) / math.sqrt(self.d)  # [B*V, T]
        logits = logits + self.pos_bias  # broadcast

        # mask非法滞后位(保持因果 & 限定最大滞后)
        big_neg = torch.finfo(logits.dtype).min
        logits_masked = torch.where(
            self.valid_mask.expand_as(logits),
            logits, torch.full_like(logits, big_neg)
        )

        # 归一化成权重
        if self.use_gumbel:
            W = F.gumbel_softmax(logits_masked / self.temperature, tau=1.0, hard=False, dim=-1)
        else:
            W = F.softmax(logits_masked / self.temperature, dim=-1)

        # 加权汇总得到对齐后的值: [B*V]
        y = (W * X.reshape(B*V, T)).sum(-1)

        # 还原形状
        Y = y.view(B, V)              # [B, V]
        W = W.view(B, V, T)           # [B, V, T] 方便可视化
        return Y, W





class LagAlignPro(nn.Module):
    """
    单变量因果注意力的“可学习滞后对齐”层(可选上下文、局部patch、位置偏置、温度 / Gumbel-Softmax)。
    作用：对 X[B,V,T] 的每个变量, 在允许的滞后集合(L_min..L_max)内学习一组权重 W[B,V,T],
         将该变量的“有效滞后表示”对齐到当前时刻,输出对齐后的 Y。
    
    Args:
        V (int): 变量数(仅用于报错友好;本层对 V 无特定参数化要求)
        T (int): 窗口长度(时间步个数)
        d (int): q/k 的隐维(patch 经 MLP 后的向量维度)
        patch (int): 用于构造 q/k 的局部片段长度(>=1;片段尾对齐当前索引,严格因果)
        L_min (int): 允许的最小滞后(例如 1 表示禁止 τ=0,也就是不能用“最新一根”)
        L_max (Optional[int]): 允许的最大滞后(默认 T-1)
        value_dim (int): 对齐后每个变量输出的通道数(=1 时直接对原值加权;>1 时使用 value 投影)
        c_ctx (int): 额外上下文 ctx 的维度(如 DTE/时钟/仓位预算等),若无上下文传 0
        temperature (float): softmax 温度(越小越尖锐)
        use_gumbel (bool): 是否用 Gumbel-Softmax(训练期可近似离散选择)
        position_bias (bool): 是否使用可学习的相对位置偏置(对滞后位置打先验)
    """

    def __init__(
        self,
        V: int,
        T: int,
        d: int = 32,
        patch: int = 3,
        L_min: int = 0,
        L_max: int = None,
        value_dim: int = 1,
        c_ctx: int = 0,
        temperature: float = 0.7,
        use_gumbel: bool = False,
        position_bias: bool = True,
    ):
        super().__init__()
        assert T >= 1, "T must be >= 1"
        assert patch >= 1 and patch <= T, "patch must satisfy 1 <= patch <= T"
        assert 0 <= L_min <= T - 1, "L_min must be in [0, T-1]"
        L_max = T - 1 if L_max is None else L_max
        assert L_min <= L_max <= T - 1, "Need L_min <= L_max <= T-1"

        self.V = V
        self.T = T
        self.d = d
        self.patch = patch
        self.L_min = L_min
        self.L_max = L_max
        self.value_dim = value_dim
        self.c_ctx = c_ctx
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        self.position_bias_flag = position_bias

        # q/k 的非线性投影：patch( [+ ctx] ) -> d
        self.key_mlp = nn.Sequential(
            nn.Linear(patch, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, d),
        )
        q_in_dim = patch + (c_ctx if c_ctx > 0 else 0)
        self.query_mlp = nn.Sequential(
            nn.Linear(q_in_dim, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, d),
        )

        # 可学习的相对位置偏置(按时间维 T)
        if position_bias:
            self.pos_bias = nn.Parameter(torch.zeros(1, 1, T))
        else:
            self.register_parameter("pos_bias", None)

        # value 投影(当需要多通道输出时)
        if value_dim > 1:
            self.value_proj = nn.Linear(1, value_dim)  # 对每个时间点的标量做通道扩展
        else:
            self.value_proj = None

        # 预构造 “因果 + 滞后范围” 的 mask(True=允许;False=禁止)
        valid_mask = self.build_causal_lag_mask(T=self.T, L_min=self.L_min, L_max=self.L_max)
        self.register_buffer("valid_mask", valid_mask)  # [1,1,T]

    @staticmethod
    def build_causal_lag_mask(T: int, L_min: int, L_max: int, device=None, dtype=torch.bool) -> torch.Tensor:
        """
        允许的时间索引区间：i ∈ [T-1-L_max,  T-1-L_min]
        对应滞后：τ ∈ [L_min, L_max],  τ = T-1-i
        """
        assert 0 <= L_min <= L_max <= T - 1
        mask = torch.zeros(1, 1, T, dtype=dtype, device=device)  # [1,1,T]
        lo = T - 1 - L_max
        hi = T - 1 - L_min
        mask[..., lo : hi + 1] = True
        return mask

    @staticmethod
    def causal_left_pad_last(x: torch.Tensor, pad_left: int) -> torch.Tensor:
        """
        在最后一维做因果左侧 padding：用第一个有效值复制 pad_left 次,防止反射/零带来伪信号。
        x: [..., T]
        return: [..., T + pad_left]
        """
        if pad_left <= 0:
            return x
        first = x[..., :1].expand(*x.shape[:-1], pad_left)
        return torch.cat([first, x], dim=-1)

    @staticmethod
    def unfold_last_dim_causal(x: torch.Tensor, patch: int) -> torch.Tensor:
        """
        将最后一维(时间维)以因果方式展开为长度=patch 的滑动窗口(步长=1),保持原时间长度。
        输入:  x [B,V,T]
        输出:  [B,V,T,patch],第 t 个窗口覆盖 [t-patch+1 .. t](左侧用首值复制补齐)
        """
        B, V, T = x.shape
        pad_left = patch - 1
        x_pad = LagAlignPro.causal_left_pad_last(x, pad_left)  # [B,V,T+patch-1]
        # unfold 在 dimension=2 (时间维) 上生成 [B,V,T,patch]
        patches = x_pad.unfold(dimension=2, size=patch, step=1)  # [B,V,T,patch]
        return patches

    def _compute_q_k(self, X: torch.Tensor, ctx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算 K(每个时间位的 key)和 q(最新位的 query)。
        X:   [B,V,T]
        ctx: [B,C_ctx] 或 None
        return:
            K: [B,V,T,d]
            q: [B,V,1,d]   (方便广播到 T)
        """
        B, V, T = X.shape
        assert T == self.T, f"Expected time length T={self.T}, got {T}"

        # K：所有时刻的因果 patch -> d
        patches = self.unfold_last_dim_causal(X, self.patch)   # [B,V,T,patch]
        K = self.key_mlp(patches)                              # [B,V,T,d]

        # q：最新时刻(T-1)的因果 patch + 可选上下文
        q_local = patches[:, :, -1, :]                         # [B,V,patch]
        if self.c_ctx > 0:
            assert ctx is not None and ctx.shape[-1] == self.c_ctx, \
                f"ctx needed with shape [B,{self.c_ctx}]"
            ctx_exp = ctx.unsqueeze(1).expand(B, V, self.c_ctx)  # [B,V,C_ctx]
            q_in = torch.cat([q_local, ctx_exp], dim=-1)         # [B,V,patch+C_ctx]
        else:
            q_in = q_local                                       # [B,V,patch]

        q = self.query_mlp(q_in).unsqueeze(-2)                 # [B,V,1,d]
        return K, q

    def _masked_softmax_over_time(self, logits: torch.Tensor) -> torch.Tensor:
        """
        对时间维做带 mask 的 softmax / gumbel-softmax。
        logits: [B,V,T]
        return: weights W: [B,V,T]
        """
        # 加位置偏置
        if self.position_bias_flag and self.pos_bias is not None:
            logits = logits + self.pos_bias  # [1,1,T] 广播到 [B,V,T]

        # 应用有效区间 mask(False 的位置置为一个大负数)
        big_neg = -1e9
        logits = logits.masked_fill(~self.valid_mask.expand_as(logits), big_neg)

        # 温度缩放
        logits = logits / self.temperature

        # 归一化
        if self.use_gumbel:
            W = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=-1)  # [B,V,T]
        else:
            W = F.softmax(logits, dim=-1)  # [B,V,T]
        return W

    def forward(self, X: torch.Tensor, ctx = None):
        """
        Args:
            X:   [B,V,T]  —— 历史窗口(仅包含过去,严格因果)
            ctx: [B,C_ctx] 或 None —— 额外上下文(没有就不传)
        Returns:
            Y: [B,V]            (value_dim=1)  或  [B,V,value_dim]  (value_dim>1)
            W: [B,V,T]          —— 对每个变量的滞后权重(可解释可视化)
        """
        assert X.dim() == 3, "X must be [B,V,T]"
        B, V, T = X.shape
        assert V == self.V or self.V <= 0, f"Expected V={self.V}, got {V} (V 仅用于检查/说明)"
        assert T == self.T, f"Expected T={self.T}, got {T}"

        # 1) q/k
        K, q = self._compute_q_k(X, ctx)                     # K:[B,V,T,d], q:[B,V,1,d]

        # 2) 缩放点积打分(沿 d 维)
        #    让 q 在时间维广播成 [B,V,T,d],按 d 求和 → logits:[B,V,T]
        logits = (q * K).sum(dim=-1) / math.sqrt(self.d)     # [B,V,T]

        # 3) 带因果+范围 mask 的 softmax(或 gumbel-softmax)
        W = self._masked_softmax_over_time(logits)           # [B,V,T]

        # 4) 对值做加权汇总
        if self.value_proj is None:  # 只加权原始标量
            Y = (W * X).sum(dim=-1)                          # [B,V]
        else:
            # 先把标量投影成多通道,再按时间加权
            Vvals = self.value_proj(X.unsqueeze(-1))         # [B,V,T, value_dim]
            Y = (W.unsqueeze(-1) * Vvals).sum(dim=-2)        # [B,V,value_dim]

        return Y, W

    @staticmethod
    def entropy(W: torch.Tensor) -> torch.Tensor:
        """
        时间权重的熵(越大越“平均”)。可在训练时加 -lambda * entropy 促进“尖锐选择”。
        W: [B,V,T]
        """
        eps = 1e-8
        H = -(W.clamp_min(eps) * (W.clamp_min(eps)).log()).sum(dim=-1)  # [B,V]
        return H.mean()