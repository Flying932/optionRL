"""
    本模块实现PPO算法的部分, 这也是整个架构最核心的部分.
    需要对接我们的模拟miniQMT环境

    本PPO算法需要实现:
        输出动作: (LONG, SHORT, CLOSE_ALL, HOLD)
        输出权重: weight in [0, 1]
"""
from math import e
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from windowEnv import windowEnv
import time, json
import sys
import pandas as pd
from datetime import datetime
import pickle
from tools.Norm import Normalization, RewardNormalization, RewardScaling
from preTrain.preMOE import PreMOE
from dataclasses import dataclass
import random

# —— 构造每个样本的权重掩码 —— #
A_HOLD, A_LONG, A_SHORT, A_CLOSE = 0, 1, 2, 3
WEIGHT_BINS_CPU = torch.tensor([0.00, 0.25, 0.50, 0.75, 1.00])  # 离散权重


DESK_PATH = 'C:/Users/David/Desktop'
DESK_PATH = 'C:/Users/Flying/Desktop'


# 输出类, 实现重定向sys.stdout的功能
class outPut():
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "w", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()
    
    def close(self):
        # self.terminal.close()
        self.logfile.close()


# 共享干路 + 双头
class ActorDualHead(nn.Module):
    def __init__(self, state_dim, hidden_dim: int=256, n_actions: int = 4, n_weights: int = 5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, n_actions)
        self.weight_head = nn.Linear(hidden_dim, n_weights)

    def forward(self, state):
        # state: (B, state_dim) 或 (state_dim,)
        state = state.to(dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        z = self.backbone(state)
        logits_action = self.action_head(z)   # (B, n_actions)
        logits_weight = self.weight_head(z)   # (B, 5)
        return logits_action, logits_weight


# 价值网络
class Value(nn.Module):
    def __init__(self, state_dim, hidden_dim: int=256):
        super(Value, self).__init__()
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state):
        return self.value_net(state)

# 轻量级网络实现维度缩减
class FeatureAdapter(nn.Module):
    def __init__(self, input_dim: int=3732, out_dim=128):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(1024, out_dim),
            # nn.Tanh(),
        )
    
    def forward(self, x, train: bool=True):
        if train:
            return self.adapter(x)
        with torch.no_grad():
            return self.adapter(x)

class ViewProjector(nn.Module):
    """
    负责处理单个模型的 (High-Dim, Low-Dim) 并将其融合
    """
    def __init__(self, high_dim, low_dim, out_dim=64):
        super().__init__()
        
        # 1. 压缩高维特征
        self.high_net = nn.Sequential(
            nn.LayerNorm(high_dim),
            nn.Linear(high_dim, out_dim), # 线性压缩
            # nn.Dropout(0.1)
        )
        
        # 2. 嵌入低维统计量
        self.low_net = nn.Sequential(
            nn.LayerNorm(low_dim),
            nn.Linear(low_dim, 32),       # 升维增强
            # nn.Tanh()                     # 赋予非线性
            nn.GELU(),
        )
        
        # 3. 视图融合
        self.fusion = nn.Sequential(
            nn.Linear(out_dim + 32, out_dim),
            nn.LayerNorm(out_dim)
        )
        
    def forward(self, x_high, x_low):
        h = self.high_net(x_high)
        l = self.low_net(x_low)
        # 拼接后融合
        return self.fusion(torch.cat([h, l], dim=-1))


class MultiViewAdapter(nn.Module):
    def __init__(self, 
                 dims_dict: dict,  # 包含各模型维度的字典
                 final_dim: int = 128):
        super().__init__()
        
        # 1. 定义三个独立的投影器 (Trainable)
        # 假设我们想让每个模型贡献 48 维的特征
        view_dim = 48
        
        self.varma_proj = ViewProjector(
            high_dim=dims_dict['varma_high'], 
            low_dim=dims_dict['varma_low'], 
            out_dim=view_dim
        )
        
        self.basis_proj = ViewProjector(
            high_dim=dims_dict['basis_high'], 
            low_dim=dims_dict['basis_low'], 
            out_dim=view_dim
        )
        
        self.itrans_proj = ViewProjector(
            high_dim=dims_dict['itrans_high'], 
            low_dim=dims_dict['itrans_low'], 
            out_dim=view_dim
        )
        
        # Router 的特征直接处理 (因为它本来就是低维高语义)
        self.router_proj = nn.Sequential(
            nn.LayerNorm(dims_dict['router']),
            nn.Linear(dims_dict['router'], 32)
        )
        
        # 2. 最终融合层
        # 输入维度 = 48*3 + 32 = 176
        self.final_net = nn.Sequential(
            nn.Linear(view_dim * 3 + 32, final_dim),
            nn.LayerNorm(final_dim) # 再次强调：不要 Tanh，用 LayerNorm
        )
        
    def raw_forward(self, inputs: dict):
        # inputs 是一个字典，包含所有模型的原始输出
        
        # 1. 并行处理各视图
        v_varma = self.varma_proj(inputs['varma_h'], inputs['varma_l'])
        v_basis = self.basis_proj(inputs['basis_h'], inputs['basis_l'])
        v_itrans = self.itrans_proj(inputs['itrans_h'], inputs['itrans_l'])
        
        v_router = self.router_proj(inputs['router'])
        
        # 2. 拼接
        combined = torch.cat([v_varma, v_basis, v_itrans, v_router], dim=-1)
        
        # 3. 输出
        return self.final_net(combined)
    
    def forward(self, inputs: dict, train: bool=True):
        if train:
            return self.raw_forward(inputs)
        with torch.no_grad():
            return self.raw_forward(inputs)

# PPO-agent
class weightPPO:
    def __init__(self,
                 action_dim: int, 
                 actor_lr: float=3e-4,
                 value_lr: float=5e-4,
                 gamma: float=0.99,
                 clip_eps: float=0.1,
                 k_epochs: int=5,
                 device: str='cpu',
                 check_path: str='./miniQMT/DL/checkout',
                 resume: bool=False,
                 window_size: int=32,
                 pre_len: int=4,
                 ):
        self.device = 'cuda' if torch.cuda.is_available() else device
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.resume = resume
        self.check_path = f'{check_path}/check_data_all.pt'

        self.actor_lr, self.value_lr = actor_lr, value_lr

        self.action_dim = action_dim

        self.actor, self.value, self.opt_a, self.opt_b = None, None, None, None

        self.window_size = window_size

        # 放设备上的权重离散值
        self.WEIGHT_BINS = WEIGHT_BINS_CPU.to(self.device)


        # 预训练基础模型MOE, 加载参数
        self.pre_len = pre_len
        self.extractor = PreMOE(seq_len=self.window_size, pred_len=self.pre_len, n_variates=13, d_router=128).to(self.device)
        self.load_moe_parameters()

        for p in self.extractor.parameters():
            p.requires_grad = False
        self.extractor.eval()

        # 轻量级维度缩减: k -> 128
        self.feature_adapter = None
        self.opt_c = None

        # self.feature_adapter = FeatureAdapter(3732, 128).to(self.device)
        # self.opt_c = optim.Adam(self.feature_adapter.parameters(), lr=actor_lr)

        # 归一化模块
        self.state_norm = None
        self.reward_norm = None

    def extract_features_batch(self, current_state: torch.Tensor, history_state: torch.Tensor, cal_dim: bool = False):
        """
        批量特征提取函数
        输入:
            current_state: (Batch_Size, state_dim)
            history_state: (Batch_Size, window_size, history_dim)
            cal_dim: 是否仅用于计算维度 (True时不进行梯度更新)
        输出:
            norm_features: (Batch_Size, feature_dim)
        """
        # 1. 确保数据在正确的设备上 (通常调用前已经处理好，这里做个双保险)
        if current_state.device != self.device:
            current_state = current_state.to(self.device)
        if history_state.device != self.device:
            history_state = history_state.to(self.device)

        # 2. 切分 Call / Put 的历史数据
        # history_state shape: (Batch, Window, Feat) -> 在 Feat 维度切分
        call_state, put_state = torch.chunk(history_state, chunks=2, dim=2)

        # 3. MOE 编码 (PreMOE 支持 Batch 输入，输出字典中各项均为 (Batch, ...))
        call_dict = self.extractor.encode_tokens(call_state)
        put_dict = self.extractor.encode_tokens(put_state)

        # 4. 首次运行时：动态初始化 Adapter 和 优化器
        if self.feature_adapter is None:
            # 从 batch 中获取维度信息 (取 shape[-1])
            dims_dict = {
                'varma_high': call_dict['varma_h'].shape[-1],
                'varma_low': call_dict['varma_l'].shape[-1],
                'basis_high': call_dict['basis_h'].shape[-1],
                'basis_low': call_dict['basis_l'].shape[-1],
                'itrans_high': call_dict['itrans_h'].shape[-1],
                'itrans_low': call_dict['itrans_l'].shape[-1],
                'router': call_dict['router'].shape[-1]
            }
            
            # 初始化多视图适配器
            self.feature_adapter = MultiViewAdapter(dims_dict, final_dim=128).to(self.device)
            
            # 重要：初始化适配器的优化器
            self.opt_c = optim.Adam(self.feature_adapter.parameters(), lr=self.actor_lr)
            print(f"[Info] FeatureAdapter & Opt_c initialized via Batch mode.")

        # 5. 维度缩减 (Adapter Forward)
        train = not cal_dim
        reduce_call = self.feature_adapter(call_dict, train=train) # (Batch, 128)
        reduce_put = self.feature_adapter(put_dict, train=train)   # (Batch, 128)

        # 6. 特征拼接
        # current_state: (Batch, Curr_Dim)
        # 拼接后: (Batch, Curr_Dim + 128 + 128)
        features = torch.cat([current_state, reduce_call, reduce_put], dim=-1)

        # 7. 状态归一化 (Running Mean Std)
        # 如果是第一次运行，初始化归一化层
        if self.state_norm is None:
            # 注意：Normalization 初始化通常需要输入特征的 shape (除去 Batch 维)
            # 或者是直接传入一个 Batch 数据，内部会自动获取 shape
            self.init_norm_state(features) 
        
        # 返回归一化后的 Batch 特征
        return self.state_norm(features)
    
    def load_moe_parameters(self):
        try:
            SAVE_PATH = f'./miniQMT/DL/preTrain/weights/preMOE_best_dummy_data_{self.window_size}_{self.pre_len}.pth'
            state_dict = torch.load(SAVE_PATH, map_location=self.device)
            self.extractor.load_state_dict(state_dict)

            print(f"[Info] 加载MOE参数成功~")
        except Exception as e:
            print(f"[Info] 加载MOE参数失败, e = {e}")

    def init_norm_state(self, x: torch.Tensor):
        self.state_norm = Normalization(x.shape)
    
    def init_norm_reward(self):
        self.reward_norm = RewardScaling(shape=(1,), gamma=self.gamma)

    def exe_reward_norm(self, x: float):
        if self.reward_norm is None:
            self.init_norm_reward()
        
        x = torch.tensor([x], dtype=torch.float32)
        return self.reward_norm(x).item()


    # 状态归一化
    def extract_features(self, current_state: list, history_state: list, cal_dim: bool=False):
        current_state = torch.tensor(current_state, dtype=torch.float32, device=self.device)
        history_state = torch.tensor(history_state, dtype=torch.float32, device=self.device)
        
        history_state = history_state.unsqueeze(0)
        call_state, put_state= torch.chunk(history_state, chunks=2, dim=2)
        call_dict = self.extractor.encode_tokens(call_state)
        put_dict = self.extractor.encode_tokens(put_state)

        dims_dict = {
            'varma_high': call_dict['varma_h'].shape[-1],
            'varma_low': call_dict['varma_l'].shape[-1],
            'basis_high': call_dict['basis_h'].shape[-1],
            'basis_low': call_dict['basis_l'].shape[-1],
            'itrans_high': call_dict['itrans_h'].shape[-1],
            'itrans_low': call_dict['itrans_l'].shape[-1],
            'router': call_dict['router'].shape[-1]
        }

        if self.feature_adapter is None:
            self.feature_adapter = MultiViewAdapter(dims_dict, final_dim=128)
            self.opt_c = optim.Adam(self.feature_adapter.parameters(), lr=self.actor_lr)

        train = not cal_dim
        reduce_call = self.feature_adapter(call_dict, train=train)
        reduce_put = self.feature_adapter(put_dict, train=train)

        current_state = current_state.unsqueeze(0)
        features = torch.cat([current_state, reduce_call, reduce_put], dim=-1)

        if self.state_norm is None:
            self.init_norm_state(features)
        return self.state_norm(features)

    def set_actor_and_value(self, state_dim: int):
        self.actor = ActorDualHead(state_dim, n_actions=self.action_dim).to(self.device)
        self.value = Value(state_dim).to(self.device)

        self.opt_a = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.opt_b = optim.Adam(self.value.parameters(), lr=self.value_lr)

    @torch.no_grad()
    def selete_action_and_weight(self, state, test: bool=False):
        """
        采样动作 + 权重,并且在 rollout 阶段就计算“联合 logprob”(与训练用掩码完全一致)
        """
        logits_a, logits_w = self.actor(state)

        # 动作分布
        dist_a = Categorical(logits=logits_a)
        if test:
            a = torch.argmax(logits_a, dim=-1)
        else:
            a = dist_a.sample()                    # [K]
        logp_a = dist_a.log_prob(a)            # [K]

        # 权重掩码(与训练端完全一致)
        K = a.shape[0]
        allowed = torch.zeros(K, 5, dtype=torch.bool, device=self.device)
        mask_ls = (a == A_LONG) | (a == A_SHORT) | (a == A_CLOSE)    # 多/空/平仓
        allowed[mask_ls, 1:] = True                  # 只能 0.25~1.0
        allowed[~mask_ls, 0] = True                  # 平/不动 -> 只能 0.0

        masked_logits_w = logits_w.clone()
        masked_logits_w[~allowed] = -1e9

        dist_w = Categorical(logits=masked_logits_w)
        if test:
            w_idx = torch.argmax(masked_logits_w, dim=-1)
        else:
            w_idx = dist_w.sample()                      # [K], 离散权重索引 0..4
        logp_w = dist_w.log_prob(w_idx)              # [K]
        w_val = self.WEIGHT_BINS[w_idx]              # [K], 实际权重值

        # —— rollout时就存联合 logp：仅开仓样本叠加权重 logp —— #
        need_w = ((a == A_LONG) | (a == A_SHORT) | (a == A_CLOSE)).float()
        logp_joint = logp_a + need_w * logp_w        # [K]

        return a, w_idx, w_val, logp_a, logp_w, logp_joint

    # 计算GAE(矢量版)
    def compute_gae_vector(
        self,
        rewards: torch.Tensor,            # 归一化后的奖励
        values: torch.Tensor,             # torch.Tensor, [T,K]
        next_value: torch.Tensor,         # torch.Tensor, [K]
        terminateds,        # list[int], len=T
        valid_mask=None,    # torch.BoolTensor, [T,K]
        weights=None,       # torch.Tensor or None, [T,K]
        gamma: float=None,
        lam: float=0.95
    ):
        device = self.device
        gamma = self.gamma if not gamma else gamma
        r = rewards
        done = torch.as_tensor(terminateds, dtype=torch.float32, device=device) # [T]

        T, K = values.shape
        if valid_mask is None:
            valid_mask = torch.ones(T, K, dtype=torch.bool, device=device)
        vm = valid_mask.float()

        next_value = next_value.squeeze()
        if next_value.dim() == 0:
            next_value = next_value.unsqueeze(0)
        assert next_value.shape == (K,), "next_value 需要是 [K]"

        adv = torch.zeros(T, K, dtype=torch.float32, device=device)
        last_gae = torch.zeros(K, dtype=torch.float32, device=device)

        for t in reversed(range(T)):
            v_t = values[t]                              # [K]
            v_tp1 = next_value if t == T-1 else values[t+1]
            m = 1.0 - done[t]
            delta = r[t].unsqueeze(0) + gamma * v_tp1 * m - v_t
            last_gae = delta + gamma * lam * m * last_gae
            adv[t] = last_gae

        # 掩码 & returns
        adv = adv * vm                                  # [T,K]
        returns = adv + values                          # [T,K]

        # 聚合权重(默认均匀平均)
        if weights is None:
            denom = vm.sum(dim=1, keepdim=True).clamp_min(1.0)
            w = vm / denom
        else:
            w = (weights * vm)
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)

        # 供 PPO 用的标量优势(按 K 加权汇总)
        adv_scalar = (adv * w).sum(dim=1)               # [T]
        return adv_scalar, returns, adv                 # returns/adv: [T,K]

    def update_parallel(self, traces, target_kl: float = 0.015, entropy_coef: float = 0.001, value_coef: float = 0.5):
        device = self.device
        
        # ===== 核心修复：使用 stack 处理并行环境的 Tensor 列表 =====
        # traces 中的每个元素 list 长度为 T，其中的 item 是 shape=(NumEnvs, ...) 的 Tensor
        # stack 后 shape 变为 (T, NumEnvs, ...)
        
        states = torch.stack(traces['states'], dim=0).to(device)  # (T, K, StateDim)

        # 之前的 torch.as_tensor 无法处理 list of tensors，改为 stack
        actions = torch.stack(traces['actions'], dim=0).to(dtype=torch.long, device=device)       # (T, K)
        w_idx = torch.stack(traces['weight_idx'], dim=0).to(dtype=torch.long, device=device)      # (T, K)
        old_logp_joint = torch.stack(traces['logp_joint'], dim=0).to(dtype=torch.float32, device=device) # (T, K)
        rewards = torch.stack(traces['rewards'], dim=0).to(dtype=torch.float32, device=device)    # (T, K)

        terminateds = torch.stack(traces['terminated'], dim=0).to(dtype=torch.bool, device=device) # (T, K)
        truncateds = torch.stack(traces['truncated'], dim=0).to(dtype=torch.bool, device=device)   # (T, K)
        
        # next_state 本身就是一个 Tensor (NumEnvs, StateDim)，不需要 stack，直接转设备即可
        next_state = traces['next_state'].to(dtype=torch.float32, device=device)

        dones = terminateds | truncateds # (T, K)

        # ===== 1. GAE & 优势标准化 =====
        with torch.no_grad():
            T, K, L = states.shape
            
            # 计算 V(s) 和 V(s')
            # view(T*K, L) 展平放入网络，出来后再 view 回 (T, K)
            v_tk = self.value(states.view(T * K, L)).squeeze(-1).view(T, K)
            v_next = self.value(next_state).squeeze(-1) # (K,)

            # 计算 GAE
            # 注意：你的 compute_gae_vector 内部需要能够处理 (T, K) 的矩阵
            # 通常只要 compute_gae_vector 里的 mask/weights 计算支持 broadcast 即可
            adv_1d, ret_tk, _ = self.compute_gae_vector(
                rewards, v_tk, v_next, dones.tolist(), # dones.tolist() 会变成 list[list]，需确保内部能处理
                valid_mask=None, weights=None, gamma=self.gamma, lam=0.95
            )

            # 优势标准化 + 裁剪
            adv_1d = torch.clamp(adv_1d, -3.0, 3.0)   # 超保守裁剪
            adv_1d = (adv_1d - adv_1d.mean()) / (adv_1d.std() + 1e-8)
            
            # 展平准备给 PPO Update 使用
            # (T * K)
            adv = adv_1d.unsqueeze(1).expand(T, K).reshape(T * K)
            rets = ret_tk.reshape(T * K)

        entropy_last = None
        actor_loss_last = None
        value_loss_last = None

        for e in range(self.k_epochs):
            T, K, L = states.shape

            # 所有数据展平为 (Batch_Size, ...) 其中 Batch_Size = T * K
            s_flat = states.view(T * K, L)
            a_flat = actions.view(T * K)
            w_flat = w_idx.view(T * K)
            old_logp_flat = old_logp_joint.view(T * K).detach()

            need_w = ((a_flat == A_LONG) | (a_flat == A_SHORT) | (a_flat == A_CLOSE)).float()
            old_v_flat = v_tk.reshape(T * K)

            # 记录更新前的 logits(用于看 policy 改变幅度)
            with torch.no_grad():
                old_logits_a, old_logits_w = self.actor(s_flat)

            # ===== 前向：当前策略 =====
            logits_a, logits_w = self.actor(s_flat)

            dist_a = Categorical(logits=logits_a)
            new_logp_a = dist_a.log_prob(a_flat)
            ent_a = dist_a.entropy()      # [T*K]

            # 权重头：只在 need_w 的时候使用 [1..],否则只用 [0]
            lw = logits_w.clone()
            allowed = torch.zeros_like(lw, dtype=torch.bool)
            allowed[need_w.bool(), 1:] = True
            allowed[~need_w.bool(), 0] = True
            lw[~allowed] = -1e9

            dist_w = Categorical(logits=lw)
            new_logp_w = dist_w.log_prob(w_flat)
            ent_w = dist_w.entropy()      # [T*K]

            logp_new = new_logp_a + need_w * new_logp_w
            ratio = torch.exp(logp_new - old_logp_flat)

            # ===== PPO-Clip Actor Loss =====
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            # ===== Value Loss(clip) =====
            v_pred = self.value(s_flat).squeeze(-1)
            v_clipped = old_v_flat + torch.clamp(v_pred - old_v_flat, -self.clip_eps, self.clip_eps)
            value_loss = torch.max((v_pred - rets)**2, (v_clipped - rets)**2).mean()

            # ===== 熵：单独算 action / weight 部分 =====
            entropy_a = ent_a.mean()
            # 只在 need_w=1 的样本上统计权重熵
            if need_w.sum() > 0:
                entropy_w_eff = (need_w * ent_w).sum() / (need_w.sum() + 1e-8)
            else:
                entropy_w_eff = torch.tensor(0.0, device=device)

            entropy = entropy_a + 0.7 * entropy_w_eff

            loss = actor_loss - entropy_coef * entropy + value_coef * value_loss

            # ===== 反向 + 梯度裁剪 =====
            self.opt_a.zero_grad()
            self.opt_b.zero_grad()

            if self.opt_c:
                self.opt_c.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.3)
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.3)

            if self.opt_c:
                torch.nn.utils.clip_grad_norm_(self.extractor.parameters(), 0.3)

            self.opt_a.step()
            self.opt_b.step()

            if self.opt_c:
                self.opt_c.step()

            # ===== 统计信息 =====
            with torch.no_grad():
                kl = (old_logp_flat - logp_new).mean().abs()

                new_logits_a, new_logits_w = self.actor(s_flat)
                # delta_a = (new_logits_a - old_logits_a).pow(2).mean().sqrt()
                # delta_w = (new_logits_w - old_logits_w).pow(2).mean().sqrt()

                act_probs = torch.softmax(new_logits_a, dim=-1).mean(0)
                w_probs = torch.softmax(new_logits_w, dim=-1).mean(0)

                # ratio_mean = ratio.mean().item()
                # ratio_min = ratio.min().item()
                # ratio_max = ratio.max().item()

            print(f" act_probs={act_probs.detach().cpu().tolist()}")
            print(f" w_probs={w_probs.detach().cpu().tolist()}")

            entropy_last = entropy.detach()
            actor_loss_last = actor_loss.detach()
            value_loss_last = value_loss.detach()

            if kl > 1.5 * target_kl:
                print(f"Early stop epoch {e+1} KL={kl:.4f}")
                break

        if entropy_last is None:
            entropy_last = torch.tensor(0.0, device=device)
        if actor_loss_last is None:
            actor_loss_last = torch.tensor(0.0, device=device)
        if value_loss_last is None:
            value_loss_last = torch.tensor(0.0, device=device)

        return entropy_last, actor_loss_last, value_loss_last, kl

    def update(self, traces, target_kl: float = 0.015, entropy_coef: float = 0.001, value_coef: float = 0.5):
        device = self.device
        states = torch.stack(traces['states'], dim=0).to(device)

        actions = torch.as_tensor(traces['actions'], dtype=torch.long, device=device)    
        w_idx = torch.as_tensor(traces['weight_idx'], dtype=torch.long, device=device)   
        old_logp_joint = torch.as_tensor(traces['logp_joint'], dtype=torch.float32, device=device)

        rewards = torch.as_tensor(traces['rewards'], dtype=torch.float32, device=device)

        terminateds = torch.as_tensor(traces['terminated'], dtype=torch.bool, device=device)    
        truncateds = torch.as_tensor(traces['truncated'], dtype=torch.bool, device=device)
        next_state = torch.as_tensor(traces['next_state'], dtype=torch.float32, device=device)  
        dones = terminateds | truncateds

        # ===== 1. GAE & 优势标准化 =====
        with torch.no_grad():
            T, K, L = states.shape
            v_tk = self.value(states.view(T * K, L)).squeeze(-1).view(T, K)
            v_next = self.value(next_state).squeeze(-1)

            adv_1d, ret_tk, _ = self.compute_gae_vector(
                rewards, v_tk, v_next, dones.tolist(),
                valid_mask=None, weights=None, gamma=self.gamma, lam=0.95
            )

            # 优势标准化 + 裁剪
            adv_1d = torch.clamp(adv_1d, -3.0, 3.0)   # 超保守裁剪
            adv_1d = (adv_1d - adv_1d.mean()) / (adv_1d.std() + 1e-8)
            
            adv = adv_1d.unsqueeze(1).expand(T, K).reshape(T * K)
            rets = ret_tk.reshape(T * K)

        # 展平
        # s_flat = states.view(T * K, L)
        # a_flat = actions.view(T * K)
        # w_flat = w_idx.view(T * K)
        # old_logp_flat = old_logp_joint.view(T * K).detach()

        # need_w = ((a_flat == A_LONG) | (a_flat == A_SHORT) | (a_flat == A_CLOSE)).float()
        # old_v_flat = v_tk.reshape(T * K)

        # print(
        #     f"[update] adv(global): mean={adv_1d.mean():.4f}, std={adv_1d.std():.4f}, "
        #     f"min={adv_1d.min():.4f}, max={adv_1d.max():.4f}"
        # )

        entropy_last = None
        actor_loss_last = None
        value_loss_last = None

        for e in range(self.k_epochs):
            T, K, L = states.shape

            s_flat = states.view(T * K, L)
            a_flat = actions.view(T * K)
            w_flat = w_idx.view(T * K)
            old_logp_flat = old_logp_joint.view(T * K).detach()

            need_w = ((a_flat == A_LONG) | (a_flat == A_SHORT) | (a_flat == A_CLOSE)).float()
            old_v_flat = v_tk.reshape(T * K)


            # 记录更新前的 logits(用于看 policy 改变幅度)
            with torch.no_grad():
                old_logits_a, old_logits_w = self.actor(s_flat)

            # ===== 前向：当前策略 =====
            logits_a, logits_w = self.actor(s_flat)

            dist_a = Categorical(logits=logits_a)
            new_logp_a = dist_a.log_prob(a_flat)
            ent_a = dist_a.entropy()      # [T*K]

            # 权重头：只在 need_w 的时候使用 [1..],否则只用 [0]
            lw = logits_w.clone()
            allowed = torch.zeros_like(lw, dtype=torch.bool)
            allowed[need_w.bool(), 1:] = True
            allowed[~need_w.bool(), 0] = True
            lw[~allowed] = -1e9

            dist_w = Categorical(logits=lw)
            new_logp_w = dist_w.log_prob(w_flat)
            ent_w = dist_w.entropy()      # [T*K]

            logp_new = new_logp_a + need_w * new_logp_w
            ratio = torch.exp(logp_new - old_logp_flat)

            # ===== PPO-Clip Actor Loss =====
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            # ===== Value Loss(clip) =====
            v_pred = self.value(s_flat).squeeze(-1)
            v_clipped = old_v_flat + torch.clamp(v_pred - old_v_flat, -self.clip_eps, self.clip_eps)
            value_loss = torch.max((v_pred - rets)**2, (v_clipped - rets)**2).mean()

            # ===== 熵：单独算 action / weight 部分 =====
            entropy_a = ent_a.mean()
            # 只在 need_w=1 的样本上统计权重熵
            if need_w.sum() > 0:
                entropy_w_eff = (need_w * ent_w).sum() / (need_w.sum() + 1e-8)
            else:
                entropy_w_eff = torch.tensor(0.0, device=device)

            entropy = entropy_a + 0.7 * entropy_w_eff

            loss = actor_loss - entropy_coef * entropy + value_coef * value_loss

            # ===== 反向 + 梯度裁剪 =====
            self.opt_a.zero_grad()
            self.opt_b.zero_grad()

            if self.opt_c:
                self.opt_c.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.3)
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.3)

            if self.opt_c:
                torch.nn.utils.clip_grad_norm_(self.extractor.parameters(), 0.3)

            self.opt_a.step()
            self.opt_b.step()

            if self.opt_c:
                self.opt_c.step()

            # ===== 统计信息 =====
            with torch.no_grad():
                kl = (old_logp_flat - logp_new).mean().abs()

                new_logits_a, new_logits_w = self.actor(s_flat)
                delta_a = (new_logits_a - old_logits_a).pow(2).mean().sqrt()
                delta_w = (new_logits_w - old_logits_w).pow(2).mean().sqrt()

                act_probs = torch.softmax(new_logits_a, dim=-1).mean(0)
                w_probs = torch.softmax(new_logits_w, dim=-1).mean(0)

                ratio_mean = ratio.mean().item()
                ratio_min = ratio.min().item()
                ratio_max = ratio.max().item()

            # print(
            #     f" actor={actor_loss:.5f} value={value_loss:.5f} "
            #     f"ent={entropy:.3f} (ent_a={entropy_a:.3f}, ent_w_eff={entropy_w_eff:.3f}) kl={kl:.5f}"
            # )
            # print(
            #     f" adv: mean={adv_1d.mean():.4f}, std={adv_1d.std():.4f}, "
            #     f"min={adv_1d.min():.4f}, max={adv_1d.max():.4f}"
            # )
            # print(
            #     f"[update {e+1}] ratio: mean={ratio_mean:.4f}, "
            #     f"min={ratio_min:.4f}, max={ratio_max:.4f}"
            # )
            # print(
            #     f"logits_delta: a={float(delta_a):.6f}, "
            #     f"w={float(delta_w):.6f}"
            # )
            print(
                f" act_probs={act_probs.detach().cpu().tolist()}"
            )
            print(
                f" w_probs={w_probs.detach().cpu().tolist()}"
            )

            entropy_last = entropy.detach()
            actor_loss_last = actor_loss.detach()
            value_loss_last = value_loss.detach()

            if kl > 1.5 * target_kl:
                print(f"Early stop epoch {e+1} KL={kl:.4f}")
                break

        if entropy_last is None:
            entropy_last = torch.tensor(0.0, device=device)
        if actor_loss_last is None:
            actor_loss_last = torch.tensor(0.0, device=device)
        if value_loss_last is None:
            value_loss_last = torch.tensor(0.0, device=device)

        return entropy_last, actor_loss_last, value_loss_last, kl

    def save(self, epoch: int = None, best_reward: float = None, path: str = None):
            """
            保存当前 actor / value 以及优化器等信息
            """
            save_path = path or self.check_path
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)

            data = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "actor_state": self.actor.state_dict(),
                "value_state": self.value.state_dict(),
                "opt_a_state": self.opt_a.state_dict(),
                "opt_b_state": self.opt_b.state_dict(),
                "features_adapter_state": self.feature_adapter.state_dict(),
                "h_params": {
                    "gamma": self.gamma,
                    "clip_eps": self.clip_eps,
                    "k_epochs": self.k_epochs,
                    "device": self.device,
                },
                "epoch": epoch,
                "best_reward": best_reward,
            }
            torch.save(data, save_path)
            # 可选：打印一下路径方便确认
            print(f"[PPO] checkpoint saved to: {save_path}")

    # 仅用于推理
    def load_infer_parameters(self, check_path: str=None, device: str=None):
        device = device if device else self.device
        path = check_path if check_path else self.check_path
        data = torch.load(path, map_location=device)

        # 加载actor, value网络和特征提取网络
        self.actor.load_state_dict(data['actor_state'])
        self.value.load_state_dict(data['value_state'])
        self.feature_adapter.load_state_dict(data['features_adapter_state'])

        print(f"[Info: 推理阶段] 加载actor, value网络和特征提取网络~")


@dataclass
class AgentConfig:
    action_dim: int
    
    # 传入环境组合
    env_list: list
    max_epochs: int=500
    max_timesteps: int=1000
    print_interval: int=5

    check_path: str='./miniQMT/DL/checkout'
    window_size: int=32
    device: str='cuda' if torch.cuda.is_available() else 'cpu'


class SyncVectorEnv:
    def __init__(self, env_fns):
        """
        env_fns: 一个列表，每个元素是一个返回 env 实例的函数，或者直接是 env 实例列表
        这里为了兼容你的写法，假设传入的是已经实例化好的 env_list
        """
        self.envs = env_fns
        self.num_envs = len(self.envs)
        
    def reset(self):
        # 重置所有环境，收集所有初始状态
        current_states = []
        history_states = []
        for env in self.envs:
            c, h, _ = env.reset()
            current_states.append(c)
            history_states.append(h)
        return current_states, history_states

    def step(self, actions, weights):
        # actions: list of int, len = num_envs
        # weights: list of float, len = num_envs
        
        next_currents, next_histories = [], []
        rewards, terminateds, truncateds = [], [], []
        infos = []

        for i, env in enumerate(self.envs):
            # 执行动作
            nc, nh, r, term, trunc = env.step(actions[i], weights[i])

            info = {}
            
            # 如果环境结束，自动 reset (这是并行训练的标准做法)
            if term or trunc:
                # 记录结束时的状态（可选，用于计算最后一步价值），这里简化处理
                # 重置环境
                if hasattr(env, 'account_controller'):
                    info['final_equity'] = env.account_controller.equity

                nc, nh, _ = env.reset() 
            
            next_currents.append(nc)
            next_histories.append(nh)
            rewards.append(r)
            terminateds.append(term)
            truncateds.append(trunc)
            
        return next_currents, next_histories, rewards, terminateds, truncateds
    
    def close(self):
        for env in self.envs:
            env.close()

# 训练模型的代理
class Agent:
    def __init__(self, config: AgentConfig):
        self.env = None
        self.action_dim = action_dim  # <<< 修复作用域
        self.ppo = None

        now = datetime.now()
        # 格式化为时分秒字符串
        time_str = now.strftime("%H-%M-%S")
        file_path = f'{DESK_PATH}/PPO_records_{time_str}.txt'
        sys.stdout = outPut(file_path)
        

        self.window_size = config.window_size
        self.device = config.device

        self.max_epochs = config.max_epochs
        self.max_timesteps = config.max_timesteps
        self.print_interval = config.print_interval
        self.check_path = config.check_path

        # 已经创建的env
        self.env_list = config.env_list
    
    def set_norm(self, state_norm: Normalization):
        self.ppo.state_norm = state_norm

        print(f"[Info] Norm设置完成, reward.n = {self.ppo.reward_norm.running_ms.n}, state.n = {self.ppo.state_norm.running_ms.n}")

    def set_env(self, env: windowEnv):
        print(f"[Info] 设置env | call = {env.call}, put = {env.put}")
        self.env = env
        current_shape, history_shape = self.env.get_raw_shape()
        current_state = torch.zeros(current_shape)
        history_state = torch.zeros(history_shape)
        if self.ppo is None:
            self.ppo = weightPPO(action_dim, window_size=self.window_size, device=self.device)
            results = self.ppo.extract_features(current_state, history_state, cal_dim=True)
            _, state_dim = results.shape

            self.ppo.set_actor_and_value(state_dim)
    
    def random_select_env(self):
        if len(self.env_list) == 0:
            return None
        
        return random.choice(self.env_list)

    def select_env(self, k: int):
        length = len(self.env_list)
        idx = k % length
        return self.env_list[idx]

    def export_to_excel(self, 
                        data, 
                        value, 
                        entropy_list, 
                        actor_loss_list, 
                        value_loss_list, 
                        kl_list, 
                        hold_ratio_list,
                        long_ratio_list,
                        short_ratio_list,
                        close_ratio_list,
                        filename=f'{DESK_PATH}/PPO_records.xlsx'):
        """
        将列表导出到 Excel 文件。
        
        参数:
        - data: 二维列表,第一行作为列标题(可选)。
        - filename: 输出 Excel 文件名,默认为 'output.xlsx'。
        
        示例:
        export_to_excel(data, 'example.xlsx')
        """
        if not data:
            print("数据为空,无法导出。")
            return
        e_data = [(i + 1) for i in range(len(data))]
        df = pd.DataFrame({
            'Epoch': e_data,
            'Rewards': data,
            'Market_value': value,
            'Entropy': entropy_list,
            'Actor_loss': actor_loss_list,
            'Value_loss': value_loss_list,
            'KL': kl_list,
            'Hold_ratio': hold_ratio_list,
            'Long_ratio': long_ratio_list,
            'Short_ratio': short_ratio_list,
            'Close_ratio': close_ratio_list
        })
        # 导出到 Excel
        df.to_excel(filename, index=False, sheet_name='Sheet1')
        print(f"数据已成功导出到 {filename}")

    # 新增辅助方法：根据当前进度计算 ek
    def _get_entropy_coef(self, current_epoch):
        """
        计算动态熵系数 (ek)
        策略: Warmup (保持) -> Linear Decay (线性衰减) -> Stable (保持)
        """
        # --- 超参数配置 ---
        ek_start = 0.05   # 初始高探索率
        ek_end = 0.001    # 最终低探索率
        
        warmup_ratio = 0.1  # 热身阶段占比 (前10%)
        decay_ratio = 0.8   # 衰减结束节点 (第80%时降到最低)
        # ----------------
        
        progress = current_epoch / self.max_epochs
        
        if progress < warmup_ratio:
            # 阶段1: 热身，保持高熵
            return ek_start
        
        elif progress < decay_ratio:
            # 阶段2: 线性衰减
            # 计算衰减阶段的进度 (0.0 -> 1.0)
            decay_progress = (progress - warmup_ratio) / (decay_ratio - warmup_ratio)
            ek = ek_start - decay_progress * (ek_start - ek_end)
            return ek
            
        else:
            # 阶段3: 稳定，保持低熵
            return ek_end
        
    def train_parallel(self):
        best_reward = -float('inf')
        
        # 1. 初始化向量环境管理器
        # 确保 SyncVectorEnv 类已定义并可用
        vec_env = SyncVectorEnv(self.env_list) 
        self.num_envs = len(self.env_list)
        
        # 2. 初始化 PPO 网络 (若尚未初始化)
        # 获取第一个环境的维度来初始化网络
        dummy_c, dummy_h, _ = self.env_list[0].reset()
        if self.ppo is None:
            self.ppo = weightPPO(self.action_dim, window_size=self.window_size, device=self.device)
            
            # 使用 batch 方式提取特征以初始化维度 (Batch=1)
            # 需要手动增加 Batch 维度: (Dim) -> (1, Dim)
            d_c_batch = torch.tensor([dummy_c], dtype=torch.float32, device=self.device)
            d_h_batch = torch.tensor([dummy_h], dtype=torch.float32, device=self.device)
            
            # 调用专门的 batch 特征提取函数 (见之前的回答)
            results = self.ppo.extract_features_batch(d_c_batch, d_h_batch, cal_dim=True)
            _, state_dim = results.shape
            self.ppo.set_actor_and_value(state_dim)

        # 3. 初始化用于 Excel 导出的记录列表
        reward_list = []
        market_value_list = []
        actor_loss_list, value_loss_list = [], []
        entropy_list = []
        kl_list = []
        hold_ratio_list, short_ratio_list, close_ratio_list, long_ratio_list = [], [], [], []

        print(f"[Info] Start Parallel Training with {self.num_envs} environments...")

        for epoch in range(self.max_epochs):
            ek = self._get_entropy_coef(epoch)
            
            # 注意：在并行环境中，RunningMeanStd 均值更新较快，视情况决定是否每个 epoch 重置
            if self.ppo.reward_norm is not None:
                # self.ppo.reward_norm.reset() 
                pass

            print(f"[Info] Epoch = {epoch + 1}, ek = {ek}, Envs = {self.num_envs}")

            # --- Reset All Envs ---
            curr_states_list, hist_states_list = vec_env.reset()
            
            # 构造 Batch State Tensor
            state = self._batch_extract_features(curr_states_list, hist_states_list)

            traces = {
                'states': [], 
                'actions': [],
                'weight_idx': [],
                'logp_joint': [],
                'rewards': [], 
                'dones': [],
                'terminated': [], 
                'truncated': []
            }
            
            # 用于统计本 Epoch 的奖励 (并行环境累积)
            episode_rewards = np.zeros(self.num_envs)
            finished_episode_rewards = [] # 存储本 epoch 内已完成 episode 的总奖励

            # --- Rollout Loop ---
            for t in range(self.max_timesteps):
                # 1. 批量采样动作
                # state: (NumEnvs, StateDim) -> a, w_val: (NumEnvs,)
                a, w_idx, w_val, logp_a, logp_w, logp_joint = self.ppo.selete_action_and_weight(state)

                # 转为 list 传给环境
                actions_list = a.cpu().numpy().tolist()
                weights_list = w_val.cpu().numpy().tolist()

                # 2. 批量环境步进
                next_curr_list, next_hist_list, rewards_list, terms, truncs = vec_env.step(actions_list, weights_list)
                
                # 3. 奖励归一化
                norm_rewards = []
                for r in rewards_list:
                    norm_rewards.append(self.ppo.exe_reward_norm(r))
                norm_rewards_tensor = torch.tensor(norm_rewards, dtype=torch.float32, device=self.device)

                # 逻辑或是 (Terminated or Truncated)
                dones = [terms[i] or truncs[i] for i in range(self.num_envs)]
                
                # 收集 Trace
                traces['states'].append(state)
                traces['actions'].append(a)
                traces['weight_idx'].append(w_idx)
                traces['logp_joint'].append(logp_joint)
                traces['rewards'].append(norm_rewards_tensor)
                traces['terminated'].append(torch.tensor(terms, dtype=torch.bool, device=self.device))
                traces['truncated'].append(torch.tensor(truncs, dtype=torch.bool, device=self.device))

                # 统计原始奖励 (用于 Logging)
                episode_rewards += np.array(rewards_list)
                for i, d in enumerate(dones):
                    if d:
                        finished_episode_rewards.append(episode_rewards[i])
                        episode_rewards[i] = 0.0 # 重置该环境的累积奖励

                # 准备下一个状态
                next_state = self._batch_extract_features(next_curr_list, next_hist_list, cal_dim=False)
                state = next_state
            
            # 记录最后一步状态 (用于计算 Advantage)
            traces['next_state'] = next_state # (NumEnv, StateDim)
            
            # --- PPO Update ---
            # 使用之前修正后的 update 方法 (支持堆叠后的 Tensor)
            entropy, actor_loss, value_loss, kl = self.ppo.update_parallel(traces, entropy_coef=ek)
            
            # --- 数据记录与统计 ---
            
            # 1. 记录 Loss & Entropy
            entropy_list.append(entropy.item())
            actor_loss_list.append(actor_loss.item())
            value_loss_list.append(value_loss.item())
            kl_list.append(kl.item())

            # 2. 记录 Reward
            # 如果本 epoch 有 episode 结束，取结束 episode 的平均奖励；否则取当前累积的平均奖励
            current_avg_reward = np.mean(finished_episode_rewards) if finished_episode_rewards else np.mean(episode_rewards)
            reward_list.append(current_avg_reward)
            
            # 3. 记录 Market Value (Equity)
            # 计算所有并行环境当前的平均净值
            current_equities = [env.account_controller.equity for env in vec_env.envs]
            avg_equity = np.mean(current_equities)
            market_value_list.append(avg_equity)

            # 4. 记录动作频率 (Action Ratios)
            # traces['actions'] 是 list of tensors (NumEnvs,), 需要 stack 后 flatten 统计所有步数所有环境的动作
            all_actions_tensor = torch.stack(traces['actions']) # (T, NumEnvs)
            all_actions_flat = all_actions_tensor.cpu().numpy().flatten()
            
            unique, counts = np.unique(all_actions_flat, return_counts=True)
            freq = {0: 0, 1: 0, 2: 0, 3: 0} # 初始化
            for u, c in zip(unique, counts):
                freq[int(u)] = int(c)
            
            total_acts = sum(freq.values()) if sum(freq.values()) > 0 else 1
            hold_ratio_list.append(freq[0] / total_acts)
            long_ratio_list.append(freq[1] / total_acts)
            short_ratio_list.append(freq[2] / total_acts)
            close_ratio_list.append(freq[3] / total_acts)

            # 打印动作分布
            maps = {0: 'HOLD', 1: 'LONG', 2: 'SHORT', 3: 'CLOSE'}
            print(f"Actions: ", end='')
            for k, v in freq.items():
                print(f"{maps[k]}: {v} ({v/total_acts:.2%}) | ", end='')
            print("")

            # --- Excel 导出 ---
            self.export_to_excel(
                reward_list, 
                market_value_list, 
                entropy_list, 
                actor_loss_list, 
                value_loss_list, 
                kl_list, 
                hold_ratio_list, 
                long_ratio_list, 
                short_ratio_list, 
                close_ratio_list
            )

            # --- Logging & Save ---
            if (epoch + 1) % self.print_interval == 0:
                print(f"[Train] Epoch {epoch+1} | Mean Reward: {current_avg_reward:.2f} | Avg Equity: {avg_equity:.0f} | "
                      f"Loss: A={actor_loss.item():.4f}/V={value_loss.item():.4f} | KL: {kl.item():.4f}")
                
            # 保存最佳模型
            if current_avg_reward > best_reward:
                best_reward = current_avg_reward
                self.ppo.save(epoch, best_reward)
                print(f"[Info] Best reward updated: {best_reward:.2f}, model saved.")

            print("\n")
            
        # 训练结束关闭环境
        vec_env.close()

    def _batch_extract_features(self, current_list, history_list, cal_dim=False):
        """辅助函数：处理批量状态输入并转换为 Tensor"""
        # 将 list of arrays 堆叠成 Tensor
        # current_list: [ (ShapeC), (ShapeC), ... ] -> (Batch, ShapeC)
        curr_tensor = torch.tensor(np.array(current_list), dtype=torch.float32, device=self.device)
        hist_tensor = torch.tensor(np.array(history_list), dtype=torch.float32, device=self.device)
        
        # 调用 PPO 中支持 batch 的特征提取
        return self.ppo.extract_features_batch(curr_tensor, hist_tensor, cal_dim)
    def _batch_extract_features(self, current_list, history_list, cal_dim=False):
        """辅助函数：处理批量状态输入"""
        # 将 list of arrays 堆叠成 Tensor
        # current_list: [ (ShapeC), (ShapeC), ... ] -> (Batch, ShapeC)
        curr_tensor = torch.tensor(np.array(current_list), dtype=torch.float32, device=self.device)
        hist_tensor = torch.tensor(np.array(history_list), dtype=torch.float32, device=self.device)
        
        # 修改 ppo.extract_features 以支持 batch
        # 或者在这里手动处理：你的 extract_features 似乎期望 history 是 (1, Window, Feat)
        # 如果传入 (Batch, Window, Feat) 需要确保 extractor 能处理
        
        # 暂时假定 extractor 能够处理 batch (通常 Linear/Conv 层都支持)
        # 唯一需要注意的是 extract_features 里的 unsqueeze(0)
        
        # 临时 Hack：直接调用 extractor 的内部逻辑，或者让 ppo 暴露一个 batch 接口
        # 为了不改动太多 weightPPO 代码，我们这里假设 extract_features 被修改为支持 batch
        # 下面是修改后的 extract_features 调用逻辑
        return self.ppo.extract_features_batch(curr_tensor, hist_tensor, cal_dim)


    # 训练
    def train(self):
        best_reward = 0
        # ek, alpha = 0.03, 0.997
        # ek, alpha = 0.05, 0.997
        reward_list = []
        market_value_list = []
        actor_loss_list, value_loss_list = [], []
        entropy_list = []
        kl_list = []

        hold_ratio_list, short_ratio_list, close_ratio_list, long_ratio_list = [], [], [], []

        reset_env_interval = len(self.env_list)
        for epoch in range(self.max_epochs):
            ek = self._get_entropy_coef(epoch)
            if (epoch + 1) % reset_env_interval == 0 or self.env is None:
                # c_env = self.select_env(epoch // reset_env_interval)
                c_env = random.choice(self.env_list)
                self.set_env(c_env)

            if self.ppo.reward_norm is not None:
                self.ppo.reward_norm.reset()

            # if epoch:
            #     ek = ek * alpha
            #     if ek <= 0.01:
            #         ek = 0.01
            print(f"[Info] Epoch = {epoch + 1}, ek = {ek}")

            current_state, history_state, _ = self.env.reset()
            state = self.ppo.extract_features(current_state, history_state).to(self.device)

    
            # print(f"[Train] Epoch = {epoch} | init_state = {state}")
            traces = {
                'states': [], 'actions': [],
                'weight_idx': [],       # 索引给 PPO
                'weights': [],          # 实际权重(可选)
                'logp_joint': [],       # <<< 存联合 logprob
                'rewards': [], 'dones': [],
                'next_state': None,
                'terminated': [], 'truncated': []
            }
            total_reward = 0.0

            for t in range(self.max_timesteps):
                # 现在返回: (动作, 权重索引, 权重值, logp_a, logp_w, logp_joint)
                a, w_idx, w_val, logp_a, logp_w, logp_joint = self.ppo.selete_action_and_weight(state)

                # tolist
                action = a.item()
                w_idx = w_idx.item()
                logp_joint = logp_joint.item()

                # 环境吃实际权重值
                current_state, history_state, reward, terminated, truncated = self.env.step(action, w_val)
                reward = self.ppo.exe_reward_norm(reward)

                next_state = self.ppo.extract_features(current_state, history_state, cal_dim=False)
                done = terminated or truncated

                traces['states'].append(state)
                traces['actions'].append(action)
                traces['weight_idx'].append(w_idx)
                traces['weights'].append(w_val)
                traces['logp_joint'].append(logp_joint)   # <<< 只存联合
                traces['rewards'].append(reward)
                traces['dones'].append(done)
                traces['terminated'].append(terminated)
                traces['truncated'].append(truncated)

                state = next_state

                total_reward += reward
                if done:
                    break
            
            value = self.env.account_controller.equity
            market_value_list.append(value)
            
            reward_list.append(total_reward)
            best_reward = max(best_reward, total_reward)
            traces['next_state'] = next_state


            entropy, actor_loss, value_loss, kl = self.ppo.update(traces, entropy_coef=ek)
            entropy_list.append(entropy.item())
            actor_loss_list.append(actor_loss.item())
            value_loss_list.append(value_loss.item())
            kl_list.append(kl.item())

            actions = traces['actions']
            unique, counts = np.unique(actions, return_counts=True)
            freq = {int(u): int(c) for u, c in zip(unique, counts)}
            maps = {
                0: 'HOLD',
                1: 'LONG',
                2: 'SHORT',
                3: 'CLOSE'
            }
            sum_of_act = freq[0] + freq[1] + freq[2] + freq[3]
            for k, v in freq.items():
                print(f"{maps[k]} -> {v} | ", end='')
            
        
            hold_ratio_list.append(freq[0] / sum_of_act)
            long_ratio_list.append(freq[1] / sum_of_act)
            short_ratio_list.append(freq[2] / sum_of_act)
            close_ratio_list.append(freq[3] / sum_of_act)


            self.export_to_excel(reward_list, market_value_list, entropy_list, actor_loss_list, value_loss_list, kl_list, hold_ratio_list, long_ratio_list, short_ratio_list, close_ratio_list)

            success_cnt = len(self.env.account_controller.Trades)
            order_cnt = len(self.env.account_controller.Orders)

            if (epoch + 1) % self.print_interval == 0:
                rs = traces['rewards']
                total_reward = sum(rs)

                print(f"[Info: Train model] epoch: {epoch + 1} / {self.max_epochs} | "
                      f"Reward: {total_reward:.2f} | Market-Value: {value} | entropy-k = {ek} | "
                      f"entropy: {entropy.item():.4f}, actor_loss: {actor_loss.item():.6f}, value_loss: {value_loss.item():.6f}")
                

                mx, mi, me = max(rs), min(rs), sum(rs) / len(rs)

                not_zeros = sum([1 for item in rs if abs(item) > 1e-6])
                not_zero_ratio = not_zeros / len(rs)

                print(f"Reward: max = {mx:.6f}, min = {mi:.6f}, mean = {me:.6f}, last_reward = {rs[-1]}, success_cnt = {success_cnt}, order_cnt = {order_cnt}")
                print(f"[Reward-ratio] not_zeros = {not_zeros}, total = {len(rs)}, ratio = {not_zero_ratio}")
                a_f = self.env.account_controller.free_money
                print(f"free_money = {a_f}, market-value = {value}")

            self.ppo.save(epoch, best_reward)




            print("\n")
        self.env.close()

        
    # 测试
    def test(self, epochs: int = 5, alpha: float = 0.5, test_mode: bool = False):
        for epoch in range(epochs):
            # 和 train 一样的 reset 逻辑
            current_state, history_state, _ = self.env.reset()
            # 第一次可以用默认 cal_dim=True，后续用 cal_dim=False
            state = self.ppo.extract_features(current_state, history_state)

            done = False
            total_reward = 0.0

            action_list = []
            weight_list = []

            for t in range(self.max_timesteps):
                # 和 train 一样：直接用特征 state 作为输入
                # 多传一个 test=test_mode，用于“测试/评估模式”（例如用均值而不是采样）
                a, w_idx, w_val, _, _, _ = self.ppo.selete_action_and_weight(
                    state, test=test_mode
                )

                # 转成 Python 标量，和 train 逻辑保持一致
                action = int(a.item()) if hasattr(a, "item") else int(a)
                w_val_scalar = float(w_val.item()) if hasattr(w_val, "item") else float(w_val)

                # 记录一下，后面统计动作频率用
                action_list.append(action)
                weight_list.append(w_val_scalar)

                # 环境吃“实际权重值”（而不是索引）
                current_state, history_state, reward, terminated, truncated = \
                    self.env.step(action, w_val_scalar)

                done = terminated or truncated

                # 提取下一步特征（这里 cal_dim=False，和训练时一致）
                next_state = self.ppo.extract_features(
                    current_state, history_state, cal_dim=False
                )

                # 如果结束，加上基于最终权益的额外 reward（和你原来的逻辑一致）
                if done:
                    e = self.env.account_controller.equity
                    init_capital = self.env.account_controller.init_capital
                    reward += alpha * (e - init_capital) / init_capital

                total_reward += reward
                state = next_state

                if done:
                    break

            # ===== 动作统计（用整条轨迹，而不是最后一步） =====
            if len(action_list) > 0:
                unique, counts = np.unique(action_list, return_counts=True)
                freq = {int(u): int(c) for u, c in zip(unique, counts)}
            else:
                freq = {}

            maps = {
                0: 'HOLD',
                1: 'LONG',
                2: 'SHORT',
                3: 'CLOSE'
            }
            for k, v in freq.items():
                print(f"{maps.get(k, k)} -> {v} | ", end='')
            print("\n")

            value = self.env.account_controller.equity
            print(f"[Info: Test model] epoch: {epoch + 1} | "
                  f"Reward: {total_reward:.4f} | Market-Value: {value}")

        self.env.close()

def save_norm(agent: Agent, filepath: str='./miniQMT/DL/checkout/norm.pkl'):
    data = {
        'reward_norm': agent.ppo.reward_norm,
        'state_norm': agent.ppo.state_norm,
    }

    # 保存
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def read_norm(filepath: str='./miniQMT/DL/checkout/norm.pkl'):
    with open(filepath, 'rb') as f:
        loaded_data = pickle.load(f)
    
    return loaded_data['reward_norm'], loaded_data['state_norm']


if __name__ == '__main__':
    action_dim = 4
    max_epochs = 2
    max_timesteps = 600
    print_interval = 1

    start_time = '20250408100000'
    end_time = '20250924150000'
    benchmark = '510050'

    option_pairs = []
    option_pairs.append({'call': '10009163', 'put': '10009164'})
    option_pairs.append({'call': '10008793', 'put': '10008802'})
    option_pairs.append({'call': '10008794', 'put': '10008803'})
    option_pairs.append({'call': '10008795', 'put': '10008804'})
    option_pairs.append({'call': '10008796', 'put': '10008805'})
    option_pairs.append({'call': '10008797', 'put': '10008806'})
    option_pairs.append({'call': '10008798', 'put': '10008807'})
    option_pairs.append({'call': '10008799', 'put': '10008808'})
    option_pairs.append({'call': '10008800', 'put': '10008809'})
    option_pairs.append({'call': '10008801', 'put': '10008810'})
    option_pairs.append({'call': '10008885', 'put': '10008886'})
    option_pairs.append({'call': '10008895', 'put': '10008896'})
    option_pairs.append({'call': '10008905', 'put': '10008906'})
    option_pairs.append({'call': '10009039', 'put': '10009040'})

    fee = 1.3
    env_list = []
    for dic in option_pairs:
        call, put = dic['call'], dic['put']
        env = windowEnv(100000, call, put, max_timesteps, fee, start_time, end_time, benchmark)
        
        env_list.append(env)
    
    config = AgentConfig(
        action_dim=action_dim,
        max_epochs=max_epochs,
        max_timesteps=max_timesteps,
        print_interval=print_interval,
        env_list=env_list
    )

    agent = Agent(config)
    print(f"[PPO-Agent] 开始训练......")
    agent.train()
    save_norm(agent)
    print(f"[PPO-Agent] 结束训练......")


    # 测试
    idx = 2
    call, put = option_pairs[idx]['call'], option_pairs[idx]['put']

    call, put = '10006038', '10006029'
    start_time = '20230928100000'
    end_time = '20231122150000'

    # calls = [calls[1]]
    # puts = [puts[1]]
    env = windowEnv(100000, call, put, max_timesteps, fee, start_time, end_time, benchmark, normalize_reward=False)
    _, state_norm = read_norm()
    
    agent.set_env(env)
    agent.set_norm(state_norm)
    # 加载模型权重
    agent.ppo.load_infer_parameters()
    print(f"[PPO-Agent] 开始测试......")
    agent.test(test_mode=False, epochs=3)


    print(0 / 0)
