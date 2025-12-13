"""
    PPO算法 (标准精度 Float32 版) - 动态并行训练重构版 (Full Optimized)
    包含: Multiprocessing Parallellism + Excel Export + Dynamic Environment Loading + Data Caching (Shared Memory)
    修复: 
    1. DynamicWindowEnv 增加 close 方法，修复 AttributeError。
    2. DataCache 使用 multiprocessing.Manager 共享内存，解决多进程重复读取导致的 Miss 刷屏。
"""
from math import e
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from windowEnv_parallel import windowEnv
import time, json
import sys
import pandas as pd
from datetime import datetime, timedelta
import pickle
from tools.Norm import Normalization, RewardNormalization, RewardScaling
from preTrain.preMOE import PreMOE
from dataclasses import dataclass, field
import random
import multiprocessing as mp
import copy
from finTool.single_window_account import single_Account  # 用于 DataCache 读取数据
import os

import warnings
# 忽略所有 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# —— 构造每个样本的权重掩码 —— #
A_HOLD, A_LONG, A_SHORT, A_CLOSE = 0, 1, 2, 3
WEIGHT_BINS_CPU = torch.tensor([0.00, 0.25, 0.50, 0.75, 1.00])  # 离散权重

DESK_PATH = 'C:/Users/Flying/Desktop' # 请根据实际路径修改
# DESK_PATH = 'C:/Users/David/Desktop' 

# -----------------------------------------------------------
# 全局数据缓存 (共享内存版)
# -----------------------------------------------------------
class DataCache:
    """
    全局数据缓存工具类。
    不再持有类变量，而是操作传入的 shared_dict (Manager.dict)。
    """

    @staticmethod
    def clean_ts(ts_series):
        """向量化时间清洗"""
        if np.issubdtype(ts_series.dtype, np.datetime64):
            return ts_series.dt.strftime('%Y%m%d%H%M%S').values
        try:
            # 尝试 Pandas 向量化字符串操作
            return ts_series.astype(str).str.replace(' ', '').str.replace('-', '').str.replace(':', '').values
        except:
            # 兜底
            return np.array([str(x).replace(' ', '').replace('-', '').replace(':', '') for x in ts_series])

    @classmethod
    def get_data(cls, shared_dict, benchmark, start_time, end_time, init_capital, fee):
        """
        获取数据: 先查共享内存，没有再读盘并写入共享内存
        """
        key = f"{benchmark}_{start_time}_{end_time}"
        
        # 1. 查共享字典 (进程安全)
        if key in shared_dict:
            return shared_dict[key]
            
        # 2. 未命中，读取数据
        # 加上进程名方便调试
        p_name = mp.current_process().name
        print(f"[DataCache][{p_name}] Miss! Loading {key}...")
        
        # 使用临时账户读取
        temp_acct = single_Account(init_capital, fee, '30m', [benchmark])
        df = temp_acct.real_info_controller.get_bars_between_from_df(benchmark, start_time, end_time)
        
        # 转 Numpy
        close_arr = df['close'].values.astype(np.float32)
        ts_arr = cls.clean_ts(df['ts'])
        
        # 封装数据包
        data_pack = {
            'close_arr': close_arr,
            'ts_arr': ts_arr,
            'benchmark': benchmark
        }
        
        # 3. 写入共享字典
        shared_dict[key] = data_pack
        print(f"[DataCache][{p_name}] Loaded & Shared {len(close_arr)} steps.")
        
        del temp_acct
        return data_pack

# 输出类
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
        state = state.to(dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        z = self.backbone(state)
        return self.action_head(z), self.weight_head(z)

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

# -----------------------------------------------------------
# 多进程环境相关类
# -----------------------------------------------------------

class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                action, weight = data
                nc, nh, r, term, trunc = env.step(action, weight)
                
                info = {}
                if hasattr(env, 'account_controller'):
                    info['equity'] = env.account_controller.equity
                
                if term or trunc:
                    info['final_equity'] = env.account_controller.equity
                    nc, nh, _ = env.reset()
                
                remote.send((nc, nh, r, term, trunc, info))
            
            elif cmd == 'reset':
                nc, nh, _ = env.reset()
                info = {}
                if hasattr(env, 'account_controller'):
                    info['equity'] = env.account_controller.equity
                remote.send((nc, nh, info))
            
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocEnv worker: got KeyboardInterrupt')
    except Exception as e:
        print(f'SubprocEnv worker error: {e}')
    finally:
        env.close() # <--- 这里会调用 DynamicWindowEnv.close()，必须保证该方法存在

class SubprocVectorEnv:
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.ps = []
        
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            p = mp.Process(target=worker, args=args, daemon=True) 
            p.start()
            self.ps.append(p)
            work_remote.close()

    def step(self, actions, weights):
        for remote, action, weight in zip(self.remotes, actions, weights):
            remote.send(('step', (action, weight)))
        results = [remote.recv() for remote in self.remotes]
        currents, histories, rewards, terms, truncs, infos = zip(*results)
        return np.stack(currents), np.stack(histories), np.stack(rewards), np.stack(terms), np.stack(truncs), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        currents, histories, infos = zip(*results)
        return np.stack(currents), np.stack(histories), infos

    def close(self):
        if self.closed: return
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

# -----------------------------------------------------------
# 动态环境包装器 (修复 Missing Close Method)
# -----------------------------------------------------------
class DynamicWindowEnv:
    def __init__(self, option_pairs, global_cfg, shared_cache, seed=0):
        self.all_pairs = option_pairs
        self.cfg = global_cfg
        self.rng = random.Random(seed)
        self.current_env = None
        self.shared_cache = shared_cache # 保存共享字典
        
    def reset(self):
        if self.current_env is not None:
            self.current_env.close()
            self.current_env = None

        pair_info = self.rng.choice(self.all_pairs)
        t_start = pair_info.get('start_time', self.cfg.start_time)
        t_end = pair_info.get('end_time', self.cfg.end_time)
        
        # 从共享内存获取数据
        preloaded_data = DataCache.get_data(
            self.shared_cache, # 传入 Manager.dict
            benchmark='510050',
            start_time=t_start,
            end_time=t_end,
            init_capital=self.cfg.init_capital,
            fee=self.cfg.fee
        )
        
        self.current_env = windowEnv(
            init_capital=self.cfg.init_capital,
            call=pair_info['call'],
            put=pair_info['put'],
            fee=self.cfg.fee,
            start_time=t_start,
            end_time=t_end,
            benchmark='510050',
            timesteps=self.cfg.max_timesteps,
            preloaded_data=preloaded_data
        )
        
        return self.current_env.reset()

    def step(self, action, weight):
        if self.current_env is None:
            return self.reset()
        return self.current_env.step(action, weight)

    # [修复] 必须显式定义 close 方法，否则 Worker 退出时会报错
    def close(self):
        if self.current_env:
            self.current_env.close()

    @property
    def account_controller(self):
        return self.current_env.account_controller
    
    def get_raw_shape(self):
        if self.current_env is None:
            self.reset()
        return self.current_env.get_raw_shape()

# -----------------------------------------------------------
# 特征适配器
# -----------------------------------------------------------
class ViewProjector(nn.Module):
    def __init__(self, high_dim, low_dim, out_dim=64):
        super().__init__()
        self.high_net = nn.Sequential(
            nn.LayerNorm(high_dim),
            nn.Linear(high_dim, out_dim), 
        )
        self.low_net = nn.Sequential(
            nn.LayerNorm(low_dim),
            nn.Linear(low_dim, 32),       
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(out_dim + 32, out_dim),
            nn.LayerNorm(out_dim)
        )
        
    def forward(self, x_high, x_low):
        h = self.high_net(x_high)
        l = self.low_net(x_low)
        return self.fusion(torch.cat([h, l], dim=-1))

class MultiViewAdapter(nn.Module):
    def __init__(self, dims_dict: dict, final_dim: int = 128):
        super().__init__()
        view_dim = 48
        self.varma_proj = ViewProjector(dims_dict['varma_high'], dims_dict['varma_low'], out_dim=view_dim)
        self.basis_proj = ViewProjector(dims_dict['basis_high'], dims_dict['basis_low'], out_dim=view_dim)
        self.itrans_proj = ViewProjector(dims_dict['itrans_high'], dims_dict['itrans_low'], out_dim=view_dim)
        self.router_proj = nn.Sequential(
            nn.LayerNorm(dims_dict['router']),
            nn.Linear(dims_dict['router'], 32)
        )
        self.final_net = nn.Sequential(
            nn.Linear(view_dim * 3 + 32, final_dim),
            nn.LayerNorm(final_dim)
        )
        
    def raw_forward(self, inputs: dict):
        v_varma = self.varma_proj(inputs['varma_h'], inputs['varma_l'])
        v_basis = self.basis_proj(inputs['basis_h'], inputs['basis_l'])
        v_itrans = self.itrans_proj(inputs['itrans_h'], inputs['itrans_l'])
        v_router = self.router_proj(inputs['router'])
        combined = torch.cat([v_varma, v_basis, v_itrans, v_router], dim=-1)
        return self.final_net(combined)
    
    def forward(self, inputs: dict, train: bool=True):
        if train: return self.raw_forward(inputs)
        with torch.no_grad(): return self.raw_forward(inputs)

# -----------------------------------------------------------
# PPO Agent 类
# -----------------------------------------------------------

class weightPPO:
    def __init__(self, action_dim: int, actor_lr: float=3e-4, value_lr: float=5e-4, 
                 gamma: float=0.99, clip_eps: float=0.1, k_epochs: int=5, 
                 device: str='cpu', check_path: str='./miniQMT/DL/checkout',
                 window_size: int=32, pre_len: int=4):
        
        self.device = device
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.check_path = f'{check_path}/check_data_all.pt'
        self.actor_lr, self.value_lr = actor_lr, value_lr
        self.action_dim = action_dim
        self.window_size = window_size
        self.WEIGHT_BINS = WEIGHT_BINS_CPU.to(self.device)

        self.actor, self.value = None, None
        self.opt_a, self.opt_b = None, None
        
        self.pre_len = pre_len
        self.extractor = PreMOE(seq_len=self.window_size, pred_len=self.pre_len, n_variates=13, d_router=128).to(self.device)
        self.load_moe_parameters()

        # 冻结 extractor 参数
        for p in self.extractor.parameters():
            p.requires_grad = False
        self.extractor.eval()
        
        self.feature_adapter = None
        self.opt_c = None
        self.state_norm = None
        self.reward_norm = None
        self.reward_norm_list = []

        print(f"[Info] Device: {self.device} | Precision: Float32 (AMP Removed)")

    def load_moe_parameters(self):
        try:
            SAVE_PATH = f'./miniQMT/DL/preTrain/weights/preMOE_best_dummy_data_{self.window_size}_{self.pre_len}.pth'
            state_dict = torch.load(SAVE_PATH, map_location=self.device)
            self.extractor.load_state_dict(state_dict)
            print(f"[Info] 加载MOE参数成功~")
        except Exception as e:
            print(f"[Info] 加载MOE参数失败, e = {e}")

    def init_norm_state(self, x: torch.Tensor):
        self.state_norm = Normalization(x.shape[1:]) 
    
    def init_norm_reward(self):
        self.reward_norm = RewardScaling(shape=(1,), gamma=self.gamma)

    def init_norm_reward_list(self, length: int):
        self.reward_norm_list = []
        for _ in range(length):
            self.reward_norm_list.append(RewardScaling(shape=(1,), gamma=self.gamma))
    
    def load_infer_parameters(self, check_path: str=None, device: str=None):
        device = device if device else self.device
        path = check_path if check_path else self.check_path
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data['actor_state'])
        self.value.load_state_dict(data['value_state'])
        self.feature_adapter.load_state_dict(data['features_adapter_state'])
        print(f"[Info: 推理阶段] 加载网络权重完成~")

    def exe_reward_norm(self, x: float):
        if self.reward_norm is None: 
            self.init_norm_reward()
        x = torch.tensor([x], dtype=torch.float32)
        return self.reward_norm(x).item()

    def extract_features_batch(self, current_state: torch.Tensor, history_state: torch.Tensor, cal_dim: bool = False):
        if current_state.device != torch.device(self.device): 
            current_state = current_state.to(self.device)
        if history_state.device != torch.device(self.device): 
            history_state = history_state.to(self.device)

        call_state, put_state = torch.chunk(history_state, chunks=2, dim=2)
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
            self.feature_adapter = MultiViewAdapter(dims_dict, final_dim=128).to(self.device)
            self.opt_c = optim.Adam(self.feature_adapter.parameters(), lr=self.actor_lr)

        train = not cal_dim
        reduce_call = self.feature_adapter(call_dict, train=train)
        reduce_put = self.feature_adapter(put_dict, train=train)
        features = torch.cat([current_state, reduce_call, reduce_put], dim=-1)

        if self.state_norm is None:
            self.init_norm_state(features)
        return self.state_norm(features, update=train)

    def set_actor_and_value(self, state_dim: int):
        self.actor = ActorDualHead(state_dim, n_actions=self.action_dim).to(self.device)
        self.value = Value(state_dim).to(self.device)
        self.opt_a = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.opt_b = optim.Adam(self.value.parameters(), lr=self.value_lr)

    @torch.no_grad()
    def selete_action_and_weight(self, state, test: bool=False):
        logits_a, logits_w = self.actor(state)
        dist_a = Categorical(logits=logits_a)
        a = torch.argmax(logits_a, dim=-1) if test else dist_a.sample()
        logp_a = dist_a.log_prob(a)

        K = a.shape[0]
        allowed = torch.zeros(K, 5, dtype=torch.bool, device=self.device)
        mask_ls = (a == A_LONG) | (a == A_SHORT) | (a == A_CLOSE)
        allowed[mask_ls, 1:] = True
        allowed[~mask_ls, 0] = True
        
        masked_logits_w = logits_w.clone()
        masked_logits_w[~allowed] = -1e9 
        
        dist_w = Categorical(logits=masked_logits_w)
        w_idx = torch.argmax(masked_logits_w, dim=-1) if test else dist_w.sample()
        logp_w = dist_w.log_prob(w_idx)
        w_val = self.WEIGHT_BINS[w_idx]

        need_w = ((a == A_LONG) | (a == A_SHORT) | (a == A_CLOSE)).float()
        logp_joint = logp_a + need_w * logp_w

        return a, w_idx, w_val, logp_a, logp_w, logp_joint
    
    def update_parallel(self, traces, target_kl=0.015, entropy_coef=0.01, value_coef=0.5):
        raw_curr = torch.stack(traces['raw_curr']).to(self.device)
        raw_hist = torch.stack(traces['raw_hist']).to(self.device)
        
        actions = torch.stack(traces['actions']).to(self.device)
        w_idx = torch.stack(traces['weight_idx']).to(self.device)
        old_logp = torch.stack(traces['logp_joint']).to(self.device)
        rewards = torch.stack(traces['rewards']).to(self.device)
        next_raw_curr = traces['next_raw_curr'].to(self.device)
        next_raw_hist = traces['next_raw_hist'].to(self.device)
        
        dones = torch.stack(traces['terminated']).to(self.device) | torch.stack(traces['truncated']).to(self.device)
        
        T, K, Dc = raw_curr.shape
        Dh = raw_hist.shape[-1]

        # --- 1. 计算 GAE (no_grad) ---
        with torch.no_grad():
            curr_flat = raw_curr.view(T*K, -1)
            hist_flat = raw_hist.view(T*K, -1, Dh)
            feat_tk = self.extract_features_batch(curr_flat, hist_flat)
            v_tk = self.value(feat_tk).view(T, K)
            
            next_feat = self.extract_features_batch(next_raw_curr, next_raw_hist)
            v_next = self.value(next_feat).squeeze(-1)
        
            v_tk_cpu = v_tk.float()
            v_next_cpu = v_next.float()
            rew_cpu = rewards.float()
            dones_cpu = dones.float()
            
            adv = torch.zeros_like(rew_cpu)
            last_gae = 0
            for t in reversed(range(T)):
                m = 1.0 - dones_cpu[t]
                v_tp1 = v_next_cpu if t == T-1 else v_tk_cpu[t+1]
                delta = rew_cpu[t] + self.gamma * v_tp1 * m - v_tk_cpu[t]
                last_gae = delta + self.gamma * 0.95 * m * last_gae
                adv[t] = last_gae
            
            returns = adv + v_tk_cpu
        
        curr_flat = raw_curr.view(T*K, -1)
        hist_flat = raw_hist.view(T*K, -1, Dh)
        a_flat = actions.view(-1)
        w_flat = w_idx.view(-1)
        old_logp_flat = old_logp.view(-1)
        adv_flat = adv.view(-1).to(self.device)
        ret_flat = returns.view(-1).to(self.device)
        
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # --- 2. 训练循环 ---
        for i in range(self.k_epochs):
            s_flat = self.extract_features_batch(curr_flat, hist_flat)
            logits_a, logits_w = self.actor(s_flat)
            
            dist_a = Categorical(logits=logits_a)
            new_logp_a = dist_a.log_prob(a_flat)
            ent_a = dist_a.entropy().mean()
            
            need_w = ((a_flat == A_LONG) | (a_flat == A_SHORT) | (a_flat == A_CLOSE)).float()
            
            lw = logits_w.clone()
            mask = torch.zeros_like(lw, dtype=torch.bool)
            mask[need_w.bool(), 1:] = True
            mask[~need_w.bool(), 0] = True
            lw[~mask] = -1e9
            
            dist_w = Categorical(logits=lw)
            new_logp_w = dist_w.log_prob(w_flat)
            ent_w = (need_w * dist_w.entropy()).sum() / (need_w.sum() + 1e-6)
            
            logp_new = new_logp_a + need_w * new_logp_w
            ratio = torch.exp(logp_new - old_logp_flat)
            
            surr1 = ratio * adv_flat
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv_flat
            loss_a = -torch.min(surr1, surr2).mean()
            
            v_pred = self.value(s_flat).squeeze(-1)
            loss_v = F.mse_loss(v_pred, ret_flat)
            loss = loss_a + value_coef * loss_v - entropy_coef * (ent_a + 0.5 * ent_w)

            self.opt_a.zero_grad()
            self.opt_b.zero_grad()
            if self.opt_c: self.opt_c.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.opt_a.step()
            self.opt_b.step()
            if self.opt_c: self.opt_c.step()

            last_actor_loss = loss_a.item()
            last_value_loss = loss_v.item()
            last_entropy = (ent_a + 0.5 * ent_w).item()

            kl = (old_logp_flat - logp_new).mean().abs()
            if kl > 1.5 * target_kl:
                print(f"Early stop at epoch {i} KL={kl.item():.4f}")
                break
    
        return loss.item(), kl.item(), last_actor_loss, last_value_loss, last_entropy
    
    def save(self, epoch: int = None, best_reward: float = None, path: str = None):
            save_path = path or self.check_path
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
                "best_reward": best_reward.item() if hasattr(best_reward, 'item') else best_reward,
                "state_norm": self.state_norm,
            }
            torch.save(data, save_path)
            print(f"[PPO] checkpoint saved to: {save_path}")
        
    def load_checkpoint(self, path: str=None):
        if path is None:
            path = self.check_path
        if not os.path.exists(path):
            print(f"[Warn] Checkpoint not found at {path}")
            return None, None
        
        print(f"[Resume] Loading checkpoint from {path}...")
        # 这里的 map_location 非常重要，防止跨设备加载报错
        data = torch.load(path, map_location=self.device)
        
        # 1. 加载网络权重
        self.actor.load_state_dict(data['actor_state'])
        self.value.load_state_dict(data['value_state'])
        self.feature_adapter.load_state_dict(data['features_adapter_state'])
        
        # 2. 加载优化器状态
        if self.opt_a: self.opt_a.load_state_dict(data['opt_a_state'])
        if self.opt_b: self.opt_b.load_state_dict(data['opt_b_state'])
        if self.opt_c and 'features_adapter_opt_state' in data:
            st = data['features_adapter_opt_state']
            if st is not None: self.opt_c.load_state_dict(st)

        # 3. [新增] 加载 Normalization 状态
        if 'state_norm' in data:
            # 直接覆盖当前的 self.state_norm
            self.state_norm = data['state_norm']
            print(f"[Resume] State Norm loaded. (count={self.state_norm.running_ms.n if hasattr(self.state_norm.running_ms, 'n') else '?'})")
        else:
            print("[Resume] Warning: No state_norm in checkpoint! Training might be unstable.")

        # 4. [新增] 加载 Reward Norm List (可选)
        if 'reward_norm_list' in data:
            self.reward_norm_list = data['reward_norm_list']

        epoch = data.get('epoch', 0)
        best_reward = data.get('best_reward', -float('inf'))
        
        print(f"[Resume] Success! Resuming from Epoch {epoch + 1}, Best Reward: {best_reward:.4f}")
        return epoch, best_reward


# -----------------------------------------------------------
# 主 Agent 类
# -----------------------------------------------------------

@dataclass
class AgentConfig:
    action_dim: int
    option_pairs: list
    max_epochs: int=300
    max_timesteps: int=1000
    device: str='cuda' if torch.cuda.is_available() else 'cpu'
    print_interval: int=1
    
    # 全局默认时间 (如果 option_pairs 里没指定则用这个)
    start_time: str='20250408100000'
    end_time: str='20250924150000'

    fee: float=1.3
    init_capital: float=100000.0
    mode: str='train'
    
    # 自动计算 Worker 数量
    num_workers: int = field(default_factory=lambda: min(mp.cpu_count() - 2, 12)) 

class Agent:
    def __init__(self, config: AgentConfig):
        self.cfg = config
        self.device = config.device
        self.env = None
        self.env_fns = []

        if config.mode == 'train':
            self.init_train()
        
        self.records = {
            'epoch': [], 'reward': [], 'avg_equity': [], 'loss': [], 'kl': [],
            'hold_ratio': [], 'long_ratio': [], 'short_ratio': [], 'close_ratio': [],
            'actor_loss': [], 'value_loss': [], 'entropy': [],
            'ratio_0': [], 'ratio_25': [], 'ratio_50': [], 'ratio_75': [], 'ratio_100': [],
        }

        # 并行个数
        self.num_workers = None

    # 训练模式初始化
# 训练模式初始化 - [已修复序列化报错]
    def init_train(self):
        print(f'[Agent-init-train] 期权组合数量 = {len(self.cfg.option_pairs)}')
        config = self.cfg
        self.ppo = weightPPO(config.action_dim, device=self.device)
        
        # 1. 启动管理器 (必须在主进程)
        self.manager = mp.Manager()
        self.shared_cache = self.manager.dict() # 创建跨进程共享字典
        
        # 2. 确定 Worker 数量
        num_workers = min(len(config.option_pairs), config.num_workers)
        if num_workers < 1: num_workers = 1
        self.num_workers = num_workers
        print(f"[Init] Detect {mp.cpu_count()} CPUs, Launching {num_workers} workers.")

        # 3. 构造环境生成器 (关键修改！！！)
        self.env_fns = []
        
        # --- 核心修复开始 ---
        # 提取为局部变量，切断与 self 的联系
        # 这样 pickle 就只会打包这几个对象，而不会打包包含 self.manager 的 Agent
        _cache = self.shared_cache
        _pairs = config.option_pairs
        _cfg = config 
        
        for i in range(num_workers):
            # 使用默认参数绑定 (seed=i, cache=_cache, ...) 
            # 这样函数体内部就不需要引用外部作用域的 self 了
            def make_env(seed=i, cache=_cache, pairs=_pairs, cfg=_cfg):
                return DynamicWindowEnv(pairs, cfg, cache, seed=seed)
            
            self.env_fns.append(make_env)
        # --- 核心修复结束 ---
        
        # 4. 预热网络
        # 注意：这里我们临时实例化一个 env 来拿 shape，用完即毁
        # 这里可以直接用局部变量初始化，避免调用 self.env_fns[0] 导致不必要的复杂性
        # 或者依然用 self.env_fns[0] 也可以，因为现在 make_env 已经是干净的了
        dummy_env = self.env_fns[0]() 
        c, h, _ = dummy_env.reset()
        dummy_env.close()
        
        c_b = torch.tensor([c], dtype=torch.float32, device=self.device)
        h_b = torch.tensor([h], dtype=torch.float32, device=self.device)
        feat = self.ppo.extract_features_batch(c_b, h_b, cal_dim=True)
        self.ppo.set_actor_and_value(feat.shape[-1])



    # 测试截断设置
    def set_env(self, env: windowEnv):
        self.env = env
        current_shape, history_shape = self.env.get_raw_shape()
        current_state = torch.zeros(current_shape)
        history_state = torch.zeros(history_shape)
        if self.ppo is None:
            self.ppo = weightPPO(self.action_dim, window_size=self.window_size, device=self.device)
            results = self.ppo.extract_features(current_state, history_state, cal_dim=True)
            _, state_dim = results.shape
            self.ppo.set_actor_and_value(state_dim)

    def set_norm(self, state_norm: Normalization):
        self.ppo.state_norm = state_norm
        print(f"[Info] Norm设置完成 | state.n = {self.ppo.state_norm.running_ms.n}")

    # 动态并行训练函数
    def train_parallel_modified_early_stop(self, from_check_point: bool=False):
        # 1. 初始化并行环境
        vec_env = SubprocVectorEnv(self.env_fns)
        print(f"[Train] Start dynamic parallel training on {self.device}...")
        
        best_reward = -float('inf')

        # 早停参数
        patience = getattr(self.cfg, 'patience', 30)          
        stop_entropy = getattr(self.cfg, 'stop_entropy', 0.6) 
        min_delta = 0.001                                     
        early_stop_counter = 0                                

        if from_check_point:
            start_epoch, best_reward = self.ppo.load_checkpoint()

        for epoch in range(self.cfg.max_epochs):
            if from_check_point and start_epoch is not None:
                if epoch < start_epoch:
                    print(f'[Skip] epoch = {epoch}')
                    continue
            print(f'epoch = {epoch}')
            start_time = time.time()
            
            self.ppo.init_norm_reward_list(length=len(self.env_fns))

            # Reset
            curr_np, hist_np, infos = vec_env.reset()
            
            traces = {
                'raw_curr': [], 'raw_hist': [],
                'actions': [], 'weight_idx': [], 'logp_joint': [], 
                'rewards': [], 
                'terminated': [], 'truncated': []
            }
            
            epoch_rewards = [] 
            current_equities = [0.0] * len(self.env_fns)
            
            # --- Rollout Loop ---
            for t in range(self.cfg.max_timesteps):
                
                c_tensor = torch.as_tensor(curr_np, dtype=torch.float32, device=self.device)
                h_tensor = torch.as_tensor(hist_np, dtype=torch.float32, device=self.device)
                
                with torch.no_grad():
                     state = self.ppo.extract_features_batch(c_tensor, h_tensor)
                     a, w_idx, w_val, _, _, logp_joint = self.ppo.selete_action_and_weight(state)
                
                actions_np = a.cpu().numpy()
                weights_np = w_val.cpu().numpy()
                
                next_curr, next_hist, rews, terms, truncs, infos = vec_env.step(actions_np, weights_np)
                
                scaled_rewards = []
                for num in range(len(self.env_fns)):
                    r = rews[num].item()
                    r_norm = self.ppo.reward_norm_list[num]
                    scaled_rewards.append(r_norm(r))
                
                for i, info in enumerate(infos):
                    if isinstance(info, dict):
                        key = 'final_equity' if (terms[i] or truncs[i]) else 'equity'
                        if key in info:
                            current_equities[i] = info[key]

                if t > 0:
                    rew_tensor = torch.as_tensor(scaled_rewards, dtype=torch.float32, device=self.device)
                    traces['rewards'].append(rew_tensor)
                    epoch_rewards.append(rews)

                traces['raw_curr'].append(c_tensor)
                traces['raw_hist'].append(h_tensor)
                traces['actions'].append(a)
                traces['weight_idx'].append(w_idx)
                traces['logp_joint'].append(logp_joint)
                traces['terminated'].append(torch.as_tensor(terms, device=self.device))
                traces['truncated'].append(torch.as_tensor(truncs, device=self.device))
                
                curr_np, hist_np = next_curr, next_hist

            # --- Soft End 补齐 ---
            hold_actions = np.zeros(len(self.env_fns), dtype=int)
            hold_weights = np.zeros(len(self.env_fns), dtype=float)
            _, _, final_rews, _, _, _ = vec_env.step(hold_actions, hold_weights)
            
            rew_tensor = torch.as_tensor(final_rews, dtype=torch.float32, device=self.device)
            traces['rewards'].append(rew_tensor)
            epoch_rewards.append(final_rews)
            
            traces['next_raw_curr'] = torch.as_tensor(curr_np, dtype=torch.float32, device=self.device)
            traces['next_raw_hist'] = torch.as_tensor(hist_np, dtype=torch.float32, device=self.device)

            # --- Update ---
            n_act = len(traces['actions'])
            if n_act > 0:
                loss, kl, actor_loss, value_loss, entropy = self.ppo.update_parallel(traces)
            else:
                loss, kl, actor_loss, value_loss, entropy = 0, 0, 0, 0, 0

            # --- Log & Excel ---
            end_time = time.time()
            fps = (n_act * len(self.env_fns)) / (end_time - start_time + 1e-5)
            
            if len(epoch_rewards) > 0:
                avg_rew = np.mean(np.concatenate(epoch_rewards))
            else:
                avg_rew = 0.0
            avg_equity = np.mean(current_equities)

            if n_act > 0:
                all_actions = torch.stack(traces['actions']).cpu().numpy().flatten()
                counts = np.bincount(all_actions, minlength=4)
                ratios = counts / (len(all_actions) + 1e-8) 

                all_weights = torch.stack(traces['weight_idx']).cpu().numpy().flatten()
                weight_counts = np.bincount(all_weights, minlength=5) 
                weight_ratios = weight_counts / (len(all_weights) + 1e-8)
            else:
                ratios = [0, 0, 0, 0]
                weight_ratios = [0, 0, 0, 0, 0]

            self.records['epoch'].append(epoch + 1)
            self.records['reward'].append(avg_rew)
            self.records['avg_equity'].append(avg_equity)
            self.records['loss'].append(loss)
            self.records['kl'].append(kl)
            self.records['actor_loss'].append(actor_loss)
            self.records['value_loss'].append(value_loss)
            self.records['entropy'].append(entropy)
            
            self.records['hold_ratio'].append(ratios[0])
            self.records['long_ratio'].append(ratios[1])
            self.records['short_ratio'].append(ratios[2])
            self.records['close_ratio'].append(ratios[3])
            
            names = ['ratio_0', 'ratio_25', 'ratio_50', 'ratio_75', 'ratio_100']
            for k in range(len(names)):
                val = weight_ratios[k] if k < len(weight_ratios) else 0.0
                self.records[names[k]].append(val)

            df = pd.DataFrame(self.records)
            excel_path = f'{DESK_PATH}/PPO_training_data.xlsx'
            try:
                df.to_excel(excel_path, index=False)
            except Exception as e:
                print(f"[Warning] Save Excel failed: {e}")

            if (epoch + 1) % self.cfg.print_interval == 0:
                print(f"[Epoch {epoch+1}/{self.cfg.max_epochs}] "
                      f"Rew: {avg_rew:.2f} | "
                      f"Val: {avg_equity:.0f} | "
                      f"Act: H{ratios[0]:.2f}/L{ratios[1]:.2f}/S{ratios[2]:.2f}/C{ratios[3]:.2f} | "
                      f"Ent: {entropy:.2f} | "
                      f"KL: {kl:.4f} | "
                      f"FPS: {fps:.0f}")
            
            # --- 早停判断 ---
            if avg_rew > best_reward + min_delta:
                best_reward = avg_rew
                early_stop_counter = 0 
                self.ppo.save(epoch, best_reward) 
                print(f"   >>> 🌟 Best Reward Updated: {best_reward:.2f} (Counter Reset)")
            else:
                early_stop_counter += 1
                print(f"   ⏳ [Patience] No improvement: {early_stop_counter}/{patience} | Best: {best_reward:.2f}")

            if early_stop_counter >= patience:
                print(f"\n🛑 [Early Stop] Triggered! Reward has not improved for {patience} epochs.")
                print(f"   Final Best Reward: {best_reward:.2f}")
                break
            
            if entropy < stop_entropy and avg_rew > 0:
                print(f"\n🛑 [Early Stop] Triggered! Entropy ({entropy:.4f}) is too low.")
                self.ppo.save(epoch, best_reward)
                break

        print(f"[Train] Finished. Data saved to {excel_path}")
        vec_env.close()


# -----------------------------------------------------------
# 入口
# -----------------------------------------------------------
if __name__ == '__main__':
    if not torch.cuda.is_available():
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)

    mp.set_start_method('spawn', force=True) 

    # 构造海量期权组合 (示例)
    all_pairs = []
    dtype = {
        'call': str,
        'put': str,
        'call_strike': int,
        'put_strike': int,
        'call_open': str,
        'put_open': str,
        'call_expire': str,
        'put_expire': str,
    }
    df = pd.read_excel('./miniQMT/datasets/all_label_data/20251213_train.xlsx', dtype=dtype)


    for index, row in df.iterrows():
        start = row['call_open']
        end = row['call_expire']

        start_time = datetime.strptime(start, "%Y%m%d")
        end_time = datetime.strptime(end, "%Y%m%d")
        days = (end_time - start_time).days

        if days <= 40:
            continue

        call = row['call']
        put = row['put']
        end_time = start_time + timedelta(days=20)
        end_time = end_time.strftime('%Y%m%d')

        start_time = start + '100000'
        end_time = end_time + '150000'
        all_pairs.append({
            'call': call,
            'put': put,
            'start_time': start_time,
            'end_time': end_time
        })



    # all_pairs.append({
    #     'call': '10006819', 'put': '10006820', 
    #     'start_time': '20240201100000', 'end_time': '20240505150000'
    # })

    # all_pairs.append({
    #     'call': '10007866', 'put': '10007875', 
    #     'start_time': '20240926100000', 'end_time': '20241113150000'
    # })

    # all_pairs.append({
    #     'call': '10008545', 'put': '10008554', 
    #     'start_time': '20250317100000', 'end_time': '20250617150000'
    # })

    cfg = AgentConfig(
        action_dim=4, 
        option_pairs=all_pairs, 
        max_epochs=500,
        max_timesteps=1000, 
        # num_workers=5      
    )
    

    agent = Agent(cfg)
    agent.train_parallel_modified_early_stop(from_check_point=False)