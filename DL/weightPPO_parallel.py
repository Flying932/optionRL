"""
    PPO算法 (标准精度 Float32 版)
    包含: Multiprocessing Parallellism + Excel Export
    已移除: Mixed Precision (AMP)
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
import multiprocessing as mp

# —— 构造每个样本的权重掩码 —— #
A_HOLD, A_LONG, A_SHORT, A_CLOSE = 0, 1, 2, 3
WEIGHT_BINS_CPU = torch.tensor([0.00, 0.25, 0.50, 0.75, 1.00])  # 离散权重

DESK_PATH = 'C:/Users/Flying/Desktop' # 请根据实际路径修改
DESK_PATH = 'C:/Users/David/Desktop' # 请根据实际路径修改

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
        # 确保输入是 float32
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
    finally:
        env.close()

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


# -----------------------------------------------------------
# PPO Agent 类 (标准 Float32 版)
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
    

    # 仅用于推理
    def load_infer_parameters(self, check_path: str=None, device: str=None):
        device = device if device else self.device
        path = check_path if check_path else self.check_path
        data = torch.load(path, map_location=self.device)

        # 加载actor, value网络和特征提取网络
        self.actor.load_state_dict(data['actor_state'])
        self.value.load_state_dict(data['value_state'])
        self.feature_adapter.load_state_dict(data['features_adapter_state'])

        print(f"[Info: 推理阶段] 加载actor, value网络和特征提取网络~")


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
        # --- 移除 Autocast ---
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
        # Float32 下可以用 -1e9
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
            # 拍扁, 得到所有轨迹所有步的curr_state
            curr_flat = raw_curr.view(T*K, -1)

            # 拍扁, 得到所有轨迹所有步的high_state
            hist_flat = raw_hist.view(T*K, -1, Dh)

            # 得到降维特征作为PPO的状态
            feat_tk = self.extract_features_batch(curr_flat, hist_flat)

            # 计算T个时间步, K个并行环境的状态价值函数
            v_tk = self.value(feat_tk).view(T, K)
            
            # 计算下一个状态
            next_feat = self.extract_features_batch(next_raw_curr, next_raw_hist)
            v_next = self.value(next_feat).squeeze(-1)
        
            # GAE 公式
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
        
        # --- 准备训练数据 ---
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
            
            # Mask
            lw = logits_w.clone()
            mask = torch.zeros_like(lw, dtype=torch.bool)
            mask[need_w.bool(), 1:] = True
            mask[~need_w.bool(), 0] = True
            
            # Float32 安全值
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

            # --- 标准反向传播 (无 Scaler) ---
            self.opt_a.zero_grad()
            self.opt_b.zero_grad()
            if self.opt_c: self.opt_c.zero_grad()
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            
            self.opt_a.step()
            self.opt_b.step()
            if self.opt_c: self.opt_c.step()

            # 记录最后一步的指标
            last_actor_loss = loss_a.item()
            last_value_loss = loss_v.item()
            last_entropy = (ent_a + 0.5 * ent_w).item()


            kl = (old_logp_flat - logp_new).mean().abs()
            if kl > 1.5 * target_kl:
                print(f"Early stop at epoch {i} KL={kl.item():.4f}")
                break
    
        return loss.item(), kl.item(), last_actor_loss, last_value_loss, last_entropy
    

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
                "best_reward": best_reward.item(),
            }
            torch.save(data, save_path)
            # 可选：打印一下路径方便确认
            print(f"[PPO] checkpoint saved to: {save_path}")

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
    start_time: str='20250408100000'
    end_time: str='20250924150000'

    # 期权环境的手续费
    fee: float=1.3

    # 期权环境的初始资金
    init_capital: float=100000.0

    # 模式: 训练/测试
    mode: str='train'

class Agent:
    def __init__(self, config: AgentConfig):
        self.cfg = config
        self.device = config.device

        # 单个测试
        self.env = None

        # 多个训练
        self.env_fns = []

        if config.mode == 'train':
            self.init_train()
        

        # --- 数据记录容器 ---
        self.records = {
            'epoch': [],
            'reward': [],
            'avg_equity': [],
            'loss': [],
            'kl': [],
            'hold_ratio': [],
            'long_ratio': [],
            'short_ratio': [],
            'close_ratio': [],
            'actor_loss': [],
            'value_loss': [],
            'entropy': [],
            'ratio_0': [],
            'ratio_25': [],
            'ratio_50': [],
            'ratio_75': [],
            'ratio_100': [],
        }

    # 训练模式初始化
    def init_train(self):
        config = self.cfg
        self.ppo = weightPPO(config.action_dim, device=self.device)
        
        self.env_fns = []
        for pair in config.option_pairs:
            def make_env(c=pair['call'], p=pair['put']):
                return windowEnv(cfg.init_capital, c, p, config.max_timesteps, 
                                 cfg.fee, config.start_time, config.end_time, '510050')
            self.env_fns.append(make_env)
        
        dummy_env = self.env_fns[0]()
        c, h, _ = dummy_env.reset()
        dummy_env.close()
        
        c_b = torch.tensor([c], dtype=torch.float32, device=self.device)
        h_b = torch.tensor([h], dtype=torch.float32, device=self.device)
        feat = self.ppo.extract_features_batch(c_b, h_b, cal_dim=True)
        self.ppo.set_actor_and_value(feat.shape[-1])


    # 测试模式设置环境, 环境的参数由外部指定
    def set_env(self, env: windowEnv):
        print(f"[Info] 设置env | call = {env.call}, put = {env.put}")
        self.env = env
        current_shape, history_shape = self.env.get_raw_shape()
        current_state = torch.zeros(current_shape)
        history_state = torch.zeros(history_shape)
        if self.ppo is None:
            self.ppo = weightPPO(self.action_dim, window_size=self.window_size, device=self.device)
            results = self.ppo.extract_features(current_state, history_state, cal_dim=True)
            _, state_dim = results.shape

            self.ppo.set_actor_and_value(state_dim)

    # 设置对state的归一化模块
    def set_norm(self, state_norm: Normalization):
        self.ppo.state_norm = state_norm
        print(f"[Info] Norm设置完成 | state.n = {self.ppo.state_norm.running_ms.n}")


    # 带早停的训练函数
    def train_parallel_modified_early_stop(self):
        # 1. 初始化并行环境
        vec_env = SubprocVectorEnv(self.env_fns)
        print(f"[Train] Start parallel training on {self.device}...")
        
        # 2. 初始化最佳奖励记录
        best_reward = -float('inf')

        # ==========================================
        # [新增] 早停机制参数 (如果没有在cfg定义，则使用默认值)
        # ==========================================
        patience = getattr(self.cfg, 'patience', 30)         # 容忍多少轮不提升
        stop_entropy = getattr(self.cfg, 'stop_entropy', 0.6) # 熵低于多少停止
        min_delta = 0.001                                    # 认为是提升的最小幅度
        early_stop_counter = 0                               # 计数器
        # ==========================================

        for epoch in range(self.cfg.max_epochs):
            start_time = time.time()
            # 每一次需要重新rewardScaling
            self.ppo.init_norm_reward_list(length=len(self.env_fns))

            # Reset 环境 (每个 Epoch 重新开始跑一遍完整数据)
            curr_np, hist_np, infos = vec_env.reset()
            
            traces = {
                'raw_curr': [], 'raw_hist': [],
                'actions': [], 'weight_idx': [], 'logp_joint': [], 
                'rewards': [], 
                'terminated': [], 'truncated': []
            }
            
            epoch_rewards = [] 
            current_equities = [0.0] * len(self.env_fns)
            
            # 标记变量
            is_hard_end = False

            # --- Rollout Loop ---
            for t in range(self.cfg.max_timesteps):

                
                # 1. 记录 State
                c_tensor = torch.as_tensor(curr_np, dtype=torch.float32, device=self.device)
                h_tensor = torch.as_tensor(hist_np, dtype=torch.float32, device=self.device)
                
                # 2. 决策
                with torch.no_grad():
                     state = self.ppo.extract_features_batch(c_tensor, h_tensor)
                     a, w_idx, w_val, _, _, logp_joint = self.ppo.selete_action_and_weight(state)
                
                actions_np = a.cpu().numpy()
                weights_np = w_val.cpu().numpy()
                
                # 3. 步进
                next_curr, next_hist, rews, terms, truncs, infos = vec_env.step(actions_np, weights_np)

                                
                scaled_rewards = []
                for num in range(len(self.env_fns)):
                    r = rews[num].item()
                    r_norm = self.ppo.reward_norm_list[num]
                    scaled_rewards.append(r_norm(r))
                
                # 更新 Log
                for i, info in enumerate(infos):
                    if isinstance(info, dict):
                        key = 'final_equity' if (terms[i] or truncs[i]) else 'equity'
                        if key in info:
                            current_equities[i] = info[key]

                # === 奖励对齐 ===
                # 当前 rews 是上一步动作的果，填入 traces['rewards']
                # 注意：第一步(t=0)产生的 rews 是无意义的(属于过去)，所以 t>0 才填
                if t > 0:
                    rew_tensor = torch.as_tensor(scaled_rewards, dtype=torch.float32, device=self.device)
                    traces['rewards'].append(rew_tensor)
                    epoch_rewards.append(rews)

                # 4. 记录当前步数据 (Action, State)
                # 此时我们还没拿到当前 Action 的 Reward
                traces['raw_curr'].append(c_tensor)
                traces['raw_hist'].append(h_tensor)
                traces['actions'].append(a)
                traces['weight_idx'].append(w_idx)
                traces['logp_joint'].append(logp_joint)
                traces['terminated'].append(torch.as_tensor(terms, device=self.device))
                traces['truncated'].append(torch.as_tensor(truncs, device=self.device))
                
                curr_np, hist_np = next_curr, next_hist
                
                # === 检测 Hard End (数据跑完了) ===
                if np.all(terms | truncs):
                    is_hard_end = True
                    # 此时 rews 包含 (A_{t-1} 的盈亏 + 终点 Bonus)
                    # 这个 rews 已经在上面 "if t > 0" 里填给了 A_{t-1}，处理完毕。
                    
                    # 关键：当前的动作 A_t (最后一步) 刚刚被 append 进 traces。
                    # 但是环境已经 Done 了，A_t 没有未来，无法结算。
                    # 我们将在循环外把它删掉。
                    break

            # --- 循环后处理 (Post-processing) ---
            
            if is_hard_end:
                # === 情况 A: Hard End ===
                # traces 里多记录了最后一步动作 A_t，但它没有 Reward (因为没有 T+1)
                # 我们必须把它删掉，让数据对齐
                
                # 1. 保存最后一步的状态作为 "Next State" (给 A_{t-1} 用)
                # 此时 traces['raw_curr'][-1] 就是发生 Done 时的状态 S_t
                traces['next_raw_curr'] = traces['raw_curr'][-1]
                traces['next_raw_hist'] = traces['raw_hist'][-1]
                
                # 2. 弹出所有列表的最后一个元素 (删除 A_t)
                for k in ['raw_curr', 'raw_hist', 'actions', 'weight_idx', 'logp_joint', 'terminated', 'truncated']:
                    traces[k].pop()
                
                # 此时 len(actions) 减少 1，与 len(rewards) 完美对齐
                
            else:
                # === 情况 B: Soft End (只是时间到了) ===
                # 数据还没完，我们可以再走一步(Probe)来结算最后一步动作 A_t
                
                # 1. 探测步
                hold_actions = np.zeros(len(self.env_fns), dtype=int)
                hold_weights = np.zeros(len(self.env_fns), dtype=float)
                _, _, final_rews, _, _, _ = vec_env.step(hold_actions, hold_weights)
                
                # 2. 补齐 Reward (给 A_t)
                rew_tensor = torch.as_tensor(final_rews, dtype=torch.float32, device=self.device)
                traces['rewards'].append(rew_tensor)
                epoch_rewards.append(final_rews)
                
                # 3. 设置 Next State
                traces['next_raw_curr'] = torch.as_tensor(curr_np, dtype=torch.float32, device=self.device)
                traces['next_raw_hist'] = torch.as_tensor(hist_np, dtype=torch.float32, device=self.device)

            # --- 最终校验 ---
            n_act = len(traces['actions'])
            n_rew = len(traces['rewards'])
            assert n_act == n_rew, f"对齐严重错误: Act={n_act}, Rew={n_rew}, HardEnd={is_hard_end}"

            # 6. 更新 (如果数据有效)
            if n_act > 0:
                loss, kl, actor_loss, value_loss, entropy = self.ppo.update_parallel(traces)
            else:
                loss, kl, actor_loss, value_loss, entropy = 0, 0, 0, 0, 0

            # --- Log & Export ---
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
                weight_counts = np.bincount(all_weights, minlength=4)
                weight_ratios = weight_counts / (len(all_weights) + 1e-8)

            else:
                ratios = [0, 0, 0, 0]

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
                self.records[names[k]].append(weight_ratios[k])

            df = pd.DataFrame(self.records)
            excel_path = f'{DESK_PATH}/PPO_training_data.xlsx'
            try:
                df.to_excel(excel_path, index=False)
                print(f"🌟 Save records to {excel_path}")
            except:
                pass

            if (epoch + 1) % self.cfg.print_interval == 0:
                print(f"[Epoch {epoch+1}/{self.cfg.max_epochs}] "
                      f"Rew: {avg_rew:.2f} | "
                      f"Val: {avg_equity:.0f} | "
                      f"Act: H{ratios[0]:.2f}/L{ratios[1]:.2f}/S{ratios[2]:.2f}/C{ratios[3]:.2f} | "
                      f"Ent: {entropy:.2f} | "
                      f"KL: {kl:.4f} | "
                      f"FPS: {fps:.0f}")
            
            # ==========================================
            # [新增] 早停逻辑核心实现
            # ==========================================
            # 1. 检查 Reward 是否创新高 (引入 min_delta 防止微小抖动)
            if avg_rew > best_reward + min_delta:
                best_reward = avg_rew
                early_stop_counter = 0 # 重置耐心值
                self.ppo.save(epoch, best_reward)
                print(f"   >>> 🌟 Best Reward Updated: {best_reward:.2f} (Counter Reset)")
            else:
                early_stop_counter += 1
                print(f"   ⏳ [Patience] No improvement: {early_stop_counter}/{patience} | Best: {best_reward:.2f}")

            # 2. 触发条件 A: 耐心耗尽
            if early_stop_counter >= patience:
                print(f"\n🛑 [Early Stop] Triggered! Reward has not improved for {patience} epochs.")
                print(f"   Final Best Reward: {best_reward:.2f}")
                break
            
            # 3. 触发条件 B: 熵过低 (策略固化)
            if entropy < stop_entropy and avg_rew > 0:
                print(f"\n🛑 [Early Stop] Triggered! Entropy ({entropy:.4f}) is too low, policy has converged.")
                break
            # ==========================================

        print(f"[Train] Finished. Data saved to {excel_path}")
        vec_env.close()


    # 测试
    def test(self, epochs: int = 5, alpha: float = 0.5, test_mode: bool = False):
        res = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for epoch in range(epochs):
            # 和 train 一样的 reset 逻辑
            current_state, history_state, _ = self.env.reset()

            # 第一次可以用默认 cal_dim=True，后续用 cal_dim=False
            current_state = torch.as_tensor(current_state, dtype=torch.float32, device=device)
            history_state = torch.as_tensor(history_state, dtype=torch.float32, device=device)
            
            if current_state.dim() == 1:
                current_state =  current_state.unsqueeze(0)
            
            if history_state.dim() == 2:
                history_state = history_state.unsqueeze(0)

            state = self.ppo.extract_features_batch(current_state, history_state, cal_dim=True)

            done = False
            total_reward = 0.0

            action_list = []
            weight_list = []

            for t in range(self.cfg.max_timesteps):
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
                
                current_state = torch.as_tensor(current_state, dtype=torch.float32, device=device)
                history_state = torch.as_tensor(history_state, dtype=torch.float32, device=device)
                
                if current_state.dim() == 1:
                    current_state =  current_state.unsqueeze(0)
                
                if history_state.dim() == 2:
                    history_state = history_state.unsqueeze(0)


                done = terminated or truncated

                # 提取下一步特征（这里 cal_dim=False，和训练时一致）
                next_state = self.ppo.extract_features_batch(
                    current_state, history_state, cal_dim=False
                )

                # 如果结束，加上基于最终权益的额外 reward
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
            res.append((value, total_reward))

        self.env.close()
        return res



def save_norm(agent, filepath: str='./miniQMT/DL/checkout/norm.pkl'):
    data = {
        'reward_norm': agent.ppo.reward_norm,
        'state_norm': agent.ppo.state_norm,
    }
    
    torch.save(data, filepath)


def read_norm(filepath: str = './miniQMT/DL/checkout/norm_fixed.pkl'):
    """
    加载由 torch.save 保存的归一化参数
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaded_data = torch.load(filepath, map_location=device, weights_only=False)
    
    return loaded_data['reward_norm'], loaded_data['state_norm']

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) 

    pairs = []
    pairs.append({'call': '10007709', 'call_strike': 2.25, 'call_expire': '20241023', 'put': '10007718', 'put_strike': 2.25, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'})
    pairs.append({'call': '10007710', 'call_strike': 2.3, 'call_expire': '20241023', 'put': '10007719', 'put_strike': 2.3, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'})
    pairs.append({'call': '10007711', 'call_strike': 2.35, 'call_expire': '20241023', 'put': '10007720', 'put_strike': 2.35, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'})
    pairs.append({'call': '10007712', 'call_strike': 2.4, 'call_expire': '20241023', 'put': '10007721', 'put_strike': 2.4, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'})
    pairs.append({'call': '10007713', 'call_strike': 2.45, 'call_expire': '20241023', 'put': '10007722', 'put_strike': 2.45, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'})
    pairs.append({'call': '10007714', 'call_strike': 2.5, 'call_expire': '20241023', 'put': '10007723', 'put_strike': 2.5, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'})
    pairs.append({'call': '10007715', 'call_strike': 2.55, 'call_expire': '20241023', 'put': '10007724', 'put_strike': 2.55, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'})
    pairs.append({'call': '10007716', 'call_strike': 2.6, 'call_expire': '20241023', 'put': '10007725', 'put_strike': 2.6, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'})
    pairs.append({'call': '10007717', 'call_strike': 2.65, 'call_expire': '20241023', 'put': '10007726', 'put_strike': 2.65, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'})
    pairs.append({'call': '10008505', 'call_strike': 2.5, 'call_expire': '20250122', 'put': '10008514', 'put_strike': 2.5, 'put_expire': '20250122', 'call_open': '20241202', 'put_open': '20241202'})
    pairs.append({'call': '10008506', 'call_strike': 2.55, 'call_expire': '20250122', 'put': '10008515', 'put_strike': 2.55, 'put_expire': '20250122', 'call_open': '20241202', 'put_open': '20241202'})
    pairs.append({'call': '10008507', 'call_strike': 2.6, 'call_expire': '20250122', 'put': '10008516', 'put_strike': 2.6, 'put_expire': '20250122', 'call_open': '20241202', 'put_open': '20241202'})
    pairs.append({'call': '10008508', 'call_strike': 2.65, 'call_expire': '20250122', 'put': '10008517', 'put_strike': 2.65, 'put_expire': '20250122', 'call_open': '20241202', 'put_open': '20241202'})
    pairs.append({'call': '10008509', 'call_strike': 2.7, 'call_expire': '20250122', 'put': '10008518', 'put_strike': 2.7, 'put_expire': '20250122', 'call_open': '20241202', 'put_open': '20241202'})
    pairs.append({'call': '10008510', 'call_strike': 2.75, 'call_expire': '20250122', 'put': '10008519', 'put_strike': 2.75, 'put_expire': '20250122', 'call_open': '20241202', 'put_open': '20241202'})
    pairs.append({'call': '10008511', 'call_strike': 2.8, 'call_expire': '20250122', 'put': '10008520', 'put_strike': 2.8, 'put_expire': '20250122', 'call_open': '20241202', 'put_open': '20241202'})
    pairs.append({'call': '10008512', 'call_strike': 2.85, 'call_expire': '20250122', 'put': '10008521', 'put_strike': 2.85, 'put_expire': '20250122', 'call_open': '20241202', 'put_open': '20241202'})
    pairs.append({'call': '10008545', 'call_strike': 2.7, 'call_expire': '20250625', 'put': '10008554', 'put_strike': 2.7, 'put_expire': '20250625', 'call_open': '20241202', 'put_open': '20241202'})
    pairs.append({'call': '10008547', 'call_strike': 2.8, 'call_expire': '20250625', 'put': '10008556', 'put_strike': 2.8, 'put_expire': '20250625', 'call_open': '20241202', 'put_open': '20241202'})
    pairs.append({'call': '10008548', 'call_strike': 2.85, 'call_expire': '20250625', 'put': '10008557', 'put_strike': 2.85, 'put_expire': '20250625', 'call_open': '20241202', 'put_open': '20241202'})
    pairs.append({'call': '10008573', 'call_strike': 2.55, 'call_expire': '20250226', 'put': '10008582', 'put_strike': 2.55, 'put_expire': '20250226', 'call_open': '20241226', 'put_open': '20241226'})
    pairs.append({'call': '10008574', 'call_strike': 2.6, 'call_expire': '20250226', 'put': '10008583', 'put_strike': 2.6, 'put_expire': '20250226', 'call_open': '20241226', 'put_open': '20241226'})
    pairs.append({'call': '10008575', 'call_strike': 2.65, 'call_expire': '20250226', 'put': '10008584', 'put_strike': 2.65, 'put_expire': '20250226', 'call_open': '20241226', 'put_open': '20241226'})
    pairs.append({'call': '10008576', 'call_strike': 2.7, 'call_expire': '20250226', 'put': '10008585', 'put_strike': 2.7, 'put_expire': '20250226', 'call_open': '20241226', 'put_open': '20241226'})
    pairs.append({'call': '10008577', 'call_strike': 2.75, 'call_expire': '20250226', 'put': '10008586', 'put_strike': 2.75, 'put_expire': '20250226', 'call_open': '20241226', 'put_open': '20241226'})
    pairs.append({'call': '10008578', 'call_strike': 2.8, 'call_expire': '20250226', 'put': '10008587', 'put_strike': 2.8, 'put_expire': '20250226', 'call_open': '20241226', 'put_open': '20241226'})
    pairs.append({'call': '10008676', 'call_strike': 2.5, 'call_expire': '20250226', 'put': '10008678', 'put_strike': 2.5, 'put_expire': '20250226', 'call_open': '20250103', 'put_open': '20250103'})
    pairs.append({'call': '10008801', 'call_strike': 2.8, 'call_expire': '20250924', 'put': '10008810', 'put_strike': 2.8, 'put_expire': '20250924', 'call_open': '20250123', 'put_open': '20250123'})
    pairs.append({'call': '10008885', 'call_strike': 2.85, 'call_expire': '20250924', 'put': '10008886', 'put_strike': 2.85, 'put_expire': '20250924', 'call_open': '20250127', 'put_open': '20250127'})
    pairs.append({'call': '10008895', 'call_strike': 2.9, 'call_expire': '20250924', 'put': '10008896', 'put_strike': 2.9, 'put_expire': '20250924', 'call_open': '20250211', 'put_open': '20250211'})
    pairs.append({'call': '10008905', 'call_strike': 2.95, 'call_expire': '20250924', 'put': '10008906', 'put_strike': 2.95, 'put_expire': '20250924', 'call_open': '20250224', 'put_open': '20250224'})
    pairs.append({'call': '10009039', 'call_strike': 3.0, 'call_expire': '20250924', 'put': '10009040', 'put_strike': 3.0, 'put_expire': '20250924', 'call_open': '20250317', 'put_open': '20250317'})
    pairs.append({'call': '10009325', 'call_strike': 2.65, 'call_expire': '20250723', 'put': '10009334', 'put_strike': 2.65, 'put_expire': '20250723', 'call_open': '20250529', 'put_open': '20250529'})
    pairs.append({'call': '10009326', 'call_strike': 2.7, 'call_expire': '20250723', 'put': '10009335', 'put_strike': 2.7, 'put_expire': '20250723', 'call_open': '20250529', 'put_open': '20250529'})
    pairs.append({'call': '10009327', 'call_strike': 2.75, 'call_expire': '20250723', 'put': '10009336', 'put_strike': 2.75, 'put_expire': '20250723', 'call_open': '20250529', 'put_open': '20250529'})
    pairs.append({'call': '10009328', 'call_strike': 2.8, 'call_expire': '20250723', 'put': '10009337', 'put_strike': 2.8, 'put_expire': '20250723', 'call_open': '20250529', 'put_open': '20250529'})
    pairs.append({'call': '10009329', 'call_strike': 2.85, 'call_expire': '20250723', 'put': '10009338', 'put_strike': 2.85, 'put_expire': '20250723', 'call_open': '20250529', 'put_open': '20250529'})
    pairs.append({'call': '10009330', 'call_strike': 2.9, 'call_expire': '20250723', 'put': '10009339', 'put_strike': 2.9, 'put_expire': '20250723', 'call_open': '20250529', 'put_open': '20250529'})
    pairs.append({'call': '10009331', 'call_strike': 2.95, 'call_expire': '20250723', 'put': '10009340', 'put_strike': 2.95, 'put_expire': '20250723', 'call_open': '20250529', 'put_open': '20250529'})
    pairs.append({'call': '10009495', 'call_strike': 3.1, 'call_expire': '20250924', 'put': '10009496', 'put_strike': 3.1, 'put_expire': '20250924', 'call_open': '20250626', 'put_open': '20250626'})
    pairs.append({'call': '10009511', 'call_strike': 2.75, 'call_expire': '20250827', 'put': '10009520', 'put_strike': 2.75, 'put_expire': '20250827', 'call_open': '20250626', 'put_open': '20250626'})
    pairs.append({'call': '10009512', 'call_strike': 2.8, 'call_expire': '20250827', 'put': '10009521', 'put_strike': 2.8, 'put_expire': '20250827', 'call_open': '20250626', 'put_open': '20250626'})
    pairs.append({'call': '10009513', 'call_strike': 2.85, 'call_expire': '20250827', 'put': '10009522', 'put_strike': 2.85, 'put_expire': '20250827', 'call_open': '20250626', 'put_open': '20250626'})
    pairs.append({'call': '10009514', 'call_strike': 2.9, 'call_expire': '20250827', 'put': '10009523', 'put_strike': 2.9, 'put_expire': '20250827', 'call_open': '20250626', 'put_open': '20250626'})
    pairs.append({'call': '10009515', 'call_strike': 2.95, 'call_expire': '20250827', 'put': '10009524', 'put_strike': 2.95, 'put_expire': '20250827', 'call_open': '20250626', 'put_open': '20250626'})
    pairs.append({'call': '10009516', 'call_strike': 3.0, 'call_expire': '20250827', 'put': '10009525', 'put_strike': 3.0, 'put_expire': '20250827', 'call_open': '20250626', 'put_open': '20250626'})
    pairs.append({'call': '10009517', 'call_strike': 3.1, 'call_expire': '20250827', 'put': '10009526', 'put_strike': 3.1, 'put_expire': '20250827', 'call_open': '20250626', 'put_open': '20250626'})
    pairs.append({'call': '10009619', 'call_strike': 3.2, 'call_expire': '20250827', 'put': '10009620', 'put_strike': 3.2, 'put_expire': '20250827', 'call_open': '20250721', 'put_open': '20250721'})
    pairs.append({'call': '10009621', 'call_strike': 3.2, 'call_expire': '20250924', 'put': '10009622', 'put_strike': 3.2, 'put_expire': '20250924', 'call_open': '20250721', 'put_open': '20250721'})


    # 202406~202412
    new_option_pairs = [
        {'call': '10006421', 'call_strike': 2.35, 'call_expire': '20240626', 'put': '10006430', 'put_strike': 2.35, 'put_expire': '20240626', 'call_open': '20231127', 'put_open': '20231127'},
        {'call': '10006422', 'call_strike': 2.4, 'call_expire': '20240626', 'put': '10006431', 'put_strike': 2.4, 'put_expire': '20240626', 'call_open': '20231127', 'put_open': '20231127'},
        {'call': '10006423', 'call_strike': 2.45, 'call_expire': '20240626', 'put': '10006432', 'put_strike': 2.45, 'put_expire': '20240626', 'call_open': '20231127', 'put_open': '20231127'},
        {'call': '10006424', 'call_strike': 2.5, 'call_expire': '20240626', 'put': '10006433', 'put_strike': 2.5, 'put_expire': '20240626', 'call_open': '20231127', 'put_open': '20231127'},
        {'call': '10006425', 'call_strike': 2.55, 'call_expire': '20240626', 'put': '10006434', 'put_strike': 2.55, 'put_expire': '20240626', 'call_open': '20231127', 'put_open': '20231127'},
        {'call': '10006426', 'call_strike': 2.6, 'call_expire': '20240626', 'put': '10006435', 'put_strike': 2.6, 'put_expire': '20240626', 'call_open': '20231127', 'put_open': '20231127'},
        {'call': '10006427', 'call_strike': 2.65, 'call_expire': '20240626', 'put': '10006436', 'put_strike': 2.65, 'put_expire': '20240626', 'call_open': '20231127', 'put_open': '20231127'},
        {'call': '10006732', 'call_strike': 2.25, 'call_expire': '20240925', 'put': '10006741', 'put_strike': 2.25, 'put_expire': '20240925', 'call_open': '20240125', 'put_open': '20240125'},
        {'call': '10006734', 'call_strike': 2.35, 'call_expire': '20240925', 'put': '10006743', 'put_strike': 2.35, 'put_expire': '20240925', 'call_open': '20240125', 'put_open': '20240125'},
        {'call': '10006735', 'call_strike': 2.4, 'call_expire': '20240925', 'put': '10006744', 'put_strike': 2.4, 'put_expire': '20240925', 'call_open': '20240125', 'put_open': '20240125'},
        {'call': '10006736', 'call_strike': 2.45, 'call_expire': '20240925', 'put': '10006745', 'put_strike': 2.45, 'put_expire': '20240925', 'call_open': '20240125', 'put_open': '20240125'},
        {'call': '10006737', 'call_strike': 2.5, 'call_expire': '20240925', 'put': '10006746', 'put_strike': 2.5, 'put_expire': '20240925', 'call_open': '20240125', 'put_open': '20240125'},
        {'call': '10006819', 'call_strike': 2.55, 'call_expire': '20240925', 'put': '10006820', 'put_strike': 2.55, 'put_expire': '20240925', 'call_open': '20240126', 'put_open': '20240126'},
        {'call': '10007333', 'call_strike': 2.35, 'call_expire': '20240724', 'put': '10007342', 'put_strike': 2.35, 'put_expire': '20240724', 'call_open': '20240523', 'put_open': '20240523'},
        {'call': '10007334', 'call_strike': 2.4, 'call_expire': '20240724', 'put': '10007343', 'put_strike': 2.4, 'put_expire': '20240724', 'call_open': '20240523', 'put_open': '20240523'},
        {'call': '10007335', 'call_strike': 2.45, 'call_expire': '20240724', 'put': '10007344', 'put_strike': 2.45, 'put_expire': '20240724', 'call_open': '20240523', 'put_open': '20240523'},
        {'call': '10007336', 'call_strike': 2.5, 'call_expire': '20240724', 'put': '10007345', 'put_strike': 2.5, 'put_expire': '20240724', 'call_open': '20240523', 'put_open': '20240523'},
        {'call': '10007337', 'call_strike': 2.55, 'call_expire': '20240724', 'put': '10007346', 'put_strike': 2.55, 'put_expire': '20240724', 'call_open': '20240523', 'put_open': '20240523'},
        {'call': '10007338', 'call_strike': 2.6, 'call_expire': '20240724', 'put': '10007347', 'put_strike': 2.6, 'put_expire': '20240724', 'call_open': '20240523', 'put_open': '20240523'},
        {'call': '10007452', 'call_strike': 2.3, 'call_expire': '20240828', 'put': '10007461', 'put_strike': 2.3, 'put_expire': '20240828', 'call_open': '20240627', 'put_open': '20240627'},
        {'call': '10007453', 'call_strike': 2.35, 'call_expire': '20240828', 'put': '10007462', 'put_strike': 2.35, 'put_expire': '20240828', 'call_open': '20240627', 'put_open': '20240627'},
        {'call': '10007454', 'call_strike': 2.4, 'call_expire': '20240828', 'put': '10007463', 'put_strike': 2.4, 'put_expire': '20240828', 'call_open': '20240627', 'put_open': '20240627'},
        {'call': '10007455', 'call_strike': 2.45, 'call_expire': '20240828', 'put': '10007464', 'put_strike': 2.45, 'put_expire': '20240828', 'call_open': '20240627', 'put_open': '20240627'},
        {'call': '10007456', 'call_strike': 2.5, 'call_expire': '20240828', 'put': '10007465', 'put_strike': 2.5, 'put_expire': '20240828', 'call_open': '20240627', 'put_open': '20240627'},
        {'call': '10007457', 'call_strike': 2.55, 'call_expire': '20240828', 'put': '10007466', 'put_strike': 2.55, 'put_expire': '20240828', 'call_open': '20240627', 'put_open': '20240627'},
        {'call': '10007458', 'call_strike': 2.6, 'call_expire': '20240828', 'put': '10007467', 'put_strike': 2.6, 'put_expire': '20240828', 'call_open': '20240627', 'put_open': '20240627'},
        {'call': '10007709', 'call_strike': 2.25, 'call_expire': '20241023', 'put': '10007718', 'put_strike': 2.25, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'},
        {'call': '10007710', 'call_strike': 2.3, 'call_expire': '20241023', 'put': '10007719', 'put_strike': 2.3, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'},
        {'call': '10007711', 'call_strike': 2.35, 'call_expire': '20241023', 'put': '10007720', 'put_strike': 2.35, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'},
        {'call': '10007712', 'call_strike': 2.4, 'call_expire': '20241023', 'put': '10007721', 'put_strike': 2.4, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'},
        {'call': '10007713', 'call_strike': 2.45, 'call_expire': '20241023', 'put': '10007722', 'put_strike': 2.45, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'},
        {'call': '10007714', 'call_strike': 2.5, 'call_expire': '20241023', 'put': '10007723', 'put_strike': 2.5, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'},
        {'call': '10007715', 'call_strike': 2.55, 'call_expire': '20241023', 'put': '10007724', 'put_strike': 2.55, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'},
        {'call': '10007716', 'call_strike': 2.6, 'call_expire': '20241023', 'put': '10007725', 'put_strike': 2.6, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'},
        {'call': '10007717', 'call_strike': 2.65, 'call_expire': '20241023', 'put': '10007726', 'put_strike': 2.65, 'put_expire': '20241023', 'call_open': '20240829', 'put_open': '20240829'},
        {'call': '10007866', 'call_strike': 2.55, 'call_expire': '20241127', 'put': '10007875', 'put_strike': 2.55, 'put_expire': '20241127', 'call_open': '20240926', 'put_open': '20240926'},
        {'call': '10007867', 'call_strike': 2.6, 'call_expire': '20241127', 'put': '10007876', 'put_strike': 2.6, 'put_expire': '20241127', 'call_open': '20240926', 'put_open': '20240926'},
        {'call': '10007868', 'call_strike': 2.65, 'call_expire': '20241127', 'put': '10007877', 'put_strike': 2.65, 'put_expire': '20241127', 'call_open': '20240926', 'put_open': '20240926'},
        {'call': '10007869', 'call_strike': 2.7, 'call_expire': '20241127', 'put': '10007878', 'put_strike': 2.7, 'put_expire': '20241127', 'call_open': '20240926', 'put_open': '20240926'},
        {'call': '10007955', 'call_strike': 2.75, 'call_expire': '20241127', 'put': '10007957', 'put_strike': 2.75, 'put_expire': '20241127', 'call_open': '20240927', 'put_open': '20240927'},
        {'call': '10007956', 'call_strike': 2.8, 'call_expire': '20241127', 'put': '10007958', 'put_strike': 2.8, 'put_expire': '20241127', 'call_open': '20240927', 'put_open': '20240927'},
        {'call': '10007985', 'call_strike': 2.85, 'call_expire': '20241127', 'put': '10007987', 'put_strike': 2.85, 'put_expire': '20241127', 'call_open': '20240930', 'put_open': '20240930'},
        {'call': '10007986', 'call_strike': 2.9, 'call_expire': '20241127', 'put': '10007988', 'put_strike': 2.9, 'put_expire': '20241127', 'call_open': '20240930', 'put_open': '20240930'},
        {'call': '10008047', 'call_strike': 2.95, 'call_expire': '20241127', 'put': '10008051', 'put_strike': 2.95, 'put_expire': '20241127', 'call_open': '20241008', 'put_open': '20241008'},
        {'call': '10008048', 'call_strike': 3.0, 'call_expire': '20241127', 'put': '10008052', 'put_strike': 3.0, 'put_expire': '20241127', 'call_open': '20241008', 'put_open': '20241008'}
    ]



    start_time: str='20250317100000'
    end_time: str='20250617150000'
    option_pairs = []
    option_pairs.append({'call': '10008545', 'put': '10008554'})
    option_pairs.append({'call': '10008547', 'put': '10008556'})
    option_pairs.append({'call': '10008548', 'put': '10008557'})
    option_pairs.append({'call': '10008801', 'put': '10008810'})
    option_pairs.append({'call': '10008885', 'put': '10008886'})
    option_pairs.append({'call': '10008895', 'put': '10008896'})
    option_pairs.append({'call': '10008905', 'put': '10008906'})
    option_pairs.append({'call': '10009039', 'put': '10009040'})

    cfg = AgentConfig(action_dim=4, option_pairs=option_pairs, start_time=start_time, end_time=end_time)
    agent = Agent(cfg)
    agent.train_parallel_modified_early_stop()
    # save_norm(agent)

    # 测试
    # idx = 2
    # call, put = option_pairs[idx]['call'], option_pairs[idx]['put']

    # call, put = '10006038', '10006029'
    # start_time = '20230928100000'
    # end_time = '20231122150000'

    # calls = [calls[1]]
    # puts = [puts[1]]

    print(f"[PPO-Agent] 开始测试......")
    # for idx in range(len(option_pairs)):
    # call, put = option_pairs[idx]['call'], option_pairs[idx]['put']
    # start_time = option_pairs[idx]['call_open'] + '100000'
    # end_time = option_pairs[idx]['call_expire'] + '150000'

    call, put = '10006819', '10006820'
    start_time = '20240201100000'
    end_time = '20240505150000'

    call, put = '10007866', '10007875'
    start_time = '20240926100000'
    end_time = '20241113150000'

    # call, put = '10008545', '10008554'
    # start_time: str='20250317100000'
    # end_time: str='20250617150000'

    call, put = '10008934', '10008943'
    start_time = '20250227100000'
    end_time = '20250410150000'


    # start_time: str='20250317100000'
    # end_time: str='20250617150000'
    # option_pairs = []
    # option_pairs.append({'call': '10008545', 'put': '10008554'})
    option_pairs = [{'call': call, 'put': put}]
    cfg = AgentConfig(action_dim=4, option_pairs=option_pairs, start_time=start_time, end_time=end_time)
    agent = Agent(cfg)

    max_timesteps = 1000
    fee = 1.3
    benchmark ='510050'
    env = windowEnv(100000, call, put, max_timesteps, fee, start_time, end_time, benchmark, normalize_reward=False)
    reward_norm, state_norm = read_norm()
    
    agent.set_env(env)
    agent.set_norm(state_norm)
    # 加载模型权重
    agent.ppo.load_infer_parameters()
    
    res = agent.test(test_mode=False, epochs=3)

    print(0 / 0)