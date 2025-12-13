"""
    本模块实现PPO算法的部分, 这也是整个架构最核心的部分.
    需要对接我们的模拟miniQMT环境

    本PPO算法需要实现:
        输出动作: (LONG, SHORT, CLOSE_ALL, HOLD)
        输出权重: weight in [0, 1]
"""
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
import os
import pickle

# —— 构造每个样本的权重掩码 —— #
A_HOLD, A_LONG, A_SHORT, A_CLOSE = 0, 1, 2, 3
WEIGHT_BINS_CPU = torch.tensor([0.00, 0.25, 0.50, 0.75, 1.00])  # 离散权重


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

now = datetime.now()
# 格式化为时分秒字符串
time_str = now.strftime("%H-%M-%S")
file_path = f'C:/Users/Flying/Desktop/PPO_records_{time_str}.txt'


sys.stdout = outPut(file_path)



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


# PPO-agent
class weightPPO:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 actor_lr: float=1e-4,
                 value_lr: float=3e-4,
                 gamma: float=0.99,
                 clip_eps: float=0.2,
                 k_epochs: int=5,
                 device: str='cpu',
                 check_path: str='./miniQMT/DL/checkout/check_data.pt',
                 resume: bool=False,
                 ):
        self.device = 'cuda' if torch.cuda.is_available() else device
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.resume = resume
        self.check_path = check_path

        self.actor = ActorDualHead(state_dim, n_actions=action_dim).to(self.device)
        self.value = Value(state_dim).to(self.device)

        self.opt_a = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.opt_b = optim.Adam(self.value.parameters(), lr=value_lr)

        # 放设备上的权重离散值
        self.WEIGHT_BINS = WEIGHT_BINS_CPU.to(self.device)

    @torch.no_grad()
    def selete_action_and_weight(self, state, test: bool=False):
        """
        采样动作 + 权重，并且在 rollout 阶段就计算“联合 logprob”（与训练用掩码完全一致）
        """
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)  # [K,L] 或 [L]
        logits_a, logits_w = self.actor(s)

        # 动作分布
        dist_a = Categorical(logits=logits_a)
        if test:
            a = torch.argmax(logits_a, dim=-1)
        else:
            a = dist_a.sample()                    # [K]
        logp_a = dist_a.log_prob(a)            # [K]

        # 权重掩码（与训练端完全一致）
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

    # 计算GAE（矢量版）
    def compute_gae_vector(
        self,
        rewards,            # list[float], len=T
        values,             # torch.Tensor, [T,K]
        next_value,         # torch.Tensor, [K]
        terminateds,        # list[int], len=T
        valid_mask=None,    # torch.BoolTensor, [T,K]
        weights=None,       # torch.Tensor or None, [T,K]
        gamma: float=None,
        lam: float=0.95
    ):
        device = self.device
        gamma = self.gamma if not gamma else gamma
        r = torch.as_tensor(rewards, dtype=torch.float32, device=device)        # [T]
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

        # 聚合权重（默认均匀平均）
        if weights is None:
            denom = vm.sum(dim=1, keepdim=True).clamp_min(1.0)
            w = vm / denom
        else:
            w = (weights * vm)
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)

        # 供 PPO 用的标量优势（按 K 加权汇总）
        adv_scalar = (adv * w).sum(dim=1)               # [T]
        return adv_scalar, returns, adv                 # returns/adv: [T,K]

    def update(self, traces, target_kl: float = 0.015, entropy_coef: float = 0.001, value_coef: float = 0.5):
        device = self.device

        states = torch.as_tensor(traces['states'], dtype=torch.float32, device=device)   # [T,K,L]
        actions = torch.as_tensor(traces['actions'], dtype=torch.long, device=device)    
        w_idx = torch.as_tensor(traces['weight_idx'], dtype=torch.long, device=device)   
        old_logp_joint = torch.as_tensor(traces['logp_joint'], dtype=torch.float32, device=device)

        rewards = torch.as_tensor(traces['rewards'], dtype=torch.float32, device=device)        
        terminateds = torch.as_tensor(traces['terminated'], dtype=torch.bool, device=device)    
        truncateds = torch.as_tensor(traces['truncated'], dtype=torch.bool, device=device)
        next_state = torch.as_tensor(traces['next_state'], dtype=torch.float32, device=device)   

        dones = terminateds | truncateds

        T, K, L = states.shape

        # ===== 1. GAE & 优势标准化 =====
        with torch.no_grad():
            v_tk = self.value(states.view(T * K, L)).squeeze(-1).view(T, K)
            v_next = self.value(next_state).squeeze(-1)

            adv_1d, ret_tk, _ = self.compute_gae_vector(
                rewards.tolist(), v_tk, v_next, dones.tolist(),
                valid_mask=None, weights=None, gamma=self.gamma, lam=0.95
            )

            # 优势标准化 + 裁剪
            adv_1d = torch.clamp(adv_1d, -3.0, 3.0)   # 超保守裁剪
            adv_1d = (adv_1d - adv_1d.mean()) / (adv_1d.std() + 1e-8)
            
            adv = adv_1d.unsqueeze(1).expand(T, K).reshape(T * K)
            rets = ret_tk.reshape(T * K)

        # 展平
        s_flat = states.view(T * K, L)
        a_flat = actions.view(T * K)
        w_flat = w_idx.view(T * K)
        old_logp_flat = old_logp_joint.view(T * K).detach()

        need_w = ((a_flat == A_LONG) | (a_flat == A_SHORT) | (a_flat == A_CLOSE)).float()
        old_v_flat = v_tk.reshape(T * K)

        print(
            f"[update] adv(global): mean={adv_1d.mean():.4f}, std={adv_1d.std():.4f}, "
            f"min={adv_1d.min():.4f}, max={adv_1d.max():.4f}"
        )

        entropy_last = None
        actor_loss_last = None
        value_loss_last = None

        for e in range(self.k_epochs):
            # 记录更新前的 logits（用于看 policy 改变幅度）
            with torch.no_grad():
                old_logits_a, old_logits_w = self.actor(s_flat)

            # ===== 前向：当前策略 =====
            logits_a, logits_w = self.actor(s_flat)

            dist_a = Categorical(logits=logits_a)
            new_logp_a = dist_a.log_prob(a_flat)
            ent_a = dist_a.entropy()      # [T*K]

            # 权重头：只在 need_w 的时候使用 [1..]，否则只用 [0]
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

            # ===== Value Loss（clip） =====
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
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.3)
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.3)

            self.opt_a.step()
            self.opt_b.step()

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

            print(
                f" actor={actor_loss:.5f} value={value_loss:.5f} "
                f"ent={entropy:.3f} (ent_a={entropy_a:.3f}, ent_w_eff={entropy_w_eff:.3f}) kl={kl:.5f}"
            )
            print(
                f" adv: mean={adv_1d.mean():.4f}, std={adv_1d.std():.4f}, "
                f"min={adv_1d.min():.4f}, max={adv_1d.max():.4f}"
            )
            print(
                f"[update {e+1}] ratio: mean={ratio_mean:.4f}, "
                f"min={ratio_min:.4f}, max={ratio_max:.4f}"
            )
            print(
                f"logits_delta: a={float(delta_a):.6f}, "
                f"w={float(delta_w):.6f}"
            )
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

        return entropy_last, actor_loss_last, value_loss_last

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
    def load_actor_only(self, check_path: str=None, device: str=None):
        device = device if device else self.device
        path = check_path if check_path else self.check_path
        data = torch.load(path, map_location=device)
        self.actor.load_state_dict(data['actor_state'])


# 训练模型的代理
class Agent:
    def __init__(self, action_dim, env: windowEnv, max_epochs: int=500, max_timesteps: int=30, print_interval: int=5, check_path: str='./miniQMT/DL/checkout'):
        self.env = env
        self.action_dim = action_dim  # <<< 修复作用域
        self.ppo = None

        if self.env:
            state_dim = self.env.get_state_dim()
            self.ppo = weightPPO(state_dim, action_dim)

        self.max_epochs = max_epochs
        self.max_timesteps = max_timesteps
        self.print_interval = print_interval
        self.check_path = check_path

    def set_env(self, env: windowEnv):
        self.env = env
        state_dim = self.env.get_state_dim()
        self.ppo = weightPPO(state_dim, self.action_dim)  # <<< 使用保存的 action_dim
    

    def export_to_excel(self, data, filename=f'C:/Users/Flying/Desktop/PPO_records.xlsx'):
        """
        将列表导出到 Excel 文件。
        
        参数:
        - data: 二维列表，第一行作为列标题（可选）。
        - filename: 输出 Excel 文件名，默认为 'output.xlsx'。
        
        示例:
        export_to_excel(data, 'example.xlsx')
        """
        if not data:
            print("数据为空，无法导出。")
            return
        e_data = [(i + 1) for i in range(len(data))]
        df = pd.DataFrame({
            'Epoch': e_data,
            'Rewards': data
        })
        # 导出到 Excel
        df.to_excel(filename, index=False, sheet_name='Sheet1')
        print(f"数据已成功导出到 {filename}")

    # 训练
    def train(self):
        best_reward = 0
        ek, alpha = 0.03, 0.997
        reward_list = []
        for epoch in range(self.max_epochs):
            if epoch:
                ek = ek * alpha
                if ek <= 0.01:
                    ek = 0.01
            print(f"start, ek = {ek}")

            state, _ = self.env.reset()
            print(f"[Train] Epoch = {epoch} | init_state = {state}")
            traces = {
                'states': [], 'actions': [],
                'weight_idx': [],       # 索引给 PPO
                'weights': [],          # 实际权重（可选）
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
                tolist = lambda x: x.tolist() if hasattr(x, "tolist") else x
                actions = tolist(a)
                w_idx_l = tolist(w_idx)
                w_val_l = tolist(w_val)
                logp_joint_l = tolist(logp_joint)

                # 环境吃实际权重值
                next_state, reward, terminated, truncated = self.env.step(actions, w_val_l)
                done = terminated or truncated

                traces['states'].append(state)
                traces['actions'].append(actions)
                traces['weight_idx'].append(w_idx_l)
                traces['weights'].append(w_val_l)
                traces['logp_joint'].append(logp_joint_l)   # <<< 只存联合
                traces['rewards'].append(reward)
                traces['dones'].append(done)
                traces['terminated'].append(terminated)
                traces['truncated'].append(truncated)

                state = next_state
                total_reward += reward
                if done:
                    break
            
            reward_list.append(total_reward)
            self.export_to_excel(reward_list)
            best_reward = max(best_reward, total_reward)
            traces['next_state'] = next_state

            value = self.env.account_controller.equity
            entropy, actor_loss, value_loss = self.ppo.update(traces, entropy_coef=ek)

            success_cnt = len(self.env.account_controller.Trades)
            order_cnt = len(self.env.account_controller.Orders)

            if (epoch + 1) % self.print_interval == 0:
                print(f"[Info: Train model] epoch: {epoch + 1} / {self.max_epochs} | "
                      f"Reward: {total_reward:.2f} | Market-Value: {value} | entropy-k = {ek} | "
                      f"entropy: {entropy.item():.4f}, actor_loss: {actor_loss.item():.6f}, value_loss: {value_loss.item():.6f}")
                rs = traces['rewards']

                mx, mi, me = max(rs), min(rs), sum(rs) / len(rs)

                not_zeros = sum([1 for item in rs if abs(item) > 1e-6])
                not_zero_ratio = not_zeros / len(rs)

                print(f"Reward: max = {mx:.6f}, min = {mi:.6f}, mean = {me:.6f}, last_reward = {rs[-1]}, success_cnt = {success_cnt}, order_cnt = {order_cnt}")
                print(f"[Reward-ratio] not_zeros = {not_zeros}, total = {len(rs)}, ratio = {not_zero_ratio}")
                a_f = self.env.account_controller.free_money
                single_f1 = self.env.account_controller.comb_info[0]['free_money']
                single_f2 = self.env.account_controller.comb_info[1]['free_money']
                single_e1 = self.env.account_controller.comb_info[0]['equity']
                single_e2 = self.env.account_controller.comb_info[1]['equity']

                print(f"a_f = {a_f} | single_f1 = {single_f1} | single_f2 = {single_f2}\n")
                print(f"a_e = {value} | single_e1 = {single_e1} | single_e2 = {single_e2}\n")

            self.ppo.save(epoch, best_reward)

            actions = traces['actions']
            actions = [k[0] for k in actions]
            unique, counts = np.unique(actions, return_counts=True)
            freq = {int(u): int(c) for u, c in zip(unique, counts)}
            maps = {
                0: 'HOLD',
                1: 'LONG',
                2: 'SHORT',
                3: 'CLOSE'
            }

            for k, v in freq.items():
                print(f"{maps[k]} -> {v} | ", end='')
            print("\n")
        self.env.close()

    # 测试
    def test(self, epochs: int = 5, alpha: float=0.5, test_mode: bool=False):
        for epoch in range(epochs):
            state, _ = self.env.reset()
            done = False
            total_reward = 0.0

            action_list = []
            weights = []

            while not done:
                a, w_idx, w_val, _, _, _ = self.ppo.selete_action_and_weight(state, test=test_mode)
    
                # action_list.append(a.item())
                # weights.append(w_val.item())

    
                actions = a.tolist() if hasattr(a, "tolist") else [int(a)]
                w_vals = w_val.tolist() if hasattr(w_val, "tolist") else [float(w_val)]
                next_state, reward, terminated, truncated = self.env.step(actions, w_vals)
                done = terminated or truncated

                # 加上最后的收益
                if done:
                    e = self.env.account_controller.equity
                    init_capital = self.env.account_controller.init_capital
                    reward += alpha * (e - init_capital) / init_capital

                state = next_state
                total_reward += reward

            unique, counts = np.unique(action_list, return_counts=True)
            freq = {int(u): int(c) for u, c in zip(unique, counts)}
            maps = {
                0: 'HOLD',
                1: 'LONG',
                2: 'SHORT',
                3: 'CLOSE'
            }
            for k, v in freq.items():
                print(f"{maps[k]} -> {v} | ", end='')
            print("\n")

            value = self.env.account_controller.equity
            print(f"[Info: Test model] epoch: {epoch + 1} | Reward: {total_reward} | Market-Value: {value}")
        self.env.close()

def save_norm(agent: Agent, filepath: str='./miniQMT/DL/checkout/norm.pkl'):
    data = {
        'reward_norm': agent.env.reward_norm,
        'state_norm': agent.env.state_norm
    }

    # 保存
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def read_norm(filepath: str='./miniQMT/DL/checkout/norm.pkl'):
    with open(filepath, 'rb') as f:
        loaded_data = pickle.load(f)
    
    return loaded_data['reward_norm'], loaded_data['state_norm']


if __name__ == '__main__':
    start_time = '20250624100000'
    end_time = '20250824150000'
    benchmark = '510050'
    
    state_dim = 10
    action_dim = 4
    max_epochs = 200
    max_timesteps = 350
    print_interval = 1

    start_time = '20250825100000'
    end_time = '20250924150000'

    option_pairs = []
    option_pairs.append({'call': '10008800','put': '10008809'})
    option_pairs.append({'call': '10008793','put': '10008802'})
    option_pairs.append({'call': '10008798','put': '10008807'})
    option_pairs.append({'call': '10008795','put': '10008804'})
    option_pairs.append({'call': '10008794','put': '10008803'})
    option_pairs.append({'call': '10008905','put': '10008906'})
    option_pairs.append({'call': '10009811','put': '10009812'})
    option_pairs.append({'call': '10009495','put': '10009496'})
    option_pairs.append({'call': '10009039','put': '10009040'})
    option_pairs.append({'call': '10008797','put': '10008806'})
    
    agent = Agent(action_dim, None, max_epochs, max_timesteps, print_interval)

    # # 训练
    calls = [option_pairs[0]['call'], option_pairs[1]['call']]
    puts = [option_pairs[0]['put'],  option_pairs[1]['put']]

    # calls = [option_pairs[0]['call']]
    # puts = [option_pairs[0]['put']]

    env = windowEnv(100000, calls, puts, max_timesteps, 1.3, start_time, end_time, benchmark)
    # agent.set_env(env)

    # print(f"[PPO-Agent] 开始训练......")
    # agent.train()
    # save_norm(agent)
    # print(f"[PPO-Agent] 结束训练......")


    # 测试
    calls = [option_pairs[2]['call'], option_pairs[2]['call']]
    puts = [option_pairs[3]['put'],  option_pairs[3]['put']]

    # calls = [calls[1]]
    # puts = [puts[1]]
    env = windowEnv(100000, calls, puts, max_timesteps, 1.3, start_time, end_time, benchmark, normalize_reward=False)
    reward_norm, state_norm = read_norm()
    env.set_norm(reward_norm, state_norm)
    agent.set_env(env)
    
    # 加载模型权重
    agent.ppo.load_actor_only()
    print(f"[PPO-Agent] 开始测试......")
    agent.test(test_mode=False)
