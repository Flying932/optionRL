"""
    本模块实现PPO算法(支持多环境并行 & 批量更新)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import time, json

# ---- 你的环境 ----
from Environment import fintechEnv

# =========================
# 向量化环境：把多个 env 打包
# =========================
class VecEnvSimple:
    """
    简单的同步并行环境包装器。
    所有子环境接口: reset()->(state,info), step(a)->(next_state,reward,terminated,truncated)
    """
    def __init__(self, envs):
        self.envs = envs
        self.num_envs = len(envs)

    def reset(self):
        states, infos = [], []
        for e in self.envs:
            s, info = e.reset()
            states.append(s)
            infos.append(info)
        # (N, state_dim), list(info)
        return np.stack(states, axis=0), infos

    def step(self, actions):
        next_states, rewards, terms, truncs = [], [], [], []
        for e, a in zip(self.envs, actions):
            ns, r, te, tr = e.step(int(a))
            next_states.append(ns)
            rewards.append(r)
            terms.append(te)
            truncs.append(tr)
        return (
            np.stack(next_states, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terms, dtype=np.int32),
            np.asarray(truncs, dtype=np.int32),
        )

    def close(self):
        for e in self.envs:
            try:
                e.close()
            except Exception:
                pass

# =========================
# 策略/价值网络
# =========================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim: int=64):
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )
    def forward(self, state):
        # state: (B, state_dim) 或 (state_dim,)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.actor_net(state)   # (B, A)

class Value(nn.Module):
    def __init__(self, state_dim, hidden_dim: int=128):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.value_net(state).squeeze(-1)  # (B,)

# =========================
# PPO
# =========================
class PPO:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 actor_lr: float=3e-4,
                 value_lr: float=1e-3,
                 gamma: float=0.99,
                 clip_eps: float=0.2,
                 k_epochs: int=10,
                 device: str='cpu',
                 check_path: str='./miniQMT/DL/checkout/check_data.pt',
                 resume: bool=False):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.device = device
        self.resume = resume
        self.check_path = check_path

        self.actor = Actor(state_dim, action_dim).to(device)
        self.value = Value(state_dim).to(device)

        self.opt_a = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.opt_b = optim.Adam(self.value.parameters(), lr=value_lr)

    # 支持批量选择动作：state:(B,dim) 或 (dim,)
    def selete_action(self, state):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        probs = self.actor(s)                          # (B,A)
        dist = Categorical(probs=probs)
        actions = dist.sample()                        # (B,)
        logp = dist.log_prob(actions)                  # (B,)
        # 若是单样本输入，返回标量，否则返回数组
        if s.dim() == 1 or (s.dim()==2 and s.size(0)==1):
            return actions.item(), logp.squeeze(0)
        return actions.detach().cpu().numpy(), logp

    # (T,N) 版 GAE
    @torch.no_grad()
    def compute_gae_batched(self, rewards, values, next_values, terminateds, gamma=0.99, lam=0.95):
        """
        rewards:    (T, N)  torch float
        values:     (T, N)  torch float
        next_values:(N,)    torch float  # 最后一步的 bootstrap
        terminateds:(T, N)  torch float in {0,1}
        """
        T, N = rewards.shape
        adv = torch.zeros_like(rewards, device=rewards.device)
        last_gae = torch.zeros(N, device=rewards.device)
        for t in reversed(range(T)):
            v_t   = values[t]                           # (N,)
            v_tp1 = next_values if t == T-1 else values[t+1]
            mask  = 1.0 - terminateds[t]               # 终止则mask=0
            delta = rewards[t] + gamma * v_tp1 * mask - v_t
            last_gae = delta + gamma * lam * mask * last_gae
            adv[t] = last_gae
        returns = adv + values
        return adv, returns

    # PPO 更新：支持 (T,N) 采样，按 minibatch 打散
    def update(self, traj, minibatch_size: int=1024):
        """
        traj 字段：
          states:      (T, N, state_dim)
          actions:     (T, N)
          log_probs:   (T, N)
          rewards:     (T, N)
          terminated:  (T, N)
          next_states: (N, state_dim)   # rollout 最后一时刻的 next_state(每个env一个)
        """
        device = self.device
        states      = torch.as_tensor(traj['states'], dtype=torch.float32, device=device)      # (T,N,D)
        actions     = torch.as_tensor(traj['actions'], dtype=torch.long,   device=device)      # (T,N)
        old_logp    = torch.as_tensor(traj['log_probs'],dtype=torch.float32, device=device)    # (T,N)
        rewards     = torch.as_tensor(traj['rewards'],  dtype=torch.float32, device=device)    # (T,N)
        terminated  = torch.as_tensor(traj['terminated'],dtype=torch.float32, device=device)   # (T,N)
        next_states = torch.as_tensor(traj['next_states'],dtype=torch.float32, device=device)  # (N,D)

        T, N, D = states.shape
        # 计算 values 与 bootstrap
        with torch.no_grad():
            v = self.value(states.view(T*N, D)).view(T, N)          # (T,N)
            v_next = self.value(next_states)                         # (N,)
            adv, ret = self.compute_gae_batched(rewards, v, v_next, terminated,
                                                gamma=self.gamma, lam=0.95)
            # 标准化优势
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 扁平化
        states_f  = states.view(T*N, D)
        actions_f = actions.reshape(T*N)
        old_logp_f= old_logp.reshape(T*N).detach()
        adv_f     = adv.reshape(T*N).detach()
        ret_f     = ret.reshape(T*N).detach()

        # 迭代 K 个 epoch，随机小批更新
        total_size = T * N
        for _ in range(self.k_epochs):
            idx = torch.randperm(total_size, device=device)
            for start in range(0, total_size, minibatch_size):
                mb = idx[start:start+minibatch_size]
                s_mb = states_f[mb]
                a_mb = actions_f[mb]
                old_lp_mb = old_logp_f[mb]
                adv_mb = adv_f[mb]
                ret_mb = ret_f[mb]

                probs = self.actor(s_mb)
                dist  = Categorical(probs=probs)
                new_lp = dist.log_prob(a_mb)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()

                values = self.value(s_mb)
                value_loss = F.mse_loss(values, ret_mb)

                loss = actor_loss + 0.5*value_loss - 0.01*entropy

                self.opt_a.zero_grad()
                self.opt_b.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters())+list(self.value.parameters()), 1.0)
                self.opt_a.step()
                self.opt_b.step()

        return actor_loss.detach().item(), value_loss.detach().item(), entropy.detach().item()

    # 保存/加载
    def save(self, epoch: int=None, best_reward: float=None):
        data = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "actor_state": self.actor.state_dict(),
            "value_state": self.value.state_dict(),
            "opt_a_state": self.opt_a.state_dict(),
            "opt_b_state": self.opt_b.state_dict(),
            "h_params": {
                "gamma": self.gamma, "clip_eps": self.clip_eps,
                "k_epochs": self.k_epochs, "device": self.device
            },
            "epoch": epoch, "best_reward": best_reward
        }
        torch.save(data, self.check_path)

    def load_actor_only(self, check_path: str=None, device: str=None):
        device = device if device else self.device
        path = check_path if check_path else self.check_path
        data = torch.load(path, map_location=device)
        self.actor.load_state_dict(data['actor_state'])

# =========================
# 训练代理（并行环境）
# =========================
class Agent:
    def __init__(self,
                 state_dim, action_dim,
                 vec_env: VecEnvSimple,
                 max_epochs: int=500,
                 rollout_len: int=128,          # 每轮收集的时间步
                 minibatch_size: int=1024,      # PPO 小批量
                 print_interval: int=5,
                 check_path: str='./miniQMT/DL/checkout',
                 device: str='cpu'):
        self.ppo = PPO(state_dim, action_dim, device=device)
        self.env = vec_env
        self.max_epochs = max_epochs
        self.rollout_len = rollout_len
        self.minibatch_size = minibatch_size
        self.print_interval = print_interval
        self.check_path = check_path
        self.device = device

    def _equity_stats_now(self):
        vals = np.array([e.account_controller.equity for e in self.env.envs], dtype=np.float64)
        return vals.mean(), vals.std(), vals.min(), vals.max()
    
    def train(self):
        best_reward = -1e9
        N = self.env.num_envs
        for epoch in range(self.max_epochs):
            states, _ = self.env.reset()

            buf_states, buf_actions, buf_log_probs = [], [], []
            buf_rewards, buf_terminated = [], []

            ep_return_sum = 0.0
            equity_time_sum = 0.0          # NEW: 用于时间平均
            for t in range(self.rollout_len):
                with torch.no_grad():
                    actions, logp = self.ppo.selete_action(states)
                    if isinstance(actions, (int, np.integer)):
                        actions = np.array([actions]); logp = logp.unsqueeze(0)

                next_states, rewards, terms, truncs = self.env.step(actions)
                dones = (terms | truncs).astype(np.int32)

                buf_states.append(states.copy())
                buf_actions.append(actions.copy())
                buf_log_probs.append(logp.detach().cpu().numpy())
                buf_rewards.append(rewards.copy())
                buf_terminated.append(terms.copy())

                ep_return_sum += rewards.mean()

                # NEW: 累计当下“跨环境平均市值”
                eq_mean, _, _, _ = self._equity_stats_now()
                equity_time_sum += eq_mean

                if dones.any():
                    for i, done in enumerate(dones):
                        if done:
                            s, _ = self.env.envs[i].reset()
                            next_states[i] = s
                states = next_states

            next_states_bootstrap = states.copy()
            traj = {
                'states': np.stack(buf_states, axis=0),
                'actions': np.stack(buf_actions, axis=0),
                'log_probs': np.stack(buf_log_probs, axis=0),
                'rewards': np.stack(buf_rewards, axis=0),
                'terminated': np.stack(buf_terminated, axis=0),
                'next_states': next_states_bootstrap,
            }

            actor_l, value_l, entropy = self.ppo.update(traj, minibatch_size=self.minibatch_size)

            # NEW: 统计两个你关心的“市值指标”
            # 1) 时间平均(本轮 rollout 的时间平均、跨环境均值)
            equity_time_avg = equity_time_sum / self.rollout_len
            # 2) 快照均值(rollout 结束瞬间跨环境均值)
            equity_snap_mean, equity_snap_std, _, _ = self._equity_stats_now()

            if (epoch + 1) % self.print_interval == 0:
                avg_ret = ep_return_sum / self.rollout_len
                print(f"[Train] epoch {epoch+1}/{self.max_epochs} | "
                    f"avg_step_return: {avg_ret:.4f} | "
                    f"equity_time_avg: {equity_time_avg:.2f} | "
                    f"equity_snap_mean±std: {equity_snap_mean:.2f}±{equity_snap_std:.2f} | "
                    f"actor: {actor_l:.4f} | value: {value_l:.4f} | entropy: {entropy:.4f}")

            best_reward = max(best_reward, ep_return_sum)
            self.ppo.save(epoch, best_reward)
        self.env.close()

    def test(self, episodes: int=10):
        N = self.env.num_envs
        for ep in range(episodes):
            states, _ = self.env.reset()
            done = np.zeros(N, dtype=bool)
            total_r = np.zeros(N, dtype=np.float32)
            while not done.all():
                with torch.no_grad():
                    actions, _ = self.ppo.selete_action(states)
                    if isinstance(actions, (int, np.integer)):
                        actions = np.array([actions])
                next_states, rewards, terms, truncs = self.env.step(actions)
                done = (terms | truncs) | done
                states = next_states
                total_r += rewards
            # 结束时的跨环境市值统计
            mean_eq, std_eq, min_eq, max_eq = self._equity_stats_now()
            print(f"[Test] ep {ep+1} | avg_return: {total_r.mean():.4f} | "
                f"final_equity_mean±std: {mean_eq:.2f}±{std_eq:.2f} "
                f"(min:{min_eq:.2f}, max:{max_eq:.2f})")
        self.env.close()



# =========================
# 用法示例（把多对合约并行训练）
# =========================
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_time = '20250825100000'
    end_time   = '20250924150000'
    benchmark  = '510050'
    action_dim = 6

    option_pairs = [
        {'call':'10008800','put':'10008809'},
        {'call':'10008793','put':'10008802'},
        {'call':'10008798','put':'10008807'},
        {'call':'10008795','put':'10008804'},
        {'call':'10008794','put':'10008803'},
        {'call':'10008905','put':'10008906'},
        {'call':'10009811','put':'10009812'},
        {'call':'10009495','put':'10009496'},
        {'call':'10009039','put':'10009040'},
        {'call':'10008797','put':'10008806'},
    ]

    # === 构造并行环境 ===
    num_envs = 4                      # 你可以自由指定并行个数
    max_timesteps = 350               # 环境内部会用到
    init_capital = 100000
    fee = 1.3

    envs = []
    for i in range(num_envs):
        pair = option_pairs[i % len(option_pairs)]
        envs.append(fintechEnv(init_capital, max_timesteps, fee,
                               start_time, end_time, benchmark,
                               pair['call'], pair['put']))
    vec_env = VecEnvSimple(envs)

    # === 估计 state_dim（从一个 env.reset() 拿）===
    tmp_state, _ = envs[0].reset()
    state_dim = len(tmp_state)

    agent = Agent(state_dim, action_dim, vec_env,
                  max_epochs=400,
                  rollout_len=128,           # 每轮收集 128 步 × num_envs
                  minibatch_size=512,       # 你的 batch_size
                  print_interval=10,
                  device=device)

    print("[PPO-Agent] 并行训练开始……")
    agent.train()
    print("[PPO-Agent] 并行测试……")

    # 测试时也可以换一批环境
    test_envs = []
    for i in range(num_envs):
        pair = option_pairs[(i+5) % len(option_pairs)]
        test_envs.append(fintechEnv(init_capital, max_timesteps, fee,
                                    start_time, end_time, benchmark,
                                    pair['call'], pair['put']))
    agent.env = VecEnvSimple(test_envs)
    agent.test(episodes=5)
