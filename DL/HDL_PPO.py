"""
两层PPO（HRL）最小可跑版
高层：每 macro_interval 步选择体制 r ∈ {FLAT, LONG_VOL, SHORT_VOL}
低层：在体制掩码约束下，从 6 动作中选择执行
"""

import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# 你的环境
from Environment import fintechEnv

# -------- 动作与体制枚举 --------
HOLD, OPEN_LONG_1x, OPEN_LONG_2x, OPEN_SHORT_1x, OPEN_SHORT_2x, CLOSE_ALL = range(6)
FLAT, LONG_VOL, SHORT_VOL = range(3)

# 根据当前的体制构造动作掩码
def build_action_mask(regime: int):
    """
    Returns:
        长度为6的bool列表, True表示允许
    """
    allow = [False]*6
    # 永远允许的：HOLD, CLOSE_ALL
    allow[HOLD] = True
    allow[CLOSE_ALL] = True
    if regime == LONG_VOL:
        allow[OPEN_LONG_1x] = True
        allow[OPEN_LONG_2x] = True
    elif regime == SHORT_VOL:
        allow[OPEN_SHORT_1x] = True
        allow[OPEN_SHORT_2x] = True
    else:  # FLAT
        pass
    return allow

# -------- 通用的 Actor/Value（离散分类，输出logits） --------
class DiscreteActor(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)  # logits
        )
    def forward(self, s):
        if s.dim()==1: s = s.unsqueeze(0)
        return self.net(s)  # (B, n_actions)

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, s):
        if s.dim()==1: s = s.unsqueeze(0)
        return self.net(s).squeeze(-1)  # (B,)

# -------- 一个可复用的 PPO 头（适配“掩码动作”） --------
class PPOHead:
    def __init__(self, state_dim, n_actions, actor_lr=3e-4, value_lr=1e-3,
                 gamma=0.99, lam=0.95, clip_eps=0.2, k_epochs=10, device='cpu', ent_coef=0.02):
        self.device = device
        self.gamma, self.lam, self.clip_eps, self.k_epochs = gamma, lam, clip_eps, k_epochs
        self.actor = DiscreteActor(state_dim, n_actions).to(device)
        self.value = ValueNet(state_dim).to(device)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.opt_v = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        self.ent_coef = ent_coef

    @torch.no_grad()
    def select_action(self, state_np, allow_mask=None):
        """state_np: (state_dim,), allow_mask: None or bool[ n_actions ]"""
        s = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
        logits = self.actor(s)  # (1,nA)
        if allow_mask is not None:
            mask = torch.tensor(allow_mask, dtype=torch.bool, device=self.device).unsqueeze(0)  # (1,nA)
            # 禁止的动作 logits 置为 -1e9
            logits = logits.masked_fill(~mask, -1e9)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), logp.detach()

# --- 在 PPOHead 中，替换 _gae / update ---

    @torch.no_grad()
    def _gae(self, rewards, values, last_value, dones, gammas=None):
        """
        rewards: (T,)
        values:  (T,)
        last_value: ()
        dones:   (T,) 1/0
        gammas:  (T,) 每一步的折扣，如果 None 则用常数 self.gamma
        """
        T = rewards.shape[0]
        adv = torch.zeros(T, device=self.device)
        last_gae = torch.zeros((), device=self.device)
        for t in reversed(range(T)):
            g = (self.gamma if gammas is None else gammas[t])
            v  = values[t]
            v1 = last_value if t == T-1 else values[t+1]
            mask = 1.0 - dones[t]
            delta = rewards[t] + g * v1 * mask - v
            last_gae = delta + g * self.lam * mask * last_gae
            adv[t] = last_gae
        ret = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def update(self, traj):
        device = self.device
        states = torch.as_tensor(traj['states'], dtype=torch.float32, device=device)
        actions = torch.as_tensor(traj['actions'], dtype=torch.long, device=device)
        logp_old = torch.as_tensor(traj['logp_old'], dtype=torch.float32, device=device)
        rewards  = torch.as_tensor(traj['rewards'], dtype=torch.float32, device=device)
        dones    = torch.as_tensor(traj['dones'], dtype=torch.float32, device=device)
        next_state = torch.as_tensor(traj['next_state'], dtype=torch.float32, device=device)
        masks = torch.as_tensor(traj.get('masks', None), dtype=torch.bool, device=device) if 'masks' in traj else None
        gammas = torch.as_tensor(traj.get('gammas', None), dtype=torch.float32, device=device) if 'gammas' in traj else None

        with torch.no_grad():
            v = self.value(states)
            v_next = self.value(next_state)
            adv, ret = self._gae(rewards, v, v_next, dones, gammas=gammas)

        for _ in range(self.k_epochs):
            logits = self.actor(states)
            if masks is not None:
                logits = logits.masked_fill(~masks, -1e9)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - logp_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            v_pred = self.value(states)
            value_loss = F.mse_loss(v_pred, ret)

            loss = actor_loss + 0.5 * value_loss - self.ent_coef * entropy
            self.opt_a.zero_grad(); self.opt_v.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters())+list(self.value.parameters()), 1.0)
            self.opt_a.step(); self.opt_v.step()
        return actor_loss.item(), value_loss.item(), entropy.item()


# -------- HRL Agent --------
class HRLAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim_low: int = 6,
                 gamma: float = 0.99,
                 macro_interval: int = 8,      # 高层每多少步决策一次（8*30min≈4小时）
                 flip_penalty: float = 0.01,   # 体制切换惩罚
                 device: str = 'cpu'):
        
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = device

        self.gamma = gamma
        self.macro_interval = macro_interval
        self.flip_penalty = flip_penalty

        # 高层：3 类体制
        self.hi_entropy_coef = 0.02
        self.hi = PPOHead(state_dim, n_actions=3, gamma=gamma ** self.macro_interval , device=device, ent_coef=self.hi_entropy_coef)
        

        # 低层：6 个原子动作
        self.lo_entropy_coef = 0.05
        self.lo = PPOHead(state_dim, n_actions=action_dim_low, gamma=gamma, device=device, ent_coef=self.lo_entropy_coef)

    def train(self, env: fintechEnv, max_epochs=200, max_timesteps=350, print_interval=10):
        """
        关键改动：
        - 段内奖励 seg_reward = ∑ step_reward （不提前折扣）
        - 无交易整段惩罚 / 轻微活跃奖励
        - 高层温度采样与熵系数退火
        """
        device = self.device

        # 超参：可以先用下面默认值，跑通后再微调
        beta_no_trade   = 1e-3   # 一整段无成交惩罚（非常小，避免“罚躺平”收敛）
        beta_has_trade  = 5e-4   # 有成交的小激励（比惩罚略小，防止过度交易）
        switch_penalty  = 1e-3   # 体制切换罚
        hi_temp_start   = 1.5    # 高层温度采样系数起点（>1 更随机）
        hi_temp_end     = 1.0    # 终点
        hi_ent_start    = 0.10   # 高层熵系数起点（更鼓励探索）
        hi_ent_end      = 0.02   # 终点
        warmup_epochs   = 20     # 前若干 epoch 强化探索

        # 把熵系数设置为起点
        self.hi.ent_coef = hi_ent_start
        self.lo.ent_coef = self.lo.ent_coef  # 低层保持你当前设置

        best = -1e9
        for ep in range(1, max_epochs+1):
            # 线性退火（也可换成余弦）
            t_frac = min(1.0, max(0.0, (ep-1) / max(1, max_epochs-1)))
            hi_temp = hi_temp_start + (hi_temp_end - hi_temp_start) * t_frac
            self.hi.ent_coef = hi_ent_start + (hi_ent_end - hi_ent_start) * t_frac

            state, _ = env.reset()
            total_ret = 0.0
            done = False

            # ------- 低层轨迹缓存 -------
            L_states, L_actions, L_logp, L_rewards, L_dones, L_masks = [], [], [], [], [], []

            # ------- 高层轨迹缓存（每段记一条） -------
            H_states, H_actions, H_logp, H_rewards, H_dones = [], [], [], [], []

            # 当前段统计
            current_regime = None
            last_regime = None
            seg_reward_sum = 0.0
            trades_at_seg_start = len(getattr(env.account_controller, "Trades", []))
            seg_steps = 0

            # 高层采样函数（带温度/epsilon）
            def sample_hi_action(s_np, temperature=1.0):
                s = torch.as_tensor(s_np, dtype=torch.float32, device=device)
                logits = self.hi.actor(s)  # (1,3)
                # 温度缩放
                logits = logits / max(1e-6, temperature)
                dist = Categorical(logits=logits)
                a = dist.sample()
                return int(a.item()), dist.log_prob(a).detach()

            t = 0
            while t < max_timesteps and not done:
                # ---- 是否高层重新决策 ----
                if (current_regime is None) or (t % self.macro_interval == 0):
                    # 把上一段的累计结果记给高层
                    if current_regime is not None:
                        trades_now = len(getattr(env.account_controller, "Trades", []))
                        seg_trades = max(0, trades_now - trades_at_seg_start)
                        # 活跃/摆烂 shaping（很小的量级）
                        activity_bonus = (beta_has_trade if seg_trades > 0 else -beta_no_trade)
                        # 体制切换罚（非常小）
                        flip = (1.0 if (last_regime is not None and current_regime != last_regime) else 0.0)
                        H_rewards.append(seg_reward_sum + activity_bonus - switch_penalty * flip)
                        H_dones.append(0.0)  # 段内未终止
                        # 重置段统计
                        seg_reward_sum = 0.0
                        seg_steps = 0
                        trades_at_seg_start = trades_now

                    # 采样新体制
                    if ep <= warmup_epochs:
                        a_hi = np.random.choice([FLAT, LONG_VOL, SHORT_VOL])
                        logp_hi = torch.tensor(0.0)
                    else:
                        a_hi, logp_hi = sample_hi_action(state, temperature=hi_temp)

                    last_regime = current_regime
                    current_regime = a_hi

                    # 记高层轨迹（状态=段首状态）
                    H_states.append(np.asarray(state, dtype=np.float32))
                    H_actions.append(a_hi)
                    H_logp.append(float(logp_hi.item()))

                # ---- 低层在体制掩码下行动 ----
                allow = build_action_mask(current_regime)
                a_lo, logp_lo = self.lo.select_action(state, allow_mask=allow)

                # 环境一步
                next_state, reward, terminated, truncated = env.step(a_lo)
                done = bool(terminated or truncated)
                total_ret += reward

                # 存低层轨迹
                L_states.append(np.asarray(state, dtype=np.float32))
                L_actions.append(a_lo)
                L_logp.append(float(logp_lo.item()))
                L_rewards.append(float(reward))
                L_dones.append(1.0 if done else 0.0)
                L_masks.append(np.asarray(allow, dtype=bool))

                # 段内累计（不折扣）
                seg_reward_sum += float(reward)
                seg_steps += 1

                state = next_state
                t += 1

            # episode 结束：把最后一段也记上
            if current_regime is not None:
                trades_now = len(getattr(env.account_controller, "Trades", []))
                seg_trades = max(0, trades_now - trades_at_seg_start)
                activity_bonus = (beta_has_trade if seg_trades > 0 else -beta_no_trade)
                flip = 0.0  # 最后一段不算切换
                H_rewards.append(seg_reward_sum + activity_bonus - switch_penalty * flip)
                H_dones.append(1.0 if done else 0.0)

            # ------- 更新低层 -------
            L_traj = {
                'states': np.stack(L_states) if len(L_states)>0 else np.empty((0, len(state)), dtype=np.float32),
                'actions': np.array(L_actions, dtype=np.int64),
                'logp_old': np.array(L_logp, dtype=np.float32),
                'rewards': np.array(L_rewards, dtype=np.float32),
                'dones':   np.array(L_dones, dtype=np.float32),
                'next_state': np.asarray(state, dtype=np.float32),
                'masks': np.stack(L_masks) if len(L_masks)>0 else np.empty((0,6), dtype=bool),
            }
            aL, vL, entL = self.lo.update(L_traj) if len(L_states)>0 else (0.0,0.0,0.0)

            # ------- 更新高层 -------
            H_traj = {
                'states': np.stack(H_states) if len(H_states)>0 else np.empty((0, len(state)), dtype=np.float32),
                'actions': np.array(H_actions, dtype=np.int64) if len(H_actions)>0 else np.empty((0,), dtype=np.int64),
                'logp_old': np.array(H_logp, dtype=np.float32) if len(H_logp)>0 else np.empty((0,), dtype=np.float32),
                'rewards': np.array(H_rewards, dtype=np.float32) if len(H_rewards)>0 else np.empty((0,), dtype=np.float32),
                'dones':   np.array(H_dones, dtype=np.float32) if len(H_dones)>0 else np.empty((0,), dtype=np.float32),
                'next_state': np.asarray(state, dtype=np.float32),
            }
            aH, vH, entH = self.hi.update(H_traj) if len(H_states)>0 else (0.0,0.0,0.0)

            actions = L_traj['actions']
            if ep % print_interval == 0:
                equality = env.account_controller.equity
                len_orders = len(env.account_controller.Orders)
                print(f"[HRL] ep {ep}/{max_epochs} | "
                    f"return:{total_ret:.2f} | "
                    f"LO(actor:{aL:.3f}, value:{vL:.3f}, ent:{entL:.3f}) | market_value = {equality} | len_orders = {len_orders}")
                
                print(f"actions = {actions}")



    def test(self, env: fintechEnv, episodes=5):
        for ep in range(1, episodes+1):
            state, _ = env.reset()
            done = False
            t = 0
            total = 0.0
            current_regime = None
            
            actions = []
            while not done and t < 1000:
                if current_regime is None or t % self.macro_interval == 0:
                    current_regime, _ = self.hi.select_action(state)
                allow = build_action_mask(current_regime)
                a, _ = self.lo.select_action(state, allow_mask=allow)
                actions.append(a)
                state, r, term, trunc = env.step(a)
                total += r
                t += 1
                done = bool(term or trunc)
            
            len_orders = len(env.account_controller.Orders)
            market_value = env.account_controller.equity

            print(f"[TEST] ep {ep} | return {total:.2f}, market_value = {market_value}, len_orders = {len_orders}")
            print(f"actions = {actions}")
        env.close()

# -------- 直接跑一个例子 --------
if __name__ == "__main__":
    # 你的数据区间/合约按需替换
    start_time = '20250825100000'
    end_time   = '20250924150000'
    benchmark  = '510050'
    call_code  = '10008800'
    put_code   = '10008809'
    init_capital = 100000
    max_timesteps = 350
    fee = 1.3

    # 构建环境
    env = fintechEnv(init_capital, max_timesteps, fee,
                     start_time, end_time, benchmark, call_code, put_code)

    # 估计 state_dim
    s0, _ = env.reset()
    state_dim = len(s0)

    agent = HRLAgent(state_dim=state_dim,
                     gamma=0.99,
                     macro_interval=16,     # 可调：体制承诺长度
                     flip_penalty=0.01,
                     device='cuda' if torch.cuda.is_available() else 'cpu')

    print("[HRL] 训练开始...")
    agent.train(env, max_epochs=200, max_timesteps=max_timesteps, print_interval=10)

    # 测试（可换一对call/put）
    test_env = fintechEnv(init_capital, max_timesteps, fee,
                          start_time, end_time, benchmark, call_code, put_code)
    print("[HRL] 测试开始...")
    agent.test(test_env, episodes=5)
