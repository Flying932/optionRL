"""
    本模块实现PPO算法的部分, 这也是整个架构最核心的部分.
    需要对接我们的模拟miniQMT环境
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from Environment import fintechEnv
import time, json
import sys

# 输出类, 实现重定向sys.stdout的功能
class outPut():
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()
    
    def close(self):
        # self.terminal.close()
        self.logfile.close()

file_path = f'C:/Users/Flying/Desktop/PPO_records.txt'
sys.stdout = outPut(file_path)


# 策略学习网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim: int=64):
        super(Actor, self).__init__()

        # 状态 -> 动作
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.actor_net(state)

# 价值学习网络
class Value(nn.Module):
    def __init__(self, state_dim, hidden_dim: int=128):
        super(Value, self).__init__()
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.value_net(state)

# PPO-agent
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
                 resume: bool=False
                 ):
        
        self.device = 'cuda' if torch.cuda.is_available() else device
        # self.device = device

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs

        # 继续之前
        self.resume = resume

        self.check_path = check_path

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.value = Value(state_dim).to(self.device)

        self.opt_a = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.opt_b = optim.Adam(self.value.parameters(), lr=value_lr)
    
    # 根据当前状态->选择动作
    def selete_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        probs = self.actor(state)

        dist = Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action)

    # 计算一个轨迹的累积回报, next_value是最后一步的
    def compute_returns(self, rewards, dones, next_value):
        returns = []
        R = next_value

        for r, done in zip(reversed(rewards), reversed(dones)):
            R =  r + self.gamma * R * (1 - done)
            returns.insert(0, R)
            
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    # 计算GAE, 广义优势估计
    def compute_gae(self, rewards, values, next_value, terminateds, gamma: float=0.99, lam: float=0.95):
        """
        Args:
            rewards: (T,)  list/np/torch
            values:  (T,)  torch
            next_value: () torch
            terminateds: (T,) 0/1, 只有环境真正结束才是1; 时间截断是0
        """
        T = len(rewards)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        terms = torch.as_tensor(terminateds, dtype=torch.float32, device=self.device)
        adv = torch.zeros(T, dtype=torch.float32, device=self.device)

        last_gae = 0
        for t in reversed(range(T)):
            v_t = values[t]
            v_tp1 = next_value if t == T - 1 else values[t + 1]
            mask = 1 - terms[t]

            delta = rewards[t] + gamma * v_tp1 * mask - v_t
            last_gae = delta + gamma * lam * mask * last_gae
            adv[t] = last_gae
        
        # A(s, a) = Q(s, a) - V(s)
        returns = adv + values

        return adv, returns

    # 更新参数, 每一条轨迹更新一次
    def update(self, traces):
        states = torch.FloatTensor(traces['states']).to(self.device)
        actions = torch.LongTensor(traces['actions']).to(self.device)
        old_log_probs = torch.stack(traces['log_probs']).to(self.device).detach()
        rewards = traces['rewards']

        dones = traces['dones']
        terminateds = traces['terminated']

        # 最后一步的下一个状态, 这个是用来估计价值函数的(如果被截断而不是结束的话)
        next_state = torch.FloatTensor(traces['next_state']).to(self.device)

        with torch.no_grad():
            # 估计最后的价值函数(若被截断则需要用上)
            next_value = self.value(next_state)

            values = self.value(states).squeeze(-1)
            
            # GAE计算回报
            advantages, returns = self.compute_gae(
                rewards,
                values, 
                next_value,
                terminateds,
                self.gamma,
                lam=0.95
            )

            # 归一化
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 计算累积回报
            # returns = self.compute_returns(rewards, dones, next_value)

            # 计算每一步的状态的价值函数
            # action_values = self.value(states)

            # 计算优势函数, squeeze()去掉维度为1的维度
            # returns是一个一维向量, action_values也需要对齐
            # advantages = returns - action_values.squeeze()
        
        for _ in range(self.k_epochs):
            probs = self.actor(states)
            values = self.value(states)

            dist = Categorical(probs)
            entropy = dist.entropy().mean()

            # 当前参数更新后, 同样动作的对数概率
            new_log_probs = dist.log_prob(actions)

            # 新旧策略比值
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            # 策略网络损失函数
            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值网络的损失函数
            value_loss = F.mse_loss(values.squeeze(), returns)

            # 总损失函数
            loss = actor_loss + 0.5 * value_loss - 0.01 * entropy
            k1 = 0.5 * value_loss / actor_loss
            k2 = 0.01 * entropy / actor_loss

            print(f"[PPO.py] actor_loss = {actor_loss}, value_loss = {value_loss}, k1 = {k1}, k2 = {k2}, entropy = {entropy}")

            # 梯度更新
            self.opt_a.zero_grad()
            self.opt_b.zero_grad()
            loss.backward()

            self.opt_a.step()
            self.opt_b.step()
        
        return entropy, actor_loss, value_loss

    # 保存权重
    def save(self, epoch: int=None, best_reward: int=None):
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
                "device": self.device
            },
            "epoch": epoch,
            "best_reward": best_reward
        }

        torch.save(data, self.check_path)
    
    # 仅用于推理
    def load_actor_only(self, check_path: str=None, device: str=None):
        device = device if device else self.device
        path = check_path if check_path else self.check_path
        data = torch.load(path, map_location=device)
        self.actor.load_state_dict(data['actor_state'])

# 训练模型的代理, 需要一端对接PPO, 一端对接env
class Agent:
    def __init__(self, state_dim, action_dim, env: fintechEnv, max_epochs: int=500, max_timesteps: int=30, print_interval: int=5, check_path: str='./miniQMT/DL/checkout'):
        self.ppo = PPO(state_dim, action_dim)
        
        # 需要一个交互的环境env
        self.env = env

        self.max_epochs = max_epochs
        self.max_timesteps = max_timesteps
        self.print_interval = print_interval

        # 保存权重
        self.check_path = check_path
    
    # 训练模型
    def train(self):
        best_reward = 0
        for epoch in range(self.max_epochs):
            state, _ = self.env.reset()
            traces = {
                'states': [],
                'actions': [],
                'log_probs': [],
                'rewards': [],
                'dones': [],
                'next_state': None,
                'terminated': [],
                'truncated': []
            }

            total_reward = 0
            for t in range(self.max_timesteps):
                # 随机选择一个操作
                action, log_prob = self.ppo.selete_action(state)

                # 环境交互, 计算下一个状态的信息
                next_state, reward, terminated, truncated = self.env.step(action)

                done = terminated or truncated

                traces['states'].append(state)
                traces['actions'].append(action)
                traces['log_probs'].append(log_prob)
                traces['rewards'].append(reward)
                traces['dones'].append(done)
                traces['terminated'].append(terminated)
                traces['truncated'].append(truncated)

                state = next_state
                total_reward += reward
                
                # 若结束则跳出本次探索
                if done:
                    break
            best_reward = max(best_reward, total_reward)
            # 记录下最后的状态, 若被截断, 计算价值函数需要用到
            traces['next_state'] = next_state

            # 最终市值
            value = self.env.account_controller.equity

            success_cnt = len(self.env.account_controller.Trades)
            order_cnt = len(self.env.account_controller.Orders)

            # 更新参数
            entropy, actor_loss, value_loss = self.ppo.update(traces)
            if (epoch + 1) % self.print_interval == 0:
                print(f"[Info: Train model] epoch: {epoch + 1} / {self.max_epochs} | Reward: {total_reward:.2f} | Value: {value} | entropy: {entropy}, actor_loss: {actor_loss}, value_loss: {value_loss}")
                print(f"Success_cnt = {success_cnt}, Order_cnt = {order_cnt}")
                rewards = traces['rewards']
                mx, mi, me = max(rewards), min(rewards), sum(rewards) / len(rewards)
                print(f"max = {mx}, min = {mi}, mean = {me}")
            self.ppo.save(epoch, best_reward)
        self.env.close()

    # 测试模型
    def test(self, epochs: int=30):
        for epoch in range(epochs):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _ = self.ppo.selete_action(state)
                next_state, reward, terminated, truncated = self.env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward

            value = self.env.account_controller.equity
            print(f"[Info: Test model] epoch: {epoch + 1} | Reward: {total_reward} | Value: {value}")
        
        self.env.close()

if __name__ == '__main__':
    start_time = '20250624100000'
    end_time = '20250824150000'
    benchmark = '510050'
    
    state_dim = 10
    action_dim = 6
    max_epochs = 400
    max_timesteps = 350
    print_interval = 20
    print_interval = 1

    start_time = '20250825100000'
    end_time = '20250924150000'

    option_pairs = []
    option_pairs.append(
        {
            'call': '10008800',
            'put': '10008809'
        }
    )

    option_pairs.append(
        {
            'call': '10008793',
            'put': '10008802'
        }
    )

    option_pairs.append(
        {
            'call': '10008798',
            'put': '10008807'
        }
    )

    option_pairs.append(
        {
            'call': '10008795',
            'put': '10008804'
        }
    )

    option_pairs.append(
        {
            'call': '10008794',
            'put': '10008803'
        }
    )

    option_pairs.append(
        {
            'call': '10008905',
            'put': '10008906'
        }
    )

    option_pairs.append(
        {
            'call': '10009811',
            'put': '10009812'
        }
    )

    option_pairs.append(
        {
            'call': '10009495',
            'put': '10009496'
        }
    )

    option_pairs.append(
        {
            'call': '10009039',
            'put': '10009040'
        }
    )

    option_pairs.append(
        {
            'call': '10008797',
            'put': '10008806'
        }
    )

    # 训练
    agent = Agent(state_dim, action_dim, None, max_epochs, max_timesteps, print_interval)
    length = len(option_pairs)
    idx = int(length * 0.7)

    print(f"[PPO-Agent] 开始训练......")
    for i in range(idx):
        print(f"idx = {i}")
        call = option_pairs[i]['call']
        put = option_pairs[i]['put']
        env = fintechEnv(100000, max_timesteps, 1.3, start_time, end_time, benchmark, call, put)
        agent.env = env
        agent.train()
    print(f"[PPO-Agent] 结束训练......")
    agent.ppo.load_actor_only()

    print(f"[PPO-Agent] 开始测试......")
    for i in range(idx + 1, length):
        call = option_pairs[i]['call']
        put = option_pairs[i]['put']

        print(f"call = {call}, put = {put}")

        new_env = fintechEnv(100000, max_timesteps, 1.3, start_time, end_time, benchmark, call, put)
        agent.env = new_env
        agent.test(5)


    print(0 / 0)
    