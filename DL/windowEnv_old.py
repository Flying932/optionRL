"""
    本模块实现一个环境模块, 这是PPO算法所需的env.
    本模块是加强版的, 时间窗口长度不等于1
    期权组合也不限为1组
"""

from abc import ABC, abstractmethod

from zmq import has
from finTool.windowAccount import windowAccount
import math
import numpy as np
import torch

# 奖励归一化
class RunningMeanStd:
    """动态计算mean 和 std"""
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)
    
    def update(self, x):
        x = np.array(x)
        if x.ndim == 0:
            x = x.reshape(1)
        
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = np.abs(x)
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S += (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(np.maximum(self.S / self.n, 1e-8))

# 对state进行标准化
class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x.tolist()

class RewardNormalization:
    """奖励归一化"""
    def __init__(self, shape=1):
        self.running_ms = RunningMeanStd(shape=shape)
    
    def __call__(self, x, update=True):
        if update:
            self.running_ms.update(x)
        
        normalized_x = (x - self.running_ms.mean) / (np.maximum(self.running_ms.std, 1e-8))
        return normalized_x[0]
    
    def get_states(self):
        return self.running_ms.mean, self.running_ms.std

class RewardScaling:
    def __init__(self, shape=1, gamma=0.99):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std

        return x[0]

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

# 环境模型基类
class baseEnv(ABC):
    @abstractmethod
    def reset(self) -> tuple:
        # 必须返回: state, info
        pass
    
    @abstractmethod
    def step(self, action) -> tuple:
        # 必须返回: next_state, reward, terminated, truncated
        pass

    @abstractmethod
    def close(self) -> None:
        pass

# 环境类
class windowEnv(baseEnv):
    def __init__(self, init_capital, calls, puts, timesteps: int=30, fee: float=1.3, start_time: str='20251021150000', end_time: str='20251022150000', benchmark: str='510050', normalize_reward: bool=True):
        self.start_time = start_time
        self.end_time = end_time
        self.benchmark = benchmark
        self.stockList = [benchmark]
        self.init_capital = init_capital
        self.fee = fee
        
        # 初始化账户对象
        self.account_controller = windowAccount(self.init_capital, self.fee, '30m', self.stockList)

        # 添加期权组合
        self.calls, self.puts = calls, puts
        self.add_comb(self.calls, self.puts)

        # 主图标的K线
        self.timesteps = timesteps
        self.run_data = self.account_controller.real_info_controller.get_bars_between_from_df(benchmark, start_time, end_time)
        self.total_length = len(self.run_data)
        self.row_index = 0
        ts = str(self.run_data.iloc[0]['ts']).replace(' ',  '').replace('-', '').replace(':', '')
        self.account_controller.init_state(ts, self.run_data.iloc[0]['close'])
        
        # 奖励归一化
        # self.reward_norm = RewardScaling() if normalize_reward else None
        self.reward_norm = RewardNormalization() if normalize_reward else None
        self.state_norm = None
    
    # 添加组合
    def add_comb(self, calls, puts):
        pairs = []
        for call, put in zip(calls, puts):
            pairs.append((call, put))

        self.account_controller.set_combos(pairs)
    
    # 设置归一化
    def set_norm(self, reward_norm, state_norm):
        self.reward_norm = reward_norm
        self.state_norm = state_norm

    # 返回state_dim
    def get_state_dim(self):
        state = self.account_controller.get_mix_total_state()
        return len(state[0])

    # 需要返回: (next_state, reward, terminated, truncated)
    def step(self, actions, weights, test: bool=False) -> tuple:
        if self.row_index >= self.total_length:
            return self.account_controller.get_total_state(), self.account_controller.getReward(), True, True
        
        ts, close = self.run_data.iloc[self.row_index]
        ts = str(ts)
        ivs = [' ', '-', ':']
        for item in ivs:
            ts = ts.replace(item, '')

        self.row_index += 1

        if self.row_index >= min(self.total_length, self.timesteps):
            # 导出excel
            # self.account_controller.out_excel()
            sharpe_like = math.log(self.account_controller.equity / self.account_controller.init_capital)
            dd = max(0, (self.account_controller.equity_peak - self.account_controller.equity) / self.account_controller.equity_peak)
            terminal = sharpe_like - 0.1 * dd  

            # reward = self.reward_norm(terminal)
            reward = terminal

            state = self.account_controller.get_mix_total_state()
            states = torch.as_tensor(state, dtype=torch.float32, device='cpu')
            if self.state_norm is None:
                self.state_norm = Normalization(states.norm)
            
            state = self.state_norm(states, update=not test)
            return state, reward, True, False

        state, reward, truncated = self.account_controller.step(actions, weights, ts, close)
        states = torch.as_tensor(state, dtype=torch.float32, device='cpu')
        
        if self.state_norm is None:
            self.state_norm = Normalization(states.shape)
        
        # reward = self.reward_norm(reward)
        return self.state_norm(states, update=not test), reward, False, truncated

    # 需要返回: (state, info)
    def reset(self) -> tuple:
        if hasattr(self, 'account_controller'):
            del self.account_controller
        if hasattr(self, 'reward_norm'):
            del self.reward_norm
        self.account_controller = windowAccount(self.init_capital, self.fee, '30m', self.stockList)
        self.reward_norm = RewardNormalization()

        # 添加期权组合
        self.add_comb(self.calls, self.puts)
        
        self.run_data = self.account_controller.real_info_controller.get_bars_between_from_df(self.benchmark, self.start_time, self.end_time)
        self.row_index = 0
        self.total_length = len(self.run_data)

        ts, close = self.run_data.iloc[0]
        ts = str(ts)
        ivs = [' ', '-', ':']

        for item in ivs:
            ts = ts.replace(item, '')


        self.account_controller.init_state(ts, close)

        state = self.account_controller.get_mix_total_state()
        info = self.account_controller.getInfo()

        return state, info
    
    def close(self):
        del self.account_controller
        del self.run_data

        

