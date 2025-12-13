"""
    本模块实现一个环境模块, 这是PPO算法所需的env.
    本模块是加强版的, 时间窗口长度不等于1
    期权组合也不限为1组
"""

from abc import ABC, abstractmethod

from finTool.single_window_account import single_Account
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
    def __init__(self, init_capital, call, put, timesteps: int=30, fee: float=1.3, start_time: str='20251021150000', end_time: str='20251022150000', benchmark: str='510050', device: str='cpu'):
        self.start_time = start_time
        self.end_time = end_time
        self.benchmark = benchmark
        self.stockList = [benchmark]
        self.init_capital = init_capital
        self.fee = fee

        self.device = device
        
        # 初始化账户对象
        # self.account_controller = windowAccount(self.init_capital, self.fee, '30m', self.stockList)
        self.account_controller = single_Account(self.init_capital, self.fee, '30m', self.stockList)

        # 添加期权组合
        self.call, self.put = call, put
        self.add_comb(self.call, self.put)

        # 主图标的K线
        self.timesteps = timesteps
        self.run_data = self.account_controller.real_info_controller.get_bars_between_from_df(benchmark, start_time, end_time)
        self.close_list = None
        self.ts_list = None
        self.total_length = len(self.run_data)
        self.row_index = 0
        ts = str(self.run_data.iloc[0]['ts']).replace(' ',  '').replace('-', '').replace(':', '')
        self.account_controller.init_hv160(self.start_time, self.end_time, self.benchmark)
        self.account_controller.init_state(ts, self.run_data.iloc[0]['close'])
        
        
    # 添加组合
    def add_comb(self, call, put):
        self.account_controller.set_combos(call, put)
    
    # 返回 (current_dim, history_dim)
    def get_raw_shape(self):
        current_state, history_state = self.account_controller.get_total_state()

        current_state = torch.tensor(current_state, dtype=torch.float32)
        history_state = torch.tensor(history_state, dtype=torch.float32)

        return current_state.shape, history_state.shape

    # 需要返回: (next_state, reward, terminated, truncated)
    def step(self, action, weight, test: bool=False) -> tuple:
        if self.row_index >= self.total_length:
            current_state, history_state = self.account_controller.get_total_state()
            return current_state, history_state, self.account_controller.getReward(), True, True
        
        ts, close = self.ts_list[self.row_index], self.close_list[self.row_index]
        self.row_index += 1

        if self.row_index >= min(self.total_length, self.timesteps):
            

            reward = 0.0
            peak = self.account_controller.equity_peak
            current = self.account_controller.equity
            dd = max(0, (peak - current) / (peak + 1e-6))

            # 回撤惩罚
            if dd > 0.1:
                reward -= 10.0
            else:
                reward -= dd * 50
            
            # 破产暴击
            if self.account_controller.equity < self.account_controller.init_capital * 0.6:
                reward -= 50.0

            # 回撤很小且在高点附近说明完美
            if dd < 0.02 and self.account_controller.equity > self.account_controller.init_capital:
                reward += 10.0

            current_state, history_state = self.account_controller.get_total_state()
            return current_state, history_state, reward, True, False

        current_state, history_state, reward, truncated = self.account_controller.step(action, weight, ts, close)

        return current_state, history_state, reward, False, truncated


    def clean(self, ts_str: str):
        if isinstance(ts_str, str):
            return ts_str.replace(' ', '').replace('-', '').replace(':', '')
        # 如果ts已经是datetime对象，需要先格式化为字符串
        return str(ts_str).replace(' ', '').replace('-', '').replace(':', '')
    
    # 需要返回: (state, info)
    def reset(self) -> tuple:
        if hasattr(self, 'account_controller'):
            del self.account_controller
        self.account_controller = single_Account(self.init_capital, self.fee, '30m', self.stockList)

        # 添加期权组合
        self.add_comb(self.call, self.put)
        self.run_data = self.account_controller.real_info_controller.get_bars_between_from_df(self.benchmark, self.start_time, self.end_time)
        
        self.account_controller.init_hv160(self.start_time, self.end_time, self.benchmark)
        self.close_list = self.run_data['close'].values.astype(np.float32)
        self.ts_list = [self.clean(ts) for ts in self.run_data['ts']]
        self.row_index = 0
        self.total_length = len(self.run_data)

        ts, close = self.ts_list[0], self.close_list[0]

        self.account_controller.init_state(ts, close)

        current_state, _ = self.account_controller.get_total_state()
        history_state = self.account_controller.get_history_state()
        info = self.account_controller.getInfo()

        return current_state, history_state, info
    
    def close(self):
        del self.account_controller
        del self.run_data

        

