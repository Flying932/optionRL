"""
    本模块实现一个环境模块, 这是PPO算法所需的env.
    本模块是加强版的, 时间窗口长度不等于1
    期权组合也不限为1组
    
    [优化说明]
    1. 支持 preloaded_data 传入，跳过 Pandas IO 和字符串处理。
    2. 使用 Numpy 数组替代 DataFrame iloc，大幅提升 step 速度。
    3. 修复了 __init__ 和 reset 重复加载数据的性能 BUG。
"""

from abc import ABC, abstractmethod
from finTool.single_window_account import single_Account
import numpy as np
import torch

# 环境模型基类
class baseEnv(ABC):
    @abstractmethod
    def reset(self) -> tuple:
        pass
    
    @abstractmethod
    def step(self, action) -> tuple:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

# 环境类
class windowEnv(baseEnv):
    def __init__(self, init_capital, call, put, timesteps: int=30, fee: float=1.3, 
                 start_time: str='20251021150000', end_time: str='20251022150000', 
                 benchmark: str='510050', device: str='cpu', 
                 preloaded_data: dict = None):
        
        self.start_time = start_time
        self.end_time = end_time
        self.benchmark = benchmark
        self.stockList = [benchmark]
        self.init_capital = init_capital
        self.fee = fee
        self.device = device
        self.timesteps = timesteps
        self.call, self.put = call, put
        
        # 核心优化：支持预加载数据
        self.preloaded_data = preloaded_data
        
        # 兼容性变量
        self.account_controller = None
        self.run_data = None
        
        # 如果没有预加载数据，我们在 init 里不进行繁重的初始化
        # 统一放到 reset 中去处理，避免 DynamicWindowEnv 实例化时进行双重初始化

        # 初始化时只做轻量级设置，确保对象可用
        self.row_index = 0
        self.total_length = 0
        self.close_arr = None
        self.ts_arr = None

        # 立即初始化第一步，以便 get_raw_shape 等方法能调用
        # 但注意：如果 preloaded_data 存在，我们不进行耗时操作
        if self.preloaded_data is None:
            # 旧模式：为了兼容直接调用 windowEnv 的代码，这里可能需要初始化
            # 但为了性能，建议外部直接调用 reset() 来启动
            pass 
        else:
            # 极速模式：直接挂载数据引用
            self.close_arr = self.preloaded_data['close_arr']
            self.ts_arr = self.preloaded_data['ts_arr']
            self.total_length = len(self.close_arr)

    # 添加组合
    def add_comb(self, call, put):
        if self.account_controller:
            self.account_controller.set_combos(call, put)
    
    # 返回 (current_dim, history_dim)
    def get_raw_shape(self):
        # 这是一个潜在的瓶颈，如果不 reset 直接调，需要临时初始化
        if self.account_controller is None:
             self.reset()
        
        current_state, history_state = self.account_controller.get_total_state()
        current_state = torch.tensor(current_state, dtype=torch.float32)
        history_state = torch.tensor(history_state, dtype=torch.float32)
        return current_state.shape, history_state.shape

    # 需要返回: (next_state, reward, terminated, truncated)
    def step(self, action, weight, test: bool=False) -> tuple:
        # 越界保护
        if self.row_index >= self.total_length:
            current_state, history_state = self.account_controller.get_total_state()
            return current_state, history_state, self.account_controller.getReward(), True, True
        
        # 优化：使用 Numpy 数组直接取值 (极快)
        ts = self.ts_arr[self.row_index]
        close = self.close_arr[self.row_index]
        
        self.row_index += 1

        # 终止条件判断
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

        # 调用账户核心 step
        current_state, history_state, reward, truncated = self.account_controller.step(action, weight, ts, close)

        return current_state, history_state, reward, False, truncated

    def clean(self, ts_str: str):
        if isinstance(ts_str, str):
            return ts_str.replace(' ', '').replace('-', '').replace(':', '')
        return str(ts_str).replace(' ', '').replace('-', '').replace(':', '')
    
    # 需要返回: (state, info)
    def reset(self) -> tuple:
        # 1. 重置账户对象
        # 优化：如果 account_controller 已经存在且支持 reset，则复用 (取决于 single_Account 实现)
        # 这里为了稳妥，我们还是重新实例化，但数据处理部分会优化
        if hasattr(self, 'account_controller'):
            del self.account_controller
        
        self.account_controller = single_Account(self.init_capital, self.fee, '30m', self.stockList)
        self.add_comb(self.call, self.put) # 设置期权组合

        # 2. 数据加载 (分支处理)
        if self.preloaded_data is not None:
            # === 分支 A: 极速模式 (使用缓存数据) ===
            # 直接引用，不拷贝，不读取，不清洗
            self.close_arr = self.preloaded_data['close_arr']
            self.ts_arr = self.preloaded_data['ts_arr']
            self.total_length = len(self.close_arr)
            
            # 如果预加载数据里包含 hv160，尝试直接注入 account_controller (如果支持)
            # 这里假设 single_Account 依然需要调用 init_hv160，但这是唯一的计算开销了
            # 如果能进一步优化 init_hv160 更好
            self.account_controller.init_hv160(self.start_time, self.end_time, self.benchmark)

        else:
            # === 分支 B: 兼容旧模式 (慢速 IO) ===
            self.run_data = self.account_controller.real_info_controller.get_bars_between_from_df(self.benchmark, self.start_time, self.end_time)
            self.account_controller.init_hv160(self.start_time, self.end_time, self.benchmark)
            self.close_arr = self.run_data['close'].values.astype(np.float32)
            # 列表推导式清洗时间 (慢)
            self.ts_arr = [self.clean(ts) for ts in self.run_data['ts']]
            self.total_length = len(self.run_data)

        # 3. 初始化状态
        self.row_index = 0
        ts, close = self.ts_arr[0], self.close_arr[0]
        self.account_controller.init_state(ts, close)

        current_state, _ = self.account_controller.get_total_state()
        history_state = self.account_controller.get_history_state()
        info = self.account_controller.getInfo()

        return current_state, history_state, info
    
    def close(self):
        if hasattr(self, 'account_controller'):
            del self.account_controller
        if hasattr(self, 'run_data'):
            del self.run_data