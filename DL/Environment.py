"""
    本模块实现一个环境模块, 这是PPO算法所需的env
"""

from abc import ABC, abstractmethod

from zmq import has
from finTool.Account import Account

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
class fintechEnv(baseEnv):
    def __init__(self, init_capital, timesteps: int=30, fee: float=1.3, start_time: str='20251021150000', end_time: str='20251022150000', benchmark: str='510050', call: str='10009039', put: str='10009040'):
        self.fee = fee
        self.start_time = start_time
        self.end_time = end_time
        self.benchmark = benchmark
        self.init_capital = init_capital
        
        
        # 初始化账户对象
        self.call = call
        self.put = put
        self.account_controller = Account(self.init_capital, self.call, self.put, self.fee)

        # 主图标的K线
        self.timesteps = timesteps
        self.run_data = self.account_controller.real_info_controller.get_bars_between_from_df(benchmark, start_time, end_time)
        self.total_length = len(self.run_data)
        self.row_index = 0

    # 需要返回: (next_state, reward, terminated, truncated)
    def step(self, action) -> tuple:
        if self.row_index >= self.total_length:
            return self.account_controller.getState(), self.account_controller.getReward(), True, True
        
        ts, close = self.run_data.iloc[self.row_index]
        ts = str(ts)
        ivs = [' ', '-', ':']
        for item in ivs:
            ts = ts.replace(item, '')

        self.row_index += 1

        if self.row_index >= min(self.total_length, self.timesteps):
            # 导出excel
            # self.account_controller.out_excel()
            return self.account_controller.getState(), 0, True, False

        state, reward, truncated = self.account_controller.step(action, ts, close)
        return state, reward, False, truncated

    # 需要返回: (state, info)
    # 这里的state是没有经过对齐和变量优先学习的
    def reset(self) -> tuple:
        if hasattr(self, 'account_controller'):
            del self.account_controller
        self.account_controller = Account(self.init_capital, self.call, self.put, self.fee)
        self.run_data = self.account_controller.real_info_controller.get_bars_between_from_df(self.benchmark, self.start_time, self.end_time)
        self.row_index = 0
        self.total_length = len(self.run_data)

        ts, close = self.run_data.iloc[0]
        ts = str(ts)
        ivs = [' ', '-', ':']

        for item in ivs:
            ts = ts.replace(item, '')

        self.account_controller.init_state(ts, close)

        state = self.account_controller.getState()
        info = self.account_controller.getInfo()

        return state, info
    
    def close(self):
        del self.account_controller
        del self.run_data

        

