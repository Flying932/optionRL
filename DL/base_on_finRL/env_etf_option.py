import gymnasium as gym  # FinRL/SB3 现在主要支持 gymnasium
from gymnasium import spaces
import numpy as np
import random
import pandas as pd
from Engine import tradeEngine  # 确保你的 Engine.py 在这个路径下

class OptionStraddleEnv(gym.Env):
    """
    FinRL 兼容的期权跨式交易环境
    特性：
    1. 多任务 (Multi-task): reset 时随机切换期权对
    2. 因果对齐: T时刻决策 -> T+1开盘成交 -> T+1收盘结算
    3. 混合观测: 包含历史序列(Transformer用)和账户特征(MLP用)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df_dict, option_pairs_list, cfg):
        super(OptionStraddleEnv, self).__init__()
        
        # 数据集: 建议传入一个 Dict, key是期权对ID, value是对应的DataFrame
        # 这样在 reset 切换任务时查询速度最快 O(1)
        self.data_map = df_dict 
        self.all_pairs = option_pairs_list
        self.cfg = cfg
        
        # --- 1. 定义动作空间 ---
        # 维度 0: 动作类型 [Hold, Long, Short, Close]
        # 维度 1: 仓位权重 [0%, 25%, 50%, 75%, 100%]
        self.action_space = spaces.MultiDiscrete([4, 5])
        
        # --- 2. 定义观测空间 ---
        # 必须与 Engine.get_total_state() 返回的维度一致
        # hist: [32, 26] (32个时间步, Call+Put各13个特征)
        # curr: [Dc] (账户资金、Greeks、持仓状态等)
        # 这里的 26 和 14 需要根据你实际的数据列数调整
        self.observation_space = spaces.Dict({
            "hist": spaces.Box(low=-np.inf, high=np.inf, shape=(32, 26), dtype=np.float32),
            "curr": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        })

        # --- 3. 实例化物理内核 ---
        self.engine = tradeEngine(init_capital=cfg.init_capital)
        self.current_task_id = None
        self.row_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # A. 随机采样任务 (多任务泛化核心)
        task = random.choice(self.all_pairs)
        self.current_task_id = task['id'] # 假设 all_pairs 里有唯一标识符
        self.current_df = self.data_map[self.current_task_id]
        
        # B. 重置物理引擎
        self.engine.reset()
        self.engine.comb = task # 载入 Call/Put 配置
        
        # C. 数据预加载 (调用你 Engine 里的优化逻辑)
        self.engine.preload_data(task['start_time'], task['end_time'])
        
        # D. 初始化状态指针
        self.row_idx = 32 # 留出历史窗口
        
        # 初始化 T 时刻状态
        row = self.current_df.iloc[self.row_idx]
        self.engine.init_state(row['ts'], row['close'])
        
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        """
        从 Engine 获取处理好的状态，并组装成 Gym Dict
        """
        # 调用 Engine 现有的方法获取状态
        # curr: 账户及当前物理特征
        # hist: 过去32个窗口的序列特征
        curr, hist = self.engine.get_total_state()
        
        # 确保数据类型为 float32
        return {
            "hist": hist.astype(np.float32),
            "curr": curr.astype(np.float32)
        }

    def step(self, action):
        """
        执行标准 RL 步进。
        逻辑：T决策 -> T+1执行 -> 返回 T+1 状态和 T+1 日内收益
        """
        # 检查是否越界
        if self.row_idx >= len(self.current_df) - 2:
            return self._get_obs(), 0.0, True, False, {}

        # 1. 解析动作
        act_type = action[0]
        weight_idx = action[1]
        weight = [0.0, 0.25, 0.50, 0.75, 1.0][weight_idx]

        # 2. 获取 T 和 T+1 的时间戳/价格
        row_now = self.current_df.iloc[self.row_idx]
        row_next = self.current_df.iloc[self.row_idx + 1]
        
        ts_now = row_now['ts']
        ts_next = row_next['ts']
        close_next = row_next['close']

        # ====================================================
        # 调用 Engine 的原子能力 (因果对齐逻辑)
        # ====================================================
        
        # A. 提交订单 (T时刻决策)
        # 你需要在 Engine 中确保 submit_orders 只是记录指令，不改变资金
        self.engine.submit_orders(act_type, weight, ts_now)

        # B. 模拟成交 (T+1 开盘)
        # 这会扣除手续费，更新持仓
        self.engine.simulate_fill(ts_next, use_open_price=True)
        
        # C. 捕捉责任起始水位 (T+1 开盘成交后的净值)
        self.engine.update_positions(ts_next, use_open=True) # 用Open价更新市值
        equity_open = self.engine.equity

        # D. 推进时间到 T+1 收盘
        self.engine.update_positions(ts_next, use_open=False) # 用Close价更新市值
        self.engine.init_state(ts_next, close_next)           # 刷新Greeks
        self.engine._update_comb_equity()
        self.engine.equity_list.append(self.engine.equity)

        # E. 计算奖励 (T+1日内收益，归功于 act_type)
        reward = self.engine.getReward(equity_open, act_type)
        
        # F. 记录上一步动作供下次参考
        self.engine.last_action = act_type

        # ====================================================

        # 3. 推进指针
        self.row_idx += 1
        done = self.row_idx >= len(self.current_df) - 2
        
        # 4. 获取新状态 (T+1收盘状态)
        obs = self._get_obs()
        
        # Info 中包含净值，供 FinRL 回测绘图
        info = {"equity": self.engine.equity}
        
        return obs, reward, done, False, info