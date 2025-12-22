import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import torch
from Engine import tradeEngine  # 确保 Engine.py 在同级目录

class OptionStraddleEnv(gym.Env):
    """
    FinRL 兼容的期权跨式交易环境 (2025-12-21 最终修正版)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, option_pairs_list, cfg):
        super(OptionStraddleEnv, self).__init__()
        
        self.all_pairs = option_pairs_list
        self.cfg = cfg
        
        # 1. 动作空间: [动作(4), 权重(5)]
        self.action_space = spaces.MultiDiscrete([4, 5])
        
        # 2. 观测空间: 匹配 Engine.get_total_state()
        # curr(14维), hist(32步, 26特征)
        self.observation_space = spaces.Dict({
            "hist": spaces.Box(low=-np.inf, high=np.inf, shape=(32, 26), dtype=np.float32),
            "curr": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        })

        # 3. 实例化物理内核
        self.engine = tradeEngine(
            init_capital=cfg.init_capital,
            fee=cfg.fee,
            period='30m',
            stockList=[cfg.benchmark],
            filepath='./miniQMT/datasets/',
            window=32
        )
        
        self.current_pair = None
        self.row_idx = 0
        self.ts_arr = []
        self.close_arr = []
        self.total_length = 0

    def clean_ts(self, ts_str):
        return str(ts_str).replace(' ', '').replace('-', '').replace(':', '')

    def get_smooth_reward(self, raw_terminal_bonus):
        """ 迁移自 windowEnv 的奖励平滑逻辑 """
        if raw_terminal_bonus >= 0:
            return np.clip(raw_terminal_bonus, 0, 1.5)
        else:
            return -5.0 * np.tanh(np.abs(raw_terminal_bonus) / 30.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 随机采样任务
        self.current_pair = random.choice(self.all_pairs)
        
        # 引擎数据预加载 (遵循 windowEnv 逻辑)
        self.engine.reset(
            start_time=self.current_pair['start_time'],
            end_time=self.current_pair['end_time'],
            targetCode=self.cfg.benchmark
        )
        self.engine.set_combos(self.current_pair['call'], self.current_pair['put'])
        
        # 加载标的时间轴
        run_data = self.engine.real_info_controller.get_bars_between_from_df(
            self.cfg.benchmark, 
            self.current_pair['start_time'], 
            self.current_pair['end_time']
        )
        self.close_arr = run_data['close'].values.astype(np.float32)
        self.ts_arr = [self.clean_ts(ts) for ts in run_data['ts']]
        self.total_length = len(run_data)
        
        # 指针从 0 开始 (Engine 会自动填充历史窗口)
        self.row_idx = 0
        ts, close = self.ts_arr[0], self.close_arr[0]
        self.engine.init_state(ts, close)
        
        obs, _ = self._get_obs()
        return obs, {}

    def _get_obs(self):
        curr, hist = self.engine.get_total_state()
        return {
            "hist": np.array(hist, dtype=np.float32),
            "curr": np.array(curr, dtype=np.float32)
        }, {}

    def step(self, action):
        if self.row_idx >= self.total_length - 1:
            obs, _ = self._get_obs()
            return obs, 0.0, True, False, {"equity": self.engine.equity}

        # 获取 T 和 T+1 信息
        ts_now = self.ts_arr[self.row_idx]
        close_now = self.close_arr[self.row_idx]
        ts_next = self.ts_arr[self.row_idx + 1]
        close_next = self.close_arr[self.row_idx + 1]

        # 物理步进
        act_type = int(action[0])
        weight_val = [0.0, 0.25, 0.50, 0.75, 1.0][int(action[1])]

        _, _, step_reward, truncated = self.engine.step(
            action=act_type,
            weight=weight_val,
            ts=ts_now,
            close=close_now,
            ts_next=ts_next,
            close_next=close_next
        )

        self.row_idx += 1
        terminated = (self.row_idx >= self.total_length - 1)
        final_reward = float(step_reward)

        info = {'equity': self.engine.equity}
        total_return_ratio = self.engine.equity / self.engine.init_capital
        total_return = total_return_ratio - 1
        annual_factor = 252 * 8 / max(1, self.row_idx)

        simple_ann = total_return * annual_factor
        if total_return_ratio > 0:
            compound_ann = (total_return_ratio ** annual_factor) - 1
        else:
            compound_ann = -1.0
        sr = self.engine.get_sharpe_ratio()

        # 每一帧都把实时指标塞进 info
        info = {
            "equity": self.engine.equity,
            "running_metrics": {
                "sharpe": sr,
                "simple_ann": simple_ann,
                "log_ann": compound_ann,
                "return": simple_ann
            }
        }

        # 终端夏普惩罚
        if terminated or truncated:
            peak = self.engine.equity_peak
            current = self.engine.equity
            drawdown = (peak - current) / (peak + 1e-6)
            
            bonus = 0.0
            if sr > 2.5: bonus -= ((sr - 2.5) ** 2) * 30 
            elif 1.0 <= sr <= 2.5: bonus += 1.5 
            elif sr < 0.5: bonus -= 0.5

            if current < self.engine.init_capital * 0.8: bonus -= 2.0 
            if drawdown > 0.08: bonus -= 1.0 
            
            final_reward += self.get_smooth_reward(bonus)

        obs, _ = self._get_obs()
        return obs, final_reward, terminated, truncated, info