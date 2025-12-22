"""
    手动实现一个回测的框架.
    本代码是账户信息(组合级子账户版).
    [最终极速完整版] 
    1. 包含向量化牛顿法 (Vectorized Newton-Raphson) 反推 IV。
    2. 包含向量化 Black-Scholes 计算 Greeks。
    3. preload_data 预计算所有数据，消除训练时的 CPU 计算压力。

    更新时间2025-12-20, 本框架需要对接finRL
"""
from dataclasses import asdict, dataclass, field
from typing import ClassVar, Dict, Tuple, List, Optional
from itertools import count
import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import deque
import warnings
from datetime import timedelta, datetime

import sys
from pathlib import Path

def setup_miniqmt_import_root():
    """
    递归查找 'miniQMT' 文件夹，并将其添加到 sys.path 中，
    从而允许使用 miniQMT 为根的绝对导入。
    """
    try:

        calling_script_path = Path(sys._getframe(1).f_globals['__file__']).resolve()
    except KeyError:

        print("⚠️ 警告: 无法确定当前脚本路径，跳过路径设置。")
        return
    
    current_path = calling_script_path
    miniqmt_root = None
    for parent in [current_path] + list(current_path.parents):
        if parent.name == 'miniQMT':
            miniqmt_root = parent
            break
    if miniqmt_root:
        miniqmt_root_str = str(miniqmt_root)
        if miniqmt_root_str not in sys.path:
            sys.path.insert(0, miniqmt_root_str)
            print(f"✅ 成功将项目根目录添加到搜索路径: {miniqmt_root_str}")
        else:
            # 已经添加过，无需重复添加
            # print(f"ℹ️ 项目根目录已在搜索路径中: {miniqmt_root_str}")
            pass
    else:
        print("❌ 错误: 未能在当前路径或其任何父目录中找到 'miniQMT' 文件夹。")
setup_miniqmt_import_root()
from DL.finTool.optionBaseInfo import optionBaseInfo
from DL.finTool.realInfo import RealInfo

# ========================== 数据结构 ==========================
@dataclass(slots=True)
class Order:
    _id_counter: ClassVar[count] = count(1)
    order_id: int = field(init=False)
    code: str
    direction: str
    init_volume: int
    success_volume: int
    time_str: str
    status: str
    info: str = ''
    c_id: int = -1
    def __post_init__(self):
        self.order_id = next(self._id_counter)

@dataclass(slots=True)
class Trade:
    order_id: int
    code: str
    direction: str
    price: float
    fee: float
    time_str: str
    success_volume: int

# ========================== 账户类 ==========================
class tradeEngine:
    def __init__(self, 
                 init_capital: float,
                 fee: float=1.3,
                 period: str='30m',
                 stockList=None,
                 filepath: str='./miniQMT/datasets/',
                 window: int=32,
                 start_time: str='',
                 end_time: str='',
                ):
        self.init_capital = init_capital
        self.filepath = filepath
        self.fee = fee
        self.period = period if period else '30m'
        self.stockList = stockList if stockList else ['510050']
        self.window_size = window if window else 32

        # 获取数据的两个类
        self.option_info_controller = optionBaseInfo(self.stockList, f'{filepath}/optionInfo')
        self.real_info_controller = RealInfo(self.stockList, f'{filepath}/realInfo',
                                             period, max_option_cache=15, date_pick='last')

        # 初始化所有变量
        self.free_money = self.init_capital
        self.frozen_money = 0
        self.equity = self.init_capital
        self.equity_peak = self.equity

        # ('买入开仓', abs(signed), new_value)
        self.positions = {}
        self.Trades: List[Trade] = []
        self.Orders: List[Order] = []
        self.has_disposed_id = -1

        self.equity_list = [self.equity]
        self.frozen_money_list = [self.frozen_money]
        self.free_money_list = [self.free_money]
        self.target_gain_list = [0]
        self.target_price = 0.0
        self.raw_returns = []
        
        
        # ppo
        self.last_action = 0

        # 初始化变量信息
        self.reset(start_time, end_time, self.stockList[0])


    # 对交易引擎初始化
    def reset(self, start_time: str, end_time: str, targetCode: str='510050'):
        self.free_money = self.init_capital
        self.frozen_money = 0
        self.equity = self.init_capital
        self.equity_peak = self.equity

        # ('买入开仓', abs(signed), new_value)
        self.positions = {}
        self.Trades: List[Trade] = []
        self.Orders: List[Order] = []
        self.has_disposed_id = -1

        self.equity_list = [self.equity]
        self.frozen_money_list = [self.frozen_money]
        self.free_money_list = [self.free_money]
        self.target_gain_list = [0]
        self.target_price = 0.0

        self.raw_returns = []

        # 期权状态信息
        self.comb = {
            'call': None, 'put': None,
            'call_price': 0.0, 'put_price': 0.0,
            'call_strike': 0.0, 'put_strike': 0.0,
            'call_ttm': 0.0, 'put_ttm': 0.0,
            'call_iv': 0.0, 'put_iv': 0.0,
            'call_theta': 0.0, 'put_theta': 0.0,
            'call_vega': 0.0, 'put_vega': 0.0,
            'call_delta': 0.0, 'put_delta': 0.0,
            'call_gamma': 0.0, 'put_gamma': 0.0,
            'call_rho': 0.0, 'put_rho': 0.0,
            'pos_dir': 0, 'pos_size': 0,
            'call_real_value': 0.0, 'put_real_value': 0.0,
            'call_time_value': 0.0, 'put_time_value': 0.0,
            'call_hv_160': 0.0, 'put_hv_160': 0.0,
        }  

        # 账户状态信息
        self.cash_ratio = 1.0
        self.margin_ratio = 0.0 

        self.eps = 1e-6
        self.h_states = deque(maxlen=self.window_size)

        # 缓存池
        self.hv_cache = {} 
        self.price_cache = {}
        self.volume_cache = {}
        self.margin_cache = {}
        self.greek_cache = {} 
        self.open_cache = {}  # 【新增】存 Open
        self.raw_returns = []  # 记录真实的单步收益率（不带任何 penalty）

        # ppo
        self.last_action = 0

        # 调用缓存加载hv160
        self.init_hv160(start_time, end_time)

        # 预加载期权greeks
        self.preload_data(start_time, end_time)

    # 设置期权组合
    def set_combos(self, call: str, put: str):
        self.comb['call'] = call
        self.comb['put'] = put
    
    # hv160的缓存, 需要根据时间进行加载, 需要调用来初始化
    def init_hv160(self, start_time: str, end_time: str, targetCode: str='510050'):
        """计算历史波动率 (作为 IV 反推失败时的兜底)"""
        if self.comb['call'] is None or self.comb['put'] is None:
            return
        try:
            before_str = self.real_info_controller.get_prev_30_days(start_time[0: 8], days=100)
            before_str = before_str + start_time[8: ]
            hv_data = self.real_info_controller.get_bars_between(targetCode, before_str, end_time, '30m')
            hv_data = hv_data[['ts', 'close']].copy()
            hv_data['close_prev'] = hv_data['close'].shift(1)
            hv_data['log_diff'] = np.log(hv_data['close'] / hv_data['close_prev'])
            window_size = 160
            hv_data['rolling_std_160'] = hv_data['log_diff'].rolling(window=window_size).std() * np.sqrt(2016)
            
            self.hv_cache = {}
            for row in hv_data.itertuples():
                ts_str = str(row.ts).replace(' ', '').replace('-', '').replace(':', '')
                if not np.isnan(row.rolling_std_160):
                    self.hv_cache[ts_str] = row.rolling_std_160
        except Exception as e:
            print(f"[Warn] HV160 init failed: {e}")

    def get_hv_160(self, ts: str):
        return self.hv_cache.get(ts, 0.0)

    # ================= 核心算法：向量化 BS & IV 反推 =================
    def _bs_price_vectorized(self, S, K, T, r, sigma, op_type, q=0.0):
        """向量化计算理论价格，用于 IV 反推时的误差计算"""
        T = np.maximum(T, 1e-5)
        sigma = np.maximum(sigma, 1e-4)
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        exp_qt = np.exp(-q * T)
        exp_rt = np.exp(-r * T)
        
        # Call Price
        call_price = S * exp_qt * norm.cdf(d1) - K * exp_rt * norm.cdf(d2)
        
        # Vega (Call/Put 相同)
        vega = S * exp_qt * np.sqrt(T) * norm.pdf(d1)
        
        if op_type == 'call':
            return call_price, vega
        else:
            # Put Price 利用平价公式: P = C - S*e^-qT + K*e^-rT
            put_price = call_price - S * exp_qt + K * exp_rt
            return put_price, vega

    def _vectorized_implied_volatility(self, S, K, T, r, market_price, op_type, q=0.0):
        """
        全向量化反推隐含波动率 (Newton-Raphson Method)
        一次性并行计算 1000+ 个数据点，极快。
        """
        # 1. 初始化猜测值 (0.5)
        sigma = np.full_like(S, 0.5) 
        
        # 2. 牛顿迭代 (8次通常足够收敛到 1e-5 精度)
        for i in range(8):
            price_theo, vega = self._bs_price_vectorized(S, K, T, r, sigma, op_type, q)
            diff = price_theo - market_price
            
            # 防止 Vega 过小导致除零
            vega = np.where(vega < 1e-8, 1e-8, vega)
            
            # 更新 Sigma
            sigma = sigma - diff / vega
            
            # 边界限制，防止飞出正常范围
            sigma = np.clip(sigma, 0.001, 5.0)
            
        return sigma

    def _bs_greeks_vectorized(self, S, K, T, r, sigma, op_type, q=0.0):
        """向量化计算所有 Greeks"""
        T = np.maximum(T, 1e-5)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        exp_qt = np.exp(-q * T)
        exp_rt = np.exp(-r * T)
        
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        cdf_neg_d1 = norm.cdf(-d1)
        cdf_neg_d2 = norm.cdf(-d2)

        gamma = (exp_qt * pdf_d1) / (S * sigma * np.sqrt(T))
        vega = S * exp_qt * np.sqrt(T) * pdf_d1

        if op_type == 'call':
            delta = exp_qt * cdf_d1
            theta = -(S * sigma * exp_qt * pdf_d1) / (2 * np.sqrt(T)) + q * S * exp_qt * cdf_d1 - r * K * exp_rt * cdf_d2
            rho = K * T * exp_rt * cdf_d2
        else:
            delta = -exp_qt * cdf_neg_d1
            theta = -(S * sigma * exp_qt * pdf_d1) / (2 * np.sqrt(T)) - q * S * exp_qt * cdf_neg_d1 + r * K * exp_rt * cdf_neg_d2
            rho = -K * T * exp_rt * cdf_neg_d2

        # 清洗极小值
        threshold = 1e-6
        delta = np.where(np.abs(delta) < threshold, 0.0, delta)
        gamma = np.where(np.abs(gamma) < threshold, 0.0, gamma)
        theta = np.where(np.abs(theta) < threshold, 0.0, theta)
        vega  = np.where(np.abs(vega)  < threshold, 0.0, vega)
        rho   = np.where(np.abs(rho)   < threshold, 0.0, rho)

        return delta, gamma, theta, vega, rho

    def get_greeks_vectorized(self, S, K, T, r, market_price, op_type='call', q=0.0):
        """自动反推 IV 并计算 Greeks"""
        # 1. 反推 IV
        iv = self._vectorized_implied_volatility(S, K, T, r, market_price, op_type, q)
        # 2. 计算 Greeks
        d, g, t_val, v, rho = self._bs_greeks_vectorized(S, K, T, r, iv, op_type, q)
        return iv, d, g, t_val, v, rho

    # ================= 预加载Greeks =================
    def preload_data(self, start_time: str, end_time: str):
        """
        在 reset 阶段一次性计算所有 Greeks。
        解决 CPU 100% 的关键函数。
        """
        codes = [self.comb['call'], self.comb['put']]
        target_code = self.stockList[0]
        
        # 1. 加载标的数据
        try:
            df_target = self.real_info_controller.get_bars_between(target_code, start_time, end_time, self.period, columns=('ts', 'close', 'volume', 'open'))
        except:
            df_target = self.real_info_controller.get_bars_between_from_df(target_code, start_time, end_time, self.period, columns=('ts', 'close', 'volume', 'open'))
        
        # 构造标的价格数组 (按时间对齐)
        target_map = {str(r.ts).replace(' ', '').replace('-', '').replace(':', ''): float(r.close) for r in df_target.itertuples()}
        
        for code in codes:
            if not code: continue
            if code in self.price_cache: continue

            try:
                df = self.real_info_controller.get_bars_between(code, start_time, end_time, self.period, columns=('ts', 'close', 'volume', 'open'))
            except:
                df = self.real_info_controller.get_bars_between_from_df(code, start_time, end_time, self.period, columns=('ts', 'close', 'volume', 'open'))
            
            # 准备向量化计算的数组
            ts_list = []
            P_arr = [] # 期权价格
            S_arr = [] # 标的价格
            T_arr = [] # 剩余时间
            
            K = self.option_info_controller.get_strikePrice(code)
            expire = self.option_info_controller.get_expireDate(code)
            op_type = self.option_info_controller.get_optionType(code)

            # 利率是写死的
            r = 1.3849 / 100
            
            p_cache = {}
            v_cache = {}
            o_cache = {} # Open Cache 【新增】
            
            for row in df.itertuples():
                ts_str = str(row.ts).replace(' ', '').replace('-', '').replace(':', '')
                close_p = float(row.close)
                vol = int(row.volume)
                open_p = float(row.open) # 【新增】
                
                p_cache[ts_str] = close_p
                v_cache[ts_str] = vol
                o_cache[ts_str] = open_p # 【新增】
                
                # 对齐标的价格
                s_val = target_map.get(ts_str)
                if s_val is not None:
                    ttm = self.real_info_controller.get_ttm(ts_str, expire)
                    
                    ts_list.append(ts_str)
                    P_arr.append(close_p)
                    S_arr.append(s_val)
                    T_arr.append(ttm)

            self.price_cache[code] = p_cache
            self.volume_cache[code] = v_cache
            self.open_cache[code] = o_cache # 【新增】
            self.margin_cache[code] = float(self.option_info_controller.get_margin(code))
            
            # 🔥 批量计算 Greeks (如果有数据)
            g_cache = {}
            if len(P_arr) > 0:
                S_np = np.array(S_arr)
                P_np = np.array(P_arr)
                T_np = np.array(T_arr)
                
                iv_v, d_v, g_v, t_v, v_v, rho_v = self.get_greeks_vectorized(
                    S_np, K, T_np, r, P_np, op_type, q=0.0
                )
                
                # 存入缓存
                for i, ts_val in enumerate(ts_list):
                    g_cache[ts_val] = {
                        'delta': float(d_v[i]), 
                        'gamma': float(g_v[i]), 
                        'theta': float(t_v[i]), 
                        'vega': float(v_v[i]), 
                        'rho': float(rho_v[i]), 
                        'iv': float(iv_v[i])
                    }
            
            self.greek_cache[code] = g_cache


    # ================= 基础查询函数 =================
    def set_fee(self, fee: float):
        self.fee = float(fee)

    def getClosePrice(self, code: str, time_str: str) -> float:
        if code in self.price_cache:
            return self.price_cache[code].get(time_str, 0.0)
        return float(self.real_info_controller.get_close_by_str(code, time_str))

    def getOpenPrice(self, code: str, time_str: str) -> float:
            """优先从缓存取 Open, 没有则读文件"""
            if code in self.open_cache:
                return self.open_cache[code].get(time_str, 0.0)
            return float(self.real_info_controller.get_open_by_str(code, time_str))

    def getRealVolume(self, code: str, time_str: str) -> int:
        if code in self.volume_cache:
            return self.volume_cache[code].get(time_str, 0)
        return int(self.real_info_controller.get_volume_by_str(code, time_str))

    def getMargin(self, optionCode: str) -> float:
        if optionCode in self.margin_cache:
            return self.margin_cache[optionCode]
        return float(self.option_info_controller.get_margin(optionCode))

    def getRealMargin(self, optionCode: str, time_str: str) -> float:
        stockCode = self.option_info_controller.get_stockCode(optionCode)
        stock_price = self.getClosePrice(stockCode, time_str)
        strike_price = self.option_info_controller.get_strikePrice(optionCode)
        option_price = self.getClosePrice(optionCode, time_str)
        op_type = self.option_info_controller.get_optionType(optionCode)

        if op_type == "put":
            delta = max(stock_price - strike_price, 0)
            m = option_price + max(0.12 * stock_price - delta, 0.07 * strike_price)
            margin = min(m, strike_price) * self.option_info_controller.get_multiplier(optionCode)
        elif op_type == "call":
            delta = max(strike_price - stock_price, 0)
            margin = option_price + max(0.12 * stock_price - delta, 0.07 * stock_price)
            margin = margin * self.option_info_controller.get_multiplier(optionCode)
        else:
            margin = 0.0
        return float(margin)


# ================= 交易逻辑 =================
    def get_option_list(self, stockCode: str = '510050', expire: str = '202512', op_type: str = 'call'):
        return self.option_info_controller.find_options_by_stock_and_expiry(stockCode, expire, op_type)

    # 提交委托
    def submit_order(self, code: str, direction: str, volume: int, time_str: str,
                     price: float = None, c_id: int = -1):
        if len(code) == 8:
            assert direction in ['买入开仓', '卖出开仓', '买入平仓', '卖出平仓']
        elif len(code) in (6, 7):
            assert direction in ['买入', '卖出']
        else:
            raise ValueError(f"未知代码格式: {code}")

        order = Order(code, direction, int(volume), 0, time_str, '已报', '', int(c_id))
        self.Orders.append(order)

    # 计算可买开跨式期权的张数
    def _pair_qty_buy_open(self, ts: str, desired: int, call: str, put: str,
                           c_id: int, free_override: Optional[float] = None) -> int:
        price_c = self.getClosePrice(call, ts)
        price_p = self.getClosePrice(put, ts)
        mul_c = self.option_info_controller.get_multiplier(call)
        mul_p = self.option_info_controller.get_multiplier(put)
        per_cost = price_c * mul_c + price_p * mul_p + 2 * self.fee

        free_money = self.free_money if free_override is None else float(free_override)
        cap_cash = int(free_money // per_cost) if per_cost > 0 else desired
        vol_cap = min(self.getRealVolume(call, ts), self.getRealVolume(put, ts))
        return max(0, min(int(desired), cap_cash, vol_cap))

    # 计算可卖开跨式期权的张数
    def _pair_qty_sell_open(self, ts: str, desired: int, call: str, put: str,
                            c_id: int, free_override: Optional[float] = None) -> int:
        m_c = self.getMargin(call)
        m_p = self.getMargin(put)
        per_margin = m_c + m_p

        free_money = self.free_money if free_override is None else float(free_override)
        cap_cash = int(free_money // per_margin) if per_margin > 0 else desired
        vol_cap = min(self.getRealVolume(call, ts), self.getRealVolume(put, ts))
        return max(0, min(int(desired), cap_cash, vol_cap))

    # 买开
    def open_long_pair(self, desired: int, ts: str, call: str, put: str, c_id: int=-1) -> int:
        q = self._pair_qty_buy_open(ts, desired, call, put, c_id)
        if q <= 0: 
            return 0
        self.submit_order(call, '买入开仓', q, ts, c_id=c_id)
        self.submit_order(put,  '买入开仓', q, ts, c_id=c_id)
        return q

    # 卖开
    def open_short_pair(self, desired: int, ts: str, call: str, put: str, c_id: int=-1) -> int:
        q = self._pair_qty_sell_open(ts, desired, call, put, c_id)
        if q <= 0: 
            return 0
        self.submit_order(call, '卖出开仓', q, ts, c_id=c_id)
        self.submit_order(put,  '卖出开仓', q, ts, c_id=c_id)
        return q

    # 平仓
    def close_pair(self, ts: str, call: str, put: str, w: float = None, c_id: int=-1) -> None:
        w = 1.0 if (w is None) else float(w)
        if w <= 0: 
            return
        for code in (call, put):
            pos = self.positions.get(code)
            if not pos: continue
            d, v, _ = pos
            if v <= 0: continue
            v_to_close = int(v * w)
            if v_to_close <= 0: v_to_close = 1
            v_to_close = min(v_to_close, v)

            if d == '卖出开仓':
                self.submit_order(code, '买入平仓', v_to_close, ts, c_id=c_id) 
            elif d == '买入开仓':
                self.submit_order(code, '卖出平仓', v_to_close, ts, c_id=c_id)

    # 空头变多头, target是总减少的组合数
    def flip_short_to_long(self, target: int, ts: str, call: str, put: str, c_id: int=-1) -> None:
        pos_dir, pos_size = self.get_pair_position(call, put)
        v = min(target, pos_size) if pos_dir < 0 else 0
        if v > 0:
            self.submit_order(call, '买入平仓', v, ts, c_id=c_id)
            self.submit_order(put,  '买入平仓', v, ts, c_id=c_id)
        l = target - v
        if l > 0:
            self.open_long_pair(l, ts, call, put, c_id)

    # 多头变空头
    def flip_long_to_short(self, target: int, ts: str, call: str, put: str, c_id: int=-1) -> None:
        pos_dir, pos_size = self.get_pair_position(call, put)
        v = min(target, pos_size) if pos_dir > 0 else 0
        if v > 0:
            self.submit_order(call, '卖出平仓', v, ts, c_id=c_id)
            self.submit_order(put,  '卖出平仓', v, ts, c_id=c_id)
        l = target - v
        if l > 0:
            self.open_short_pair(l, ts, call, put, c_id)

    # 获取持仓的方向和仓位
    def get_pair_position(self, call: str, put: str) -> Tuple[int, int]:
        d1, v1 = self.positions.get(call, (None, 0, 0.0))[0:2]
        d2, v2 = self.positions.get(put,  (None, 0, 0.0))[0:2]
        if v1 <= 0 or v2 <= 0 or d1 is None or d2 is None: return 0, 0
        if d1 == d2 == '买入开仓': return 1, min(v1, v2)
        if d1 == d2 == '卖出开仓': return -1, min(v1, v2)
        return 0, 0

    # 撮合成交
    def dispose_order(self, code: str, dispose_volume: int, price: float,
                      free_money_delta: float, frozen_money_delta: float):
        if code in self.positions:
            direction, volume, _ = self.positions[code]
            signed = volume if direction == '买入开仓' else -volume
            signed += dispose_volume
            new_value = price * abs(signed) * self.option_info_controller.get_multiplier(code)
            if signed > 0:
                self.positions[code] = ('买入开仓', abs(signed), new_value)
            elif signed < 0:
                self.positions[code] = ('卖出开仓', abs(signed), new_value)
            else:
                del self.positions[code]
        else:
            if dispose_volume > 0:
                val = price * dispose_volume * self.option_info_controller.get_multiplier(code)
                self.positions[code] = ('买入开仓', dispose_volume, val)
            elif dispose_volume < 0:
                val = price * abs(dispose_volume) * self.option_info_controller.get_multiplier(code)
                self.positions[code] = ('卖出开仓', abs(dispose_volume), val)

        self.frozen_money += float(frozen_money_delta)
        self.free_money += float(free_money_delta)

    # 根据当前的持仓市值更新账户持仓数量、总市值(浮动盈亏)
    def _update_comb_equity(self):
        """
        仅负责更新：当前净值、持仓组合状态、资金利用率。
        不负责计算收益率 (target_gain)，收益率由 step 函数统一控制。
        """
        total_value = 0.0
        # 1. 计算持仓市值
        for code, (direction, volume, value) in self.positions.items():  
            if direction == '买入开仓':
                total_value += value
            elif direction == '卖出开仓':
                total_value -= value # 卖方持仓是负债
        
        self.equity = self.free_money + self.frozen_money + total_value

        # 2. 更新组合状态标签 (Call/Put Pair)
        call, put = self.comb['call'], self.comb['put']
        p_dir, p_size = self.get_pair_position(call, put)
        self.comb['pos_dir'] = p_dir
        self.comb['pos_size'] = p_size

        # 3. 更新风控指标
        # 避免除以 0
        denom = self.equity if abs(self.equity) > 1e-6 else 1.0
        self.cash_ratio = self.free_money / denom
        self.margin_ratio = self.frozen_money / denom
        self.equity_peak = max(self.equity_peak, self.equity)


    # 根据最新价格更新持仓市值, 到期的强制清算, 没到期的根据市值更新
    def update_positions(self, time_str: str, use_open: bool=False):
        delete_list = []
        for code, (direction, volume, _) in list(self.positions.items()):
            if len(code) != 8: 
                continue

            expire = self.option_info_controller.get_expireDate(code)

            # 到期的期权强制清算
            if expire <= time_str[0: 8]: 
                price = self.getClosePrice(code, time_str)
                mul = self.option_info_controller.get_multiplier(code)
                margin = self.getMargin(code)
                c_id = -1

                if direction == '买入开仓':
                    order = Order(code, '卖出平仓', volume, volume, time_str, '成交', '强制卖出平仓', c_id)
                    self.Orders.append(order)
                    trade = Trade(order.order_id, code, '卖出平仓', price * volume * mul, 0.0, time_str, volume)
                    self.Trades.append(trade)
                    free_delta = price * volume * mul
                    self.dispose_order(code, -volume, price, free_delta, 0.0)
                else:
                    fee = self.fee * volume
                    order = Order(code, '买入平仓', volume, volume, time_str, '成交', '强制买入平仓', c_id)
                    self.Orders.append(order)
                    trade = Trade(order.order_id, code, '买入平仓', price * volume * mul, fee, time_str, volume)
                    self.Trades.append(trade)
                    frozen_delta = -margin * volume
                    free_delta = margin * volume - price * volume * mul - fee
                    self.dispose_order(code, +volume, price, free_delta, frozen_delta)
                delete_list.append(code)
            
            else:
                # 未到期的更新其市值
                if use_open:
                    price = self.getOpenPrice(code, time_str)
                else:
                    price = self.getClosePrice(code, time_str)
                mul = self.option_info_controller.get_multiplier(code)
                self.positions[code] = (direction, volume, price * volume * mul)

        for code in delete_list:
            self.positions.pop(code, None)


        self.frozen_money = 0 if abs(self.frozen_money) < self.eps else self.frozen_money
        self.free_money = 0 if abs(self.free_money) < self.eps else self.free_money
        self.frozen_money_list.append(self.frozen_money)
        self.free_money_list.append(self.free_money)

    def simulate_fill_moc(self, time_str: str):
        """
        MOC 专用撮合函数 (修正版)
        1. 以当前收盘价 + 随机滑点 进行撮合。
        2. 【修复】平仓时严格遵循下单数量 (volume)，不再强制全平。
        """
        # 如果没有新订单，直接返回
        if self.has_disposed_id >= len(self.Orders) - 1: 
            return
        
        # 遍历所有未处理的订单
        for order in self.Orders[self.has_disposed_id + 1:]:
            self.has_disposed_id += 1 # 标记为已处理
            
            code = order.code
            direction = order.direction
            volume = int(order.init_volume) # 这是下单时指定的数量 (受 w 影响)
            order_id = order.order_id
            
            # 跳过强制平仓单
            if '强制' in (order.info or ''): continue

            # 期权到期检查
            if len(code) == 8:
                expire = self.option_info_controller.get_expireDate(code)
                if expire < time_str[0:8]:
                    order.status = '废单'
                    order.info = '期权到期后无法下单'
                    continue

            # 获取真实成交量限制
            real_volume = self.getRealVolume(code, time_str)
            
            # 1. 获取基础收盘价
            raw_close = self.getClosePrice(code, time_str)
            
            # 2. 计算滑点
            slippage_rate = abs(np.random.normal(0, 0.0005)) 
            
            if '买' in direction:
                 price = raw_close * (1 + slippage_rate)
            else:
                 price = raw_close * (1 - slippage_rate)

            # 3. 资金与持仓处理
            mul = self.option_info_controller.get_multiplier(code)
            margin = self.getMargin(code) if len(code) == 8 else 0.0
            free_sub = self.free_money

            if len(code) == 8:
                if direction == '买入开仓':
                    cost_per_unit = mul * price + self.fee
                    num_can_buy = int(free_sub // cost_per_unit) if cost_per_unit > 0 else 0
                    max_cnt = max(0, min(volume, num_can_buy, real_volume))
                    
                    if max_cnt <= 0:
                        order.status = '废单'
                        order.info = '组合资金不足开仓'
                        continue
                        
                    order.success_volume = max_cnt
                    order.status = '成交' if max_cnt == volume else '部分成交'
                    
                    frozen_delta = 0.0
                    free_delta = -price * max_cnt * mul - max_cnt * self.fee
                    
                    self.dispose_order(code, +max_cnt, price, free_delta, frozen_delta)
                    trade = Trade(order_id, code, direction, price * max_cnt * mul, self.fee * max_cnt, time_str, max_cnt)
                    self.Trades.append(trade)

                elif direction == '卖出开仓':
                    num_can_sell_open = int(free_sub // margin) if margin > 0 else 0
                    max_cnt = max(0, min(volume, num_can_sell_open, real_volume))
                    
                    if max_cnt <= 0:
                        order.status = '废单'
                        order.info = '组合资金不足开仓'
                        continue
                        
                    order.success_volume = max_cnt
                    order.status = '成交' if max_cnt == volume else '部分成交'
                    
                    frozen_delta = margin * max_cnt
                    free_delta = price * mul * max_cnt - margin * max_cnt
                    
                    self.dispose_order(code, -max_cnt, price, free_delta, frozen_delta)
                    trade = Trade(order_id, code, direction, price * max_cnt * mul, 0.0, time_str, max_cnt)
                    self.Trades.append(trade)

                elif direction == '买入平仓':
                    raw = self.positions.get(code, ('无仓位', 0, 0.0))
                    raw_dir, raw_vol = raw[0], raw[1]
                    
                    # 检查持仓是否足够
                    if raw_vol == 0 or raw_dir != '卖出开仓':
                        order.status = '废单'
                        continue
                        
                    # 【修复】这里必须取 min(下单量, 现有持仓量)
                    # 之前错误的写成了 max_cnt = raw_vol，导致 w 失效
                    max_cnt = min(volume, raw_vol)
                    
                    if max_cnt <= 0:
                         order.status = '废单'
                         continue

                    fee = self.fee * max_cnt
                    
                    frozen_delta = -margin * max_cnt # 释放保证金
                    free_delta = margin * max_cnt - fee - price * max_cnt * mul 
                    
                    order.success_volume = max_cnt
                    order.status = '成交'
                    
                    self.dispose_order(code, +max_cnt, price, free_delta, frozen_delta)
                    trade = Trade(order_id, code, direction, price * max_cnt * mul, fee, time_str, max_cnt)
                    self.Trades.append(trade)

                elif direction == '卖出平仓':
                    raw = self.positions.get(code, ('无仓位', 0, 0.0))
                    raw_dir, raw_vol = raw[0], raw[1]
                    
                    if raw_vol == 0 or raw_dir != '买入开仓':
                        order.status = '废单'
                        continue
                    
                    # 【修复】同上，遵循下单量
                    max_cnt = min(volume, raw_vol)
                    
                    if max_cnt <= 0:
                         order.status = '废单'
                         continue
                    
                    frozen_delta = 0.0
                    free_delta = price * max_cnt * mul 
                    
                    order.success_volume = max_cnt
                    order.status = '成交'
                    
                    self.dispose_order(code, -max_cnt, price, free_delta, frozen_delta)
                    trade = Trade(order_id, code, direction, price * max_cnt * mul, 0.0, time_str, max_cnt)
                    self.Trades.append(trade)
            else:
                order.status = '废单'
                order.info = '目前暂不交易股票'


    def simulate_fill(self, time_str: str, use_open_price: bool = True):
        """
        Args:
            time_str: 当前时间
            use_open_price: 如果为 True,强制使用 Open 价格进行撮合 (Next Open 模式)
                            如果为 False,使用 Close 价格 (MOC 模式)
        """
        if self.has_disposed_id >= len(self.Orders) - 1: 
            return
        
        for order in self.Orders[self.has_disposed_id + 1:]:
            self.has_disposed_id += 1
            code = order.code
            direction = order.direction
            volume = int(order.init_volume)
            order_id = order.order_id
            
            if '强制' in (order.info or ''): continue

            if len(code) == 8:
                expire = self.option_info_controller.get_expireDate(code)
                if expire < time_str[0:8]:
                    order.status = '废单'
                    order.info = '期权到期后无法下单'
                    continue

            real_volume = self.getRealVolume(code, time_str)
    
            # 【核心修改】：根据参数决定用 Open 还是 Close
            if use_open_price:
                price = self.getOpenPrice(code, time_str)
            else:
                price = self.getClosePrice(code, time_str)

            mul = self.option_info_controller.get_multiplier(code)
            margin = self.getMargin(code) if len(code) == 8 else 0.0
            free_sub = self.free_money

            if len(code) == 8:
                if direction == '买入开仓':
                    num_can_buy = int(free_sub // (mul * price + self.fee))
                    max_cnt = max(0, min(volume, num_can_buy, real_volume))
                    if max_cnt <= 0:
                        order.status = '废单'
                        order.info = '组合资金不足开仓'
                        continue
                    order.success_volume = max_cnt
                    order.status = '成交' if max_cnt == volume else '部分成交'
                    frozen_delta = 0.0
                    free_delta = -price * max_cnt * mul - max_cnt * self.fee
                    self.dispose_order(code, +max_cnt, price, free_delta, frozen_delta)
                    trade = Trade(order_id, code, direction, price * max_cnt * mul, self.fee * max_cnt, time_str, max_cnt)
                    self.Trades.append(trade)

                elif direction == '卖出开仓':
                    num_can_sell_open = int(free_sub // margin) if margin > 0 else 0
                    max_cnt = max(0, min(volume, num_can_sell_open, real_volume))
                    if max_cnt <= 0:
                        order.status = '废单'
                        order.info = '组合资金不足开仓'
                        continue
                    order.success_volume = max_cnt
                    order.status = '成交' if max_cnt == volume else '部分成交'
                    frozen_delta = margin * max_cnt
                    free_delta = price * mul * max_cnt - margin * max_cnt
                    self.dispose_order(code, -max_cnt, price, free_delta, frozen_delta)
                    trade = Trade(order_id, code, direction, price * max_cnt * mul, 0.0, time_str, max_cnt)
                    self.Trades.append(trade)

                elif direction == '买入平仓':
                    raw = self.positions.get(code, ('无仓位', 0, 0.0))
                    raw_dir, raw_vol = raw[0], raw[1]
                    if raw_vol == 0 or raw_dir != '卖出开仓':
                        order.status = '废单'
                        continue
                    max_cnt = raw_vol
                    fee = self.fee * max_cnt
                    frozen_delta = -margin * max_cnt
                    free_delta = margin * max_cnt - fee - price * max_cnt * mul
                    order.success_volume = max_cnt
                    order.status = '成交'
                    self.dispose_order(code, +max_cnt, price, free_delta, frozen_delta)
                    trade = Trade(order_id, code, direction, price * max_cnt * mul, fee, time_str, max_cnt)
                    self.Trades.append(trade)

                elif direction == '卖出平仓':
                    raw = self.positions.get(code, ('无仓位', 0, 0.0))
                    raw_dir, raw_vol = raw[0], raw[1]
                    if raw_vol == 0 or raw_dir != '买入开仓':
                        order.status = '废单'
                        continue
                    max_cnt = raw_vol
                    frozen_delta = 0.0
                    free_delta = price * max_cnt * mul
                    order.success_volume = max_cnt
                    order.status = '成交'
                    self.dispose_order(code, -max_cnt, price, free_delta, frozen_delta)
                    trade = Trade(order_id, code, direction, price * max_cnt * mul, 0.0, time_str, max_cnt)
                    self.Trades.append(trade)
            else:
                order.status = '废单'
                order.info = '目前暂不交易股票'


    # ================= 状态相关 =================
    def if_truncated(self) -> bool:
        return (self.equity / self.init_capital) < 0.05

    def has_positions(self):
        if self.comb['pos_size'] != 0:
            return True
        return False  

    def init_state(self, time_str: str, close: float):
        """仅查表，不计算，极速更新当前时刻的市场状态"""
        self.target_price = float(close)
        call, put = self.comb['call'], self.comb['put']
        
        self.comb['call_strike'] = self.option_info_controller.get_strikePrice(call)
        self.comb['put_strike'] = self.option_info_controller.get_strikePrice(put)
        
        # 从缓存读取 Greeks
        c_greeks = self.greek_cache.get(call, {}).get(time_str, {})
        p_greeks = self.greek_cache.get(put, {}).get(time_str, {})
        
        self.comb['call_delta'] = c_greeks.get('delta', 0.0)
        self.comb['put_delta']  = p_greeks.get('delta', 0.0)
        self.comb['call_gamma'] = c_greeks.get('gamma', 0.0)
        self.comb['put_gamma']  = p_greeks.get('gamma', 0.0)
        self.comb['call_vega']  = c_greeks.get('vega', 0.0)
        self.comb['put_vega']   = p_greeks.get('vega', 0.0)
        self.comb['call_theta'] = c_greeks.get('theta', 0.0)
        self.comb['put_theta']  = p_greeks.get('theta', 0.0)
        self.comb['call_iv']    = c_greeks.get('iv', 0.0)
        self.comb['put_iv']     = p_greeks.get('iv', 0.0)
        
        # 【新增】更新 rho (之前的代码漏了 rho，但你的状态里需要)
        self.comb['call_rho']   = c_greeks.get('rho', 0.0)
        self.comb['put_rho']    = p_greeks.get('rho', 0.0)

        # 【新增】更新 HV160 (历史波动率)
        # 假设 HV 是基于标的(510050)的，所以 Call/Put 共用同一个 HV
        hv_val = self.get_hv_160(time_str)
        self.comb['call_hv_160'] = hv_val
        self.comb['put_hv_160']  = hv_val
        
        self.comb['call_price'] = self.getClosePrice(call, time_str)
        self.comb['put_price'] = self.getClosePrice(put, time_str)
        
        self.comb['call_real_value'] = max(0, close - self.comb['call_strike'])
        self.comb['call_time_value'] = self.comb['call_price'] - self.comb['call_real_value']
        self.comb['put_real_value'] = max(0, self.comb['put_strike'] - close)
        self.comb['put_time_value'] = self.comb['put_price'] - self.comb['put_real_value']
        
        # 计算 TTM (虽然 get_total_state 里有判空逻辑，这里更新一下更稳)
        # 注意：这里需要 RealInfo 支持，或者直接从 Greek Cache 反推 TTM (如果有存)
        # 简单起见，如果 cache 里没存 ttm，这里可以用 self.real_info_controller 计算
        # self.comb['call_ttm'] = ... (由于 calculate_score 或其他地方可能算过了，这里暂且不强更，或者在 preload 里加 ttm)
        # 为防止 0，建议在 preload_data 的 greek_cache 构造时把 ttm 也存进去，或者这里实时算：
        expire = self.option_info_controller.get_expireDate(call)
        self.comb['call_ttm'] = self.real_info_controller.get_ttm(time_str, expire)
        self.comb['put_ttm']  = self.comb['call_ttm']

    def get_history_state(self):
        # 修复 NoneType 错误
        if not self.h_states:
             return [[0.0]*26] * self.window_size

        hist = list(self.h_states)
        if len(hist) < self.window_size:
             hist = [hist[0]] * (self.window_size - len(hist)) + hist
        return hist

    def get_total_state(self):
        """
        严格匹配神经网络输入维度的状态获取函数。
        Scalar Dim = 9
        Seq Feature Dim = 26
        """
        current_state = []
        gs = {
            'cash_ratio': self.cash_ratio if abs(self.cash_ratio) > self.eps else 0,
            'margin_ratio': self.margin_ratio if abs(self.margin_ratio) > self.eps else 0,
            'draw_down': 0.0 if self.equity_peak <= 0 else (self.equity_peak - self.equity) / self.equity_peak,
            'max_equity': self.equity_peak / self.init_capital,
        }

        comb = self.comb
        for _, v in gs.items():
            current_state.append(v)
        current_state.append(comb['pos_dir'])
        current_state.append(comb['pos_size'])
        current_state.append(self.free_money / self.init_capital)
        current_state.append(self.frozen_money / self.init_capital)
        current_state.append(self.equity / self.init_capital)

        # 确保分母不为0，虽然 +1e-6 已经处理了
        close = self.target_price + 1e-6
        
        single = [
            # --- Call (13 vars) ---
            comb['call_strike'] / close,
            comb['call_ttm'] if comb['call_ttm'] else 0,
            comb['call_real_value'] / close,
            comb['call_time_value'] / close,
            self.target_gain_list[-1], # Log Return
            comb['call_hv_160'],
            comb['call_iv'],
            comb['call_theta'],
            comb['call_vega'],
            comb['call_gamma'],
            comb['call_delta'],
            comb['call_rho'],
            1, # Flag
            
            # --- Put (13 vars) ---
            comb['put_strike'] / close,
            comb['put_ttm'] if comb['put_ttm'] else 0,
            comb['put_real_value'] / close,
            comb['put_time_value'] / close,
            self.target_gain_list[-1], # Log Return
            comb['put_hv_160'],
            comb['put_iv'],
            comb['put_theta'],
            comb['put_vega'],
            comb['put_gamma'],
            comb['put_delta'],
            comb['put_rho'],
            -1 # Flag
        ]
        
        self.h_states.append(single)
        return current_state, self.get_history_state()

    # ================= 强化学习接口 =================
    def step(self, action, weight, ts, close, ts_next, close_next):
        """
        Args:
            action/weight(a_T, w_T): T收盘时的决策
            ts/close: T的时间戳和收盘价
            ts_next/close_next: T+1的时间戳和收盘价

        Logic:
            * T时刻提交动作, 在T+1开盘价成交
            * T+1收盘时查看持仓市值和账户信息
        """
        target_gain = np.log(close_next / close)
        self.target_gain_list.append(target_gain)
        self.target_price = close_next

        # 立即提交订单: 结合 Weight 动态计算下单量 (购买力约束 90%)
        call, put = self.comb['call'], self.comb['put']
        max_margin_allow = self.free_money * 0.9
        margin_per_pair = self.getMargin(call) + self.getMargin(put)
        cap_vol = int(max_margin_allow / (margin_per_pair + self.eps))
        target_vol = int(min(cap_vol, 50) * weight) 

        if weight > 0 and target_vol < 1: 
            target_vol = 1

        if action == 1:
            self.open_long_pair(target_vol, ts, call, put)
        elif action == 2: 
            self.open_short_pair(target_vol, ts, call, put)
        elif action == 3: 
            self.close_pair(ts, call, put, w=weight)

        # 立即成交, 按照T+1的开盘价, 这会更新资金
        self.simulate_fill(ts_next, use_open_price=True)

        # 推进时间观察, 根据T+1更新持仓的市值
        self.update_positions(ts_next, use_open=True)

        # T+1开盘价市值
        equity_open = self.equity

        self.update_positions(ts_next, use_open=False)
        
        # 刷新T+1时的状态(greeks)
        self.init_state(ts_next, close_next)
        self._update_comb_equity()
        self.equity_list.append(self.equity)
        self.raw_returns.append(np.log(self.equity_list[-1] / self.equity_list[-2]))

        # 计算reward
        reward = self.getReward(equity_open, action)
        self.last_action = action

        # 获取状态(T+1收盘时的状态)
        curr, hist = self.get_total_state()

        return curr, hist, reward, self.if_truncated()


    def getReward_1222(self, equity_open: float, action: int, eps: float=1e-6):
        if len(self.equity_list) <= 1: 
            return 0.0
        
        # perv: T+1开盘净值 | cur: T+1收盘净值
        prev, cur = equity_open, self.equity_list[-1]
        peak = self.equity_peak
        
        # 使用对数收益率
        step_ret = np.log((cur + eps) / (prev + eps))
        
        # 1. 基础收益映射 
        final_reward = step_ret * 150.0

        # 2. 🔥 强制诱导做多 (A_LONG = 1)
        # 给买方单一点点生存补偿，降低模型对权利金时间损耗的“习得性恐惧”
        if action == 1: 
            final_reward += 0.005
            if step_ret > 0:
                final_reward += 0.01  # 多头盈利时额外奖励
        elif action == 2:
            final_reward = final_reward * 0.5 - 0.01  # 空头惩罚减半 + 固定惩罚

        # 3. 利润回吐重罚 (强化版)
        # 如果当前处于盈利状态但收益率转负，加大惩罚，强制模型学会止盈
        if cur > self.init_capital and step_ret < 0:
            final_reward += step_ret * 100.0 # 额外增加 60 倍权重的回吐惩罚

        # 4. 差分移动回撤惩罚
        cur_dd = (peak - cur) / (peak + eps)
        prev_dd = (peak - prev) / (peak + eps)
        if cur_dd > prev_dd:
            # 只要回撤在扩大，就根据回撤增量施加重罚
            final_reward -= (cur_dd - prev_dd) * 100.0 # 权重从 50 提至 100，压制 30w-50w 的巨震

        # 5. 破产/重亏 Step 惩罚 (保持)
        if cur < self.init_capital * 0.7:
            final_reward -= 0.1 

        # 6. 交易频率惩罚 (防止无意义的高频对冲)
        # if action in [1, 2]: 
        #     final_reward -= 0.01 

        # 7. Reward Clipping
        final_reward = np.clip(final_reward, -3.0, 3.0) # 允许负向惩罚更大，约束模型
            
        return float(final_reward)

    # Engine.py -> getReward 优化建议
    def getReward(self, equity_open: float, action: int, eps: float=1e-6):
        prev, cur = equity_open, self.equity_list[-1]
        step_ret = np.log((cur + eps) / (prev + eps))
        
        # 1. 降低基础放大倍数，避免 Value Loss 炸裂
        # 将 300 降到 100-150，使单步奖励主要落在 [-1, 1]
        final_reward = step_ret * 150.0 

        # 2. 引入风险调整后的“持仓奖励”
        # 只有在持仓且净值波动的 standard deviation 较小时才给额外奖励
        if action in [1, 2]: # Long 或 Short
            final_reward += 0.002 # 极微小的鼓励，对抗手续费和时间价值
        
        # 3. 线性化回撤惩罚 (不要用 100 倍这么夸张)
        cur_dd = (self.equity_peak - cur) / (self.equity_peak + eps)
        if cur_dd > 0.05: # 只在回撤超过 5% 时才触发惩罚
            final_reward -= cur_dd * 10.0

        return np.clip(float(final_reward), -3.0, 3.0)

    # 增加计算夏普的方法：
    def get_sharpe_ratio(self):
        """
        计算当前 Episode 的年化夏普比率
        年化因子 = sqrt(252天 * 每天8根30分钟K线) = sqrt(2016)
        """
        if len(self.raw_returns) < 2:
            return 0.0
        
        returns_arr = np.array(self.raw_returns)
        mean_ret = np.mean(returns_arr)
        std_ret = np.std(returns_arr) + 1e-9 # 防止除零
        
        # 30分钟K线的年化因子
        annual_factor = np.sqrt(252 * 8) 
        sharpe = annual_factor * (mean_ret / std_ret)
        return float(sharpe)