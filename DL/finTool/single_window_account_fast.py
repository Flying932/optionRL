"""
    手动实现一个回测的框架.
    本代码是账户信息(组合级子账户版).
    [最终极速完整版] 
    1. 包含向量化牛顿法 (Vectorized Newton-Raphson) 反推 IV。
    2. 包含向量化 Black-Scholes 计算 Greeks。
    3. preload_data 预计算所有数据，消除训练时的 CPU 计算压力。
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

# 引入包 (根据你的文件结构保持不变)
if __name__ != '__main__':
    from finTool.optionBaseInfo import optionBaseInfo
    from finTool.realInfo import RealInfo
else:
    from optionBaseInfo import optionBaseInfo
    from realInfo import RealInfo

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
class single_Account:
    def __init__(self, init_capital: float, fee: float = 1.3, period: str = '30m',
                 stockList: Optional[List[str]] = None, filepath: str = './miniQMT/datasets/',
                 window: int=32):
        self.init_capital = float(init_capital)
        self.filepath = filepath
        self.fee = float(fee)
        self.period = period if period else '30m'
        self.stockList = stockList if stockList else ['510050', '588000']

        self.free_money = float(init_capital)
        self.frozen_money = 0.0
        self.equity = float(init_capital)
        self.positions: Dict[str, Tuple[str, int, float]] = {}

        self.Trades: List[Trade] = []
        self.Orders: List[Order] = []
        self.has_disposed_id = -1

        self.option_info_controller = optionBaseInfo(self.stockList, f'{filepath}/optionInfo')
        self.real_info_controller = RealInfo(self.stockList, f'{filepath}/realInfo',
                                             period, max_option_cache=15, date_pick='last')

        self.equity_list: List[float] = [self.init_capital]
        self.time_list: List[str] = []
        self.frozen_money_list: List[float] = [self.frozen_money]
        self.free_money_list: List[float] = [self.free_money]
        self.target_gain_list = []

        # 初始化 comb 避免 NoneType 报错
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

        self.target_price: float = 0.0
        self.equity_peak = float(init_capital)
        self.cash_ratio = 1.0
        self.margin_ratio = 0.0
        self.target_gain = 0.0
        self.eps = 1e-6
        self.info = {"message": "initial"}

        self.window_size = window
        self.h_states = deque(maxlen=window)

        # 缓存池
        self.hv_cache = {} 
        self.price_cache = {}
        self.volume_cache = {}
        self.margin_cache = {}
        self.greek_cache = {} 

    def set_combos(self, call: str, put: str):
        self.comb['call'] = call
        self.comb['put'] = put

    def init_hv160(self, start_time: str, end_time: str, targetCode: str):
        """计算历史波动率 (作为 IV 反推失败时的兜底)"""
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

    # ================= 预加载逻辑 (CPU 救星) =================

    def preload_data(self, start_time: str, end_time: str):
        """
        在 reset 阶段一次性计算所有 Greeks。
        解决 CPU 100% 的关键函数。
        """
        codes = [self.comb['call'], self.comb['put']]
        target_code = '510050'
        
        # 1. 加载标的数据
        try:
            df_target = self.real_info_controller.get_bars_between(target_code, start_time, end_time, self.period, columns=('ts', 'close', 'volume'))
        except:
            df_target = self.real_info_controller.get_bars_between_from_df(target_code, start_time, end_time, self.period, columns=('ts', 'close', 'volume'))
        
        # 构造标的价格数组 (按时间对齐)
        target_map = {str(r.ts).replace(' ', '').replace('-', '').replace(':', ''): float(r.close) for r in df_target.itertuples()}
        
        for code in codes:
            if not code: continue
            if code in self.price_cache: continue

            try:
                df = self.real_info_controller.get_bars_between(code, start_time, end_time, self.period, columns=('ts', 'close', 'volume'))
            except:
                df = self.real_info_controller.get_bars_between_from_df(code, start_time, end_time, self.period, columns=('ts', 'close', 'volume'))
            
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
            
            for row in df.itertuples():
                ts_str = str(row.ts).replace(' ', '').replace('-', '').replace(':', '')
                close_p = float(row.close)
                vol = int(row.volume)
                
                p_cache[ts_str] = close_p
                v_cache[ts_str] = vol
                
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

    # ================= 基础查询 =================

    def set_fee(self, fee: float):
        self.fee = float(fee)

    def getClosePrice(self, code: str, time_str: str) -> float:
        if code in self.price_cache:
            return self.price_cache[code].get(time_str, 0.0)
        return float(self.real_info_controller.get_close_by_str(code, time_str))

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

    # ================= 交易逻辑 (保持原样) =================

    def get_option_list(self, stockCode: str = '510050', expire: str = '202512', op_type: str = 'call'):
        return self.option_info_controller.find_options_by_stock_and_expiry(stockCode, expire, op_type)

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

    def _pair_qty_sell_open(self, ts: str, desired: int, call: str, put: str,
                            c_id: int, free_override: Optional[float] = None) -> int:
        m_c = self.getMargin(call)
        m_p = self.getMargin(put)
        per_margin = m_c + m_p

        free_money = self.free_money if free_override is None else float(free_override)
        cap_cash = int(free_money // per_margin) if per_margin > 0 else desired
        vol_cap = min(self.getRealVolume(call, ts), self.getRealVolume(put, ts))
        return max(0, min(int(desired), cap_cash, vol_cap))

    def open_long_pair(self, desired: int, ts: str, call: str, put: str, c_id: int=-1) -> int:
        q = self._pair_qty_buy_open(ts, desired, call, put, c_id)
        if q <= 0: return 0
        self.submit_order(call, '买入开仓', q, ts, c_id=c_id)
        self.submit_order(put,  '买入开仓', q, ts, c_id=c_id)
        return q

    def open_short_pair(self, desired: int, ts: str, call: str, put: str, c_id: int=-1) -> int:
        q = self._pair_qty_sell_open(ts, desired, call, put, c_id)
        if q <= 0: return 0
        self.submit_order(call, '卖出开仓', q, ts, c_id=c_id)
        self.submit_order(put,  '卖出开仓', q, ts, c_id=c_id)
        return q

    def close_pair(self, ts: str, call: str, put: str, w: float = None, c_id: int=-1) -> None:
        w = 1.0 if (w is None) else float(w)
        if w <= 0: return
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

    def flip_short_to_long(self, target: int, ts: str, call: str, put: str, c_id: int=-1) -> None:
        pos_dir, pos_size = self.get_pair_position(call, put)
        v = min(target, pos_size) if pos_dir < 0 else 0
        if v > 0:
            self.submit_order(call, '买入平仓', v, ts, c_id=c_id)
            self.submit_order(put,  '买入平仓', v, ts, c_id=c_id)
        l = target - v
        if l > 0:
            self.open_long_pair(l, ts, call, put, c_id)

    def flip_long_to_short(self, target: int, ts: str, call: str, put: str, c_id: int=-1) -> None:
        pos_dir, pos_size = self.get_pair_position(call, put)
        v = min(target, pos_size) if pos_dir > 0 else 0
        if v > 0:
            self.submit_order(call, '卖出平仓', v, ts, c_id=c_id)
            self.submit_order(put,  '卖出平仓', v, ts, c_id=c_id)
        l = target - v
        if l > 0:
            self.open_short_pair(l, ts, call, put, c_id)

    def get_pair_position(self, call: str, put: str) -> Tuple[int, int]:
        d1, v1 = self.positions.get(call, (None, 0, 0.0))[0:2]
        d2, v2 = self.positions.get(put,  (None, 0, 0.0))[0:2]
        if v1 <= 0 or v2 <= 0 or d1 is None or d2 is None: return 0, 0
        if d1 == d2 == '买入开仓': return 1, min(v1, v2)
        if d1 == d2 == '卖出开仓': return -1, min(v1, v2)
        return 0, 0

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

    def _update_position_values(self, time_str: str):
        for code, (direction, volume, _) in list(self.positions.items()):
            price = self.getClosePrice(code, time_str)
            mul = self.option_info_controller.get_multiplier(code)
            self.positions[code] = (direction, volume, price * volume * mul)

    def _update_comb_equity(self):
        total_value = 0
        for code, (direction, volume, value) in self.positions.items():
            signed_val = value if direction == '买入开仓' else -value
            total_value += signed_val
        self.equity = self.free_money + self.frozen_money + total_value

        last_equity = self.equity
        self.equity_gain = self.equity - last_equity
        self.cash_ratio = 0.0 if abs(self.equity) < self.eps else (self.free_money / self.equity)
        self.margin_ratio = 0.0 if abs(self.equity) < self.eps else (self.frozen_money / self.equity)
        self.equity_peak = max(self.equity_peak, self.equity)

    def update_positions(self, time_str: str):
        delete_list = []
        for code, (direction, volume, _) in list(self.positions.items()):
            if len(code) != 8: continue
            expire = self.option_info_controller.get_expireDate(code)
            if expire > time_str[0:8]: continue

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

        for code in delete_list:
            self.positions.pop(code, None)

        self._update_position_values(time_str)
        self._update_comb_equity()
        self.frozen_money = 0 if abs(self.frozen_money) < self.eps else self.frozen_money
        self.free_money = 0 if abs(self.free_money) < self.eps else self.free_money
        self.frozen_money_list.append(self.frozen_money)
        self.free_money_list.append(self.free_money)

    def simulate_fill(self, time_str: str):
        if self.has_disposed_id >= len(self.Orders) - 1: return
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

        self._update_position_values(time_str)
        self._update_comb_equity()

    def out_excel(self):
        if len(self.time_list) <= 0: return
        filepath = f'{self.filepath}/outs/account_info.xlsx'
        peak = 0.0
        for i, eq in enumerate(self.equity_list):
            peak = max(peak, eq)
            if i == 0:
                self.gain_rate.append((eq - self.init_capital) / self.init_capital)
                self.draw_down.append(0.0)
            else:
                prev = self.equity_list[i - 1]
                self.gain_rate.append((eq - prev) / (prev if prev != 0 else 1.0))
                self.draw_down.append((peak - eq) / (peak if peak != 0 else 1.0))

        df = pd.DataFrame({'时间': self.time_list, '市值': self.equity_list, '收益率': self.gain_rate, '回撤': self.draw_down})
        df.to_excel(filepath, sheet_name='账户信息', index=False)
        
        if self.Orders:
            filepath = f'{self.filepath}/outs/order_list.xlsx'
            df = pd.DataFrame([asdict(o) for o in self.Orders])
            df.to_excel(filepath, sheet_name='委托记录', index=False)

        if self.Trades:
            filepath = f'{self.filepath}/outs/trade_list.xlsx'
            df = pd.DataFrame([asdict(t) for t in self.Trades])
            df.to_excel(filepath, sheet_name='交易记录', index=False)
        
    def get_step_length(self, code: str, start_time: str, end_time: str):
        df = self.real_info_controller.get_bars_between(code, start_time, end_time)
        return len(df)

    # ================= 强化学习接口 =================

    def step(self, action, weight, ts, close):
        # 1. 更新环境状态 (走极速缓存)
        self.init_state(ts, close)
        
        # 2. 模拟撮合
        self.simulate_fill(ts)
        
        # 3. 执行 Action
        c_id = 0
        call, put = self.comb['call'], self.comb['put']
        
        if action == 1: # Long
            target_vol = 10 
            self.open_long_pair(target_vol, ts, call, put, c_id)
        elif action == 2: # Short
            target_vol = 10
            self.open_short_pair(target_vol, ts, call, put, c_id)
        elif action == 3: # Close
            self.close_pair(ts, call, put, w=weight, c_id=c_id)
        
        # 4. 更新持仓与净值
        self.update_positions(ts)
        
        # 5. 返回
        curr, hist = self.get_total_state()
        reward = self.getReward(action)
        truncated = self.if_truncated()
        
        return curr, hist, reward, truncated

    def init_state(self, time_str: str, close: float):
        """仅查表，不计算，极速"""
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
        
        self.comb['call_price'] = self.getClosePrice(call, time_str)
        self.comb['put_price'] = self.getClosePrice(put, time_str)
        
        self.comb['call_real_value'] = max(0, close - self.comb['call_strike'])
        self.comb['call_time_value'] = self.comb['call_price'] - self.comb['call_real_value']
        self.comb['put_real_value'] = max(0, self.comb['put_strike'] - close)
        self.comb['put_time_value'] = self.comb['put_price'] - self.comb['put_real_value']
        
        self._update_position_values(time_str)
        self._update_comb_equity()

    def get_history_state(self):
        # 修复 NoneType 错误
        if not self.h_states:
             return [[0.0]*26] * self.window_size

        hist = list(self.h_states)
        if len(hist) < self.window_size:
             hist = [hist[0]] * (self.window_size - len(hist)) + hist
        return hist

    def get_total_state(self):
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

        close = self.target_price + 1e-6
        single = [
            comb['call_strike'] / close,
            comb['call_ttm'] if comb['call_ttm'] else 0,
            comb['call_real_value'] / close,
            comb['call_time_value'] / close,
            self.target_gain,
            comb['call_hv_160'],
            comb['call_iv'],
            comb['call_theta'],
            comb['call_vega'],
            comb['call_gamma'],
            comb['call_delta'],
            comb['call_rho'],
            1, 
            
            comb['put_strike'] / close,
            comb['put_ttm'] if comb['put_ttm'] else 0,
            comb['put_real_value'] / close,
            comb['put_time_value'] / close,
            self.target_gain,
            comb['put_hv_160'],
            comb['put_iv'],
            comb['put_theta'],
            comb['put_vega'],
            comb['put_gamma'],
            comb['put_delta'],
            comb['put_rho'],
            -1 
        ]
        
        self.h_states.append(single)
        return current_state, self.get_history_state()

    def getReward(self, action: int, eps: float=1e-6):
        if len(self.equity_list) <= 1:
            step_ret = 0.0
        else:
            prev, cur = self.equity_list[-2], self.equity_list[-1]
            step_ret = np.log((cur + eps) / (prev + eps))
        
        scale = 10.0
        if step_ret > 0:
            final_reward = step_ret * scale
        else:
            if step_ret >= -0.005:
                final_reward = step_ret * scale * 1.0
            else:
                final_reward = step_ret * scale * 1.5
            
        peak = self.equity_peak
        current = self.equity
        dd = (peak - current) / peak
        dd_penalty = -10.0 * dd 
        final_reward += dd_penalty
        
        if self.equity < self.init_capital * 0.5:
            final_reward -= 5 

        if action == 0 and self.has_positions():
            final_reward += 0.1

        return float(final_reward)

    def if_truncated(self) -> bool:
        return (self.equity / self.init_capital) < 0.05
    
    def getInfo(self):
        return self.info

    def has_positions(self):
        if self.comb['pos_size'] != 0:
            return True
        return False

    def combine_label_step(self, ts: str, close: float, targetCode: str='510050'):
        return {}
    
# ========================== 用例 ==========================
if __name__ == '__main__':
    # 示例：单组合跨式 + 逐步调用step
    start_time = '20250825100000'
    # start_time = '20251025100000'
    # start_time = '20250923143000'
    end_time = '20250925150000'
    # end_time = '20251125150000'

    calls, puts = [], []

    call = '10008800'
    put = '10008809'
   
    account = single_Account(100000, fee=1.3, period='30m', stockList=['510050'])

    account.set_combos(call, put)
    target = '510050'

    # dtype = {
    #     'call': str,
    #     'put': str,
    #     'call_strike': int,
    #     'put_strike': int,
    #     'call_open': str,
    #     'put_open': str,
    #     'call_expire': str,
    #     'put_expire': str,
    #     'ignore_days': int,
    #     'steps': int,
    # }

    # def calculate_score(row):
    #     """自定义计算逻辑，输入是一行数据"""
    #     start = row['call_open']
    #     end = row['call_expire']

    #     start_time = datetime.strptime(start, "%Y%m%d")
    #     end_time = datetime.strptime(end, "%Y%m%d")
    #     days = (end_time - start_time).days

    #     if days <= 40:
    #         return 0

    #     end_time = start_time + timedelta(days=20)
    #     end_time = end_time.strftime('%Y%m%d')

    #     start_time = start + '100000'
    #     end_time = end_time + '150000'
    #     call = row['call']
    #     return account.get_step_length(call, start_time, end_time)

    # df = pd.read_excel('./miniQMT/datasets/all_label_data/20251213_train.xlsx', dtype=dtype)
    # df['steps'] = df.apply(calculate_score, axis=1)
    # df['ignore_days'] = df.apply(lambda row: 20, axis=1)
    # df.to_excel('./miniQMT/datasets/all_label_data/20251213_train.xlsx', dtype=dtype)
    # print(0 / 0)

    # 取样本数据(你的 RealInfo 里方法名可能是 get_bars_between 或 get_bars_between_from_df)
    try:
        data = account.real_info_controller.get_bars_between(target, start_time, end_time, '30m')
    except AttributeError:
        data = account.real_info_controller.get_bars_between_from_df(target, start_time, end_time, '30m')

    account.preload_data(start_time, end_time)
    # 初始化一次
    first_close = float(data.iloc[0].close)
    first_ts = str(data.iloc[0].ts).replace(' ', '').replace('-', '').replace(':', '')
    account.init_state(first_ts, first_close)

    for i in range(len(data)):
        ts = str(data.iloc[i].ts).replace(' ', '').replace('-', '').replace(':', '')
        close = float(data.iloc[i].close)

        if i == 0:
            account.step(2, 1, ts, close)
        elif ts[0: 8] != '20250925':
            react_state, state, reward, truncated = account.step(0, 0, ts, close)

        
    print(0 / 0)

    # account.out_excel()
