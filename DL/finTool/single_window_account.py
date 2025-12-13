"""
    手动实现一个回测的框架.
    本代码是账户信息(组合级子账户版).
"""
from dataclasses import asdict, dataclass, field
from typing import ClassVar, Dict, Tuple, List, Optional
from itertools import count
import numpy as np
import pandas as pd
import math
from collections import deque

from regex import T

# 引入包
if __name__ != '__main__':
    from finTool.optionBaseInfo import optionBaseInfo
    from finTool.realInfo import RealInfo
else:
    from optionBaseInfo import optionBaseInfo
    from realInfo import RealInfo

# 函数太多, 不知道删除与否, 实现一个弃用的修饰器
import warnings
import functools
def deprecated(reason="", version=""):
    def decorator(func):
        msg = f"Function {func.__name__} is deprecated"
        if version:
            msg += f" since version {version}"
        if reason:
            msg += f": {reason}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ========================== 数据结构 ==========================

# 订单类
@dataclass(slots=True)
class Order:
    # 类变量: 订单号自增
    _id_counter: ClassVar[count] = count(1)
    order_id: int = field(init=False)

    # 代码, 10000101 / 588000
    code: str
    # 买卖方向: '买入开仓' / '卖出开仓' / '买入平仓' / '卖出平仓' ；股票: '买入' / '卖出'
    direction: str
    # 下单数量(张/股)
    init_volume: int
    # 成交数量
    success_volume: int
    # 委托时间: str(YYYYmmddHHMMSS)
    time_str: str
    # 订单状态: 已报 / 部分成交 / 废单 / 成交
    status: str
    # 说明信息
    info: str = ''
    # 对应哪个期权组合(为了多期权跨式组合同时操作)
    c_id: int = -1

    def __post_init__(self):
        self.order_id = next(self._id_counter)


# 成交类(若部分成交, 只产生部分成交的成交记录)
@dataclass(slots=True)
class Trade:
    order_id: int
    code: str
    direction: str
    # 成交总价格(合计金额)
    price: float
    # 总手续费
    fee: float
    # 成交时间: 20251018093000
    time_str: str
    # 成交张数
    success_volume: int


# ========================== 账户类 ==========================

class single_Account:
    """
        仅支持单个期权组合
    """

    # ------------------ 初始化与结构 ------------------

    def __init__(self,
                 init_capital: float,
                 fee: float = 1.3,
                 period: str = '30m',
                 stockList: Optional[List[str]] = None,
                 filepath: str = './miniQMT/datasets/',
                 window: int=32,
                 ):
        # 账户资金信息(全局)
        self.init_capital = float(init_capital)
        self.filepath = filepath
        self.fee = float(fee)
        self.period = period if period else '30m'
        self.stockList = stockList if stockList else ['510050', '588000']

        # 账户实时值(全局,注意：在 set_combos 之后会与子账户对齐)
        self.free_money = float(init_capital)      # 可用资金
        self.frozen_money = 0.0                    # 冻结(保证金)
        self.margin = 0.0
        self.equity = float(init_capital)

        # 账户持仓: code -> (direction, volume, value)
        # direction: '买入开仓' / '卖出开仓'(都是正volume)
        self.positions: Dict[str, Tuple[str, int, float]] = {}

        # 交易流水
        self.Trades: List[Trade] = []
        self.Orders: List[Order] = []
        self.has_disposed_id = -1  # 已经处理的订单编号(序号)

        # 控制器, 分别负责基本信息和实时信息
        self.option_info_controller = optionBaseInfo(self.stockList, f'{filepath}/optionInfo')
        self.real_info_controller = RealInfo(self.stockList, f'{filepath}/realInfo',
                                             period, max_option_cache=15, date_pick='last')

        # 记账(全局)
        self.equity_list: List[float] = [self.init_capital]
        self.time_list: List[str] = []
        self.gain_rate: List[float] = []
        self.draw_down: List[float] = []
        self.frozen_money_list: List[float] = [self.frozen_money]
        self.free_money_list: List[float] = [self.free_money]

        # HV
        self.target_gain_list = []


        # 组合期权信息
        self.comb = {
            'call': None,
            'put': None,
            'call_price': None,
            'put_price': None,
            'call_strike': None,
            'put_strike': None,
            'call_ttm': None,
            'put_ttm': None,
            'call_iv': None,
            'put_iv': None,
            'call_theta': None,
            'put_theta': None,
            'call_vega': None,
            'put_vega': None,
            'call_delta': None,
            'put_delta': None,
            'call_rho': None,
            'put_rho': None,
            'pos_dir': 0,
            'pos_size': 0,
            'call_real_value': None,
            'put_real_value': None,
            'call_time_value': None,
            'put_time_value': None,
            'call_hv_160': None,
            'put_hv_160':None,
        }   


        # 全局/ENV相关
        self.last_close: Optional[float] = None
        self.target_price: Optional[float] = None

        # 指标与奖励控制
        self.last_equity = float(init_capital)
        self.equity_peak = float(init_capital)
        self.last_reward = 0.0
        self.last_trade_cnt = 0
        self.use_penalties = False
        self.lam_dd = 1.0
        self.lam_mr = 1.0
        self.lam_trade_cnt = 0.0
        self.dd_floor = 0.15
        self.mr_floor = 0.30
        self.free_trades_per_step = 1

        self.cash_ratio = 1.0
        self.margin_ratio = 0.0
        self.target_gain = 0.0
        self.equity_gain = 0.0

        self.global_state = {
            # 'target_gain': self.target_gain,
            'cash_ratio': self.cash_ratio,
            'margin_ratio': self.margin_ratio,
            'target_price': self.target_price,
            'draw_down': 0.0,
            'max_equity': self.equity / self.init_capital,  # 字段名统一
        }

        self.record = []
        self.eps = 1e-6
        self.info = {"message": "initial"}

        # 历史状态
        self.window_size = window
        self.h_states = deque(maxlen=window)

        # hv160 (30m)
        self.hv_data = None

    # 设置期权组合
    def set_combos(self, call: str, put: str):
        self.comb['call'] = call
        self.comb['put'] = put

    # 初始化HV160
    def init_hv160(self, start_time: str, end_time: str, targetCode: str):
        before_str = self.real_info_controller.get_prev_30_days(start_time[0: 8], days=100)
        before_str = before_str + start_time[8: ]
        hv_data = self.real_info_controller.get_bars_between(targetCode, before_str, end_time, '30m')
        hv_data = hv_data[['ts', 'close']]
        hv_data['close_prev'] = hv_data['close'].shift(1)
        hv_data['log_diff'] = np.log(hv_data['close'] / hv_data['close_prev'])
        window_size = 160
        hv_data['rolling_std_160'] = hv_data['log_diff'].rolling(window=window_size).std() * np.sqrt(2016)
        hv_data = hv_data.set_index('ts')

        self.hv_data = hv_data

    def get_hv_160(self, ts: str):
        if self.hv_data is None:
            return None
        
        search_format = '%Y%m%d%H%M%S'
        target_timestamp = pd.to_datetime(ts, format=search_format)
        std_dev = self.hv_data.loc[ts, 'rolling_std_160']

        return std_dev

    # ------------------ 基础查询/计算 ------------------
    def set_fee(self, fee: float):
        self.fee = float(fee)

    def getMargin(self, optionCode: str) -> float:
        """期权静态保证金"""
        return float(self.option_info_controller.get_margin(optionCode))

    def getRealMargin(self, optionCode: str, time_str: str) -> float:
        """期权动态保证金(保留原先示例公式)"""
        stockCode = self.option_info_controller.get_stockCode(optionCode)
        stock_price = self.real_info_controller.get_close_by_str(stockCode, time_str)
        strike_price = self.option_info_controller.get_strikePrice(optionCode)
        option_price = self.real_info_controller.get_close_by_str(optionCode, time_str)
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
            print(f"[Info: getRealMargin错误] optionCode = {optionCode}, op_type = {op_type}")
            margin = 0.0
        return float(margin)

    def getClosePrice(self, code: str, time_str: str = "20251008093000") -> float:
        return float(self.real_info_controller.get_close_by_str(code, time_str))

    def getRealVolume(self, code: str, time_str: str = "20251008093000") -> int:
        return int(self.real_info_controller.get_volume_by_str(code, time_str))

    def get_option_list(self, stockCode: str = '510050', expire: str = '202512', op_type: str = 'call'):
        """(标的, 到期日, 期权类型) -> 期权代码列表"""
        return self.option_info_controller.find_options_by_stock_and_expiry(stockCode, expire, op_type)

    # ------------------ 订单录入 ------------------
    def submit_order(self, code: str, direction: str, volume: int, time_str: str,
                     price: float = None, c_id: int = -1):
        if len(code) == 8:  # 期权
            assert direction in ['买入开仓', '卖出开仓', '买入平仓', '卖出平仓']
        elif len(code) in (6, 7):  # 股票(暂不支持)
            assert direction in ['买入', '卖出']
        else:
            raise ValueError(f"未知代码格式: {code}")

        order = Order(code, direction, int(volume), 0, time_str, '已报', '', int(c_id))
        self.Orders.append(order)

    # ------------------ 组合原子化操作 ------------------
    def _pair_qty_buy_open(self, ts: str, desired: int, call: str, put: str,
                           c_id: int, free_override: Optional[float] = None) -> int:
        """跨式买开时, 两腿共同可成交的手数 q(按组合 free_money 计算)"""
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
        """跨式卖开时, 两腿共同可成交的手数 q(按组合 free_money 的保证金能力)"""
        m_c = self.getMargin(call)
        m_p = self.getMargin(put)
        per_margin = m_c + m_p

        free_money = self.free_money if free_override is None else float(free_override)
        cap_cash = int(free_money // per_margin) if per_margin > 0 else desired
        vol_cap = min(self.getRealVolume(call, ts), self.getRealVolume(put, ts))
        return max(0, min(int(desired), cap_cash, vol_cap))

    def open_long_pair(self, desired: int, ts: str, call: str, put: str, c_id: int=-1) -> int:
        q = self._pair_qty_buy_open(ts, desired, call, put, c_id)
        if q <= 0:
            return 0
        self.submit_order(call, '买入开仓', q, ts, c_id=c_id)
        self.submit_order(put,  '买入开仓', q, ts, c_id=c_id)
        return q

    def open_short_pair(self, desired: int, ts: str, call: str, put: str, c_id: int=-1) -> int:
        q = self._pair_qty_sell_open(ts, desired, call, put, c_id)
        if q <= 0:
            return 0
        self.submit_order(call, '卖出开仓', q, ts, c_id=c_id)
        self.submit_order(put,  '卖出开仓', q, ts, c_id=c_id)
        return q

    def close_pair(self, ts: str, call: str, put: str, w: float = None, c_id: int=-1) -> None:
        """按权重 w 平掉两腿(w∈[0,1])"""
        w = 1.0 if (w is None) else float(w)
        if w <= 0:
            return

        for code in (call, put):
            pos = self.positions.get(code)
            if not pos:
                continue
            d, v, _ = pos
            if v <= 0:
                continue
            v_to_close = int(v * w)
            if v_to_close <= 0:
                v_to_close = 1  # 有比例但不足1张时,至少平1张
            v_to_close = min(v_to_close, v)

            if d == '卖出开仓':
                self.submit_order(code, '买入平仓', v_to_close, ts, c_id=c_id)   # 关空头
            elif d == '买入开仓':
                self.submit_order(code, '卖出平仓', v_to_close, ts, c_id=c_id)   # 关多头

    def flip_short_to_long(self, target: int, ts: str, call: str, put: str, c_id: int=-1) -> None:
        """从空翻多：先平空,再买开补足到 target"""
        pos_dir, pos_size = self.get_pair_position(call, put)
        v = min(target, pos_size) if pos_dir < 0 else 0
        if v > 0:
            self.submit_order(call, '买入平仓', v, ts, c_id=c_id)
            self.submit_order(put,  '买入平仓', v, ts, c_id=c_id)
        l = target - v
        if l > 0:
            self.open_long_pair(l, ts, call, put, c_id)

    def flip_long_to_short(self, target: int, ts: str, call: str, put: str, c_id: int=-1) -> None:
        """从多翻空：先平多,再卖开补足到 target"""
        pos_dir, pos_size = self.get_pair_position(call, put)
        v = min(target, pos_size) if pos_dir > 0 else 0
        if v > 0:
            self.submit_order(call, '卖出平仓', v, ts, c_id=c_id)
            self.submit_order(put,  '卖出平仓', v, ts, c_id=c_id)
        l = target - v
        if l > 0:
            self.open_short_pair(l, ts, call, put, c_id)

    def get_pair_position(self, call: str, put: str) -> Tuple[int, int]:
        """返回(方向, 手数). 方向：-1空 / 0无 / 1多"""
        d1, v1 = self.positions.get(call, (None, 0, 0.0))[0:2]
        d2, v2 = self.positions.get(put,  (None, 0, 0.0))[0:2]
        if v1 <= 0 or v2 <= 0 or d1 is None or d2 is None:
            return 0, 0
        if d1 == d2 == '买入开仓':
            return 1, min(v1, v2)
        if d1 == d2 == '卖出开仓':
            return -1, min(v1, v2)
        return 0, 0  # 非同向/不成对

    # ------------------ 资金与仓位更新 ------------------

    def dispose_order(self, code: str, dispose_volume: int, price: float,
                      free_money_delta: float, frozen_money_delta: float):
        """更新全局资金与positions | dispose_volume>0视为买方向(多开/买平), <0视为卖方向(空开/卖平)"""
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

        # 全局资金先更新(随后用子账户汇总回写全局)
        self.frozen_money += float(frozen_money_delta)
        self.free_money += float(free_money_delta)


    def _update_position_values(self, time_str: str):
        """重估所有持仓的 value 字段(不做资金变动)"""
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

        # 比率/峰值
        self.cash_ratio = 0.0 if abs(self.equity) < self.eps else (self.free_money / self.equity)
        self.margin_ratio = 0.0 if abs(self.equity) < self.eps else (self.frozen_money / self.equity)
        self.equity_peak = max(self.equity_peak, self.equity)

    def update_positions(self, time_str: str):
        """重估 & 处理到期强平(强平也要按组合记账)"""
        delete_list = []

        # 先处理到期强平
        for code, (direction, volume, _) in list(self.positions.items()):
            if len(code) != 8:
                continue
            expire = self.option_info_controller.get_expireDate(code)
            if expire > time_str[0:8]:
                continue

            # 到期 -> 强制了结
            price = self.getClosePrice(code, time_str)
            mul = self.option_info_controller.get_multiplier(code)
            margin = self.getMargin(code)
            c_id = -1

            if direction == '买入开仓':
                # 强制卖出平仓：收回剩余价值(费用设为0,保持与撮合一致)
                order = Order(code, '卖出平仓', volume, volume, time_str, '成交', '强制卖出平仓', c_id)
                self.Orders.append(order)
                trade = Trade(order.order_id, code, '卖出平仓', price * volume * mul, 0.0, time_str, volume)
                self.Trades.append(trade)

                free_delta = price * volume * mul
                frozen_delta = 0.0
                # 更新仓位与资金
                self.dispose_order(code, -volume, price, free_delta, frozen_delta)
                

            else:  # 卖出开仓
                # 强制买入平仓：释放保证金并支付平仓+手续费
                fee = self.fee * volume
                order = Order(code, '买入平仓', volume, volume, time_str, '成交', '强制买入平仓', c_id)
                self.Orders.append(order)
                trade = Trade(order.order_id, code, '买入平仓', price * volume * mul, fee, time_str, volume)
                self.Trades.append(trade)

                frozen_delta = -margin * volume
                free_delta = margin * volume - price * volume * mul - fee
                # 更新仓位与资金
                self.dispose_order(code, +volume, price, free_delta, frozen_delta)
                

            delete_list.append(code)

        for code in delete_list:
            self.positions.pop(code, None)

        # 重估持仓价值
        self._update_position_values(time_str)

        # 刷新市值
        self._update_comb_equity()

        self.frozen_money = 0 if abs(self.frozen_money) < self.eps else self.frozen_money
        self.free_money = 0 if abs(self.free_money) < self.eps else self.free_money

        self.frozen_money_list.append(self.frozen_money)
        self.free_money_list.append(self.free_money)

    # ------------------ 撮合引擎(逐笔撮合Orders) ------------------
    def simulate_fill(self, time_str: str):
        if self.has_disposed_id >= len(self.Orders) - 1:
            return

        for order in self.Orders[self.has_disposed_id + 1:]:
            self.has_disposed_id += 1
            code = order.code
            direction = order.direction
            volume = int(order.init_volume)
            order_id = order.order_id
            c_id = getattr(order, "c_id", -1)

            # 强制单(到期清算)已经在 update_positions 中生成并处理,这里跳过
            if '强制' in (order.info or ''):
                continue

            # 到期后拒单
            if len(code) == 8:  # 期权
                expire = self.option_info_controller.get_expireDate(code)
                if expire < time_str[0:8]:
                    order.status = '废单'
                    order.info = '期权到期后无法下单'
                    continue

            # 市场信息
            real_volume = self.getRealVolume(code, time_str)
            price = self.getClosePrice(code, time_str)
            mul = self.option_info_controller.get_multiplier(code)
            margin = self.getMargin(code) if len(code) == 8 else 0.0

            # 组合可用资金
            free_sub = self.free_money

            # ----- 期权撮合 -----
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
                    if max_cnt < volume:
                        info = []
                        if volume > real_volume:
                            info.append('下单数量超过真实成交量')
                        if volume > num_can_buy:
                            info.append('资金不足')
                        order.info = ' | '.join(info)

                    frozen_delta = 0.0
                    free_delta = -price * max_cnt * mul - max_cnt * self.fee
                    # 更新仓位与资金
                    self.dispose_order(code, +max_cnt, price, free_delta, frozen_delta)


                    trade = Trade(order_id, code, direction, price * max_cnt * mul,
                                  self.fee * max_cnt, time_str, max_cnt)
                    self.Trades.append(trade)

                elif direction == '卖出开仓':
                    # 以组合 free_money 的保证金能力为准
                    num_can_sell_open = int(free_sub // margin) if margin > 0 else 0
                    max_cnt = max(0, min(volume, num_can_sell_open, real_volume))

                    if max_cnt <= 0:
                        order.status = '废单'
                        order.info = '组合资金不足开仓'
                        continue

                    order.success_volume = max_cnt
                    order.status = '成交' if max_cnt == volume else '部分成交'
                    if max_cnt < volume:
                        info = []
                        if volume > real_volume:
                            info.append('下单数量超过真实成交量')
                        if volume > num_can_sell_open:
                            info.append('资金不足')
                        order.info = ' | '.join(info)

                    frozen_delta = margin * max_cnt
                    free_delta = price * mul * max_cnt - margin * max_cnt
                    self.dispose_order(code, -max_cnt, price, free_delta, frozen_delta)
                    
                    trade = Trade(order_id, code, direction, price * max_cnt * mul,
                                  0.0, time_str, max_cnt)
                    self.Trades.append(trade)

                elif direction == '买入平仓':
                    raw = self.positions.get(code, ('无仓位', 0, 0.0))
                    raw_dir, raw_vol = raw[0], raw[1]
                    if raw_vol == 0 or raw_dir != '卖出开仓':
                        order.status = '废单'
                        order.info = f'没有期权持仓或者方向错误, 无法买入平仓, direction = {raw_dir}'
                        continue

                    max_cnt = raw_vol
                    fee = self.fee * max_cnt
                    frozen_delta = -margin * max_cnt
                    free_delta = margin * max_cnt - fee - price * max_cnt * mul

                    order.success_volume = max_cnt
                    order.init_volume = max_cnt
                    order.status = '成交'

                    self.dispose_order(code, +max_cnt, price, free_delta, frozen_delta)
                    
                    trade = Trade(order_id, code, direction, price * max_cnt * mul, fee, time_str, max_cnt)
                    self.Trades.append(trade)

                elif direction == '卖出平仓':
                    raw = self.positions.get(code, ('无仓位', 0, 0.0))
                    raw_dir, raw_vol = raw[0], raw[1]
                    if raw_vol == 0 or raw_dir != '买入开仓':
                        order.status = '废单'
                        order.info = f'没有期权持仓或者方向错误, 无法卖出平仓, direction = {raw_dir}'
                        continue

                    max_cnt = raw_vol
                    frozen_delta = 0.0
                    free_delta = price * max_cnt * mul  # 卖出收到现金(未计费)

                    order.success_volume = max_cnt
                    order.init_volume = max_cnt
                    order.status = '成交'

                    self.dispose_order(code, -max_cnt, price, free_delta, frozen_delta)
                   
                    trade = Trade(order_id, code, direction, price * max_cnt * mul, 0.0, time_str, max_cnt)
                    self.Trades.append(trade)

            else:
                # 股票暂不支持
                order.status = '废单'
                order.info = '目前暂不交易股票'

        # 每次撮合后,刷新持仓, 刷新市值
        self._update_position_values(time_str)
        self._update_comb_equity()

    # ------------------ 导出 ------------------

    def out_excel(self):
        if len(self.time_list) <= 0:
            return

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

        df = pd.DataFrame({
            '时间': self.time_list,
            '市值': self.equity_list,
            '收益率': self.gain_rate,
            '回撤': self.draw_down
        }, columns=['时间', '市值', '收益率', '回撤'])
        df.to_excel(filepath, sheet_name='账户信息', index=False)
        print(f'[Info] 导出账户信息记录成功 -> {filepath}')

        # 订单
        filepath = f'{self.filepath}/outs/order_list.xlsx'
        if self.Orders:
            df = pd.DataFrame([asdict(o) for o in self.Orders])
            df = df.rename({
                'order_id': '委托号',
                'code': '证券代码',
                'direction': '交易方向',
                'init_volume': '下单数量',
                'success_volume': '成交数量',
                'time_str': '成交时间',
                'status': '委托状态',
                'info': '说明信息',
                'c_id': '组合ID'
            }, axis=1)
            df.to_excel(filepath, sheet_name='委托记录', index=False)
            print(f'[Info] 导出委托记录成功 -> {filepath}')

        # 成交
        filepath = f'{self.filepath}/outs/trade_list.xlsx'
        if self.Trades:
            df = pd.DataFrame([asdict(t) for t in self.Trades])
            df = df.rename({
                'order_id': '委托号',
                'code': '证券代码',
                'direction': '交易方向',
                'price': '总价格',
                'fee': '总手续费',
                'success_volume': '成交数量',
                'time_str': '成交时间',
            }, axis=1)
            df.to_excel(filepath, sheet_name='交易记录', index=False)
            print(f'[Info] 导出交易记录成功 -> {filepath}')

    # ------------------ ENV/策略 ------------------
    def init_state(self, time_str: str, close: float):
        """初始化目标价格/Greeks等(可按需补充)."""
        self.target_price = float(close)
        self.last_close = float(close)

        call, put = self.comb['call'], self.comb['put']
        dic = self.comb

        # 更稳妥的获取价格(不同数据源可能需要 period)
        try:
            call_price = self.real_info_controller.get_close_by_str(call, time_str, self.period)
        except TypeError:
            call_price = self.real_info_controller.get_close_by_str(call, time_str)
        try:
            put_price = self.real_info_controller.get_close_by_str(put, time_str, self.period)
        except TypeError:
            put_price = self.real_info_controller.get_close_by_str(put, time_str)

        call_expire = self.option_info_controller.get_expireDate(call)
        call_ttm = self.real_info_controller.get_ttm(time_str, call_expire)

        put_expire = self.option_info_controller.get_expireDate(put)
        put_ttm = self.real_info_controller.get_ttm(time_str, put_expire)

        call_strike = self.option_info_controller.get_strikePrice(call)
        put_strike = self.option_info_controller.get_strikePrice(put)

        dic['call_price'] = call_price
        dic['put_price'] = put_price

        dic['call_ttm'] = call_ttm
        dic['put_ttm'] = put_ttm

        dic['call_hv_160'] = self.get_hv_160(time_str)
        dic['put_hv_160'] = self.get_hv_160(time_str)


        dic['call_strike'] = call_strike
        dic['put_strike'] = put_strike

        dic['call_real_value'] = max(0, close - call_strike)
        dic['call_time_value'] = (call_price - max(0, close - call_strike))

        dic['put_real_value'] = max(0, put_strike - close)
        dic['put_time_value'] = (put_price - max(0, put_strike - close))

        # Greeks(按需使用)
        call_greeks = self.real_info_controller.cal_greeks(time_str, self.stockList[0], call, call_strike, call_expire, 'call')
        put_greeks  = self.real_info_controller.cal_greeks(time_str, self.stockList[0], put, put_strike, put_expire, 'put')
        dic['call_iv'] = call_greeks['iv']
        dic['put_iv'] = put_greeks['iv']
        dic['call_theta'] = call_greeks['theta']
        dic['put_theta'] = put_greeks['theta']
        dic['call_vega'] = call_greeks['vega']
        dic['put_vega'] = put_greeks['vega']
        dic['call_gamma'] = call_greeks['gamma']
        dic['put_gamma'] = put_greeks['gamma']
        dic['call_delta'] = call_greeks['delta']
        dic['put_delta'] = put_greeks['delta']
        dic['call_rho'] = call_greeks['rho']
        dic['put_rho'] = put_greeks['rho']

        # 初始估值同步到子账户/全局
        self._update_position_values(time_str)
        self._update_comb_equity()

    # 直接补充到self.window_size
    def get_history_state(self):
        first = None
        single = []
        for item in self.h_states:
            single.append(item)

            if first is None:
                first = item
                for _ in range(self.window_size - len(self.h_states)):
                    single.append(first)
            

        return single

    def get_total_state(self):
        current_state = []
        gs = {
            # 'target_gain': self.target_gain,
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

        single = []
        close = self.target_price
        call_strike, put_strike = comb['call_strike'], comb['put_strike']
        call_price, put_price = comb['call_price'], comb['put_price']
        comb['call_real_value'] = max(0, close - call_strike)
        comb['call_time_value'] = (call_price - max(0, close - call_strike))
        comb['put_real_value'] = max(0, put_strike - close)
        comb['put_time_value'] = (put_price - max(0, put_strike - close))

        single.append(comb['call_strike'] / close)
        single.append(comb['call_ttm'])
        single.append(comb['call_real_value'] / close)
        single.append(comb['call_time_value'] / close)
        single.append(self.target_gain)
        single.append(comb['call_hv_160'])
        single.append(comb['call_iv'])
        single.append(comb['call_theta'])
        single.append(comb['call_vega'])
        single.append(comb['call_gamma'])
        single.append(comb['call_delta'])
        single.append(comb['call_rho'])
        single.append(1)
        

        # single.append(comb['put_price'] / close)
        single.append(comb['put_strike'] / close)
        single.append(comb['put_ttm'])
        single.append(comb['put_real_value'] / close)
        single.append(comb['put_time_value'] / close)
        single.append(self.target_gain)
        single.append(comb['put_hv_160'])
        single.append(comb['put_iv'])
        single.append(comb['put_theta'])
        single.append(comb['put_vega'])
        single.append(comb['put_gamma'])
        single.append(comb['put_delta'])
        single.append(comb['put_rho'])
        single.append(-1)
        
        self.h_states.append(single)
        return current_state, self.get_history_state()
    

    def getInfo(self):
        return self.info

    def has_positions(self):
        if self.comb['pos_size'] != 0:
            return True
        return False

    def getReward(self, action: int, eps: float=1e-6):
            # 1. 计算对数收益率 (保持不变)
            if len(self.equity_list) <= 1:
                step_ret = 0.0
            else:
                prev, cur = self.equity_list[-2], self.equity_list[-1]
                step_ret = np.log((cur + eps) / (prev + eps))
            
            # 2. 对称且带有惩罚的奖励机制
            # 放大 100 倍，让 1% 的波动对应 1.0 的奖励值，方便 PPO 学习
            scale = 10.0
            
            if step_ret > 0:
                # 盈利：正常奖励
                final_reward = step_ret * scale
            else:
                # 亏损: 小亏就是正常, 大亏才加倍惩罚
                if step_ret >= -0.005:
                    final_reward = step_ret * scale * 1.0
                else:
                    final_reward = step_ret * scale * 1.5
                
            # 4. 回撤重罚 (Drawdown Penalty) - 关键！
            # 计算当前动态回撤
            peak = self.equity_peak
            current = self.equity
            dd = (peak - current) / peak
            
            # 只要有回撤，就给予持续的惩罚压力
            # 例如回撤 10%，dd=0.1，惩罚 -10 * 0.1 = -1.0
            # 这会迫使 AI 尽快从坑里爬出来，或者干脆别掉进去
            dd_penalty = -10.0 * dd 
            
            final_reward += dd_penalty
            
            # 5. 破产保护 (可选)
            if self.equity < self.init_capital * 0.5: # 亏损超过 50%
                final_reward -= 5 # 给予一次性巨额惩罚，甚至可以考虑结束 Episode

            if action == 0:
                # 持仓的话可以给一点甜头
                if self.has_positions():
                    final_reward += 0.1

            return float(final_reward)

    def if_truncated(self) -> bool:
        return (self.equity / self.init_capital) < 0.05


    # 监督学习遍历
    def combine_label_step(self, ts: str, close: float, targetCode: str='510050'):
        dic = {}

        dic['时间'] = ts[0: 4] + '-' + ts[4: 6] + '-' + ts[6: 8] + ' ' + ts[8: 10] + ':' + ts[10: 12] + ':' + ts[12: 14]

        call, put = self.label_pairs
        call_price = self.getClosePrice(call, ts)
        put_price = self.getClosePrice(put, ts)
        
        call_strike = self.option_info_controller.get_strikePrice(call)
        put_strike = self.option_info_controller.get_strikePrice(put)
        assert call_strike == put_strike, 'Strike不同'

        # 内在价值
        call_value = max(0, close - call_strike) / close
        put_value = max(0, put_strike - close) / close

        # 时间价值
        call_time_value = call_price / close - call_value
        put_time_value = put_price / close - put_value

        # 行权价虚实
        strike = call_strike / close

        # 到期日 (转化为年)
        expire = self.option_info_controller.get_expireDate(call)
        ttm = self.real_info_controller.get_ttm(ts, expire)

        dic['相对行权价'] = strike
        dic['ttm'] = ttm

        dic['call_value'] = call_value
        dic['put_value'] = put_value
        dic['call_time_value'] = call_time_value
        dic['put_time_value'] = put_time_value
        

        # greeks
        call_greeks = self.real_info_controller.cal_greeks(ts, targetCode, call, call_strike, expire, 'call')
        put_greeks  = self.real_info_controller.cal_greeks(ts, targetCode, put, put_strike, expire, 'put')

        dic['call_iv'] = call_greeks['iv']
        dic['put_iv'] = put_greeks['iv']

        dic['call_theta'] = call_greeks['theta']
        dic['put_theta'] = put_greeks['theta']
        dic['call_vega'] = call_greeks['vega']
        dic['put_vega'] = put_greeks['vega']
        dic['call_gamma'] = call_greeks['gamma']
        dic['put_gamma'] = put_greeks['gamma']
        dic['call_delta'] = call_greeks['delta']
        dic['put_delta'] = put_greeks['delta']
        dic['call_rho'] = call_greeks['rho']
        dic['put_rho'] = put_greeks['rho']

        return dic

    # 监督学习遍历(单个期权)
    def label_step(self, optionCode: str, ts: str, close: float, op_type: str, targetCode: str='510050', valid: bool=True):
        dic = {}

        dic['时间'] = ts[0: 4] + '-' + ts[4: 6] + '-' + ts[6: 8] + ' ' + ts[8: 10] + ':' + ts[10: 12] + ':' + ts[12: 14]

        strike = self.option_info_controller.get_strikePrice(optionCode)
        price = self.getClosePrice(optionCode, ts)

        # 内在价值
        if op_type == 'call':
            value = max(0, close - strike) / close
        elif op_type == 'put':
            value = max(0, strike - close) / close

        # 时间价值
        time_value = price / close - value

        # 行权价虚实
        relative_strike = strike / close

        # 到期日 (转化为年)
        expire = self.option_info_controller.get_expireDate(optionCode)
        ttm = self.real_info_controller.get_ttm(ts, expire)

        dic['相对行权价'] = relative_strike
        dic['ttm'] = ttm

        dic['内在价值'] = value
        dic['时间价值'] = time_value

        # greeks
        if valid:
            greeks = self.real_info_controller.cal_greeks(ts, targetCode, optionCode, strike, expire, op_type)
            dic['隐含波动率'] = greeks['iv']
            dic['Theta'] = greeks['theta']
            dic['Vega'] = greeks['vega']
            dic['Gamma'] = greeks['gamma']
            dic['Delta'] = greeks['delta']
            dic['Rho'] = greeks['rho']
            dic['greeks_valid'] = greeks['iv_valid']
        else:
            dic['隐含波动率'] = 0.0
            dic['Theta'] = 0.0
            dic['Vega'] = 0.0
            dic['Gamma'] = 0.0
            dic['Delta'] = 0.0
            dic['Rho'] = 0.0
            dic['greeks_valid'] = 0
        return dic
    
    def step(self, action: int, w: float, ts: str, close: float):
        """
        action: 0 HOLD, 1 LONG(买开跨式), 2 SHORT(卖开跨式), 3 CLOSE(按比例w平仓)
        w: 权重(0~1), 控制本组合可用资金使用比例或平仓比例
        idx: 组合编号
        ts: 时间串(YYYYmmddHHMMSS)
        close: 标的收盘价
        """
        ts = str(ts)
        for it in [' ', ':', '-']:
            ts = ts.replace(it, '')
        self.time_list.append(ts)


        # 策略动作
        assert action in (0, 1, 2, 3), 'action错误'
        call, put = self.comb['call'], self.comb['put']
        w = float(w)
        
        call_expire = self.option_info_controller.get_expireDate(call)
        put_expire = self.option_info_controller.get_expireDate(put)

        assert call_expire == put_expire, 'Error: 组合到期日不同!'

        if call_expire > ts[0: 8]:
            if action == 1:  # LONG(按组合资金计算可开手数)
                cap = self.free_money * max(0.0, min(1.0, w))
                q = self._pair_qty_buy_open(ts, desired=10**9, call=call, put=put, c_id=-1, free_override=cap)
                if q > 0:
                    pos_dir, _ = self.get_pair_position(call, put)
                    if pos_dir < 0:
                        self.flip_short_to_long(q, ts, call, put, -1)
                    else:
                        self.open_long_pair(q, ts, call, put, -1)

            elif action == 2:  # SHORT
                cap = self.free_money * max(0.0, min(1.0, w))
                q = self._pair_qty_sell_open(ts, desired=10**9, call=call, put=put, c_id=-1, free_override=cap)
                if q > 0:
                    pos_dir, _ = self.get_pair_position(call, put)
                    if pos_dir > 0:
                        self.flip_long_to_short(q, ts, call, put, -1)
                    else:
                        self.open_short_pair(q, ts, call, put, -1)

            elif action == 3:  # CLOSE
                self.close_pair(ts, call, put, w, -1)

        # 撮合与估值
        self.simulate_fill(ts)
        self.update_positions(ts)
        self.equity_list.append(self.equity)

        # 目标价格/收益
        if self.last_close is None:
            self.target_gain = 0.0
        else:
            self.target_gain = math.log(float(close) / (self.last_close if self.last_close != 0 else float(close)))
        self.target_gain_list.append(self.target_gain)

        self.last_close = float(close)
        self.target_price = float(close)


        # 更新组合持仓快照
        dic = self.comb
        pos_dir, pos_size = self.get_pair_position(call, put)
        dic['pos_dir'], dic['pos_size'] = pos_dir, pos_size
        try:
            dic['call_price'] = self.real_info_controller.get_close_by_str(call, ts)
            dic['put_price'] = self.real_info_controller.get_close_by_str(put, ts)
        except TypeError:
            dic['call_price'] = self.getClosePrice(call, ts)
            dic['put_price'] = self.getClosePrice(put, ts)

        call_expire = self.option_info_controller.get_expireDate(call)
        dic['call_ttm'] = self.real_info_controller.get_ttm(ts, call_expire)
        dic['call_strike'] = self.option_info_controller.get_strikePrice(call)

        put_expire = self.option_info_controller.get_expireDate(put)
        dic['put_ttm'] = self.real_info_controller.get_ttm(ts, put_expire)
        dic['put_strike'] = self.option_info_controller.get_strikePrice(put)


        current_state, history_state = self.get_total_state()
        reward = self.getReward(action)
        truncated = self.if_truncated()
        self.info = {"message": "ok"}
        return current_state, history_state, reward, truncated

    # ------------------ 批量K线回测 ------------------
    def handlebar(self, data: pd.DataFrame, func):
        first = True
        for row in data.itertuples(index=False):
            time_str, stock_close = str(row.ts), float(row.close)
            for it in [' ', ':', '-']:
                time_str = time_str.replace(it, '')

            if first:
                self.update_positions(time_str)
                first = False

            # 策略函数(内部会调用 submit_order)
            func(time_str, stock_close)

            # 撮合 + 更新
            self.simulate_fill(time_str)
            self.update_positions(time_str)
            self.equity_list.append(self.equity)

    # 一个最简单的策略示例(随时可替换)
    def strategy(self, time_str: str, stock_close: float):
        # 仅示例：对第0个组合买开1手
        if 0 in self.comb:
            call = self.comb[0]['call']
            put = self.comb[0]['put']
            self.open_long_pair(1, time_str, call, put, c_id=0)

    def main(self, start_str: str, end_str: str, benchmark: str = '510050', period: str = ''):
        period = period or self.period
        data = self.real_info_controller.get_bars_between_from_df(benchmark, start_str, end_str, period)
        self.handlebar(data, self.strategy)
        self.out_excel()

    # ------------------ 内部同步 ------------------

    def _sync_global_from_subaccounts(self):
        """将全局 free/frozen/equity 与子账户求和对齐"""
        self.free_money = sum(v['free_money'] for v in self.comb_info.values()) if self.comb_info else self.free_money
        self.frozen_money = sum(v['frozen_money'] for v in self.comb_info.values()) if self.comb_info else self.frozen_money
        # 注意：equity 在 _update_comb_equity 中会按持仓估值更新


def gen_label_data(target: str='510050'):
    account = single_Account(100000, fee=1.3, period='30m', stockList=[target])
    # option_list = account.option_info_controller.get_option_list(target)
    option_list = account.get_option_list(target, '202412', 'call')
    option_list.extend(account.get_option_list(target, '202412', 'put'))

    # 统计什么样的相对行权价会导致iv无法计算
    min_more_1, max_more_1 = None, None
    min_less_1, max_less_1 = None, None

    total_length = len(option_list)

    for index, optionCode in enumerate(option_list):
        start_time = account.option_info_controller.get_openDate(optionCode)
        end_time = account.option_info_controller.get_expireDate(optionCode)

        avg_volume = account.real_info_controller.get_avg_volume(optionCode)

        # 平均交易量太小, 认为无流动性
        if avg_volume <= 150:
            continue

        op_type = account.option_info_controller.get_optionType(optionCode)
        multiplier = account.option_info_controller.get_multiplier(optionCode)
        strike = account.option_info_controller.get_strikePrice(optionCode)

        start_close = account.real_info_controller.get_close_by_str(target, start_time, period='1d')

        if strike / start_close < 0.88 or strike / start_close > 1.12:
            continue
        
        if multiplier != 10000:
            continue
        
        # 取样本数据(你的 RealInfo 里方法名可能是 get_bars_between 或 get_bars_between_from_df)
        try:
            data = account.real_info_controller.get_bars_between(target, start_time, end_time, '30m')
        except AttributeError:
            data = account.real_info_controller.get_bars_between_from_df(target, start_time, end_time, '30m')

        filename = f'./miniQMT/datasets/label_train_data/{optionCode}_{target}.xlsx'

        result_list = []
        for idx in range(len(data)):

            close = float(data.iloc[idx].close)
            ts = str(data.iloc[idx].ts).replace(' ', '').replace('-', '').replace(':', '')

            result = account.label_step(optionCode, ts, close, op_type, target)

            # ttm大于15天
            if result and result['ttm'] >= 15 / 365:
                result_list.append(result)
                if result['greeks_valid'] == 0:
                    strike = result['相对行权价']
                    if strike <= 1:
                        min_less_1 = strike if min_less_1 is None or min_less_1 > strike else min_less_1
                        max_less_1 = strike if max_less_1 is None or max_less_1 < strike else max_less_1
                    else:
                        min_more_1 = strike if min_more_1 is None or min_more_1 > strike else min_more_1
                        max_more_1 = strike if max_more_1 is None or max_more_1 < strike else max_more_1

                    print(f"[Warning] 期权{optionCode}存在iv = 0 | idx: {idx}")
            else:
                break
        
        df = pd.DataFrame(result_list)
        df.to_excel(filename, index=False, sheet_name='期权信息')

        print(f"[Info] 进度: {index + 1} / {total_length} | optionCode = {optionCode} | ({min_less_1, max_less_1}) | ({min_more_1, max_more_1})")    
    print(f"[Info] 结束 ({min_less_1, max_less_1}) | ({min_more_1, max_more_1})")

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

    # 取样本数据(你的 RealInfo 里方法名可能是 get_bars_between 或 get_bars_between_from_df)
    try:
        data = account.real_info_controller.get_bars_between(target, start_time, end_time, '30m')
    except AttributeError:
        data = account.real_info_controller.get_bars_between_from_df(target, start_time, end_time, '30m')

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
        
    print(0 / 0)

    # account.out_excel()
