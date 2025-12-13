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

import matplotlib.pyplot as plt

# 引入包
try:
    from finTool.optionBaseInfo import optionBaseInfo
    from finTool.realInfo import RealInfo
except Exception as e:
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

class windowAccount:
    """
    账户端：支持多组合的跨式(call, put)回测账户(含组合级子账户 comb_info)
    用法：
        acc = windowAccount(100000, fee=1.3, period='30m', stockList=['510050'])
        acc.set_combos([(call1, put1), (call2, put2), ...])  # 至少一个
        ...
        state, reward, truncated = acc.step(action, w, idx, ts, close)
    """

    # ------------------ 初始化与结构 ------------------

    def __init__(self,
                 init_capital: float,
                 fee: float = 1.3,
                 period: str = '30m',
                 stockList: Optional[List[str]] = None,
                 filepath: str = './miniQMT/datasets/',
                 window: int=32,
                 label_pairs: tuple=None,
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

        # 组合期权信息与资金子账户
        self.comb: Dict[int, Dict] = {}        # 每个组合的期权/持仓快照
        self.comb_info: Dict[int, Dict] = {}   # 每个组合的资金子账户
        self.code2cid: Dict[str, int] = {}     # 期权代码 -> 组合ID
        self.num_option = 0  # 组合个数

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
            'target_gain': self.target_gain,
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

        # 保存状态供监督学习 (call, put)
        self.label_pairs = label_pairs

    # 注册组合(跨式)：传入[(call, put), ...],并按等权分配初始资金
    def set_combos(self, pairs: List[Tuple[str, str]]):
        self.num_option = len(pairs)
        self.comb.clear()
        self.comb_info.clear()
        self.code2cid.clear()

        sub_capital = self.init_capital / max(1, self.num_option)

        for idx, pair in enumerate(pairs):
            # 组合期权信息
            call, put = pair

            # 注意, 这里存的都是不归一化的信息, 归一化需要在获取状态的时候做
            self.comb[idx] = {
                'call': call, 'put': put,
                'call_price': None, 'put_price': None,
                'call_strike': None, 'put_strike': None,
                'call_ttm': None, 'put_ttm': None,
                'call_iv': None, 'put_iv': None,
                'call_theta': None, 'put_theta': None,
                'call_vega': None, 'put_vega': None,
                'call_gamma': None, 'put_gamma': None,
                'call_delta': None, 'put_delta': None,
                'call_rho': None, 'put_rho': None,
                'pos_dir': 0, 'pos_size': 0,
                'call_real_value': None, 'call_time_value': None,
                'put_real_value': None, 'put_time_value': None,
            }

            # 组合资金子账户
            self.comb_info[idx] = {
                'init_capital': sub_capital,
                'free_money': sub_capital,
                'frozen_money': 0.0,
                'equity': sub_capital,
                'last_equity': sub_capital
            }

            # 代码 -> 组合ID
            self.code2cid[call] = idx
            self.code2cid[put] = idx
        # 全局资金与子账户对齐
        self._sync_global_from_subaccounts()

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

        free_money = self.comb_info[c_id]['free_money'] if free_override is None else float(free_override)
        cap_cash = int(free_money // per_cost) if per_cost > 0 else desired
        vol_cap = min(self.getRealVolume(call, ts), self.getRealVolume(put, ts))
        return max(0, min(int(desired), cap_cash, vol_cap))

    def _pair_qty_sell_open(self, ts: str, desired: int, call: str, put: str,
                            c_id: int, free_override: Optional[float] = None) -> int:
        """跨式卖开时, 两腿共同可成交的手数 q(按组合 free_money 的保证金能力)"""
        m_c = self.getMargin(call)
        m_p = self.getMargin(put)
        per_margin = m_c + m_p

        free_money = self.comb_info[c_id]['free_money'] if free_override is None else float(free_override)
        cap_cash = int(free_money // per_margin) if per_margin > 0 else desired
        vol_cap = min(self.getRealVolume(call, ts), self.getRealVolume(put, ts))
        return max(0, min(int(desired), cap_cash, vol_cap))

    def open_long_pair(self, desired: int, ts: str, call: str, put: str, c_id: Optional[int] = None) -> int:
        assert c_id is not None and c_id in self.comb_info, "open_long_pair 需要有效 c_id"
        q = self._pair_qty_buy_open(ts, desired, call, put, c_id)
        if q <= 0:
            return 0
        self.submit_order(call, '买入开仓', q, ts, c_id=c_id)
        self.submit_order(put,  '买入开仓', q, ts, c_id=c_id)
        return q

    def open_short_pair(self, desired: int, ts: str, call: str, put: str, c_id: Optional[int] = None) -> int:
        assert c_id is not None and c_id in self.comb_info, "open_short_pair 需要有效 c_id"
        q = self._pair_qty_sell_open(ts, desired, call, put, c_id)
        if q <= 0:
            return 0
        self.submit_order(call, '卖出开仓', q, ts, c_id=c_id)
        self.submit_order(put,  '卖出开仓', q, ts, c_id=c_id)
        return q

    def close_pair(self, ts: str, call: str, put: str, w: float = None, c_id: Optional[int] = None) -> None:
        """按权重 w 平掉两腿(w∈[0,1])"""
        w = 1.0 if (w is None) else float(w)
        if w <= 0:
            return
        assert c_id is not None and c_id in self.comb_info, "close_pair 需要有效 c_id"

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

    def flip_short_to_long(self, target: int, ts: str, call: str, put: str, c_id: Optional[int] = None) -> None:
        """从空翻多：先平空,再买开补足到 target"""
        assert c_id is not None and c_id in self.comb_info
        pos_dir, pos_size = self.get_pair_position(call, put)
        v = min(target, pos_size) if pos_dir < 0 else 0
        if v > 0:
            self.submit_order(call, '买入平仓', v, ts, c_id=c_id)
            self.submit_order(put,  '买入平仓', v, ts, c_id=c_id)
        l = target - v
        if l > 0:
            self.open_long_pair(l, ts, call, put, c_id)

    def flip_long_to_short(self, target: int, ts: str, call: str, put: str, c_id: Optional[int] = None) -> None:
        """从多翻空：先平多,再卖开补足到 target"""
        assert c_id is not None and c_id in self.comb_info
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
        """更新全局资金与positions；dispose_volume>0视为买方向(多开/买平), <0视为卖方向(空开/卖平)"""
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

    def dispose_single_comb(self, c_id: int, free_delta: float, frozen_delta: float):
        """更新单个组合的资金变动"""
        if c_id not in self.comb_info:
            return
        info = self.comb_info[c_id]
        info['free_money'] += float(free_delta)
        info['frozen_money'] += float(frozen_delta)
        # equity 在 update_comb_equity 中按最新市值统一刷新

    def _update_position_values(self, time_str: str):
        """重估所有持仓的 value 字段(不做资金变动)"""
        for code, (direction, volume, _) in list(self.positions.items()):
            price = self.getClosePrice(code, time_str)
            mul = self.option_info_controller.get_multiplier(code)
            self.positions[code] = (direction, volume, price * volume * mul)

    def _update_comb_equity(self):
        """按当前市值更新各组合 equity；并把全局资金用子账户汇总同步回写"""
        # 1) 先把每个组合的持仓市值(多为正,空为负)算出来
        comb_pos_value: Dict[int, float] = {i: 0.0 for i in self.comb_info.keys()}

        for code, (direction, volume, value) in self.positions.items():
            c_id = self.code2cid.get(code, -1)
            if c_id not in comb_pos_value:
                continue
            signed_val = value if direction == '买入开仓' else -value
            comb_pos_value[c_id] += signed_val

        # 2) 刷新组合 equity
        for c_id, info in self.comb_info.items():
            info['equity'] = info.get('equity', info['init_capital'])
            pos_val = comb_pos_value.get(c_id, 0.0)
            info['equity'] = info['free_money'] + info['frozen_money'] + pos_val

        # 3) 汇总全局
        self.free_money = sum(v['free_money'] for v in self.comb_info.values())
        self.frozen_money = sum(v['frozen_money'] for v in self.comb_info.values())
        last_equity = self.equity
        self.equity = sum(v['equity'] for v in self.comb_info.values())

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
            c_id = self.code2cid.get(code, -1)

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
                self.dispose_single_comb(c_id, free_delta, frozen_delta)

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
                self.dispose_single_comb(c_id, free_delta, frozen_delta)

            delete_list.append(code)

        for code in delete_list:
            self.positions.pop(code, None)

        # 重估持仓价值
        self._update_position_values(time_str)
        # 刷新每个组合 equity,并同步回全局
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

            # 组合子账户
            free_sub = self.comb_info.get(c_id, {'free_money': 0.0})['free_money']

            # ----- 期权撮合 -----
            if len(code) == 8:
                if direction == '买入开仓':
                    # 以组合 free_money 为准
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
                    # 更新仓位与资金(全局)
                    self.dispose_order(code, +max_cnt, price, free_delta, frozen_delta)
                    # 同步到组合
                    self.dispose_single_comb(c_id, free_delta, frozen_delta)

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
                    self.dispose_single_comb(c_id, free_delta, frozen_delta)

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
                    self.dispose_single_comb(c_id, free_delta, frozen_delta)

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
                    self.dispose_single_comb(c_id, free_delta, frozen_delta)

                    trade = Trade(order_id, code, direction, price * max_cnt * mul, 0.0, time_str, max_cnt)
                    self.Trades.append(trade)

            else:
                # 股票暂不支持
                order.status = '废单'
                order.info = '目前暂不交易股票'

        # 每次撮合后,统一重估并同步子账户->全局
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

        for idx, dic in self.comb.items():
            call = dic['call']
            put = dic['put']
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

    def getState(self, idx: Optional[int] = None):
        """返回当前状态(dict). idx为组合索引 | None则返回全局+各组合概览."""

        gs = {
            'target_gain': self.target_gain,
            'cash_ratio': self.cash_ratio,
            'margin_ratio': self.margin_ratio,
            'draw_down': 0.0 if self.equity_peak <= 0 else (self.equity_peak - self.equity) / self.equity_peak,
            'max_equity': self.equity_peak / self.init_capital,
        }

        if idx is None:
            overview = {}
            for k, dic in self.comb.items():
                call, put = dic['call'], dic['put']
                pos_dir, pos_size = self.get_pair_position(call, put)
                info = self.comb_info[k]
                overview[k] = {
                    'call': call, 'put': put,
                    'pos_dir': pos_dir, 'pos_size': pos_size,
                    'call_price': dic.get('call_price'), 'put_price': dic.get('put_price'),
                    'call_ttm': dic.get('call_ttm'), 'call_strike': dic.get('call_strike'),
                    'put_ttm': dic.get('put_ttm'), 'put_strike': dic.get('put_strike'),
                    'sub_free': info['free_money'], 'sub_frozen': info['frozen_money'],
                    'sub_equity': info['equity'], 'sub_last_equity': info['equity'],
                }
            return {'global': gs, 'combos': overview}

        dic = self.comb[idx]
        call, put = dic['call'], dic['put']
        pos_dir, pos_size = self.get_pair_position(call, put)
        info = self.comb_info[idx]
        local = {
            'call': call, 'put': put,
            'pos_dir': pos_dir, 'pos_size': pos_size,
            'call_price': dic.get('call_price'), 'put_price': dic.get('put_price'),
            'ttm': dic.get('ttm'), 'strike': dic.get('strike'),
            'target_price': self.target_price,
            'sub_free': info['free_money'], 'sub_frozen': info['frozen_money'],
            'sub_equity': info['equity'], 'sub_last_equity': info['equity'],
        }

        state = []
        for _, v in gs.items():
            state.append(v)
        for _, v in local.items():
            state.append(v)
        # return {'global': gs, 'combo': local, 'idx': idx}
        return state

    def get_global_state(self):
        gs = {
            'target_gain': self.target_gain,
            'cash_ratio': self.cash_ratio if abs(self.cash_ratio) > self.eps else 0,
            'margin_ratio': self.margin_ratio if abs(self.margin_ratio) > self.eps else 0,
            'draw_down': 0.0 if self.equity_peak <= 0 else (self.equity_peak - self.equity) / self.equity_peak,
            'max_equity': self.equity_peak / self.init_capital,
        }
        
        state = []
        for _, v in gs.items():
            state.append(v)
        
        return state

    # 直接补充到self.window_size
    def get_history_state(self):
        states = []
        for idx in range(self.num_option):
            first = None
            single = []
            for item in self.h_states:
                single.append(item[idx])

                if first is None:
                    first = item[idx]
                    
                    for _ in range(self.window_size - len(self.h_states)):
                        single.append(first)
            
            states.append(single)

        return states

    def get_total_state(self):
        state = []

        gs = {
            'target_gain': self.target_gain,
            'cash_ratio': self.cash_ratio if abs(self.cash_ratio) > self.eps else 0,
            'margin_ratio': self.margin_ratio if abs(self.margin_ratio) > self.eps else 0,
            'draw_down': 0.0 if self.equity_peak <= 0 else (self.equity_peak - self.equity) / self.equity_peak,
            'max_equity': self.equity_peak / self.init_capital,
        }

        for idx in range(self.num_option):
            single = []
            for _, v in gs.items():
                single.append(v)
            
            comb = self.comb[idx]

            # 子账户信息(包括期权交易信息)
            comb_info = self.comb_info[idx]

            single.append(comb['pos_dir'])
            single.append(comb['pos_size'])
            single.append(comb_info['free_money'] / comb_info['init_capital'])
            single.append(comb_info['frozen_money'] / comb_info['init_capital'])
            single.append(comb_info['equity'] / comb_info['init_capital'])


            # 期权实际信息
            # single.append(comb['call_price'] / self.target_price)
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
            single.append(comb['call_iv'])
            single.append(comb['call_theta'])
            single.append(comb['call_vega'])
            single.append(comb['call_gamma'])
            single.append(comb['call_delta'])
            single.append(comb['call_rho'])
            

            # single.append(comb['put_price'] / close)
            single.append(comb['put_strike'] / close)
            single.append(comb['put_ttm'])
            single.append(comb['put_real_value'] / close)
            single.append(comb['put_time_value'] / close)
            single.append(comb['put_iv'])
            single.append(comb['put_theta'])
            single.append(comb['put_vega'])
            single.append(comb['put_gamma'])
            single.append(comb['put_delta'])
            single.append(comb['put_rho'])
            
            state.append(single)
        
        self.h_states.append(state)
        return state
    

    def get_window_state(self):
        states = []
        for idx in range(self.num_option):
            for item in self.h_states:
                states.append(item[idx])
            
        return states

    def getInfo(self):
        return self.info

    def has_positions(self):
        for idx in self.comb:
            if self.comb[idx]['pos_size'] != 0:
                return True
        return False

    def getReward(self, eps: float=1e-6):
        # 1) step_ret 不变
        if len(self.equity_list) <= 1:
            step_ret = 0.0
        else:
            prev, cur = self.equity_list[-2], self.equity_list[-1]
            step_ret = np.log((cur + eps) / (prev + eps))
        
        return float(np.clip(step_ret, -0.1, 0.1))

        # 2) dd 罚 (0.1, 只有仓)
        dd_penalty = 0.0
        if self.has_positions():
            dd = max(0.0, (self.equity_peak - self.equity) / (self.equity_peak + eps))
            dd_penalty = 0.1 * dd

        # 5) 组合
        raw = step_ret - dd_penalty
        return raw
    
    def old_getReward(self, eps: float = 1e-6):
        # 基础 log-return(非累加)
        a = self.equity if self.equity > 0 else eps
        b = self.last_equity if self.last_equity > 0 else eps
        r_base = math.log(a / b) - self.target_gain
        self.last_equity = self.equity
        # return r_base
    
        if not self.use_penalties:
            return float(np.clip(r_base, -1, 1))

        # 回撤惩罚
        peak = self.equity_peak if self.equity_peak > eps else eps
        dd = (peak - self.equity) / peak
        pen_dd = self.lam_dd * (max(0.0, dd - self.dd_floor) ** 2)

        # 保证金占比惩罚
        pen_mr = self.lam_mr * (max(0.0, self.margin_ratio - self.mr_floor) ** 2)

        # 交易频次惩罚
        new_trades = max(0, len(self.Trades) - self.last_trade_cnt)
        self.last_trade_cnt = len(self.Trades)
        pen_trade_cnt = self.lam_trade_cnt * max(0, new_trades - self.free_trades_per_step)

        r = r_base - (pen_dd + pen_mr + pen_trade_cnt)
        return float(np.clip(r, -0.10, 0.10))

    def if_truncated(self) -> bool:
        return (self.equity / self.init_capital) < 0.05


    @deprecated('暂时弃用~', '0.0')
    def single_step(self, action: int, w: float, idx: int, ts: str, close: float):
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
        dic = self.comb[idx]
        call, put = dic['call'], dic['put']
        w = float(w)
        
        call_expire = self.option_info_controller.get_expireDate(call)
        put_expire = self.option_info_controller.get_expireDate(put)
        assert call_expire == put_expire, '组合到期日不同!'

        if call_expire > ts[0: 8]:
            if action == 1:  # LONG(按组合资金计算可开手数)
                cap = self.comb_info[idx]['free_money'] * max(0.0, min(1.0, w))
                q = self._pair_qty_buy_open(ts, desired=10**9, call=call, put=put, c_id=idx, free_override=cap)
                if q > 0:
                    pos_dir, _ = self.get_pair_position(call, put)
                    if pos_dir < 0:
                        self.flip_short_to_long(q, ts, call, put, idx)
                    else:
                        self.open_long_pair(q, ts, call, put, idx)

            elif action == 2:  # SHORT
                cap = self.comb_info[idx]['free_money'] * max(0.0, min(1.0, w))
                q = self._pair_qty_sell_open(ts, desired=10**9, call=call, put=put, c_id=idx, free_override=cap)
                if q > 0:
                    pos_dir, _ = self.get_pair_position(call, put)
                    if pos_dir > 0:
                        self.flip_long_to_short(q, ts, call, put, idx)
                    else:
                        self.open_short_pair(q, ts, call, put, idx)

            elif action == 3:  # CLOSE
                self.close_pair(ts, call, put, w, idx)

        # 撮合与估值
        self.simulate_fill(ts)
        self.update_positions(ts)
        self.equity_list.append(self.equity)

        # 目标价格/收益
        if self.last_close is None:
            self.target_gain = 0.0
        else:
            self.target_gain = math.log(float(close) / (self.last_close if self.last_close != 0 else float(close)))
        self.last_close = float(close)
        self.target_price = float(close)

        # 更新组合持仓快照
        pos_dir, pos_size = self.get_pair_position(call, put)
        dic['pos_dir'], dic['pos_size'] = pos_dir, pos_size
        try:
            dic['call_price'] = self.real_info_controller.get_close_by_str(call, ts)
            dic['put_price'] = self.real_info_controller.get_close_by_str(put, ts)
        except TypeError:
            dic['call_price'] = self.getClosePrice(call, ts)
            dic['put_price'] = self.getClosePrice(put, ts)
        expire = self.option_info_controller.get_expireDate(call)
        dic['ttm'] = self.real_info_controller.get_ttm(ts, expire)
        dic['strike'] = self.option_info_controller.get_strikePrice(call)

        # 状态/奖励/截断
        self.get_total_state()
        state = self.get_history_state()
        reward = self.getReward()
        truncated = self.if_truncated()
        self.info = {"message": "ok"}
        return state, reward, truncated
    
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
    def label_step(self, optionCode: str, ts: str, close: float, op_type: str, targetCode: str='510050', last_close: float=0, valid: bool=True, hv_160: float=0.0):
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

        dic['对数收益率'] = 0 if last_close == 0.0 else np.log(close / (last_close + 1e-8))
        dic['HV160'] = hv_160

        # dic['HV20'] = self.real_info_controller.get_history_iv(ts, targetCode)

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
    
    def step(self, action_list: int, w_list: list, ts: str, close: float):
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

        assert len(w_list) == len(action_list), 'w, action LIST长度不一样!'

        # TEST
        if action_list[0] == 0:
            self.record.append('HOLD')
        elif action_list[0] == 1:
            self.record.append(f'做多: {w_list[0]}')
        elif action_list[0] == 2:
            self.record.append(f'做空: {w_list[0]}')
        elif action_list[0] == 3:
            self.record.append(f'平仓: {w_list[0]}')

        # 遍历所有(action, weight, idx)
        for idx, (action, w) in enumerate(zip(action_list, w_list)):

            # 策略动作
            assert action in (0, 1, 2, 3), 'action错误'
            dic = self.comb[idx]
            call, put = dic['call'], dic['put']
            w = float(w)
            
            call_expire = self.option_info_controller.get_expireDate(call)
            put_expire = self.option_info_controller.get_expireDate(put)
            assert call_expire == put_expire, '组合到期日不同!'

            if call_expire > ts[0: 8]:
                if action == 1:  # LONG(按组合资金计算可开手数)
                    cap = self.comb_info[idx]['free_money'] * max(0.0, min(1.0, w))
                    q = self._pair_qty_buy_open(ts, desired=10**9, call=call, put=put, c_id=idx, free_override=cap)
                    if q > 0:
                        pos_dir, _ = self.get_pair_position(call, put)
                        if pos_dir < 0:
                            self.flip_short_to_long(q, ts, call, put, idx)
                        else:
                            self.open_long_pair(q, ts, call, put, idx)

                elif action == 2:  # SHORT
                    cap = self.comb_info[idx]['free_money'] * max(0.0, min(1.0, w))
                    q = self._pair_qty_sell_open(ts, desired=10**9, call=call, put=put, c_id=idx, free_override=cap)
                    if q > 0:
                        pos_dir, _ = self.get_pair_position(call, put)
                        if pos_dir > 0:
                            self.flip_long_to_short(q, ts, call, put, idx)
                        else:
                            self.open_short_pair(q, ts, call, put, idx)

                elif action == 3:  # CLOSE
                    self.close_pair(ts, call, put, w, idx)

        # 撮合与估值
        self.simulate_fill(ts)
        self.update_positions(ts)
        self.equity_list.append(self.equity)

        # 目标价格/收益
        if self.last_close is None:
            self.target_gain = 0.0
        else:
            self.target_gain = math.log(float(close) / (self.last_close if self.last_close != 0 else float(close)))
        
        self.last_close = float(close)
        self.target_price = float(close)

        for idx in range(self.num_option):
            dic = self.comb[idx]
            # 更新组合持仓快照
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


        # 状态/奖励/截断
        react_state = self.get_total_state()
        state = self.get_history_state()
        reward = self.getReward()
        truncated = self.if_truncated()
        self.info = {"message": "ok"}
        return react_state, state, reward, truncated

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

def draw(target: str='510050'):
    start_time = '20240425'
    end_time = '20250510'
    account = windowAccount(100000, fee=1.3, period='30m', stockList=[target])
    data = account.real_info_controller.get_bars_between(target, start_time, end_time, '1d')

    h_iv_list = []
    idx_list = []


    for idx in range(len(data)):
        # close = float(data.iloc[idx].close)
        ts = str(data.iloc[idx].ts).replace(' ', '').replace('-', '').replace(':', '')
        h_iv_list.append(account.real_info_controller.get_history_iv(ts, target))
        idx_list.append(idx + 1)


    plt.figure(figsize=(10, 6))  # 设置图表大小

    # 2. 绘制折线图
    # X轴使用 idx_list，Y轴使用 h_iv_list
    plt.plot(idx_list, h_iv_list, 
            marker='o',         # 在每个数据点上显示一个圆圈标记
            linestyle='-',      # 使用实线连接数据点
            color='skyblue',    # 设置线条颜色
            label='Historical Volatility (HV)') # 图例标签

    # 3. 添加图表元素
    plt.title('Historical Volatility Over Time', fontsize=16) # 设置图表标题
    plt.xlabel('Index / Observation Point', fontsize=12)      # 设置 X 轴标签
    plt.ylabel('Historical Volatility (HV)', fontsize=12)     # 设置 Y 轴标签
    plt.grid(True, linestyle='--', alpha=0.7)                # 添加网格线

    # 4. 添加图例和调整布局
    plt.legend()                                             # 显示图例
    plt.tight_layout()                                       # 自动调整子图参数，使之填充整个图像区域

    # 5. 显示图表
    plt.show()

import os
import glob

def get_excel_files(file_path):
    # 获取所有 .xlsx 文件
    files_xlsx = glob.glob(os.path.join(file_path, "*.xlsx"))
    # 获取所有 .xls 文件
    files_xls = glob.glob(os.path.join(file_path, "*.xls"))

    files = files_xlsx + files_xls

    # 过滤掉隐藏文件（以 . 开头）
    files = [
        f for f in files
        if not os.path.basename(f).startswith(".")
           and os.path.isfile(f)
    ]
    
    return files

def get_exist_option_list(file_path: str='./miniQMT/datasets/test_label_train_data'):
    files = get_excel_files(file_path)
    option_list = []
    for f in files:
        if not f.endswith('_510050.xlsx'):
            continue
        first = f.find('_510050.xlsx')

        option = f[first - 8: first]
        if len(option) == 8:
            option_list.append(option)  
    
    return option_list



def gen_label_data(target: str='510050'):
    account = windowAccount(100000, fee=1.3, period='30m', stockList=[target])
    # option_list = account.option_info_controller.get_option_list(target)

    option_list = account.get_option_list(target, '202412', 'call')
    option_list.extend(account.get_option_list(target, '202412', 'put'))

    option_list.extend(account.get_option_list(target, '202411', 'call'))
    option_list.extend(account.get_option_list(target, '202411', 'put'))


    option_list.extend(account.get_option_list(target, '202410', 'call'))
    option_list.extend(account.get_option_list(target, '202410', 'put'))
    
    option_list.extend(account.get_option_list(target, '202409', 'call'))
    option_list.extend(account.get_option_list(target, '202409', 'put'))

    option_list.extend(account.get_option_list(target, '202408', 'call'))
    option_list.extend(account.get_option_list(target, '202408', 'put'))


    option_list.extend(account.get_option_list(target, '202407', 'call'))
    option_list.extend(account.get_option_list(target, '202407', 'put'))

    option_list.extend(account.get_option_list(target, '202406', 'call'))
    option_list.extend(account.get_option_list(target, '202406', 'put'))

    

    # ---- 已经存在的 ----
    # option_list = account.get_option_list(target, '202410', 'call')
    # option_list.extend(account.get_option_list(target, '202410', 'put'))

    option_list.extend(account.get_option_list(target, '202509', 'call'))
    option_list.extend(account.get_option_list(target, '202509', 'put'))

    option_list.extend(account.get_option_list(target, '202508', 'call'))
    option_list.extend(account.get_option_list(target, '202508', 'put'))

    option_list.extend(account.get_option_list(target, '202507', 'call'))
    option_list.extend(account.get_option_list(target, '202507', 'put'))

    option_list.extend(account.get_option_list(target, '202506', 'call'))
    option_list.extend(account.get_option_list(target, '202506', 'put'))

    option_list.extend(account.get_option_list(target, '202505', 'call'))
    option_list.extend(account.get_option_list(target, '202505', 'put'))

    option_list.extend(account.get_option_list(target, '202504', 'call'))
    option_list.extend(account.get_option_list(target, '202504', 'put'))

    option_list.extend(account.get_option_list(target, '202503', 'call'))
    option_list.extend(account.get_option_list(target, '202503', 'put'))

    option_list.extend(account.get_option_list(target, '202502', 'call'))
    option_list.extend(account.get_option_list(target, '202502', 'put'))

    option_list.extend(account.get_option_list(target, '202501', 'call'))
    option_list.extend(account.get_option_list(target, '202501', 'put'))


    lis = get_exist_option_list(file_path='./miniQMT/datasets/test_label_train_data')
    lis.extend(get_exist_option_list(file_path='./miniQMT/datasets/label_train_data'))

    new_lis = []
    for op in lis:
        if op in option_list:
            new_lis.append(op)

    pairs = []

    for call in new_lis:
        if account.option_info_controller.get_optionType(call) != 'call':
            continue

        call_strike = account.option_info_controller.get_strikePrice(call)
        call_expire = account.option_info_controller.get_expireDate(call)
        call_open = account.option_info_controller.get_openDate(call)

        for put in new_lis:
            if account.option_info_controller.get_optionType(put) != 'put':
                continue

            put_strike = account.option_info_controller.get_strikePrice(put)
            put_expire = account.option_info_controller.get_expireDate(put)
            put_open = account.option_info_controller.get_openDate(put)

            if call_strike == put_strike and call_expire == put_expire:
                pairs.append(
                    {
                        'call': call,
                        'call_strike': call_strike,
                        'call_expire': call_expire,
                        'put': put,
                        'put_strike': put_strike,
                        'put_expire': put_expire,
                        'call_open': call_open,
                        'put_open': put_open
                    }
                )
                break
    df = pd.DataFrame(pairs)
    df.to_excel('./miniQMT/datasets/all_label_data/20251213_train.xlsx')

    
    print(0 / 0)

    # option_list.extend(account.get_option_list(target, '202511', 'call'))
    # option_list.extend(account.get_option_list(target, '202511', 'put'))

    # option_list.extend(account.get_option_list(target, '202512', 'call'))
    # option_list.extend(account.get_option_list(target, '202512', 'put'))

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

        filename = f'./miniQMT/datasets/test_label_train_data/{optionCode}_{target}.xlsx'

        result_list = []
        last_close = 0

        before_str = account.real_info_controller.get_prev_30_days(start_time, days=100)
        before_str = before_str + start_time[8: ]
        hv_data = account.real_info_controller.get_bars_between(target, before_str, end_time, '30m')
        hv_data = hv_data[['ts', 'close']]
        hv_data['close_prev'] = hv_data['close'].shift(1)
        hv_data['log_diff'] = np.log(hv_data['close'] / hv_data['close_prev'])
        window_size = 160
        hv_data['rolling_std_160'] = hv_data['log_diff'].rolling(window=window_size).std() * np.sqrt(2016)
        hv_data = hv_data.set_index('ts')


        for idx in range(len(data)):
            ts = data.iloc[idx].ts
            close = float(data.iloc[idx].close)
            std_dev = hv_data.loc[ts, 'rolling_std_160']
            
            ts = str(ts)
            ts = ts.replace(' ', '').replace('-', '').replace(':', '')

            result = account.label_step(optionCode, ts, close, op_type, target, last_close=last_close, hv_160=std_dev)
            last_close = close

            # ttm大于15天
            if result and result['ttm'] >= 15 / 365:
                if op_type == 'call':
                    result['op_type'] = 1
                elif op_type == 'put':
                    result['op_type'] = -1
                else:
                    result['op_type'] = 0
                
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
        
        if len(result_list) >= 128:
            df = pd.DataFrame(result_list)
            df.to_excel(filename, index=False, sheet_name='期权信息')
        else:
            print(f"[Info] 期权{optionCode}信息太少, 排除, len = {len(result_list)}")

        print(f"[Info] 进度: {index + 1} / {total_length} | optionCode = {optionCode} | ({min_less_1, max_less_1}) | ({min_more_1, max_more_1})")    
    print(f"[Info] 结束 ({min_less_1, max_less_1}) | ({min_more_1, max_more_1})")


# gen_label_data()

# ========================== 用例 ==========================
if __name__ == '__main_':
    # 示例：单组合跨式 + 逐步调用step
    start_time = '20250825100000'
    # start_time = '20251025100000'
    # start_time = '20250923143000'
    end_time = '20250924150000'
    # end_time = '20251125150000'

    calls, puts = [], []

    call = '10008800'
    put = '10008809'
    calls.append(call)
    puts.append(put)

    call = '10009039'
    put = '10009040'
    calls.append(call)
    puts.append(put)

    option_pairs = []
    option_pairs.append({'call': '10008800','put': '10008809'})
    option_pairs.append({'call': '10008793','put': '10008802'})
    option_pairs.append({'call': '10008798','put': '10008807'})
    option_pairs.append({'call': '10008795','put': '10008804'})
    option_pairs.append({'call': '10008794','put': '10008803'})
    option_pairs.append({'call': '10008905','put': '10008906'})
    option_pairs.append({'call': '10009811','put': '10009812'})
    option_pairs.append({'call': '10009495','put': '10009496'})
    option_pairs.append({'call': '10009039','put': '10009040'})
    option_pairs.append({'call': '10008797','put': '10008806'})


    account = windowAccount(100000, fee=1.3, period='30m', stockList=['510050', '588000'])
    account.set_combos([(calls[0], puts[0]), (calls[1], puts[1])])

    target = '588000'
    target = '510050'

    # 取样本数据(你的 RealInfo 里方法名可能是 get_bars_between 或 get_bars_between_from_df)
    try:
        data = account.real_info_controller.get_bars_between(target, start_time, end_time, '30m')
    except AttributeError:
        data = account.real_info_controller.get_bars_between_from_df(target, start_time, end_time, '30m')
    

    call = option_pairs[1]['call']
    put = option_pairs[1]['put']

    # call = '10008893'
    # put = '10008894'

    # call, put = '10009309', '10009310'
    call, put = '10009495', '10009496'

    for i in range(len(option_pairs)):
        # call, put = option_pairs[i]['call'], option_pairs[i]['put']

        account.label_pairs = (call, put)
        result_list = []
        for idx in range(len(data)):
            close = float(data.iloc[idx].close)
            ts = str(data.iloc[idx].ts).replace(' ', '').replace('-', '').replace(':', '')

            result = account.combine_label_step(ts, close, target)
            if result:
                result_list.append(result)
        
        
        filename = f'./miniQMT/datasets/label_train_data/{call}_{put}_{target}.xlsx'
        df = pd.DataFrame(result_list)
        df.to_excel(filename, index=False, sheet_name='期权组合信息')

        print(f"[进度] {idx + 1} / {len(option_pairs)}")

    # # 初始化一次
    # first_close = float(data.iloc[0].close)
    # first_ts = str(data.iloc[0].ts).replace(' ', '').replace('-', '').replace(':', '')
    # account.init_state(first_ts, first_close)

    # actions = [2, 0, 2, 0, 2, 0, 1, 2, 0]
    # weights = [1, 0, 0.75, 0, 1, 0, 0.75, 1, 0]
    # # TEST
    # # idx = 0
    # # close = float(data.iloc[idx].close)
    # # ts = str(data.iloc[idx].ts).replace(' ', '').replace('-', '').replace(':', '')
    # # act, w = 2, 1
    # # state, reward, truncated = account.step([act], [w], ts=ts, close=close)
    # # idx += 1

    # for i, (ac, w) in enumerate(zip(actions, weights)):
    #     ts = str(data.iloc[i].ts)
    #     for c in [' ', ':', '-']:
    #         ts = ts.replace(c, '')
    #     close = float(data.iloc[i].close)
    #     state, reward, truncated = account.step([ac, ac], [w, w], ts=ts, close=close)

    #     if truncated:
    #         print('[Info] 触发截断,账户爆仓保护.')
    #         break
    

    # account.out_excel()
