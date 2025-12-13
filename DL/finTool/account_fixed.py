
"""
账户与撮合：修订版
- 修复与改进：
  1) out_excel 中的条件判断改为 <= 0
  2) 默认参数中的列表改为 None，避免共享可变对象
  3) submit_order 股票分支遗漏 append 的问题
  4) 统一用【有符号数量】来管理仓位：多头 > 0，空头 < 0
  5) dispose_order / simulate_fill 现金流与保证金的处理更严谨
  6) 每根 K 线动态重算保证金 (frozen_money)，使 free_money 保持总现金恒定
  7) 到期日处理：按内在价值现金交割，释放保证金，生成成交记录
  8) 多处文案/异常修正（如 raise Exception / 卖出平仓提示等）
- 兼容性：尽量保持原方法名，但内部逻辑更安全；如果外部代码只依赖这里提供的方法，应可无缝替换。
"""

from dataclasses import asdict, dataclass, field
from typing import ClassVar, Dict, Optional, Tuple, List
from itertools import count
import pandas as pd

# 外部依赖（按原代码名保留）
from optionBaseInfo import optionBaseInfo
from realInfo import RealInfo


# ========== 数据结构 ==========
@dataclass(slots=True)
class Order:
    _id_counter: ClassVar[count] = count(1)
    order_id: int = field(init=False)

    # 证券代码（期权 8 位 / 标的 6 位或 7 位）
    code: str
    # 方向：'买入开仓' / '卖出开仓' / '买入平仓' / '卖出平仓'；股票：'买入' / '卖出'
    direction: str
    # 申报数量（张）
    init_volume: int
    # 实际成交数量（张）
    success_volume: int
    # 时间戳字符串：YYYYMMDDhhmmss
    time_str: str
    # 订单状态：'已报' / '部分成交' / '废单' / '成交'
    status: str
    # 备注
    info: str = ''

    def __post_init__(self):
        self.order_id = next(self._id_counter)


@dataclass(slots=True)
class Trade:
    order_id: int
    code: str
    direction: str  # 成交方向（同 Order.direction）
    price: float    # 成交单价（不是总价）；记录时请存单价，避免歧义
    fee: float      # 总手续费
    time_str: str   # 成交时间
    success_volume: int  # 成交张数（合约张）


@dataclass(slots=True)
class Position:
    code: str
    qty: int                 # 有符号数量：多>0，空<0
    avg_cost: float          # 成本均价（单价），仅用于参考；空头这里记录为开仓均价
    multiplier: int          # 合约乘数


# ========== 账户 ==========
class Account:
    def __init__(
        self,
        init_capital: float,
        fee: float = 1.3,
        period: str = '30m',
        stockList: Optional[List[str]] = None,
        filepath: str = './miniQMT/datasets/'
    ):
        # 基础属性
        self.init_capital = init_capital
        self.filepath = filepath
        self.fee = fee
        self.period = period

        if stockList is None:
            stockList = ['510050', '588000']

        # 资金（现金）相关：free + frozen 始终等于总现金
        self.free_money = init_capital     # 可用资金
        self.frozen_money = 0.0            # 冻结保证金

        # 组合市值（持仓按市值计） + 现金 = equity
        self.equity = init_capital

        # 仓位：code -> Position
        self.positions: Dict[str, Position] = {}

        # 记录
        self.Trades: List[Trade] = []
        self.Orders: List[Order] = []
        self.has_disposed_id = -1

        # 控制器
        self.option_info_controller = optionBaseInfo(stockList, f'{filepath}/optionInfo')
        self.real_info_controller = RealInfo(stockList, f'{filepath}/realInfo', period, max_option_cache=15, date_pick='last')

        # 序列记录
        self.equity_list: List[float] = []
        self.time_list: List[str] = []
        self.gain_rate: List[float] = []
        self.draw_down: List[float] = []

    # ---------- 导出 ----------
    def out_excel(self):
        filepath = f'{self.filepath}/outs/account_info.xlsx'
        if len(self.time_list) <= 0:
            return

        peak = 0.0
        self.gain_rate = []
        self.draw_down = []
        for i in range(len(self.time_list)):
            peak = max(peak, self.equity_list[i])
            if i == 0:
                self.gain_rate.append((self.equity_list[i] - self.init_capital) / self.init_capital)
                self.draw_down.append(0.0)
            else:
                prev = self.equity_list[i - 1]
                self.gain_rate.append((self.equity_list[i] - prev) / prev if prev != 0 else 0.0)
                self.draw_down.append((peak - self.equity_list[i]) / peak if peak != 0 else 0.0)

        df = pd.DataFrame({
            '时间': self.time_list,
            '市值': self.equity_list,
            '收益率': self.gain_rate,
            '回撤': self.draw_down
        }, columns=['时间', '市值', '收益率', '回撤'])
        df.to_excel(filepath, sheet_name='账户信息', index=False)

        # 订单
        filepath = f'{self.filepath}/outs/order_list.xlsx'
        if len(self.Orders) > 0:
            df = pd.DataFrame([asdict(o) for o in self.Orders])
            df = df.rename({
                'order_id': '委托号',
                'code': '证券代码',
                'direction': '交易方向',
                'init_volume': '下单数量',
                'success_volume': '成交数量',
                'time_str': '成交时间',
                'status': '委托状态',
                'info': '说明信息'
            }, axis=1)
            df.to_excel(filepath, sheet_name='委托记录', index=False)

        # 成交
        filepath = f'{self.filepath}/outs/trade_list.xlsx'
        if len(self.Trades) > 0:
            df = pd.DataFrame([asdict(t) for t in self.Trades])
            df = df.rename({
                'order_id': '委托号',
                'code': '证券代码',
                'direction': '交易方向',
                'price': '成交单价',
                'fee': '总手续费',
                'success_volume': '成交数量',
                'time_str': '成交时间',
            }, axis=1)
            df.to_excel(filepath, sheet_name='交易记录', index=False)

    # ---------- 公共查询 ----------
    def set_fee(self, fee: float):
        self.fee = float(fee)

    def getMargin(self, optionCode: str) -> float:
        # 保留接口：静态保证金（若上层只想用交易所公式，不考虑标的波动）
        return self.option_info_controller.get_margin(optionCode)

    def getRealMargin(self, optionCode: str, time_str: str) -> float:
        # 动态保证金
        stockCode = self.option_info_controller.get_stockCode(optionCode)
        stock_price = self.real_info_controller.get_close_by_str(stockCode, time_str)
        strike_price = self.option_info_controller.get_strikePrice(optionCode)
        option_price = self.real_info_controller.get_close_by_str(optionCode, time_str)
        op_type = self.option_info_controller.get_optionType(optionCode)
        mul = self.option_info_controller.get_multiplier(optionCode)

        if op_type == "put":
            delta = max(stock_price - strike_price, 0.0)
            m = option_price + max(0.12 * stock_price - delta, 0.07 * strike_price)
            margin = min(m, strike_price) * mul
        elif op_type == "call":
            delta = max(strike_price - stock_price, 0.0)
            margin = (option_price + max(0.12 * stock_price - delta, 0.07 * stock_price)) * mul
        else:
            raise Exception(f"Unknown option type for {optionCode}")
        return float(margin)

    def getClosePrice(self, code: str, time_str: str) -> float:
        return float(self.real_info_controller.get_close_by_str(code, time_str))

    def getRealVolume(self, code: str, time_str: str) -> int:
        return int(self.real_info_controller.get_volume_by_str(code, time_str))

    # ---------- 下单 ----------
    def submit_order(self, code: str, direction: str, volume: int, time_str: str, price: Optional[float] = None):
        if len(code) == 8:
            assert direction in ['买入开仓', '卖出开仓', '买入平仓', '卖出平仓']
            order = Order(code, direction, volume, 0, time_str, '已报')
            self.Orders.append(order)
        elif len(code) in (6, 7):
            assert direction in ['买入', '卖出']
            order = Order(code, direction, volume, 0, time_str, '已报')
            self.Orders.append(order)
        else:
            raise Exception(f"无法识别的 code: {code}")

    # ---------- 内部工具 ----------
    def _total_cash(self) -> float:
        return float(self.free_money + self.frozen_money)

    def _set_frozen_by_positions(self, time_str: str):
        "根据当前【空头】头寸动态重算保证金，并保持总现金不变"
        total_cash = self._total_cash()
        frozen = 0.0
        for pos in self.positions.values():
            if len(pos.code) == 8 and pos.qty < 0:
                # 空头期权需要保证金
                frozen += self.getRealMargin(pos.code, time_str) * abs(pos.qty)
        self.frozen_money = float(frozen)
        self.free_money = float(total_cash - self.frozen_money)

    def _mark_to_market_value(self, time_str: str) -> float:
        "按签名数量计算持仓市值（可为负）"
        total = 0.0
        for pos in self.positions.values():
            price = self.getClosePrice(pos.code, time_str)
            pos_value = price * pos.multiplier * pos.qty  # qty 有符号
            total += pos_value
        return float(total)

    def _get_multiplier(self, code: str) -> int:
        if len(code) == 8:
            return int(self.option_info_controller.get_multiplier(code))
        else:
            # 股票或 ETF：默认 1
            return 1

    # ---------- 撮合与持仓更新 ----------
    def dispose_order(self, code: str, delta_qty: int, price: float, time_str: str, order_id: int, fee_total: float):
        """
        统一的持仓与现金更新（内部使用）：
        - delta_qty > 0: 买（开多或平空）
        - delta_qty < 0: 卖（开空或平多）
        - price 为成交单价
        - fee_total 为本次成交的总手续费
        """
        mul = self._get_multiplier(code)
        cash_delta = - price * mul * delta_qty  # 买 -> 现金减少；卖 -> 现金增加
        self.free_money += cash_delta - fee_total

        # 更新持仓（签名数量）
        if code in self.positions:
            pos = self.positions[code]
            new_qty = pos.qty + delta_qty

            # 更新均价（仅对同向加仓有意义）
            if (pos.qty >= 0 and delta_qty > 0) or (pos.qty <= 0 and delta_qty < 0):
                # 同向加仓：加权更新均价
                notional_old = abs(pos.qty) * pos.avg_cost
                notional_new = abs(delta_qty) * price
                new_avg = (notional_old + notional_new) / max(abs(new_qty), 1)
                pos.avg_cost = new_avg
            pos.qty = new_qty

            if pos.qty == 0:
                del self.positions[code]
        else:
            self.positions[code] = Position(code=code, qty=delta_qty, avg_cost=price, multiplier=mul)

        # 更新成交记录
        self.Trades.append(Trade(order_id, code, '买入' if delta_qty > 0 else '卖出', price, fee_total, time_str, abs(delta_qty)))

    def _settle_expired(self, time_str: str):
        """
        期权到期结算（现金结算）：
        - 多头：收取内在价值
        - 空头：支付内在价值，并释放保证金
        """
        to_delete = []
        for code, pos in list(self.positions.items()):
            if len(code) != 8:
                continue
            expire = self.option_info_controller.get_expireDate(code)  # 'YYYYMMDD'
            if expire < time_str[0:8]:
                # 计算内在价值
                stock = self.option_info_controller.get_stockCode(code)
                K = float(self.option_info_controller.get_strikePrice(code))
                S = float(self.getClosePrice(stock, time_str))
                mul = pos.multiplier
                op_type = self.option_info_controller.get_optionType(code)
                if op_type == 'call':
                    intrinsic = max(S - K, 0.0) * mul
                elif op_type == 'put':
                    intrinsic = max(K - S, 0.0) * mul
                else:
                    intrinsic = 0.0

                qty_abs = abs(pos.qty)
                cash_change = intrinsic * (1 if pos.qty > 0 else -1)  # 多收空付
                # 释放保证金（空头）——在后续 _set_frozen_by_positions 统一处理

                # 生成结算成交（视为平仓）
                direction = '卖出平仓' if pos.qty > 0 else '买入平仓'
                # 手续费按 0（也可替换为券商行权费规则）
                self.Trades.append(Trade(0, code, direction, 0.0, 0.0, time_str, qty_abs))

                # 更新现金与持仓
                self.free_money += cash_change
                to_delete.append(code)

        for code in to_delete:
            if code in self.positions:
                del self.positions[code]

    def update_positions(self, time_str: str):
        # 到期先结算
        self._settle_expired(time_str)

        # 动态重算保证金（保持总现金恒定）
        self._set_frozen_by_positions(time_str)

        # 重新计算 equity
        position_value = self._mark_to_market_value(time_str)
        self.equity = position_value + self._total_cash()

    # ---------- 撮合主流程 ----------
    def simulate_fill(self, time_str: str):
        if self.has_disposed_id >= len(self.Orders) - 1:
            return

        for order in self.Orders[self.has_disposed_id + 1:]:
            self.has_disposed_id += 1

            code = order.code
            direction = order.direction
            volume_req = int(order.init_volume)
            order_id = order.order_id

            # 已到期禁止下单
            if len(code) == 8:
                expire = self.option_info_controller.get_expireDate(code)
                if expire < time_str[0:8]:
                    order.status = '废单'
                    order.info = '期权到期后无法下单'
                    continue

            real_volume = self.getRealVolume(code, time_str)
            price = self.getClosePrice(code, time_str)
            mul = self._get_multiplier(code)

            # 简单撮合：限制为【申报量】【当根真实量】中的最小值
            # 可按需加入“最大占比”与滑点模型
            max_by_liquidity = min(volume_req, real_volume) if real_volume > 0 else 0

            # 每一笔按最优可能成交（不考虑拆单细节）
            filled = 0
            fee_total = 0.0

            if len(code) == 8:
                # ===== 期权 =====
                if direction == '买入开仓':
                    # 现金约束
                    unit_cost = price * mul + self.fee
                    num_by_cash = int(self.free_money // unit_cost) if unit_cost > 0 else 0
                    can_fill = max(0, min(max_by_liquidity, num_by_cash))

                    if can_fill <= 0:
                        order.status = '废单'
                        order.info = '资金或流动性不足开仓'
                        continue

                    filled = can_fill
                    fee_total = self.fee * filled
                    self.dispose_order(code, +filled, price, time_str, order_id, fee_total)
                    order.status = '成交' if filled == volume_req else '部分成交'
                    order.success_volume = filled
                    if filled < volume_req and max_by_liquidity < volume_req:
                        order.info = '下单数量超过真实成交量'

                elif direction == '卖出开仓':
                    # 保证金约束
                    margin_unit = self.getRealMargin(code, time_str)
                    num_by_margin = int(self.free_money // margin_unit) if margin_unit > 0 else 0
                    can_fill = max(0, min(max_by_liquidity, num_by_margin))

                    if can_fill <= 0:
                        order.status = '废单'
                        order.info = '资金（保证金）或流动性不足开仓'
                        continue

                    filled = can_fill
                    # 手续费可按卖出方向计入，也可以 0；此处计入
                    fee_total = self.fee * filled
                    self.dispose_order(code, -filled, price, time_str, order_id, fee_total)
                    order.status = '成交' if filled == volume_req else '部分成交'
                    order.success_volume = filled
                    if filled < volume_req and max_by_liquidity < volume_req:
                        order.info = '下单数量超过真实成交量'

                elif direction == '买入平仓':
                    pos = self.positions.get(code, None)
                    short_pos = abs(pos.qty) if (pos and pos.qty < 0) else 0
                    if short_pos <= 0:
                        order.status = '废单'
                        order.info = '无空头可平（或方向错误）'
                        continue
                    can_fill = min(short_pos, max_by_liquidity, volume_req)
                    filled = can_fill
                    fee_total = self.fee * filled
                    self.dispose_order(code, +filled, price, time_str, order_id, fee_total)  # 买入 -> delta_qty > 0
                    order.status = '成交' if filled == volume_req else '部分成交'
                    order.success_volume = filled

                elif direction == '卖出平仓':
                    pos = self.positions.get(code, None)
                    long_pos = pos.qty if (pos and pos.qty > 0) else 0
                    if long_pos <= 0:
                        order.status = '废单'
                        order.info = '无多头可平（或方向错误）'
                        continue
                    can_fill = min(long_pos, max_by_liquidity, volume_req)
                    filled = can_fill
                    fee_total = self.fee * filled
                    self.dispose_order(code, -filled, price, time_str, order_id, fee_total)  # 卖出 -> delta_qty < 0
                    order.status = '成交' if filled == volume_req else '部分成交'
                    order.success_volume = filled

            elif len(code) in (6, 7):
                # ===== 股票/ETF（示意，默认乘数1） =====
                if direction == '买入':
                    unit_cost = price * mul + self.fee
                    num_by_cash = int(self.free_money // unit_cost) if unit_cost > 0 else 0
                    can_fill = max(0, min(max_by_liquidity, num_by_cash))
                    if can_fill <= 0:
                        order.status = '废单'
                        order.info = '资金或流动性不足'
                        continue
                    filled = can_fill
                    fee_total = self.fee * filled
                    self.dispose_order(code, +filled, price, time_str, order_id, fee_total)
                    order.status = '成交' if filled == volume_req else '部分成交'
                    order.success_volume = filled

                elif direction == '卖出':
                    pos = self.positions.get(code, None)
                    long_pos = pos.qty if (pos and pos.qty > 0) else 0
                    if long_pos <= 0:
                        order.status = '废单'
                        order.info = '无多头可卖'
                        continue
                    can_fill = min(long_pos, max_by_liquidity, volume_req)
                    filled = can_fill
                    fee_total = self.fee * filled
                    self.dispose_order(code, -filled, price, time_str, order_id, fee_total)
                    order.status = '成交' if filled == volume_req else '部分成交'
                    order.success_volume = filled

            # 每笔撮合后，重算保证金与 equity
            self.update_positions(time_str)

    # ---------- 期权列表/辅助 ----------
    def get_option_list(self, stockCode: str='510050', expire: str='202512', op_type: str='call'):
        return self.option_info_controller.find_options_by_stock_and_expiry(stockCode, expire, op_type)

    def _nearest_atm_pair(self, stockCode: str, time_str: str, expire: str) -> Tuple[str, str]:
        "找到最接近 ATM 的 (call, put) 期权代码"
        S = self.getClosePrice(stockCode, time_str)
        calls = self.get_option_list(stockCode, expire, 'call')
        puts = self.get_option_list(stockCode, expire, 'put')
        # 选择行权价最接近标的价的合约
        def nearest(options: List[str]):
            best, best_diff = None, 1e18
            for oc in options:
                K = float(self.option_info_controller.get_strikePrice(oc))
                diff = abs(K - S)
                if diff < best_diff:
                    best, best_diff = oc, diff
            return best
        return nearest(calls), nearest(puts)

    def open_straddle(self, stockCode: str, time_str: str, expire: str, volume_each: int, side: str='buy'):
        "简化下单：在给定到期上买/卖跨式（各 volume_each 张）"
        call_code, put_code = self._nearest_atm_pair(stockCode, time_str, expire)
        if side == 'buy':
            self.submit_order(call_code, '买入开仓', volume_each, time_str)
            self.submit_order(put_code, '买入开仓', volume_each, time_str)
        elif side == 'sell':
            self.submit_order(call_code, '卖出开仓', volume_each, time_str)
            self.submit_order(put_code, '卖出开仓', volume_each, time_str)
        else:
            raise Exception("side 必须为 'buy' 或 'sell'")

    # ---------- 回测主入口 ----------
    def handlebar(self, data: pd.DataFrame, func):
        for row in data.itertuples(index=False):
            time_str, stock_close = str(row.ts), float(row.close)
            self.time_list.append(time_str)

            # 格式化时间串
            for it in [' ', ':', '-']:
                time_str = time_str.replace(it, '')

            # 先更新市值（含到期结算与保证金重算）
            self.update_positions(time_str)

            # 策略回调（可能产生订单）
            func(time_str, stock_close)

            # 撮合成交
            self.simulate_fill(time_str)

            # 再次更新市值
            self.update_positions(time_str)
            self.equity_list.append(self.equity)

    # 示例策略（保留原样但更小心）
    def strategy(self, time_str: str, stock_close: float):
        code = '10002064'  # 示例代码
        self.submit_order(code, '买入开仓', 10, time_str)  # 用小量示意

    def main(self, start_str: str, end_str: str, benchmark: str='510050', period: str=''):
        if period == '':
            period = self.period
        data = self.real_info_controller.get_bars_between_from_df(benchmark, start_str, end_str, period)
        self.handlebar(data, self.strategy)
        self.out_excel()


# 运行样例（按需注释掉，保持与原始脚本结构近似）
if __name__ == "__main__":
    # 510050: K线: 20180504~20251017, 期权最早20171123
    account = Account(15000, 1.3, '30m', ['510050'])
    account.main('20200122100000', '20200123100000', '510050')
