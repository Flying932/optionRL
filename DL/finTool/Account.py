"""
    手动实现一个回测的框架.
    本代码是账户信息.
"""
from dataclasses import asdict, dataclass, field
from re import M
from typing import ClassVar
import random
import numpy as np

# 引入包
if __name__ != '__main__':
    from finTool.optionBaseInfo import optionBaseInfo
    from finTool.realInfo import RealInfo
else:
    from optionBaseInfo import optionBaseInfo
    from realInfo import RealInfo

from itertools import count
import pandas as pd
import math

# 订单类
@dataclass(slots=True)
class Order:
    # 类变量 (int): 订单号, 从1开始
    _id_counter: ClassVar[count] = count(1)
    order_id: int=field(init=False)

    # 代码, 10000101 / 588000
    code: str

    # 买卖方向: 'buy' / 'sell'
    direction: str

    # 下单数量
    init_volume: int

    # 成交数量
    success_volume: int

    time_str: str

    # 订单状态: 已报 / 部分成交 / 废单 / 成交
    status: str

    # 说明信息
    info: str=''

    # 订单号
    def __post_init__(self):
        self.order_id = next(self._id_counter)

# 成交类(若部分成交, 只产生部分成交的成交记录)
@dataclass(slots=True)
class Trade:
    order_id: int
    code: str
    direction: str

    # 成交总价格
    price: float

    # 总手续费
    fee: float

    # 成交时间: 20251018093000
    time_str: str

    # 成交张数
    success_volume: int

class Account:
    def __init__(self, init_capital: float, call: str, put: str, fee: float=1.3, period: str='30m', stockList=['510050', '588000'], filepath: str=f'./miniQMT/datasets/'):
        # 账户资金信息
        self.init_capital = init_capital

        self.filepath = filepath

        # 手续费
        self.fee = fee

        # 可用资金
        self.free_money = init_capital

        # 冻结资金
        self.frozen_money = 0

        # 账户市值
        self.equity = init_capital

        # 账户持仓信息: code -> (direction, volume, value)
        self.positions = {}

        # 账户交易记录: time (str, like 20250310093000), type(买开买平, etc.) price, volume, fee
        self.Trades = []

        # 订单清单(通常下一K线或者本K线结束成交) time, type, price, volume, fee
        self.Orders = []
        
        # 已经处理的订单编号
        self.has_disposed_id = -1

        # 期权信息类
        self.option_info_controller = optionBaseInfo(stockList, f'{filepath}/optionInfo')

        # 期权/股票K线类
        self.real_info_controller = RealInfo(stockList, f'{filepath}/realInfo', period, max_option_cache=15, date_pick='last')

        # 数据类型
        self.period = period

        # 市值变化
        self.equity_list = []

        # 运行的time_list
        self.time_list = []

        # 收益率
        self.gain_rate = []

        # 回撤列表
        self.draw_down = []

        # env相关变量, 目前先设计两个期权的跨式组合
        self.info = {}
        self.call = call
        self.put = put
        self.strike = 3.0
        self.last_close = None
        self.target_gain = 0
        self.call_price = 0
        self.put_price = 0
        self.ttm = 0
        self.pos_dir = 0 # 净方向: 多=1, 空=-1, 空仓=0
        self.pos_size = 0 # 净手数
        self.cash_ratio = self.free_money / self.equity
        self.margin_ratio = self.frozen_money / self.equity
        self.last_equity = self.equity
        self.last_reward = 0
        self.target_price = None
        self.equity_peak = self.equity
        self.down = 0

        # 奖励函数惩罚相关
        self.use_penalties = True

        self.lam_dd = 0.02      # 回撤
        self.lam_mr = 0.01      # 保证金占比
        self.lam_trade_cnt = 1e-6     # 交易频次
        self.lam_ttm = 0.01     # 到期
        self.lam_dlt = 0.03     # delta风险
        self.last_trade_cnt = 0

        self.eps = 1e-6


        # env状态
        self.state = {
            # 标的的对数增长率
            'target_gain': self.target_gain,

            # call和put的实时价格与标的价格比值
            'call_price': self.call_price / self.target_price if self.target_price else self.call_price,
            'put_price': self.put_price / self.target_price if self.target_price else self.put_price,

            # 到期日
            'ttm': self.ttm,

            # 多空方向
            'pos_dir': self.pos_dir,

            # 头寸大小
            'pos_size': self.pos_size,

            # 可用资金比例
            'cash_ratio': self.cash_ratio,

            # 冻结资金比例
            'margin_ratio': self.margin_ratio,

            # 行权价与标的价格比值
            'strike_price': self.strike / self.target_price if self.target_price else self.strike,

            # # 标的价格
            # 'target_price': self.target_price
            'draw_down': 0,
        }



        self.info = {
            "message": "initial"
        }

        self.frozen_money_list = [self.frozen_money]

    # env函数
    def init_state(self, time_str: str, close: float):
        self.call_price = self.real_info_controller.get_close_by_str(self.call, time_str, self.period)
        self.put_price = self.real_info_controller.get_close_by_str(self.put, time_str, self.period)
        expire = self.option_info_controller.get_expireDate(self.call)
        self.ttm = self.real_info_controller.get_ttm(time_str, expire)
        self.target_price = close
        self.strike = self.option_info_controller.get_strikePrice(self.call)

        self.state['call_price'] = self.call_price / self.target_price
        self.state['put_price'] = self.put_price / self.target_price
        self.state['ttm'] = self.ttm
        # self.state['target_price'] = self.target_price
        self.state['strike_price'] = self.strike / self.target_price
    
    def getState(self):
        state = []
        for _, v in self.state.items():
            state.append(v)
        return state

    def getInfo(self):
        return self.info


    # 导出excel
    def out_excel(self):
        filepath = f'{self.filepath}/outs/account_info.xlsx'
        if len(self.time_list) <= 0:
            return
        
        peak = 0
        for i in range(len(self.time_list)):
            peak = max(peak, self.equity_list[i])
            if i == 0:
                self.gain_rate.append((self.equity_list[i] - self.init_capital) / self.init_capital)
                self.draw_down.append(0)
            else:
                self.gain_rate.append((self.equity_list[i] - self.equity_list[i - 1]) / self.equity_list[i - 1])
                self.draw_down.append((peak - self.equity_list[i]) / peak)
        
        df = pd.DataFrame({
            '时间': self.time_list,
            '市值': self.equity_list,
            '收益率': self.gain_rate,
            '回撤': self.draw_down
        }, columns=['时间', '市值', '收益率', '回撤'])
        df.to_excel(filepath, sheet_name='账户信息', index=False)
        print(f'[Info] 导出账户信息记录成功.')

        # 导出order_list
        filepath = f'{self.filepath}/outs/order_list.xlsx'

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
        print(f'[Info] 导出委托记录记录成功.')

        filepath = f'{self.filepath}/outs/trade_list.xlsx'
        df = pd.DataFrame([asdict(o) for o in self.Trades])
        df = df.rename({
            'order_id': '委托号',
            'code': '证券代码',
            'direction': '交易方向',
            'price': '总下单价格',
            'fee': '总手续费',
            'success_volume': '成交数量',
            'time_str': '成交时间',
        }, axis=1)

        df.to_excel(filepath, sheet_name='交易记录', index=False)
        print(f'[Info] 导出交易记录记录成功.')

    # 设置手续费
    def set_fee(self, fee):
        self.fee = fee

    # 计算期权的保证金(静态)
    def getMargin(self, optionCode: str):
        return self.option_info_controller.get_margin(optionCode)

    # 计算期权的保证金(动态)
    def getRealMargin(self, optionCode: str, time_str: str):

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
        elif op_type == "unknown":
            print(f"[Info: getMargin错误] optionCode = {optionCode}, op_type = unknown")

        return margin       
    
    # 获取收盘价
    def getClosePrice(self, code: str, time_str: str="20251008093000"):
        return self.real_info_controller.get_close_by_str(code, time_str)

    # 获取成交量
    def getRealVolume(self, code: str, time_str: str="20251008093000"):
        return self.real_info_controller.get_volume_by_str(code, time_str)
    
    # 下单
    def submit_order(self, code: str, direction: str, volume: int, time_str: str, price=None):
        if len(code) == 8:
            # 期权, 期权买入平仓或者卖出平仓, volume随便传一个值就行
            assert direction in ['买入开仓', '卖出开仓', '买入平仓', '卖出平仓']
            order = Order(code, direction, volume, 0, time_str, '已报')
            self.Orders.append(order)

        elif len(code) == 6:
            # 股票
            assert direction in ['买入', '卖出']
            order = Order(code, direction, volume, 0, time_str, '已报')
            self.Orders.append(order)
    
    # 处理订单, 更新仓位和资金
    def dispose_order(self, code, dispose_volume, price: float, free_money_delta: float, frozen_money_delta: float):
        if code in self.positions:
            direction, volume, value = self.positions[code]
            if direction == '卖出开仓':
                volume = -volume
            
            volume += dispose_volume
            value = price * volume * self.option_info_controller.get_multiplier(code)

            if volume > 0:
                # value最后再更新
                self.positions[code] = ('买入开仓', volume, value)
            elif volume < 0:
                self.positions[code] = ('卖出开仓', -volume, value)
            else:
                del self.positions[code]
        else:
            if dispose_volume > 0:
                self.positions[code] = ('买入开仓', dispose_volume, price * dispose_volume * self.option_info_controller.get_multiplier(code))
            else:
                self.positions[code] = ('卖出开仓', -dispose_volume, price * dispose_volume * self.option_info_controller.get_multiplier(code))
        self.frozen_money += frozen_money_delta
        self.free_money += free_money_delta

    # 更新仓位市值, 到期的需要直接了结
    def update_positions(self, time_str: str):
        delete_list = []

        total_equity = 0

        for key in self.positions.keys():
            # 暂时没有处理股票

            if len(key) == 8:
                direction, volume, value = self.positions[key]

                expire = self.option_info_controller.get_expireDate(key)
                if expire < time_str[0: 8]:
                    delete_list.append(key)

                    if direction == '买入开仓':
                        v = value
                        if v > 0:
                            order = Order(key, '卖出平仓', volume, volume, time_str, '成交', '强制卖出平仓')
                            self.Orders.append(order)

                            trade = Trade(order.order_id, key, '卖出平仓', value, 0, time_str, volume)
                            self.Trades.append(trade)

                            self.free_money += v
                            continue

                    elif direction == '卖出开仓':
                        margin = self.getMargin(key)

                        order = Order(key, '买入平仓', volume, volume, time_str, '成交', '强制买入平仓')
                        self.Orders.append(order)

                        trade = Trade(order.order_id, key, '买入平仓', value, self.fee * volume, time_str, volume)
                        self.Trades.append(trade)

                        self.frozen_money -= margin * volume
                        self.free_money = self.free_money - self.fee * volume - value + margin * volume
                        continue

                    print(f"[Info: 期权到期强制了结] code = {key}")
                    continue

            
            # 获取最新价格
            price = self.getClosePrice(key, time_str)
            self.positions[key] = (direction, abs(volume), price * volume * self.option_info_controller.get_multiplier(key))

            if direction == '买入开仓' or direction == '买入':
                total_equity += (price * volume * self.option_info_controller.get_multiplier(key))

            elif direction == '卖出开仓':
                total_equity -= (price * volume * self.option_info_controller.get_multiplier(key))
        for key in delete_list:
            del self.positions[key]

        self.frozen_money_list.append(self.frozen_money)
        self.equity = total_equity + self.free_money + self.frozen_money

    # 处理订单
    def simulate_fill(self, time_str: str):
        if self.has_disposed_id >= len(self.Orders) - 1:
            return
        
        temp_id = self.has_disposed_id
        for order in self.Orders[temp_id + 1: ]:
            self.has_disposed_id += 1

            code = order.code
            direction = order.direction
            volume = order.init_volume
            order_id = order.order_id

            if order.info == '强制买入平仓' or order.info == '强制卖出平仓':
                continue
            
            # 已经到期
            expire = self.option_info_controller.get_expireDate(code)

            if expire < time_str[0: 8]:
                order.status = '废单'
                order.info = '期权到期后无法下单'
                continue
            
            # 查询交易量
            real_volume = self.getRealVolume(code, time_str)

            # 查询价格 保证金 合约乘数
            price = self.getClosePrice(code, time_str)
            margin = self.getMargin(code)
            mul = self.option_info_controller.get_multiplier(code)

            if len(code) == 8:
                # 期权
                if direction == '买入开仓':
                    num_can_buy = int(self.free_money / (mul * price + self.fee))
                    max_cnt = min(volume, num_can_buy)
                    max_cnt = min(real_volume, max_cnt)

                    if max_cnt <= 0:
                        order.status = '废单'
                        order.info = '账户资金不足开仓'
                        continue
                    elif max_cnt == volume:
                        order.status = '成交'
                        order.success_volume = max_cnt
                    elif max_cnt < volume:
                        order.status = '部分成交'
                        order.success_volume = max_cnt
                        order.info = ''

                        if volume > real_volume:
                            order.info = '下单数量超过真实成交量'
                        if volume > num_can_buy:
                            if order.info == '':
                                order.info = '资金不足'
                            else:
                                order.info += ' | 资金不足'
                            
                    frozen_delta = 0
                    free_delta = -price * max_cnt * mul - max_cnt * self.fee
                    
                    self.dispose_order(code, max_cnt, price, free_delta, frozen_delta)
                    trade = Trade(order_id, code, direction, price * max_cnt * mul, self.fee * max_cnt, time_str, max_cnt)
                    self.Trades.append(trade)
                elif direction == '卖出开仓':

                    num_can_buy = int(self.free_money / margin)

                    max_cnt = min(volume, num_can_buy)
                    max_cnt = min(max_cnt, real_volume)

                    if max_cnt <= 0:
                        order.status = '废单'
                        order.info = '账户资金不足开仓'
                        continue
                    elif max_cnt == volume:
                        order.status = '成交'
                        order.success_volume = max_cnt
                    elif max_cnt < volume:
                        order.status = '部分成交'
                        order.success_volume = max_cnt
                        order.info = ''

                        if volume > real_volume:
                            order.info = '下单数量超过真实成交量'
                        if volume > num_can_buy:
                            if order.info == '':
                                order.info = '资金不足'
                            else:
                                order.info += ' | 资金不足'

                    frozen_delta = margin * max_cnt
                    free_delta = price * mul * max_cnt - margin * max_cnt

                    # print(f"max_cnt = {max_cnt}, frozen_delta = {frozen_delta}, free_delta = {free_delta}")

                    self.dispose_order(code, -max_cnt, price, free_delta, frozen_delta)
                    trade = Trade(order_id, code, direction, price * max_cnt * mul, 0, time_str, max_cnt)
                    self.Trades.append(trade)

                elif direction == '买入平仓':
                    raw_direction, raw_volume, _ = self.positions.get(code, ('无仓位', 0, 0))
                    if raw_volume == 0 or (raw_volume > 0 and raw_direction == '买入开仓'):
                        order.status = '废单'
                        order.info = f'没有期权持仓或者方向错误, 无法买入平仓, direction = {raw_direction}'
                        continue
                    elif raw_volume > 0 and raw_direction == '卖出开仓':
                        # 先释放保证金, 再交易, 保证金一定能覆盖买入的费用, 目前我先不设判断
                        frozen_delta = -margin * raw_volume
                        free_delta = margin * raw_volume - raw_volume * self.fee - raw_volume * price * mul

                        # print(f"raw_volume = {raw_volume}, frozen_delta = {frozen_delta}, free_delta = {free_delta}")

                        order.status = '成交'
                        order.success_volume = raw_volume
                        order.init_volume = raw_volume

                        self.dispose_order(code, raw_volume, price, free_delta, frozen_delta)
                        trade = Trade(order_id, code, direction, price * raw_volume * mul, self.fee * raw_volume, time_str, raw_volume)
                        self.Trades.append(trade)

                elif direction == '卖出平仓':
                    raw_direction, raw_volume, _ = self.positions.get(code, ('无仓位', 0, 0))
                    if raw_volume == 0 or (raw_volume > 0 and raw_direction == '卖出开仓'):
                        order.status = '废单'
                        order.info = order.info = f'没有期权持仓或者方向错误, 无法买入平仓, direction = {raw_direction}'
                        continue
                    
                    elif raw_volume > 0 and raw_direction == '买入开仓':
                        frozen_delta = 0
                        free_delta = raw_volume * price * mul

                        # 目前没考虑资金卖出开仓后为负值的情况, 后面再改吧, 先认为一定可以平掉
                        order.status = '成交'
                        order.success_volume = raw_volume
                        order.init_volume = raw_volume

                        self.dispose_order(code, -raw_volume, price, free_delta, frozen_delta)
                        trade = Trade(order_id, code, direction, price * raw_volume * mul, 0, time_str, raw_volume)
                        self.Trades.append(trade)

            elif len(code) == 6 or len(code) == 7:
                # 510050.SH 或者 510050
                raise '目前暂时不交易股票'

    # ---------- combo helpers: 近似原子化的成对下单 ----------

    def _pair_qty_buy_open(self, ts: str, desired: int, call: str, put: str) -> int:
        """跨式买开时，两腿共同可成交的手数 q"""
        price_c = self.getClosePrice(call, ts)
        price_p = self.getClosePrice(put, ts)
        mul_c = self.option_info_controller.get_multiplier(call)
        mul_p = self.option_info_controller.get_multiplier(put)

        # 两腿合计现金成本/手
        per_cost = price_c * mul_c + price_p * mul_p + 2 * self.fee                
        cap_cash = int(self.free_money // per_cost) if per_cost > 0 else desired

        # 两腿真实可成交量的下界
        vol_cap = min(self.getRealVolume(call, ts), self.getRealVolume(put, ts))
        return max(0, min(desired, cap_cash, vol_cap))

    def _pair_qty_sell_open(self, ts: str, desired: int, call: str, put: str) -> int:
        """跨式卖开时，两腿共同可成交的手数 q (按静态保证金)"""
        m_c = self.getMargin(call)
        m_p = self.getMargin(put)

        per_margin = m_c + m_p

        cap_cash = int(self.free_money // per_margin) if per_margin > 0 else desired
        vol_cap = min(self.getRealVolume(call, ts), self.getRealVolume(put, ts))

        return max(0, min(desired, cap_cash, vol_cap))

    def open_long_pair(self, desired: int, ts: str, call: str, put: str) -> int:
        """买入开仓：两腿同手数"""
        q = self._pair_qty_buy_open(ts, desired, call, put)
        if q <= 0: return 0
        self.submit_order(call, '买入开仓', q, ts)
        self.submit_order(put,  '买入开仓', q, ts)
        return q

    def open_short_pair(self, desired: int, ts: str, call: str, put: str) -> int:
        """卖出开仓：两腿同手数"""
        q = self._pair_qty_sell_open(ts, desired, call, put)
        if q <= 0: return 0
        self.submit_order(call, '卖出开仓', q, ts)
        self.submit_order(put, '卖出开仓', q, ts)
        return q

    def close_pair_fully(self, ts: str, call: str, put: str) -> None:
        """把当前两腿各自完全平掉，不再依赖全局 pos_size"""
        for code in [call, put]:
            pos = self.positions.get(code)
            if not pos: continue
            d, v, _ = pos
            if v <= 0: continue
            if d == '卖出开仓':
                self.submit_order(code, '买入平仓', v, ts)   # 关空头
            elif d == '买入开仓':
                self.submit_order(code, '卖出平仓', v, ts)   # 关多头

    def flip_short_to_long(self, target: int, ts: str, call: str, put: str) -> None:
        """从空翻多，先平空 v, 再买开补足到 target"""
        v = min(target, self.pos_size)
        if v > 0:
            self.submit_order(call, '买入平仓', v, ts)
            self.submit_order(put, '买入平仓', v, ts)
        l = target - v
        if l > 0:
            self.open_long_pair(l, ts, call, put)

    def flip_long_to_short(self, target: int, ts: str, call: str, put: str) -> None:
        """从多翻空，先平多 v, 再卖开补足到 target"""
        v = min(target, self.pos_size)
        if v > 0:
            self.submit_order(call, '卖出平仓', v, ts)
            self.submit_order(put,  '卖出平仓', v, ts)
        l = target - v
        if l > 0:
            self.open_short_pair(l, ts, call, put)

    # 辅助函数: (标的, 到期日, 期权类型) -> 期权代码列表
    def get_option_list(self, stockCode: str='510050', expire: str='202512', op_type: str='call'):
        return self.option_info_controller.find_options_by_stock_and_expiry(stockCode, expire, op_type)
    
    # 逐个K线运行, data是K线
    def handlebar(self, data: pd.DataFrame, func):
        first_flag = True

        for row in data.itertuples(index=False):
            time_str, stock_close = row.ts, row.close
            time_str = str(time_str)
            self.time_list.append(time_str)

            for it in [' ', ':', '-']:
                time_str = time_str.replace(it, '')
            
            # 更新信息
            if first_flag:
                self.update_positions(time_str)

            # 执行策略(策略里会下单)
            func(time_str, stock_close)

            # 处理订单
            self.simulate_fill(time_str)
            
            # 更新持仓信息
            self.update_positions(time_str)
            self.equity_list.append(self.equity)

    # 策略函数
    def strategy(self, time_str: str, stock_close: float):
        # code_list = self.get_option_list('510050', time_str[0: 6], 'put')
        # code = code_list[0]

        code = '10002064'
        self.submit_order(code, '买入开仓', 1, time_str)
    

    def main(self, start_str, end_str, benchmark: str='510050', period: str=''):
        if period == '':
            period = self.period
        data = self.real_info_controller.get_bars_between_from_df(benchmark, start_str, end_str, period)
        self.handlebar(data, self.strategy)

        self.out_excel()

    # ---------- 以下是env环境相关 ----------
    # 计算奖励函数
    # def getReward(self, eps: float=1e-6):
    #     a = self.equity if self.equity > 0 else eps
    #     b = self.last_equity if self.last_equity > 0 else eps
    #     self.last_equity = self.equity
    #     r = math.log(a / b)
    #     len_trades = len(self.Trades)

    #     # r = r + self.last_reward
    #     # self.last_reward = r

    #     if not self.use_penalties:
    #         return r
        
    #     # 风险惩罚
    #     self.equity_peak = max(self.equity, self.equity_peak)
    #     peak = self.equity_peak if self.equity_peak > eps else eps
    #     dd = (peak - self.equity) / peak
        
    #     # 1.回撤惩罚, 超过15%再惩罚
    #     pen_dd = self.lam_dd * (max(0, dd - 0.15) ** 2)

    #     # 2.保证金占比惩罚, 超过30%再惩罚
    #     pen_mr = self.lam_mr * (max(0.0, self.margin_ratio - 0.3)) ** 2

    #     # 3.delta风险 / 4.ttm (暂时没有写)

    #     # 5.交易频次
    #     pen_trade_cnt = self.lam_trade_cnt * max(0, len_trades - self.last_trade_cnt)
    #     self.last_trade_cnt = len_trades

    #     # 总奖励
    #     # print(f"奖励: r = {r}, pen_dd = {pen_dd}, pen_mr = {pen_mr}, pen_order_cnt = {pen_trade_cnt}")
    #     r = r - (pen_dd + pen_mr + pen_trade_cnt)

    #     return r

    def getReward(self, eps: float = 1e-6):
        # --- 基础收益：非累加 log-return ---
        a = self.equity if self.equity > 0 else eps
        b = self.last_equity if self.last_equity > 0 else eps
        r_base = math.log(a / b)
        self.last_equity = self.equity

        r = r_base + self.last_reward
        self.last_reward = r
        return r


        # 无惩罚开关：直接返回剪裁后的基线
        if not getattr(self, 'use_penalties', True):
            return float(np.clip(r_base, -0.05, 0.05))

        # ---------- 风险与成本惩罚 ----------
        # （1）回撤惩罚：只罚超出阈值的部分
        self.equity_peak = max(self.equity_peak, self.equity)
        peak = self.equity_peak if self.equity_peak > eps else eps
        dd = (peak - self.equity) / peak
        dd_floor = getattr(self, 'dd_floor', 0.15)  # 超过 15% 才罚
        pen_dd = self.lam_dd * (max(0.0, dd - dd_floor) ** 2) if hasattr(self, 'lam_dd') else 0.0

        # （2）保证金占比惩罚：只罚超额
        mr_floor = getattr(self, 'mr_floor', 0.30)
        if self.margin_ratio < 0:
            print(0 / 0)
        pen_mr = self.lam_mr * (max(0.0, self.margin_ratio - mr_floor) ** 2) if hasattr(self, 'lam_mr') else 0.0

        # （3）交易惩罚：仅罚超出“免费额度”的新增成交
        # 若已把手续费/滑点计入 equity，建议把 lam_trade_cnt 设为 0
        new_trades = max(0, len(self.Trades) - self.last_trade_cnt)
        self.last_trade_cnt = len(self.Trades)
        free_trades = getattr(self, 'free_trades_per_step', 1)  # 每步允许 1 笔不罚
        pen_trade_cnt = self.lam_trade_cnt * max(0, new_trades - free_trades) if hasattr(self, 'lam_trade_cnt') else 0.0


        # --- 合成与剪裁 ---
        penalties = pen_dd + pen_mr + pen_trade_cnt
        r = r_base - penalties

        # print(f"r_base = {r_base}, pen_dd = {pen_dd}, pen_mr = {pen_mr}, pen_trade = {pen_trade_cnt}, r = {r}")

        # 适度剪裁保障稳定
        return float(np.clip(r, -0.10, 0.10))


    def if_truncated(self):
        if self.equity / self.init_capital < 0.05:
            return True
        return False
    
    # 符合环境, 对action采取对应操作 -> (next_state, reward, terminated, truncated)
    def step(self, action: int, ts, close) -> tuple:
        self.time_list.append(ts)
        # 先采取acction的操作: (还未写)
        action_dict = {
            0: 'HOLD',
            1: 'OPEN_LONG_1x',
            2: 'OPEN_LONG_2x',
            3: 'OPEN_SHORT_1x',
            4: 'OPEN_SHORT_2x',
            5: 'CLOSE_ALL',
        }
        assert action in range(6), 'action错误'
        act = action_dict[action]

        # print(f"act = {act}")

        if act == "OPEN_LONG_1x":
            if self.pos_dir == -1:
                self.flip_short_to_long(1, ts, self.call, self.put)
            elif self.pos_dir in [0, 1]:
                self.open_long_pair(1, ts, self.call, self.put)
        elif act == "OPEN_LONG_2x":
            if self.pos_dir == -1:
                self.flip_short_to_long(2, ts, self.call, self.put)
            elif self.pos_dir in [0, 1]:
                self.open_long_pair(2, ts, self.call, self.put)
        elif act == "OPEN_SHORT_1x":
            if self.pos_dir == 1:
                self.flip_long_to_short(1, ts, self.call, self.put)
            else:
                self.open_short_pair(1, ts, self.call, self.put)
        elif act == "OPEN_SHORT_2x":
            if self.pos_dir == 1:
                self.flip_long_to_short(2, ts, self.call, self.put)
            else:
                self.open_short_pair(2, ts, self.call, self.put)
        elif act == "CLOSE_ALL":
            self.close_pair_fully(ts, self.call, self.put)
        
        # 订单执行
        self.simulate_fill(ts)

        # 更新持仓信息
        self.update_positions(ts)

        if self.call in self.positions:
            direction, volume, value = self.positions[self.call]

            if direction == '买入开仓':
                self.pos_dir = 1
                self.pos_size = volume
            elif direction == '卖出开仓':
                self.pos_dir = -1
                self.pos_size = volume
        else:
            self.pos_size = 0
            self.pos_dir = 0

        self.equity_list.append(self.equity)
        self.equity_peak = max(self.equity_peak, self.equity)
        self.down = max(self.down, (self.equity_peak - self.equity) / self.equity_peak)

        if self.last_close is None:
            self.target_gain = 0
        else:
            self.target_gain = math.log(close / self.last_close)
        self.last_close = close
        self.target_price = close

        self.call_price = self.real_info_controller.get_close_by_str(self.call, ts)
        self.put_price = self.real_info_controller.get_close_by_str(self.put, ts)

        # 账户持仓信息: code -> (direction, volume, value)
        if self.call in self.positions and self.put in self.positions:
            call_d, call_volume, call_value = self.positions[self.call]
            put_d, put_volume, put_value = self.positions[self.put]

            assert (call_volume == put_volume or call_d == put_d), "组合数量异常"

            if call_d == '买入开仓':
                self.pos_dir = 1
            elif call_d == '卖出开仓':
                self.pos_dir = -1

            self.pos_size = call_volume
        else:
            self.pos_dir, self.pos_size = 0, 0
        
        if abs(self.frozen_money) < self.eps:
            self.frozen_money = 0

        self.cash_ratio = self.free_money / self.equity
        self.margin_ratio = self.frozen_money / self.equity
        expire = self.option_info_controller.get_expireDate(self.call)
        self.ttm = self.real_info_controller.get_ttm(ts, expire)
        reward = self.getReward()

        if abs(self.cash_ratio) < self.eps:
            self.cash_ratio = 0


        self.state = {
            'target_gain': self.target_gain,
            'call_price': self.call_price / self.target_price,
            'put_price': self.put_price / self.target_price,
            'ttm': self.ttm,
            'pos_dir': self.pos_dir,
            'pos_size': self.pos_size,
            'cash_ratio': self.cash_ratio,
            'margin_ratio': self.margin_ratio,
            'strike_price': self.strike / self.target_price,
            # 'target_price': self.target_price
            'draw_down': self.down
        }

        self.info = {
            "message": "test"
        }

        truncated = self.if_truncated()

        return self.getState(), reward, truncated


"""
    510050: K线: 20180504~20251017, 期权最早20171123
"""

if __name__ == '__main__':
    start_time = '20250825100000'
    end_time = '20250924150000'

    call = '10008800'
    put = '10008809'
    account = Account(100000, call, put, 1.3, '30m', ['510050'])
    data = account.real_info_controller.get_bars_between_from_df('510050', start_time, end_time)
    account.init_state(start_time, data.loc[0]['close'])
    
    # actions = '13331542504010000212422242040000202000020212000200000000220000002002000022020040220020200000220042200000000040202020400000020400000040200440024202202202200204022000000002200224'
    

    actions = '04445'

    for idx, ac in enumerate(actions):
        # ac = int(ac)
        # ac = random.randint(0, 5)
        ac = int(ac)
        ts, close = data.iloc[idx]
        ts = str(ts)
        ivs = [' ', '-', ':']

        for item in ivs:
            ts = ts.replace(item, '')

        account.step(ac, ts, close)
    print(0/ 0)
    # account.main('20200122100000', '20200123100000', '510050')

