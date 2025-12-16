# baseline_runner.py
import numpy as np

# 你的三件套
from miniQMT.DL.finTool.single_window_account import SingleWindowAccount
from miniQMT.DL.finTool.realInfo import RealInfo
from miniQMT.DL.finTool.optionBaseInfo import OptionBaseInfo


# ---------- 工具：绩效 ----------
def calc_performance(nav, bars_per_year=16*250):
    nav = np.asarray(nav, dtype=float)
    if len(nav) < 2:
        return {}
    ret = np.diff(nav) / nav[:-1]
    ann_ret = (1 + ret.mean()) ** bars_per_year - 1
    ann_vol = ret.std() * np.sqrt(bars_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(nav)
    mdd = ((nav - peak) / peak).min()
    return {"ann_return": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe), "max_drawdown": float(mdd)}


# ---------- 工具：选 ATM + 近月 ----------
def pick_near_month_atm_straddle(real: RealInfo, opt: OptionBaseInfo, min_days=5):
    """
    返回 (call_contract, put_contract)
    TODO：这里需要你 optionBaseInfo 提供的“合约列表 + 行权价 + 到期日 + call/put 标识”
    """
    # TODO 示例：opt.get_chain(date_time) -> list[contract]
    chain = opt.get_chain(real.dt)  # <- TODO：改成你的接口

    spot = real.spot  # <- TODO：改成 realInfo 里的字段
    # 过滤：到期天数 >= min_days
    chain = [c for c in chain if c.days_to_expiry(real.dt) >= min_days]  # <- TODO：改成你的接口

    # 找最近到期的 expiry
    expiries = sorted(set(c.expiry for c in chain))  # <- TODO
    if not expiries:
        return None, None
    near_exp = expiries[0]
    near = [c for c in chain if c.expiry == near_exp]

    # ATM strike：最接近 spot 的 strike
    strikes = sorted(set(c.strike for c in near))
    atm_k = min(strikes, key=lambda k: abs(k - spot))

    call = next((c for c in near if c.strike == atm_k and c.is_call), None)
    put  = next((c for c in near if c.strike == atm_k and (not c.is_call)), None)
    return call, put


# ---------- 策略基类 ----------
class BaseStrategy:
    def reset(self):
        pass

    def on_bar(self, real: RealInfo, opt: OptionBaseInfo, account: SingleWindowAccount):
        """
        返回 orders: list[dict]，每个 dict 描述一个交易
        例如：{"symbol":..., "qty":..., "side":"BUY"/"SELL", "price":...}
        """
        raise NotImplementedError


# ---------- B1：滚动 ATM 多头跨式 ----------
class RollingLongStraddle(BaseStrategy):
    def __init__(self, notional=1.0, roll_days=3):
        self.notional = notional
        self.roll_days = roll_days

    def on_bar(self, real, opt, account):
        orders = []

        call, put = pick_near_month_atm_straddle(real, opt)
        if call is None or put is None:
            return orders

        # TODO：判断当前是否已持有该跨式；或到期前 roll_days 触发换月
        # pos = account.get_position(call.symbol) ...
        need_roll = False  # <- TODO：用 account/contract 的到期日判断
        has_pos = account.has_straddle_position()  # <- TODO：改成你的接口

        if (not has_pos) or need_roll:
            # 先平旧仓（如有）
            orders += account.close_all_option_positions()  # <- TODO

            # 再开新跨式：买 call + 买 put
            call_px = opt.get_mid_price(call, real.dt)  # <- TODO
            put_px  = opt.get_mid_price(put, real.dt)   # <- TODO
            qty = self.notional  # 你可以映射成“张数/权重”，由 account 定义

            orders += [
                {"symbol": call.symbol, "side": "BUY", "qty": qty, "price": call_px},
                {"symbol": put.symbol,  "side": "BUY", "qty": qty, "price": put_px},
            ]

        return orders


# ---------- B2：Delta 对冲多头跨式 ----------
class DeltaHedgedLongStraddle(BaseStrategy):
    def __init__(self, notional=1.0, hedge_every_bar=True):
        self.notional = notional
        self.hedge_every_bar = hedge_every_bar

    def on_bar(self, real, opt, account):
        orders = []

        call, put = pick_near_month_atm_straddle(real, opt)
        if call is None or put is None:
            return orders

        # 1) 确保有跨式仓位（同 B1，可简化：若无则开）
        has_pos = account.has_straddle_position()  # <- TODO
        if not has_pos:
            call_px = opt.get_mid_price(call, real.dt)  # <- TODO
            put_px  = opt.get_mid_price(put, real.dt)   # <- TODO
            qty = self.notional
            orders += [
                {"symbol": call.symbol, "side": "BUY", "qty": qty, "price": call_px},
                {"symbol": put.symbol,  "side": "BUY", "qty": qty, "price": put_px},
            ]

        # 2) 计算组合 delta，并用标的对冲到 0
        if self.hedge_every_bar:
            # TODO：opt 或 realInfo 或 account 是否能给 greeks？
            # 典型做法：delta_total = qty_call*delta_call + qty_put*delta_put
            delta_total = account.get_portfolio_delta(real, opt)  # <- TODO
            target_underlying = -delta_total

            # TODO：用 account 下单标的（50ETF）
            orders += account.rebalance_underlying(target_underlying, real)  # <- TODO

        return orders


# ---------- B3：Delta 对冲卖出跨式（VRP） ----------
class DeltaHedgedShortStraddle(BaseStrategy):
    def __init__(self, notional=1.0, hedge_every_bar=True):
        self.notional = notional
        self.hedge_every_bar = hedge_every_bar

    def on_bar(self, real, opt, account):
        orders = []
        call, put = pick_near_month_atm_straddle(real, opt)
        if call is None or put is None:
            return orders

        has_pos = account.has_straddle_position()  # <- TODO
        if not has_pos:
            call_px = opt.get_mid_price(call, real.dt)  # <- TODO
            put_px  = opt.get_mid_price(put, real.dt)   # <- TODO
            qty = self.notional
            orders += [
                {"symbol": call.symbol, "side": "SELL", "qty": qty, "price": call_px},
                {"symbol": put.symbol,  "side": "SELL", "qty": qty, "price": put_px},
            ]

        if self.hedge_every_bar:
            delta_total = account.get_portfolio_delta(real, opt)  # <- TODO
            target_underlying = -delta_total
            orders += account.rebalance_underlying(target_underlying, real)  # <- TODO

        return orders


# ---------- B4：IV 期限结构择时：预测 RV 决定做多/做空/空仓 ----------
class IVTermStructureTimingStraddle(BaseStrategy):
    def __init__(self, notional=1.0, z_open=0.5, z_close=0.1):
        self.notional = notional
        self.z_open = z_open
        self.z_close = z_close

    def on_bar(self, real, opt, account):
        orders = []

        # TODO：计算近月/次近月 ATM IV，做一个 slope 或 zscore
        # iv_near = opt.get_atm_iv(expiry=near, dt=real.dt)
        # iv_next = opt.get_atm_iv(expiry=next, dt=real.dt)
        iv_near, iv_next = opt.get_near_next_atm_iv(real.dt)  # <- TODO
        slope = iv_near - iv_next

        # 简单规则：slope 高（近月更贵）=> 市场预期短期波动大：做多跨式
        # slope 低/负（近月更便宜）=> 做空跨式或空仓
        pos = account.get_straddle_net_position()  # <- TODO：正=多头跨式，负=空头跨式

        if slope > self.z_open and pos <= 0:
            orders += account.close_all_option_positions()  # <- TODO
            call, put = pick_near_month_atm_straddle(real, opt)
            if call and put:
                qty = self.notional
                orders += [
                    {"symbol": call.symbol, "side": "BUY", "qty": qty, "price": opt.get_mid_price(call, real.dt)},
                    {"symbol": put.symbol,  "side": "BUY", "qty": qty, "price": opt.get_mid_price(put,  real.dt)},
                ]

        elif slope < -self.z_open and pos >= 0:
            orders += account.close_all_option_positions()  # <- TODO
            call, put = pick_near_month_atm_straddle(real, opt)
            if call and put:
                qty = self.notional
                orders += [
                    {"symbol": call.symbol, "side": "SELL", "qty": qty, "price": opt.get_mid_price(call, real.dt)},
                    {"symbol": put.symbol,  "side": "SELL", "qty": qty, "price": opt.get_mid_price(put,  real.dt)},
                ]

        elif abs(slope) < self.z_close and pos != 0:
            # 信号消失：平仓
            orders += account.close_all_option_positions()  # <- TODO

        return orders


def run_one_year(data_iter, strategy: BaseStrategy, init_cash=1_000_000):
    """
    data_iter: 迭代器，每次产出 (real:RealInfo, opt:OptionBaseInfo)
    """
    account = SingleWindowAccount(init_cash=init_cash)  # <- TODO：按你构造函数
    strategy.reset()

    nav = []
    for real, opt in data_iter:
        orders = strategy.on_bar(real, opt, account)

        # 执行订单
        for od in orders:
            account.place_order(**od)  # <- TODO：对齐你 account 的下单接口

        # 盯市/结算
        account.mark_to_market(real, opt)  # <- TODO：对齐接口
        nav.append(account.get_nav())      # <- TODO：对齐接口

    return nav, calc_performance(nav)
