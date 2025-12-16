
"""
Baselines for 50ETF straddle trading using ONLY the "three-piece" backtest framework:
- optionBaseInfo.py
- realInfo.py
- single_window_account_fast.py

Important fix vs v1:
- We MUST set the (call, put) into account.comb BEFORE calling account.init_state(),
  because init_state reads self.comb['call'/'put'] and uses cached Greeks.
- To keep your PPO integration intact, we do NOT change single_Account; instead we import
  single_account_baseline_ext.py which monkey-patches helper wrappers.

Baselines implemented (1-year backtest):
B1) Always Long ATM near-month straddle, rolling before expiry
B2) Always Short ATM near-month straddle, rolling before expiry
B3) VRP Timing: avg(IV_call,IV_put) - HV160 -> long/short/flat
B4) IV term-structure slope timing: IV(next) - IV(near) -> long/short/flat

NOTE: Delta-hedging with underlying is NOT implemented because single_Account.simulate_fill()
      currently rejects 6/7-digit codes (e.g., 510050 ETF). So we keep option-only baselines.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from collections import defaultdict
from datetime import datetime

import os
import math
import numpy as np
import pandas as pd

import sys
# 导入路径修改正
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir))) 
miniqmt_dir = os.path.join(project_root_dir, 'miniQMT') 
if miniqmt_dir not in sys.path:
    sys.path.append(miniqmt_dir)
from DL.finTool.single_window_account_fast import single_Account 

# -----------------------
# Helpers
# -----------------------

def ts_to_str(ts) -> str:
    """pandas Timestamp/datetime -> 'YYYYMMDDHHMMSS' string"""
    return str(ts).replace(' ', '').replace('-', '').replace(':', '')

def ymd(s: str) -> str:
    return str(s).strip()[:8]

def parse_ymd(s: str) -> datetime:
    return datetime.strptime(ymd(s), "%Y%m%d")

def normalize_opt_type(x) -> str:
    s = str(x).lower()
    if "call" in s or "认购" in s or s in ("c",):
        return "call"
    if "put" in s or "认沽" in s or s in ("p",):
        return "put"
    return s

@dataclass
class Pair:
    call: str
    put: str
    strike: float
    expire: str  # YYYYMMDD
    dte: int


class StraddleUniverse:
    """
    expiry_date -> strike -> {'call': code, 'put': code}
    """
    def __init__(self, account: single_Account, stock_code: str):
        self.stock_code = stock_code
        opt = account.option_info_controller

        m: Dict[str, Dict[float, Dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
        for code in opt.get_option_list(stock_code):
            exp = ymd(opt.get_expireDate(code))
            strike = float(opt.get_strikePrice(code))
            t = normalize_opt_type(opt.get_optionType(code))
            if t in ("call", "put"):
                if t not in m[exp][strike]:
                    m[exp][strike][t] = str(code)

        cleaned = defaultdict(dict)
        for exp, smap in m.items():
            for k, d in smap.items():
                if "call" in d and "put" in d:
                    cleaned[exp][k] = d
        self.map = cleaned

    def select_atm_pair(
        self,
        account: single_Account,
        ts_str: str,
        spot: float,
        min_dte: int,
        max_dte: int,
        after_expire: Optional[str] = None,
        require_liquidity: bool = True,
    ) -> Optional[Pair]:
        now = parse_ymd(ts_str[:8])
        after_dt = parse_ymd(after_expire) if after_expire else None

        expiries = []
        for exp in self.map.keys():
            exp_dt = parse_ymd(exp)
            if after_dt and exp_dt <= after_dt:
                continue
            dte = (exp_dt - now).days
            if dte < min_dte or dte > max_dte:
                continue
            expiries.append((exp_dt, exp))
        expiries.sort(key=lambda x: x[0])

        for _, exp in expiries:
            strikes = list(self.map[exp].keys())
            strikes.sort(key=lambda k: abs(float(k) - float(spot)))
            for k in strikes:
                call = self.map[exp][k]["call"]
                put  = self.map[exp][k]["put"]

                if require_liquidity:
                    pc = float(account.getClosePrice(call, ts_str) or 0.0)
                    pp = float(account.getClosePrice(put, ts_str) or 0.0)
                    vc = int(account.getRealVolume(call, ts_str) or 0)
                    vp = int(account.getRealVolume(put, ts_str) or 0)
                    if pc <= 0 or pp <= 0 or vc <= 0 or vp <= 0:
                        continue

                dte = (parse_ymd(exp) - now).days
                return Pair(call=call, put=put, strike=float(k), expire=exp, dte=int(dte))

        return None


def compute_metrics(equity: np.ndarray, bars_per_year: int = 2016) -> Dict[str, float]:
    equity = np.asarray(equity, dtype=float)
    if len(equity) < 2:
        return {"total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}

    rets = np.diff(np.log(np.maximum(equity, 1e-12)))
    total_return = equity[-1] / equity[0] - 1.0

    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-12)
    max_dd = float(np.max(dd))

    mu = float(np.mean(rets))
    sig = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    sharpe = 0.0 if sig <= 1e-12 else (mu / sig) * math.sqrt(bars_per_year)

    return {"total_return": float(total_return), "max_drawdown": float(max_dd), "sharpe": float(sharpe)}


# -----------------------
# Strategy API aligned with single_Account.step()
# init_state -> simulate_fill -> strategy_action -> update_positions
# -----------------------

class StrategyBase:
    def reset(self, account: single_Account, start_time: str, end_time: str):
        raise NotImplementedError

    def select_pair_before_init_state(self, account: single_Account, ts_str: str, spot: float):
        """Must ensure account.comb['call'/'put'] are set BEFORE init_state."""
        raise NotImplementedError

    def act_after_fill(self, account: single_Account, ts_str: str, spot: float):
        """Place orders after simulate_fill."""
        raise NotImplementedError


class RollStraddleBase(StrategyBase):
    def __init__(self, stock_code="510050", target_vol=10, roll_days=3, min_dte=7, max_dte=60):
        self.stock_code = stock_code
        self.target_vol = int(target_vol)
        self.roll_days = int(roll_days)
        self.min_dte = int(min_dte)
        self.max_dte = int(max_dte)

        self.univ: Optional[StraddleUniverse] = None
        self.pair: Optional[Pair] = None
        self.pending_roll_to: Optional[Pair] = None
        self._start = None
        self._end = None

    def reset(self, account: single_Account, start_time: str, end_time: str):
        self._start = start_time
        self._end = end_time
        self.univ = StraddleUniverse(account, self.stock_code)
        self.pair = None
        self.pending_roll_to = None

    def _pair_flat(self, account: single_Account, pair: Pair) -> bool:
        return (not account.has_positions(pair.call)) and (not account.has_positions(pair.put))

    def _need_roll(self, ts_str: str) -> bool:
        if not self.pair:
            return True
        now = parse_ymd(ts_str[:8])
        exp = parse_ymd(self.pair.expire)
        return (exp - now).days <= self.roll_days

    def _choose_initial_pair(self, account: single_Account, ts_str: str, spot: float):
        if self.pair is None:
            self.pair = self.univ.select_atm_pair(
                account, ts_str, spot, min_dte=self.min_dte, max_dte=self.max_dte
            )

    def select_pair_before_init_state(self, account: single_Account, ts_str: str, spot: float):
        # Choose pair if none
        self._choose_initial_pair(account, ts_str, spot)
        if not self.pair:
            return

        # If rolling needed: decide next pair and trigger close
        if self.pending_roll_to is None and self._need_roll(ts_str):
            nxt = self.univ.select_atm_pair(
                account, ts_str, spot, min_dte=self.min_dte, max_dte=self.max_dte,
                after_expire=self.pair.expire
            )
            if nxt:
                self.pending_roll_to = nxt
                # We close in act_after_fill (align with "action after fill")
                # but init_state must still have a valid comb; keep current pair for now.

        # Ensure comb is set to CURRENT pair for init_state
        account.ensure_combos_loaded(self.pair.call, self.pair.put, self._start, self._end)

    def _complete_roll_if_flat(self, account: single_Account, ts_str: str, spot: float):
        if self.pending_roll_to is not None and self._pair_flat(account, self.pair):
            self.pair = self.pending_roll_to
            self.pending_roll_to = None
            account.ensure_combos_loaded(self.pair.call, self.pair.put, self._start, self._end)


class AlwaysLongStraddle(RollStraddleBase):
    def act_after_fill(self, account: single_Account, ts_str: str, spot: float):
        if not self.pair:
            return

        # roll: close current if needed
        if self.pending_roll_to is not None:
            account.close_pair(ts_str, self.pair.call, self.pair.put, w=1.0)

        # complete roll if positions are flat
        self._complete_roll_if_flat(account, ts_str, spot)
        if not self.pair:
            return

        # ensure long
        d, sz = account.get_pair_position(self.pair.call, self.pair.put)
        if d == 0 and self.pending_roll_to is None:
            account.open_long_pair(self.target_vol, ts_str, self.pair.call, self.pair.put)


class AlwaysShortStraddle(RollStraddleBase):
    def act_after_fill(self, account: single_Account, ts_str: str, spot: float):
        if not self.pair:
            return

        if self.pending_roll_to is not None:
            account.close_pair(ts_str, self.pair.call, self.pair.put, w=1.0)

        self._complete_roll_if_flat(account, ts_str, spot)
        if not self.pair:
            return

        d, sz = account.get_pair_position(self.pair.call, self.pair.put)
        if d == 0 and self.pending_roll_to is None:
            account.open_short_pair(self.target_vol, ts_str, self.pair.call, self.pair.put)


class VRPTimingStraddle(RollStraddleBase):
    def __init__(self, *args, spread_th=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.spread_th = float(spread_th)

    def reset(self, account: single_Account, start_time: str, end_time: str):
        super().reset(account, start_time, end_time)
        # hv cache for realized vol proxy
        account.init_hv160(start_time, end_time, self.stock_code)

    def act_after_fill(self, account: single_Account, ts_str: str, spot: float):
        if not self.pair:
            return

        # roll close
        if self.pending_roll_to is not None:
            account.close_pair(ts_str, self.pair.call, self.pair.put, w=1.0)

        self._complete_roll_if_flat(account, ts_str, spot)
        if not self.pair:
            return

        # Signal uses current comb IV from init_state (must have been called)
        iv = 0.5 * (float(account.comb.get("call_iv", 0.0)) + float(account.comb.get("put_iv", 0.0)))
        hv = float(account.get_hv_160(ts_str) or 0.0)
        if iv <= 0 or hv <= 0:
            return
        spread = iv - hv

        d, sz = account.get_pair_position(self.pair.call, self.pair.put)
        if spread > self.spread_th:
            if d >= 0:
                account.flip_long_to_short(self.target_vol, ts_str, self.pair.call, self.pair.put)
        elif spread < -self.spread_th:
            if d <= 0:
                account.flip_short_to_long(self.target_vol, ts_str, self.pair.call, self.pair.put)
        else:
            if d != 0:
                account.close_pair(ts_str, self.pair.call, self.pair.put, w=1.0)


class IVTermSlopeTiming(RollStraddleBase):
    def __init__(
        self, *args,
        min_dte_near=7, max_dte_near=30,
        min_dte_next=31, max_dte_next=90,
        slope_hi=0.02, slope_lo=-0.02,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.min_dte_near = int(min_dte_near)
        self.max_dte_near = int(max_dte_near)
        self.min_dte_next = int(min_dte_next)
        self.max_dte_next = int(max_dte_next)
        self.slope_hi = float(slope_hi)
        self.slope_lo = float(slope_lo)

    def reset(self, account: single_Account, start_time: str, end_time: str):
        self._start = start_time
        self._end = end_time
        self.univ = StraddleUniverse(account, self.stock_code)
        self.pair = None
        self.pending_roll_to = None

    def select_pair_before_init_state(self, account: single_Account, ts_str: str, spot: float):
        # Near pair is traded instrument
        if self.pair is None:
            self.pair = self.univ.select_atm_pair(
                account, ts_str, spot, min_dte=self.min_dte_near, max_dte=self.max_dte_near
            )
        if not self.pair:
            return

        if self.pending_roll_to is None and self._need_roll(ts_str):
            nxt_near = self.univ.select_atm_pair(
                account, ts_str, spot, min_dte=self.min_dte_near, max_dte=self.max_dte_near,
                after_expire=self.pair.expire
            )
            if nxt_near:
                self.pending_roll_to = nxt_near

        account.ensure_combos_loaded(self.pair.call, self.pair.put, self._start, self._end)

    def act_after_fill(self, account: single_Account, ts_str: str, spot: float):
        if not self.pair:
            return

        if self.pending_roll_to is not None:
            account.close_pair(ts_str, self.pair.call, self.pair.put, w=1.0)

        # if flat, switch to new near
        if self.pending_roll_to is not None and self._pair_flat(account, self.pair):
            self.pair = self.pending_roll_to
            self.pending_roll_to = None
            account.ensure_combos_loaded(self.pair.call, self.pair.put, self._start, self._end)

        # Pick next maturity for IV measurement
        nxt = self.univ.select_atm_pair(
            account, ts_str, spot, min_dte=self.min_dte_next, max_dte=self.max_dte_next
        )
        if not nxt:
            return

        iv_near = 0.5 * (float(account.comb.get("call_iv", 0.0)) + float(account.comb.get("put_iv", 0.0)))
        if iv_near <= 0:
            return

        # Next IV via RealInfo.cal_greeks (works without preload)
        opt = account.option_info_controller
        ri = account.real_info_controller

        g1 = ri.cal_greeks(ts_str, self.stock_code, nxt.call,
                           strike=float(opt.get_strikePrice(nxt.call)),
                           expire=ymd(opt.get_expireDate(nxt.call)),
                           op_type=normalize_opt_type(opt.get_optionType(nxt.call)),
                           period=account.period)
        g2 = ri.cal_greeks(ts_str, self.stock_code, nxt.put,
                           strike=float(opt.get_strikePrice(nxt.put)),
                           expire=ymd(opt.get_expireDate(nxt.put)),
                           op_type=normalize_opt_type(opt.get_optionType(nxt.put)),
                           period=account.period)
        iv_next = 0.5 * (float(g1.get("iv", 0.0)) + float(g2.get("iv", 0.0)))
        if iv_next <= 0:
            return

        slope = iv_next - iv_near
        d, sz = account.get_pair_position(self.pair.call, self.pair.put)

        if slope >= self.slope_hi:
            if d <= 0:
                account.flip_short_to_long(self.target_vol, ts_str, self.pair.call, self.pair.put)
        elif slope <= self.slope_lo:
            if d >= 0:
                account.flip_long_to_short(self.target_vol, ts_str, self.pair.call, self.pair.put)
        else:
            if d != 0:
                account.close_pair(ts_str, self.pair.call, self.pair.put, w=1.0)


# -----------------------
# Runner
# -----------------------

def run_backtest(
    strategy: StrategyBase,
    stock_code: str,
    start_time: str,
    end_time: str,
    init_capital: float = 100000.0,
    fee: float = 1.3,
    period: str = "30m",
    dataset_root: str = "./miniQMT/datasets",
):
    account = single_Account(init_capital, fee=fee, period=period, stockList=[stock_code], filepath=dataset_root)

    # Underlying bars
    try:
        df = account.real_info_controller.get_bars_between(stock_code, start_time, end_time, period, columns=("ts", "close", "volume"))
    except Exception:
        df = account.real_info_controller.get_bars_between_from_df(stock_code, start_time, end_time, period, columns=("ts", "close", "volume"))

    if df.empty:
        raise RuntimeError("Underlying Kline is empty. Check datasets path / time range / period.")

    strategy.reset(account, start_time, end_time)

    ts_list, spot_list, equity_list = [], [], []

    for row in df.itertuples(index=False):
        ts_str = ts_to_str(row.ts)
        spot = float(row.close)

        # 0) Strategy MUST ensure comb legs are set BEFORE init_state()
        strategy.select_pair_before_init_state(account, ts_str, spot)

        # 1) Update state (fast cache)
        account.init_state(ts_str, spot)

        # 2) Match previous orders
        account.simulate_fill(ts_str)

        # 3) Strategy acts (places new orders for next bar fill)
        strategy.act_after_fill(account, ts_str, spot)

        # 4) Update positions & equity
        account.update_positions(ts_str)

        ts_list.append(ts_str)
        spot_list.append(spot)
        equity_list.append(float(account.equity))

    curve = pd.DataFrame({"ts_str": ts_list, "spot": spot_list, "equity": equity_list})
    metrics = compute_metrics(np.array(equity_list, dtype=float), bars_per_year=2016)
    return curve, metrics, account


def save_results(curve: pd.DataFrame, metrics: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path) as w:
        curve.to_excel(w, index=False, sheet_name="equity_curve")
        pd.DataFrame([metrics]).to_excel(w, index=False, sheet_name="metrics")


if __name__ == "__main__":
    stock = "510050"
    start = "20240102093000"
    end   = "20241231150000"
    out_dir = "./miniQMT/datasets/outs"
    os.makedirs(out_dir, exist_ok=True)

    # B1
    strat = AlwaysLongStraddle(stock_code=stock, target_vol=10, roll_days=3, min_dte=7, max_dte=60)
    curve, metrics, _ = run_backtest(strat, stock, start, end, init_capital=100000, period="30m", dataset_root="./miniQMT/datasets")
    print("B1 AlwaysLongStraddle:", metrics)
    save_results(curve, metrics, os.path.join(out_dir, "baseline_B1_always_long.xlsx"))

    # B2
    strat = AlwaysShortStraddle(stock_code=stock, target_vol=10, roll_days=3, min_dte=7, max_dte=60)
    curve, metrics, _ = run_backtest(strat, stock, start, end, init_capital=100000, period="30m", dataset_root="./miniQMT/datasets")
    print("B2 AlwaysShortStraddle:", metrics)
    save_results(curve, metrics, os.path.join(out_dir, "baseline_B2_always_short.xlsx"))

    # B3
    strat = VRPTimingStraddle(stock_code=stock, target_vol=10, roll_days=3, min_dte=7, max_dte=60, spread_th=0.05)
    curve, metrics, _ = run_backtest(strat, stock, start, end, init_capital=100000, period="30m", dataset_root="./miniQMT/datasets")
    print("B3 VRPTimingStraddle:", metrics)
    save_results(curve, metrics, os.path.join(out_dir, "baseline_B3_vrp_timing.xlsx"))

    # B4
    strat = IVTermSlopeTiming(stock_code=stock, target_vol=10, roll_days=3,
                              min_dte_near=7, max_dte_near=30, min_dte_next=31, max_dte_next=90,
                              slope_hi=0.02, slope_lo=-0.02)
    curve, metrics, _ = run_backtest(strat, stock, start, end, init_capital=100000, period="30m", dataset_root="./miniQMT/datasets")
    print("B4 IVTermSlopeTiming:", metrics)
    save_results(curve, metrics, os.path.join(out_dir, "baseline_B4_iv_term_slope.xlsx"))
