
"""
Baselines for 50ETF straddle trading using ONLY the "three-piece" backtest framework:
- optionBaseInfo.py
- realInfo.py
- single_window_account_fast.py  (account + execution simulator)

Core ideas implemented:
1) Always Long ATM Straddle (roll when near expiry)
2) Always Short ATM Straddle (roll when near expiry)
3) VRP Timing (Implied Vol - Realized Vol spread): buy/sell/flat
4) IV Term-Structure Slope Timing (Jones & Wang-style idea): slope = IV(next) - IV(near)

Notes:
- Execution follows your PPO step convention: simulate_fill() first, then submit orders, then update_positions().
  That implies orders submitted at bar t are filled at bar t+1 (one-bar latency).
- Underlying (stock) trading is NOT supported by single_Account.simulate_fill() currently,
  so these baselines do NOT delta-hedge with the ETF.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from collections import defaultdict
from datetime import datetime, timedelta

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
    """Convert pandas Timestamp / datetime to 'YYYYMMDDHHMMSS' string format used in your framework."""
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
    # fallback: return raw lower
    return s

@dataclass
class Pair:
    call: str
    put: str
    strike: float
    expire: str  # YYYYMMDD
    dte: int     # days to expiry at selection time

class StraddleUniverse:
    """
    Metadata-only map:
        expiry_date -> strike -> {'call': code, 'put': code}
    """
    def __init__(self, account: single_Account, stock_code: str):
        self.stock_code = stock_code
        opt = account.option_info_controller

        self.expiry_strike_map: Dict[str, Dict[float, Dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
        for code in opt.get_option_list(stock_code):
            exp = ymd(opt.get_expireDate(code))
            strike = float(opt.get_strikePrice(code))
            t = normalize_opt_type(opt.get_optionType(code))
            if t in ("call", "put"):
                # keep first seen per type (usually unique anyway)
                if t not in self.expiry_strike_map[exp][strike]:
                    self.expiry_strike_map[exp][strike][t] = str(code)

        # only keep strikes that have BOTH call+put
        cleaned = defaultdict(dict)
        for exp, smap in self.expiry_strike_map.items():
            for k, d in smap.items():
                if "call" in d and "put" in d:
                    cleaned[exp][k] = d
        self.expiry_strike_map = cleaned

    def select_atm_pair(
        self,
        account: single_Account,
        ts_str: str,
        spot: float,
        min_dte: int = 7,
        max_dte: int = 60,
        after_expire: Optional[str] = None,
        require_liquidity: bool = True,
    ) -> Optional[Pair]:
        """
        Choose nearest-expiry ATM straddle under [min_dte, max_dte], optionally strictly after a given expiry.
        Also checks that both legs have non-zero close & volume at ts_str to avoid "0 price" artifacts.
        """
        now = parse_ymd(ts_str[:8])

        after_dt = parse_ymd(after_expire) if after_expire else None

        # candidate expiries
        expiries = []
        for exp in self.expiry_strike_map.keys():
            exp_dt = parse_ymd(exp)
            if after_dt and exp_dt <= after_dt:
                continue
            dte = (exp_dt - now).days
            if dte < min_dte or dte > max_dte:
                continue
            expiries.append((exp_dt, exp))
        expiries.sort(key=lambda x: x[0])

        for _, exp in expiries:
            strikes = list(self.expiry_strike_map[exp].keys())
            strikes.sort(key=lambda k: abs(float(k) - float(spot)))

            for k in strikes:
                call = self.expiry_strike_map[exp][k]["call"]
                put  = self.expiry_strike_map[exp][k]["put"]

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

    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
    }


# -----------------------
# Strategies
# -----------------------

class StrategyBase:
    def reset(self, account: single_Account, df_target: pd.DataFrame, start_time: str, end_time: str):
        raise NotImplementedError

    def on_bar(self, account: single_Account, ts_str: str, spot: float):
        raise NotImplementedError


class RollStraddleBase(StrategyBase):
    def __init__(
        self,
        stock_code: str = "510050",
        target_vol: int = 10,
        roll_days: int = 3,
        min_dte: int = 7,
        max_dte: int = 60,
    ):
        self.stock_code = stock_code
        self.target_vol = int(target_vol)
        self.roll_days = int(roll_days)
        self.min_dte = int(min_dte)
        self.max_dte = int(max_dte)

        self.universe: Optional[StraddleUniverse] = None
        self.pair: Optional[Pair] = None
        self.pending_roll_to: Optional[Pair] = None

    def _has_leg_position(self, account: single_Account, code: str) -> bool:
        return code in account.positions and account.positions[code][1] > 0

    def _pair_flat(self, account: single_Account, pair: Pair) -> bool:
        return (not self._has_leg_position(account, pair.call)) and (not self._has_leg_position(account, pair.put))

    def _need_roll(self, ts_str: str) -> bool:
        if not self.pair:
            return True
        now = parse_ymd(ts_str[:8])
        exp = parse_ymd(self.pair.expire)
        return (exp - now).days <= self.roll_days

    def _ensure_pair(self, account: single_Account, ts_str: str, spot: float):
        if self.universe is None:
            self.universe = StraddleUniverse(account, self.stock_code)

        if self.pair is None:
            self.pair = self.universe.select_atm_pair(
                account, ts_str, spot, min_dte=self.min_dte, max_dte=self.max_dte
            )
            if self.pair:
                account.set_combos(self.pair.call, self.pair.put)
                account.preload_data(self._bt_start_time, self._bt_end_time)

    def reset(self, account: single_Account, df_target: pd.DataFrame, start_time: str, end_time: str):
        self._bt_start_time = start_time
        self._bt_end_time = end_time
        self.universe = StraddleUniverse(account, self.stock_code)
        self.pair = None
        self.pending_roll_to = None


class AlwaysLongStraddle(RollStraddleBase):
    def on_bar(self, account: single_Account, ts_str: str, spot: float):
        self._ensure_pair(account, ts_str, spot)
        if not self.pair:
            return

        # if rolling needed: close current; schedule next expiry
        if self.pending_roll_to is None and self._need_roll(ts_str):
            nxt = self.universe.select_atm_pair(
                account, ts_str, spot, min_dte=self.min_dte, max_dte=self.max_dte,
                after_expire=self.pair.expire
            )
            if nxt:
                self.pending_roll_to = nxt
                account.close_pair(ts_str, self.pair.call, self.pair.put, w=1.0)

        # complete roll once flat
        if self.pending_roll_to is not None and self._pair_flat(account, self.pair):
            self.pair = self.pending_roll_to
            self.pending_roll_to = None
            account.set_combos(self.pair.call, self.pair.put)
            account.preload_data(self._bt_start_time, self._bt_end_time)
            account.open_long_pair(self.target_vol, ts_str, self.pair.call, self.pair.put)

        # normal: ensure long
        pos_dir, pos_size = account.get_pair_position(self.pair.call, self.pair.put)
        if pos_dir == 0 and self.pending_roll_to is None:
            account.open_long_pair(self.target_vol, ts_str, self.pair.call, self.pair.put)


class AlwaysShortStraddle(RollStraddleBase):
    def on_bar(self, account: single_Account, ts_str: str, spot: float):
        self._ensure_pair(account, ts_str, spot)
        if not self.pair:
            return

        if self.pending_roll_to is None and self._need_roll(ts_str):
            nxt = self.universe.select_atm_pair(
                account, ts_str, spot, min_dte=self.min_dte, max_dte=self.max_dte,
                after_expire=self.pair.expire
            )
            if nxt:
                self.pending_roll_to = nxt
                account.close_pair(ts_str, self.pair.call, self.pair.put, w=1.0)

        if self.pending_roll_to is not None and self._pair_flat(account, self.pair):
            self.pair = self.pending_roll_to
            self.pending_roll_to = None
            account.set_combos(self.pair.call, self.pair.put)
            account.preload_data(self._bt_start_time, self._bt_end_time)
            account.open_short_pair(self.target_vol, ts_str, self.pair.call, self.pair.put)

        pos_dir, pos_size = account.get_pair_position(self.pair.call, self.pair.put)
        if pos_dir == 0 and self.pending_roll_to is None:
            account.open_short_pair(self.target_vol, ts_str, self.pair.call, self.pair.put)


class VRPTimingStraddle(RollStraddleBase):
    """
    Simple VRP timing:
      spread = avg(IV_call, IV_put) - HV160
      if spread > +th: short straddle
      if spread < -th: long straddle
      else: flat
    """
    def __init__(self, *args, spread_th: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.spread_th = float(spread_th)

    def reset(self, account: single_Account, df_target: pd.DataFrame, start_time: str, end_time: str):
        super().reset(account, df_target, start_time, end_time)
        # HV cache used as "realized vol" proxy (annualized)
        account.init_hv160(start_time, end_time, self.stock_code)

    def on_bar(self, account: single_Account, ts_str: str, spot: float):
        self._ensure_pair(account, ts_str, spot)
        if not self.pair:
            return

        # roll management identical
        if self.pending_roll_to is None and self._need_roll(ts_str):
            nxt = self.universe.select_atm_pair(
                account, ts_str, spot, min_dte=self.min_dte, max_dte=self.max_dte,
                after_expire=self.pair.expire
            )
            if nxt:
                self.pending_roll_to = nxt
                account.close_pair(ts_str, self.pair.call, self.pair.put, w=1.0)

        if self.pending_roll_to is not None and self._pair_flat(account, self.pair):
            self.pair = self.pending_roll_to
            self.pending_roll_to = None
            account.set_combos(self.pair.call, self.pair.put)
            account.preload_data(self._bt_start_time, self._bt_end_time)

        # compute signals (requires init_state already called by runner)
        iv = 0.5 * (float(account.comb.get("call_iv", 0.0)) + float(account.comb.get("put_iv", 0.0)))
        hv = float(account.get_hv_160(ts_str) or 0.0)
        if iv <= 0 or hv <= 0:
            return

        spread = iv - hv
        pos_dir, pos_size = account.get_pair_position(self.pair.call, self.pair.put)

        if spread > self.spread_th:
            # short
            if pos_dir >= 0:
                account.flip_long_to_short(self.target_vol, ts_str, self.pair.call, self.pair.put)
        elif spread < -self.spread_th:
            # long
            if pos_dir <= 0:
                account.flip_short_to_long(self.target_vol, ts_str, self.pair.call, self.pair.put)
        else:
            # flat
            if pos_dir != 0:
                account.close_pair(ts_str, self.pair.call, self.pair.put, w=1.0)


class IVTermSlopeTiming(RollStraddleBase):
    """
    IV term-structure slope timing:
      slope = IV(next_maturity) - IV(near_maturity)
      if slope is high -> long near straddle
      if slope is low  -> short near straddle
      else -> flat

    Near maturity selection: [min_dte_near, max_dte_near]
    Next maturity selection: [min_dte_next, max_dte_next]
    """
    def __init__(
        self,
        *args,
        min_dte_near: int = 7,
        max_dte_near: int = 30,
        min_dte_next: int = 31,
        max_dte_next: int = 90,
        slope_hi: float = 0.02,
        slope_lo: float = -0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_dte_near = int(min_dte_near)
        self.max_dte_near = int(max_dte_near)
        self.min_dte_next = int(min_dte_next)
        self.max_dte_next = int(max_dte_next)
        self.slope_hi = float(slope_hi)
        self.slope_lo = float(slope_lo)

    def on_bar(self, account: single_Account, ts_str: str, spot: float):
        if self.universe is None:
            self.universe = StraddleUniverse(account, self.stock_code)

        # pick near pair (this becomes the trade instrument)
        if self.pair is None:
            near = self.universe.select_atm_pair(
                account, ts_str, spot, min_dte=self.min_dte_near, max_dte=self.max_dte_near
            )
            if near:
                self.pair = near
                account.set_combos(near.call, near.put)
                account.preload_data(self._bt_start_time, self._bt_end_time)

        if not self.pair:
            return

        # roll near pair if too close to expiry
        if self.pending_roll_to is None and self._need_roll(ts_str):
            nxt_near = self.universe.select_atm_pair(
                account, ts_str, spot, min_dte=self.min_dte_near, max_dte=self.max_dte_near,
                after_expire=self.pair.expire
            )
            if nxt_near:
                self.pending_roll_to = nxt_near
                account.close_pair(ts_str, self.pair.call, self.pair.put, w=1.0)

        if self.pending_roll_to is not None and self._pair_flat(account, self.pair):
            self.pair = self.pending_roll_to
            self.pending_roll_to = None
            account.set_combos(self.pair.call, self.pair.put)
            account.preload_data(self._bt_start_time, self._bt_end_time)

        # select next maturity pair just for IV measurement (no trading)
        nxt = self.universe.select_atm_pair(
            account, ts_str, spot, min_dte=self.min_dte_next, max_dte=self.max_dte_next
        )
        if not nxt:
            return

        # near IV from cached init_state (runner calls init_state before on_bar)
        iv_near = 0.5 * (float(account.comb.get("call_iv", 0.0)) + float(account.comb.get("put_iv", 0.0)))
        if iv_near <= 0:
            return

        # next IV: compute via RealInfo.cal_greeks (safe fallback if not preloaded)
        opt = account.option_info_controller
        ri = account.real_info_controller

        # call
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

        pos_dir, pos_size = account.get_pair_position(self.pair.call, self.pair.put)
        if slope >= self.slope_hi:
            # long near straddle
            if pos_dir <= 0:
                account.flip_short_to_long(self.target_vol, ts_str, self.pair.call, self.pair.put)
        elif slope <= self.slope_lo:
            # short near straddle
            if pos_dir >= 0:
                account.flip_long_to_short(self.target_vol, ts_str, self.pair.call, self.pair.put)
        else:
            # flat
            if pos_dir != 0:
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
) -> Tuple[pd.DataFrame, Dict[str, float], single_Account]:
    """
    Returns:
      - df_curve: columns [ts_str, spot, equity]
      - metrics: dict
      - account: to inspect Orders/Trades/positions
    """
    account = single_Account(init_capital, fee=fee, period=period, stockList=[stock_code], filepath=dataset_root)

    # underlying bars
    try:
        df = account.real_info_controller.get_bars_between(stock_code, start_time, end_time, period, columns=("ts", "close", "volume"))
    except Exception:
        df = account.real_info_controller.get_bars_between_from_df(stock_code, start_time, end_time, period, columns=("ts", "close", "volume"))

    if df.empty:
        raise RuntimeError("Underlying Kline is empty. Check your datasets path / time range / period.")

    strategy.reset(account, df, start_time, end_time)

    equity = []
    ts_list = []
    spot_list = []

    # Initialize state once (needed so comb has iv for strategies that read account.comb['*_iv'])
    first_ts_str = ts_to_str(df.iloc[0].ts)
    first_spot = float(df.iloc[0].close)
    account.init_state(first_ts_str, first_spot)

    for row in df.itertuples(index=False):
        ts_str = ts_to_str(row.ts)
        spot = float(row.close)

        # framework step order (same as your RL step): init_state -> simulate_fill -> (strategy submits orders) -> update_positions
        account.init_state(ts_str, spot)
        account.simulate_fill(ts_str)

        strategy.on_bar(account, ts_str, spot)

        account.update_positions(ts_str)

        ts_list.append(ts_str)
        spot_list.append(spot)
        equity.append(float(account.equity))

    df_curve = pd.DataFrame({"ts_str": ts_list, "spot": spot_list, "equity": equity})
    metrics = compute_metrics(np.array(equity, dtype=float), bars_per_year=2016)

    return df_curve, metrics, account


def save_results(df_curve: pd.DataFrame, metrics: Dict[str, float], out_path: str):
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path) as w:
        df_curve.to_excel(w, index=False, sheet_name="equity_curve")
        pd.DataFrame([metrics]).to_excel(w, index=False, sheet_name="metrics")


if __name__ == "__main__":
    # Example: run 1 year (edit dates to the year you want)
    stock = "510050"
    start = "20240102093000"
    end   = "20241231150000"

    # 1) Always long straddle
    strat = AlwaysLongStraddle(stock_code=stock, target_vol=10, roll_days=3, min_dte=7, max_dte=60)
    curve, metrics, acc = run_backtest(strat, stock, start, end, init_capital=100000, period="30m", dataset_root="./miniQMT/datasets")
    print("AlwaysLongStraddle metrics:", metrics)
    save_results(curve, metrics, "./miniQMT/datasets/outs/baseline_always_long.xlsx")

    # 2) Always short straddle
    strat = AlwaysShortStraddle(stock_code=stock, target_vol=10, roll_days=3, min_dte=7, max_dte=60)
    curve, metrics, acc = run_backtest(strat, stock, start, end, init_capital=100000, period="30m", dataset_root="./miniQMT/datasets")
    print("AlwaysShortStraddle metrics:", metrics)
    save_results(curve, metrics, "./miniQMT/datasets/outs/baseline_always_short.xlsx")

    # 3) VRP timing
    strat = VRPTimingStraddle(stock_code=stock, target_vol=10, roll_days=3, min_dte=7, max_dte=60, spread_th=0.05)
    curve, metrics, acc = run_backtest(strat, stock, start, end, init_capital=100000, period="30m", dataset_root="./miniQMT/datasets")
    print("VRPTimingStraddle metrics:", metrics)
    save_results(curve, metrics, "./miniQMT/datasets/outs/baseline_vrp_timing.xlsx")

    # 4) IV term slope timing
    strat = IVTermSlopeTiming(stock_code=stock, target_vol=10, roll_days=3,
                              min_dte_near=7, max_dte_near=30, min_dte_next=31, max_dte_next=90,
                              slope_hi=0.02, slope_lo=-0.02)
    curve, metrics, acc = run_backtest(strat, stock, start, end, init_capital=100000, period="30m", dataset_root="./miniQMT/datasets")
    print("IVTermSlopeTiming metrics:", metrics)
    save_results(curve, metrics, "./miniQMT/datasets/outs/baseline_iv_term_slope.xlsx")
