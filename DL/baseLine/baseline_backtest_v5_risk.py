
"""
baseline_backtest_v5_risk.py
在不改三件套任何原函数的前提下，提升 baseline 的“战斗力”：
- 卖跨式加入：趋势过滤 + 止损(premium multiple) + 冷却期
- 买跨式加入：IV 相对 HV 便宜才买（hysteresis）
仍按：init_state -> simulate_fill -> 下单 -> update_positions -> _update_comb_equity
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Deque
from collections import defaultdict, deque
from datetime import datetime
import os, sys, math
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
miniqmt_dir = os.path.join(project_root_dir, 'miniQMT')
if miniqmt_dir not in sys.path:
    sys.path.append(miniqmt_dir)

from DL.finTool.single_window_account_fast import single_Account

def ts_to_str(ts) -> str:
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

def digits_only(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit())

def is_listed_on(account: single_Account, code: str, ts_str: str) -> bool:
    opt = account.option_info_controller
    od = digits_only(opt.get_openDate(code))
    t  = digits_only(ts_str)
    if len(od) >= 14 and len(t) >= 14:
        return t[:14] >= od[:14]
    if len(od) >= 8 and len(t) >= 8:
        return t[:8] >= od[:8]
    return False

def set_pair_and_preload(account: single_Account, call: str, put: str, start_time: str, end_time: str):
    account.set_combos(call, put)
    account.preload_data(start_time, end_time)

def is_leg_flat(account: single_Account, code: str) -> bool:
    pos = account.positions.get(code)
    return (pos is None) or (int(pos[1]) <= 0)

def pair_premium(account: single_Account, ts_str: str, call: str, put: str) -> float:
    pc = float(account.getClosePrice(call, ts_str) or 0.0)
    pp = float(account.getClosePrice(put, ts_str) or 0.0)
    return pc + pp

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

@dataclass
class Pair:
    call: str
    put: str
    strike: float
    expire: str
    dte: int

class StraddleUniverse:
    def __init__(self, account: single_Account, stock_code: str):
        self.stock_code = stock_code
        opt = account.option_info_controller
        m: Dict[str, Dict[float, Dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
        for code in opt.get_option_list(stock_code):
            exp = ymd(opt.get_expireDate(code))
            strike = float(opt.get_strikePrice(code))
            t = normalize_opt_type(opt.get_optionType(code))
            if t in ("call", "put"):
                m[exp][strike][t] = str(code)
        cleaned = defaultdict(dict)
        for exp, smap in m.items():
            for k, d in smap.items():
                if "call" in d and "put" in d:
                    cleaned[exp][k] = d
        self.map = cleaned

    def select_atm_pair(self, account: single_Account, ts_str: str, spot: float,
                        min_dte: int, max_dte: int, after_expire: Optional[str]=None,
                        require_liquidity: bool=True) -> Optional[Pair]:
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
                call = self.map[exp][k]["call"]; put = self.map[exp][k]["put"]
                if (not is_listed_on(account, call, ts_str)) or (not is_listed_on(account, put, ts_str)):
                    continue

                if account.option_info_controller.get_multiplier(call) != 10000:
                    continue

                if account.option_info_controller.get_multiplier(put) != 10000:
                    continue


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

class StrategyBase:
    def reset(self, account: single_Account, start_time: str, end_time: str): ...
    def select_pair_before_init_state(self, account: single_Account, ts_str: str, spot: float): raise NotImplementedError
    def act_after_fill(self, account: single_Account, ts_str: str, spot: float): raise NotImplementedError

class RollStraddleBase(StrategyBase):
    def __init__(self, stock_code="510050", target_vol=3, roll_days=3, min_dte=14, max_dte=60):
        self.stock_code=stock_code; self.target_vol=int(target_vol)
        self.roll_days=int(roll_days); self.min_dte=int(min_dte); self.max_dte=int(max_dte)
        self.univ=None; self.pair=None; self.pending_roll_to=None
        self._start=None; self._end=None; self._loaded_pairs=set()

    def reset(self, account, start_time, end_time):
        self._start=start_time; self._end=end_time
        self.univ=StraddleUniverse(account, self.stock_code)
        self.pair=None; self.pending_roll_to=None; self._loaded_pairs=set()

    def _pair_flat(self, account, pair: Pair) -> bool:
        return is_leg_flat(account, pair.call) and is_leg_flat(account, pair.put)

    def _need_roll(self, ts_str: str) -> bool:
        if not self.pair: return True
        now=parse_ymd(ts_str[:8]); exp=parse_ymd(self.pair.expire)
        return (exp-now).days <= self.roll_days

    def _ensure_loaded(self, account, pair: Pair):
        key=(pair.call,pair.put)
        if key not in self._loaded_pairs:
            set_pair_and_preload(account, pair.call, pair.put, self._start, self._end)
            self._loaded_pairs.add(key)
        else:
            account.set_combos(pair.call, pair.put)

    def select_pair_before_init_state(self, account, ts_str, spot):
        if self.pair is None:
            self.pair=self.univ.select_atm_pair(account, ts_str, spot, self.min_dte, self.max_dte)
        if not self.pair: return
        if self.pending_roll_to is None and self._need_roll(ts_str):
            nxt=self.univ.select_atm_pair(account, ts_str, spot, self.min_dte, self.max_dte, after_expire=self.pair.expire)
            if nxt: self.pending_roll_to=nxt
        self._ensure_loaded(account, self.pair)

    def _complete_roll_if_flat(self, account):
        if self.pending_roll_to is not None and self._pair_flat(account, self.pair):
            self.pair=self.pending_roll_to; self.pending_roll_to=None
            self._ensure_loaded(account, self.pair)

class LongStraddleCheapIV(RollStraddleBase):
    def __init__(self, *args, k_open=-0.005, k_close=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_open=float(k_open); self.k_close=float(k_close)

    def act_after_fill(self, account, ts_str, spot):
        if not self.pair: return
        if self.pending_roll_to is not None:
            account.close_pair(ts_str, self.pair.call, self.pair.put, w=1.0)
        self._complete_roll_if_flat(account)
        if not self.pair: return
        iv=0.5*(float(account.comb.get("call_iv",0.0))+float(account.comb.get("put_iv",0.0)))
        hv=float(account.get_hv_160(ts_str) or 0.0)
        if iv<=0 or hv<=0: return
        d,_=account.get_pair_position(self.pair.call,self.pair.put)
        if iv <= hv + self.k_open:
            if d <= 0:
                account.flip_short_to_long(self.target_vol, ts_str, self.pair.call, self.pair.put)
        elif iv >= hv + self.k_close:
            if d != 0:
                account.close_pair(ts_str, self.pair.call, self.pair.put, w=1.0)

class ShortStraddleRiskManaged(RollStraddleBase):
    def __init__(self, *args, lookback_bars=8, trend_th=0.012, stop_mult=1.5, cooldown_bars=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookback_bars=int(lookback_bars); self.trend_th=float(trend_th)
        self.stop_mult=float(stop_mult); self.cooldown_bars=int(cooldown_bars)
        self.spots: Deque[float] = deque(maxlen=self.lookback_bars+1)
        self.entry_premium=None
        self.cooldown_left=0

    def reset(self, account, start_time, end_time):
        super().reset(account, start_time, end_time)
        self.spots.clear(); self.entry_premium=None; self.cooldown_left=0

    def act_after_fill(self, account, ts_str, spot):
        self.spots.append(spot)
        if not self.pair: return
        if self.pending_roll_to is not None:
            account.close_pair(ts_str, self.pair.call, self.pair.put, w=1.0)
        self._complete_roll_if_flat(account)
        if not self.pair: return

        d,_=account.get_pair_position(self.pair.call,self.pair.put)

        trend_block=False
        if len(self.spots) >= self.lookback_bars+1:
            r=(self.spots[-1]/self.spots[0])-1.0
            if abs(r) >= self.trend_th:
                trend_block=True

        if d < 0:
            if not self.entry_premium:
                self.entry_premium = pair_premium(account, ts_str, self.pair.call, self.pair.put)
            prem = pair_premium(account, ts_str, self.pair.call, self.pair.put)
            if prem>0 and self.entry_premium and prem >= self.entry_premium*(1.0+self.stop_mult):
                account.close_pair(ts_str, self.pair.call, self.pair.put, w=1.0)
                self.cooldown_left=self.cooldown_bars
                self.entry_premium=None
                return

        if trend_block:
            if d != 0:
                account.close_pair(ts_str, self.pair.call, self.pair.put, w=1.0)
                self.entry_premium=None
                self.cooldown_left=max(self.cooldown_left, self.cooldown_bars//2)
            return

        if self.cooldown_left>0:
            self.cooldown_left-=1
            return

        if d == 0 and self.pending_roll_to is None:
            account.open_short_pair(self.target_vol, ts_str, self.pair.call, self.pair.put)
            self.entry_premium = pair_premium(account, ts_str, self.pair.call, self.pair.put)

def run_backtest(strategy: StrategyBase, stock_code: str, start_time: str, end_time: str,
                 init_capital: float=100000.0, fee: float=1.3, period: str="30m",
                 dataset_root: str="./miniQMT/datasets", use_open_price: bool=True):
    account = single_Account(init_capital, fee=fee, period=period, stockList=[stock_code], filepath=dataset_root)
    df = account.real_info_controller.get_bars_between(stock_code, start_time, end_time, period, columns=("ts","close","volume","open"))
    if df.empty:
        raise RuntimeError("Underlying Kline is empty.")
    account.init_hv160(start_time, end_time, stock_code)
    strategy.reset(account, start_time, end_time)

    ts_list=[]; spot_list=[]; equity_list=[]
    for row in df.itertuples(index=False):
        ts_str=ts_to_str(row.ts); spot=float(row.close)
        strategy.select_pair_before_init_state(account, ts_str, spot)
        account.init_state(ts_str, spot)
        account.simulate_fill(ts_str, use_open_price=use_open_price)
        strategy.act_after_fill(account, ts_str, spot)
        account.update_positions(ts_str)
        account._update_comb_equity()
        ts_list.append(ts_str); spot_list.append(spot); equity_list.append(float(account.equity))

    curve=pd.DataFrame({"ts_str":ts_list,"spot":spot_list,"equity":equity_list})
    metrics=compute_metrics(np.array(equity_list,dtype=float), bars_per_year=2016)
    return curve, metrics, account

def save_results(curve: pd.DataFrame, metrics: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path) as w:
        curve.to_excel(w, index=False, sheet_name="equity_curve")
        pd.DataFrame([metrics]).to_excel(w, index=False, sheet_name="metrics")

if __name__=="__main__":
    stock="510050"
    start="20240102093000"
    end="20241231150000"
    out_dir="./miniQMT/datasets/outs"
    os.makedirs(out_dir, exist_ok=True)

    strat=LongStraddleCheapIV(stock_code=stock, target_vol=5, roll_days=3, min_dte=10, max_dte=60, k_open=-0.005, k_close=0.01)
    curve, metrics, _=run_backtest(strat, stock, start, end, init_capital=100000, period="30m", dataset_root="./miniQMT/datasets", use_open_price=True)
    print("B1 LongStraddleCheapIV:", metrics)
    save_results(curve, metrics, os.path.join(out_dir, "baseline_v5_B1_long_cheap_iv.xlsx"))

    strat=ShortStraddleRiskManaged(stock_code=stock, target_vol=3, roll_days=3, min_dte=14, max_dte=60,
                                   lookback_bars=8, trend_th=0.012, stop_mult=1.5, cooldown_bars=16)
    curve, metrics, _=run_backtest(strat, stock, start, end, init_capital=100000, period="30m", dataset_root="./miniQMT/datasets", use_open_price=True)
    print("B2 ShortStraddleRiskManaged:", metrics)
    save_results(curve, metrics, os.path.join(out_dir, "baseline_v5_B2_short_risk_managed.xlsx"))
