from __future__ import annotations
import os
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple
import pandas as pd
from datetime import datetime, time, timedelta

try:
    from finTool.BS import BS
except Exception as _:
    from BS import BS

import numpy as np

class RealInfo:
    PERIOD_TO_K = {
            '1m': '1分钟', '5m': '5分钟', '15m': '15分钟', '30m': '30分钟', '60m': '60分钟',
            '1d': '日', 'd': '日', 'day': '日', '日': '日'
        }
    USECOLS = ['交易时间', '收盘价', '成交量']
    RENAME  = {'交易时间':'ts','收盘价':'close','成交量':'volume'}

    def __init__(self,
                 stock_list: List[str],
                 filepath: str = './miniQMT/datasets/realInfo',
                 period: str = '30m',
                 max_option_cache: int = 10,
                 date_pick: str = 'last'   # <--- 新增：'last'（默认）或 'first'
                 ):
        self.stock_list = list(stock_list)
        self.filepath = filepath
        self.period = period.lower()
        self.date_pick = date_pick  # 'first' / 'last'

        # 缓存键：(code, k)
        self.stock_data_dict: Dict[Tuple[str, str], pd.DataFrame] = {}
        self.option_data_dict: "OrderedDict[Tuple[str, str], pd.DataFrame]" = OrderedDict()

        # LRU缓存机制, 内存中最多放置max_option_cache(默认10)个期权excel表
        self.max_option_cache = max_option_cache
    

        # 无风险利率
        self.rate_dict = {}
        self.read_bond_rate()

        # BS-model
        self.bs = BS()

    # ---------------- 公共接口 ----------------
    def get_df(self, code: str, period: Optional[str] = None) -> pd.DataFrame:
        period = (period or self.period).lower()
        k = self._period_to_k(period)
        key = (code, k)

        if self._is_stock(code):
            df = self.stock_data_dict.get(key)
            if df is None:
                df = self._load_excel_minimal(code, k)
                self.stock_data_dict[key] = df
            return df

        if key in self.option_data_dict:
            df = self.option_data_dict.pop(key)
            self.option_data_dict[key] = df
            return df

        df = self._load_excel_minimal(code, k)
        self._option_put_lru(key, df)
        return df

    def get_close_by_str(self, code: str, ts_str: str, period: Optional[str] = None) -> Optional[float]:
        period = (period or self.period).lower()
        df = self.get_df(code, period)
        if df.empty:
            return None

        dt_min, dt_day = self._parse_keys(ts_str)
        if self._is_daily_period(period):
            if dt_day is None: return None
            if 'ts_date' not in df.columns:
                df['ts_date'] = df['ts'].dt.date
            hit = df.loc[df['ts_date'] == dt_day.date(), 'close']
            return float(hit.iloc[-1]) if not hit.empty else None
        else:
            # 分钟线：优先精确 14 位；8 位则用当日第一/最后一根的“规范时间戳”
            if dt_min is not None:
                hit = df.loc[df['ts'] == dt_min, 'close']
                if not hit.empty:
                    return float(hit.iloc[-1])
            if dt_day is not None:
                target_ts = self._pick_bar_time_for_day(period, dt_day, which=self.date_pick)
                if target_ts is None:
                    return None
                hit = df.loc[df['ts'] == target_ts, 'close']
                # 如果恰好那根缺失，可退化为当日首/末可见一根
                if hit.empty:
                    day_mask = (df['ts'].dt.date == dt_day.date())
                    series = df.loc[day_mask, 'close']
                    if series.empty:
                        return None
                    return float(series.iloc[0] if self.date_pick == 'first' else series.iloc[-1])
                return float(hit.iloc[-1])
            return None
    
    def get_avg_volume(self, code: str, period: Optional[str] = None):
        period = (period or self.period).lower()
        df = self.get_df(code, period)

        if df.empty:
            return 0
        return df['volume'].mean()

    def get_volume_by_str(self, code: str, ts_str: str, period: Optional[str] = None) -> Optional[float]:
        period = (period or self.period).lower()
        df = self.get_df(code, period)
        if df.empty:
            return None

        dt_min, dt_day = self._parse_keys(ts_str)
        if self._is_daily_period(period):
            if dt_day is None: return None
            if 'ts_date' not in df.columns:
                df['ts_date'] = df['ts'].dt.date
            hit = df.loc[df['ts_date'] == dt_day.date(), 'volume']
            return float(hit.iloc[-1]) if not hit.empty else None
        else:
            if dt_min is not None:
                hit = df.loc[df['ts'] == dt_min, 'volume']
                if not hit.empty:
                    return float(hit.iloc[-1])
            if dt_day is not None:
                target_ts = self._pick_bar_time_for_day(period, dt_day, which=self.date_pick)
                if target_ts is None:
                    return None
                hit = df.loc[df['ts'] == target_ts, 'volume']
                if hit.empty:
                    day_mask = (df['ts'].dt.date == dt_day.date())
                    series = df.loc[day_mask, 'volume']
                    if series.empty:
                        return None
                    return float(series.iloc[0] if self.date_pick == 'first' else series.iloc[-1])
                return float(hit.iloc[-1])
            return None

    # ---------------- 内部：加载/缓存 ----------------
    def _load_excel_minimal(self, code: str, k: str) -> pd.DataFrame:
        path = self._excel_path(code, k)
        if not os.path.exists(path):
            return pd.DataFrame(columns=['ts','close','volume'])

        df = pd.read_excel(path, sheet_name=0, usecols=self.USECOLS, engine='openpyxl').rename(columns=self.RENAME)
        if k == '日':
            df['ts'] = pd.to_datetime(df['ts'], format="%Y-%m-%d", errors='coerce')
        else:
            df['ts'] = pd.to_datetime(df['ts'], format="%Y-%m-%d %H:%M:%S", errors='coerce')

        df = (df.dropna(subset=['ts'])
                .sort_values('ts')
                .drop_duplicates(subset=['ts'], keep='last')
                .reset_index(drop=True))

        df['close']  = pd.to_numeric(df['close'],  errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        return df[['ts','close','volume']]

    def _option_put_lru(self, key: Tuple[str, str], df: pd.DataFrame):
        if key in self.option_data_dict:
            self.option_data_dict.pop(key)
        self.option_data_dict[key] = df
        while len(self.option_data_dict) > self.max_option_cache:
            self.option_data_dict.popitem(last=False)

    # ---------------- 工具 ----------------
    def _excel_path(self, code: str, k: str) -> str:
        return os.path.join(self.filepath, f'K线导出_{code}_{k}线数据.xlsx')

    def _period_to_k(self, period: str) -> str:
        p = period.lower()
        if p in self.PERIOD_TO_K:
            return self.PERIOD_TO_K[p]
        if p.endswith('m'):
            return p[:-1]
        raise ValueError(f'不可识别的 period: {period}')

    def _is_daily_period(self, period: str) -> bool:
        return self._period_to_k(period) == '日'

    def _is_stock(self, code: str) -> bool:
        return len(code) == 6

    def _parse_keys(self, ts_str: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        s = ts_str.strip()
        if len(s) == 14:
            dt = pd.to_datetime(s, format="%Y%m%d%H%M%S", errors='coerce')
            if pd.isna(dt): return (None, None)
            return (dt, pd.to_datetime(s[:8], format="%Y%m%d"))
        elif len(s) == 8:
            d = pd.to_datetime(s, format="%Y%m%d", errors='coerce')
            if pd.isna(d): return (None, None)
            return (None, d)
        else:
            return (None, None)

    # ==== 新增：根据 A 股规则返回某日该周期的“第一根/最后一根”的结束时间戳 ====
    def _pick_bar_time_for_day(self, period: str, day_ts: pd.Timestamp, which: str = 'last') -> Optional[pd.Timestamp]:
        """
        day_ts: 当天 00:00 的 Timestamp（无时分秒）
        返回：这一天该周期 K 线的第一/最后一根“结束时间戳”（用于等值匹配）
        规则（区间结束时间）：
          - 1m:  09:31 ~ 11:30, 13:01 ~ 15:00  → 第一根 09:31, 最后一根 15:00
          - 5m:  09:35 第一根，15:00 最后一根
          - 15m: 09:45 第一根，15:00 最后一根
          - 30m: 10:00 第一根，15:00 最后一根
          - 60m: 10:30 第一根，15:00 最后一根
        """
        p = period.lower()
        first_map = {
            '1m':  time(9, 31, 0),
            '5m':  time(9, 35, 0),
            '15m': time(9, 45, 0),
            '30m': time(10, 0, 0),
            '60m': time(10, 30, 0),
        }
        last_map = {
            '1m':  time(15, 0, 0),
            '5m':  time(15, 0, 0),
            '15m': time(15, 0, 0),
            '30m': time(15, 0, 0),
            '60m': time(15, 0, 0),
        }
        if p.endswith('d') or self._period_to_k(p) == '日':
            return None
        if p not in first_map:
            # 其它自定义分钟粒度可在此扩展
            return None
        t = first_map[p] if which == 'first' else last_map[p]
        return pd.Timestamp(datetime.combine(day_ts.date(), t))

    # ========= 新增：把边界字符串规范成 Timestamp ==========
    def _normalize_bound(self, period: str, ts_str: str, is_start: bool) -> Optional[pd.Timestamp]:
        """
        把 'YYYYMMDD' / 'YYYYMMDDHHMMSS' 规范成用于筛选的 Timestamp。
        - 日线&分钟线（传 8 位日期）：起点=00:00:00，终点=15:00:00（同一自然日）
        - 分钟线（传 14 位时间）：精确时间；日线（传 14 位）也按同日09:30/15:00处理
        """
        period = period.lower()
        s = ts_str.strip()

        # —— 是否日线 —— 
        is_daily = self._is_daily_period(period)

        # 14 位：分钟线精确，日线取同日 09:30/15:00
        if len(s) == 14:
            dt = pd.to_datetime(s, format="%Y%m%d%H%M%S", errors="coerce")
            if pd.isna(dt):
                return None
            if is_daily:
                return pd.Timestamp(f"{dt.date()} {'09:30:00' if is_start else '15:00:00'}")
            return dt

        # 8 位：同日 09:30 / 15:00
        if len(s) == 8:
            d = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
            if pd.isna(d):
                return None
            return pd.Timestamp(f"{d.date()} {'00:00:00' if is_start else '15:00:00'}")

        return None

    # ========= 新增：区间获取收盘价/成交量 ==========
    def get_bars_between(self,
                         code: str,
                         start_str: str,
                         end_str: str,
                         period: Optional[str] = None,
                         columns: tuple = ('ts', 'close'),
                         closed: str = 'both') -> pd.DataFrame:
        """
        返回区间 [A, B] 的K线数据（默认只含 ts/close；传 columns=('ts','close','volume') 可一起返回成交量）。
        - A/B 接受 'YYYYMMDD' 或 'YYYYMMDDHHMMSS'
        - period 不传用实例默认；会按你文件名规则读取相应 Excel
        - 日线：A=当天0点，B=当天23:59:59.999999
        - 分钟线：A=当天第一根的结束时刻，B=当天最后一根的结束时刻；若 A/B 是14位则精确到秒
        - closed: 'both' | 'left' | 'right' | 'neither' 控制边界是否包含
        """
        period = (period or self.period).lower()
        df = self.get_df(code, period)
        if df.empty:
            return pd.DataFrame(columns=list(columns))

        a = self._normalize_bound(period, start_str, is_start=True)
        b = self._normalize_bound(period, end_str,   is_start=False)
        if a is None or b is None:
            return pd.DataFrame(columns=list(columns))
        if b < a:
            # 交换，保证 a <= b
            a, b = b, a

        # 构造布尔掩码（边界闭开控制）
        if closed == 'both':
            mask = (df['ts'] >= a) & (df['ts'] <= b)
        elif closed == 'left':
            mask = (df['ts'] >= a) & (df['ts'] <  b)
        elif closed == 'right':
            mask = (df['ts'] >  a) & (df['ts'] <= b)
        else:  # 'neither'
            mask = (df['ts'] >  a) & (df['ts'] <  b)

        out = df.loc[mask, list(columns)].copy()
        # 防御：按时间排序并去重
        if not out.empty:
            out = out.sort_values('ts').drop_duplicates(subset=['ts'], keep='last').reset_index(drop=True)
        return out

    def get_bars_between_from_df(self,
                                code: str,
                                start_str: str,
                                end_str: str,
                                period: Optional[str] = None,
                                columns: tuple = ('ts', 'close'),
                                closed: str = 'both') -> pd.DataFrame:
        """
        直接在给定的 df 上做区间筛选（df 需含列 ts/close/volume，ts 为 datetime64[ns]）。
        """

        period = (period or self.period).lower()
        df = self.get_df(code, period)

        a = self._normalize_bound(period, start_str, is_start=True)
        b = self._normalize_bound(period, end_str,   is_start=False)
        if a is None or b is None:
            return pd.DataFrame(columns=list(columns))
        if b < a:
            a, b = b, a

        if closed == 'both':
            mask = (df['ts'] >= a) & (df['ts'] <= b)
        elif closed == 'left':
            mask = (df['ts'] >= a) & (df['ts'] <  b)
        elif closed == 'right':
            mask = (df['ts'] >  a) & (df['ts'] <= b)
        else:
            mask = (df['ts'] >  a) & (df['ts'] <  b)

        out = df.loc[mask, list(columns)].copy()
        if not out.empty:
            out = (out.sort_values('ts')
                    .drop_duplicates(subset=['ts'], keep='last')
                    .reset_index(drop=True))  # pandas <2.1 用 reset_index(drop=True)
        return out

    # 读取无风险利率
    def read_bond_rate(self):
        bond_path = './miniQMT/datasets/realInfo/十年期国债利率.xlsx'
        df = pd.read_excel(bond_path, sheet_name=0)
        

        for _, row in df.iterrows():
            date = row['日期']
            rate = row['10年期国债收益率']

            int_date = int(date[0: 4] + date[5: 7] + date[8: 10])
            self.rate_dict[int_date] = rate
    
    # 获取无风险利率
    def get_bond_rate(self, time_str: str='20240808150000'):
        time_str = time_str[0: 8]
        int_time = int(time_str)
        
        return self.rate_dict[int_time] / 100

    # 计算ttm
    def get_ttm(self, now: str='', expire: str=''):
        # now要么20250808 / 20250808150000
        # expire: 20251025

        expire_day = datetime.strptime(expire, '%Y%m%d')
        if len(now) == 8:
            now_time = datetime.strptime(now, '%Y%m%d')
        elif len(now) == 14:
            now_time = datetime.strptime(now, '%Y%m%d%H%M%S')
            expire_day = expire_day.replace(hour=15)

        delta = expire_day - now_time

        return (delta.total_seconds() / 86400 / 365)

    # 计算隐含波动率
    def cal_iv(self, 
               time_str: str, 
               stockCode: str='510050', 
               optionCode: str='10010101', 
               strike: float=0, 
               expire: str='', 
               op_type: str='call',
               q: float=0,
               period=None):
        """
        Args:
            time_str: 时间
            stockCode/optionCode: 标的和期权代码
            strike: 行权价
            expire: 到期日, str = '20250827'
            op_type: call/put
            q: 分红率, 默认0, 由外部传入
            period: 周期, 默认None则采用类实例化的period
        Returns:
            iv: 期权的隐含波动率(无意义则返回-1)
        """
                
        period = period if period else self.period

        option_price = self.get_close_by_str(optionCode, time_str, period)
        stock_price = self.get_close_by_str(stockCode, time_str, period)
        rate = self.get_bond_rate(time_str)
        ttm = self.get_ttm(time_str, expire)

        iv, _ = BS.implied_volatility(option_price, stock_price, strike, rate, q, ttm, op_type)
        if iv is np.nan:
            return -1
        return iv

    def get_prev_30_days(self, date_str: str, days: int=30):
        # 输入格式：'20250808'
        date = datetime.strptime(date_str, '%Y%m%d')
        prev_date = date - timedelta(days=days)
        return prev_date.strftime('%Y%m%d')


    # 计算标的的历史波动率
    def get_history_iv(self, time_str, stockCode: str='510050'):
        end = time_str[0: 8]
        start = self.get_prev_30_days(end)

        data = self.get_bars_between(stockCode, start, end, '1d')
        close = data['close']
        log_ret = np.log(close / close.shift(1)).dropna()
        std = log_ret.std()
        history_iv = np.sqrt(252) * std

        return history_iv


    # 计算希腊字母
    def cal_greeks(self, 
               time_str: str, 
               stockCode: str='510050', 
               optionCode: str='10010101', 
               strike: float=0, 
               expire: str='', 
               op_type: str='call',
               q: float=0,
               period=None,
               rate: float=1.3849/100,
               ):
        """
        Args:
            time_str: 时间
            stockCode/optionCode: 标的和期权代码
            strike: 行权价
            expire: 到期日, str = '20250827'
            op_type: call/put
            q: 分红率, 默认0, 由外部传入
            period: 周期, 默认None则采用类实例化的period
        Returns:
            greeks: 
            {
                'iv': iv, 为0代表异常
                'delta': delta,
                'gamma': gamma,
                'theta': theta(),
                'vega': vega(),
                'rho': rho
            }

        """
        period = period if period else self.period

        option_price = self.get_close_by_str(optionCode, time_str, period)
        stock_price = self.get_close_by_str(stockCode, time_str, period)

        if rate is None:
            rate = self.get_bond_rate(time_str)
        ttm = self.get_ttm(time_str, expire)

        # print(f"option_price = {option_price}, stock_price = {stock_price}, ttm = {ttm}, expire = {expire}")

        # 获取历史波动率
        history_iv = self.get_history_iv(time_str, stockCode)

        # print(f"stock_price = {stock_price}, strike = {strike}, ttm = {ttm}, rate = {rate}, option_price = {option_price}, op_type = {op_type}, history_iv = {history_iv}")

        # res = bs.get_greeks(S, K, T, r, option_price, 'put', q=q)
        greeks = self.bs.get_greeks(stock_price, strike, ttm, rate, option_price, op_type, input_iv=None)
        return greeks


# ctx = RealInfo(['510050', '510500', '510300', '159915'])
# res = ctx.bs.get_greeks(3.094, 2.5, 30 / 365, 1.8 / 100, 0.5976, 'call', 0.1115)

# print(f"res = {res}")
# # start = '20200101093000'
# # end = '20200110150000'
# start_time = '20251107150000'
# end_time = '20250924150000'
# code = '10009218'

# results = ctx.cal_greeks(start_time, '510050', code, 2.55, '20251224', 'call')


# ctx.cal_greeks(self, )
# close = ctx.get_close_by_str('10002064', '20200102100000', '30m')
# close = ctx.get_bars_between_from_df('510050', '20200101093000', '20200110150000', '30m')
# print(f'close = {close}')
# print(0 / 0)