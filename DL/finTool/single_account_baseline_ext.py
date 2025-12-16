
"""
Runtime extension (monkey-patch) for single_Account to support baseline backtests
WITHOUT changing any existing PPO-integrated methods.

Adds:
- ensure_combos_loaded(call, put, start_time=None, end_time=None)
- init_state_with_pair(time_str, close, call, put, start_time=None, end_time=None)

Usage:
    from single_account_baseline_ext import single_Account
    acc = single_Account(...)
    acc.init_state_with_pair(ts, close, call, put, start, end)
"""

try:
    from finTool.single_window_account_fast import single_Account
except Exception:
    from single_window_account_fast import single_Account


def ensure_combos_loaded(self: single_Account, call: str, put: str, start_time: str=None, end_time: str=None):
    """Set comb legs if changed; optionally preload greek/price caches for [start_time, end_time]."""
    curr_call = self.comb.get('call', None)
    curr_put  = self.comb.get('put', None)
    if curr_call != call or curr_put != put:
        self.set_combos(call, put)
        if start_time is not None and end_time is not None:
            self.preload_data(start_time, end_time)


def init_state_with_pair(self: single_Account, time_str: str, close: float, call: str, put: str,
                         start_time: str=None, end_time: str=None):
    """
    Safe wrapper: ensure comb legs are set (and optionally preloaded), then run the original init_state.
    This avoids the issue where init_state expects self.comb['call'/'put'] already set.
    """
    ensure_combos_loaded(self, call, put, start_time, end_time)
    return self.init_state(time_str, close)


# Monkey-patch onto the class (keeps original file unchanged)
single_Account.ensure_combos_loaded = ensure_combos_loaded
single_Account.init_state_with_pair = init_state_with_pair
