"""
    PPO算法 (标准精度 Float32 版) - 动态并行训练重构版 (Full Optimized)
    包含: Multiprocessing Parallellism + Excel Export + Dynamic Environment Loading + Data Caching (Shared Memory)
    修复: 
    1. DynamicWindowEnv 增加 close 方法，修复 AttributeError。
    2. DataCache 使用 multiprocessing.Manager 共享内存，解决多进程重复读取导致的 Miss 刷屏。

    这个是2025.12.14版本, 目前已经实现了多线程, 整个rollout更新的操作
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from windowEnv_parallel_fast import windowEnv
import time, json
import sys
import pandas as pd
from datetime import datetime, timedelta

from typing import Any, Dict, List, Optional, Tuple
from tools.Norm import Normalization, RewardNormalization, RewardScaling
from preTrain.preMOE import PreMOE
from dataclasses import dataclass, field
import random
import multiprocessing as mp
from finTool.single_window_account_fast import single_Account
import os
import traceback
from datetime import datetime, timedelta

import warnings
# 忽略所有 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


# =========================
# Constants / actions
# =========================
A_HOLD, A_LONG, A_SHORT, A_CLOSE = 0, 1, 2, 3
WEIGHT_BINS = np.array([0.00, 0.25, 0.50, 0.75, 1.00], dtype=np.float32)


# 输出类, 输出日志, 防止中断后看不到信息
class outPut():
    """
    自定义输出类，将输出同时写入终端和日志文件。
    支持两种模式：'w' (覆盖重写) 和 'a' (续写)。
    """
    def __init__(self, filename, mode='w'):
        """
        初始化 outPut 实例。

        :param filename: 日志文件名。
        :param mode: 文件打开模式，'w' 为覆盖重写，'a' 为续写。默认为 'w'。
        """
        # 检查 mode 参数是否合法
        if mode not in ['w', 'a']:
            raise ValueError("mode 参数必须是 'w' (覆盖) 或 'a' (续写)")

        self.terminal = sys.stdout
        # 根据 mode 参数打开文件
        self.logfile = open(filename, mode, encoding="utf-8")
        
        # 可选：打印当前模式到终端，方便调试
        print(f"日志文件 '{filename}' 已以模式 '{mode}' 打开。")


    def write(self, message):
        """将消息同时写入终端和日志文件。"""
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        """强制将缓冲区内容写入目标（终端和文件）。"""
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        """关闭日志文件。"""
        self.logfile.close()
        print("日志文件已关闭。")

# =========================
# Normalization (n==0 -> return x)
# =========================
# class RunningMeanStd:
#     def __init__(self, shape, dtype=torch.float32, eps=1e-8, device="cpu"):
#         self.eps = float(eps)
#         self.n = 0  # keep int
#         self.mean = torch.zeros(shape, dtype=dtype, device=device)
#         self.var = torch.ones(shape, dtype=dtype, device=device)
#         self.std = torch.sqrt(self.var).clamp_min(self.eps)

#     def update(self, x: torch.Tensor):
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
#         B = int(x.shape[0])
#         if B <= 0:
#             return

#         n_old = int(self.n)
#         n_new = n_old + B

#         if n_old == 0:
#             mean = x.mean(dim=0)
#             var = x.var(dim=0, unbiased=False)
#             self.mean = mean
#             self.var = var
#             self.std = torch.sqrt(self.var.clamp_min(self.eps))
#             self.n = n_new
#             return

#         old_mean = self.mean
#         old_var = self.var
#         batch_mean = x.mean(dim=0)
#         batch_var = x.var(dim=0, unbiased=False)

#         delta = batch_mean - old_mean
#         mean = old_mean + delta * (B / n_new)

#         m2_old = old_var * n_old
#         m2_batch = batch_var * B
#         m2 = m2_old + m2_batch + (delta ** 2) * (n_old * B / n_new)
#         var = m2 / n_new

#         self.mean = mean
#         self.var = var
#         self.std = torch.sqrt(self.var.clamp_min(self.eps))
#         self.n = n_new


# class Normalization:
#     def __init__(self, shape, dtype=torch.float32, eps=1e-8, device="cpu"):
#         self.running_ms = RunningMeanStd(shape=shape, dtype=dtype, eps=eps, device=device)
#         self.eps = float(eps)

#     def __call__(self, x: torch.Tensor, update=True):
#         if update:
#             self.running_ms.update(x.detach())
#         if int(self.running_ms.n) == 0:
#             return x
#         return (x - self.running_ms.mean) / (self.running_ms.std + self.eps)

#     def state_dict(self):
#         return {
#             "n": int(self.running_ms.n),
#             "mean": self.running_ms.mean.detach().cpu(),
#             "var": self.running_ms.var.detach().cpu(),
#             "std": self.running_ms.std.detach().cpu(),
#             "eps": self.eps,
#         }

#     def load_state_dict(self, d: Dict[str, Any], device="cpu"):
#         self.eps = float(d.get("eps", self.eps))
#         self.running_ms.n = int(d["n"])
#         self.running_ms.mean = d["mean"].to(device)
#         self.running_ms.var = d["var"].to(device)
#         self.running_ms.std = d.get("std", torch.sqrt(self.running_ms.var)).to(device)


# =========================
# Networks
# =========================
class ActorDualHead(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_actions=4, n_weights=5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, n_actions)
        self.weight_head = nn.Linear(hidden_dim, n_weights)

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = self.backbone(x.float())
        return self.action_head(z), self.weight_head(z)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x.float())


class ViewProjector(nn.Module):
    def __init__(self, high_dim, low_dim, out_dim=48):
        super().__init__()
        self.high_net = nn.Sequential(
            nn.LayerNorm(high_dim),
            nn.Linear(high_dim, out_dim),
        )
        self.low_net = nn.Sequential(
            nn.LayerNorm(low_dim),
            nn.Linear(low_dim, 32),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(out_dim + 32, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x_high: torch.Tensor, x_low: torch.Tensor):
        h = self.high_net(x_high)
        l = self.low_net(x_low)
        return self.fusion(torch.cat([h, l], dim=-1))


class MultiViewAdapter(nn.Module):
    def __init__(self, dims_dict: Dict[str, int], final_dim=128, view_dim=48):
        super().__init__()
        self.varma_proj = ViewProjector(dims_dict["varma_high"], dims_dict["varma_low"], out_dim=view_dim)
        self.basis_proj = ViewProjector(dims_dict["basis_high"], dims_dict["basis_low"], out_dim=view_dim)
        self.itrans_proj = ViewProjector(dims_dict["itrans_high"], dims_dict["itrans_low"], out_dim=view_dim)
        self.router_proj = nn.Sequential(
            nn.LayerNorm(dims_dict["router"]),
            nn.Linear(dims_dict["router"], 32),
        )
        self.final_net = nn.Sequential(
            nn.Linear(view_dim * 3 + 32, final_dim),
            nn.LayerNorm(final_dim),
        )

    def forward(self, tok: Dict[str, torch.Tensor]):
        v_varma = self.varma_proj(tok["varma_h"], tok["varma_l"])
        v_basis = self.basis_proj(tok["basis_h"], tok["basis_l"])
        v_itrans = self.itrans_proj(tok["itrans_h"], tok["itrans_l"])
        v_router = self.router_proj(tok["router"])
        return self.final_net(torch.cat([v_varma, v_basis, v_itrans, v_router], dim=-1))


# =========================
# Feature pipeline (encode_tokens ONLY)
# =========================
class FeaturePipeline:
    def __init__(self, extractor: PreMOE, device="cpu", adapter_dim=128):
        self.device = device
        self.extractor = extractor.to(device)
        self.extractor.eval()
        for p in self.extractor.parameters():
            p.requires_grad = False

        self.adapter_dim = adapter_dim
        self.adapter: Optional[MultiViewAdapter] = None
        self.norm: Optional[Normalization] = None

    @torch.no_grad()
    def _encode_tokens_only(self, call_state: torch.Tensor, put_state: torch.Tensor):
        # IMPORTANT: only encode_tokens (no forward/predict)
        call_tok = self.extractor.encode_tokens(call_state)
        put_tok = self.extractor.encode_tokens(put_state)
        return call_tok, put_tok

    def build_adapter_from_dims(self, dims: Dict[str, int]):
        if self.adapter is None:
            self.adapter = MultiViewAdapter(dims, final_dim=self.adapter_dim).to(self.device)

    def build_norm_if_needed(self, state_dim: int):
        if self.norm is None:
            self.norm = Normalization(shape=(state_dim,), device=self.device)

    @torch.no_grad()
    def obs_to_state_raw(self, curr: torch.Tensor, hist: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        curr: (1,Dc)  hist: (1,L,Dh)
        returns raw_state (1, state_dim) and dims_dict (for adapter init)
        """
        if curr.dim() == 1:
            curr = curr.unsqueeze(0)
        if hist.dim() == 2:
            hist = hist.unsqueeze(0)

        call_state, put_state = torch.chunk(hist, chunks=2, dim=2)
        physical_call, physical_put = call_state[:, -1, :], put_state[:, -1, :]
        call_tok, put_tok = self._encode_tokens_only(call_state, put_state)

        dims = {
            "varma_high": int(call_tok["varma_h"].shape[-1]),
            "varma_low": int(call_tok["varma_l"].shape[-1]),
            "basis_high": int(call_tok["basis_h"].shape[-1]),
            "basis_low": int(call_tok["basis_l"].shape[-1]),
            "itrans_high": int(call_tok["itrans_h"].shape[-1]),
            "itrans_low": int(call_tok["itrans_l"].shape[-1]),
            "router": int(call_tok["router"].shape[-1]),
        }
        self.build_adapter_from_dims(dims)
        assert self.adapter is not None

        reduce_call = self.adapter(call_tok)
        reduce_put = self.adapter(put_tok)
        # raw = torch.cat([curr.float(), reduce_call.float(), reduce_put.float()], dim=-1)
        raw = torch.cat([curr.float(),  physical_call.float(), physical_put.float(), reduce_call.float(), reduce_put.float()], dim=-1)
        return raw, dims

    @torch.no_grad()
    def obs_to_state_normed(self, curr: torch.Tensor, hist: torch.Tensor, update_norm=False) -> Tuple[torch.Tensor, Dict[str, int]]:
        raw, dims = self.obs_to_state_raw(curr, hist)
        self.build_norm_if_needed(state_dim=int(raw.shape[-1]))
        assert self.norm is not None
        return self.norm(raw, update=update_norm), dims


# =========================
# Env wrapper (task switching)
# =========================
class DynamicWindowEnv:
    def __init__(self, option_pairs: List[Dict[str, Any]], cfg: Dict[str, Any], seed=0):
        self.all_pairs = option_pairs
        self.cfg = cfg
        self.fixed_idx: Optional[int] = None
        self.current_env = None
        random.seed(seed)

    def set_task(self, idx: int):
        self.fixed_idx = int(idx)

    def set_fee(self, fee: float):
        self.cfg['fee'] = fee

    def reset(self):
        if self.current_env is not None:
            try:
                self.current_env.close()
            except Exception:
                pass
            self.current_env = None

        if self.fixed_idx is None:
            pair = random.choice(self.all_pairs)
        else:
            pair = self.all_pairs[self.fixed_idx % len(self.all_pairs)]


        self.current_env = windowEnv(
            init_capital=self.cfg["init_capital"],
            call=pair["call"],
            put=pair["put"],
            fee=self.cfg["fee"],
            start_time=pair.get("start_time", self.cfg["start_time"]),
            end_time=pair.get("end_time", self.cfg["end_time"]),
            benchmark=self.cfg["benchmark"],
            timesteps=pair['steps'] + 1,
        )
        return self.current_env.reset()

    def step(self, action: int, weight: float):
        return self.current_env.step(action, weight)

    def close(self):
        if self.current_env is not None:
            try:
                self.current_env.close()
            except Exception:
                pass
            self.current_env = None

    @property
    def account_controller(self):
        return self.current_env.account_controller


# =========================
# IPC wrapper
# =========================
class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def _set_worker_threads():
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

_set_worker_threads()

# =========================
# Worker process
# =========================
def worker(remote, parent_remote, env_fn_wrapper, worker_cfg: Dict[str, Any]):
    parent_remote.close()
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    try:
        env = env_fn_wrapper.x()
    except Exception:
        tb = traceback.format_exc()
        remote.send(("__init_error__", tb))
        remote.close()
        return

    # build extractor (CPU)
    try:
        extractor = PreMOE(
            seq_len=worker_cfg["window_size"],
            pred_len=worker_cfg["pre_len"],
            n_variates=worker_cfg["n_variates"],
            d_router=worker_cfg["d_router"],
        ).to("cpu")
        if worker_cfg.get("pretrained_path") and os.path.exists(worker_cfg["pretrained_path"]):
            sd = torch.load(worker_cfg["pretrained_path"], map_location="cpu")
            extractor.load_state_dict(sd, strict=True)
        extractor.eval()
        for p in extractor.parameters():
            p.requires_grad = False
    except Exception:
        tb = traceback.format_exc()
        remote.send(("__init_error__", tb))
        remote.close()
        return

    feat = FeaturePipeline(extractor, device="cpu", adapter_dim=worker_cfg["adapter_dim"])
    actor: Optional[ActorDualHead] = None
    critic: Optional[ValueNet] = None

    pending_payload: Optional[Dict[str, Any]] = None
    adapter_dims: Optional[Dict[str, int]] = None

    def ensure_policy(curr_np: np.ndarray, hist_np: np.ndarray):
        nonlocal actor, critic, pending_payload, adapter_dims
        curr = torch.from_numpy(curr_np).float().unsqueeze(0)
        hist = torch.from_numpy(hist_np).float().unsqueeze(0)
        s, dims = feat.obs_to_state_normed(curr, hist, update_norm=False)
        adapter_dims = dims
        state_dim = int(s.shape[-1])
        if actor is None:
            actor = ActorDualHead(state_dim, hidden_dim=worker_cfg["hidden_dim"]).to("cpu")
            critic = ValueNet(state_dim, hidden_dim=worker_cfg["hidden_dim"]).to("cpu")
        if pending_payload is not None:
            apply_payload(pending_payload)
            pending_payload = None

    def apply_payload(payload: Dict[str, Any]):
        nonlocal actor, critic, adapter_dims
        if adapter_dims is None and payload.get("adapter_dims") is not None:
            adapter_dims = payload["adapter_dims"]

        if adapter_dims is not None:
            feat.build_adapter_from_dims(adapter_dims)

        if payload.get("norm_state") is not None:
            feat.build_norm_if_needed(int(payload["norm_state"]["mean"].numel()))
            assert feat.norm is not None
            feat.norm.load_state_dict(payload["norm_state"], device="cpu")

        if payload.get("adapter_state") is not None:
            assert feat.adapter is not None
            feat.adapter.load_state_dict(payload["adapter_state"], strict=True)

        if actor is not None and payload.get("actor_state") is not None:
            actor.load_state_dict(payload["actor_state"], strict=True)
        if critic is not None and payload.get("critic_state") is not None:
            critic.load_state_dict(payload["critic_state"], strict=True)

    def sample_action_weight(state_1d: torch.Tensor, deterministic=False) -> Tuple[int, int, float, float, float]:
        with torch.no_grad():
            assert actor is not None and critic is not None
            logits_a, logits_w = actor(state_1d)
            logits_a = logits_a.squeeze(0)
            logits_w = logits_w.squeeze(0)

            dist_a = Categorical(logits=logits_a)
            
            # --- 动作 A 选择 ---
            if deterministic:
                a = int(torch.argmax(logits_a, dim=-1).item())
            else:
                a = int(dist_a.sample().item())
            
            logp_a = float(dist_a.log_prob(torch.tensor(a)).item())

            # 掩码逻辑
            allowed = torch.zeros(5, dtype=torch.bool)
            if a in (A_LONG, A_SHORT, A_CLOSE):
                allowed[1:] = True
                need_w = 1.0
            else:
                allowed[0] = True
                need_w = 0.0

            masked = logits_w.clone()
            masked[~allowed] = -1e9
            dist_w = Categorical(logits=masked)

            # --- 权重 WI 选择 ---
            if deterministic:
                wi = int(torch.argmax(masked, dim=-1).item())
            else:
                wi = int(dist_w.sample().item())
                
            logp_w = float(dist_w.log_prob(torch.tensor(wi)).item())

            wv = float(WEIGHT_BINS[wi])
            logp_joint = logp_a + need_w * logp_w
            v = float(critic(state_1d).squeeze(-1).item())
            return a, wi, wv, logp_joint, v

    def sample_action_weight_old(state_1d: torch.Tensor, deterministic=False) -> Tuple[int, int, float, float, float]:
        """
        注意：这个函数本身不包 no_grad，但它只会在 rollout 循环的 with torch.no_grad() 内被调用
        所以不会产生梯度图，也不会占 GPU/CPU 的反传开销。
        """
        with torch.no_grad():
            assert actor is not None and critic is not None
            logits_a, logits_w = actor(state_1d)
            logits_a = logits_a.squeeze(0)
            logits_w = logits_w.squeeze(0)

            dist_a = Categorical(logits=logits_a)

            a = int(dist_a.sample().item())
            logp_a = float(dist_a.log_prob(torch.tensor(a)).item())

            allowed = torch.zeros(5, dtype=torch.bool)
            if a in (A_LONG, A_SHORT, A_CLOSE):
                allowed[1:] = True
                need_w = 1.0
            else:
                allowed[0] = True
                need_w = 0.0

            masked = logits_w.clone()
            masked[~allowed] = -1e9
            dist_w = Categorical(logits=masked)
            wi = int(dist_w.sample().item())
            logp_w = float(dist_w.log_prob(torch.tensor(wi)).item())

            wv = float(WEIGHT_BINS[wi])
            logp_joint = logp_a + need_w * logp_w
            v = float(critic(state_1d).squeeze(-1).item())
            return a, wi, wv, logp_joint, v

    # 用于 rewardScaling 的 gamma
    gamma = float(worker_cfg.get("gamma", 0.99))

    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break

        try:
            if cmd == "__ping__":
                remote.send(("__pong__", None))

            elif cmd == "set_task":
                idx = int(data)
                if hasattr(env, "set_task"):
                    env.set_task(idx)
                remote.send(("ok", None))
            
            elif cmd == 'set_fee':
                new_fee = float(data)
                env.set_fee(new_fee)

                remote.send(("ok", None))

            elif cmd == "set_weights":
                payload = data
                if actor is None or critic is None or feat.adapter is None or feat.norm is None:
                    pending_payload = payload
                else:
                    apply_payload(payload)
                remote.send(("ok", None))

            elif cmd == "rollout":
                T = int(data["T"])

                # ---- reset env ----
                reset_out = env.reset()
                if isinstance(reset_out, (tuple, list)) and len(reset_out) >= 2:
                    curr_np, hist_np = reset_out[0], reset_out[1]
                else:
                    raise RuntimeError(f"env.reset unexpected: {type(reset_out)}")

                curr_np = np.asarray(curr_np, np.float32)
                hist_np = np.asarray(hist_np, np.float32)

                ensure_policy(curr_np, hist_np)

                Dc = int(curr_np.shape[-1])
                L = int(hist_np.shape[-2])
                Dh = int(hist_np.shape[-1])

                # ---- buffers ----
                raw_curr = np.zeros((T, Dc), np.float32)
                raw_hist = np.zeros((T, L, Dh), np.float32)
                actions = np.zeros((T,), np.int64)
                w_idx = np.zeros((T,), np.int64)
                w_val = np.zeros((T,), np.float32)
                logp_old = np.zeros((T,), np.float32)
                value_old = np.zeros((T,), np.float32)

                rewards = np.zeros((T,), np.float32)

                done = np.zeros((T,), np.bool_)    # done[t] 对应 “t 动作之后是否终止”
                valid = np.zeros((T,), np.bool_)   # valid[t] 表示这一步 transition 是否可用于训练（必须有对齐后的 reward）

                # ---- per-episode RewardScaling (每个 worker/episode 独立) ----
                r_scaler = RewardScaling(shape=1, gamma=gamma)
                try:
                    r_scaler.reset()
                except Exception:
                    pass

                terminated_early = False

                # 你这个环境的 reward 延迟：t 返回 R_t，但属于 t-1 动作
                # 因此：t=0 的 reward 无意义；最后一个动作没有 reward（除非你额外再 step 一次）
                for t in range(T):
                    # 1) record state at time t
                    raw_curr[t] = curr_np
                    raw_hist[t] = hist_np

                    if terminated_early:
                        # padding
                        actions[t] = A_HOLD
                        w_idx[t] = 0
                        w_val[t] = 0.0
                        logp_old[t] = 0.0
                        value_old[t] = 0.0
                        rewards[t] = 0.0
                        done[t] = True
                        valid[t] = False
                        continue

                    # 2) decide action using policy (NO GRAD)
                    with torch.no_grad():
                        curr = torch.from_numpy(curr_np).unsqueeze(0)
                        hist = torch.from_numpy(hist_np).unsqueeze(0)
                        s, _ = feat.obs_to_state_normed(curr, hist, update_norm=False)
                        a, wi, wv, lp, v = sample_action_weight(s)

                    # 3) env step
                    step_out = env.step(a, wv)
                    if isinstance(step_out, (tuple, list)) and len(step_out) >= 5:
                        next_curr, next_hist, r, term, trunc = step_out[0], step_out[1], step_out[2], step_out[3], step_out[4]
                    else:
                        raise RuntimeError(f"env.step return length={len(step_out)} unexpected")

                    d = bool(term or trunc)

                    # 4) write transition fields for time t (but reward for t will come at t+1)
                    actions[t] = a
                    w_idx[t] = wi
                    w_val[t] = wv
                    logp_old[t] = lp
                    value_old[t] = v
                    done[t] = d

                    # 5) reward alignment + rewardScaling:
                    # 当前 step 返回的 r 属于 (t-1) 的动作，所以写入 rewards[t-1]
                    # 并且 t=0 的 r 丢弃
                    try:
                        r_in = torch.as_tensor([float(r)], dtype=torch.float32)
                        # r_scaled = float(r_scaler(r_in).item())
                    except Exception:
                        # 兜底：如果 RewardScaling 支持 float 输入
                        # r_scaled = float(r_scaler(float(r)))
                        pass
                    
                    # rewards[t] = r_scaled
                    rewards[t] = r_in
                    valid[t] = True
                    # if t > 0:
                    #     # 只有 (t-1) 这个 transition 才真正拿到了属于它的 reward，所以才 valid
                    #     rewards[t - 1] = r_scaled
                    #     # 注意：t-1 这步是否有效，还得看 t-1 自己是否是“最后一步”/是否被提前终止
                    #     # 这里只保证 reward 已对齐到 t-1
                    #     valid[t - 1] = True
                    

                    # 当前 t 这步：reward 还没来，所以先不置 valid[t]
                    # 如果这一刻终止了，那么这一步永远等不到 reward -> valid[t] 必须 False
                    if d:
                        terminated_early = True
                        valid[t] = False  # 强制
                        # 终止时不再更新 curr_np/hist_np
                        continue

                    # 6) move to next state
                    curr_np = np.asarray(next_curr, np.float32)
                    hist_np = np.asarray(next_hist, np.float32)

                # ---- 关键：最后一步永远没有下一步 reward（延迟机制下），必须 mask 掉 ----
                valid[T - 1] = False
                rewards[T - 1] = 0.0

                # ---- bootstrap last_value（保留你原逻辑） ----
                if not terminated_early:
                    with torch.no_grad():
                        curr = torch.from_numpy(curr_np).unsqueeze(0)
                        hist = torch.from_numpy(hist_np).unsqueeze(0)
                        s_last, _ = feat.obs_to_state_normed(curr, hist, update_norm=False)
                        assert critic is not None
                        last_value = float(critic(s_last).squeeze(-1).item())
                else:
                    last_value = 0.0

                # equity_end：尽量从 env.account_controller 取
                equity_end = None
                try:
                    if hasattr(env, "account_controller") and hasattr(env.account_controller, "equity"):
                        equity_end = float(env.account_controller.equity)
                except Exception:
                    equity_end = None
                if equity_end is None:
                    equity_end = float("nan")

                env_sharpe = env.account_controller.get_sharpe_ratio()
                remote.send(("traj", {
                    "raw_curr": raw_curr,
                    "raw_hist": raw_hist,
                    "actions": actions,
                    "w_idx": w_idx,
                    "w_val": w_val,
                    "logp_old": logp_old,
                    "value_old": value_old,
                    "rewards": rewards,
                    "done": done,
                    "valid": valid,
                    "last_value": np.array([last_value], np.float32),
                    "equity_end": np.array([equity_end], np.float32),
                    "env_sharpe": env_sharpe, # 🔥 新增
                }))

            elif cmd == "close":
                try:
                    env.close()
                except Exception:
                    pass
                remote.send(("ok", None))
                remote.close()
                break

            else:
                raise NotImplementedError(cmd)

        except Exception:
            tb = traceback.format_exc()
            try:
                remote.send(("error", tb))
            except Exception:
                pass
            break



# =========================
# SubprocVectorEnv (rollout in one IPC)
# =========================
class SubprocVectorEnv:
    def __init__(self, env_fns: List, worker_cfg: Dict[str, Any]):
        self.closed = False
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.ps = []

        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            p = mp.Process(
                target=worker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn), worker_cfg),
                daemon=True,
            )
            p.start()
            self.ps.append(p)
            work_remote.close()

        # detect init error early
        for r in self.remotes:
            r.send(("__ping__", None))
        for r in self.remotes:
            tag, payload = r.recv()
            if tag != "__pong__":
                raise RuntimeError("worker did not respond to ping")

    def set_tasks(self, task_indices: List[int]):
        for r, idx in zip(self.remotes, task_indices):
            r.send(("set_task", int(idx)))
        for r in self.remotes:
            tag, payload = r.recv()
            if tag == "error":
                raise RuntimeError(payload)

    def set_fee_all(self, fee: float):
        for r in self.remotes:
            r.send(("set_fee", fee))
        for r in self.remotes:
            tag, _ = r.recv()
            if tag == "error":
                raise RuntimeError("Failed to set fee in worker")
            
    def set_weights_all(self, payload: Dict[str, Any]):
        for r in self.remotes:
            r.send(("set_weights", payload))
        for r in self.remotes:
            tag, payload2 = r.recv()
            if tag == "error":
                raise RuntimeError(payload2)

    def rollout(self, T: int) -> List[Dict[str, Any]]:
        # 1. 发送命令
        for r in self.remotes:
            r.send(("rollout", {"T": int(T)}))

        trajs = []
        # 2. 逐个接收，接收完立刻断开引用，防止积压
        for r in self.remotes:
            try:
                # recv 这一步是最耗内存的，因为它要反序列化 huge object
                tag, payload = r.recv()
                
                if tag == "error":
                    raise RuntimeError(payload)
                if tag != "traj":
                    raise RuntimeError(f"unexpected tag from worker: {tag}")
                
                trajs.append(payload)
                
                # 🔥 显式删除 payload 引用 (虽然 Python 有 GC，但显式删除更保险)
                del payload 
            except Exception as e:
                # 遇到错误也要尝试把剩下的收完，防止死锁
                print(f"[Error] Recv failed: {e}")
                raise e
                
        return trajs


    def close(self):
        if self.closed:
            return
        for r in self.remotes:
            r.send(("close", None))
        for r in self.remotes:
            try:
                r.recv()
            except Exception:
                pass
        for p in self.ps:
            p.join()
        self.closed = True


# =========================
# Learner PPO (GPU update)
# =========================
class LearnerPPO:
    def __init__(
        self,
        device: str,
        window_size: int,
        pre_len: int,
        n_variates: int,
        d_router: int,
        pretrained_path: str,
        adapter_dim: int,
        hidden_dim: int,
        gamma: float,
        gae_lambda: float,
        clip_eps: float,
        k_epochs: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        check_path: str='./miniQMT/DL/checkout/check_data_parallel.pt',
        update_mb_size: int=2048,
        total_epochs: int=1000,
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.adapter_dim = adapter_dim
        self.hidden_dim = hidden_dim

        # self.check_path = f'{check_path}/check_data_parallel.pt'
        self.check_path = check_path
        self.update_mb_size = update_mb_size

        # frozen extractor on GPU
        self.extractor = PreMOE(
            seq_len=window_size, pred_len=pre_len, n_variates=n_variates, d_router=d_router
        ).to(device)
        if pretrained_path and os.path.exists(pretrained_path):
            sd = torch.load(pretrained_path, map_location=device)
            self.extractor.load_state_dict(sd, strict=True)
        else:
            print(f"[Learner] WARNING: pretrained_path not found: {pretrained_path}")
        self.extractor.eval()
        for p in self.extractor.parameters():
            p.requires_grad = False

        self.adapter: Optional[MultiViewAdapter] = None
        self.actor: Optional[ActorDualHead] = None
        self.critic: Optional[ValueNet] = None

        # 状态归一化模块
        self.norm: Optional[Normalization] = None
        self.adapter_dims: Optional[Dict[str, int]] = None

        self.opt_adapter = None
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.opt_actor = None
        self.opt_critic = None

        # 学习率衰减
        self.total_epochs = total_epochs

    def save(self, epoch: int = None, best_reward: float = None, path: str = None):
            save_path = path or self.check_path

            scheduler_state = {
            'actor': self.scheduler_actor.state_dict(),
            'critic': self.scheduler_critic.state_dict(),
            'adapter': self.scheduler_adapter.state_dict(),
        }
            data = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "actor_state": self.actor.state_dict(),
                "value_state": self.critic.state_dict(),
                "adpter_state": self.adapter.state_dict(),
                "opt_actor_state": self.opt_actor.state_dict(),
                "opt_critic_state": self.opt_critic.state_dict(),
                "opt_adpter_state": self.opt_adapter.state_dict(),
                "scheduler_state": scheduler_state,
                "h_params": {
                    "gamma": self.gamma,
                    "clip_eps": self.clip_eps,
                    "k_epochs": self.k_epochs,
                    "device": self.device,
                },
                "epoch": epoch,
                "best_reward": best_reward.item() if hasattr(best_reward, 'item') else best_reward,
                "state_norm": self.norm,
            }
            torch.save(data, save_path)
            print(f"[PPO] checkpoint saved to: {save_path}")

    def load_checkpoint(self, path: str=None):
        if path is None:
            path = self.check_path
        if not os.path.exists(path):
            print(f"[Warn] Checkpoint not found at {path}")
            return None, None
        
        print(f"[Resume] Loading checkpoint from {path}...")
        # 这里的 map_location 非常重要，防止跨设备加载报错
        data = torch.load(path, map_location=self.device, weights_only=False)
        
        # 1. 加载网络权重
        self.actor.load_state_dict(data['actor_state'])
        self.critic.load_state_dict(data['value_state'])
        self.adapter.load_state_dict(data['adpter_state'])
        
        # 2. 加载优化器状态
        if self.opt_actor: self.opt_actor.load_state_dict(data['opt_actor_state'])
        if self.opt_critic: self.opt_critic.load_state_dict(data['opt_critic_state'])
        if self.opt_adapter and 'adapter_state' in data:
            st = data['opt_adapter_state']
            if st is not None: 
                self.opt_adapter.load_state_dict(st)

        # 3. [新增] 加载 Normalization 状态
        if 'state_norm' in data:
            # 直接覆盖当前的 self.state_norm
            self.norm = data['state_norm']
            print(f"[Resume] State Norm loaded. (count={self.norm.running_ms.n if hasattr(self.norm.running_ms, 'n') else '?'})")
        else:
            print("[Resume] Warning: No state_norm in checkpoint! Training might be unstable.")

        # 4. [新增] 加载学习率调度器状态
        if 'scheduler_state' in data:
            self.scheduler_actor.load_state_dict(data['scheduler_state']['actor'])
            self.scheduler_critic.load_state_dict(data['scheduler_state']['critic'])
            self.scheduler_adapter.load_state_dict(data['scheduler_state']['adapter'])
            print("[Resume] Schedulers loaded.")    


        epoch = data.get('epoch', 0)
        best_reward = data.get('best_reward', -float('inf'))
        
        print(f"[Resume] Success! Resuming from Epoch {epoch + 1}, Best Reward: {best_reward:.4f}")
        return epoch, best_reward


    @torch.no_grad()
    def _encode_tokens_only(self, call_state, put_state):
        call_tok = self.extractor.encode_tokens(call_state)
        put_tok = self.extractor.encode_tokens(put_state)
        return call_tok, put_tok

    def _maybe_build(self, curr: torch.Tensor, hist: torch.Tensor):
        if curr.dim() == 1:
            curr = curr.unsqueeze(0)
        if hist.dim() == 2:
            hist = hist.unsqueeze(0)

        call_state, put_state = torch.chunk(hist, chunks=2, dim=2)

        physical_call, physical_put = call_state[:, -1, :], put_state[:, -1, :]

        call_tok, put_tok = self._encode_tokens_only(call_state, put_state)

        if self.adapter_dims is None:
            self.adapter_dims = {
                "varma_high": int(call_tok["varma_h"].shape[-1]),
                "varma_low": int(call_tok["varma_l"].shape[-1]),
                "basis_high": int(call_tok["basis_h"].shape[-1]),
                "basis_low": int(call_tok["basis_l"].shape[-1]),
                "itrans_high": int(call_tok["itrans_h"].shape[-1]),
                "itrans_low": int(call_tok["itrans_l"].shape[-1]),
                "router": int(call_tok["router"].shape[-1]),
            }

        if self.adapter is None:
            self.adapter = MultiViewAdapter(self.adapter_dims, final_dim=self.adapter_dim).to(self.device)

        reduce_call = self.adapter(call_tok)
        reduce_put = self.adapter(put_tok)
        
        raw = torch.cat([curr.float(),  physical_call.float(), physical_put.float(), reduce_call.float(), reduce_put.float()], dim=-1)
        state_dim = int(raw.shape[-1])


        if self.norm is None:
            self.norm = Normalization(shape=(state_dim,), device=self.device)

        if self.actor is None:
            self.actor = ActorDualHead(state_dim, hidden_dim=self.hidden_dim).to(self.device)
            self.critic = ValueNet(state_dim, hidden_dim=self.hidden_dim).to(self.device)
            self.opt_adapter = torch.optim.Adam(self.adapter.parameters(), lr=self.actor_lr)
            self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

            self.scheduler_actor = torch.optim.lr_scheduler.LinearLR(self.opt_actor, start_factor=1.0, end_factor=0.01, total_iters=self.total_epochs)
            self.scheduler_critic = torch.optim.lr_scheduler.LinearLR(self.opt_critic, start_factor=1.0, end_factor=0.01, total_iters=self.total_epochs)
            self.scheduler_adapter = torch.optim.lr_scheduler.LinearLR(self.opt_adapter, start_factor=1.0, end_factor=0.01, total_iters=self.total_epochs)


    def export_payload(self) -> Dict[str, Any]:
        assert self.adapter is not None and self.actor is not None and self.critic is not None
        norm_state = self.norm.state_dict() if self.norm is not None else None

        return {
            "adapter_dims": self.adapter_dims,
            "adapter_state": {k: v.detach().cpu() for k, v in self.adapter.state_dict().items()},
            "actor_state": {k: v.detach().cpu() for k, v in self.actor.state_dict().items()},
            "critic_state": {k: v.detach().cpu() for k, v in self.critic.state_dict().items()},
            "norm_state": norm_state,
        }

    def _build_state(self, curr: torch.Tensor, hist: torch.Tensor, norm_update: bool):
        self._maybe_build(curr, hist)
        assert self.adapter is not None and self.norm is not None

        # 1. 拆分状态
        call_state, put_state = torch.chunk(hist, chunks=2, dim=2)
        
        # 2. 提取物理特征 (与 _maybe_build 对齐)
        physical_call, physical_put = call_state[:, -1, :], put_state[:, -1, :]
        
        # 3. 提取特征 Token
        call_tok, put_tok = self._encode_tokens_only(call_state, put_state)

        # 4. Adapter 投影
        reduce_call = self.adapter(call_tok)
        reduce_put = self.adapter(put_tok)
        
        # 5. 🔥 核心修复：这里必须拼 5 个部分，总维度才是 547
        raw = torch.cat([
            curr.float(), 
            physical_call.float(), 
            physical_put.float(), 
            reduce_call.float(), 
            reduce_put.float()
        ], dim=-1)
        
        s = self.norm(raw, update=norm_update)
        return raw, s
    
    def update_from_trajs(self, trajs: List[Dict[str, Any]], target_kl: float = 0.015):
            """
            针对 4090D 优化的平稳更新版：
            1. 引入 Explained Variance 监控价值网络质量
            2. 使用 Huber Loss 处理价值损失，抑制极端惩罚导致的梯度抖动
            3. 调整 Loss 权重分配：VF=0.5 (增强预判), Ent=0.02 (保持探索)
            """
            assert self.actor is not None and self.critic is not None
            
            # ---------------------------------------------------------------------
            # 1. 数据堆叠 (CPU)
            # ---------------------------------------------------------------------
            def stack_key(key, dtype):
                N = len(trajs)
                if N == 0: return torch.tensor([], dtype=dtype)
                first_val = trajs[0][key]
                sample_shape = torch.as_tensor(first_val, dtype=dtype).shape
                target_shape = list(sample_shape)
                target_shape.insert(1, N)
                out = torch.empty(target_shape, dtype=dtype)
                for i, tr in enumerate(trajs):
                    out.select(1, i).copy_(torch.as_tensor(tr[key], dtype=dtype))
                return out

            raw_curr   = stack_key("raw_curr", torch.float32)
            raw_hist   = stack_key("raw_hist", torch.float32)
            actions    = stack_key("actions", torch.long)
            w_idx      = stack_key("w_idx", torch.long)
            logp_old   = stack_key("logp_old", torch.float32)
            value_old  = stack_key("value_old", torch.float32)
            rewards    = stack_key("rewards", torch.float32)
            done       = stack_key("done", torch.float32)
            valid      = stack_key("valid", torch.float32)
            last_value = torch.stack([torch.as_tensor(tr["last_value"], dtype=torch.float32) for tr in trajs], dim=0).squeeze()

            # ---------------------------------------------------------------------
            # 2. 计算 GAE (CPU)
            # ---------------------------------------------------------------------
            T, N = raw_curr.shape[:2]
            with torch.no_grad():
                adv = torch.zeros((T, N), dtype=torch.float32)
                last_gae = torch.zeros((N,), dtype=torch.float32)
                for t in reversed(range(T)):
                    m = (1.0 - done[t]) * valid[t]
                    v_tp1 = last_value if t == T - 1 else value_old[t + 1]
                    delta = rewards[t] + self.gamma * v_tp1 * m - value_old[t]
                    last_gae = delta + self.gamma * self.gae_lambda * m * last_gae
                    adv[t] = last_gae * valid[t]
                ret = adv + value_old

            # ---------------------------------------------------------------------
            # 3. Flatten & Filter
            # ---------------------------------------------------------------------
            mask_flat = valid.view(-1) > 0.5
            if not mask_flat.any(): return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            def flat_and_filter(tensor_in):
                return tensor_in.flatten(0, 1)[mask_flat]

            curr_flat = flat_and_filter(raw_curr)
            hist_flat = flat_and_filter(raw_hist)
            act_flat  = flat_and_filter(actions)
            widx_flat = flat_and_filter(w_idx)
            logp_old_flat = flat_and_filter(logp_old)
            value_old_flat = flat_and_filter(value_old)
            adv_raw_flat = flat_and_filter(adv)
            ret_flat  = flat_and_filter(ret)

            # 计算 Explained Variance (衡量旧 Value 对实际 Return 的预测准确度)
            with torch.no_grad():
                y_true = ret_flat.numpy()
                y_pred = value_old_flat.numpy()
                var_y = np.var(y_true)
                explained_var = 1.0 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0.0
                explained_var = max(-1.0, float(explained_var)) # 截断异常负值

            # 优势归一化
            adv_flat = (adv_raw_flat - adv_raw_flat.mean()) / (adv_raw_flat.std() + 1e-8)
            adv_flat = torch.clamp(adv_flat, -5.0, 5.0)

            # ---------------------------------------------------------------------
            # 4. PPO Training Loop (Mini-batch)
            # ---------------------------------------------------------------------
            M = curr_flat.shape[0]
            mb = self.update_mb_size
            stats = {"loss": [], "kl": [], "actor_loss": [], "value_loss": [], "entropy": []}
            
            for _ep in range(self.k_epochs):
                indices = torch.randperm(M)
                epoch_kl = []
                for st in range(0, M, mb):
                    idx = indices[st:st + mb]
                    if len(idx) == 0: continue
                    
                    # 异步推送到 GPU (4090D 优势)
                    curr_b = curr_flat[idx].to(self.device, non_blocking=True)
                    hist_b = hist_flat[idx].to(self.device, non_blocking=True)
                    act_b  = act_flat[idx].to(self.device, non_blocking=True)
                    widx_b = widx_flat[idx].to(self.device, non_blocking=True)
                    logp_old_b = logp_old_flat[idx].to(self.device, non_blocking=True)
                    adv_b  = adv_flat[idx].to(self.device, non_blocking=True)
                    ret_b  = ret_flat[idx].to(self.device, non_blocking=True)
                    v_old_b = value_old_flat[idx].to(self.device, non_blocking=True)

                    _, s = self._build_state(curr_b, hist_b, norm_update=False)
                    logits_a, logits_w = self.actor(s)
                    
                    # Actor Logic
                    dist_a = Categorical(logits=logits_a)
                    new_logp_a = dist_a.log_prob(act_b)
                    ent_a = dist_a.entropy().mean()

                    # Weight Logic
                    need_w = ((act_b == 1) | (act_b == 2) | (act_b == 3)).float()
                    lw = logits_w.clone()
                    if need_w.sum() > 0:
                        maskw = torch.zeros_like(lw, dtype=torch.bool)
                        maskw[need_w.bool(), 1:] = True
                        maskw[~need_w.bool(), 0] = True
                        lw[~maskw] = -1e9
                    dist_w = Categorical(logits=lw)
                    new_logp_w = dist_w.log_prob(widx_b)
                    ent_w = (need_w * dist_w.entropy()).sum() / (need_w.sum() + 1e-6)

                    logp_new = new_logp_a + need_w * new_logp_w
                    ratio = torch.exp(logp_new - logp_old_b)

                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # --- 🔥 关键改进：Huber Loss + Value Clipping ---
                    v_pred = self.critic(s).squeeze(-1)
                    # 使用 Huber Loss 代替 MSE，缓解 -150 等极端值冲击
                    v_loss_unclipped = F.smooth_l1_loss(v_pred, ret_b)
                    
                    v_pred_clipped = v_old_b + torch.clamp(v_pred - v_old_b, -self.clip_eps, self.clip_eps)
                    v_loss_clipped = F.smooth_l1_loss(v_pred_clipped, ret_b)
                    value_loss = torch.max(v_loss_unclipped, v_loss_clipped)

                    entropy = ent_a + 0.5 * ent_w
                    
                    # --- 🔥 总 Loss 计算 (VF权重提升至0.5) ---
                    loss = actor_loss + 0.5 * value_loss - 0.05 * entropy
                    
                    self.opt_adapter.zero_grad(set_to_none=True)
                    self.opt_actor.zero_grad(set_to_none=True)
                    self.opt_critic.zero_grad(set_to_none=True)
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 0.5)
                    
                    self.opt_adapter.step()
                    self.opt_actor.step()
                    self.opt_critic.step()
                    
                    with torch.no_grad():
                        kl_b = (logp_old_b - logp_new).mean().abs().item()
                        epoch_kl.append(kl_b)
                        stats["loss"].append(loss.item())
                        stats["kl"].append(kl_b)
                        stats["actor_loss"].append(actor_loss.item())
                        stats["value_loss"].append(value_loss.item())
                        stats["entropy"].append(entropy.item())

                if (sum(epoch_kl)/len(epoch_kl)) > 1.5 * target_kl: break

            # 5. 更新状态归一化
            if M > 0:
                with torch.no_grad():
                    for st in range(0, M, mb):
                        self._build_state(curr_flat[st:st+mb].to(self.device), hist_flat[st:st+mb].to(self.device), norm_update=True)

            self.scheduler_actor.step()
            self.scheduler_critic.step()
            self.scheduler_adapter.step()
            
            def get_avg(k): return sum(stats[k]) / len(stats[k]) if stats[k] else 0.0
            return get_avg("loss"), get_avg("kl"), get_avg("actor_loss"), get_avg("value_loss"), get_avg("entropy"), explained_var


    def oldold_update_from_trajs(self, trajs: List[Dict[str, Any]], target_kl: float = 0.015):
        """
        针对 110 Epoch 后的异常修正版：
        1. 使用 Huber Loss 替换 MSE，平滑异常收益冲击
        2. 进一步削减 Value 权重 (0.01)，增加熵权重 (0.02) 以打破 Hold/Close 僵局
        3. 强制 Value Clipping
        """
        assert self.actor is not None and self.critic is not None
        
        # ---------------------------------------------------------------------
        # 1. 统一堆叠 (CPU)
        # ---------------------------------------------------------------------
        def stack_key(key, dtype):
            N = len(trajs)
            if N == 0: return torch.tensor([], dtype=dtype)
            first_val = trajs[0][key]
            sample_shape = torch.as_tensor(first_val, dtype=dtype).shape
            target_shape = list(sample_shape)
            target_shape.insert(1, N)
            out = torch.empty(target_shape, dtype=dtype)
            for i, tr in enumerate(trajs):
                out.select(1, i).copy_(torch.as_tensor(tr[key], dtype=dtype))
            return out

        raw_curr  = stack_key("raw_curr", torch.float32)
        raw_hist  = stack_key("raw_hist", torch.float32)
        actions   = stack_key("actions", torch.long)
        w_idx     = stack_key("w_idx", torch.long)
        logp_old  = stack_key("logp_old", torch.float32)
        value_old = stack_key("value_old", torch.float32)
        rewards   = stack_key("rewards", torch.float32)
        done      = stack_key("done", torch.float32)
        valid     = stack_key("valid", torch.float32)
        last_value = torch.stack([torch.as_tensor(tr["last_value"], dtype=torch.float32) for tr in trajs], dim=0).squeeze()

        # ---------------------------------------------------------------------
        # 2. 计算 GAE
        # ---------------------------------------------------------------------
        T, N = raw_curr.shape[:2]
        with torch.no_grad():
            adv = torch.zeros((T, N), dtype=torch.float32)
            last_gae = torch.zeros((N,), dtype=torch.float32)
            for t in reversed(range(T)):
                m = (1.0 - done[t]) * valid[t]
                v_tp1 = last_value if t == T - 1 else value_old[t + 1]
                delta = rewards[t] + self.gamma * v_tp1 * m - value_old[t]
                last_gae = delta + self.gamma * self.gae_lambda * m * last_gae
                adv[t] = last_gae * valid[t]
            ret = adv + value_old

        # ---------------------------------------------------------------------
        # 3. Flatten & Filter
        # ---------------------------------------------------------------------
        mask_flat = valid.view(-1) > 0.5
        if not mask_flat.any(): return 0.0, 0.0, 0.0, 0.0, 0.0
        
        def flat_and_filter(tensor_in):
            return tensor_in.flatten(0, 1)[mask_flat]

        curr_flat = flat_and_filter(raw_curr)
        hist_flat = flat_and_filter(raw_hist)
        act_flat  = flat_and_filter(actions)
        widx_flat = flat_and_filter(w_idx)
        logp_old_flat = flat_and_filter(logp_old)
        value_old_flat = flat_and_filter(value_old)
        adv_raw_flat = flat_and_filter(adv)
        ret_flat  = flat_and_filter(ret)

        # 优势归一化与裁剪
        adv_flat = (adv_raw_flat - adv_raw_flat.mean()) / (adv_raw_flat.std() + 1e-8)
        adv_flat = torch.clamp(adv_flat, -5.0, 5.0)

        # ---------------------------------------------------------------------
        # 4. PPO Training Loop
        # ---------------------------------------------------------------------
        M = curr_flat.shape[0]
        mb = self.update_mb_size
        stats = {"loss": [], "kl": [], "actor_loss": [], "value_loss": [], "entropy": []}
        
        for _ep in range(self.k_epochs):
            indices = torch.randperm(M)
            epoch_kl = []
            for st in range(0, M, mb):
                idx = indices[st:st + mb]
                if len(idx) == 0: continue
                
                curr_b = curr_flat[idx].to(self.device, non_blocking=True)
                hist_b = hist_flat[idx].to(self.device, non_blocking=True)
                act_b  = act_flat[idx].to(self.device, non_blocking=True)
                widx_b = widx_flat[idx].to(self.device, non_blocking=True)
                logp_old_b = logp_old_flat[idx].to(self.device, non_blocking=True)
                adv_b  = adv_flat[idx].to(self.device, non_blocking=True)
                ret_b  = ret_flat[idx].to(self.device, non_blocking=True)
                v_old_b = value_old_flat[idx].to(self.device, non_blocking=True)

                _, s = self._build_state(curr_b, hist_b, norm_update=False)
                logits_a, logits_w = self.actor(s)
                
                # Actor 逻辑
                dist_a = Categorical(logits=logits_a)
                new_logp_a = dist_a.log_prob(act_b)
                ent_a = dist_a.entropy().mean()

                need_w = ((act_b == 1) | (act_b == 2) | (act_b == 3)).float()
                lw = logits_w.clone()
                if need_w.sum() > 0:
                    maskw = torch.zeros_like(lw, dtype=torch.bool)
                    maskw[need_w.bool(), 1:] = True
                    maskw[~need_w.bool(), 0] = True
                    lw[~maskw] = -1e9
                
                dist_w = Categorical(logits=lw)
                new_logp_w = dist_w.log_prob(widx_b)
                ent_w = (need_w * dist_w.entropy()).sum() / (need_w.sum() + 1e-6)

                logp_new = new_logp_a + need_w * new_logp_w
                ratio = torch.exp(logp_new - logp_old_b)

                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- 🔥 改进：Huber Loss + Value Clipping ---
                v_pred = self.critic(s).squeeze(-1)
                v_pred_clipped = v_old_b + torch.clamp(v_pred - v_old_b, -self.clip_eps, self.clip_eps)
                # 使用 Huber Loss (smooth_l1_loss) 对极端 Return 不敏感
                v_loss1 = F.smooth_l1_loss(v_pred, ret_b)
                v_loss2 = F.smooth_l1_loss(v_pred_clipped, ret_b)
                value_loss = torch.max(v_loss1, v_loss2)

                entropy = ent_a + 0.5 * ent_w
                
                # --- 🔥 总 Loss 计算 (进一步调低 VF 权重) ---
                loss = actor_loss + 0.5 * value_loss - 0.02 * entropy
                
                self.opt_adapter.zero_grad(set_to_none=True)
                self.opt_actor.zero_grad(set_to_none=True)
                self.opt_critic.zero_grad(set_to_none=True)
                loss.backward()
                
                # 梯度裁剪保持 0.5
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 0.5)
                
                self.opt_adapter.step()
                self.opt_actor.step()
                self.opt_critic.step()
                
                with torch.no_grad():
                    kl_b = (logp_old_b - logp_new).mean().abs().item()
                    epoch_kl.append(kl_b)
                    stats["loss"].append(loss.item())
                    stats["kl"].append(kl_b)
                    stats["actor_loss"].append(actor_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy.item())

            if (sum(epoch_kl)/len(epoch_kl)) > 1.5 * target_kl: break

        # 更新 Norm & Scheduler
        if M > 0:
            with torch.no_grad():
                for st in range(0, M, mb):
                    self._build_state(curr_flat[st:st+mb].to(self.device), hist_flat[st:st+mb].to(self.device), norm_update=True)

        self.scheduler_actor.step()
        self.scheduler_critic.step()
        self.scheduler_adapter.step()
        
        def get_avg(k): return sum(stats[k]) / len(stats[k]) if stats[k] else 0.0
        return get_avg("loss"), get_avg("kl"), get_avg("actor_loss"), get_avg("value_loss"), get_avg("entropy")


    def p_update_from_trajs(self, trajs: List[Dict[str, Any]], target_kl: float = 0.015):
        """
        重构完整版：
        1. 加入 Value Clipping 防止价值网络预测炸裂导致 loss 飙升
        2. 调整 Loss 权重分配 (VF Coeff=0.1, Entropy Coeff=0.01)
        3. 全程使用 torch.stack/as_tensor 优化内存
        """
        assert self.actor is not None and self.critic is not None
        
        # ---------------------------------------------------------------------
        # 1. 统一堆叠 (Stacking) - CPU 上完成，避免内存峰值
        # ---------------------------------------------------------------------
        def stack_key(key, dtype):
            N = len(trajs)
            if N == 0: return torch.tensor([], dtype=dtype)
            
            first_val = trajs[0][key]
            sample_tensor = torch.as_tensor(first_val, dtype=dtype)
            sample_shape = sample_tensor.shape
            
            target_shape = list(sample_shape)
            target_shape.insert(1, N) # (T, N, ...)
            
            out = torch.empty(target_shape, dtype=dtype)
            for i, tr in enumerate(trajs):
                out.select(1, i).copy_(torch.as_tensor(tr[key], dtype=dtype))
            return out

        # 提取所有基础数据
        raw_curr  = stack_key("raw_curr", torch.float32) # (T, N, Dc)
        raw_hist  = stack_key("raw_hist", torch.float32) # (T, N, L, Dh)
        actions   = stack_key("actions", torch.long)     # (T, N)
        w_idx     = stack_key("w_idx", torch.long)       # (T, N)
        logp_old  = stack_key("logp_old", torch.float32) # (T, N)
        value_old = stack_key("value_old", torch.float32)# (T, N)
        rewards   = stack_key("rewards", torch.float32)  # (T, N)
        done      = stack_key("done", torch.float32)     # (T, N)
        valid     = stack_key("valid", torch.float32)    # (T, N)
        
        last_value = torch.stack([torch.as_tensor(tr["last_value"], dtype=torch.float32) for tr in trajs], dim=0).squeeze()

        T, N = raw_curr.shape[:2]

        # ---------------------------------------------------------------------
        # 2. 计算 GAE (CPU 计算)
        # ---------------------------------------------------------------------
        with torch.no_grad():
            adv = torch.zeros((T, N), dtype=torch.float32)
            last_gae = torch.zeros((N,), dtype=torch.float32)
            
            for t in reversed(range(T)):
                # m: mask，只有在未完成且有效的步数才累积 GAE
                m = (1.0 - done[t]) * valid[t]
                v_tp1 = last_value if t == T - 1 else value_old[t + 1]
                delta = rewards[t] + self.gamma * v_tp1 * m - value_old[t]
                last_gae = delta + self.gamma * self.gae_lambda * m * last_gae
                adv[t] = last_gae * valid[t]

            # 这里的 ret 是 Critic 的拟合目标 (Returns)
            ret = adv + value_old

        # ---------------------------------------------------------------------
        # 3. Flatten & Filter (基于 valid 过滤无效数据)
        # ---------------------------------------------------------------------
        mask_flat = valid.view(-1) > 0.5
        if not mask_flat.any():
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        def flat_and_filter(tensor_in):
            return tensor_in.flatten(0, 1)[mask_flat]

        curr_flat = flat_and_filter(raw_curr)
        hist_flat = flat_and_filter(raw_hist)
        act_flat  = flat_and_filter(actions)
        widx_flat = flat_and_filter(w_idx)
        logp_old_flat = flat_and_filter(logp_old)
        value_old_flat = flat_and_filter(value_old) # 关键：用于 Value Clipping
        adv_raw_flat = flat_and_filter(adv)
        ret_flat  = flat_and_filter(ret)

        # 优势归一化与强制裁剪 [-5, 5] 防止梯度爆炸
        adv_flat = (adv_raw_flat - adv_raw_flat.mean()) / (adv_raw_flat.std() + 1e-8)
        adv_flat = torch.clamp(adv_flat, -5.0, 5.0)

        # 打印监控数据
        with torch.no_grad():
            x1 = (torch.abs(adv_flat) > 3.0).sum().item() / adv_flat.shape[0]
            x2 = (torch.abs(adv_flat) > 2.0).sum().item() / adv_flat.shape[0]
            x3 = (torch.abs(adv_flat) > 1.0).sum().item() / adv_flat.shape[0]
            print(f"【Adv Ratio】 >3.0: {x1:.2%} | >2.0: {x2:.2%} | >1.0: {x3:.2%}")
            print(f"Advantage Max: {adv_flat.max().item():.4f} | Min: {adv_flat.min().item():.4f}")

        # ---------------------------------------------------------------------
        # 4. PPO Training Loop (Mini-batch)
        # ---------------------------------------------------------------------
        M = curr_flat.shape[0]
        mb = self.update_mb_size
        stats = {"loss": [], "kl": [], "actor_loss": [], "value_loss": [], "entropy": []}
        
        for _ep in range(self.k_epochs):
            indices = torch.randperm(M)
            epoch_kl = []
            
            for st in range(0, M, mb):
                idx = indices[st:st + mb]
                if len(idx) == 0: continue
                
                # --- 异步加载到 GPU ---
                curr_b = curr_flat[idx].to(self.device, non_blocking=True)
                hist_b = hist_flat[idx].to(self.device, non_blocking=True)
                act_b  = act_flat[idx].to(self.device, non_blocking=True)
                widx_b = widx_flat[idx].to(self.device, non_blocking=True)
                logp_old_b = logp_old_flat[idx].to(self.device, non_blocking=True)
                adv_b  = adv_flat[idx].to(self.device, non_blocking=True)
                ret_b  = ret_flat[idx].to(self.device, non_blocking=True)
                v_old_b = value_old_flat[idx].to(self.device, non_blocking=True) # 用于计算 Value Clipping

                # --- Actor Forward ---
                _, s = self._build_state(curr_b, hist_b, norm_update=False)
                logits_a, logits_w = self.actor(s)
                
                # Action A 分布
                dist_a = Categorical(logits=logits_a)
                new_logp_a = dist_a.log_prob(act_b)
                ent_a = dist_a.entropy().mean()

                # Weight Action 掩码逻辑
                need_w = ((act_b == 1) | (act_b == 2) | (act_b == 3)).float() # 1:Long, 2:Short, 3:Close
                lw = logits_w.clone()
                if need_w.sum() > 0:
                    maskw = torch.zeros_like(lw, dtype=torch.bool)
                    maskw[need_w.bool(), 1:] = True
                    maskw[~need_w.bool(), 0] = True
                    lw[~maskw] = -1e9
                
                dist_w = Categorical(logits=lw)
                new_logp_w = dist_w.log_prob(widx_b)
                ent_w = (need_w * dist_w.entropy()).sum() / (need_w.sum() + 1e-6)

                # 计算 PPO Ratio
                logp_new = new_logp_a + need_w * new_logp_w
                ratio = torch.exp(logp_new - logp_old_b)

                # Actor Loss (Clipped Surrogate Objective)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- 🔥 改进的 Critic Forward (Value Clipping) ---
                v_pred = self.critic(s).squeeze(-1)
                
                # v_loss1: 原始 MSE
                v_loss_unclipped = (v_pred - ret_b) ** 2
                # v_loss2: 基于采样时旧 Value 的裁剪 MSE
                v_pred_clipped = v_old_b + torch.clamp(v_pred - v_old_b, -self.clip_eps, self.clip_eps)
                v_loss_clipped = (v_pred_clipped - ret_b) ** 2
                
                # 综合 Value Loss，取最大值能带来更稳健的梯度
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # 熵正则化
                entropy = ent_a + 0.5 * ent_w
                
                # --- 🔥 总 Loss 计算 (VF Coeff=0.1, Ent Coeff=0.01) ---
                # 降低价值网络权重，从 0.5 降到 0.1，平滑由于 Return 波动带来的 loss 飙升
                loss = actor_loss + 0.1 * value_loss - 0.01 * entropy
                
                # --- Backward & Step ---
                self.opt_adapter.zero_grad(set_to_none=True)
                self.opt_actor.zero_grad(set_to_none=True)
                self.opt_critic.zero_grad(set_to_none=True)
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 0.5)
                
                self.opt_adapter.step()
                self.opt_actor.step()
                self.opt_critic.step()
                
                # 统计
                with torch.no_grad():
                    kl_b = (logp_old_b - logp_new).mean().abs().item()
                    epoch_kl.append(kl_b)
                    stats["loss"].append(loss.item())
                    stats["kl"].append(kl_b)
                    stats["actor_loss"].append(actor_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy.item())

            # 早停检查
            avg_kl = sum(epoch_kl) / len(epoch_kl) if epoch_kl else 0
            if avg_kl > 1.5 * target_kl: break

        # ---------------------------------------------------------------------
        # 5. 更新状态归一化统计量 (RunningMeanStd)
        # ---------------------------------------------------------------------
        if M > 0:
            with torch.no_grad():
                for st in range(0, M, mb):
                    idx = slice(st, min(st + mb, M))
                    curr_b = curr_flat[idx].to(self.device, non_blocking=True)
                    hist_b = hist_flat[idx].to(self.device, non_blocking=True)
                    self._build_state(curr_b, hist_b, norm_update=True)

        # 学习率 Scheduler 步进
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        self.scheduler_adapter.step()
        
        def get_avg(k):
            return sum(stats[k]) / len(stats[k]) if stats[k] else 0.0

        return get_avg("loss"), get_avg("kl"), get_avg("actor_loss"), get_avg("value_loss"), get_avg("entropy")


    def old_update_from_trajs(self, trajs: List[Dict[str, Any]], target_kl: float = 0.015):
        """
        returns: loss, kl, actor_loss, value_loss, entropy
        重构重点：
        1. 全程移除 np.stack，全部改为 torch.stack/as_tensor，彻底解决 _ArrayMemoryError。
        2. 统一数据流：List -> CPU Tensor -> Flatten & Filter -> GPU Mini-batch。
        """
        assert self.actor is not None and self.critic is not None
        
        # ---------------------------------------------------------------------
        # 1. 统一堆叠 (Stacking) - 全部在 CPU 上完成，避免 Numpy 内存峰值
        # ---------------------------------------------------------------------
        # 辅助函数：处理不同类型的数据并转为 Tensor

        def stack_key(key, dtype):
            # --- 优化开始 ---
            # 获取 trajs 的长度 (N)
            N = len(trajs)
            if N == 0:
                return torch.tensor([], dtype=dtype)

            # 1. 拿第一个元素来确定形状 (避免把所有元素都转成 Tensor 放在列表里)
            # 注意：这里只转第一个，开销很小
            first_val = trajs[0][key]
            sample_tensor = torch.as_tensor(first_val, dtype=dtype)
            sample_shape = sample_tensor.shape

            # 2. 计算目标 Tensor 的形状
            # 原代码是 dim=1，意味着在第 1 维插入 N
            # 例如: 单个是 (T, L, Dh)，结果就是 (T, N, L, Dh)
            target_shape = list(sample_shape)
            target_shape.insert(1, N)

            # 3. 预分配内存
            # 使用 empty 分配内存比 zeros 快，因为我们要马上覆盖它
            out = torch.empty(target_shape, dtype=dtype)

            # 4. 逐个填充 (这是省内存的关键)
            # 这样每次循环只处理一个 trajectory 的数据，处理完就释放中间变量
            for i, tr in enumerate(trajs):
                # out.select(1, i) 选中第 i 个切片，将数据 copy 进去
                # torch.as_tensor 会尽可能共享内存，减少复制
                val = tr[key]
                # 如果 val 已经是 tensor 且设备与 out 一致，copy_ 很快
                out.select(1, i).copy_(torch.as_tensor(val, dtype=dtype))
                
            return out
        # --- 优化结束 ---

        # 只要 trajs 里的 raw_hist 是 numpy array，torch.as_tensor 会很高效
        raw_curr  = stack_key("raw_curr", torch.float32) # (T, N, Dc)
        raw_hist  = stack_key("raw_hist", torch.float32) # (T, N, L, Dh)
        
        actions   = stack_key("actions", torch.long)     # (T, N)
        w_idx     = stack_key("w_idx", torch.long)       # (T, N)
        logp_old  = stack_key("logp_old", torch.float32) # (T, N)
        value_old = stack_key("value_old", torch.float32)# (T, N)
        rewards   = stack_key("rewards", torch.float32)  # (T, N)
        done      = stack_key("done", torch.float32)     # (T, N)
        valid     = stack_key("valid", torch.float32)    # (T, N)
        
        # last_value 处理稍微不同，它是 (N,)
        last_value = torch.stack([torch.as_tensor(tr["last_value"], dtype=torch.float32) for tr in trajs], dim=0).squeeze()

        T, N = raw_curr.shape[:2]

        # ---------------------------------------------------------------------
        # 2. 计算 GAE (仍在 CPU 上，计算量很小，不需要 GPU)
        # ---------------------------------------------------------------------
        with torch.no_grad():
            adv = torch.zeros((T, N), dtype=torch.float32)
            last_gae = torch.zeros((N,), dtype=torch.float32)
            
            # 这里的计算逻辑完全不需要改，但现在变量全是 Tensor 了
            for t in reversed(range(T)):
                m = (1.0 - done[t]) * valid[t]
                v_tp1 = last_value if t == T - 1 else value_old[t + 1]
                delta = rewards[t] + self.gamma * v_tp1 * m - value_old[t]
                last_gae = delta + self.gamma * self.gae_lambda * m * last_gae
                adv[t] = last_gae * valid[t]

            ret = adv + value_old

            # -----------------------------------------------------------------
            # 3. Flatten & Filter (基于 valid mask 过滤无效步)
            # -----------------------------------------------------------------
            # 创建 bool mask (T*N)
            mask_flat = valid.view(-1) > 0.5
            if not mask_flat.any():
                return 0.0, 0.0, 0.0, 0.0, 0.0
            
            # 定义 Flatten 函数
            def flat_and_filter(tensor_in):
                # reshape 到 (T*N, ...) 然后取 mask
                return tensor_in.flatten(0, 1)[mask_flat]

            # 大块内存操作：这里会产生一份新的过滤后的内存，但因为过滤掉了无效数据，通常比原数据小
            curr_flat = flat_and_filter(raw_curr)       # (M, Dc)
            hist_flat = flat_and_filter(raw_hist)       # (M, L, Dh)
            act_flat  = flat_and_filter(actions)        # (M,)
            widx_flat = flat_and_filter(w_idx)          # (M,)
            logp_old_flat = flat_and_filter(logp_old)   # (M,)
            adv_flat  = flat_and_filter(adv)            # (M,)
            ret_flat  = flat_and_filter(ret)            # (M,)

            # 归一化 Advantage
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        
        advs = adv_flat
        adv_flat = torch.clamp(advs, -5.0, 5.0)
        x1 = (torch.abs(advs) > 3.0).sum().item() / advs.shape[0]
        x2 = (torch.abs(advs) > 2.0).sum().item() / advs.shape[0]
        x3 = (torch.abs(advs) > 1.0).sum().item() / advs.shape[0]
        print(f"【Ratio】 (abs(adv) > 3.0 | 2.0 | 1.0) : {x1} | {x2} | {x3}")
        print(f"Advantage Max: {advs.max().item():.4f} | Min: {advs.min().item():.4f} | Mean: {advs.mean().item():.4f}")

        # ---------------------------------------------------------------------
        # 4. PPO Training Loop (Mini-batch)
        # ---------------------------------------------------------------------
        M = curr_flat.shape[0] # 有效样本总数
        mb = self.update_mb_size
        
        # 记录器
        stats = {"loss": [], "kl": [], "actor_loss": [], "value_loss": [], "entropy": []}
        
        for _ep in range(self.k_epochs):
            # 生成随机索引
            indices = torch.randperm(M)
            
            epoch_kl = []
            
            for st in range(0, M, mb):
                idx = indices[st:st + mb]
                if len(idx) == 0: continue
                
                # --- Move to GPU (显存只在这里消耗) ---
                curr_b = curr_flat[idx].to(self.device, non_blocking=True)
                hist_b = hist_flat[idx].to(self.device, non_blocking=True)
                act_b  = act_flat[idx].to(self.device, non_blocking=True)
                widx_b = widx_flat[idx].to(self.device, non_blocking=True)
                logp_old_b = logp_old_flat[idx].to(self.device, non_blocking=True)
                adv_b  = adv_flat[idx].to(self.device, non_blocking=True)
                ret_b  = ret_flat[idx].to(self.device, non_blocking=True)
                
                # --- Forward ---
                # norm_update=False: 训练阶段不更新 running_mean/std，保持稳定
                _, s = self._build_state(curr_b, hist_b, norm_update=False)

                logits_a, logits_w = self.actor(s)
                dist_a = Categorical(logits=logits_a)
                new_logp_a = dist_a.log_prob(act_b)
                ent_a = dist_a.entropy().mean()

                # Weight Action 处理逻辑
                need_w = ((act_b == A_LONG) | (act_b == A_SHORT) | (act_b == A_CLOSE)).float()
                
                # 优化 maskw 的创建，避免 clone 整个 logits
                lw = logits_w  # 如果不需要 inplace 修改，直接用
                # 这里为了安全还是 clone 一下，防止 inplace 报错，但操作可以简化
                if need_w.sum() > 0:
                    # 只有当需要处理权重时才进行复杂的 mask 操作
                    # (由于你的逻辑比较特殊，这里保留原逻辑，但加上 clone)
                    lw = logits_w.clone()
                    maskw = torch.zeros_like(lw, dtype=torch.bool)
                    maskw[need_w.bool(), 1:] = True
                    maskw[~need_w.bool(), 0] = True
                    lw[~maskw] = -1e9
                
                dist_w = Categorical(logits=lw)
                new_logp_w = dist_w.log_prob(widx_b)
                ent_w = (need_w * dist_w.entropy()).sum() / (need_w.sum() + 1e-6)

                logp_new = new_logp_a + need_w * new_logp_w
                ratio = torch.exp(logp_new - logp_old_b)

                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()

                v_pred = self.critic(s).squeeze(-1)
                
                value_loss = F.mse_loss(v_pred, ret_b)

                entropy = ent_a + 0.5 * ent_w
                loss = actor_loss + 0.5 * value_loss - 0.005 * entropy
                
                # --- Backward ---
                self.opt_adapter.zero_grad(set_to_none=True)
                self.opt_actor.zero_grad(set_to_none=True)
                self.opt_critic.zero_grad(set_to_none=True)
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 0.5)
                
                self.opt_adapter.step()
                self.opt_actor.step()
                self.opt_critic.step()
                
                # Stats
                with torch.no_grad():
                    kl_b = (logp_old_b - logp_new).mean().abs().item()
                    epoch_kl.append(kl_b)
                    
                    stats["loss"].append(loss.item())
                    stats["kl"].append(kl_b)
                    stats["actor_loss"].append(actor_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy.item())

            # Early Stopping Check (Epoch Level)
            avg_kl = sum(epoch_kl) / len(epoch_kl) if epoch_kl else 0
            if avg_kl > 1.5 * target_kl:
                break
        
        # ---------------------------------------------------------------------
        # 5. Update Norm Stats (单独一轮，不再计算梯度，仅更新 Running Mean/Std)
        # ---------------------------------------------------------------------
        # 这一步如果不做 backward，其实非常快
        if M > 0:
            with torch.no_grad():
                # 不需要 shuffle，直接顺序过一遍
                for st in range(0, M, mb):
                    idx = slice(st, min(st + mb, M))
                    curr_b = curr_flat[idx].to(self.device, non_blocking=True)
                    hist_b = hist_flat[idx].to(self.device, non_blocking=True)
                    # 只要前向传播，Adapter 内部的 RunningMeanStd 就会更新
                    self._build_state(curr_b, hist_b, norm_update=True)

        # Learning Rate Schedule
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        self.scheduler_adapter.step()
        
        # Helper to get average
        def get_avg(k):
            return sum(stats[k]) / len(stats[k]) if stats[k] else 0.0

        return get_avg("loss"), get_avg("kl"), get_avg("actor_loss"), get_avg("value_loss"), get_avg("entropy")
   


# =========================
# Agent
# =========================
@dataclass
class AgentConfig:
    option_pairs: List[Dict[str, Any]]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # env
    start_time: str = "20250408100000"
    end_time: str = "20250924150000"
    benchmark: str = "510050"
    fee: float = 1.3
    init_capital: float = 100000.0
    max_timesteps: int = 2000

    # rollout chunk
    rollout_T: int = 512
    num_workers: int = field(default_factory=lambda: max(1, min(mp.cpu_count() - 2, 12)))

    # model
    window_size: int = 32
    pre_len: int = 4
    n_variates: int = 13
    d_router: int = 128
    # adapter_dim: int = 128
    adapter_dim: int = 256

    # hidden_dim: int = 256
    hidden_dim: int = 512
    pretrained_path: str = "./miniQMT/DL/preTrain/weights/preMOE_best_dummy_data_32_4.pth"

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    k_epochs: int = 8
    epochs: int = 50,
    actor_lr: float = 3e-4
    critic_lr: float = 5e-4
    check_path: str = './miniQMT/DL/checkout/check_data_parallel.pt'

    # logging
    save_excel: bool = False
    excel_path: str = "./miniQMT/DL/results/PPO_training_data.xlsx"

    mini_batch: int=2048


class Agent:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.cfg.pretrained_path = f"./miniQMT/DL/preTrain/weights/preMOE_best_dummy_data_{cfg.window_size}_{cfg.pre_len}.pth"
        self.device = cfg.device
        self.pairs = cfg.option_pairs
        assert len(self.pairs) > 0, "option_pairs is empty"

        # build learner
        self.learner = LearnerPPO(
            device=cfg.device,
            window_size=cfg.window_size,
            pre_len=cfg.pre_len,
            n_variates=cfg.n_variates,
            d_router=cfg.d_router,
            pretrained_path=cfg.pretrained_path,
            adapter_dim=cfg.adapter_dim,
            hidden_dim=cfg.hidden_dim,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_eps=cfg.clip_eps,
            k_epochs=cfg.k_epochs,
            update_mb_size=cfg.mini_batch,
            total_epochs = cfg.epochs,
            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr,
            check_path=cfg.check_path
        )

        # build vector env
        workers = min(cfg.num_workers, len(self.pairs))
        workers = max(1, workers)

        env_fns = []
        base_env_cfg = {
            "start_time": cfg.start_time,    # 如果 cfg 没有，就删掉这两行
            "end_time": cfg.end_time,
            "benchmark": cfg.benchmark,
            "fee": cfg.fee,
            "init_capital": cfg.init_capital,
        }

        for i in range(workers):
            # 关键：每个 make_env 用一个“独立拷贝”的 cfg，避免闭包/引用问题
            c = dict(base_env_cfg)

            def make_env(seed=i, pairs=self.pairs, c=c):
                return DynamicWindowEnv(pairs, c, seed=seed)
            env_fns.append(make_env)


        worker_cfg = {
            "window_size": cfg.window_size,
            "pre_len": cfg.pre_len,
            "n_variates": cfg.n_variates,
            "d_router": cfg.d_router,
            "pretrained_path": cfg.pretrained_path,
            "adapter_dim": cfg.adapter_dim,
            "hidden_dim": cfg.hidden_dim,
        }

        self.vec_env = SubprocVectorEnv(env_fns, worker_cfg=worker_cfg)
        self.workers = workers

        # warmup build learner modules
        tmp = env_fns[0]()
        c0, h0, _ = tmp.reset()
        tmp.close()
        c0_t = torch.from_numpy(np.asarray(c0, np.float32)).to(self.device)
        h0_t = torch.from_numpy(np.asarray(h0, np.float32)).to(self.device)

        self.learner._maybe_build(c0_t.unsqueeze(0), h0_t.unsqueeze(0))

        self.records = {
            'epoch': [], 'reward': [], 'avg_equity': [], 'loss': [], 'kl': [],
            'hold_ratio': [], 'long_ratio': [], 'short_ratio': [], 'close_ratio': [],
            'actor_loss': [], 'value_loss': [], 'entropy': [],
            'ratio_0': [], 'ratio_25': [], 'ratio_50': [], 'ratio_75': [], 'ratio_100': [],
            'sharpe': [],
            'right_sharpe_ratio': [],
            'golden_task_count': [],   # ✅ 新增：进入黄金区间的任务数量
            'gambling_task_count': [], # ✅ 新增：在赌博区间的任务数量
        }


        # Warmup Normalization (修复版)
        print("[Info] Warming up Normalization layers...")
        # 1. 随机派发任务
        self.vec_env.set_tasks([random.randint(0, len(self.pairs)-1) for _ in range(workers)])
        
        # 2. 跑数据
        trajs = self.vec_env.rollout(512)
        
        # 3. 🔥关键修复：收集所有 raw data 并更新 Learner 的 Norm
        # 我们借用 learner 内部的方法来构建状态，强制 update=True
        all_curr = np.concatenate([t["raw_curr"] for t in trajs], axis=0)
        all_hist = np.concatenate([t["raw_hist"] for t in trajs], axis=0)
        
        # 转为 Tensor
        c_t = torch.from_numpy(all_curr).float().to(self.device)
        h_t = torch.from_numpy(all_hist).float().to(self.device)
        
        # 强制更新 Norm
        self.learner._build_state(c_t, h_t, norm_update=True)
        print(f"[Info] Warmup done. Norm counts: {self.learner.norm.running_ms.n}")


    def train_dynamic(self, from_check_point: bool = False):
        print(f'shape = {self.learner.norm.running_ms.mean.shape}')
        if from_check_point:
            sys.stdout = outPut("./miniQMT/DL/results/PPO_records.txt", mode='a')
        else:
            sys.stdout = outPut("./miniQMT/DL/results/PPO_records.txt", mode='w')
        

        """
        ✅ 新逻辑：按 steps 排序 -> 按 num_workers 分组 -> 按“组内 steps 总和”比例随机抽组
        累积抽到的组的总 steps >= rollout_T_big(默认8192) 后，才执行一次 PPO update。

        注意：
        - 每次抽到一个组，会让所有 worker 各跑一个 pair（组不满 workers 会 padding 重复最后一个 idx，但 update/统计时会丢弃重复部分）
        - 每个组的 rollout_len = 该组内 max(steps)（保证组内每个 pair 都能跑完整周期；短的会提前 done -> valid=False padding）
        - 收集多次 rollout 得到的 traj 长度不同：update 前统一 pad 到 Tmax，并用 valid mask 屏蔽 padding。
        """

        # -------------------------
        # 0) 基本参数
        # -------------------------
        total_pairs = len(self.pairs)
        print(f'[Agent-init-train] 期权组合数量 = {total_pairs} | 进程数量 = {self.cfg.num_workers}')
        current_time = datetime.now()
        formatted_time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")

        print(f' ------ Start to train PPO on {self.device}, Time_stamp: {formatted_time_string} --- ')
        
        # PPO 算法核心参数
        print(f'PPO Hyperparams 1: gamma = {self.learner.gamma}, gae_lambda = {self.learner.gae_lambda}, clip_eps = {self.learner.clip_eps}')
        
        # 训练迭代和批次大小
        print(f'PPO Hyperparams 2: k_epochs = {self.learner.k_epochs}, update_mb_size = {self.learner.update_mb_size}, total_epochs = {self.learner.total_epochs}')
        
        # 学习率配置
        print(f'PPO Hyperparams 3: actor_lr = {self.learner.actor_lr}, critic_lr = {self.learner.critic_lr}')
        
        # 模型结构维度
        print(f'Model Dims: adapter_dim = {self.learner.adapter_dim}, hidden_dim = {self.learner.hidden_dim}')
        print(f'------------------------------------------------------------------------------')
        workers = self.workers

        # 大的采样目标：默认 8192（优先用 cfg.rollout_T_big；否则退化用 cfg.rollout_T；否则 8192）
        rollout_T_big = int(getattr(self.cfg, "rollout_T_big", getattr(self.cfg, "rollout_T", 8192)))
        if rollout_T_big <= 0:
            rollout_T_big = 8192
        
        print(f"[Info] 采样目标, rollout_T_big = {rollout_T_big}")

        # 早停相关
        best_reward = -float('inf')
        patience = getattr(self.cfg, 'patience', 15)
        stop_entropy = getattr(self.cfg, 'stop_entropy', 0.8)
        min_delta = 0.001
        early_stop_counter = 0

        # records 确保存在
        if not hasattr(self, "records") or self.records is None:
            self.records = {
                "epoch": [], "reward": [], "avg_equity": [], "loss": [], "kl": [],
                "hold_ratio": [], "long_ratio": [], "short_ratio": [], "close_ratio": [],
                "actor_loss": [], "value_loss": [], "entropy": [],
                "ratio_0": [], "ratio_25": [], "ratio_50": [], "ratio_75": [], "ratio_100": [],
            }

        # checkpoint
        start_epoch = 0
        if from_check_point:
            start_epoch, best_reward = self.learner.load_checkpoint()

            print(f'[Info] 从checkpoint开始训练, last_epoch = {start_epoch}, last_best_reward = {best_reward}')

        # -------------------------
        # 1) 预处理：按 steps 排序，并按 workers 分组
        # -------------------------
        # 每个 pair 必须有 steps
        steps_arr = np.array([int(p.get("steps", 0)) for p in self.pairs], dtype=np.int64)
        if (steps_arr <= 0).any():
            bad = np.where(steps_arr <= 0)[0][:10].tolist()
            raise ValueError(f"[train] Found invalid steps<=0 at indices: {bad} (showing first 10).")

        sorted_ids = np.argsort(-steps_arr)  # desc

        groups = []          # list[list[int]]
        group_sum_steps = [] # list[int]  采样概率用：组内 steps 总和（组不满 workers 也没问题）
        group_max_steps = [] # list[int]  rollout_len 用：组内 max steps

        for i in range(0, total_pairs, workers):
            g = sorted_ids[i:i + workers].tolist()
            if len(g) == 0:
                continue
            groups.append(g)
            group_sum_steps.append(int(steps_arr[g].sum()))
            group_max_steps.append(int(steps_arr[g].max()))

        # 概率（与组内 steps 总和成正比）
        sum_all = float(sum(group_sum_steps))
        probs = [float(s) / (sum_all + 1e-12) for s in group_sum_steps]

        # 用随机数采样组（不依赖 np.random.choice，方便你调试）
        def sample_group_index():
            r = random.random()
            c = 0.0
            for idx, p in enumerate(probs):
                c += p
                if r <= c:
                    return idx
            return len(probs) - 1

        # -------------------------
        # 2) pad helper：统一 traj 长度到 Tmax，padding 部分 valid=False
        # -------------------------
        def pad_traj_to_T(tr, T_max: int):
            T0 = int(tr["raw_curr"].shape[0])
            if T0 == T_max:
                return tr

            def pad_arr(x, pad_value, dtype=None):
                x = np.asarray(x)
                if dtype is not None:
                    x = x.astype(dtype, copy=False)
                if x.ndim == 1:
                    out = np.full((T_max,), pad_value, dtype=x.dtype)
                    out[:T0] = x
                    return out
                if x.ndim == 2:
                    out = np.full((T_max, x.shape[1]), pad_value, dtype=x.dtype)
                    out[:T0, :] = x
                    return out
                if x.ndim == 3:
                    out = np.full((T_max, x.shape[1], x.shape[2]), pad_value, dtype=x.dtype)
                    out[:T0, :, :] = x
                    return out
                raise ValueError(f"Unsupported ndim={x.ndim} for padding.")

            out = dict(tr)
            out["raw_curr"] = pad_arr(tr["raw_curr"], 0.0, dtype=np.float32)
            out["raw_hist"] = pad_arr(tr["raw_hist"], 0.0, dtype=np.float32)

            out["actions"] = pad_arr(tr["actions"], A_HOLD, dtype=np.int64)
            out["w_idx"] = pad_arr(tr["w_idx"], 0, dtype=np.int64)
            out["w_val"] = pad_arr(tr["w_val"], 0.0, dtype=np.float32)

            out["logp_old"] = pad_arr(tr["logp_old"], 0.0, dtype=np.float32)
            out["value_old"] = pad_arr(tr["value_old"], 0.0, dtype=np.float32)
            out["rewards"] = pad_arr(tr["rewards"], 0.0, dtype=np.float32)

            out["done"] = pad_arr(np.asarray(tr["done"], np.bool_), True, dtype=np.bool_)
            out["valid"] = pad_arr(np.asarray(tr["valid"], np.bool_), False, dtype=np.bool_)

            # bootstrap：padding 后直接置 0（不影响）
            out["last_value"] = np.array([0.0], dtype=np.float32)
            return out

        # -------------------------
        # 3) 主循环：每个 ep 做 1 次 PPO 更新（但 rollout 可以抽多组累积）
        # -------------------------
        sys.stdout.flush() # 强制将缓冲区写入磁盘

        for ep in range(self.cfg.epochs):
            if from_check_point and start_epoch is not None and ep <= start_epoch:
                print(f'[Skip] epoch = {ep}')
                continue

            t0 = time.time()
            
            if ep == 0 or ep == 1: # 每50轮打印并同步一次即可
                #  print(f"[Curriculum-warmup] Set Fee to {current_fee}")
                #  self.vec_env.set_fee_all(current_fee)
                 self.vec_env.set_fee_all(1.3)

            # 本轮累计的组 steps（按你要求用“组内 steps 总和”来判断是否够 8192）
            sampled_steps_sum = 0

            # 收集到的 traj（注意：每次抽组会返回 workers 条，但最后一组可能 < workers，我们会丢弃 padding 部分）
            collected_trajs = []
            epoch_all_env_sharpes = []

            # 统计（按 valid 统计）
            total_reward_sum = 0.0
            total_valid_steps = 0
            action_counts = np.zeros(4, dtype=np.int64)
            weight_counts = np.zeros(5, dtype=np.int64)
            equity_sum = 0.0
            equity_cnt = 0


            total_annual_ret_sum = 0.0
            count = 0
            task_sharpe_tracker = {} # {task_id: [sr1, sr2, ...]}
            # 反复抽组直到累计组 steps >= rollout_T_big
            while sampled_steps_sum < rollout_T_big:
                g_idx = sample_group_index()
                group = groups[g_idx]
                true_cnt = len(group)

                task_ids_raw = group.copy()

                # 【新增】预判逻辑：如果加上这组会超过目标太多(比如1.1倍)，且当前已经有不少数据了，就跳过这组或直接 break
                # 防止内存压力太大
                if sampled_steps_sum > rollout_T_big * 0.8 and (sampled_steps_sum + group_sum_steps[g_idx]) > rollout_T_big * 1.2:
                     # 这一组太大，容易爆内存，换一组小的试试，或者直接 break
                    if sampled_steps_sum >= rollout_T_big: 
                        break
                    continue


                # 组贡献的“长度”（采样概率/累计步数使用 sum steps）
                sampled_steps_sum += int(group_sum_steps[g_idx])

                # rollout_len 用 max steps，确保组内每个 pair 都能跑到自己 done（完整周期）
                rollout_len = int(group_max_steps[g_idx])


                # padding tasks：填满 workers（避免 vec_env 断言）
                task_ids = group.copy()
                print(f"choose_task_ids = {task_ids}, sum_roll_len = {sampled_steps_sum}")

                mx = max(self.pairs[i]["steps"] for i in task_ids)
                mn = min(self.pairs[i]["steps"] for i in task_ids)
                # print("group steps min/max:", mn, mx)

                while len(task_ids) < workers:
                    task_ids.append(task_ids[-1])

                # 1) set tasks
                self.vec_env.set_tasks(task_ids)

                # 2) broadcast weights + norm snapshot
                payload = self.learner.export_payload()

                self.vec_env.set_weights_all(payload)

                # 3) one-shot rollout（一次 IPC 收全轨迹）
                trajs = self.vec_env.rollout(rollout_len)

                # 只保留真实组内的 true_cnt 条（丢弃 padding 重复）
                trajs = trajs[:true_cnt]
                collected_trajs.extend(trajs)

            # ✅ 新增：将夏普归类到具体的 Task ID 下
                for tid, tr in zip(task_ids_raw, trajs):
                    sr = tr.get("env_sharpe", 0.0)
                    if tid not in task_sharpe_tracker:
                        task_sharpe_tracker[tid] = []
                    task_sharpe_tracker[tid].append(sr)
                    epoch_all_env_sharpes.append(sr)

                # 年化(252交易日, 1天8个30分钟K线)
                STEPS_PER_YEAR = 252 * 8

                # 统计（只统计 valid 的部分）
                for tr in trajs:
                    es = tr.get("env_sharpe", 0.0)
                    epoch_all_env_sharpes.append(es)

                    valid = np.asarray(tr["valid"], dtype=np.bool_)
                    valid_steps = valid.sum()

                    r = np.asarray(tr["rewards"], dtype=np.float32)

                    total_reward_sum += float(r[valid].sum())
                    total_valid_steps += int(valid.sum())

                    acts = np.asarray(tr["actions"], dtype=np.int64)[valid]
                    if acts.size > 0:
                        action_counts += np.bincount(acts, minlength=4)

                    widx = np.asarray(tr["w_idx"], dtype=np.int64)[valid]
                    if widx.size > 0:
                        weight_counts += np.bincount(widx, minlength=5)

                    if "equity_end" in tr:
                        eq = float(np.asarray(tr["equity_end"]).reshape(-1)[0])
                        if not np.isnan(eq):
                            equity_sum += eq
                            equity_cnt += 1

                            if valid_steps > 0:
                                abs_ret = (eq - self.cfg.init_capital) / self.cfg.init_capital
                                annual_ret = abs_ret * (STEPS_PER_YEAR / valid_steps)
                                total_annual_ret_sum += annual_ret
                                count += 1

            # --- 🔥 核心改进：在 while 循环结束后，进行任务画像分析 ---
            task_analysis = {"golden": [], "gambling": [], "failing": [], "normal": []}
            task_avg_srs = {}
            for tid, srs in task_sharpe_tracker.items():
                avg_sr = np.mean(srs)
                task_avg_srs[tid] = avg_sr.item()
                if 1.0 <= avg_sr <= 2.5:
                    task_analysis["golden"].append(tid)
                elif avg_sr > 3.0:
                    task_analysis["gambling"].append(tid)
                elif avg_sr < 0.5:
                    task_analysis["failing"].append(tid)
                else:
                    task_analysis["normal"].append(tid)

            # 打印详细任务报告
            print(f"\n--- 📊 Task Performance Report (Epoch {ep+1}) ---")
            print(f"✅ [Golden] (1.0<=SR<=2.5) 数量: {len(task_analysis['golden'])} | IDs: {task_analysis['golden']}")
            print(f"🚀 [Gambling] (SR > 3.0) 数量: {len(task_analysis['gambling'])} | IDs: {task_analysis['gambling']}")
            print(f"📉 [Failing] (SR < 0.5) 数量: {len(task_analysis['failing'])} | IDs: {task_analysis['failing'][:10]}...")
            
            # 找到表现最好和最差的 3 个组合
            sorted_tasks = sorted(task_avg_srs.items(), key=lambda x: x[1], reverse=True)
            if sorted_tasks:
                print(f"🏆 顶梁柱 Top 3 (ID, SR): {sorted_tasks[:3]}")
                print(f"💀 拖后腿 Bottom 3 (ID, SR): {sorted_tasks[-3:]}")
            print(f"--------------------------------------------\n")

            # 🔥 3. 在 while 循环结束后，计算全体统计量
            if len(epoch_all_env_sharpes) > 0:
                sr_mean = np.mean(epoch_all_env_sharpes)
                sr_max = np.max(epoch_all_env_sharpes)
                sr_min = np.min(epoch_all_env_sharpes)
            else:
                sr_mean = sr_max = sr_min = 0.0   

            sp_cnt = 0
            for sp in epoch_all_env_sharpes:
                if sp >= 1.0 and sp <= 2.0:
                    sp_cnt += 1
            sp_cnt = sp_cnt / len(epoch_all_env_sharpes)

            # -------------------------
            # 4) PPO update：把所有 collected_trajs pad 到 Tmax 后一次更新
            # -------------------------

            # 1. 先计算最大长度 Tmax
            # 注意：这里只读 shape，不复制数据，很快且不占内存
            Tmax = max(int(tr["raw_curr"].shape[0]) for tr in collected_trajs)

            # 2. 逐个处理并释放旧数据 (关键修改)
            trajs_for_update = []
            
            # 使用 while 循环配合 pop(0)，处理一个就扔掉一个原始数据
            # 这样内存始终只维持一份数据，不会翻倍
            while collected_trajs:
                # 弹出第一个原始轨迹 (collected_trajs 长度 -1)
                tr = collected_trajs.pop(0) 
                
                # 生成 Padding 后的新轨迹
                padded_tr = pad_traj_to_T(tr, Tmax)
                
                # 加入新列表
                trajs_for_update.append(padded_tr)
                
                # 显式删除引用，告诉 Python "这块旧内存可以释放了"
                del tr
            
            # 3. 强制触发垃圾回收，整理内存碎片
            import gc
            gc.collect()


            # trajs_for_update = [pad_traj_to_T(tr, Tmax) for tr in collected_trajs]

            valid_lens = [int(tr["valid"].sum()) for tr in trajs_for_update]
            # print("valid_len min/mean/max:", min(valid_lens), sum(valid_lens)/len(valid_lens), max(valid_lens))

            rs = np.concatenate([
                            tr["rewards"][tr["valid"].astype(bool)] 
                            for tr in trajs_for_update
                        ])
            print("[Info] Scaled reward mean/std/min/max:", rs.mean(), rs.std(), rs.min(), rs.max())

            if ep >= 50:
                self.k_epochs = 5

            loss, kl, a_loss, v_loss, ent, ev = self.learner.update_from_trajs(trajs_for_update)
            # loss, kl, a_loss, v_loss, ent = self.learner.update_from_trajs(trajs_for_update)

            if 'ev' not in self.records: self.records['ev'] = []
            self.records['ev'].append(float(ev))
            
            # -------------------------
            # 5) 写 records + 打印 + 早停
            # -------------------------
            avg_reward = total_reward_sum / (total_valid_steps + 1e-8)
            
            # avg_equity = equity_sum / max(1, equity_cnt)
            avg_anual_ret = total_annual_ret_sum / max(1, count) 
            avg_equity = (1 + avg_anual_ret) * self.cfg.init_capital

            act_total = int(action_counts.sum())
            if act_total == 0:
                hold_ratio = long_ratio = short_ratio = close_ratio = 0.0
            else:
                hold_ratio = float(action_counts[A_HOLD] / act_total)
                long_ratio = float(action_counts[A_LONG] / act_total)
                short_ratio = float(action_counts[A_SHORT] / act_total)
                close_ratio = float(action_counts[A_CLOSE] / act_total)

            w_total = int(weight_counts.sum())
            if w_total == 0:
                w_ratios = [0.0] * 5
            else:
                w_ratios = [float(weight_counts[k] / w_total) for k in range(5)]

            self.records["sharpe"].append(float(sr_mean))
            self.records["right_sharpe_ratio"].append(sp_cnt)
            self.records["epoch"].append(ep + 1)
            self.records["reward"].append(float(avg_reward))
            self.records["avg_equity"].append(float(avg_equity))
            self.records["loss"].append(float(loss))
            self.records["kl"].append(float(kl))

            self.records["hold_ratio"].append(float(hold_ratio))
            self.records["long_ratio"].append(float(long_ratio))
            self.records["short_ratio"].append(float(short_ratio))
            self.records["close_ratio"].append(float(close_ratio))

            self.records["actor_loss"].append(float(a_loss))
            self.records["value_loss"].append(float(v_loss))
            self.records["entropy"].append(float(ent))

            self.records["ratio_0"].append(float(w_ratios[0]))
            self.records["ratio_25"].append(float(w_ratios[1]))
            self.records["ratio_50"].append(float(w_ratios[2]))
            self.records["ratio_75"].append(float(w_ratios[3]))
            self.records["ratio_100"].append(float(w_ratios[4]))

            self.records["golden_task_count"].append(len(task_analysis["golden"]))
            self.records["gambling_task_count"].append(len(task_analysis["gambling"]))


            dt = time.time() - t0
            print(
                f"[Epoch {ep+1} / {self.cfg.epochs}] "
                f"sampled_group_steps={sampled_steps_sum} target={rollout_T_big} | "
                f"valid_steps={total_valid_steps} | "
                f"Reward:{avg_reward:.6f} | EV:{ev:.4f} | Market_value:{avg_equity:.2f} \n"
                f"Sharpe Ratio -> Mean: {sr_mean:.4f} | Max: {sr_max:.4f} | Min: {sr_min:.4f} | "
                f"Sharpe Ratio Percent(1.0~2.0) -> Ratio: {sp_cnt:.4f}\n"
                f"loss={loss:.4f} kl={kl:.4f} | "
                f"act(H/L/S/C)={hold_ratio:.2f}/{long_ratio:.2f}/{short_ratio:.2f}/{close_ratio:.2f} | "
                f"entropy={ent:.3f} time={dt:.1f}s"
            )

            # 保存 Excel
            if getattr(self.cfg, "save_excel", False):
                if from_check_point is False:
                    pd.DataFrame(self.records).to_excel(self.cfg.excel_path, index=False)
                else:
                    df_old = pd.read_excel(self.cfg.excel_path)
                    df_combined = pd.concat([df_old, pd.DataFrame(self.records)], ignore_index=True)
                    df_combined.to_excel(self.cfg.excel_path, index=False)
            
            # 保存断点
            self.learner.save(ep, best_reward)
            print(f'>>> 🌟 Model updated, saved, epoch = {ep}, best_ward = {best_reward}')

            # B. 每 50 个 Epoch 强制备份一个历史版本，方便随时回滚
            if (ep + 1) % 50 == 0:
                history_path = f"{self.learner.check_path.replace('.pt', '')}_epoch_{ep+1}.pt"
                self.learner.save(ep, best_reward, path=history_path)
                print(f">>> 💾 Historical Checkpoint Saved: {history_path}")

            # --- 早停判断 ---
            if avg_reward > best_reward + min_delta:
                best_reward = avg_reward
                early_stop_counter = 0
                print(f"   >>> 🌟 Best Reward Updated: {best_reward:.4f} (Counter Reset)")
            else:
                early_stop_counter += 1
                print(f"   ⏳ [Patience] No improvement: {early_stop_counter}/{patience} | Best: {best_reward:.4f}")

            if early_stop_counter >= patience:
                print(f"\n🛑 [Early Stop] Triggered! Reward has not improved for {patience} epochs.")
                print(f"   Final Best Reward: {best_reward:.4f}")
                break

            if ent < stop_entropy and avg_reward > 0:
                print(f"\n🛑 [Early Stop] Triggered! Entropy ({ent:.4f}) is too low.")
                self.learner.save(ep, best_reward)
                break

            current_time = datetime.now()
            formatted_time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
            print(f'[Info] Finish train epoch {ep + 1} | Time-stamp: {formatted_time_string}')
            sys.stdout.flush() # 强制将缓冲区写入磁盘

        print(f"[Train] Finished. Data saved to {self.cfg.excel_path}")
        

    def close(self):
        self.vec_env.close()
        sys.stdout.flush() # 强制将缓冲区写入磁盘


def run_oos_test(model_path, test_pairs, cfg):
    """
    样本外测试 (Out-of-Sample Test)
    """
    # 1. 初始化测试环境（注意 is_test=True）
    test_cfg = cfg.__dict__.copy()
    test_cfg['is_test'] = True
    
    # 2. 加载 Learner 并加载权重
    learner = LearnerPPO(...) # 按参数初始化
    learner.load_checkpoint(model_path)
    learner.actor.eval() # 开启评估模式
    
    all_equities = []
    
    for pair in test_pairs:
        env = DynamicWindowEnv([pair], test_cfg)
        curr, hist, _ = env.reset()
        done = False
        equities = [cfg.init_capital]
        
        while not done:
            # 转 Tensor
            c_t = torch.from_numpy(curr).float().unsqueeze(0).to(learner.device)
            h_t = torch.from_numpy(hist).float().unsqueeze(0).to(learner.device)
            
            # 确定性推理
            with torch.no_grad():
                _, s = learner._build_state(c_t, h_t, norm_update=False)
                # 调用 deterministic=True 的预测
                a, wi, wv, _, _ = learner.sample_action_weight(s, deterministic=True)
            
            # 环境步进
            curr, hist, r, term, trunc = env.step(a, wv)
            equities.append(env.account_controller.equity)
            done = term or trunc
            
        all_equities.append(equities)
        print(f"Task {pair['call']} Finished. Final Equity: {equities[-1]:.2f}")
    
    # 绘制净值图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for eq in all_equities:
        plt.plot(eq)
    plt.title("Out-of-Sample Performance")
    plt.show()

# =========================
# Entry
# =========================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # 1. 加载原始数据
    dtype = {
        'call': str, 'put': str,
        'call_strike': float, 'put_strike': float,
        'call_open': str, 'call_expire': str,
    }
    df = pd.read_excel('./miniQMT/datasets/all_label_data/20251213_train.xlsx', dtype=dtype)

    # 排除名单
    exclude_list = [
        '10007347', '10007466', '10007467', '10006436', '10007346', '10006435', '10007465', 
        '10007726', '10007725', '10007724', '10008052', '10007723', '10006434', '10007722', 
        '10008051', '10007345', '10007721', '10007464', '10007344', '10007988', '10006433', 
        '10006820', '10007720', '10007987', '10006746', '10006745', '10007463', '10006432', '10007719'
    ]
    
    all_pairs = []

    for index, row in df.iterrows():
        start = row['call_open']
        end = row['call_expire']

        start_time = datetime.strptime(start, "%Y%m%d")
        end_time = datetime.strptime(end, "%Y%m%d")
        days = (end_time - start_time).days

        if days <= 40:
            continue

        call = row['call']
        put = row['put']

        if call in exclude_list or put in exclude_list:
            continue
        end_time = start_time + timedelta(days=20)
        end_time = end_time.strftime('%Y%m%d')

        start_time = start + '100000'
        end_time = end_time + '150000'
        all_pairs.append({
            'call': call,
            'put': put,
            'start_time': start_time,
            'end_time': end_time
        })


    # 2. 动态分类采样逻辑
    all_pairs = []
    # 分类桶，目标总数约 20 个
    buckets = {"ITM": [], "ATM": [], "OTM": []}
    target_per_bucket = 20

    # 预加载一个账户用于获取初始标的价格
    temp_account = single_Account(100000, stockList=['510050'])

    # 打乱原始数据顺序，保证采样的随机性
    # df = df.sample(frac=1).reset_index(drop=True)

    for index, row in df.iterrows():
        call, put = row['call'], row['put']
        if call in exclude_list or put in exclude_list:
            continue

        # 时间逻辑
        start_str, expire_str = row['call_open'], row['call_expire']
        start_dt = datetime.strptime(start_str, "%Y%m%d")
        expire_dt = datetime.strptime(expire_str, "%Y%m%d")
        
        if (expire_dt - start_dt).days <= 40:
            continue

        # 计算初始 Moneyness (行权价 / 标的价格)
        # 假设 start_time 为开盘 10:00:00
        start_time_full = start_str + '100000'
        try:
            spot_price = temp_account.getOpenPrice('510050', start_time_full)
            if spot_price <= 0: continue
            
            # 以 Call 的行权价计算
            moneyness = row['call_strike'] / spot_price
            
            # 简单分类：0.98-1.02 为平值，大于 1.05 为虚值，小于 0.95 为实值
            if 0.97 <= moneyness <= 1.03:
                cat = "ATM"
            elif moneyness > 1.03:
                cat = "OTM"
            else:
                cat = "ITM"
            
            # 填桶
            if len(buckets[cat]) < target_per_bucket:
                end_time_full = (expire_dt - timedelta(days=20)).strftime('%Y%m%d') + '150000'
                buckets[cat].append({
                    'call': call, 'put': put,
                    'start_time': start_time_full,
                    'end_time': end_time_full,
                    'steps': int(row['steps']),
                    'init_moneyness': moneyness
                })
        except:
            continue

        # if sum(len(v) for v in buckets.values()) >= 21:
        #     break

    # 合并采样结果
    for cat_list in buckets.values():
        all_pairs.extend(cat_list)
    
    total_steps = sum(p['steps'] for p in all_pairs)
    print(f"[Info] 采样完成。ITM:{len(buckets['ITM'])}, ATM:{len(buckets['ATM'])}, OTM:{len(buckets['OTM'])}")
    print(f"[Info] 总组合数: {len(all_pairs)}, 总步数: {total_steps}")


    # 3. 配置 Agent
    cfg = AgentConfig(
        option_pairs=all_pairs,
        window_size=32,
        pre_len=4,
        epochs=300,
        rollout_T=2048 * 12,     # 每次更新采样的基础长度
        num_workers=16,      # 维持 12 线程
        save_excel=True,
        # --- 核心参数调整 ---
        hidden_dim=256,      # 提升网络宽度以适应复杂逻辑
        adapter_dim=128,     # 提升特征投影维度
        mini_batch=2048 * 12, # 保持大 Batch 稳定梯度
        actor_lr=2e-4,
        critic_lr=5e-4,
        check_path='./miniQMT/DL/checkout/check_data_parallel.pt',
        k_epochs=10
    )

    agent = Agent(cfg)

    try:
        # 如果你修改了 hidden_dim，建议从 False 开始，因为旧权重形状不匹配
        agent.train_dynamic(from_check_point=False) 
    finally:
        agent.close()
