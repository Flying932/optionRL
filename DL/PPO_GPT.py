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
from finTool.single_window_account import single_Account  # 用于 DataCache 读取数据
import os
import traceback

import warnings
# 忽略所有 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


# =========================
# Constants / actions
# =========================
A_HOLD, A_LONG, A_SHORT, A_CLOSE = 0, 1, 2, 3
WEIGHT_BINS = np.array([0.00, 0.25, 0.50, 0.75, 1.00], dtype=np.float32)




# =========================
# Normalization (n==0 -> return x)
# =========================
class RunningMeanStd:
    def __init__(self, shape, dtype=torch.float32, eps=1e-8, device="cpu"):
        self.eps = float(eps)
        self.n = 0  # keep int
        self.mean = torch.zeros(shape, dtype=dtype, device=device)
        self.var = torch.ones(shape, dtype=dtype, device=device)
        self.std = torch.sqrt(self.var).clamp_min(self.eps)

    def update(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = int(x.shape[0])
        if B <= 0:
            return

        n_old = int(self.n)
        n_new = n_old + B

        if n_old == 0:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.mean = mean
            self.var = var
            self.std = torch.sqrt(self.var.clamp_min(self.eps))
            self.n = n_new
            return

        old_mean = self.mean
        old_var = self.var
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        delta = batch_mean - old_mean
        mean = old_mean + delta * (B / n_new)

        m2_old = old_var * n_old
        m2_batch = batch_var * B
        m2 = m2_old + m2_batch + (delta ** 2) * (n_old * B / n_new)
        var = m2 / n_new

        self.mean = mean
        self.var = var
        self.std = torch.sqrt(self.var.clamp_min(self.eps))
        self.n = n_new


class Normalization:
    def __init__(self, shape, dtype=torch.float32, eps=1e-8, device="cpu"):
        self.running_ms = RunningMeanStd(shape=shape, dtype=dtype, eps=eps, device=device)
        self.eps = float(eps)

    def __call__(self, x: torch.Tensor, update=True):
        if update:
            self.running_ms.update(x.detach())
        if int(self.running_ms.n) == 0:
            return x
        return (x - self.running_ms.mean) / (self.running_ms.std + self.eps)

    def state_dict(self):
        return {
            "n": int(self.running_ms.n),
            "mean": self.running_ms.mean.detach().cpu(),
            "var": self.running_ms.var.detach().cpu(),
            "std": self.running_ms.std.detach().cpu(),
            "eps": self.eps,
        }

    def load_state_dict(self, d: Dict[str, Any], device="cpu"):
        self.eps = float(d.get("eps", self.eps))
        self.running_ms.n = int(d["n"])
        self.running_ms.mean = d["mean"].to(device)
        self.running_ms.var = d["var"].to(device)
        self.running_ms.std = d.get("std", torch.sqrt(self.running_ms.var)).to(device)


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
        raw = torch.cat([curr.float(), reduce_call.float(), reduce_put.float()], dim=-1)
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
            timesteps=self.cfg["max_timesteps"],
            # 如果你的 windowEnv 需要额外参数，在这里补齐
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

    def sample_action_weight(state_1d: torch.Tensor) -> Tuple[int, int, float, float, float]:
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

            elif cmd == "set_weights":
                payload = data
                if actor is None or critic is None or feat.adapter is None or feat.norm is None:
                    pending_payload = payload
                else:
                    apply_payload(payload)
                remote.send(("ok", None))

            elif cmd == "rollout":
                T = int(data["T"])

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

                raw_curr = np.zeros((T, Dc), np.float32)
                raw_hist = np.zeros((T, L, Dh), np.float32)
                actions = np.zeros((T,), np.int64)
                w_idx = np.zeros((T,), np.int64)
                w_val = np.zeros((T,), np.float32)
                logp_old = np.zeros((T,), np.float32)
                value_old = np.zeros((T,), np.float32)
                rewards = np.zeros((T,), np.float32)
                done = np.ones((T,), np.bool_)
                valid = np.zeros((T,), np.bool_)

                terminated_early = False

                for t in range(T):
                    raw_curr[t] = curr_np
                    raw_hist[t] = hist_np

                    if terminated_early:
                        actions[t] = A_HOLD
                        w_idx[t] = 0
                        w_val[t] = 0.0
                        logp_old[t] = 0.0
                        value_old[t] = 0.0
                        rewards[t] = 0.0
                        done[t] = True
                        valid[t] = False
                        continue

                    with torch.no_grad():
                        curr = torch.from_numpy(curr_np).unsqueeze(0)
                        hist = torch.from_numpy(hist_np).unsqueeze(0)
                        s, _ = feat.obs_to_state_normed(curr, hist, update_norm=False)
                        a, wi, wv, lp, v = sample_action_weight(s)

                    step_out = env.step(a, wv)
                    if isinstance(step_out, (tuple, list)) and len(step_out) >= 5:
                        next_curr, next_hist, r, term, trunc = step_out[0], step_out[1], step_out[2], step_out[3], step_out[4]
                    else:
                        raise RuntimeError(f"env.step return length={len(step_out)} unexpected")

                    d = bool(term or trunc)

                    actions[t] = a
                    w_idx[t] = wi
                    w_val[t] = wv
                    logp_old[t] = lp
                    value_old[t] = v
                    rewards[t] = float(r)
                    done[t] = d
                    valid[t] = True

                    if d:
                        terminated_early = True
                        continue

                    curr_np = np.asarray(next_curr, np.float32)
                    hist_np = np.asarray(next_hist, np.float32)

                # bootstrap last_value
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

    def set_weights_all(self, payload: Dict[str, Any]):
        for r in self.remotes:
            r.send(("set_weights", payload))
        for r in self.remotes:
            tag, payload2 = r.recv()
            if tag == "error":
                raise RuntimeError(payload2)

    def rollout(self, T: int) -> List[Dict[str, Any]]:
        for r in self.remotes:
            r.send(("rollout", {"T": int(T)}))

        trajs = []
        for r in self.remotes:
            tag, payload = r.recv()
            if tag == "error":
                raise RuntimeError(payload)
            if tag != "traj":
                raise RuntimeError(f"unexpected tag from worker: {tag}")
            trajs.append(payload)
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
        check_path: str='./miniQMT/DL/checkout',
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.adapter_dim = adapter_dim
        self.hidden_dim = hidden_dim

        self.check_path = f'{check_path}/check_data_parallel.pt'

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

    def save(self, epoch: int = None, best_reward: float = None, path: str = None):
            save_path = path or self.check_path
            data = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "actor_state": self.actor.state_dict(),
                "value_state": self.critic.state_dict(),
                "adpter_state": self.adapter.state_dict(),
                "opt_actor_state": self.opt_actor.state_dict(),
                "opt_critic_state": self.opt_critic.state_dict(),
                "opt_adpter_state": self.opt_adapter.state_dict(),
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
        data = torch.load(path, map_location=self.device)
        
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
            print(f"[Resume] State Norm loaded. (count={self.state_norm.running_ms.n if hasattr(self.state_norm.running_ms, 'n') else '?'})")
        else:
            print("[Resume] Warning: No state_norm in checkpoint! Training might be unstable.")

        # 4. [新增] 加载 Reward Norm List (可选)
        if 'reward_norm_list' in data:
            self.reward_norm_list = data['reward_norm_list']

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
        raw = torch.cat([curr.float(), reduce_call.float(), reduce_put.float()], dim=-1)
        state_dim = int(raw.shape[-1])

        if self.norm is None:
            self.norm = Normalization(shape=(state_dim,), device=self.device)

        if self.actor is None:
            self.actor = ActorDualHead(state_dim, hidden_dim=self.hidden_dim).to(self.device)
            self.critic = ValueNet(state_dim, hidden_dim=self.hidden_dim).to(self.device)
            self.opt_adapter = torch.optim.Adam(self.adapter.parameters(), lr=self.actor_lr)
            self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

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

        call_state, put_state = torch.chunk(hist, chunks=2, dim=2)
        call_tok, put_tok = self._encode_tokens_only(call_state, put_state)

        reduce_call = self.adapter(call_tok)
        reduce_put = self.adapter(put_tok)
        raw = torch.cat([curr.float(), reduce_call.float(), reduce_put.float()], dim=-1)
        s = self.norm(raw, update=norm_update)
        return raw, s

    def update_from_trajs(self, trajs: List[Dict[str, Any]], target_kl: float = 0.015):
        """
        returns: loss, kl, actor_loss, value_loss, entropy
        """
        assert self.actor is not None and self.critic is not None and self.adapter is not None
        assert self.opt_actor is not None and self.opt_critic is not None and self.opt_adapter is not None

        T = int(trajs[0]["raw_curr"].shape[0])
        N = len(trajs)

        raw_curr = np.stack([tr["raw_curr"] for tr in trajs], axis=1)
        raw_hist = np.stack([tr["raw_hist"] for tr in trajs], axis=1)
        actions = np.stack([tr["actions"] for tr in trajs], axis=1)
        w_idx = np.stack([tr["w_idx"] for tr in trajs], axis=1)
        logp_old = np.stack([tr["logp_old"] for tr in trajs], axis=1)
        value_old = np.stack([tr["value_old"] for tr in trajs], axis=1)
        rewards = np.stack([tr["rewards"] for tr in trajs], axis=1)
        done = np.stack([tr["done"] for tr in trajs], axis=1)
        valid = np.stack([tr["valid"] for tr in trajs], axis=1)
        last_value = np.stack([tr["last_value"] for tr in trajs], axis=1).squeeze(0)

        curr_t = torch.from_numpy(raw_curr).to(self.device)
        hist_t = torch.from_numpy(raw_hist).to(self.device)
        act_t = torch.from_numpy(actions).to(self.device).long()
        widx_t = torch.from_numpy(w_idx).to(self.device).long()
        logp_old_t = torch.from_numpy(logp_old).to(self.device).float()
        v_old_t = torch.from_numpy(value_old).to(self.device).float()
        rew_t = torch.from_numpy(rewards).to(self.device).float()
        done_t = torch.from_numpy(done.astype(np.float32)).to(self.device).float()
        valid_t = torch.from_numpy(valid.astype(np.float32)).to(self.device).float()
        last_v = torch.from_numpy(last_value).to(self.device).float()

        # ---- GAE (padding-aware) ----
        with torch.no_grad():
            adv = torch.zeros((T, N), device=self.device)
            last_gae = torch.zeros((N,), device=self.device)

            for t in reversed(range(T)):
                m = (1.0 - done_t[t]) * valid_t[t]
                v_tp1 = last_v if t == T - 1 else v_old_t[t + 1]
                delta = rew_t[t] + self.gamma * v_tp1 * m - v_old_t[t]
                last_gae = delta + self.gamma * self.gae_lambda * m * last_gae
                adv[t] = last_gae * valid_t[t]

            ret = adv + v_old_t

            mask = (valid_t.view(-1) > 0.5)
            adv_f = adv.view(-1)[mask]
            ret_f = ret.view(-1)[mask]
            act_f = act_t.view(-1)[mask]
            widx_f = widx_t.view(-1)[mask]
            logp_old_f = logp_old_t.view(-1)[mask]

            curr_f = curr_t.view(T * N, -1)[mask]
            hist_f = hist_t.view(T * N, hist_t.shape[2], hist_t.shape[3])[mask]

            adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        last_loss = 0.0
        last_kl = 0.0
        last_actor_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0

        for _ in range(self.k_epochs):
            raw_state, s = self._build_state(curr_f, hist_f, norm_update=False)

            logits_a, logits_w = self.actor(s)
            dist_a = Categorical(logits=logits_a)
            new_logp_a = dist_a.log_prob(act_f)
            ent_a = dist_a.entropy().mean()

            need_w = ((act_f == A_LONG) | (act_f == A_SHORT) | (act_f == A_CLOSE)).float()

            lw = logits_w.clone()
            maskw = torch.zeros_like(lw, dtype=torch.bool)
            maskw[need_w.bool(), 1:] = True
            maskw[~need_w.bool(), 0] = True
            lw[~maskw] = -1e9

            dist_w = Categorical(logits=lw)
            new_logp_w = dist_w.log_prob(widx_f)
            ent_w = (need_w * dist_w.entropy()).sum() / (need_w.sum() + 1e-6)

            logp_new = new_logp_a + need_w * new_logp_w
            ratio = torch.exp(logp_new - logp_old_f)

            surr1 = ratio * adv_f
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_f
            actor_loss = -torch.min(surr1, surr2).mean()

            v_pred = self.critic(s).squeeze(-1)
            value_loss = F.mse_loss(v_pred, ret_f)

            entropy = ent_a + 0.5 * ent_w
            loss = actor_loss + 0.5 * value_loss - 0.01 * entropy

            self.opt_adapter.zero_grad()
            self.opt_actor.zero_grad()
            self.opt_critic.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.opt_adapter.step()
            self.opt_actor.step()
            self.opt_critic.step()

            with torch.no_grad():
                kl = (logp_old_f - logp_new).mean().abs()
                last_loss = float(loss.item())
                last_kl = float(kl.item())
                last_actor_loss = float(actor_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy = float(entropy.item())
                if kl > 1.5 * target_kl:
                    break

        # update norm AFTER update
        with torch.no_grad():
            _raw2, _ = self._build_state(curr_f, hist_f, norm_update=True)

        return last_loss, last_kl, last_actor_loss, last_value_loss, last_entropy



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
    adapter_dim: int = 128
    hidden_dim: int = 256
    pretrained_path: str = "./miniQMT/DL/preTrain/weights/preMOE_best_dummy_data_32_4.pth"

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    k_epochs: int = 5
    epochs: int = 50

    # logging
    save_excel: bool = False
    excel_path: str = "./miniQMT/DL/results/PPO_training_data.xlsx"


class Agent:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
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
        )

        # build vector env
        workers = min(cfg.num_workers, len(self.pairs))
        workers = max(1, workers)


        env_fns = []
        for i in range(workers):
            dic = self.pairs[i]
            env_cfg = {
                "start_time": dic['start_time'],
                "end_time": dic['end_time'],
                "benchmark": cfg.benchmark,
                "fee": cfg.fee,
                "init_capital": cfg.init_capital,
                "max_timesteps": dic['steps'] + 1,
            }

            def make_env(seed=i, pairs=self.pairs, c=env_cfg):
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
        }

    
    def train(self, from_check_point: bool=False):
        """
        记录字段必须包含：
        epoch, reward, avg_equity, loss, kl,
        hold_ratio, long_ratio, short_ratio, close_ratio,
        actor_loss, value_loss, entropy,
        ratio_0, ratio_25, ratio_50, ratio_75, ratio_100
        """

        
        total_pairs = len(self.pairs)
        print(f'[Agent-init-train] 期权组合数量 = {total_pairs}')
        print(f'Start to train PPO on {self.device}')

        workers = self.workers

        best_reward = -float('inf')
        patience = getattr(self.cfg, 'patience', 30)
        stop_entropy = getattr(self.cfg, 'stop_entropy', 0.6)
        min_delta = 0.001
        early_stop_counter = 0


        # 若你还没初始化 records，这里确保存在
        if not hasattr(self, "records") or self.records is None:
            self.records = {
                "epoch": [], "reward": [], "avg_equity": [], "loss": [], "kl": [],
                "hold_ratio": [], "long_ratio": [], "short_ratio": [], "close_ratio": [],
                "actor_loss": [], "value_loss": [], "entropy": [],
                "ratio_0": [], "ratio_25": [], "ratio_50": [], "ratio_75": [], "ratio_100": [],
            }

        if from_check_point:
            start_epoch, best_reward = self.learner.load_checkpoint()


        for ep in range(self.cfg.epochs):
            if from_check_point and start_epoch is not None:
                if ep < start_epoch:
                    print(f'[Skip] epoch = {ep}')
                    continue

            t0 = time.time()
            order = list(range(total_pairs))
            random.shuffle(order)

            # epoch 累积统计
            total_reward_sum = 0.0
            total_valid_steps = 0

            action_counts = np.zeros(4, dtype=np.int64)
            weight_counts = np.zeros(5, dtype=np.int64)

            equity_sum = 0.0
            equity_cnt = 0

            losses, kls, actor_losses, value_losses, entropies = [], [], [], [], []

            for i in range(0, total_pairs, workers):
                task_ids_true = order[i:i + workers]
                true_cnt = len(task_ids_true)

                # padding task_ids 以满足 worker 数量
                task_ids = task_ids_true.copy()
                while len(task_ids) < workers:
                    task_ids.append(task_ids[-1])

                # 1) set tasks
                self.vec_env.set_tasks(task_ids)

                # 2) broadcast weights + norm snapshot
                payload = self.learner.export_payload()
                self.vec_env.set_weights_all(payload)

                # 3) one-shot rollout
                trajs = self.vec_env.rollout(self.cfg.rollout_T)

                # 4) PPO update
                loss, kl, a_loss, v_loss, ent = self.learner.update_from_trajs(trajs)
                losses.append(loss)
                kls.append(kl)
                actor_losses.append(a_loss)
                value_losses.append(v_loss)
                entropies.append(ent)

                # ---- 统计 reward / action / weight / equity（只统计真实组合 true_cnt，忽略 padding 重复的 worker）----
                for wi in range(true_cnt):
                    tr = trajs[wi]
                    valid = tr["valid"].astype(bool)
                    r = tr["rewards"].astype(np.float32)

                    total_reward_sum += float(r[valid].sum())
                    total_valid_steps += int(valid.sum())

                    acts = tr["actions"].astype(np.int64)[valid]
                    if acts.size > 0:
                        action_counts += np.bincount(acts, minlength=4)

                    widx = tr["w_idx"].astype(np.int64)[valid]
                    if widx.size > 0:
                        weight_counts += np.bincount(widx, minlength=5)

                    # equity_end
                    if "equity_end" in tr:
                        eq = float(np.asarray(tr["equity_end"]).reshape(-1)[0])
                        if not np.isnan(eq):
                            equity_sum += eq
                            equity_cnt += 1

            # epoch 结果
            avg_reward = total_reward_sum / (total_valid_steps + 1e-8)
            avg_equity = equity_sum / max(1, equity_cnt)

            act_total = action_counts.sum()
            if act_total == 0:
                hold_ratio = long_ratio = short_ratio = close_ratio = 0.0
            else:
                hold_ratio = float(action_counts[0] / act_total)
                long_ratio = float(action_counts[1] / act_total)
                short_ratio = float(action_counts[2] / act_total)
                close_ratio = float(action_counts[3] / act_total)

            w_total = weight_counts.sum()
            if w_total == 0:
                w_ratios = [0.0] * 5
            else:
                w_ratios = [float(weight_counts[k] / w_total) for k in range(5)]

            # 写 records
            self.records["epoch"].append(ep + 1)
            self.records["reward"].append(avg_reward)
            self.records["avg_equity"].append(avg_equity)

            self.records["loss"].append(float(np.mean(losses)) if losses else 0.0)
            self.records["kl"].append(float(np.mean(kls)) if kls else 0.0)

            self.records["hold_ratio"].append(hold_ratio)
            self.records["long_ratio"].append(long_ratio)
            self.records["short_ratio"].append(short_ratio)
            self.records["close_ratio"].append(close_ratio)

            self.records["actor_loss"].append(float(np.mean(actor_losses)) if actor_losses else 0.0)
            self.records["value_loss"].append(float(np.mean(value_losses)) if value_losses else 0.0)
            self.records["entropy"].append(float(np.mean(entropies)) if entropies else 0.0)

            entropy = self.records["entropy"][-1]

            self.records["ratio_0"].append(w_ratios[0])
            self.records["ratio_25"].append(w_ratios[1])
            self.records["ratio_50"].append(w_ratios[2])
            self.records["ratio_75"].append(w_ratios[3])
            self.records["ratio_100"].append(w_ratios[4])

            dt = time.time() - t0
            print(
                f"[Epoch {ep+1} / {self.cfg.epochs}] "
                f"Reward:{avg_reward:.6f} | Market_value: {avg_equity:.2f} |"
                f"loss={self.records['loss'][-1]:.4f} kl={self.records['kl'][-1]:.4f} |"
                f"act(H/L/S/C)={hold_ratio:.2f}/{long_ratio:.2f}/{short_ratio:.2f}/{close_ratio:.2f} |"
                f"entropy={self.records['entropy'][-1]:.3f} time={dt:.1f}s"
            )

            # 保存 excel（如果你原先有开关/路径，按你原逻辑即可；这里给一个最常用写法）
            if getattr(self.cfg, "save_excel", False):
                pd.DataFrame(self.records).to_excel(self.cfg.excel_path, index=False)
            # --- 早停判断 ---
            if avg_reward > best_reward + min_delta:
                best_reward = avg_reward
                early_stop_counter = 0 
                self.learner.save(ep, best_reward) 
                print(f"   >>> 🌟 Best Reward Updated: {best_reward:.4f} (Counter Reset)")
            else:
                early_stop_counter += 1
                print(f"   ⏳ [Patience] No improvement: {early_stop_counter}/{patience} | Best: {best_reward:.4f}")

            if early_stop_counter >= patience:
                print(f"\n🛑 [Early Stop] Triggered! Reward has not improved for {patience} epochs.")
                print(f"   Final Best Reward: {best_reward:.4f}")
                break
            
            if entropy < stop_entropy and avg_reward > 0:
                print(f"\n🛑 [Early Stop] Triggered! Entropy ({entropy:.4f}) is too low.")
                self.learner.save(ep, best_reward)
                break

        print(f"[Train] Finished. Data saved to {self.cfg.excel_path}")



    def train_dynamic(self, from_check_point: bool = False):
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
        print(f'[Agent-init-train] 期权组合数量 = {total_pairs}')
        print(f'Start to train PPO on {self.device}')

        workers = self.workers

        # 大的采样目标：默认 8192（优先用 cfg.rollout_T_big；否则退化用 cfg.rollout_T；否则 8192）
        rollout_T_big = int(getattr(self.cfg, "rollout_T_big", getattr(self.cfg, "rollout_T", 8192)))
        if rollout_T_big <= 0:
            rollout_T_big = 8192
        
        print(f"[Info] 采样目标, rollout_T_big = {rollout_T_big}")

        # 早停相关
        best_reward = -float('inf')
        patience = getattr(self.cfg, 'patience', 30)
        stop_entropy = getattr(self.cfg, 'stop_entropy', 0.6)
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
        for ep in range(self.cfg.epochs):
            if from_check_point and start_epoch is not None and ep < start_epoch:
                print(f'[Skip] epoch = {ep}')
                continue

            t0 = time.time()

            # 本轮累计的组 steps（按你要求用“组内 steps 总和”来判断是否够 8192）
            sampled_steps_sum = 0

            # 收集到的 traj（注意：每次抽组会返回 workers 条，但最后一组可能 < workers，我们会丢弃 padding 部分）
            collected_trajs = []

            # 统计（按 valid 统计）
            total_reward_sum = 0.0
            total_valid_steps = 0
            action_counts = np.zeros(4, dtype=np.int64)
            weight_counts = np.zeros(5, dtype=np.int64)
            equity_sum = 0.0
            equity_cnt = 0

            # 反复抽组直到累计组 steps >= rollout_T_big
            while sampled_steps_sum < rollout_T_big:
                g_idx = sample_group_index()
                group = groups[g_idx]
                true_cnt = len(group)

                # 组贡献的“长度”（采样概率/累计步数使用 sum steps）
                sampled_steps_sum += int(group_sum_steps[g_idx])

                # rollout_len 用 max steps，确保组内每个 pair 都能跑到自己 done（完整周期）
                rollout_len = int(group_max_steps[g_idx])


                # padding tasks：填满 workers（避免 vec_env 断言）
                task_ids = group.copy()
                print(f"choose_task_ids = {task_ids}, sum_roll_len = {sampled_steps_sum}")
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

                # 统计（只统计 valid 的部分）
                for tr in trajs:
                    valid = np.asarray(tr["valid"], dtype=np.bool_)
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

            # -------------------------
            # 4) PPO update：把所有 collected_trajs pad 到 Tmax 后一次更新
            # -------------------------
            Tmax = max(int(tr["raw_curr"].shape[0]) for tr in collected_trajs)
            trajs_for_update = [pad_traj_to_T(tr, Tmax) for tr in collected_trajs]

            loss, kl, a_loss, v_loss, ent = self.learner.update_from_trajs(trajs_for_update)

            # -------------------------
            # 5) 写 records + 打印 + 早停
            # -------------------------
            avg_reward = total_reward_sum / (total_valid_steps + 1e-8)
            avg_equity = equity_sum / max(1, equity_cnt)

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

            dt = time.time() - t0
            print(
                f"[Epoch {ep+1} / {self.cfg.epochs}] "
                f"sampled_group_steps={sampled_steps_sum} target={rollout_T_big} | "
                f"valid_steps={total_valid_steps} | "
                f"Reward:{avg_reward:.6f} | Market_value:{avg_equity:.2f} | "
                f"loss={loss:.4f} kl={kl:.4f} | "
                f"act(H/L/S/C)={hold_ratio:.2f}/{long_ratio:.2f}/{short_ratio:.2f}/{close_ratio:.2f} | "
                f"entropy={ent:.3f} time={dt:.1f}s"
            )

            # 保存 Excel
            if getattr(self.cfg, "save_excel", False):
                pd.DataFrame(self.records).to_excel(self.cfg.excel_path, index=False)

            # --- 早停判断 ---
            if avg_reward > best_reward + min_delta:
                best_reward = avg_reward
                early_stop_counter = 0
                self.learner.save(ep, best_reward)
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

        print(f"[Train] Finished. Data saved to {self.cfg.excel_path}")

    def close(self):
        self.vec_env.close()


# =========================
# Entry
# =========================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)


    # 构造海量期权组合 (示例)
    all_pairs = []
    dtype = {
        'call': str,
        'put': str,
        'call_strike': int,
        'put_strike': int,
        'call_open': str,
        'put_open': str,
        'call_expire': str,
        'put_expire': str,
    }
    df = pd.read_excel('./miniQMT/datasets/all_label_data/20251213_train.xlsx', dtype=dtype)


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
        end_time = start_time + timedelta(days=20)
        end_time = end_time.strftime('%Y%m%d')

        start_time = start + '100000'
        end_time = end_time + '150000'
        steps = row['steps']
        all_pairs.append({
            'call': call,
            'put': put,
            'start_time': start_time,
            'end_time': end_time,
            'steps': steps
        })

    option_pairs = all_pairs
    cfg = AgentConfig(
        option_pairs=option_pairs,
        pretrained_path="./miniQMT/DL/preTrain/weights/preMOE_best_dummy_data_32_4.pth",
        epochs=500,
        rollout_T=2048,
        num_workers=10,
        save_excel=True,
    )

    agent = Agent(cfg)
    try:
        agent.train_dynamic(from_check_point=False)
    finally:
        agent.close()
