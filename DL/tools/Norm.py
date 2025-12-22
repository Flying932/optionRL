import torch
import torch

class RunningMeanStd:
    """动态计算 mean 和 std，支持指定 device"""
    def __init__(self, shape, device=torch.device("cpu"), dtype=torch.float32):
        self.device = device
        self.n = 0
        self.mean = torch.zeros(shape, device=device, dtype=dtype)
        self.var = torch.ones(shape, device=device, dtype=dtype)  # 初始化为1，避免初次除零
        self.std = torch.sqrt(self.var)
    
    def update(self, x: torch.Tensor):
        # 确保输入在指定的设备上
        x = x.to(self.device)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        K, L = x.shape
        self.n += K

        if self.n == K:
            self.mean = x.mean(dim=0)
            self.var = x.var(dim=0, unbiased=False)
            self.std = torch.sqrt(self.var.clamp_min_(1e-8))
        else:
            old_mean = self.mean.clone()
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # 更新 mean
            self.mean = (self.n - K) / self.n * old_mean + K / self.n * batch_mean

            # 更新 variance (使用并入公式)
            cross_var = K * (self.n - K) / self.n * (batch_mean - old_mean) ** 2
            self.var = (self.n - K) / self.n * self.var + K / self.n * batch_var + cross_var / self.n
            self.std = torch.sqrt(self.var.clamp_min_(1e-8))

    # 手动实现 state_dict
    def state_dict(self):
        return {
            "n": self.n,
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu()
        }

    # 手动实现 load_state_dict
    def load_state_dict(self, state_dict, device: str=None):
        if device is None:
            device = self.device

        self.n = state_dict["n"]
        self.mean = state_dict["mean"].to(device)
        self.var = state_dict["var"].to(device)
        self.std = torch.sqrt(self.var.clamp_min_(1e-8)).to(device)
        
class Normalization:
    def __init__(self, shape, device=torch.device("cpu")):
        self.device = device
        self.running_ms = RunningMeanStd(shape=shape, device=device)

    def __call__(self, x: torch.Tensor, update=True):
        x = x.to(self.device).detach()
        if update:  
            self.running_ms.update(x)

        if self.running_ms.n == 0:
            return x

        return (x - self.running_ms.mean) / (self.running_ms.std)

    def state_dict(self):
        return self.running_ms.state_dict()

    def load_state_dict(self, state_dict, device: str=None):
        if device is None:
            device = self.device
        self.running_ms.load_state_dict(state_dict, device)

class RewardNormalization:
    def __init__(self, shape=1, device=torch.device("cpu")):
        self.device = device
        self.running_ms = RunningMeanStd(shape=shape, device=device)
    
    def __call__(self, x: torch.Tensor, update=True):
        x = x.to(self.device).detach()
        if update:
            self.running_ms.update(x)
        
        return (x - self.running_ms.mean) / (self.running_ms.std)

class RewardScaling:
    def __init__(self, shape=1, gamma=0.99, device=torch.device("cpu")):
        self.device = device
        self.shape = shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=self.shape, device=device)
        self.R = torch.zeros(shape, device=device)

    def __call__(self, x: torch.Tensor):
        x = x.to(self.device)
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        # 仅进行标准差缩放
        x = x / (self.running_ms.std + 1e-8)
        return x

    def reset(self):
        self.R = torch.zeros(self.shape, device=self.device)