import torch
# 奖励归一化
class RunningMeanStd:
    """动态计算mean 和 std"""
    def __init__(self, shape, dtype=torch.float32):
        # 总样本数
        self.n = 0
        self.mean = torch.zeros(shape, dtype=dtype)
        self.S = torch.zeros(shape, dtype=dtype)
        self.std = torch.sqrt(self.S)
    
    def update(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # K是期权组合数, L是特征数
        K, L = x.shape

        self.n += K

        if self.n == K:
            self.mean = x.mean(dim=0)
            self.var = x.var(dim=0, unbiased=False)
            self.std = torch.sqrt(self.var.clamp_min_(1e-8))
        else:
            old_mean = self.mean.clone()
            batch_mean = x.mean(dim=0)

            # 更新mean
            self.mean = (self.n - K) / self.n * old_mean +  K / self.n * batch_mean

            batch_var = x.var(dim=0, unbiased=False)
            cross_var = K * (self.n - K) / self.n * (batch_mean - old_mean) ** 2

            self.var = (self.n - K) / self.n * self.var + K / self.n * batch_var + cross_var / self.n
            self.std = torch.sqrt(self.var.clamp_min_(1e-8))



# 对state进行标准化
class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x: torch.Tensor, update=True):
        x = x.detach()
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:  
            self.running_ms.update(x)
        new_x = (x - self.running_ms.mean) / (self.running_ms.std)
        return new_x

class RewardNormalization:
    """奖励归一化"""
    def __init__(self, shape=1):
        self.running_ms = RunningMeanStd(shape=shape)
    
    def __call__(self, x: torch.Tensor, update=True):
        x = x.detach()
        if update:
            self.running_ms.update(x)
        
        normalized_x = (x - self.running_ms.mean) / (self.running_ms.std)
        return normalized_x
    
    def get_states(self):
        return self.running_ms.mean, self.running_ms.std

class RewardScaling:
    def __init__(self, shape=1, gamma=0.99):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = torch.zeros(shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std

        return x[0]

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = torch.zeros(self.shape)
