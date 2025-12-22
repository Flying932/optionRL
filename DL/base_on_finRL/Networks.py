import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict

import sys
from pathlib import Path
def setup_miniqmt_import_root():
    """
    递归查找 'miniQMT' 文件夹，并将其添加到 sys.path 中，
    从而允许使用 miniQMT 为根的绝对导入。
    """
    # 1. 获取当前脚本的绝对路径
    # stack[0] 是当前正在执行的帧，其 f_globals['__file__'] 是脚本路径
    try:
        # 获取调用此函数的脚本的路径
        calling_script_path = Path(sys._getframe(1).f_globals['__file__']).resolve()
    except KeyError:
        # 如果在交互式环境或某些特殊环境中，可能无法获取文件路径，则退出
        print("⚠️ 警告: 无法确定当前脚本路径，跳过路径设置。")
        return
    
    current_path = calling_script_path
    miniqmt_root = None
    
    # 2. 向上递归查找
    # current_path.parents 是一个包含所有父目录的序列
    for parent in [current_path] + list(current_path.parents):
        if parent.name == 'miniQMT':
            miniqmt_root = parent
            break
        
    # 3. 检查并添加路径
    if miniqmt_root:
        # 将找到的 miniQMT 目录添加到 sys.path
        miniqmt_root_str = str(miniqmt_root)
        if miniqmt_root_str not in sys.path:
            sys.path.insert(0, miniqmt_root_str)
            print(f"✅ 成功将项目根目录添加到搜索路径: {miniqmt_root_str}")
        else:
            # 已经添加过，无需重复添加
            # print(f"ℹ️ 项目根目录已在搜索路径中: {miniqmt_root_str}")
            pass
    else:
        print("❌ 错误: 未能在当前路径或其任何父目录中找到 'miniQMT' 文件夹。")
setup_miniqmt_import_root()
from DL.preTrain.preMOE import PreMOE 

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


class HybridFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, cfg):
        # 1. 动态计算输出维度
        # 我们需要先跑一次模型看看 adapter 输出多大，或者硬编码
        # 根据你之前的代码：Adapter 输出 final_dim=128
        # Call语义(128) + Put语义(128) + 账户特征(14) + Call物理(13) + Put物理(13)
        # 128 + 128 + 14 + 13 + 13 = 296
        total_dim = 296
        
        super(HybridFeatureExtractor, self).__init__(observation_space, features_dim=total_dim)
        
        # 2. 加载预训练 Transformer (PreMOE)
        self.pre_moe = PreMOE(
            seq_len=cfg.window_size, 
            pred_len=cfg.pre_len, 
            n_variates=cfg.n_variates, 
            d_router=cfg.d_router
        )
        
        # 加载权重
        print(f"[Network] Loading Pretrained MOE from {cfg.pretrained_path} ...")
        # map_location='cpu' 确保在任何机器上都能加载
        state_dict = torch.load(cfg.pretrained_path, map_location='cpu')
        self.pre_moe.load_state_dict(state_dict, strict=False) # strict=False 防止版本微小差异报错
        
        # 冻结参数
        self.pre_moe.eval()
        for param in self.pre_moe.parameters():
            param.requires_grad = False
            
        # 3. 初始化 MultiViewAdapter
        # 为了知道 dims_dict 到底填什么，我们构建一个假数据跑一次
        # 假设 batch=1, seq=32, feature=13 (Call)
        dummy_input = torch.zeros(1, cfg.window_size, cfg.n_variates)
        
        with torch.no_grad():
            # 获取 encode_tokens 的输出结构
            tokens = self.pre_moe.encode_tokens(dummy_input)
            
            # 自动提取维度，不需要手动填数字
            dims_dict = {
                "varma_high": int(tokens["varma_h"].shape[-1]),
                "varma_low": int(tokens["varma_l"].shape[-1]),
                "basis_high": int(tokens["basis_h"].shape[-1]),
                "basis_low": int(tokens["basis_l"].shape[-1]),
                "itrans_high": int(tokens["itrans_h"].shape[-1]),
                "itrans_low": int(tokens["itrans_l"].shape[-1]),
                "router": int(tokens["router"].shape[-1]),
            }
            print(f"[Network] Auto-detected adapter dims: {dims_dict}")

        # 实例化 Adapter (这是可训练的部分)
        # 这里使用 cfg.adapter_dim，通常是你设置的 128 或 256
        self.adapter = MultiViewAdapter(dims_dict=dims_dict, final_dim=128) 

    def forward(self, observations):
        # observations 是一个 Dict (来自 Gym space)
        hist = observations["hist"] # [Batch, 32, 26]
        curr = observations["curr"] # [Batch, 14]
        
        # 1. 拆分 Call 和 Put 的历史序列
        # 假设 hist 的前13列是 Call，后13列是 Put
        # dim=2 是特征维度 (Batch, Time, Feat)
        call_seq, put_seq = torch.chunk(hist, 2, dim=2) 
        
        # 2. 提取物理特征 (Residual Connection - 强直连)
        # 取最后一个时间步 (Time=-1)
        # phys_call: [Batch, 13]
        phys_call = call_seq[:, -1, :] 
        phys_put = put_seq[:, -1, :]
        
        # 3. Transformer 语义提取
        with torch.no_grad():
            call_tok = self.pre_moe.encode_tokens(call_seq)
            put_tok = self.pre_moe.encode_tokens(put_seq)
        
        # 4. 维度缩减 (Adapter)
        call_latent = self.adapter(call_tok) # [Batch, 128]
        put_latent = self.adapter(put_tok)   # [Batch, 128]
        
        # 5. 最终拼接
        # [账户特征(14), Call物理(13), Put物理(13), Call语义(128), Put语义(128)]
        # 总维度 = 296
        combined = torch.cat([
            curr, 
            phys_call, 
            phys_put, 
            call_latent, 
            put_latent
        ], dim=-1)
        
        return combined