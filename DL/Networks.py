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
        self.varma_proj = ViewProjector(dims_dict["varma_h"], dims_dict["varma_l"], out_dim=view_dim)
        self.basis_proj = ViewProjector(dims_dict["basis_h"], dims_dict["basis_l"], out_dim=view_dim)
        self.itrans_proj = ViewProjector(dims_dict["itrans_h"], dims_dict["itrans_l"], out_dim=view_dim)
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


class HybridOptionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, cfg):
        # 1. 提取基础维度 (不赋值给 self，仅作为局部变量计算)
        curr_dim = observation_space["curr"].shape[0]
        hist_total_dim = observation_space["hist"].shape[1]
        n_variates = hist_total_dim // 2
        adapter_final_dim = 128 

        self.device = cfg.device
        
        # 2. 临时创建一个 PreMOE 实例来推导维度
        # 注意：这里我们还没调用 super().__init__，所以不能赋给 self.pre_moe
        temp_moe = PreMOE(
            seq_len=cfg.window_size, 
            pred_len=cfg.pre_len, 
            n_variates=n_variates, 
            d_router=cfg.d_router
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, cfg.window_size, n_variates)
            toks = temp_moe.encode_tokens(dummy_input)
            dims_dict = {k: v.shape[-1] for k, v in toks.items()}
            
        # 3. 自动计算总特征维度
        # 账户(9) + 物理直连(26) + 语义(128*2) = 291
        total_features_dim = curr_dim + hist_total_dim + (adapter_final_dim * 2)
        
        # 4. 🔥 【核心修正】先调用父类初始化，再进行模块赋值
        super(HybridOptionExtractor, self).__init__(observation_space, features_dim=total_features_dim)
        
        # 5. 现在可以安全地赋值了
        self.curr_dim = curr_dim
        self.hist_total_dim = hist_total_dim
        self.pre_moe = temp_moe # 将刚才创建的实例挂载到 self
        
        # 加载权重并冻结
        self.pre_moe.load_state_dict(torch.load(cfg.pretrained_path, map_location=self.device), strict=False)
        self.pre_moe.eval()
        for p in self.pre_moe.parameters():
            p.requires_grad = False
            
        # 初始化 Adapter
        self.adapter = MultiViewAdapter(dims_dict=dims_dict, final_dim=adapter_final_dim).to(self.device)
        
        print(f"[Network] Auto-calculated feature_dim: {total_features_dim}")

    def forward(self, observations):
        hist = observations["hist"] 
        curr = observations["curr"] 
        
        # 1. 拆分序列
        call_seq, put_seq = torch.chunk(hist, 2, dim=2)
        
        # 2. 物理现状直连 (最后一帧)
        phys_call = call_seq[:, -1, :] 
        phys_put = put_seq[:, -1, :]
        
        # 3. Transformer 语义降维
        with torch.no_grad():
            c_tok = self.pre_moe.encode_tokens(call_seq)
            p_tok = self.pre_moe.encode_tokens(put_seq)
            
        c_latent = self.adapter(c_tok)
        p_latent = self.adapter(p_tok)
        
        # 4. 最终拼接
        combined = torch.cat([
            curr, 
            phys_call, 
            phys_put, 
            c_latent, 
            p_latent
        ], dim=-1)
        
        return combined