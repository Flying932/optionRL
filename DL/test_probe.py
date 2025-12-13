import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# 导入依赖
from preTrain.preTrainDataGen import OptionTrainingDataGenerator
from preTrain.preMOE import PreMOE 

# ==========================================
# 1. 适配器 (保持 GELU 版不变)
# ==========================================
class ViewProjector(nn.Module):
    def __init__(self, high_dim, low_dim, out_dim=64):
        super().__init__()
        self.high_net = nn.Sequential(nn.LayerNorm(high_dim), nn.Linear(high_dim, out_dim))
        self.low_net = nn.Sequential(nn.LayerNorm(low_dim), nn.Linear(low_dim, 32), nn.GELU()) 
        self.fusion = nn.Sequential(nn.Linear(out_dim + 32, out_dim), nn.LayerNorm(out_dim))
    def forward(self, x_high, x_low):
        return self.fusion(torch.cat([self.high_net(x_high), self.low_net(x_low)], dim=-1))

class MultiViewAdapter(nn.Module):
    def __init__(self, dims_dict, final_dim=128):
        super().__init__()
        view_dim = 48
        self.varma_proj = ViewProjector(dims_dict['varma_h'], dims_dict['varma_l'], view_dim)
        self.basis_proj = ViewProjector(dims_dict['basis_h'], dims_dict['basis_l'], view_dim)
        self.itrans_proj = ViewProjector(dims_dict['itrans_h'], dims_dict['itrans_l'], view_dim)
        self.router_proj = nn.Sequential(nn.LayerNorm(dims_dict['router']), nn.Linear(dims_dict['router'], 32))
        self.final_net = nn.Sequential(nn.Linear(view_dim * 3 + 32, final_dim), nn.LayerNorm(final_dim))
    def forward(self, inputs):
        v_varma = self.varma_proj(inputs['varma_h'], inputs['varma_l'])
        v_basis = self.basis_proj(inputs['basis_h'], inputs['basis_l'])
        v_itrans = self.itrans_proj(inputs['itrans_h'], inputs['itrans_l'])
        v_router = self.router_proj(inputs['router'])
        return self.final_net(torch.cat([v_varma, v_basis, v_itrans, v_router], dim=-1))

class ProbeNetwork(nn.Module):
    def __init__(self, pre_moe, adapter):
        super().__init__()
        self.extractor = pre_moe
        self.adapter = adapter
        self.head = nn.Linear(128, 1)

    def forward(self, x, x_mark=None):
        # 这里的输入 x 应该是 (Batch, Seq, Channel)
        # PreMOE 内部通常需要 (Batch, Seq, Channel) 或者会自己 Permute
        # 我们假设 PreMOE.encode_tokens 能处理 (B, L, C)
        # 如果 PreMOE 报错，可能需要在这里手动 permute(0, 2, 1)
        with torch.no_grad():
            raw_inputs = self.extractor.encode_tokens(x, x_mark=x_mark)
        features = self.adapter(raw_inputs)
        delta_pred = self.head(features)
        return delta_pred

# ==========================================
# 2. 训练逻辑 (维度索引修正版)
# ==========================================
def run_correct_probe():
    SEQ_LEN = 32
    PRED_LEN = 4
    C_IN = 10
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 10
    
    # [关键] 你的 feature_cols 中 IV 是第 5 个 (index 4)
    # feature_cols = [KS, ttm, Intr, Time, IV, Delta, ...]
    IV_IDX = 4 
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    PRETRAIN_PATH = './miniQMT/DL/preTrain/weights/preMOE_best_dummy_data.pth'

    print(f">>> 启动 IV 增量预测探针 (维度索引已修正)")
    
    # 1. 数据
    gen = OptionTrainingDataGenerator(window_size=SEQ_LEN, predict_horizon=PRED_LEN)
    train_loader, _, test_loader = gen.get_data_loader()
    
    # 2. 模型
    pre_moe = PreMOE(seq_len=SEQ_LEN, pred_len=PRED_LEN, n_variates=C_IN, d_router=128).to(DEVICE)
    
    # === 这里的开关用于验证 ===
    LOAD_WEIGHTS = True 
    # ========================

    if LOAD_WEIGHTS and os.path.exists(PRETRAIN_PATH):
        print(f"✅ 加载预训练权重: {PRETRAIN_PATH}")
        state = torch.load(PRETRAIN_PATH, map_location=DEVICE)
        state = {k.replace('module.', ''): v for k, v in state.items()}
        try:
            pre_moe.load_state_dict(state)
        except Exception as e:
            print(f"⚠️ 权重加载警告: {e}")
    else:
        print("⚠️ 使用随机权重 (Baseline Test)")
        
    pre_moe.eval()

    # 自动推断维度
    with torch.no_grad():
        dummy_x = torch.randn(2, SEQ_LEN, C_IN).to(DEVICE)
        # 注意：这里传入 (B, L, C)
        raw_out = pre_moe.encode_tokens(dummy_x) 
        dims = {k: v.shape[-1] for k, v in raw_out.items()}
        print(f"--- 特征维度: {dims} ---")

    adapter = MultiViewAdapter(dims_dict=dims).to(DEVICE)
    probe = ProbeNetwork(pre_moe, adapter).to(DEVICE)
    opt = optim.Adam([{'params': probe.adapter.parameters()}, {'params': probe.head.parameters()}], lr=LR)
    loss_fn = nn.MSELoss()

    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Ratio':<12} | {'Status'}")
    print("-" * 55)

    for epoch in range(EPOCHS):
        probe.train()
        train_loss = 0
        count = 0
        
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            # bx shape: (Batch, Seq_Len, Channels) = (64, 32, 10)
            # by shape: (Batch, Pred_Len, Channels) = (64, 4, 10)
            
            # --- [关键修正] 正确的切片逻辑 ---
            # 取当前时刻 (Time=-1) 的 IV (Feat=IV_IDX)
            curr_iv = bx[:, -1, IV_IDX].unsqueeze(-1) # (B, 1)
            
            # 取未来所有时刻的 IV 均值
            future_iv = by[:, :, IV_IDX].mean(dim=1, keepdim=True) # (B, 1)
            
            target_delta = future_iv - curr_iv
            
            # 前向
            pred_delta = probe(bx)
            loss = loss_fn(pred_delta, target_delta)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss += loss.item() * bx.size(0)
            count += bx.size(0)
            
        avg_train = train_loss / count
        
        # 验证
        probe.eval()
        loss_m, loss_n = 0.0, 0.0
        t_count = 0
        
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                
                curr_iv = bx[:, -1, IV_IDX].unsqueeze(-1)
                future_iv = by[:, :, IV_IDX].mean(dim=1, keepdim=True)
                target_delta = future_iv - curr_iv
                
                pred_delta = probe(bx)
                
                loss_m += loss_fn(pred_delta, target_delta).item() * bx.size(0)
                loss_n += loss_fn(torch.zeros_like(target_delta), target_delta).item() * bx.size(0)
                t_count += bx.size(0)
                
        ratio = (loss_m / t_count) / (loss_n / t_count + 1e-10)
        status = "✅ 有效" if ratio < 0.99 else "❌ 无效"
        
        print(f"{epoch+1:<6} | {avg_train:.6f}     | {ratio:.4f}       | {status}")

if __name__ == '__main__':
    run_correct_probe()