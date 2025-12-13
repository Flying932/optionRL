import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys

# ==========================================
# 1. 基础组件定义 (MultiViewAdapter - GELU版)
# ==========================================

class ViewProjector(nn.Module):
    """
    单视图投影器：融合 High-Dim 和 Low-Dim
    [核心修正] 使用 GELU 代替 Tanh，保证梯度流在极端值下不消失
    """
    def __init__(self, high_dim, low_dim, out_dim=64):
        super().__init__()
        # 1. 高维流：线性压缩
        self.high_net = nn.Sequential(
            nn.LayerNorm(high_dim),
            nn.Linear(high_dim, out_dim),
        )
        # 2. 低维流：归一化 + 升维 + 非线性激活
        self.low_net = nn.Sequential(
            nn.LayerNorm(low_dim),
            nn.Linear(low_dim, 32),
            nn.GELU() # 使用 GELU 替代 Tanh
        )
        # 3. 融合
        self.fusion = nn.Sequential(
            nn.Linear(out_dim + 32, out_dim),
            nn.LayerNorm(out_dim)
        )
        
    def forward(self, x_high, x_low):
        h = self.high_net(x_high)
        l = self.low_net(x_low)
        return self.fusion(torch.cat([h, l], dim=-1))

class MultiViewAdapter(nn.Module):
    """
    多视图适配器：将 VARMA, Basis, iTrans 的特征分流处理后再融合
    """
    def __init__(self, dims_dict, final_dim=128):
        super().__init__()
        view_dim = 48
        
        # 定义三个视图的投影器
        self.varma_proj = ViewProjector(dims_dict['varma_h'], dims_dict['varma_l'], view_dim)
        self.basis_proj = ViewProjector(dims_dict['basis_h'], dims_dict['basis_l'], view_dim)
        self.itrans_proj = ViewProjector(dims_dict['itrans_h'], dims_dict['itrans_l'], view_dim)
        
        # Router 特征 (本身就是低维高语义，直接处理)
        self.router_proj = nn.Sequential(
            nn.LayerNorm(dims_dict['router']),
            nn.Linear(dims_dict['router'], 32)
        )
        
        # 最终融合层
        total_in = view_dim * 3 + 32
        self.final_net = nn.Sequential(
            nn.Linear(total_in, final_dim),
            nn.LayerNorm(final_dim)
        )
        
    def forward(self, inputs):
        """
        inputs: 字典，包含各模型的输出 {'varma_h': ..., 'varma_l': ...}
        """
        v_varma = self.varma_proj(inputs['varma_h'], inputs['varma_l'])
        v_basis = self.basis_proj(inputs['basis_h'], inputs['basis_l'])
        v_itrans = self.itrans_proj(inputs['itrans_h'], inputs['itrans_l'])
        v_router = self.router_proj(inputs['router'])
        
        combined = torch.cat([v_varma, v_basis, v_itrans, v_router], dim=-1)
        return self.final_net(combined)

# ==========================================
# 2. 探针网络定义
# ==========================================

class ProbeNetwork(nn.Module):
    def __init__(self, pre_moe, adapter):
        super().__init__()
        self.extractor = pre_moe  # 冻结的基座
        self.adapter = adapter    # 待训练的适配器
        self.head = nn.Linear(128, 1) # 预测头 (输出 IV 标量)

    def forward(self, x, x_mark=None):
        # 1. 提取特征 (No Gradient)
        with torch.no_grad():
            raw_inputs = self.extractor.encode_tokens(x, x_mark=x_mark)
        
        # 2. 适配与预测 (Gradient Flow)
        features = self.adapter(raw_inputs)
        prediction = self.head(features)
        return prediction

# ==========================================
# 3. 核心测试逻辑
# ==========================================

def run_validity_test():
    # --- 配置区域 ---
    SEQ_LEN = 32
    PRED_LEN = 4
    C_IN = 10
    BATCH_SIZE = 64
    LR = 2e-3          # 稍微大一点的学习率，加速验证
    EPOCHS = 15
    IV_IDX = 3         # 隐含波动率在特征中的索引
    
    # 路径配置 (请根据实际情况修改)
    PRETRAIN_WEIGHTS = './miniQMT/DL/preTrain/weights/preMOE_best_dummy_data.pth'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f">>> 开始特征有效性探针测试 (Device: {DEVICE})")

    # 1. 动态导入依赖
    try:
        from preTrain.preTrainDataGen import OptionTrainingDataGenerator
        from preTrain.preMOE import PreMOE 
    except ImportError:
        print("❌ 错误: 找不到 preTrainDataGen.py 或 preMOE.py，请确保脚本在正确目录下。")
        return

    # 2. 准备数据
    print("--- 1. 加载数据 ---")
    gen = OptionTrainingDataGenerator(window_size=SEQ_LEN, predict_horizon=PRED_LEN)
    train_loader, _, test_loader = gen.get_data_loader()
    
    # 3. 初始化基座
    print("--- 2. 初始化预训练基座 ---")
    pre_moe = PreMOE(seq_len=SEQ_LEN, pred_len=PRED_LEN, n_variates=C_IN, d_router=128).to(DEVICE)
    
    # 加载权重
    if os.path.exists(PRETRAIN_WEIGHTS):
        print(f"✅ 加载权重: {PRETRAIN_WEIGHTS}")
        state = torch.load(PRETRAIN_WEIGHTS, map_location=DEVICE)
        # 移除 DDP 可能产生的 'module.' 前缀
        state = {k.replace('module.', ''): v for k, v in state.items()}
        try:
            pre_moe.load_state_dict(state)
        except Exception as e:
            print(f"⚠️ 权重加载警告: {e}")
            print(">> 将尝试使用部分加载或随机初始化继续测试...")
    else:
        print(f"❌ 严重警告: 权重文件不存在！测试结果将无效 (Random Guess)。")
    
    pre_moe.eval() # 冻结基座

    # 4. 自动维度推断 (Auto-Config)
    print("--- 3. 自动推断特征维度 ---")
    dummy_x = torch.randn(2, SEQ_LEN, C_IN).to(DEVICE) # (B, L, C)
    # 尝试调用 encode_tokens，如果报错说明接口没改好
    try:
        with torch.no_grad():
            raw_out = pre_moe.encode_tokens(dummy_x) # 假设不需要 permute，内部处理
            
        if not isinstance(raw_out, dict):
            print("❌ 错误: PreMOE.encode_tokens 返回的不是字典！")
            print(">> 请修改 preMOE.py，确保它返回 {'varma_h': ..., 'varma_l': ...} 格式。")
            print(">> 这是一个强制要求，否则 MultiViewAdapter 无法工作。")
            return

        dims_dict = {k: v.shape[-1] for k, v in raw_out.items()}
        print(f"✅ 推断成功: {dims_dict}")
        
    except Exception as e:
        print(f"❌ 推断失败: {e}")
        print(">> 请检查输入形状是否为 (Batch, Seq, Channel)。")
        return

    # 5. 构建探针
    adapter = MultiViewAdapter(dims_dict, final_dim=128).to(DEVICE)
    probe = ProbeNetwork(pre_moe, adapter).to(DEVICE)
    
    optimizer = optim.Adam(probe.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # 6. 训练循环
    print("\n--- 4. 开始训练与验证 ---")
    print(f"{'Epoch':<6} | {'Model Loss':<12} | {'Naïve Loss':<12} | {'Ratio':<8} | {'Dir Acc':<8} | {'Status'}")
    print("-" * 75)

    for epoch in range(EPOCHS):
        # A. 训练
        probe.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            # 目标: 未来 IV 均值
            target = by[:, :, IV_IDX].mean(dim=1, keepdim=True)
            
            pred = probe(bx)
            loss = loss_fn(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # B. 验证 (核心环节)
        probe.eval()
        loss_m_sum, loss_n_sum = 0.0, 0.0
        correct_dir = 0
        total = 0
        
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                
                # 真实未来
                target = by[:, :, IV_IDX].mean(dim=1, keepdim=True)
                # 傻瓜基准 (最后时刻的 IV)
                naive_pred = bx[:, -1, IV_IDX].unsqueeze(-1)
                # 模型预测
                model_pred = probe(bx)
                
                loss_m_sum += loss_fn(model_pred, target).item() * bx.size(0)
                loss_n_sum += loss_fn(naive_pred, target).item() * bx.size(0)
                
                # 方向准确率
                diff_pred = model_pred - naive_pred
                diff_true = target - naive_pred
                # 符号相同即为方向正确 (乘积 > 0)
                correct_dir += ((diff_pred * diff_true) > 0).float().sum().item()
                total += bx.size(0)
        
        avg_m = loss_m_sum / total
        avg_n = loss_n_sum / total
        ratio = avg_m / (avg_n + 1e-8)
        da = correct_dir / total
        
        # 判定
        if ratio < 0.98 and da > 0.52:
            status = "✅ 有效"
        elif ratio > 1.0:
            status = "❌ 差于基准"
        else:
            status = "➖ 持平"
            
        print(f"{epoch+1:<6} | {avg_m:.6f}     | {avg_n:.6f}     | {ratio:.4f}   | {da*100:.1f}%    | {status}")

    print("\n--- 最终诊断 ---")
    if ratio < 0.95:
        print("结论：特征提取器【有效】。")
        print("分析：模型成功击败了 Naïve Baseline，说明特征中包含了关于未来走势的有效信息。")
        print("建议：既然特征有效，PPO 训练不好的原因主要在于 Reward 函数或 RL 参数，请集中精力调优 RL 部分。")
    else:
        print("结论：特征提取器【效能不足】。")
        print("分析：模型无法超越简单的惯性预测。可能原因：")
        print("1. 预训练基座没有收敛。")
        print("2. 市场本身处于随机游走状态，IV 不可预测。")
        print("3. encode_tokens 没有正确返回关键的统计特征 (mu/sigma)。")

if __name__ == "__main__":
    run_validity_test()