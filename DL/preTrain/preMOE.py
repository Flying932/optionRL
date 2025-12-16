import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


if __name__ == '__main__' or __name__ == '__mp_main__':
    from preTrainITransformer import iTransformerConfig, iTransformer
    from preTrainBasisFormer import BasisFormerConfig, BasisFormer
    from preTrainVARMA import VARMAConfig, VARMAformer
else:
    from preTrain.preTrainITransformer import iTransformerConfig, iTransformer
    from preTrain.preTrainBasisFormer import BasisFormerConfig, BasisFormer
    from preTrain.preTrainVARMA import VARMAConfig, VARMAformer
from torch.utils.data import DataLoader
import time


class GatingNetwork(nn.Module):
    """
    MoE 路由器 (Router): 接收输入特征 (B, L, C)，输出动态权重 (B, 3)。
    """
    def __init__(self, seq_len: int, n_variates: int, d_router: int = 128):
        super().__init__()
        
        # 路由器的内部特征维度
        self.d_router = d_router
        self.op_type_emb = nn.Embedding(2, d_router)
        
        # 特征聚合：将 (B, L, C) 降维到 (B, d_router) 的固定向量
        input_dim = seq_len * n_variates

        
        self.feature_aggregator = nn.Sequential(
            nn.Linear(input_dim, d_router), # 展平 (L*C) -> d_router
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 权重输出层：输出 3 个专家的 logits
        self.router_layer = nn.Linear(self.d_router * 2, 3)
        
    def forward(self, x_enc: torch.Tensor, op_type=None) -> torch.Tensor:
        op_vec = self.op_type_emb(op_type)
        # x_enc 维度: (B, L, C)
        B = x_enc.shape[0]
        
        # 1. 展平并聚合特征 (B, L, C) -> (B, L*C) -> (B, d_router)
        x_flat = x_enc.reshape(B, -1)
        aggregated_feature = self.feature_aggregator(x_flat)

        combined = torch.cat([aggregated_feature, op_vec], dim=-1)
        
        # 2. 计算 logits 并应用 Softmax
        # logits: (B, 3) -> weights: (B, 3)
        logits = self.router_layer(combined)
        weights = F.softmax(logits, dim=-1) # [alpha, beta, gamma]
        
        return weights

# -----------------------------------------------------------
    # ⭐ 新增的特征提取函数：encode_tokens
    # -----------------------------------------------------------
    @torch.no_grad()
    def encode_tokens(self, x_enc: torch.Tensor, op_type: torch.Tensor=None):
        """
        提取 GatingNetwork 的核心特征向量，用于 RL 状态表示。
        
        参数:
            x_enc: (B, L, C) 输入特征
            op_type: (B, 1) 操作类型
            
        返回:
            combined: (B, 2 * d_router) 聚合了序列信息和操作类型的特征向量
        """
        return self._extract_features(x_enc, op_type)

    def _extract_features(self, x_enc: torch.Tensor, op_type: torch.Tensor=None) -> torch.Tensor:
        """
        内部辅助函数：统一处理特征提取和 op_type 维度修正
        """
        # 1. 处理 op_type (维度修正 + 类型转换)
        if op_type is not None:
            # squeeze(-1) 将 (B, 1) -> (B,)
            # long() 确保是整数索引
            op_idx = op_type.squeeze(-1).long()
            op_vec = self.op_type_emb(op_idx) # (B, d_router)
        else:
            # 如果 op_type 为 None，抛出异常或使用全0填充
            # 这里假设 op_type 必须存在，因为 Linear 层是固定的 2*d_router
            raise ValueError("GatingNetwork requires op_type to be provided.")

        # x_enc 维度: (B, L, C)
        B = x_enc.shape[0]
        
        # 2. 展平并聚合序列特征 (B, L, C) -> (B, L*C) -> (B, d_router)
        x_flat = x_enc.reshape(B, -1)
        aggregated_feature = self.feature_aggregator(x_flat)

        if op_vec.dim() == 1:
            op_vec = op_vec.unsqueeze(0)

        # 3. 拼接特征 (B, d_router * 2)
        combined = torch.cat([aggregated_feature, op_vec], dim=-1)
        
        return combined

class PreMOE(nn.Module):
    """
    DIBM-MoE 预训练模型类。
    专家模型实例 (iTransformer, BasisFormer, VARMAFilter) 和 Router 均由外部传入。
    """
    def __init__(
        self, 
        seq_len: int, 
        pred_len: int, 
        n_variates: int=13,
        d_router: int=128,
        target_indices: list=[0, 6, 10]
    ):
        """
        参数:
            router (nn.Module): 路由网络实例 (例如 GatingNetwork)，输出 [alpha, beta, gamma] 权重。
            itransformer_expert (nn.Module): iTransformer 模型实例。
            basisformer_expert (nn.Module): BasisFormer 模型实例。
            varma_expert (nn.Module): VARMAFilter 或其他线性模型实例。
        """
        super().__init__()


        # 被预测的变量
        self.target_indices = target_indices
        
        # MOE路由
        self.router = GatingNetwork(seq_len=seq_len, n_variates=n_variates-1, d_router=d_router)

        # 三个experts
        itransformer_config = iTransformerConfig(seq_len=seq_len, pred_len=pred_len, target_indices=self.target_indices)
        self.iTransformer_expert = iTransformer(itransformer_config)

        basisformer_config = BasisFormerConfig(c_in=n_variates-1, seq_len=seq_len, pred_len=pred_len, target_indices=self.target_indices)
        self.BasisFormer_expert = BasisFormer(basisformer_config)
        
        varma_config = VARMAConfig(c_in=n_variates-1, lookback=seq_len, horizon=pred_len, target_indices=self.target_indices)
        self.VARMA_expert = VARMAformer(varma_config)
        
        self.criterion = nn.MSELoss()
        
    def forward(
        self, 
        x_enc: torch.Tensor, 
        x_mark_enc: Optional[torch.Tensor] = None,
        y_target: Optional[torch.Tensor] = None,
        train: bool = True,
        y_mark_dec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        
        # ----------------------------------------
        # I. 专家预测 (Experts Forecasting)
        # ----------------------------------------
        x_op = x_enc[:, 0, -1].long()
        x_op = (x_op > 0).long().to(x_enc.device)
        x_enc = x_enc[:, :, :-1].to(x_enc.device)

        # 1. VARMA 预测 (线性/统计)
        # 假设 VARMAFilter.forward 接口为 (x_enc) -> Y_pred (B, T, C)
        Y_VARMA = self.VARMA_expert(x_enc, op_type=x_op)
        
        # 2. iTransformer 预测 (通用非线性/Attention)
        # 假设 iTransformer.forward 接口兼容 (x_enc, x_mark_enc, x_dec=None, x_mark_dec)
        Y_iTrans_raw = self.iTransformer_expert(x_enc, x_mark_enc, None, y_mark_dec, op_type=x_op) 
        # 兼容 iTransformer 可能返回 attns 的情况，只取预测结果
        Y_iTrans = Y_iTrans_raw[0] if isinstance(Y_iTrans_raw, tuple) else Y_iTrans_raw
        
        # 3. BasisFormer 预测 (结构化/基底分解)
        # 假设 BasisFormer.forward 接口为 (x, mark, y, train, y_mark)
        basis_output = self.BasisFormer_expert(x_enc, x_mark_enc, y_target, train=train, y_mark=y_mark_dec, op_type=x_op)
        
        if train:
            # 训练模式返回：Y_pred, l_entropy, l_smooth, ...
            Y_Basis, l_entropy, l_smooth, *_ = basis_output
        else:
            # 推理模式返回：Y_pred, score, m, ... (只取第一个输出 Y_pred)
            Y_Basis, *_ = basis_output 
        
        # ----------------------------------------
        # II. 路由决策与预测融合 (Routing & Fusion)
        # ----------------------------------------
        
        # 获取动态权重 [alpha, beta, gamma] (B, 3)
        # 假设 Router 接收 x_enc 并输出权重
        weights = self.router(x_enc, op_type=x_op)
        
        # 调整权重维度以进行批量乘法 (B, 1, 1)
        alpha = weights[:, 0].reshape(-1, 1, 1) # VARMA 权重
        beta  = weights[:, 1].reshape(-1, 1, 1) # BasisFormer 权重
        gamma = weights[:, 2].reshape(-1, 1, 1) # iTransformer 权重
        
        # 动态加权融合预测结果
        Y_pred = alpha * Y_VARMA + beta * Y_Basis + gamma * Y_iTrans
        
        # ----------------------------------------
        # III. 损失计算 (仅在预训练时需要)
        # ----------------------------------------
        if train and y_target is not None:
            # 1. 主损失 (MoE 融合结果的预测损失)
            main_loss = self.criterion(Y_pred, y_target[:, :, self.target_indices])
            
            # 2. 辅助损失 (BasisFormer 内部损失)
            # 权重系数 0.1 可调整
            basis_aux_loss = l_entropy + 0.1 * l_smooth 
            
            # 3. 总损失
            total_loss = main_loss + basis_aux_loss
            
            # 返回: 最终预测, 总损失, 专家权重
            return Y_pred, total_loss, weights
        
        # 推理模式
        Y_LogIV = Y_pred[:, :, 1]
        Y_IV_Actual = torch.exp(Y_LogIV) - 1
        Y_pred[:, :, 1] = Y_IV_Actual

        return Y_pred, weights

    @torch.no_grad()
    def encode_tokens(self, x: torch.Tensor, x_mark: torch.Tensor=None):
        x_op = x[:, 0, -1].long()
        x_op = (x_op > 0).long().to(x.device)

        x = x[:, :, :-1].to(x.device)

        # basisFormer提取到的特征(基底&当前表示)
        basis_high, basis_low = self.BasisFormer_expert.encode_tokens(x, mark=x_mark, op_type=x_op)

        # itransformer提取到的变量间的关系
        i_high, i_low = self.iTransformer_expert.encode_tokens(x, x_mark_enc=x_mark, op_type=x_op)

        # 全局决策状态
        r_agg = self.router.encode_tokens(x, op_type=x_op)

        # VARMA特征
        varma_high, varma_low = self.VARMA_expert.encode_tokens(x, op_type=x_op)

        features_dict = {
            'varma_h': varma_high,
            'varma_l': varma_low,
            'basis_h': basis_high,
            'basis_l': basis_low,
            'itrans_h': i_high,
            'itrans_l': i_low,
            'router': r_agg
        }

        return features_dict
    

def train_one_epoch(model: PreMOE, loader, optim, device, grad_clip=1.0):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(device)  # (B,C,L)
        y = y.to(device)  # (B,C,T)

        # 损失函数已经在MOE中计算出来了
        pred, loss, weights = model(x, y_target=y[:, :, :-1])
        optim.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def evaluate(model: PreMOE, loader, device, loss_fn=None):
    model.eval()
    if loss_fn is None:
        loss_fn = F.mse_loss
    
    total_squared_error = 0.0      # 用于计算最终的 MSE
    total_absolute_error = 0.0     # 用于计算最终的 MAE
    total_reliable_points = 0      # 累加所有批次中的有效点总数

    indices = model.target_indices
    
    with torch.no_grad():
        # 确保 loader 返回 (x, y, mask)
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y = y[:, :, indices]
            y[:, :, 1] = torch.exp(y[:, :, 1]) - 1
            
            pred, weights = model(x) # (B, T, C)

            # --- 修改位置 1: 计算 MSE ---
            # 使用 loss_fn (masked_mse_loss) 并设置 reduction='sum' 来获取总平方误差
            current_squared_error_sum = loss_fn(
                pred, 
                y, 
                reduction='sum'
            ).item()
            
            total_squared_error += current_squared_error_sum
            
            # --- 修改位置 2: 计算 MAE ---
            absolute_error = torch.abs(pred - y)
            masked_absolute_error = absolute_error
            total_absolute_error += masked_absolute_error.sum().item()
            
            total_reliable_points = absolute_error.shape[0]
            

    # 最终的 MSE 和 MAE 是总误差除以所有批次的有效点总数
    if total_reliable_points == 0:
        return 0.0, 0.0
        
    final_mse = total_squared_error / total_reliable_points
    final_mae = total_absolute_error / total_reliable_points
    
    return final_mse, final_mae

def train_model(
    model: PreMOE, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    test_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device, 
    epochs: int = 100, 
    grad_clip: float = 1.0,
    patience: int = 5,
    save_path='./miniQMT/DL/perTrain/weights/iTransformer_best_dummy_data.pth'
):
    """
    主训练函数，负责迭代 epoch，调用训练/评估工具，并实现早停和模型保存。
    
    :param model: 要训练的 PyTorch 模型
    :param train_loader: 训练集 DataLoader
    :param val_loader: 验证集 DataLoader (用于早停)
    :param test_loader: 测试集 DataLoader (用于最终评估)
    :param optimizer: 优化器
    :param loss_fn: 损失函数 (实际在 train_one_epoch 中使用了 F.mse_loss)
    :param device: 设备 (e.g., 'cuda', 'cpu')
    :param epochs: 最大训练轮数
    :param grad_clip: 梯度裁剪阈值
    :param patience: 早停容忍的 epoch 数
    :param save_path: 最佳模型保存路径
    """
    print(f"--- Training preMOE on {device} ---")
    print(f"Total epochs: {epochs}, Early Stopping Patience: {patience}")

    best_val_mse = float('inf')
    epochs_no_improve = 0
    
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 1. 训练阶段
        epoch_start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip)
        
        # 2. 验证阶段
        val_mse, val_mae = evaluate(model, val_loader, device)
        
        epoch_duration = time.time() - epoch_start_time

        # 3. 打印/记录结果
        print(
            f"Epoch {epoch:02d} | Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f}", 
            end=''
        )

        # 4. 模型保存和早停逻辑 (基于验证集 MSE)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(" -> **Saving Best Model**")
        else:
            epochs_no_improve += 1
            print()
        
        # 5. 早停检查
        if epochs_no_improve >= patience:
            print(f"\n[Early Stop] Validation MSE did not improve for {patience} epochs.")
            break

    # 6. 最终测试阶段 (加载最佳模型并评估)
    print("\n--- Final Evaluation ---")
    try:
        model.load_state_dict(torch.load(save_path))
        model.to(device)
        test_mse, test_mae = evaluate(model, test_loader, device)
        print(f"Loaded best model from '{save_path}'.")
        print(f"Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f}")
    except Exception as e:
        print(f"Error loading best model or running test: {e}")
        # 如果加载失败，用当前模型评估一次
        test_mse, test_mae = evaluate(model, test_loader, device)
        print(f"Using last epoch model for test: Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f}")


    total_time = time.time() - start_time
    print(f"--- Total training finished in {total_time/60:.2f} minutes. ---")
    return best_val_mse # 返回最佳验证 MSE



def evaluate_and_plot(model, excel_path, columns, target_col_index=0):
    """
    加载Excel数据，构造滑动窗口，使用模型预测，并绘制预测均值与真实均值的对比图。
    
    参数:
    - model: 已训练好的PyTorch模型，接收 (1, 32, 10) 输出 (1, 4, 10) (假设输出也是10个特征)
    - excel_path: Excel文件路径
    - columns: 需要读取的列名列表 (长度应为10)
    - target_col_index: 想要绘图的列索引 (0-9)，默认为0 (即 columns[0])
    """

    # 1. 设置字体为 SimHei (黑体)，以正常显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei'] 

    # 2. 解决保存图像时负号 '-' 显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 读取数据
    print(f"正在读取文件: {excel_path} ...")
    df = pd.read_excel(excel_path)
    
    # 检查列名是否存在
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Excel中缺少以下列: {missing_cols}")
    
    # 提取数据并转换为 Numpy 数组
    data_raw = df[columns].values
    
    # 检查数据长度
    total_len = len(data_raw)
    input_len = 32
    pred_len = 4
    
    if total_len < input_len + pred_len:
        raise ValueError("数据行数太少，不足以构建一个完整的训练窗口 (32+4 行)")

    true_means = [] # 存储真实的未来4步均值
    pred_means = [] # 存储预测的未来4步均值
    
    print("开始进行滑动窗口预测...")
    
    # 2. 切换模型到评估模式
    model.eval()
    
    # 3. 滑动窗口遍历
    # 我们能够预测的最后一个窗口的起始点是 total_len - 32 - 4
    with torch.no_grad():
        for i in range(total_len - input_len - pred_len + 1):
            # 准备输入 (32, 10)
            x_window = data_raw[i : i + input_len]
            x_window[:, :, 6] = torch.log(x_window[:, :, 6] + 1)
            
            # 准备真实值 (4, 10) -> 取紧接着的4步
            y_true_window = data_raw[i + input_len : i + input_len + pred_len]
            
            # 转换为 Tensor 并增加 Batch 维度 -> (1, 32, 10)
            input_tensor = torch.tensor(x_window, dtype=torch.float32).unsqueeze(0)
            
            # 模型推理
            # 假设模型输出形状为 (1, 4, 10)
            output, _ = model(input_tensor)
            
            # 去掉 batch 维度 -> (4, 10)
            output_np = output.squeeze(0).numpy()
            
            # 4. 计算均值
            # 针对我们关注的特定列 (target_col_index) 计算4步的均值
            # 如果你想看所有特征的均值，可以去掉 [:, target_col_index]
            
            # 真实值的均值 (标量)
            t_mean = np.mean(y_true_window[:, target_col_index])
            true_means.append(t_mean)
            
            # 预测值的均值 (标量)
            p_mean = np.mean(output_np[:, target_col_index])
            pred_means.append(p_mean)

    # 4. 绘图
    target_name = columns[target_col_index]
    
    plt.figure(figsize=(12, 6))
    plt.plot(true_means, label='Actual Mean (Next 4 Steps)', color='black', linewidth=1.5, alpha=0.8)
    plt.plot(pred_means, label='Predicted Mean (Next 4 Steps)', color='red', linestyle='--', linewidth=1.5)
    
    plt.title(f'Prediction vs Actual: {target_name} (Mean of Future 4 Steps)', fontsize=14)
    plt.xlabel('Time Window Index', fontsize=12)
    plt.ylabel(f'{target_name} Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("绘图完成。")

def old_test_predict():
    C_IN = 10
    LOOKBACK = 32
    HORIZON = 4
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 200
    PATIENCE = 5
    MIN_RATIO = 0.8
    D_ROUTER = 128 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PreMOE(seq_len=LOOKBACK, pred_len=HORIZON, n_variates=C_IN, d_router=D_ROUTER)


    SAVE_PATH = './miniQMT/DL/preTrain/weights/preMOE_best_dummy_data.pth'
    state_dict = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 假设你的列名列表
    my_columns = [
        '相对行权价', 'ttm', '内在价值', '时间价值', '隐含波动率', 
        'Theta', 'Vega', 'Gamma', 'Delta', 'Rho', 'op_type'
    ]

    # 调用函数
    # 注意：excel_path 换成你的真实文件路径
    target_col_index=2
    evaluate_and_plot(model, './miniQMT/datasets/label_train_data/10009896_510050.xlsx', my_columns, target_col_index=target_col_index)


def evaluate_and_plot_results(model: PreMOE, 
                              data_dir='./miniQMT/datasets/label_train_data', 
                              output_dir='./miniQMT/datasets/results_test', 
                              start_idx=0,
                              device='cuda' if torch.cuda.is_available() else 'cpu',
                              lookback: int=96,
                              pre_len: int=4,
                              ):
    """
    评估模型并绘制预测均值与实际均值的对比图。
    参数:
    - model: 训练好的模型 (PyTorch nn.Module)
    - data_dir: 训练数据文件夹路径
    - output_dir: 结果导出文件夹路径
    - start_idx: 从文件列表的第几个文件开始处理
    - device: 运行设备
    """


    # 1. 设置字体为 SimHei (黑体)，以正常显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei'] 

    # 2. 解决保存图像时负号 '-' 显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 准备配置
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    model.to(device)
    model.eval() # 切换到评估模式
    
    # 定义列名
    input_columns = [
        '相对行权价', 'ttm', '内在价值', '时间价值', '对数收益率', 'HV160', '隐含波动率', 
        'Theta', 'Vega', 'Gamma', 'Delta', 'Rho', 'op_type'
    ]
    target_columns = ['相对行权价', '隐含波动率', 'Delta']
    
    # 窗口设置
    input_window = lookback
    pred_window = pre_len
    
    # 2. 获取文件列表
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    # 简单的排序，保证顺序一致
    all_files.sort() 
    
    process_files = all_files[start_idx:]
    print(f"共发现 {len(all_files)} 个文件，将从索引 {start_idx} 开始处理 {len(process_files)} 个文件。")

    # 3. 循环处理文件
    for file_name in process_files:
        file_path = os.path.join(data_dir, file_name)
        base_name = os.path.splitext(file_name)[0] # 去除后缀，如 10009896_510050
        
        try:
            # 读取数据
            df = pd.read_excel(file_path)
            
            # 数据长度检查
            if len(df) <= (input_window + pred_window):
                print(f"跳过文件 {file_name}: 数据行数不足 ({len(df)})")
                continue
            
            # ==========================================
            # 注意：此处假设你的模型接受的是Raw Data或者你已经在模型内部处理了归一化。
            # 如果模型是基于归一化数据训练的，你需要在这里加载Scaler对df进行transform。
            # ==========================================
            
            # 准备容器
            X_list = []
            Y_true_list = [] # 存储未来4步的真实均值
            
            # 滑动窗口构建数据
            # 我们需要预测的位置是从 idx = input_window 开始，到 len(df) - pred_window 结束
            # i 代表输入序列的起始点
            for i in range(len(df) - input_window - pred_window + 1):
                # 输入: i 到 i+32
                x_seq = df[input_columns].iloc[i : i + input_window].values
                # 真实标签: i+32 到 i+32+4
                y_seq = df[target_columns].iloc[i + input_window : i + input_window + pred_window].values
                
                X_list.append(x_seq)
                # 直接计算真实值的均值 (axis=0 代表在时间步维度取平均)
                Y_true_list.append(np.mean(y_seq, axis=0)) 
            
            # 转为Tensor
            X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
            Y_true_arr = np.array(Y_true_list) # Shape: (Samples, 3)
            
            # 4. 模型推理
            with torch.no_grad():
                # 假设模型输出 Shape 为 (Batch, 4, 3)
                X_tensor[:, :, 6] = torch.log(X_tensor[:, :, 6] + 1)
                preds, weights = model(X_tensor) 
                
                # 如果模型输出不是(Batch, 4, 3)，请根据实际情况调整，比如可能是(Batch, 3)
                # 根据需求：绘制预测的三个变量未来4个窗口"均值"
                # 对 dim=1 (时间步) 求均值 -> Shape 变为 (Batch, 3)
                if preds.dim() == 3 and preds.shape[1] == pred_window:
                    preds_mean = torch.mean(preds, dim=1)
                else:
                    # 如果模型直接输出了均值或者单步，直接使用
                    preds_mean = preds
                
                preds_np = preds_mean.cpu().numpy()

            # 5. 绘图 (画在一张图上，使用子图区分三个变量)
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            fig.suptitle(f'Model Eval: {base_name} (Win: {input_window}->{pred_window} Mean)', fontsize=16)
            
            # 变量中文名与索引对应
            targets_map = {0: target_columns[0], 1: target_columns[1], 2: target_columns[2]}
            
            for idx, ax in enumerate(axes):
                # 绘制真实值
                ax.plot(Y_true_arr[:, idx], label='Actual (4-step Mean)', color='black', alpha=0.7, linewidth=1.5)
                # 绘制预测值
                ax.plot(preds_np[:, idx], label='Predicted (4-step Mean)', color='red', linestyle='--', alpha=0.8)
                
                ax.set_ylabel(targets_map[idx])
                ax.legend(loc='upper right')
                ax.grid(True, linestyle=':', alpha=0.6)
            
            plt.xlabel('Time Step (Rolling Window Index)')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局给suptitle留空间
            
            # 6. 导出图片
            save_path = os.path.join(output_dir, f"{base_name}.png")
            plt.savefig(save_path, dpi=100)
            plt.close(fig) # 关闭画布释放内存
            
            print(f"已生成: {save_path}")

        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
            continue

    print("所有指定文件处理完成。")


def new_test(lookback: int=32, pre_len: int=4):
    C_IN = 13
    LOOKBACK = lookback
    HORIZON = pre_len
    D_ROUTER = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PreMOE(seq_len=LOOKBACK, pred_len=HORIZON, n_variates=C_IN, d_router=D_ROUTER).to(device)
    SAVE_PATH = f'./miniQMT/DL/preTrain/weights/preMOE_best_dummy_data_{lookback}_{pre_len}_20251214_2.pth'
    state_dict = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    output_dir = f'./miniQMT/datasets/results_test_{lookback}_{pre_len}_20251214'
    data_dir = './miniQMT/datasets/test_label_train_data'
    data_dir = './miniQMT/datasets/all_label_data'

    evaluate_and_plot_results(model, start_idx=0, output_dir=output_dir, lookback=LOOKBACK, pre_len=HORIZON, data_dir=data_dir)


def main(lookback: int=32, pre_len: int=4, batch_size: int=256, num_workers: int=6):
    C_IN = 13
    LOOKBACK = lookback
    HORIZON = pre_len
    BATCH_SIZE = batch_size
    LEARNING_RATE = 1e-4
    EPOCHS = 200
    PATIENCE = 5
    MIN_RATIO = 0.8
    D_ROUTER = 128

    NUM_WORKERS = num_workers

    from preTrainDataGen import OptionTrainingDataGenerator
    generator = OptionTrainingDataGenerator(window_size=LOOKBACK, predict_horizon=HORIZON, min_ratio=MIN_RATIO)
    train_loader, valid_loader, test_loader = generator.get_data_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, tolerate=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PreMOE(seq_len=LOOKBACK, pred_len=HORIZON, n_variates=C_IN, d_router=D_ROUTER).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    train_model(
            model=model,
            train_loader=train_loader,
            val_loader=valid_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=device,
            epochs=EPOCHS,
            grad_clip=1.0,
            patience=PATIENCE,
            save_path=f'./miniQMT/DL/preTrain/weights/preMOE_best_dummy_data_{lookback}_{pre_len}.pth'
        )

    # SAVE_PATH = './miniQMT/DL/preTrain/weights/preMOE_best_dummy_data.pth'
    # state_dict = torch.load(SAVE_PATH, map_location=device)
    # model.load_state_dict(state_dict)
    # model.eval()

    # print(0 / 0)


if __name__ == "__main__":
    new_test(48, 4)
    # lookback_list = [32, 48, 64, 96, 80, 128]
    # pre_len_list = [4, 6, 8, 10, 12]

    # for lookback in lookback_list:
    #     for pre_len in pre_len_list:
    #         try:
    #             main(lookback=lookback, pre_len=pre_len, batch_size=512, num_workers=20)
    #             print(f"Success, lookback = {lookback}, pre_len = {pre_len}")
    #         except Exception as e:
    #             print(f"Error, lookback = {lookback}, pre_len = {pre_len}, e = {e}")



