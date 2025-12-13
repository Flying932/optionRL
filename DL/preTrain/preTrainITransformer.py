from sympy import flatten
import torch.nn as nn
import torch
try:
    from utils.Transformer_EncDec import Encoder, EncoderLayer
    from utils.SelfAttention_Family import FullAttention, AttentionLayer
    from utils.Embed import DataEmbedding_inverted
except Exception as e:
    from preTrain.utils.Transformer_EncDec import Encoder, EncoderLayer
    from preTrain.utils.SelfAttention_Family import FullAttention, AttentionLayer
    from preTrain.utils.Embed import DataEmbedding_inverted

# from utils.Transformer_EncDec import Encoder, EncoderLayer
# from utils.SelfAttention_Family import FullAttention, AttentionLayer
# from utils.Embed import DataEmbedding_inverted

import torch.nn.functional as F
import time
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from typing import List

@dataclass
class iTransformerConfig:
    # 序列长度相关
    seq_len: int=32          # 输入序列长度
    pred_len: int=4        # 预测长度

    # 模型结构
    d_model: int=128          # 每个 token 的隐藏维度
    e_layers: int=4         # Encoder 层数
    n_heads: int=8          # Multi-head Attention 的 head 数
    d_ff: int=2048             # FFN 隐藏层维度
    factor: int=1           # FullAttention 中用到的 factor（论文中的采样因子）

    # Embedding 配置, 似乎并没有被调用
    embed: str='fixed'           # embedding 类型，例如 "timeF"
    freq: str='h'             # 时间频率，例如 "h", "t", "s" 等
    dropout: float=0.1        # dropout 比例

    # 激活 & 归一化
    activation: str='gelu'       # FFN 激活函数名，例如 "gelu"、"relu"
    use_norm: bool=True       # 是否使用 Non-stationary Transformer 的归一化

    # 其他
    output_attention: bool=False          # forward 时是否返回 attention
    class_strategy: str='projection'     # 分类/投影策略（原实现里预留的字段）projection/average/cls_token

    # 设置变量idx
    target_indices: List[int] = field(default_factory=list)


class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs: iTransformerConfig):
        super(iTransformer, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # 是否使用归一化, 来自: non-stationary Transformer
        self.use_norm = configs.use_norm

        # 定义预测的位置
        self.target_indices = configs.target_indices
        self.num_targets = len(self.target_indices)

        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # op_embedding
        self.op_type_embedding = nn.Embedding(2, configs.d_model)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, op_type=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        if op_type is not None:
            op_embed = self.op_type_embedding(op_type.squeeze(-1).long())
            enc_out = enc_out + op_embed.unsqueeze(1)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        # 需要预测的三个变量
        target_dec_out = dec_out[:, :, self.target_indices]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            # dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            # dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            target_stdev = stdev[:, 0, self.target_indices].unsqueeze(1).repeat(1, self.pred_len, 1)
            target_means = means[:, 0, self.target_indices].unsqueeze(1).repeat(1, self.pred_len, 1)
            
            target_dec_out = target_dec_out * target_stdev + target_means

        return target_dec_out, attns


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, op_type=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, op_type)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        

# -----------------------------------------------------------
    # ⭐ 新增的特征提取函数：encode_token (已修正)
    # -----------------------------------------------------------
    @torch.no_grad()
    def encode_tokens(self, x_enc: torch.Tensor, x_mark_enc=None, op_type=None):
        """
        提取 iTransformer 的核心特征 Token。
        
        参数:
            x_enc: (B, L, N) 输入时间序列数据
            x_mark_enc: (B, L, D_mark) 输入时间戳特征
            op_type: (B, 1) 操作类型索引 (可选)
            
        返回:
            flatten_enc_out: (B, N*E) 经过 Encoder 及其它嵌入后的高维特征
            low_features:    (B, N*2) 统计特征 (均值 + 标准差)
        """
        
        # 1. 归一化 (Normalization)
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        else:
            # 【修正】如果不使用归一化，使用 0 均值和 1 标准差占位，防止后续 reshape 报错
            B, _, N = x_enc.shape
            means = torch.zeros((B, 1, N), device=x_enc.device)
            stdev = torch.ones((B, 1, N), device=x_enc.device)

        # 2. Embedding (Inverted DataEmbedding)
        # B L N -> B N E 
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # 3. Encoder (Self-Attention & FFN)
        # B N E -> B N E
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # 【新增】处理 op_type，保持与 forecast 逻辑一致
        if op_type is not None:
            # op_embed = self.op_type_embedding(op_type.squeeze(-1).long())
            op_idx = op_type.view(-1).long() # Shape 变成 [1]
            op_embed = self.op_type_embedding(op_idx)
            # op_embed: (B, D) -> (B, 1, D)
            enc_out = enc_out + op_embed.unsqueeze(1)

        # 4. 展平与特征拼接
        # Flatten enc_out: (B, N, E) -> (B, N*E)
        flatten_enc_out = enc_out.reshape(enc_out.shape[0], -1)
        
        # Flatten stats: (B, 1, N) -> (B, N)
        flatten_stdev = stdev.reshape(stdev.shape[0], -1)
        flatten_means = means.reshape(means.shape[0], -1)

        # 拼接低维统计特征
        low_features = torch.cat([flatten_stdev, flatten_means], dim=1)

        # 返回高维度和低维度两个语义特征
        return flatten_enc_out, low_features



def train_one_epoch(model, loader, optim, device, grad_clip=1.0):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(device)  # (B,C,L)
        y = y.to(device)  # (B,C,T)
        pred = model(x)

        # 损失函数(损失掩码)
        loss = F.mse_loss(pred, y)

        optim.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def evaluate(model, loader, device, loss_fn=None):
    model.eval()
    if loss_fn is None:
        loss_fn = F.mse_loss
    
    total_squared_error = 0.0      # 用于计算最终的 MSE
    total_absolute_error = 0.0     # 用于计算最终的 MAE
    total_reliable_points = 0      # 累加所有批次中的有效点总数
    
    with torch.no_grad():
        # 确保 loader 返回 (x, y, mask)
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x) # (B, T, C)


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
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    test_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    loss_fn: nn.Module, # 实际上在 train_one_epoch 里是 MSELoss，这里只作签名占位
    device: torch.device, 
    epochs: int = 20, 
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
    print(f"--- Training VARMAformer on {device} ---")
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


if __name__ == "__main__":
    C_IN = 10
    LOOKBACK = 32
    HORIZON = 4
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    PATIENCE = 5
    MIN_RATIO = 0.8

    from preTrainDataGen import OptionTrainingDataGenerator
    generator = OptionTrainingDataGenerator(window_size=LOOKBACK, predict_horizon=HORIZON, min_ratio=MIN_RATIO)
    train_loader, valid_loader, test_loader = generator.get_data_loader()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    cfg = iTransformerConfig(
        seq_len=LOOKBACK, pred_len=HORIZON,
        d_model=128, e_layers=4, n_heads=8,
        d_ff=2048, dropout=0.1, output_attention=False
    )

    model = iTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss() # 实际在 train_one_epoch 中使用了 F.mse_loss

    # train_model(
    #         model=model,
    #         train_loader=train_loader,
    #         val_loader=valid_loader,
    #         test_loader=test_loader,
    #         optimizer=optimizer,
    #         loss_fn=loss_fn,
    #         device=device,
    #         epochs=300,
    #         grad_clip=1.0,
    #         patience=5,
    #         save_path='./miniQMT/DL/preTrain/weights/iTransformer_best_dummy_data.pth'
    #     )
    
    SAVE_PATH = './miniQMT/DL/preTrain/weights/iTransformer_best_dummy_data.pth'
    state_dict = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(0 / 0)

