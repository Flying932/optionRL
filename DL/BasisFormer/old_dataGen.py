import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

import os
from typing import Tuple, List, Dict, Any, Set

def get_all_excel_paths(folder_path: str='./miniQMT/datasets/label_train_data'):
    """
    获取指定目录下所有非临时 .xlsx 文件的完整路径
    """
    excel_paths = []
    for file in os.listdir(folder_path):
        # 排除临时文件 ~$
        if file.lower().endswith('.xlsx') and not file.startswith('~$'):
            full_path = os.path.join(folder_path, file)
            excel_paths.append(full_path)
    return excel_paths


class OptionTrainingDataGenerator:
    def __init__(self, window_size=32, predict_horizon=1, ks_range=(0.9, 1.1)):
        """
        初始化数据生成器
        :param window_size: 历史观察窗口大小 (T)
        :param predict_horizon: 预测未来多少步 (用于构建 Label)
        :param ks_range: 相对行权价的有效范围，默认 [0.9, 1.1]
        """
        self.window_size = window_size
        self.predict_horizon = predict_horizon
        self.min_ks, self.max_ks = ks_range
        
        # 用于存储处理后的所有样本
        # X: (N, window_size, feature_dim)
        # Y: (N, predict_horizon, feature_dim) 或者自定义 Label
        self.all_X = []
        self.all_Y = []

        # 样本来自于哪个期权id
        self.sample_metadata: List[str] = []
        self.unique_contract_ids: Set[str] = set()
        
        # 定义特征列名映射 (根据你的Excel列名修改)
        self.feature_cols = [
            '相对行权价',   # 相对行权价 (K/S)
            'ttm',          # 到期时间
            '内在价值', # 内在价值/S (假设你计算好了)
            '时间价值', # 时间价值/S (假设你计算好了)
            '隐含波动率',           # 隐含波动率
            'Delta', 'Gamma', 'Vega', 'Theta', 'Rho' # 希腊字母
        ]
        
        self.valid_col = 'greeks_valid' # 有效性标记列
        self.ks_col = '相对行权价'      # 用于筛选的列

    def load_and_process_file(self, file_path: str):
        """
        读取单个期权文件，并提取符合条件的连续时间窗口
        :param file_path: excel文件路径
        """
        idx = file_path.find('.xlsx')
        contract_id = file_path[idx - 15: idx - 7]
        self.unique_contract_ids.add(contract_id)

        try:
            # 1. 读取数据
            df = pd.read_excel(file_path)
            
            # 确保按时间排序
            if 'ts' in df.columns:
                df = df.sort_values('ts').reset_index(drop=True)
            
            # 2. 生成有效性掩码 (Mask)
            # 条件A: Greeks_valid 必须为 1 (或 True)
            # 条件B: 相对行权价在 [0.9, 1.1] 之间
            # 条件C: 必须有交易量或其他你定义的清洗规则 (可选)
            
            condition_valid = (df[self.valid_col] == 1)
            condition_ks = (df[self.ks_col] >= self.min_ks) & (df[self.ks_col] <= self.max_ks)
            
            # 最终掩码
            valid_mask = condition_valid & condition_ks
            
            # 3. 寻找连续片段 (核心逻辑)
            # 利用 cumsum() 给每一段连续的 True 区域分配一个唯一的 group_id
            # 如果中间出现 False，group_id 会增加
            df['group_id'] = (valid_mask != valid_mask.shift()).cumsum()
            
            # 只保留 valid_mask 为 True 的行
            valid_df = df[valid_mask].copy()
            
            # 按 group_id 分组处理每一段连续数据
            for g_id, group_data in valid_df.groupby('group_id'):
                # group_data 就是一段连续的、满足 K/S 范围且 Greeks 有效的时间序列
                self._slice_windows(group_data, contract_id)
                
            print(f"[Success] Loaded {file_path}: Total samples now: {len(self.all_X)}")

        except Exception as e:
            print(f"[Error] Failed to process {file_path}: {e}")

    def _slice_windows(self, segment_df: pd.DataFrame, contract_id: str):
        """
        内部函数：对一段连续的数据进行滑动窗口切片
        """
        # 提取特征矩阵 (L, Features)

        data_values = segment_df[self.feature_cols].values.astype(np.float32)
        L = len(data_values)

        
        # 至少需要 window_size + predict_horizon 长度才能构建一对 (X, Y)
        min_len = self.window_size + self.predict_horizon
        
        if L < min_len:
            return # 长度不够，丢弃该片段
        
        
        # 滑动窗口切片
        # 例如: T=0..31 为 X, T=32 为 Y
        for i in range(L - self.window_size - self.predict_horizon + 1):
            # 构建 X (Input Sequence)
            x_window = data_values[i : i + self.window_size]
            
            # 构建 Y (Label)
            # 这里演示取未来一步的 IV 变化，或者未来一步的所有特征
            # 你可以根据需要修改 Y 的定义
            y_window = data_values[i + self.window_size : i + self.window_size + self.predict_horizon]
            
            # 这里我简单把 Y 设为未来的特征本身，用于自回归预训练
            # 如果你要做 InfoNCE，可能只需要 X，或者构造正负样本
            self.all_X.append(x_window)
            self.all_Y.append(y_window)

            self.sample_metadata.append(contract_id)

    def get_dataset(self):
        """
        返回 PyTorch 格式的 Tensor 数据集
        """
        if not self.all_X:
            print("Warning: No data loaded.")
            return None, None
            
        X_tensor = torch.tensor(np.array(self.all_X), dtype=torch.float32)
        Y_tensor = torch.tensor(np.array(self.all_Y), dtype=torch.float32)
        
        return X_tensor, Y_tensor

    # 按照合约分割数据集
    def get_split_datasets_by_contract(self, 
                                       train_ratio: float = 0.7,
                                       val_ratio: float = 0.15) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
            """
            根据期权合约 ID 进行分割 (无需排序)，确保训练/测试集的合约不重叠。
            :param train_ratio: 训练集包含的合约比例
            :param val_ratio: 验证集包含的合约比例
            :return: (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
            """
            
            # 1. 随机打乱所有唯一的合约 ID
            all_contract_ids = list(self.unique_contract_ids)
            np.random.shuffle(all_contract_ids)
            
            N_contracts = len(all_contract_ids)
            
            # 2. 计算分割点
            N_train_contracts = int(N_contracts * train_ratio)
            N_val_contracts = int(N_contracts * val_ratio)
            
            # 3. 切分合约 ID 集合
            train_ids = set(all_contract_ids[:N_train_contracts])
            val_ids = set(all_contract_ids[N_train_contracts:N_train_contracts+N_val_contracts])
            test_ids = set(all_contract_ids[N_train_contracts+N_val_contracts:])
            
            # 4. 构建样本索引列表
            train_indices, val_indices, test_indices = [], [], []
            
            for idx, metadata in enumerate(self.sample_metadata):
                cid = metadata
                if cid in train_ids:
                    train_indices.append(idx)
                elif cid in val_ids:
                    val_indices.append(idx)
                elif cid in test_ids:
                    test_indices.append(idx)

            # 5. 转换为 PyTorch Tensor
            X_all = torch.tensor(np.array(self.all_X), dtype=torch.float32)
            Y_all = torch.tensor(np.array(self.all_Y), dtype=torch.float32)

            # 6. 使用索引切分样本
            X_train, Y_train = X_all[train_indices], Y_all[train_indices]
            X_val, Y_val = X_all[val_indices], Y_all[val_indices]
            X_test, Y_test = X_all[test_indices], Y_all[test_indices]
            
            print(f"\n--- Data Split Summary (By Contract ID) ---")
            print(f"Total Contracts: {N_contracts}")
            print(f"Train Contracts: {len(train_ids)} | Train Samples: {len(X_train)}")
            print(f"Validation Contracts: {len(val_ids)} | Validation Samples: {len(X_val)}")
            print(f"Test Contracts: {len(test_ids)} | Test Samples: {len(X_test)}")
            
            return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

    # 直接获取最终的loader
    def get_data_loader(self, train_size: float=0.7, val_size: float=0.2, file_list: str=None):
        if file_list is None:
            file_list = get_all_excel_paths()
        for f in file_list:
            self.load_and_process_file(f)

        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = self.get_split_datasets_by_contract(
            train_ratio=train_size, 
            val_ratio=val_size
        )
        train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=64, shuffle=False)
        valid_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=64, shuffle=False)

        return train_loader, test_loader, valid_loader

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 1. 实例化生成器
    generator = OptionTrainingDataGenerator(
        window_size=32, 
        predict_horizon=1, 
        ks_range=(0.9, 1.1) # 严格限制在平值附近
    )
    
    # 2. 模拟传入多个期权文件路径
    # 假设你有 50ETF_10001.xlsx, 50ETF_10002.xlsx ...
    file_list = [
        "./miniQMT/datasets/label_train_data/90006028_159915.xlsx",
        # "data/510050_opt_2.xlsx" 
    ]

    file_list = get_all_excel_paths()
    
    # 3. 循环加载
    # 这里其实就是实现了“身份漂移”处理：
    # 如果 opt_1 在中间某段时间变成了深度实值，它会被 _slice_windows 自动切断，
    # 只取它还是平值的那段时间作为训练数据。
    for f in file_list:
        generator.load_and_process_file(f)

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = generator.get_split_datasets_by_contract(
            train_ratio=0.6, 
            val_ratio=0.2
        )

    # 验证分割结果
    print(f"\n最终训练集 X Shape: {X_train.shape}")
    print(f"最终测试集 X Shape: {X_test.shape}")


    # 5. 放入 DataLoader 供模型训练
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=64, shuffle=False)
    valid_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=64, shuffle=False)