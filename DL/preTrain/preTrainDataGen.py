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
    if not os.path.exists(folder_path):
        return excel_paths
        
    for file in os.listdir(folder_path):
        # 排除临时文件 ~$
        if file.lower().endswith('.xlsx') and not file.startswith('~$'):
            full_path = os.path.join(folder_path, file)
            if full_path.endswith('_510050.xlsx'):
                excel_paths.append(full_path)
    return excel_paths


class OptionTrainingDataGenerator:
    def __init__(self, window_size=32, predict_horizon=1, ks_range=(0.9, 1.1), min_ratio: float=0.8, path: str=None):
        """
        初始化数据生成器
        :param window_size: 历史观察窗口大小 (T)
        :param predict_horizon: 预测未来多少步 (用于构建 Label)
        :param ks_range: 相对行权价的有效范围，默认 [0.9, 1.1]
        :parma min_ratio: 最小可容忍非有效希腊字母个数
        """
        self.window_size = window_size
        self.predict_horizon = predict_horizon
        self.min_ks, self.max_ks = ks_range
        self.min_ratio = min_ratio

        self.path = path
        
        # 用于存储处理后的所有样本
        # X: (N, window_size, feature_dim)
        # Y: (N, predict_horizon, feature_dim) 或者自定义 Label
        # MASK: (N, window_size, 1) or (N, window_size, feature_dim)
        self.all_X = []
        self.all_Y = []

        self.mask_cols = [
            '隐含波动率', 
            'Delta', 'Gamma', 'Vega', 'Theta', 'Rho' 
        ]
        
        # 样本来自于哪个期权id
        self.sample_metadata: List[str] = []
        self.unique_contract_ids: Set[str] = set()
        
        # 定义特征列名映射 (根据你的Excel列名修改)
        self.feature_cols = [
            '相对行权价',   # 相对行权价 (K/S)
            'ttm',          # 到期时间
            '内在价值',     # 内在价值/S (假设你计算好了)
            '时间价值',     # 时间价值/S (假设你计算好了)
            '对数收益率',
            'HV160',
            '隐含波动率',
            'Theta','Vega', 'Gamma', 'Delta', 'Rho', 'op_type'
        ]
        
        self.valid_col = 'greeks_valid' # 有效性标记列
        self.ks_col = '相对行权价'       # 用于筛选的列

    def load_and_process_file(self, file_path: str, tolerate: int = None):
        """
        读取单个期权文件，并提取符合条件的连续时间窗口。
        包含了 contract_id 的定义，以及 LOCF 填充和 R_threshold 过滤逻辑。
        """
        if tolerate is None:
            tolerate = max(min(1, self.window_size * 0.15), 8)
        

        # >>> 修正：contract_id 定义 <<<
        idx = file_path.find('.xlsx')
        if idx > 15:
            # 假设文件名中有 15 个字符 ID
            contract_id = file_path[idx - 15: idx - 7]
        else:
            # 使用文件名作为 ID
            contract_id = os.path.basename(file_path).replace('.xlsx', '')
        
        old_len_x = len(self.all_X)
            
        self.unique_contract_ids.add(contract_id)
        # >>> contract_id 定义结束 <<<

        # 1. 读取数据
        df = pd.read_excel(file_path)
        if 'ts' in df.columns:
            df = df.sort_values('ts').reset_index(drop=True)
        
        if self.valid_col not in df.columns:
            df[self.valid_col] = 1

        mask_cols_to_zero = ['隐含波动率', 'Theta','Vega', 'Gamma', 'Delta', 'Rho']
        
        # 2. **数据填充 (Feature Imputation) - LOCF**
        df[mask_cols_to_zero] = df[mask_cols_to_zero].where(df[self.valid_col] == 1).ffill()
    
        # log_candidates = ['隐含波动率', 'Vega', 'Gamma', '历史波动率', 'hv20', 'hv_short']
        log_candidates = ['隐含波动率']
        cols_to_log = [c for c in log_candidates if c in df.columns]

        for col in cols_to_log:
            # 1. 安全截断：防止出现负数导致 log 报错 (理论上 IV 不应为负，但可能有脏数据)
            #    同时也处理了极小的 0 值，log1p(0) = 0
            df[col] = df[col].clip(lower=0.0)
            
            # 2. 对数变换：ln(x + 1)
            #    这样 1.8 (180%) 会变成 1.03，0.3 会变成 0.26
            df[col] = np.log1p(df[col])


        # 3. **时间窗口截断逻辑 (基于容忍度)**
        condition_valid_raw = (df[self.valid_col] == 1)
        run_id = (condition_valid_raw != condition_valid_raw.shift()).cumsum()
        run_counts = df.groupby(run_id)[self.valid_col].transform('count')
        
        is_tolerable = (~condition_valid_raw) & (run_counts <= tolerate)
        condition_valid_extended = condition_valid_raw | is_tolerable
        
        condition_ks = (df[self.ks_col] >= self.min_ks) & (df[self.ks_col] <= self.max_ks)
        valid_mask_segment = condition_valid_extended & condition_ks
        
        # 4. 寻找连续片段并切片
        df['group_id'] = (valid_mask_segment != valid_mask_segment.shift()).cumsum()
        
        valid_df = df[valid_mask_segment].copy()

        # 提取原始 greeks_valid 数据数组，用于在 _slice_windows 中切片
        greeks_valid_data = df[self.valid_col]

        
        # 按 group_id 分组处理每一段连续数据
        for g_id, group_data in valid_df.groupby('group_id'):
            if valid_mask_segment.loc[group_data.index[0]]:
                
                # 提取该分组对应的 greeks_valid 数据段
                indices_in_full_data = group_data.index.values
                group_mask_data = greeks_valid_data[indices_in_full_data]
                
                # 调用切片函数
                self._slice_windows(group_data, group_mask_data, contract_id)
        if len(self.all_X) == old_len_x:
            self.unique_contract_ids.remove(contract_id)
            print(f"[Warning] Loaded {file_path}, 但数据无意义, 没有添加~")
        else:
            print(f"[Success] Loaded {file_path}: Total samples now: {len(self.all_X)}")


    def _slice_windows(self, segment_df: pd.DataFrame, segment_mask_df: pd.DataFrame, contract_id: str):
        """
        内部函数：对一段连续的数据进行滑动窗口切片
        同时切片 Feature (X) 和 Mask (M)
        """
        # 提取特征矩阵 (L, Features)
        data_values = segment_df[self.feature_cols].values.astype(np.float32)

        # 提取 Mask 矩阵 (L, )
        mask_values = segment_mask_df.values.astype(np.float32)
        
        # 检查是否有 NaN (例如文件开头就是无效数据，无法 ffill)
        if np.isnan(data_values).any():
            return 

        L = len(data_values)
        min_len = self.window_size + self.predict_horizon
        
        if L < min_len:
            return # 长度不够，丢弃该片段
        
        # 滑动窗口切片
        for i in range(L - self.window_size - self.predict_horizon + 1):
            # 构建 X (Input Sequence)
            x_window = data_values[i : i + self.window_size]
            
            # 构建 Y (Label)
            y_window = data_values[i + self.window_size : i + self.window_size + self.predict_horizon]
            
            # 构建 Mask
            mask_window = mask_values[i : i + self.window_size]

            not_zero_ratio = mask_window.sum() / len(mask_window)

            if not_zero_ratio < self.min_ratio:
                continue

            
            self.all_X.append(x_window)
            self.all_Y.append(y_window)

            self.sample_metadata.append(contract_id)

    def get_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回 PyTorch 格式的 Tensor 数据集 (X, Y, Mask)
        """
        if not self.all_X:
            print("Warning: No data loaded.")
            return None, None, None
            
        X_tensor = torch.tensor(np.array(self.all_X), dtype=torch.float32)
        Y_tensor = torch.tensor(np.array(self.all_Y), dtype=torch.float32)

        
        return X_tensor, Y_tensor

    # 按照合约分割数据集
    def get_split_datasets_by_contract(self, 
                                        train_ratio: float = 0.7,
                                        val_ratio: float = 0.15) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        根据期权合约 ID 进行分割 (无需排序)，确保训练/测试集的合约不重叠。
        返回 (X_train, Y_train, M_train), (X_val, Y_val, M_val), (X_test, Y_test, M_test)
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

        # 5. 转换为 PyTorch Tensor (同时包含 Mask)
        if not self.all_X:
            print("Error: No data to split.")
            return (torch.empty(0), torch.empty(0), torch.empty(0)), (torch.empty(0), torch.empty(0), torch.empty(0)), (torch.empty(0), torch.empty(0), torch.empty(0))

        X_all = torch.tensor(np.array(self.all_X), dtype=torch.float32)
        Y_all = torch.tensor(np.array(self.all_Y), dtype=torch.float32)

        # 6. 使用索引切分样本
        if train_indices:
            X_train, Y_train = X_all[train_indices], Y_all[train_indices]
        else:
            X_train, Y_train = torch.empty(0), torch.empty(0)

        if val_indices:
            X_val, Y_val = X_all[val_indices], Y_all[val_indices]
        else:
            X_val, Y_val = torch.empty(0), torch.empty(0)
            
        if test_indices:
            X_test, Y_test = X_all[test_indices], Y_all[test_indices]
        else:
            X_test, Y_test = torch.empty(0), torch.empty(0)
        
        print(f"\n--- Data Split Summary (By Contract ID) ---")
        print(f"Total Contracts: {N_contracts}")
        print(f"Train Contracts: {len(train_ids)} | Train Samples: {len(X_train)} ")
        print(f"Validation Contracts: {len(val_ids)} | Validation Samples: {len(X_val)}")
        print(f"Test Contracts: {len(test_ids)} | Test Samples: {len(X_test)}")
        
        return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


    # 直接获取最终的loader
    def get_data_loader(self, train_size: float=0.7, val_size: float=0.2, file_list: str=None, batch_size: int=64, num_workers: int=6, tolerate: int=4):
        if file_list is None:
            if self.path is not None:
                file_list = get_all_excel_paths(self.path)
            else:
                file_list = get_all_excel_paths()
        
        print(f"Start loading {len(file_list)} files...")
        for i, f in enumerate(file_list):
            self.load_and_process_file(f, tolerate=tolerate)
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(file_list)} files...")

        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = self.get_split_datasets_by_contract(
            train_ratio=train_size, 
            val_ratio=val_size
        )
        
        train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, valid_loader, test_loader
    
# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 1. 实例化生成器
    generator = OptionTrainingDataGenerator(
        window_size=32, 
        predict_horizon=1, 
        ks_range=(0.9, 1.1),
        min_ratio=0.8
    )
    
    # 2. 获取文件列表
    file_list = get_all_excel_paths() # 使用自动获取函数
    # 或者手动指定列表测试
    # file_list = [
    #    "./miniQMT/datasets/label_train_data/10001234_123456.xlsx",
    # ]
    
    # 如果没有文件，可能会报错，添加一个检查
    if not file_list:
        print("No excel files found or list is empty.")
    else:
        # 3. 循环加载
        # 这里 tolerate 默认为 32 // 2 = 16
        for f in file_list:
            generator.load_and_process_file(f, tolerate=30) # 也可以手动指定 tolerate

        if generator.all_X:
            (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = generator.get_split_datasets_by_contract(
                    train_ratio=0.7, 
                    val_ratio=0.2
                )

            # 验证分割结果
            print(f"\n最终训练集 X Shape: {X_train.shape}")
            print(f"最终测试集 X Shape: {X_test.shape}")

            # 5. 放入 DataLoader
            train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)
    


