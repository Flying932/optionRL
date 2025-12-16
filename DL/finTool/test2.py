import os
import pandas as pd
from typing import List, Tuple

def check_greeks_val_ratio(directory_path: str) -> List[Tuple[str, float]]:
    """
    遍历指定目录下所有以 "_510050.xlsx" 结尾的 Excel 文件，
    计算 'greeks_val' 列的和与总行数的比值。

    :param directory_path: 包含 Excel 文件的目录路径。
    :return: 一个列表，包含所有比值不等于 1.0 的文件名和对应的比值。
    """
    
    # 用于存储结果的列表 (文件名, 比值)
    mismatched_files = []
    
    print(f"--- 🚀 开始处理目录: {directory_path} ---")

    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory_path):
        
        # 筛选条件 1: 文件名必须以 "_510050.xlsx" 结尾
        if filename.endswith("_510050.xlsx"):
            
            full_file_path = os.path.join(directory_path, filename)
            
            # 筛选条件 2: 确保它是文件，而不是目录
            if not os.path.isfile(full_file_path):
                continue
                
            print(f"✅ 正在处理文件: {filename}")
            
            try:
                # 使用 pandas 读取 Excel 文件
                # header=0 表示第一行是列头
                df = pd.read_excel(full_file_path, header=0)
                
                # 检查 'greeks_val' 列是否存在
                if 'greeks_valid' not in df.columns:
                    print(f"⚠️ 警告: 文件 {filename} 中缺少 'greeks_valid' 列，跳过。")
                    continue
                
                # 1. 计算总行数
                total_rows = len(df)
                
                # 如果文件为空，则跳过
                if total_rows == 0:
                    print(f"ℹ️ 文件 {filename} 为空，跳过。")
                    continue
                    
                # 2. 计算 'greeks_val' 列的和
                # 使用 .sum() 计算数值列的总和
                greeks_sum = df['greeks_valid'].sum()
                
                # 3. 计算比值
                ratio = greeks_sum / total_rows
                
                print(f"   总行数: {total_rows}, greeks_valid 之和: {greeks_sum}, 比值: {ratio:.4f}")
                
                # 4. 检查比值是否不等于 1.0
                # 由于浮点数精度问题，我们使用一个小的容忍度 (epsilon) 来比较
                epsilon = 1e-9
                if abs(ratio - 1.0) > epsilon:
                    mismatched_files.append((filename, ratio))
                
            except Exception as e:
                print(f"❌ 处理文件 {filename} 时发生错误: {e}")
                
    print("--- ✅ 处理完成 ---")
    return mismatched_files

# --- 示例用法 ---
# 假设你的 Excel 文件都放在名为 'data_files' 的文件夹中
# 请将下面的路径替换为你实际的文件夹路径
data_directory = './miniQMT/datasets/all_label_data'

# 确保文件夹存在，用于测试
if not os.path.exists(data_directory):
    print(f"创建测试目录: {data_directory}")
    os.makedirs(data_directory)

# 调用函数并获取结果
mismatched_results = check_greeks_val_ratio(data_directory)

## 输出最终结果
if mismatched_results:
    print("\n--- 🚨 结果不等于 1.0 的文件列表 (greeks_valid 之和 / 总行数) ---")
    res = []
    for filename, ratio in mismatched_results:
        print(f"文件名: {filename} | 比值: {ratio:.4f}")

        if ratio != 1.0:
            res.append((filename, ratio))
else:
    print("\n--- 🎉 所有符合要求的文件，其 'greeks_val' 比值均等于 1.0。---")

res_sort = sorted(res, key=lambda x: x[1])
result = []
for name, ratio in res_sort:
    if ratio < 0.9:
        result.append(name[0: 8])
print(0 / 0)