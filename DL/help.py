import torch
import pickle
import io

# ==========================================
# 1. 必须确保类定义存在
# ==========================================
# (注意：运行这段代码时，你的 RunningMeanStd 和 Normalization 类必须在当前文件中可见)
# 如果你已经在当前文件里定义了这两个类，或者 import 进来了，这里就不用管。
# 如果没有，请确保代码能找到它们，否则 pickle 会报 AttributeError。

# ==========================================
# 2. 自定义解包器 (救火队员)
# ==========================================
class LegacyGPU_Unpickler(pickle.Unpickler):
    """
    专门用于读取用 pickle.dump 保存的、含有 GPU Tensor 的文件。
    它会拦截 torch 的存储加载，强制重定向到 CPU。
    """
    def find_class(self, module, name):
        # 拦截 PyTorch 内部存储的加载调用
        if module == 'torch.storage' and name == '_load_from_bytes':
            # 返回一个强制 map_location='cpu' 的加载函数
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        
        # 其他正常的类（如 RunningMeanStd），交给父类处理
        return super().find_class(module, name)

# ==========================================
# 3. 读取函数
# ==========================================
def read_old_gpu_norm(filepath: str = './miniQMT/DL/checkout/norm.pkl'):
    print(f"正在尝试以 CPU 模式加载旧版 GPU 文件: {filepath} ...")
    
    with open(filepath, 'rb') as f:
        # 使用我们自定义的 Unpickler，而不是 pickle.load
        try:
            loaded_data = LegacyGPU_Unpickler(f).load()
            print("加载成功！")
        except Exception as e:
            print(f"加载失败，请检查类定义是否一致。错误信息: {e}")
            raise e

    return loaded_data['reward_norm'], loaded_data['state_norm']

# ==========================================
# 4. (可选) 转换脚本：读取旧的 -> 存成新的
# ==========================================
if __name__ == "__main__":
    # 假设你想把旧格式转换成新格式（torch.save），方便以后读取
    try:
        # 1. 读取旧数据
        r_norm, s_norm = read_old_gpu_norm('./miniQMT/DL/checkout/norm.pkl')
        
        # 2. 构造数据字典
        new_data = {
            'reward_norm': r_norm,
            'state_norm': s_norm,
        }
        
        # 3. 用正确的方式重新保存（以后加载就不用这么麻烦了）
        new_path = './miniQMT/DL/checkout/norm_fixed.pkl'
        torch.save(new_data, new_path)
        print(f"已转换为标准 PyTorch 格式并保存至: {new_path}")
        print("以后请使用 torch.load(..., map_location='cpu') 读取这个新文件。")
        
    except FileNotFoundError:
        print("找不到文件，请检查路径。")
    except Exception as e:
        print(f"发生错误: {e}")