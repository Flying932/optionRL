import os
import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent
from env_etf_option import OptionStraddleEnv as FinRLOptionEnv
from Networks import HybridFeatureExtractor
from dataclasses import dataclass

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



@dataclass
class Config:
    pretrained_path: str = "./miniQMT/DL/preTrain/weights/preMOE_best.pth"
    window_size: int = 32
    pre_len: int = 4
    n_variates: int = 13
    d_router: int = 128
    init_capital: float = 100000.0
    num_workers: int = 14

def get_option_pairs():
    """
    扫描目录，生成任务清单。这部分逻辑直接复用你 PPO_GPT 结尾的部分。
    """

    # 1. 加载原始数据
    dtype = {
        'call': str, 'put': str,
        'call_strike': float, 'put_strike': float,
        'call_open': str, 'call_expire': str,
    }
    df = pd.read_excel('./miniQMT/datasets/all_label_data/20251213_train.xlsx', dtype=dtype)

    # 排除名单
    exclude_list = [
        '10007347', '10007466', '10007467', '10006436', '10007346', '10006435', '10007465', 
        '10007726', '10007725', '10007724', '10008052', '10007723', '10006434', '10007722', 
        '10008051', '10007345', '10007721', '10007464', '10007344', '10007988', '10006433', 
        '10006820', '10007720', '10007987', '10006746', '10006745', '10007463', '10006432', '10007719'
    ]

    print(0 / 0)


    return

def train():
    cfg = Config()
    pairs = get_option_pairs()
    
    # 1. 包装环境 (FinRL 自动实现 SubprocVecEnv 并行)
    # 这里的 env 是类，不是实例
    agent = DRLAgent(env=None)
    env_train, _ = agent.get_sb3_env(
        env=FinRLOptionEnv, 
        n_envs=cfg.num_workers,
        env_kwargs={'option_pairs': pairs, 'cfg': cfg}
    )
    
    # 2. 策略配置
    policy_kwargs = dict(
        features_extractor_class=HybridFeatureExtractor,
        features_extractor_kwargs=dict(cfg=cfg),
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    
    # 3. 启动 PPO 训练
    model = agent.get_model(
        "ppo", 
        policy_kwargs=policy_kwargs, 
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=4096,
        ent_coef=0.02
    )
    
    agent.train_model(model, tb_log_name="ppo_finrl", total_timesteps=1000000)

if __name__ == "__main__":
    train()