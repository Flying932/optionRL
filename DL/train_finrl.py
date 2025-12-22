import os
import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent
import torch
from env_etf_option import OptionStraddleEnv
from Networks import HybridOptionExtractor as HybridFeatureExtractor
from dataclasses import dataclass
from finTool.single_window_account_fast import single_Account
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import sys
from pathlib import Path
from datetime import datetime, timedelta

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

class FinancialMetricsCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(FinancialMetricsCallback, self).__init__(verbose)
        # 用于存储本轮采样中所有步骤的指标
        self.step_buffer = {"sharpe": [], "simple_ann": [], "log_ann": []}

    def _on_step(self) -> bool:
        # 获取所有并行环境的实时 info
        for info in self.locals.get("infos", []):
            if "running_metrics" in info:
                m = info["running_metrics"]
                self.step_buffer["sharpe"].append(m["sharpe"])
                self.step_buffer["simple_ann"].append(m["simple_ann"])
                self.step_buffer["log_ann"].append(m["log_ann"])
        return True

    def _on_rollout_end(self) -> None:
        # 在一个采样周期（如 2048 步）结束时打印均值
        if len(self.step_buffer["sharpe"]) > 0:
            avg_metrics = {k: np.mean(v) for k, v in self.step_buffer.items()}
            
            # 记录到 SB3 的 Logger，使其在控制台表格显示
            self.logger.record("finance/running_sharpe_avg", float(avg_metrics["sharpe"]))
            self.logger.record("finance/ann_return_simple", float(avg_metrics["simple_ann"]))
            self.logger.record("finance/ann_return_log_cont", float(avg_metrics["log_ann"]))
            
            if self.verbose > 0:
                print(f"\n[Running Metrics] Iteration Summary:")
                print(f"Avg Sharpe: {avg_metrics['sharpe']:.4f} | "
                      f"Simple Ann: {avg_metrics['simple_ann']:.2%} | "
                      f"Log Ann: {avg_metrics['log_ann']:.2%}")
            
            # 重置缓冲区
            for k in self.step_buffer: self.step_buffer[k] = []

@dataclass
class Config:
    pretrained_path: str = "./miniQMT/DL/preTrain/weights/preMOE_best_dummy_data_32_4.pth"
    window_size: int = 32
    pre_len: int = 4
    n_variates: int = 13
    d_router: int = 128
    num_workers: int = 1

    # env
    benchmark: str = '510050'
    fee: float = 1.3
    init_capital: float = 100000.0

    # dl
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    # 2. 动态分类采样逻辑
    all_pairs = []
    # 分类桶，目标总数约 20 个
    buckets = {"ITM": [], "ATM": [], "OTM": []}
    target_per_bucket = 20

    # 预加载一个账户用于获取初始标的价格
    temp_account = single_Account(100000, stockList=['510050'])

    # 打乱原始数据顺序，保证采样的随机性
    # df = df.sample(frac=1).reset_index(drop=True)

    for index, row in df.iterrows():
        call, put = row['call'], row['put']
        if call in exclude_list or put in exclude_list:
            continue

        # 时间逻辑
        start_str, expire_str = row['call_open'], row['call_expire']
        start_dt = datetime.strptime(start_str, "%Y%m%d")
        expire_dt = datetime.strptime(expire_str, "%Y%m%d")
        
        if (expire_dt - start_dt).days <= 40:
            continue

        # 计算初始 Moneyness (行权价 / 标的价格)
        # 假设 start_time 为开盘 10:00:00
        start_time_full = start_str + '100000'
        try:
            spot_price = temp_account.getOpenPrice('510050', start_time_full)
            if spot_price <= 0: continue
            
            # 以 Call 的行权价计算
            moneyness = row['call_strike'] / spot_price
            
            # 简单分类：0.98-1.02 为平值，大于 1.05 为虚值，小于 0.95 为实值
            if 0.97 <= moneyness <= 1.03:
                cat = "ATM"
            elif moneyness > 1.03:
                cat = "OTM"
            else:
                cat = "ITM"
            
            # 填桶
            if len(buckets[cat]) < target_per_bucket:
                end_time_full = (expire_dt - timedelta(days=20)).strftime('%Y%m%d') + '150000'
                buckets[cat].append({
                    'call': call, 'put': put,
                    'start_time': start_time_full,
                    'end_time': end_time_full,
                    'steps': int(row['steps']),
                    'init_moneyness': moneyness
                })
        except:
            continue

        # if sum(len(v) for v in buckets.values()) >= 21:
        #     break

    # 合并采样结果
    for cat_list in buckets.values():
        all_pairs.extend(cat_list)

    return all_pairs

def train():
    cfg = Config(
        pretrained_path="./miniQMT/DL/preTrain/weights/preMOE_best_dummy_data_32_4.pth",
        window_size=32,
        pre_len=4,
        n_variates=13,
        d_router=128,
        init_capital=100000,
        num_workers=2)
    
    pairs = get_option_pairs()
    
# --- 修正后的环境向量化包装 ---
    def make_env():
        return OptionStraddleEnv(option_pairs_list=pairs, cfg=cfg)

    # 替代 get_sb3_env 的标准 SB3 方法
    if cfg.num_workers > 1:
        env_train = SubprocVecEnv([make_env for _ in range(cfg.num_workers)])
    else:
        env_train = DummyVecEnv([make_env])

    fin_metrics_cb = FinancialMetricsCallback(verbose=1)

    # 4. 初始化 DRLAgent
    agent = DRLAgent(env=env_train)
    
    policy_kwargs = dict(
        features_extractor_class=HybridFeatureExtractor,
        features_extractor_kwargs=dict(cfg=cfg),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    
    # --- 🔥 关键修正：超参数必须放在 model_kwargs 字典中 ---
    n_steps = 256
    model_kwargs = {
        "learning_rate": 1e-4,
        "n_steps": n_steps,
        "batch_size": 256,
        "ent_coef": 0.01,
        "clip_range": 0.15,
        "gamma": 0.99,
        "gae_lambda": 0.95
    }

    num_epochs = 1
    total_training_steps = num_epochs * n_steps * cfg.num_workers

    #   获取 PPO 模型
    model = agent.get_model(
        model_name="ppo", 
        policy="MultiInputPolicy", # 必须是这个政策！l
        policy_kwargs=policy_kwargs,
        model_kwargs=model_kwargs, # 传递修正后的字典
        verbose=1,
        tensorboard_log="./miniQMT/DL/results"
    )

    
    print(f"🚀 开始训练! 总步数: {total_training_steps} (约 {num_epochs} 个更新周期)")
    print("🚀 启动训练，已开启双口径年化收益监控...")
    model = agent.train_model(
        model=model, 
        tb_log_name="ppo_finrl_fix", 
        total_timesteps=total_training_steps,
        callbacks=[fin_metrics_cb]
    )

    model.save("ppo_option_final_model")
    print("✅ 训练完成，模型已保存。")
    print(0 / 0)

if __name__ == "__main__":
    train()