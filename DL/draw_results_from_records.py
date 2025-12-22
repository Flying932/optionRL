import re
import pandas as pd
import matplotlib.pyplot as plt
import os
# 修改这里
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP'] 
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

def parse_ppo_logs(file_path):
    """
    解析 PPO 训练日志文件，提取 Epoch 数据和 Task Performance Report。
    """
    if not os.path.exists(file_path):
        print(f"Error: 文件 {file_path} 不存在。")
        return pd.DataFrame(), pd.DataFrame()

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 适配修改后的 Epoch 数据格式
    # 包含了 EV 匹配组，并调整了后续匹配组的索引
    epoch_pattern = re.compile(
        r"\[Epoch (\d+) / \d+\].*?"                  # Group 1: Epoch
        r"Reward:([-0-9.]+).*?"                      # Group 2: Reward
        r"EV:([-0-9.]+).*?"                          # Group 3: EV (新增)
        r"Market_value:([-0-9.]+).*?\n"              # Group 4: Market Value
        r"Sharpe Ratio -> Mean: ([-0-9.]+).*?"       # Group 5: SR Mean
        r"Max: ([-0-9.]+).*?"                        # Group 6: SR Max
        r"Min: ([-0-9.]+).*?Ratio: ([-0-9.]+)\n"     # Group 7: SR Min, Group 8: Ratio
        r"loss=([-0-9.]+) kl=([-0-9.]+) \| "         # Group 9: Loss, Group 10: KL
        r"act\(H/L/S/C\)=([-0-9./]+) \| "            # Group 11: Action Ratios
        r"entropy=([-0-9.]+)",                       # Group 12: Entropy
        re.DOTALL
    )

    # 2. 解析 Task Performance Report
    report_pattern = re.compile(
        r"--- 📊 Task Performance Report \(Epoch (\d+)\) ---\n"
        r"✅ \[Golden\] .*? 数量: (\d+).*?\n"
        r"🚀 \[Gambling\] .*? 数量: (\d+).*?\n"
        r"📉 \[Failing\] .*? 数量: (\d+)",
        re.DOTALL
    )

    epochs = []
    for match in epoch_pattern.finditer(content):
        d = {
            'epoch': int(match.group(1)),
            'reward': float(match.group(2)),
            'ev': float(match.group(3)),            # 新增 EV 解析
            'market_val': float(match.group(4)),
            'sr_mean': float(match.group(5)),
            'sr_max': float(match.group(6)),
            'sr_min': float(match.group(7)),
            'right_sr_ratio': float(match.group(8)),
            'loss': float(match.group(9)),
            'kl': float(match.group(10)),
            'entropy': float(match.group(12))       # 索引变为 12
        }
        # 解析动作比例 H/L/S/C (索引变为 11)
        act_ratios = match.group(11).split('/')
        d['hold_ratio'] = float(act_ratios[0])
        d['long_ratio'] = float(act_ratios[1])
        d['short_ratio'] = float(act_ratios[2])
        d['close_ratio'] = float(act_ratios[3])
        epochs.append(d)

    reports = []
    for match in report_pattern.finditer(content):
        reports.append({
            'epoch': int(match.group(1)),
            'golden_cnt': int(match.group(2)),
            'gambling_cnt': int(match.group(3)),
            'failing_cnt': int(match.group(4))
        })

    return pd.DataFrame(epochs), pd.DataFrame(reports)

def plot_all_results(df_epochs, df_reports, save_dir='./miniQMT/DL/results'):
    """
    完全英文化的绘图函数，适配 Ubuntu 服务器环境，规避中文字体乱码问题。
    """
    if df_epochs.empty:
        print("Error: No valid Epoch data found to plot.")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 关键修改：不再指定 SimHei，使用系统通用的默认字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial'] 
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示

    # --- Fig 1: Performance Overview ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10))
    metrics1 = [
        ('reward', 'Reward Mean'), 
        ('ev', 'Explained Variance (EV)'), 
        ('sr_mean', 'Sharpe Ratio Mean'), 
        ('market_val', 'Average Market Value')
    ]
    for i, (col, title) in enumerate(metrics1):
        ax = axes1[i // 2, i % 2]
        ax.plot(df_epochs['epoch'], df_epochs[col], color='tab:blue', linewidth=1.5)
        if col == 'ev':
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.set_ylim(-1.1, 1.1)
        ax.set_title(title, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/ppo_summary.png')

    # --- Fig 2: Training Stability ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    metrics2 = [
        ('loss', 'Total Loss'), 
        ('kl', 'KL Divergence'), 
        ('entropy', 'Policy Entropy')
    ]
    colors = ['tab:red', 'tab:green', 'tab:orange']
    for i, (col, title) in enumerate(metrics2):
        axes2[i].plot(df_epochs['epoch'], df_epochs[col], color=colors[i], linewidth=1.5)
        axes2[i].set_title(title, fontsize=14)
        axes2[i].grid(True, linestyle='--', alpha=0.6)
        axes2[i].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/ppo_stability.png')

    # --- Fig 3: Action Distribution ---
    plt.figure(figsize=(12, 6))
    actions = ['hold_ratio', 'long_ratio', 'short_ratio', 'close_ratio']
    labels = ['Hold', 'Long', 'Short', 'Close']
    for act, lab in zip(actions, labels):
        plt.plot(df_epochs['epoch'], df_epochs[act], label=lab, linewidth=2)
    plt.title('Action Selection Ratios Over Time', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'{save_dir}/ppo_actions.png')

    # --- Fig 4: Task Quality Distribution ---
    if not df_reports.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(df_reports['epoch'], df_reports['golden_cnt'], label='Golden Tasks (SR 1.0-2.5)', marker='o', color='green')
        plt.plot(df_reports['epoch'], df_reports['gambling_cnt'], label='Gambling Tasks (SR >3.0)', marker='x', color='red')
        plt.plot(df_reports['epoch'], df_reports['failing_cnt'], label='Failing Tasks (SR <0.5)', marker='s', color='grey')
        plt.title('Task Quality Distribution Trends', fontsize=16)
        plt.xlabel('Epoch')
        plt.ylabel('Task Count')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f'{save_dir}/ppo_task_quality.png')

    print(f"Success: All plots saved to {save_dir}")

def plot_all_results_en(df_epochs, df_reports, save_dir='./miniQMT/DL/results'):
    """
    绘制并保存所有的训练分析图。
    """
    if df_epochs.empty:
        print("未找到有效的 Epoch 数据。")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False

    # --- 图 1: 收益与表现概览 (增加 EV 展示) ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10))
    metrics1 = [
        ('reward', '奖励均值 (Reward)'), 
        ('ev', '解释方差 (Explained Variance)'), # 替换为 EV，观察 Critic 质量
        ('sr_mean', '夏普比率均值 (Sharpe Mean)'), 
        ('market_val', '平均市值 (Market Value)')
    ]
    for i, (col, title) in enumerate(metrics1):
        ax = axes1[i // 2, i % 2]
        ax.plot(df_epochs['epoch'], df_epochs[col], color='tab:blue', linewidth=1.5)
        if col == 'ev':
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5) # EV=0 参考线
            ax.set_ylim(-1.1, 1.1) # EV 通常在 -1 到 1
        ax.set_title(title, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/ppo_summary.png')

    # --- 图 2: 训练稳定性指标 ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    metrics2 = [('loss', '总损失 (Total Loss)'), ('kl', 'KL 散度 (KL Divergence)'), ('entropy', '策略熵 (Entropy)')]
    colors = ['tab:red', 'tab:green', 'tab:orange']
    for i, (col, title) in enumerate(metrics2):
        axes2[i].plot(df_epochs['epoch'], df_epochs[col], color=colors[i], linewidth=1.5)
        axes2[i].set_title(title, fontsize=14)
        axes2[i].grid(True, linestyle='--', alpha=0.6)
        axes2[i].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/ppo_stability.png')

    # --- 图 3: 动作分布演变 ---
    plt.figure(figsize=(12, 6))
    actions = ['hold_ratio', 'long_ratio', 'short_ratio', 'close_ratio']
    labels = ['观望 (Hold)', '做多 (Long)', '做空 (Short)', '平仓 (Close)']
    for act, lab in zip(actions, labels):
        plt.plot(df_epochs['epoch'], df_epochs[act], label=lab, linewidth=2)
    plt.title('动作选择比例演变趋势', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('比例')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'{save_dir}/ppo_actions.png')

    # --- 图 4: 任务质量分布趋势 ---
    if not df_reports.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(df_reports['epoch'], df_reports['golden_cnt'], label='黄金任务 (1.0-2.5)', marker='o', color='green')
        plt.plot(df_reports['epoch'], df_reports['gambling_cnt'], label='赌博任务 (>3.0)', marker='x', color='red')
        plt.plot(df_reports['epoch'], df_reports['failing_cnt'], label='失败任务 (<0.5)', marker='s', color='grey')
        plt.title('期权组合任务质量分布趋势', fontsize=16)
        plt.xlabel('Epoch')
        plt.ylabel('任务数量')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f'{save_dir}/ppo_task_quality.png')

    print(f"所有分析图表已保存至: {save_dir}")

# --- 使用示例 ---
LOG_FILE = './miniQMT/DL/results/PPO_records.txt' 
df_e, df_r = parse_ppo_logs(LOG_FILE)
plot_all_results(df_e, df_r)