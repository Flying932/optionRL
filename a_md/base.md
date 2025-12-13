---
marp: true
# 可选设置：定义主题或背景
# theme: uncover
math:  katex
---

# 基于 PAB-VarTST 与 Skill-PPO 的期权波动率交易策略研究
## 组会汇报与核心工程思路总结

**汇报人：** 许骏飞 | **日期：** 2025.11.27

---

# Ⅰ. 研究大纲与核心创新点

## Agenda
1.  **工程痛点回顾**：从 PPO 直训 TST 的失败谈起
2.  **创新点一：PAB-VarTST** 特征学习器详解
3.  **创新点二：Skill-PPO** 分层策略与 Greeks 对齐
4.  **数据工程**：期权数据处理与环境设计
5.  **工作进展**、挑战与下一步计划

---

# Ⅰ. 研究背景、问题定位与工程思考

## 核心挑战：特征学习与决策难收敛

| 痛点 | 核心思考 (我的方案) | 解决目标 |
| :--- | :--- | :--- |
| **PPO 信号稀疏且不连续** | **解耦训练**：引入**预训练 (Pre-training)** 和 **InfoNCE 对齐**，将特征学习与稀疏的 RL 信号解耦。 | 提高特征质量，加速 PPO 收敛。 |
| **Transformer 退化**  | **结构先验融合**：结合 **VARMA (局部先验)** 和 **BasisFormer (全局结构)**，避免 Attention 机制失效。 | 鲁棒、可解释、高泛化性的特征。 |
| **高维动作与组合风险** | **分层决策**：使用 **Skill-PPO** 和 **Greeks 对齐**，将“风格选择”与“动作执行”解耦。 | 提高策略可控性和风格稳定性。 |

---

# Ⅱ. 特征学习的探索与迭代：从失败到融合

## 1. 初始尝试：PPO 端到端训练 TST (效果极差)

* **实现方法**：最初尝试直接将 **Transformer (TST)** 或 **BasisFormer** 作为 PPO 的 Feature Extractor（如 `weightPPO_TST.py` 中所示），并与 Actor/Critic 网络一起使用稀疏的 PPO 梯度进行端到端更新。
* **观察结果 (思考)**：
     **PPO 信号噪声**：策略网络尚未发现有效策略时，其梯度信号对特征提取器来说是纯噪声。
     **效果极差**：模型收敛速度极慢，且难以收敛到合理的策略。这验证了在复杂任务中，不能直接依赖稀疏 RL 信号来训练高参数量的特征器。

---
## 2. 工程校正：基准学习 (Pre-training) 的引入

* **目的**：为特征提取器提供一个**强监督/自监督**的预训练基准，使其具备基础的时序分析能力，再接入 PPO 进行微调。
* **具体进展**：
     **VARMAformer 独立训练**：已成功跑通 `VARMAformer` 的 VFE 模块独立训练（如日志所示），并验证了其预测精度（Test MSE/MAE）。
     **基准模型选择**：
         **VARMA-VFE**：用于捕捉短期的 **局部 AR/MA 动态**。
         **BasisFormer**：用于捕捉可解释的 **中长期结构**。

---

# Ⅲ. 创新点一：PAB-VarTST 特征学习器的融合思路

## PAB-VarTST = VARMA + BasisFormer + iTransformer + RL 对齐

* **组件融合的思考 (PAB-VarTST)**
    1.  **VARMA-VFE**：提供**时间轴**上的局部归纳偏置，解决 TST 对短期波动的解释力不足问题。
    2.  **BasisFormer**：提供**全局结构**，利用其可学习基底捕捉时间序列的宏观趋势和周期性。
    3.  **iTransformer 范式**：将希腊值、IV 等**多变量特征**作为 Token。在高维特征空间中，iTransformer 相比传统 TST 更稳定，专门建模**变量间的相关性**。

---
* **收益对齐的对比学习 (RL-weighted InfoNCE)**
     **目的**：弥合预训练目标与最终 RL 收益目标之间的差距。
     **方法**：在预训练中，使用 PPO **优势函数 (Advantage)** 对 InfoNCE Loss 进行加权。
     **作用**：让特征提取器更加关注那些 “被策略证实为高价值”的状态，实现 **Policy-Adaptive** 的特征学习。

---

# Ⅳ. 创新点二：Skill-PPO 策略与数据生成

## 1. Skill-PPO 的分层决策架构

* **动机**：在高风险的期权组合交易中，需要策略具备**多风格切换**和**风险控制**能力。
* **架构思路 (CDT-PPO-MOE)**：
    1.  **预训练 (CDT)**：使用离线交易轨迹数据进行 **Causal Decision Transformer** 预训练，利用 Return-to-Go 信号，为 PPO 提供一个高质量的**策略初始化**。
    2.  **分层执行 (MoE)**：
         **Manager**：根据市场特征动态选择**交易技能/专家 (Expert)** 。
         **Experts (Workers)**：每个 Expert 学习一种特定的 Greeks 暴露风格（如 Long Vega Expert, Delta Neutral Expert）。
    3.  **Greeks 对齐**：通过奖励设计，显式地将 Expert 的行为与其预期的希腊值暴露（如 $\Delta, \Gamma, \Theta, \nu$）对齐，确保策略的可解释性。

---
## 2. 数据的生成与处理 (Data Engineering)

* **数据挑战**：期权合约的**生命周期短**、**流动性不连续**（导致 IV 缺失）。
* **解决方案**：
    1.  **抽象合约**：不追踪固定合约 ID，而是动态追踪满足特定条件（如平值、$\text{TTM} > 10$ 天）的合约组合。
    2.  **特征处理**：使用 **LOCF** 填充瞬时缺失值，并使用 **相对化特征**（如 $\text{K/S}$，$\text{IV}_{rel}$）代替绝对价格，提高模型泛化性。
    3.  **RL 环境封装**：已实现 **`windowEnv`**（多窗口/多组合）和 **`windowAccount`**，为 PPO 训练提供了一个高效、可控的组合交易环境。

---

# Ⅴ. 工作进展、挑战与下一步计划

## 1. 已完成的工作 (Implemented)

* **基础 RL 框架**：PPO 核心算法 (`weightPPO`)，支持 `VARMA`/`BasisFormer` 特征模式。
* **VARMAformer与训练、BasisFormer 框架搭建**：独立训练和加载测试通过，可作为 PAB-VarTST 的基础组件。
* **数据环境**：`windowEnv` / `windowAccount` 已实现，支持期权组合的动态跟踪和多时间窗口状态输入。

---
## 2. 当前挑战 (Challenges)

* **特征联训**：将预训练的 VARMAformer/BasisFormer 融合为 **PAB-VarTST** 并接入 PPO **进行联合训练**（实现 RL-weighted InfoNCE）。这是目前最大的工程挑战。
* **Skill 奖励设计**：如何设计有效的相对奖励或风控软约束，确保 MoE Experts 能够学习到期望的 **Greeks 风格**。

## 3. 下一步计划 (Next Steps)

1.  **PAB-VarTST 融合与基线**：完成 **BasisFormer** 模块实现，构建 **PAB-VarTST**，并进行首次 PPO 端到端训练，对比纯 PPO/VARMA 基线的收益和稳定性。
2.  **Skill-PPO 雏形**：实现 **Manager-Expert** 路由层和基础的 **Greeks-Aligned Reward Shaping**，验证分层架构的有效性。
3.  **消融实验准备**：构建消融实验所需的训练脚本和数据切片，包括“无预训练” vs “完整 PAB-VarTST” 等消融实验。