基于 PAB-VarTST 与 Skill-PPO 的期权波动率交易策略研究

硕士研究生学位论文开题报告

汇报人： 许骏飞 | 日期： 2025.11.27

Ⅰ. 研究背景、问题定位与工程思考

1.1 选题原因与研究意义

期权交易因其高杠杆、非线性收益以及多元风险敞口（Greeks）的特性，是金融市场中最复杂的交易领域之一。传统的量化模型难以有效处理期权价格的高维时序特征和稀疏的收益信号。

本研究旨在构建一个可泛化、可解释的深度强化学习（DRL）框架，解决以下核心工程挑战：

特征与决策解耦：解决 PPO 稀疏奖励信号难以有效训练高参数量时序特征提取器的问题。

策略可控性：解决端到端 DRL 策略不可解释、风格不稳定的问题，尤其在风险管理至关重要的期权交易中。

多组合泛化：实现一套能够灵活处理变长或固定数量期权组合的通用策略，提升模型的实用性。

1.2 研究现状与不足

领域

现有研究现状

存在的主要不足

时序特征提取

Transformer/iTransformer/BasisFormer：利用自注意力机制捕捉序列依赖。

缺乏对金融序列多体制、高噪声的适应性，易退化，泛化性不足。

深度强化学习

PPO/SAC/CDT：用于序列决策。Skill-PPO：用于分层和技能学习。

奖励稀疏时收敛困难；缺乏将决策与金融风险（Greeks）显式对齐的机制。

期权策略

大多基于 Black-Scholes 或 Garch 模型，或直接使用简单 MLP/LSTM。

难以处理海量历史窗口输入和多变量特征间的非线性关系。

Ⅱ. 核心方法论与训练框架

2.1 PAB-VarTST 特征学习器

PAB-VarTST（Policy-Adaptive Basis with VARMA + Inverted TST）旨在通过异构模型融合和策略对齐，提取高鲁棒性的市场状态表示。

特征融合架构：采用 MoE (Mixture of Experts) 思路，融合多种归纳偏置：

VARMA-VFE：捕获短期局部统计依赖（如 AR/MA 动态）。

BasisFormer：利用可学习基底捕捉中长期的趋势和结构。

iTransformer 范式：通过转置输入，聚焦多变量特征（Greeks, IV, TTM）之间的相关性。

策略自适应对齐：通过 RL-weighted InfoNCE Loss 对预训练损失进行加权，其中权重来源于 PPO 的优势函数 (Advantage)。这强制特征提取器将注意力集中在那些能够带来高价值收益的状态上，弥合预训练目标与 RL 目标之间的差距。

2.2 Skill-PPO 分层决策架构

为了提高策略的可控性和可解释性，采用 Skill-PPO 分层架构：

策略初始化 (CDT Pre-training)：首先利用历史交易数据，通过 Causal Decision Transformer (CDT) 进行策略初始化，加速 PPO 的冷启动过程。

分层执行 (Manager-Expert)：

Manager (管理者)：负责根据市场状态选择最合适的 Expert (专家技能)。

Experts (工人)：每个专家负责执行一种预设的交易风格。

Greeks 对齐奖励 (Reward Shaping)：在 PPO 奖励中，显式地加入对 Greeks 暴露的软约束。例如，惩罚 $\Delta$ 中性专家出现高 $\Delta$ 暴露的行为，确保专家决策与其预期的风格（如 Long Vega 或 Short Gamma）保持一致，从而提高策略的可解释性。

Ⅲ. 状态融合与工程实现

3.1 状态融合架构设计

我们遵循分层处理、高维缩减、低维保留的原则，设计最终的 PPO 状态 $\mathbf{S}_{\text{Final}}$：

$$\mathbf{S}_{\text{Final}} = [\mathbf{S}_{\text{Adapter}} \oplus \mathbf{S}_{\text{Portfolio}} \oplus \mathbf{S}_{\text{Account}}]$$

信息源

维度/特性

处理方式

目的

期权基本信息 ($\mathbf{O}_{\text{MoE}}$)

高维（如 3732 维），时序抽象

$\mathbf{O}_{\text{MoE}} \xrightarrow{\text{Adapter (MLP)}} \mathbf{S}_{\text{Adapter}}$ (降维)

压缩冗余，提取市场认知。

期权交易信息 ($\mathbf{S}_{\text{Portfolio}}$)

低维，高语义（如持仓方向、数量）

直接拼接（不缩减）

保留关键的局部决策信息，避免语义丢失。

账户整体信息 ($\mathbf{S}_{\text{Account}}$)

极低维（如可用资金比例、冻结资金、历史收益等）

直接拼接（不缩减）

提供全局风险和流动性约束。

工程考量：将高维特征缩减放在 MoE Adapter 阶段完成，保证 PPO 网络接收的 $\mathbf{S}_{\text{Portfolio}}$ 和 $\mathbf{S}_{\text{Account}}$ 具有清晰语义，加速 PPO 收敛。

3.2 多组合处理策略（聚焦当前）

考虑到信号微弱和收敛难度，我们决定采取渐进式扩展策略：

核心训练：PPO 策略 只处理单个期权组合。奖励信号为该组合产生的局部收益，信号强且清晰。

初步扩展：部署时，将总账户资金平分给 $N$ 个策略副本，实现策略复用。

高级扩展：未来通过一个独立的、轻量级 MLP 分配网络 (Allocator) 来学习动态资金分配，Allocator 的奖励信号为账户总收益。

这种方式有效避免了在 PPO 训练初期使用复杂网络和微弱信号导致的收敛失败。

Ⅵ. 工作进展与下一步计划

6.1 已完成的工作

基础 RL 框架：基于 PPO 算法，实现了支持 MoE/Adapter 模式的训练框架。

环境类封装：完成了 single_window_account 类，实现了单期权组合的账户逻辑，支持设置、更改和修改期权组合。

特征预训练框架：搭建了 VARMAformer 的 VFE 模块和 BasisFormer 的基本结构，并完成了独立预训练测试。

数据处理：实现了期权数据的相对化处理和滚动抽象机制。

6.2 当前挑战与下一步计划

挑战

下一步计划

PAB-VarTST 联训

完成 MoE Router 的实现，将预训练专家权重接入 PPO，实现 RL-weighted InfoNCE 进行联合微调。

Skill 奖励设计

设计并测试基于 Greeks 暴露的 Reward Shaping 函数，验证 Experts 风格对