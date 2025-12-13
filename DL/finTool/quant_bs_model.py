import QuantLib as ql

def bs_greeks(
    S, K, T, r, market_price, 
    option_type: str = 'put',   # 'call' 或 'put'
    q: float = 0.0
):
    """
    返回一个字典，包含：
    - implied_vol  : 隐含波动率
    - delta       : Delta
    - gamma       : Gamma
    - vega        : Vega（对 1% 波动率变化的敏感度）
    - theta       : Theta（一天的时间衰减，年化/365）
    - rho         : Rho（对 1% 利率变化的敏感度）
    - price       : 模型理论价格（用计算出的 IV 重新定价，验证用）
    """
    # ---- 1. 判断期权类型 ----
    opt_type = ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put

    # ---- 2. 构建最轻量 BSM 环境（和你原来一模一样）----
    spot = ql.SimpleQuote(S)
    payoff = ql.PlainVanillaPayoff(opt_type, K)
    exercise = ql.EuropeanExercise(ql.Date.todaysDate() + int(T*365 + 0.5))
    option = ql.VanillaOption(payoff, exercise)

    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(spot),
        ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), q, ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), r, ql.Actual365Fixed())),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.TARGET(), 0.3, ql.Actual365Fixed()))
    )

    engine = ql.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)

    # ---- 3. 先算隐含波动率 ----
    iv = option.impliedVolatility(market_price, process, 1e-10, 1000, 1e-8, 4.0)

    # ---- 4. 把计算出的真实 IV 写回波动率曲线（关键一步！）----
    flat_vol = ql.BlackConstantVol(0, ql.TARGET(), iv, ql.Actual365Fixed())
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(spot),
        ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), q, ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), r, ql.Actual365Fixed())),
        ql.BlackVolTermStructureHandle(flat_vol)
    )
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    # ---- 5. 直接取所有 Greeks（QuantLib 一行一个）----
    return {
        'implied_vol': iv,           # 百分比
        # 'price'      : option.NPV(),       # 理论公平价（应该≈market_price）
        'theta'      : option.thetaPerDay() * 365, 
        'vega'       : option.vega(),   # 原始 vega 是对波动率+1（即+100%）的敏感度，除100才是对1%波动率的敏感度（国内习惯）
        'gamma'      : option.gamma(),     # Γ
        'delta'      : option.delta(),     # Δ
         # 已经是一天的 theta（负数表示时间价值流失）
        'rho'        : option.rho(),    # 对利率+1%（100bp）的敏感度，除100更直观
    }

# ================= 你的看跌期权数据 ==================
S = 1.382
K = 1.0
# T = 0.07985159817351598    # 29.1458 / 365
T = 28 / 365
r = 1.3849 / 100
market_price = 0.0008

greeks = bs_greeks(S, K, T, r, market_price, option_type='put', q=0.0)

# 漂亮打印
for k, v in greeks.items():
    print(f"{k:12s} : {v: .8f}")