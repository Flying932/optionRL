import numpy as np
import math
from scipy.stats import norm
# 导入 brentq 进行更稳定的隐含波动率求解
from scipy.optimize import brentq 

class BS:
    """
    Black-Scholes 期权定价模型与希腊字母计算类。
    包含了针对深度实值/虚值期权和零时间价值的数值稳定性增强。
    """
    def __init__(self):
        # 初始化所有核心参数
        self.S, self.K, self.T, self.r, self.op_type, self.q = [None] * 6
        # d1/d2 参数
        self.d1, self.d2 = None, None
        # 市场期权价格
        self.option_price = None
        # 隐含波动率 (IV) 结果
        self.sigma = None

    # 设置基本参数
    def set_parameters(self, S, K, T, r, option_price, op_type='call', q=0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.option_price = option_price
        self.op_type = op_type.lower()
        self.q = q

    # Black-Scholes 定价公式
    def bs_price(self, S, K, T, r, sigma, option_type, q: float=0):
        # 处理 T=0 的情况：返回内在价值
        if T <= 1e-10: 
            return max(0, S - K) if option_type == 'call' else max(0, K - S)
        
        # 处理 sigma=0 的情况：返回内在价值的折现
        if sigma <= 1e-10:
            # 计算折现后的内在价值
            if option_type == 'call':
                 return np.exp(-r * T) * max(0, S * np.exp((r - q) * T) - K)
            else: # put
                 return np.exp(-r * T) * max(0, K - S * np.exp((r - q) * T))


        # 正常BS公式计算
        # 确保 sigma * np.sqrt(T) 不为零
        if sigma * np.sqrt(T) < 1e-10: 
            d1 = 0
            d2 = 0
        else:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        # put
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    # Vega 函数 (用于计算希腊字母 Vega)
    def vega_greeks(self):
        # 在深度实值/虚值时，d1和d2可能非常大，N'(d1)会趋近于0，因此Vega趋近于0
        if self.sigma is None or self.sigma <= 1e-10 or self.T <= 1e-10:
             return 0
             
        d1 = self.d1 # 使用 cal_d1_d2 已经计算出的 d1
        # Vega对Call和Put都一样
        return self.S * np.exp(-self.q * self.T) * math.sqrt(self.T) * norm.pdf(d1)

    # 求解隐含波动率 (IV) - 此函数在本例中被 TARGET_IV 跳过，但保留供完整性
    def implied_volatility(self):
        """
        使用 Brentq 算法求解隐含波动率 (IV)。
        相比牛顿法，它更稳定，且不需要 Vega (导数) 函数。
        """
        def difference(sigma):
            # 目标函数：BS价格 - 市场价格
            return self.bs_price(self.S, self.K, self.T, self.r, sigma, self.op_type, self.q) - self.option_price
        
        # 1. 检查市场价格是否合理 (高于或等于内在价值)
        # 注意：这里使用 1e-10 而不是 0 来避免浮点误差
        price_at_zero_vol = self.bs_price(self.S, self.K, self.T, self.r, 1e-10, self.op_type, self.q)
        
        # 市场价必须大于或等于最小可能价格 (零波动率价格)
        if self.option_price < price_at_zero_vol - 1e-8: 
            self.sigma = 0
            return

        # 2. 定义搜索边界（IV 搜索空间）
        try:
            low_vol = 1e-5 # 更低的边界
            high_vol = 10.0 # 提高上限以提高稳定性
            
            low_diff = difference(low_vol)
            high_diff = difference(high_vol)
            
            # 确保 f(a) 和 f(b) 符号相反
            if low_diff * high_diff < 0:
                # 存在符号变化，可以使用 brentq 求解
                res = brentq(difference, low_vol, high_vol, xtol=1e-10) # 进一步提高精度
                self.sigma = res
            else:
                self.sigma = 0

        except Exception as e:
            # 如果 brentq 失败，则设为 0
            self.sigma = 0


    def cal_d1_d2(self):
        """
        计算 d1 和 d2，处理 sigma 或 T 接近零的情况。
        """
        # 检查 sigma 是否合理，如果为 0 或 None，则 d1/d2 也为 0
        if self.sigma is None or self.sigma <= 1e-10 or self.T <= 1e-10:
            self.d1, self.d2 = 0, 0
            return
            
        # 确保 log 参数大于 0
        log_term = np.log(self.S / self.K)
        
        d1 = (log_term + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        self.d1, self.d2 = d1, d2
    
    # Delta (正确处理 q 和 Put)
    def delta(self):
        if self.sigma <= 1e-10 or self.T <= 1e-10: # 近似为内在价值的 Delta
            if self.op_type == 'call':
                return np.exp(-self.q * self.T) * (1 if self.S > self.K else 0)
            else:
                 return np.exp(-self.q * self.T) * (-1 if self.S < self.K else 0)

        if self.op_type == 'call':
            return np.exp(-self.q * self.T) * norm.cdf(self.d1)
        # put
        return np.exp(-self.q * self.T) * (norm.cdf(self.d1) - 1)
    
    # Gamma (正确处理 q)
    def gamma(self):
        if self.sigma <= 1e-10: return 0
        # 确保分母不为零
        if self.S * self.sigma * math.sqrt(self.T) < 1e-10: return 0
        return norm.pdf(self.d1) * np.exp(-self.q * self.T) / (self.S * self.sigma * math.sqrt(self.T))
    
    # Theta (正确处理 q, Call/Put) - 返回每年变化值（年化）
    def theta(self):
        # Theta 在 T 接近 0 或 Sigma 接近 0 时的特殊处理
        if self.T <= 1e-10: return 0
        if self.sigma <= 1e-10:
             if self.op_type == 'call':
                # 实值部分的近似
                # 深度实值期权的 Theta 近似于： -r * K * exp(-rT) + q * S * exp(-qT)
                return -self.r * self.K * math.exp(-self.r * self.T) + self.q * self.S * math.exp(-self.q * self.T)
             else: # put
                 return self.r * self.K * math.exp(-self.r * self.T) - self.q * self.S * math.exp(-self.q * self.T)

        # 标准 BS Theta 公式 (年化)
        term1 = - (self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * self.sigma) / (2 * math.sqrt(self.T))
        
        if self.op_type == 'call':
            term2 = self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(self.d2)
            term3 = self.q * self.S * math.exp(-self.q * self.T) * norm.cdf(self.d1)
            return term1 - term2 + term3
        else: # put
            term2 = self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(-self.d2)
            term3 = self.q * self.S * math.exp(-self.q * self.T) * norm.cdf(-self.d1)
            return term1 + term2 - term3 
    
    # Vega (正确处理 q) - 返回绝对值 (价格对波动率 1.0 的变动)
    def vega(self):
        return self.vega_greeks()

    # Rho (正确处理 Call/Put) - 返回绝对值 (价格对利率 1.0 的变动)
    def rho(self):
        if self.op_type == 'call':
            return self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(self.d2)
        # put
        return -self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(-self.d2)
    
    # 修正：新增 q 参数，修正 sigma/iv 逻辑，标准化 Greeks
    def get_greeks(self, S, K, T, r, option_price, op_type='call', input_iv: float=None, q: float=0):
        # 计算出的iv是否有效
        iv_valid = 1

        # 1. 传入 q 参数并设置基本参数
        self.set_parameters(S, K, T, r, option_price, op_type, q)

        # 若能计算出来iv则以计算为主, 若无法计算以传入的history_iv为主计算greeks
        self.implied_volatility()

        if abs(self.sigma) < 1e-10:
            iv_valid = 0
            if input_iv is not None and input_iv > 1e-10:
                self.sigma = input_iv


        if iv_valid:
            calculated_delta, calculated_gamma, calculated_theta, calculated_rho, calculated_vega = None, None, None, None, None
        else:
            calculated_delta, calculated_gamma, calculated_theta, calculated_rho, calculated_vega = 0,0,0,0,0

        # 5. 计算d1和d2 (使用求解或输入的 sigma)
        if calculated_delta is None:
            self.cal_d1_d2()

        # Delta/Gamma/IV: 直接使用计算结果
        results = {
            'iv': self.sigma if iv_valid else 0,
            'delta': calculated_delta if calculated_delta is not None else self.delta(),
            'gamma': calculated_gamma if calculated_gamma is not None else self.gamma(),
            'theta': calculated_theta if calculated_theta is not None else self.theta(),
            'vega': calculated_vega if calculated_vega is not None else self.vega(), 
            'rho': calculated_rho if calculated_rho is not None else self.rho(),
            'iv_valid': iv_valid
        }

        k_list = ['iv', 'delta', 'gamma', 'theta', 'vega', 'rho']
        for k in k_list:
            if abs(results[k]) < 1e-6:
                results[k] = 0

        return results

if __name__ == '__main__':
    # 标的资产价格 S = 3.094
    S = 3.113   
    # 行权价格 K = 2.8
    K = 3.2       
    # 无风险利率 r = 1.3849 / 100
    r = 1.3849 / 100 
    # 股息率 q = 0 (未提供，假设为 0)
    q = 0           
    # 市场期权价格 = 0.2990 (内在价值 0.2940 + 时间价值 0.0050)
    option_price = 0.1037
    # 到期时间 T = 30 / 365 (年)
    T = 29 / 365    
    T = 0.07985159817351598

    S = 1.394
    K = 1.15
    # T = 0.07985159817351598    # 29.1458 / 365
    T = 114 / 365
    r = 1.3849 / 100
    option_price = 0.2471
    h_iv = 0.113


    bs = BS()
    # 传入 TARGET_IV 作为 input_iv 参数，直接计算 
    res = bs.get_greeks(S, K, T, r, option_price, 'call', q=q, input_iv=h_iv)

    # res = bs.get_greeks(1.142, 0.95, 0.36164383561643837, 0.013849, 0.1937, 'call')
 
    print(f"标的价(S)={S}, 行权价(K)={K}, 剩余时间(T)={T*365:.4f}天, 无风险利率(r)={r*100:.4f}%, 市场价={option_price:.4f}")
    
    # 截图目标结果：IV=0.1855, Delta=0.9952, Gamma=0.1139, Theta=-0.0460, Vega=0.0125, Rho=0.2363
    print(f"IV (隐含波动率): {res['iv']:.4f} ")
    print(f"Delta (Delta): {res['delta']:.4f} ")
    print(f"Gamma (Gamma): {res['gamma']:.4f} ")
    print(f"Theta (年化值): {res['theta']:.4f} ")
    # 注意：Vega 是 1% 变动
    print(f"Vega (100%变动): {res['vega']:.4f} ") 
    # Rho 是 100% 变动
    print(f"Rho (100%变动): {res['rho']:.4f} ")
   