import numpy as np
import scipy.stats as si

def d(s, k, r, T, sigma):
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def delta(s, k, r, T, sigma, n):
    """
    认购期权的 n 为 1
    认沽期权的 n 为 -1
    """
    d1 = d(s, k, r, T, sigma)[0]
    delta_val = n * si.norm.cdf(n * d1)
    return delta_val

def gamma(s, k, r, T, sigma):
    d1 = d(s, k, r, T, sigma)[0]
    gamma_val = si.norm.pdf(d1) / (s * sigma * np.sqrt(T))
    return gamma_val

def vega(s, k, r, T, sigma):
    d1 = d(s, k, r, T, sigma)[0]
    vega_val = (s * si.norm.pdf(d1) * np.sqrt(T)) / 100
    return vega_val

def theta(s, k, r, T, sigma, n):
    """
    认购期权的 n 为 1
    认沽期权的 n 为 -1
    """
    d1, d2 = d(s, k, r, T, sigma)
    theta_val = (-1 * (s * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - n * r * k * np.exp(-r * T) * si.norm.cdf(n * d2)) / 365
    return theta_val

g = gamma(3.094, 2.8, 1.8/100, 30/365, 0.1855)
print(f"gamma = {g}")