#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np

class CorrelatedRandomNumberGenerator:
    def __init__(self, num_steps, num_assets, cov_matrix, seed=None):
        self.num_steps = num_steps  # 时间步数
        self.num_assets = num_assets  # 资产数量
        self.cov_matrix = cov_matrix  # 协方差矩阵
        self.seed = seed  # 随机数种子
        if seed is not None:
            np.random.seed(seed)  # 设置种子，使得每次运行结果一致

    def generate(self):
        """ 生成相关的标准正态分布随机数，形状为 (num_steps, num_assets) """
        # 生成独立的标准正态分布随机数
        Z = np.random.normal(0, 1, (self.num_steps, self.num_assets))
        
        # Cholesky分解协方差矩阵
        L = np.linalg.cholesky(self.cov_matrix)  # 获取协方差矩阵的Cholesky分解

        # 将独立的标准正态分布随机数转化为相关的随机数
        correlated_Z = np.dot(Z, L.T)  # 相关的标准正态分布随机数

        return correlated_Z


# In[31]:


# 初始化参数
num_steps = 252  # 时间步数（252个交易日）
num_assets = 2   # 资产数量
cov_matrix = np.array([
    [0.02, 0.01],
    [0.01, 0.03]

])  # 协方差矩阵
seed = 42        # 随机数种子

# 创建有相关性的随机数生成器
correlated_random_number_generator = CorrelatedRandomNumberGenerator(num_steps=num_steps, num_assets=num_assets, cov_matrix=cov_matrix, seed=seed)

# 生成相关的随机数
correlated_Z = correlated_random_number_generator.generate()
print(correlated_Z.shape)  


# In[27]:


class PathGenerator:
    def __init__(self, S0, r, sigma, T, dt, num_assets, random_number_generator):
        self.S0 = S0  
        self.r = r   
        self.sigma = sigma 
        self.T = T    
        self.dt = dt  
        self.num_steps = int(T / dt) 
        self.num_assets = num_assets
        self.random_number_generator = random_number_generator

    def generate_paths(self):
        """ Time-first: generates prices for all assets in time steps """
        Z = self.random_number_generator.generate() 
        paths = np.zeros((self.num_steps + 1, self.num_assets))
        paths[0] = self.S0  
        
        for n in range(1, self.num_steps + 1):
            paths[n] = paths[n-1] * np.exp(
                (self.r - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * Z[n-1]
            )
        return paths

   


# In[32]:


import matplotlib.pyplot as plt

path_generator = PathGenerator(S0, r, sigma, T, dt, num_assets, correlated_random_number_generator)

# 生成路径
paths = path_generator.generate_paths()

# 绘制资产价格路径
plt.figure(figsize=(10, 6))
for i in range(num_assets):
    plt.plot(paths[:, i], label=f'Asset {i+1}')
plt.title('Simulated Asset Price Paths')
plt.xlabel('Time Step (Days)')
plt.ylabel('Asset Price')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[99]:


class RangeAccrualPayoff:
    def __init__(self, S0, K, T, dt, early_termination_times, c1, c2, B_KO):
        self.S0 = S0  
        self.K = K   
        self.T = T   
        self.dt = dt  
        self.early_termination_times = early_termination_times  
        self.c1 = c1  # 未触发KO时的低息票
        self.c2 = c2  # 触发KO后的高息票（需根据条款调整）
        self.B_KO = B_KO  # KO障碍值（可以是列表或标量）

    def compute(self, paths):
        payoffs = []
        knocked_out = False  # 标记是否已触发KO事件

        for t in range(1, len(self.early_termination_times)):
            if knocked_out:
                payoffs.append(0.0)
                continue

            t_k = self.early_termination_times[t]
            t_k_1 = self.early_termination_times[t-1]
            
            # 检查当前时间点t_k是否触发KO
            S_tk = paths[t_k]  # 假设paths的索引对应时间点
            performance_tk = S_tk / self.S0  # 当前时间点的表现
            
            # 获取当前KO障碍（假设B_KO为列表，与时间点一一对应）
            B_KO_k = self.B_KO[t] if isinstance(self.B_KO, list) else self.B_KO
            
            if np.any(performance_tk >= B_KO_k):
                knocked_out = True  # 触发KO事件
            
            # 计算当前区间的支付
            path_segment = paths[t_k_1:t_k]
            worst_performance = np.min(path_segment / self.S0, axis=1)
            A_k = np.sum(worst_performance >= self.K)
            C_k = (1 - A_k/(t_k - t_k_1)) * self.c1 + (A_k/(t_k - t_k_1)) * self.c2
            
            payoffs.append(C_k)
        
        return payoffs


# In[98]:


S0 = 100  # 初始资产价格
r = 0.05  # 无风险利率
sigma = 0.2  # 波动率
T = 1  # 到期时间（1年）
dt = 1/252  # 每天为时间步长
num_paths = 10000  # 模拟路径数量
num_assets = 2 
L = 1
early_termination_times = [0, 50, 100, 150, 200, 252]  # 早期终止时间点（以天为单位）
c1 = 1  # 低支付
c2 = 2  # 高支付
B_KO = 1.02

# 随机数生成器和路径生成器实例
cov_matrix = np.array([
    [0.02, 0.01], 
    [0.01, 0.03]
])  # 协方差矩阵

random_number_generator = CorrelatedRandomNumberGenerator(
    num_steps=252, num_assets=num_assets, cov_matrix=cov_matrix, seed=42)

path_generator = PathGenerator(S0, r, sigma, T, dt, num_assets, correlated_random_number_generator)

# 生成路径
paths = path_generator.generate_paths()

payoff = RangeAccrualPayoff(S0, K=1.01, T=T, dt=dt, early_termination_times=early_termination_times, c1=c1, c2=c2, B_KO=B_KO)
print(payoff.compute(paths))


# In[85]:


class MonteCarloPricer:
    def __init__(self, path_generator, payoff, num_paths, delta_S=1.0):
        self.path_generator = path_generator  # 路径生成器
        self.payoff = payoff  # 支付计算器
        self.num_paths = num_paths  # 模拟路径数
        self.delta_S = delta_S

        # 从 path_generator 和 payoff 中提取共享参数
        self.S0 = self.path_generator.S0
        self.r = self.path_generator.r
        self.sigma = self.path_generator.sigma
        self.T = self.path_generator.T
        self.dt = self.path_generator.dt
        self.early_termination_times = self.payoff.early_termination_times
        
    def price(self):
        """ 使用蒙特卡洛方法计算期权的现值 """
        payoffs = []  # 存储所有路径的支付
        
        for _ in range(self.num_paths):
            paths = self.path_generator.generate_paths()
            path_payoffs = self.payoff.compute(paths)  # 每个时间点的支付
            
            # 计算每个时间点的折现现值并累加
            discounted_payoffs = []
            for t, C_k in enumerate(path_payoffs):
                # 对每个支付使用不同的折现因子
                t_k = self.early_termination_times[t + 1]  # 当前时间点
                discount_factor = np.exp(-self.path_generator.r * (t_k / 252))  # 折现因子
                discounted_payoffs.append(C_k * discount_factor)  # 折现后的支付
            
            payoffs.append(np.sum(discounted_payoffs))  # 累加每条路径的支付
        
        pv = np.mean(payoffs)
        return pv
    
    def compute_greeks(self, S0):
        """ 计算 Delta 和 Gamma """
        original_S0 = self.path_generator.S0
        
        pv_plus = self.compute_pv(S0 + self.delta_S)
        pv_minus = self.compute_pv(S0 - self.delta_S)
        pv_base = self.compute_pv(S0)
        
        # 恢复原始 S0
        self.path_generator.S0 = original_S0
        self.payoff.S0 = original_S0
        
        delta = (pv_plus - pv_minus) / (2 * self.delta_S)
        gamma = (pv_plus - 2 * pv_base + pv_minus) / (self.delta_S ** 2)
        return delta, gamma
    


# In[86]:


# 设置参数
S0 = 100  # 初始资产价格
r = 0.05  # 无风险利率
sigma = 0.2  # 波动率
T = 1  # 到期时间（1年）
dt = 1/252  # 每天为时间步长
num_paths = 10000  # 模拟路径数量
num_assets = 2 
L = 1
early_termination_times = [0, 50, 100, 150, 200, 252]  # 早期终止时间点（以天为单位）
c1 = 1  # 低支付
c2 = 2  # 高支付


# 随机数生成器和路径生成器实例
cov_matrix = np.array([
    [0.02, 0.01], 
    [0.01, 0.03]
])  # 协方差矩阵

random_number_generator = CorrelatedRandomNumberGenerator(
    num_steps=252, num_assets=num_assets, cov_matrix=cov_matrix, seed=42)

path_generator = PathGenerator(S0, r, sigma, T, dt, num_assets, random_number_generator)

# 创建Range Accrual期权实例
payoff = RangeAccrualPayoff(S0, K=1.01, T=T, dt=dt, early_termination_times=early_termination_times, c1=c1, c2=c2)
#print(payoff.compute(paths))

# 创建Monte Carlo定价器
monte_carlo_pricer = MonteCarloPricer(path_generator, payoff, num_paths=10000)

# 计算期权现值
option_value = monte_carlo_pricer.price()
print("Option Price (Monte Carlo):", option_value)

delta, gamma = monte_carlo_pricer.compute_greeks()
print("Delta values for each S0:", delta)
print("Gamma values for each S0:", gamma)


# In[ ]:




