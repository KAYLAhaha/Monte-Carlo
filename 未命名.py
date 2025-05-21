#!/usr/bin/env python
# coding: utf-8


import numpy as np
from tqdm import tqdm

class CorrelatedRandomNumberGenerator:
    def __init__(self, num_steps, num_assets, cov_matrix, seed=None):
        self.num_steps = num_steps  
        self.num_assets = num_assets 
        self.cov_matrix = cov_matrix  
        self.seed = seed 
        if seed is not None:
            np.random.seed(seed)  

    def generate(self):
        """ Generate relevant standard normally distributed random numbers, Shape of(num_steps, num_assets) """
        Z = np.random.normal(0, 1, (self.num_steps, self.num_assets))
        
        # Cholesky decomposition of the covariance matrix
        L = np.linalg.cholesky(self.cov_matrix)  

        # Converting independent standard normally distributed random numbers into correlated random numbers
        correlated_Z = np.dot(Z, L.T)  

        return correlated_Z


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
paths = path_generator.generate_paths()

# Mapping asset price paths
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
        self.c1 = c1  
        self.c2 = c2  
        self.B_KO = B_KO 

    def compute(self, paths):
        payoffs = []
        knocked_out = False 

        for t in range(1, len(self.early_termination_times)):
            if knocked_out:
                payoffs.append(0.0)
                continue

            t_k = self.early_termination_times[t]
            t_k_1 = self.early_termination_times[t-1]
            
            S_tk = paths[t_k]  
            performance_tk = S_tk / self.S0  
            
            # Get current knockout barriers (assuming B_KO is a list, corresponding to the point in time)
            B_KO_k = self.B_KO[t] if isinstance(self.B_KO, list) else self.B_KO
            
            if np.any(performance_tk >= B_KO_k):
                knocked_out = True  
            
            path_segment = paths[t_k_1:t_k]
            worst_performance = np.min(path_segment / self.S0, axis=1)
            A_k = np.sum(worst_performance >= self.K)
            C_k = (1 - A_k/(t_k - t_k_1)) * self.c1 + (A_k/(t_k - t_k_1)) * self.c2
            
            payoffs.append(C_k)
        
        return payoffs


# In[85]:


class MonteCarloPricer:
    def __init__(self, path_generator, payoff, num_paths, delta_S=1.0):
        self.path_generator = path_generator  
        self.payoff = payoff 
        self.num_paths = num_paths 
        self.delta_S = delta_S

        # Extracting shared parameters from path_generator and payoff
        self.S0 = self.path_generator.S0
        self.r = self.path_generator.r
        self.sigma = self.path_generator.sigma
        self.T = self.path_generator.T
        self.dt = self.path_generator.dt
        self.num_assets = self.path_generator.num_assets
        self.random_number_generator = self.path_generator.random_number_generator
        self.early_termination_times = self.payoff.early_termination_times
        
    def price(self):
        """ Calculate the PV at the current S0 """
        payoffs = [] 
        
        for _ in range(self.num_paths):
            paths = self.path_generator.generate_paths()
            path_payoffs = self.payoff.compute(paths) 
            
            discounted_payoffs = []
            for t, C_k in enumerate(path_payoffs):
                t_k = self.early_termination_times[t + 1] 
                discount_factor = np.exp(-self.path_generator.r * (t_k / 252))  
                discounted_payoffs.append(C_k * discount_factor)  
            
            payoffs.append(np.sum(discounted_payoffs)) 
        
        pv = np.mean(payoffs)
        return pv
    
    def compute_greeks(self):
        """ Calculate Delta and Gamma """
        original_S0 = self.path_generator.S0  
        
        # calculate PV(S0 + delta_S)
        self.path_generator.S0 = original_S0 + self.delta_S # Update S0 of path_generator
        self.payoff.S0 = original_S0 + self.delta_S  # Update S0 of payoff
        pv_up = self.price()
        
        # calculate PV(S0 - delta_S)
        self.path_generator.S0 = original_S0 - self.delta_S
        self.payoff.S0 = original_S0 - self.delta_S
        pv_down = self.price()
     
        # calculate PV(S0)
        self.path_generator.S0 = original_S0
        self.payoff.S0 = original_S0
        pv_base = self.price()
        
        # calculate Delta 和 Gamma
        delta = (pv_up - pv_down) / (2 * self.delta_S)
        gamma = (pv_up - 2 * pv_base + pv_down) / (self.delta_S ** 2)
        return delta, gamma
    
    def generate_spot_ladders(self, S0_range):
        """ Generate spot ladder data """
        pv_ladder = []
        delta_ladder = []
        gamma_ladder = []
        
        original_S0 = self.S0  
        
        for S in tqdm(S0_range, desc="Generating Spot Ladders"):
            self.path_generator.S0 = S
            self.payoff.S0 = S
        
            pv = self.price()
            delta, gamma = self.compute_greeks()
        
            pv_ladder.append(pv)
            delta_ladder.append(delta)
            gamma_ladder.append(gamma)
        
        # Restore the original S0
        self.path_generator.S0 = original_S0
        self.payoff.S0 = original_S0
        
        return {
            "S0": S0_range,
            "PV": np.array(pv_ladder),
            "Delta": np.array(delta_ladder),
            "Gamma": np.array(gamma_ladder)
        }
    
    def plot_ladders(ladder_data):
        plt.figure(figsize=(15, 5))
        
        # PV-Spot
        plt.subplot(1, 3, 1)
        plt.plot(ladder_data["S0"], ladder_data["PV"], "b-o")
        plt.title("PV-Spot Ladder")
        plt.xlabel("Spot Price")
        plt.ylabel("Present Value")
        plt.grid(True)
        
        # Delta-Spot
        plt.subplot(1, 3, 2)
        plt.plot(ladder_data["S0"], ladder_data["Delta"], "r-o")
        plt.title("Delta-Spot Ladder")
        plt.xlabel("Spot Price")
        plt.ylabel("Delta")
        plt.grid(True)
        
        # Gamma-Spot
        plt.subplot(1, 3, 3)
        plt.plot(ladder_data["S0"], ladder_data["Gamma"], "g-o")
        plt.title("Gamma-Spot Ladder")
        plt.xlabel("Spot Price")
        plt.ylabel("Gamma")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# In[86]:


# Setting parameters
S0 = 100 
r = 0.05  
sigma = 0.2  
T = 1  
dt = 1/252  
num_paths = 10000  
num_assets = 2 
early_termination_times = [0, 50, 100, 150, 200, 252]  
c1 = 1  
c2 = 2  
B_KO = 1.04


cov_matrix = np.array([
    [0.02, 0.01], 
    [0.01, 0.03]
])  # Covariance matrix of the two assets

random_number_generator = CorrelatedRandomNumberGenerator(
    num_steps=252, num_assets=num_assets, cov_matrix=cov_matrix, seed=42)

path_generator = PathGenerator(S0, r, sigma, T, dt, num_assets, random_number_generator)

payoff = RangeAccrualPayoff(S0, K=1.01, T=T, dt=dt, early_termination_times=early_termination_times, c1=c1, c2=c2, B_KO=B_KO)

monte_carlo_pricer = MonteCarloPricer(path_generator, payoff, num_paths=10000)

option_value = monte_carlo_pricer.price()
delta, gamma = monte_carlo_pricer.compute_greeks()
ladder_data = monte_carlo_pricer.generate_spot_ladders(S0_range)
monte_carlo_pricer.plot_ladders(ladder_data)





