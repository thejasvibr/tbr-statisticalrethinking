#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
My attempts at running 
@author: autumn
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import scipy.stats as stats

# %% R code 3.2 Running the sampling method:  
def prob_posterior_grid(num_points, wins, tosses):
    p_grid =  np.linspace(0,1,num_points)
    prob_p_values = np.tile(1, p_grid.size)
    prob_wins = stats.binom.pmf(wins, tosses,p_grid)
    unstd_posterior = prob_wins*prob_p_values
    std_posterior = unstd_posterior/unstd_posterior.sum()
    return p_grid, std_posterior

# %% Sampling from the 'underlying distribution'
p_values, posterior_prob = prob_posterior_grid(10000, 30,30)
num_samples = 10**6


sampled_pvalues = np.random.choice(p_values, size=num_samples, 
                                   replace=True, p=posterior_prob)

plt.figure()
az.plot_kde(sampled_pvalues)

# %% Sum probability of p<0.5
print(sum(posterior_prob[p_values<0.5]))


# %% Sum probability of sampled pvalues 
print(sum(sampled_pvalues<0.5)/sampled_pvalues.size)

# %% Quantile limits of the sampled p-values
np.percentile(sampled_pvalues,[10,90])

# %% HDPI 
az.hdi(sampled_pvalues,hdi_prob=0.9)

# %% R code 3.2 - 
prob_water = 0.7
tosses = 2 
num_waters = np.arange(0,3)
print(stats.binom.pmf(num_waters, tosses, prob_water))

# %% R code 3.21
np.random.binomial(tosses,p=prob_water)

# %% R code 3.22
np.random.binomial(tosses,p=prob_water,size=10)

# %% R code 3.23
many_tosses = 10**5
dummy_waters = np.random.binomial(tosses,p=prob_water,size=many_tosses)
values, counts = np.unique(dummy_waters, return_counts=True)
value_freq = counts/sum(counts)
pd.DataFrame(data={'value':values, 'frequency':value_freq})

# %% R code 3.24
# Here the number of tosses are increased to 9
tosses = 90
dummy_waters = np.random.binomial(tosses,p=prob_water,size=10**5)
values, counts = np.unique(dummy_waters, return_counts=True)
value_freq = counts/sum(counts)
pd.DataFrame(data={'value':values, 'frequency':value_freq})

plt.figure()
plt.hist(dummy_waters)
