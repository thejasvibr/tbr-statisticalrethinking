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
import pymc3 as pm

# %% R code 3.2 Running the sampling method:  
def prob_posterior_grid(num_points, wins, tosses, **kwargs):
    p_grid =  np.linspace(0,1,num_points)
    prob_p_values = kwargs.get('prior',np.tile(1, p_grid.size))
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

# %% Problems of Chapter 3
p_frac, post_prob_frac = prob_posterior_grid(1000, 6, 9)
np.random.seed(100)
samples = np.random.choice(p_frac, size=10**4, p=post_prob_frac)

# %% Prob. 3E1-E3
conditions = [lambda X : X<0.2, lambda X:X>0.8, lambda X: np.logical_and(X>=0.2, X<=0.8)]
post_prob_sums = [np.sum(each_cond(samples))/samples.size for each_cond in conditions]

# %% Prob. 3E4-E5
print(np.percentile(samples, [20, 80]))

# %% Prob 3E6
print(az.hdi(samples, hdi_prob=0.66))

# %% Prob 3M1
# What if the globe tossing lead to 8 waters in 15 tosses?

water_frac, posterior_prob = prob_posterior_grid(1000, 8, 15)

# %% Prob 3M2
# Draw 10,000 samples from the posterior-prob and construct 90% HPDI
sampled_newexpt = np.random.choice(water_frac, p=posterior_prob, replace=True, size=10**4)
plt.figure()
az.plot_kde(sampled_newexpt, label='Sampled')

# 90% HPDI 
nientypct_hpdi = az.hdi(sampled_newexpt, hdi=0.9)
print('Uninformed prior: HPDI',nientypct_hpdi)
# %% Prob 3M3
# Generate the posterior predictive checks using the sampled posterior distribution

pred_posterior_check = np.random.binomial(15,sampled_newexpt)
waters, counts  = np.unique(pred_posterior_check, return_counts=True)
prob_8s = sum(pred_posterior_check==8)/pred_posterior_check.size

plt.figure()
plt.plot(waters, counts/sum(counts),'-*')

# %% Prob 3M4
# Using the posterior distribution from the 8/15 data - check the probability of
# 6/9 waters

pred_posterior2_check = np.random.binomial(9,sampled_newexpt)
waters, counts  = np.unique(pred_posterior2_check, return_counts=True)
prob_6s = sum(pred_posterior2_check==5)/pred_posterior2_check.size
print(prob_6s)

plt.figure()
plt.plot(waters, counts/sum(counts),'-*')

# %% Prob 3M5 
# Start with 3M1's posterior distribution, but use a step-function prior, where
# at p>=0.5, the probability is 1, while p<0.5 the prob is 0.
informed_prior = np.concatenate((np.zeros(500), np.ones(500)))
water_frac_informed, posterior_prob_informed = prob_posterior_grid(1000, 8, 
                                                                   15,
                                                                   prior=informed_prior)
sampled_posterior_informed = np.random.choice(water_frac_informed,
                                              p=posterior_prob_informed,size=10**4)

plt.figure()
az.plot_kde(sampled_posterior_informed)
# Now let's calculate the 90%hpdi
print('Informed prior: HPDI',az.hdi(sampled_posterior_informed))

# Now generate the predictive posterior check
pred_posteriorcheck_informed = np.random.binomial(15, p=sampled_posterior_informed)
waters_informed, counts_informed  = np.unique(pred_posteriorcheck_informed, return_counts=True)

plt.figure()
plt.plot(waters_informed, counts_informed/sum(counts_informed),'-*', label='Informed')
plt.plot(waters, counts/sum(counts),'-*', label='Uninformed')
plt.legend()

# Both informed and flat priors result in matching posterior check predictions. 
# In the informed prior however, the 90%HPDI is much narrower (0.5-0.73 vs 0.31-0.75)!

# %% Prob 3M6 
# If you want a very narrow 99% HPDI estimate that is at most 0.05 wide, how many times
# will you need to toss the globe? 
water_fraction = 0.7
num_trials = [10,100,1000,3000,10**4]

def measure_99pct_hpdi_width(num_trials, fraction):
    observed = np.random.binomial(num_trials, fraction)
    # assume a flat prior and 1000 point grid sampling
    frac_values, posterior_prob_frac = prob_posterior_grid(1000, observed, num_trials)
    # sample from the underlying posterior probability 
    sampled_posterior_prob = np.random.choice(frac_values, p=posterior_prob_frac, size=10**4)
    min_hdi, max_hdi = az.hdi(sampled_posterior_prob, 0.99)
    return max_hdi-min_hdi, (f'({np.round(min_hdi,2)},{np.round(max_hdi,2)})')

for each in num_trials:
    print('# trials:',each,'99%hpdi width:',measure_99pct_hpdi_width(each, water_fraction))

# %% Prob Hard 
# Thanks to https://github.com/pymc-devs/resources/blob/master/Rethinking_2/Chp_03.ipynb
# so I  could copy the data :p

birth1 = np.array([1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0,
                   1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1])
birth2 = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0,
                   1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1,
                   0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
print(sum(birth1) + sum(birth2))
birth1n2 = np.column_stack((birth1,birth2))

# %% Prob 3H1
num_boys = sum(birth1n2.flatten())
num_births =  birth1n2.flatten().size

# assuming  a flat prior - what p(boy birth) explain the data?
p_boy, posterior_pboy = prob_posterior_grid(1000, num_boys, num_births)
plt.figure()
plt.plot(p_boy, posterior_pboy)

# %% Prob 3H2
# If we were to sample from the underlying distribution, what values of p-boy would appear?
sampled_pboy = np.random.choice(p_boy,p=posterior_pboy,size=10**4)
plt.figure()
az.plot_kde(sampled_pboy)

# calculate the various intervals
intervals = np.array([0.5, 0.89, 0.97])
print([az.hdi(sampled_pboy,hdi_prob=each) for each in intervals])

# %% Prob 3H3
# Simulate 10,000 runs of 200 births and count the number of boys born. 
num_boys = np.random.binomial(200,p=sampled_pboy)
boys_born, freq = np.unique(num_boys, return_counts=True)
plt.figure()
plt.plot(boys_born, freq/sum(freq))

# %% Prob 3H4
num_firstboys = np.random.binomial(100,p=sampled_pboy)
boys_born, freq = np.unique(num_firstboys, return_counts=True)
plt.figure()
plt.plot(boys_born, freq/sum(freq),'-*', label='$Simulated P_{boys}$ from all births')
plt.vlines(sum(birth1), 0, np.max(freq/sum(freq)), label='Observed $P_{boy}$ from 1st birth')
plt.legend()

# Does this mean we are overestimating the fraction of boys to be born -- or is there
# some kind of correlatin between the sex of the first and second child?

# %% Prob 3H5
girl_firstbirths = birth1n2[birth1==0,:]
# number of boys born after first girl birth
nboys_aftergirl = sum(girl_firstbirths[:,1])
frction_boys_aftergirl =  nboys_aftergirl/girl_firstbirths.size

sim_numboys_girls1st =  np.random.binomial(girl_firstbirths.shape[0], p=sampled_pboy)
num_simboys, count_simboys = np.unique(sim_numboys_girls1st, return_counts=True)

plt.figure()
plt.plot(num_simboys, count_simboys/sum(count_simboys), '-*', label='expected boys, girls 1st born')
plt.vlines(nboys_aftergirl, 0, max(count_simboys/sum(count_simboys)), label='Observed boy from 1st birth')
plt.legend()

# %% Chpapter 4
# -------------
# The sum of multiple processes, irrespective of the original distribution -
# will be a norml distribution. 

# np.random.binomial(10, p=np.tile(0.5,10**3)) - binomial process

source_process = np.array(([sum(np.random.binomial(10,p=np.tile(0.2,10))) for each in range(10**4)]))

plt.figure()
plt.hist(source_process)

# %% Normal distributions by multiplicative processes

multipl_distbn = [ np.prod(np.random.uniform(1.0,1.5,12)) for each in range(10**3)]

plt.figure()
plt.hist(multipl_distbn)

# %% Normal distributions by log-additive processes

logmultipl_distbn = [ np.log10(np.prod(np.random.uniform(2.0,5,12))) for each in range(10**3)]

plt.figure()
plt.hist(logmultipl_distbn)

# %% !Kung San dataset exercise 

d = pd.read_csv('datasets/Howell1.csv', sep=';')
print(az.summary(d.to_dict(orient="list"), kind="stats", hdi_prob=0.89))

d_above18 = d['height'][d['age']>18]
plt.figure()
az.plot_kde(d_above18)

x = np.linspace(100, 250, 100)
plt.figure()
plt.plot(x, stats.norm.pdf(x, 178, 20));

# %% Checking validity of priors
numsamples = 10**4
sample_mu = np.random.normal(178, 20, numsamples)
sample_sigma = np.random.uniform(0,50,numsamples)
prior_h = np.random.normal(sample_mu, sample_sigma)
plt.figure()
az.plot_kde(prior_h, label='informed $\mu$ prior')

# 'uninformed' prior's effect
flat_sample_mu = np.random.normal(178,100,numsamples)
flat_prior_h = np.random.normal(flat_sample_mu, sample_sigma)
az.plot_kde(flat_prior_h, label='uninformed $\mu$ prior')

# %% Rcode 4.16 - well commented function 
calc_sum_logLL = lambda musigma, inputdata : sum(stats.norm.logpdf(inputdata, musigma[0], musigma[1]))

def height_posterior_prob(heights,mu_est,sigma_est, mu_prior_params,sigma_prior_params):
    # generate grid points
    mu_points = np.linspace(mu_est[0],mu_est[1],100)
    sigma_points = np.linspace(sigma_est[0], sigma_est[1], 100)
    # make parameter combinations
    mu_sigma_combis = np.concatenate(np.meshgrid(mu_points,sigma_points)).flatten().reshape(2,-1).T
    # non-uniform prior for all param-combis 
    mu_prior = stats.norm.logpdf(mu_sigma_combis[:,0], mu_prior_params[0],
                                                         mu_prior_params[1])
    sigma_prior = stats.uniform.logpdf(mu_sigma_combis[:,1], sigma_prior_params[0],
                                                        sigma_prior_params[1])
    # calculate (sum) likelihoods of the data from each param-combi
    param_combi_likelihood = np.apply_along_axis(calc_sum_logLL, 1, mu_sigma_combis, heights)
    param_combi_priors  = sigma_prior+mu_prior
    posterior_loglikelihood = param_combi_likelihood + param_combi_priors
    posterior_probs_norm = np.exp(posterior_loglikelihood-np.max(posterior_loglikelihood))
    return mu_sigma_combis, posterior_loglikelihood, posterior_probs_norm
        
paramcombis, posterior_ll, probs = height_posterior_prob(d_above18, [150,160], [7,9],
                                                  [178,20], [0,50])
plt.figure()
az.plot_kde(posterior_ll)

# %% Plotting the posterior probabilities of mu and sigma
from scipy.interpolate import griddata
xi = np.linspace(paramcombis[:, 0].min(), paramcombis[:, 0].max(), 100)
yi = np.linspace(paramcombis[:, 1].min(), paramcombis[:, 1].max(), 100)
zi = griddata((paramcombis[:,0], paramcombis[:,1]), probs,(xi[None, :], yi[:, None]))

plt.figure()
a0 = plt.subplot(111)
a0.imshow(zi, origin='lower')
a0.set_xticks([0, 50, 100])
a0.set_xticklabels([150, 155, 160])
a0.set_yticks([0, 50, 100])
a0.set_yticklabels([7, 8, 9])
a0.grid(False); plt.title('Posterior probability of $\mu$ (x-axis) and $\sigma$ (y-axis)')

# %% Sampling to generate all the underlying possible mu and sigma estimates
# and now let's sample from the posterior distribution to generate values of expected
# height. 

sampled_paramcombi_rows = np.random.choice(np.arange(0,paramcombis.shape[0]),
                                           p=probs/sum(probs),replace=True,
                                           size=paramcombis.shape[0])
sampled_paramcombis = paramcombis[sampled_paramcombi_rows,:]

plt.figure()
plt.plot(sampled_paramcombis[:,0],sampled_paramcombis[:,1],'*')

plt.figure()
a00 = plt.subplot(211)
az.plot_kde(sampled_paramcombis[:,0],label='posterior sampled $\mu$')
mu_hpdi_89 = az.hdi(sampled_paramcombis[:,0],0.89)
plt.text(0.3,0.1, f'$\mu$ 89% interval: {np.round(mu_hpdi_89,2)}',
                                                 transform = a00.transAxes)
a01 = plt.subplot(212)
az.plot_kde(sampled_paramcombis[:,1],label='posterior sampled $\sigma$')
sig_hpdi_89 = az.hdi(sampled_paramcombis[:,1],0.89)
plt.text(0.3,0.1, f'$\sigma$ 89% interval: {np.round(sig_hpdi_89,2)}',
                                                     transform = a01.transAxes)



# %% Using a subset of all heights - let's see how the parameter estimation works
# out now 

d3 = np.random.choice(d_above18, 20, replace=False)
paramcombis, posterior_ll, probs = height_posterior_prob(d3, [150,170], [4,20],
                                                  [178,20], [0,50])

sampled_paramcombi_rows = np.random.choice(np.arange(0,paramcombis.shape[0]),
                                           p=probs/sum(probs),replace=True,
                                           size=paramcombis.shape[0])
sampled_paramcombis = paramcombis[sampled_paramcombi_rows,:]

plt.figure()
plt.plot(sampled_paramcombis[:,0],sampled_paramcombis[:,1],'*')

plt.figure()
a00 = plt.subplot(211)
az.plot_kde(sampled_paramcombis[:,0],label='posterior sampled $\mu$')
mu_hpdi_89 = az.hdi(sampled_paramcombis[:,0],0.89)
plt.text(0.3,0.1, f'$\mu$ 89% interval: {np.round(mu_hpdi_89,2)}',
                                                 transform = a00.transAxes)
a01 = plt.subplot(212)
az.plot_kde(sampled_paramcombis[:,1],label='posterior sampled $\sigma$')
sig_hpdi_89 = az.hdi(sampled_paramcombis[:,1],0.89)
plt.text(0.3,0.1, f'$\sigma$ 89% interval: {np.round(sig_hpdi_89,2)}',
                                                     transform = a01.transAxes)
# %% R code 4.27
# Setting up a model with quad

with pm.Model() as m41:
    mu = pm.Normal('mu',mu=178,sd=20)
    sigma = pm.Uniform('sigma',lower=0,upper=50)
    height = pm.Normal("height",mu=mu,sd=sigma,observed=d_above18)

map_estimate_m41 = pm.find_MAP(model=m41)


with m41:
    trace_41 = pm.sample(1000, tune=1000)

az.plot_trace(trace_41)

az.summary(trace_41, round_to=2, kind="stats", hdi_prob=0.89)


# %% R code 4.30
# Now let's use an informative prior - wht is the best loaction of the 

m42 = pm.Model()
with m42:
    mu = pm.Normal('mu',mu=178,sd=0.1)
    sigma = pm.Uniform('sigma',lower=0,upper=50)
    height = pm.Normal("height",mu=mu,sd=sigma,observed=d_above18)

map_estimate_m42 = pm.find_MAP(model=m42)
print(map_estimate_m42)

with m42:
    trace_42 = pm.sample(1000, tune=1000)
az.plot_trace(trace_42)

az.summary(trace_42, round_to=2, kind="stats", hdi_prob=0.89)


# %% Variance-covariance matrices of the models 
trace_df = pm.trace_to_dataframe(trace_41)
trace_df.cov()

variances = np.diag(trace_df.cov())
variances

# %% Plotting height and weight
dabove18 = d[d['age']>18]
plt.figure()
plt.plot(dabove18['height'],dabove18['weight'],'*')

# %% R code 4.38

num_lines = 100
a = np.random.normal(178, 20, num_lines)
b = np.random.normal(0, 10, num_lines)
blognorm =  np.random.lognormal(0,3,num_lines)
#  Now plot the lines
line_function = lambda aval,bval,weights : aval+bval*(weights-np.mean(weights))

plt.figure()
plt.subplot(211)
for ai,bi in zip(a,b):
    pred_height = line_function(ai,bi,dabove18['weight'])
    plt.plot(dabove18['weight'], pred_height)
plt.subplot(212)
for ai,bi in zip(a,blognorm):
    pred_height = line_function(ai,bi,dabove18['weight'])
    plt.plot(dabove18['weight'], pred_height)


# %% Rcode 4.42
# Define a model which estimates parameters of a linear regression 

m_43 = pm.Model()
with m_43:
    a = pm.Normal('a', mu=178, sd=20)
    b = pm.Lognormal('b', mu=0, sd=1)
    sigma = pm.Uniform('sigma', 0, 50)
    mu = a + b*(dabove18['weight']-dabove18['weight'].mean())
    h_i = pm.Normal('h_i', mu=mu, sd=sigma, observed=dabove18['height'].to_numpy())
    trace_43 = pm.sample(10000, tune=1000)

az.summary(trace_43, kind='stats', hdi_prob=0.89)

# inspecting the variance-covariance of the posterior distribution
trace_43df = pm.trace_to_dataframe(trace_43)
trace_43_varcov = trace_43df.cov().round(3)

# %% R code 4.46 

mean_pred_line =  line_function(trace_43df['a'].mean(), trace_43df['b'].mean(),
                                dabove18.weight)
plt.figure()
plt.plot(dabove18.weight, dabove18.height,'o')
plt.plot(dabove18.weight, mean_pred_line)

# %% R code 4.48
m_44 = pm.Model()
numpoints = 50
part10points = dabove18.loc[:numpoints,:]

with m_44:
    a = pm.Normal('a', mu=178, sd=20)
    b = pm.Lognormal('b', mu=0, sd=1)
    sigma = pm.Uniform('sigma', 0, 50)
    mu = a + b*(part10points['weight']-part10points['weight'].mean())
    h_i = pm.Normal('h_i', mu=mu, sd=sigma, observed=part10points['height'].to_numpy())
    trace_44 = pm.sample(10000, tune=1000)

az.summary(trace_44, kind='stats', hdi_prob=0.89)

# inspecting the variance-covariance of the posterior distribution
trace_44df = pm.trace_to_dataframe(trace_44)
trace_44_varcov = trace_44df.cov().round(3)


mean_predline_p10 =  line_function(trace_44df['a'].mean(), trace_44df['b'].mean(),
                                part10points.weight)

plt.figure()
plt.plot(part10points.weight, part10points.height,'o')
plt.plot(part10points.weight, mean_predline_p10)

# the other lines possible are:
mean_predline_p10_multi =  []
for ai, bi in zip(trace_44df.a[:numpoints], trace_44df.b[:numpoints]):
    mean_predline_p10_multi.append(line_function(ai,bi, part10points.weight))

for each in mean_predline_p10_multi:
    plt.plot(part10points.weight, each, 'r', alpha=0.3)

# %% Rcode 4.57-4.58
weight_range = np.arange(25,71)
trace_43_thinned = trace_43df[::10]
mu_pred = trace_43_thinned["a"] + trace_43_thinned["b"] * (dabove18.weight - dabove18.weight.mean())


# %% Rcode 4.64
# Association between height and weight for all individuals

plt.figure()
plt.plot(d.weight, d.height,'*')




