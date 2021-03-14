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
import patsy
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

# %% R code 4.65
# Fitting the height data as a parabolic curve 

d.std_weight = (d.weight-np.mean(d.weight))/np.std(d.weight)
d.stdweight_sq = d.std_weight**2.0

m_4_65 = pm.Model()

with m_4_65:
    b1 = pm.Lognormal('b1', mu=0,sd=1)
    b2 = pm.Normal('b2', mu=0, sd=1)
    a = pm.Normal('a', mu=178, sd=20)
    mu_i = pm.Deterministic('mu_i',a + b1*d.std_weight + b2*d.stdweight_sq)
    #mu_i = a + b1*d.std_weight + b2*d.stdweight_sq
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    height = pm.Normal('height',mu=mu_i,sd=sigma, observed=d.height)
    trace_4_65 = pm.sample(10000,tune=1000)

# now summarise the posterior distbn of the parameters
trace_4_65df = pm.trace_to_dataframe(trace_4_65)
az.summary(trace_4_65, kind='stats', hdi_prob=0.89)

# %% Plot the output
mu_predicted = trace_4_65['mu_i']
height_pred = pm.sample_posterior_predictive(trace_4_65, 200, m_4_65)
ax = az.plot_hdi(d.std_weight, mu_predicted, hdi_prob=0.89)
az.plot_hdi(d.std_weight, height_pred["height"], ax=ax)
plt.scatter(d.std_weight, d.height, c="C0", alpha=0.3)


# %% Cherry blossoms dataset 
cherry = pd.read_csv('datasets/cherry_blossoms.csv')
cherry_nona = cherry.dropna(subset=['doy'])
print(cherry.describe())

# create knot points across the years in the dataset 
n_knots = 15
knot_list = np.quantile(cherry_nona.year, np.linspace(0,1,n_knots))

# %% setup the B-spline matrix and form the underlying basis
# functions 

from patsy import dmatrix
B = dmatrix(
    "bs(year, knots=knots, degree=3, include_intercept=True) - 1",
    {"year": cherry_nona.year.values, "knots": knot_list[1:-1]},
)

# plot the underlying b splines
plt.figure()
ax = plt.subplot(111)
for i in range(17):
    ax.plot(cherry_nona.year, (B[:, i]), color="C0")
ax.set_xlabel("year")
ax.set_ylabel("basis");

# %% Estimating parameters for the B-splines 
m4_7 = pm.Model()

with m4_7:
    a = pm.Normal('a', mu=100, sd=10)
    w = pm.Normal('w', mu=0, sd=10, shape=B.shape[1])
    mu = pm.Deterministic('mu', a+pm.math.dot(np.asarray(B, order="F"), w.T))
    sigma = pm.Exponential('sigma', 1)
    D = pm.Normal("D", mu, sigma, observed=cherry_nona.doy)
    trace_m4_7 = pm.sample(10000, tune=1000)
    
# %% Plot the estimaated basis functions long with their weights
_, ax = plt.subplots(1, 1,)
wp = trace_m4_7[w].mean(0)
for i in range(17):
    ax.plot(cherry_nona.year, (wp[i] * B[:, i]), color="C0")
ax.set_xlim(812, 2015)
ax.set_ylim(-6, 6);


# %% Now, plot 
ax = az.plot_hdi(cherry_nona.year, trace_m4_7["mu"], color="k")
ax.plot(cherry_nona.year, cherry_nona.doy, "o", alpha=0.3)
ax.set_xlabel("year")
ax.set_ylabel("days in year")
# also plot the predictive hdpi
mu_predicted = trace_m4_7['mu']
doy_pred = pm.sample_posterior_predictive(trace_m4_7,
                                          samples=200,
                                          model=m4_7,var_names=['mu','D'])
ax = az.plot_hdi(cherry_nona.year, doy_pred['mu'], hdi_prob=0.89)
ax = az.plot_hdi(cherry_nona.year, doy_pred['D'], hdi_prob=0.89)

# %% Chapter 4 Problems
# 4M1 
n_sims = 1000
sigma = np.random.exponential(0.5, n_sims)
mu = np.random.normal(0,10, n_sims)
y = np.random.normal(mu, sigma)
plt.figure()
plt.plot(y)

# 4M2 
m_p42 = pm.Model()
with m_p42:
    mu = pm.Normal('mu', mu=0, sd=10)
    sigma = pm.Exponential('sigm', 1)
    y = pm.Normal('y', mu=mu, sd=sigma, observed=cherry_nona.doy)
    trace_m_p42 = pm.sample(1000)


# %% Problem 4M7
# Regressing the !Kung San data without standardising the
# explanatory weight variable. 
from pymc3 import math 

dabove18['weight_std'] = dabove18.weight-np.mean(dabove18.weight)

m_p4m7 = pm.Model()
m_p4m7unstd = pm.Model()

with m_p4m7:
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    logb = pm.Normal('logb', mu=0, sd=1)
    a = pm.Normal('a', mu=178, sigma=20)
    mu = pm.Deterministic('mu',a + math.exp(logb)*(dabove18.weight_std))
    height = pm.Normal('height', mu=mu, sd=sigma, observed=dabove18.height)    
    trace_p4m7 = pm.sample()
trace_p4m7df= pm.trace_to_dataframe(trace_p4m7)

with m_p4m7unstd:
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    logb = pm.Normal('logb', mu=0, sd=1)
    a = pm.Normal('a', mu=178, sigma=20)
    mu = pm.Deterministic('mu',a + math.exp(logb)*(dabove18.weight))
    height = pm.Normal('height', mu=mu, sd=sigma, observed=dabove18.height)    
    trace_p4m7unstd = pm.sample()
trace_p4m7unstddf= pm.trace_to_dataframe(trace_p4m7unstd)

# %% Comparing the covariance matrices of the standardised and 
# unstandardised regressions

# the posterior  prediction interval
post_predint_std = pm.sample_posterior_predictive(trace_p4m7, 200, model=m_p4m7,
                                                  var_names=['height'])
post_predint_unstd = pm.sample_posterior_predictive(trace_p4m7unstd, 200,
                                                    model=m_p4m7unstd,
                                                  var_names=['height'])

plt.figure()
plt.subplot(211)
plt.scatter(dabove18.weight, dabove18.height, alpha=0.4)
az.plot_hdi(dabove18.weight, trace_p4m7['mu'])
az.plot_hdi(dabove18.weight, post_predint_std['height'], color='y')
plt.title('X = standardised weights')

plt.subplot(212)
plt.scatter(dabove18.weight, dabove18.height, alpha=0.4)
az.plot_hdi(dabove18.weight, trace_p4m7unstd['mu'], color='r')
az.plot_hdi(dabove18.weight, post_predint_unstd['height'], color='g')
plt.title('X = Raw weights')

# The difference in variance covariance matrices:

trace_p4m7df.cov()

# %% Problem 4M8
# Increase the number of knots used to fit splines in the cherry blossom dataset
# Also change the width and prior on the weights. 


# create knot points across the years in the dataset 
def fit_bplines(nknots, cherrydata, **kwargs):
    
    knot_list = np.quantile(cherrydata.year, np.linspace(0,1,nknots))
    
    B = dmatrix(
        "bs(year, knots=knots, degree=3, include_intercept=True) - 1",
        {"year": cherrydata.year.values, "knots": knot_list[1:-1]},)
    # estimating the spline weights
    spline_model = pm.Model()

    with spline_model:
        a = pm.Normal('a', mu=kwargs.get('a_muprior',100),
                           sd=kwargs.get('a_sdprior',10))
        w = pm.Normal('w', mu=kwargs.get('w_muprior',0),
                           sd=kwargs.get('w_sdprior',10),
                                               shape=B.shape[1])
        mu = pm.Deterministic('mu', 
                                  a+pm.math.dot(np.asarray(B, order="F"), w.T))
        sigma = pm.Exponential('sigma', 1)
        D = pm.Normal("D", mu, sigma, observed=cherrydata.doy)
        trace_spline = pm.sample(1000, tune=1000)

    return B, trace_spline, spline_model 
# %% 

weird_aprior = {'w_sdprior':50}
b_matrix, trace_out, model = fit_bplines(5, cherry_nona, **weird_aprior)

# %% plot the underlying b splines
plt.figure()
ax = plt.subplot(111)
for i in range(b_matrix.shape[1]):
    ax.plot(cherry_nona.year, np.asarray(b_matrix[:, i]), color="C0")
ax.set_xlabel("year")
ax.set_ylabel("basis");

# %% plot the posterior mean 89% CI
#plt.figure()
ax = az.plot_hdi(cherry_nona.year, trace_out["mu"], color="k")
ax.plot(cherry_nona.year, cherry_nona.doy, "o", alpha=0.3)
ax.set_xlabel("year")
ax.set_ylabel("days in year")
# also plot the predictive hdpi
mu_predicted = trace_out['mu']
doy_pred = pm.sample_posterior_predictive(trace_out,
                                          samples=200,
                                          model=model,var_names=['mu'])
ax = az.plot_hdi(cherry_nona.year, doy_pred['mu'], hdi_prob=0.89)

# %% Problem 4H1 

obs_weights = pd.DataFrame(data={'weight':[46.95, 43.72, 64.78, 32.59,54.63], 
                                 'individual':range(1,6)},)


# %% Let's re-run the whole regression once more using the above 18 data
m_geq18 = pm.Model()

with m_geq18:
    beta = pm.Lognormal('beta', mu=0, tau=1)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    alpha = pm.Normal('alpha', mu=178, sd=20)
    mean = pm.Deterministic('mean', alpha + beta*dabove18.weight.values)
    height = pm.Normal('height', mu=mean, sd=sigma, observed = dabove18.height)
    heights_map = pm.find_MAP()
    print(heights_map)

# %% 
obs_weights['expected_height'] = heights_map['alpha'] + heights_map['beta']*obs_weights['weight']
# sample 10,000 runs with the MAP means + sigma. 
mean_predn_samples = []
for i in range(10**3):
    mean_predn_samples.append(np.random.normal(obs_weights['expected_height'].values,
                        np.tile(heights_map['sigma'], obs_weights.shape[0])))
mean_predn_samples = np.concatenate(mean_predn_samples).reshape(10**3,-1)
hdi_intvl = az.hdi(mean_predn_samples, hdi_prob=0.89)
obs_weights['interval_89_lower'] = hdi_intvl[:,0]
obs_weights['interval_89_upperer'] = hdi_intvl[:,1]

# %% Problem 4H2

kids = d[d.age<18]


m_le18 = pm.Model()
with m_le18:
    beta = pm.Lognormal('beta', mu=0, tau=1)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    alpha = pm.Normal('alpha', mu=178, sd=20)
    mean = pm.Deterministic('mean', alpha + beta*kids.weight.values)
    height = pm.Normal('height', mu=mean, sd=sigma, observed = kids.height)
    heights_map_k = pm.find_MAP()
    trace_le18 = pm.sample(5000)
# %% 
plt.figure()
plt.plot(kids.weight, kids.height,'*')
map_fit = heights_map_k['alpha'] + heights_map_k['beta']*kids.weight
plt.plot(kids.weight, map_fit,'r')

plt.plot(dabove18.weight, dabove18.height,'*')
map_fit = heights_map['alpha'] + heights_map['beta']*dabove18.weight
plt.plot(dabove18.weight, map_fit,'g')

# %% 
# Plot the MAP line, the 89% mean interval and 89% prediction interval s
plt.figure()
plt.plot(kids.weight, kids.height,'*')
#map_fit = heights_map_k['alpha'] + heights_map_k['beta']*kids.weight
#plt.plot(kids.weight, map_fit,'r')

# generate the 89% prediction interval 
az.plot_hdi(kids.weight, trace_le18['mean'], hdi_prob=0.97)
prediction_samples = pm.sample_posterior_predictive(trace_le18,
                                                    2000,
                                                    model=m_le18)

az.plot_hdi(kids.weight, prediction_samples['height'], hdi_prob=0.89)

# %% Generating the log-weight vs height estimates

d['logweight'] = np.log10(d.weight)


m_all = pm.Model()
with m_all:
    beta = pm.Lognormal('beta', mu=0, tau=1)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    alpha = pm.Normal('alpha', mu=178, sd=20)
    mean = pm.Deterministic('mean', alpha + beta*d.logweight.values)
    height = pm.Normal('height', mu=mean, sd=sigma, observed = d.height)
    heights_map_k = pm.find_MAP()
    trace_all = pm.sample(5000)

# %% plot the 
plt.figure()
plt.scatter(d.logweight, d.height)
#map_fit = heights_map_k['alpha'] + heights_map_k['beta']*kids.weight
#plt.plot(kids.weight, map_fit,'r')

# generate the 89% prediction interval 
az.plot_hdi(d.logweight, trace_all['mean'], hdi_prob=0.97)
prediction_samples = pm.sample_posterior_predictive(trace_all,
                                                    200,
                                                    model=m_all)

az.plot_hdi(d.logweight, prediction_samples['height'], hdi_prob=0.97)

# %% Problem 4H4
# fit the parabolic (quadratic) line to the the whole height vs weight
# data - using only what you know about weight and height. 


num_samples = 1000
weight = np.linspace(2,100,num_samples)
weightsq = weight**2
alpha = np.random.normal(30,20, num_samples)
beta1 = np.random.uniform(0.001,0.01, num_samples)
beta2 = np.random.uniform(0.001,0.02, num_samples)
sigma = np.random.uniform(0,50, num_samples)
mean = alpha + beta1*weight + beta2*weightsq

plt.figure()
plt.plot(weight, mean, '*')

# %% 
# find sensible values of b1 and b2 given the 
# final heights must be between 0-270cm
b1_range = np.linspace(0.01,1.0,100)
b2_range = np.linspace(10**-3,0.025,100)

b1b2 = np.array(np.meshgrid(b1_range, b2_range)).reshape(2,-1).T

def calc_height(inputb, weight=np.linspace(2,100,100)):
    b1,b2 = inputb
    return 30 + b1*weight+b2*weight**2

def within_valid_range(allheights):
    minheight = 30
    maxheights = 270
    within_limits = np.logical_and(np.sum(allheights<=minheight*1.5)>1,
                                   allheights<=maxheights)
    all_withinlimits = np.all(within_limits)
    return all_withinlimits
    

combi_heights = np.apply_along_axis(calc_height, 1, b1b2)
valid_paramcombis = np.apply_along_axis(within_valid_range, 1, combi_heights)
print(np.sum(valid_paramcombis))
sensible_paramcombis = b1b2[valid_paramcombis,:]
print(np.percentile(sensible_paramcombis[:,0], [0,100]))
print(np.percentile(sensible_paramcombis[:,1], [0,100]))
# %% Cherry blossom with time 
# model doy with march temperature

plt.figure()
plt.scatter(cherry_nona.temp,cherry_nona.doy)

# %% Let's first do a linear model. 
# The whole season is between jan-may or doy of 01-150, so let's just assume
# a uniform distbn here.
# 
cherry_nontempna = cherry[['doy','temp']].dropna()

# equation is doy = intercept + alpha*temp
intcpt = np.random.normal(150,5)
m = -15
t = np.linspace(0,10,100)
doy_sim = intcpt + m*t
plt.figure()
plt.scatter(t, doy_sim)
# %%  Generate some priors
m_blossom = pm.Model()
with m_blossom:
    intcpt = pm.Normal('intcpt',mu=150,sd=5)
    alpha = pm.Uniform('alpha',lower=-20, upper=-1)
    mean = pm.Deterministic('mean',intcpt + alpha*cherry_nontempna.temp)
    sigma = pm.Uniform('sigma',lower=1,upper=10)
    doy = pm.Normal('doy', mu=mean,sd=sigma, observed=cherry_nontempna.doy)
    trace_blossom_lm = pm.sample(10000)
    map_fit = pm.find_MAP()

map_line = map_fit['intcpt']+map_fit['alpha']*cherry_nontempna.temp
post_predint = pm.sample_posterior_predictive(trace_blossom_lm, 200,m_blossom,var_names=['doy','mean'])

# %% 
plt.figure()
plt.scatter(cherry_nontempna.temp, cherry_nontempna.doy)
plt.plot(cherry_nontempna.temp,map_line)
az.plot_hdi(cherry_nontempna.temp, post_predint['doy'], hdi_prob=0.89)
az.plot_hdi(cherry_nontempna.temp, post_predint['mean'], hdi_prob=0.89)

# In general temperature explains some of the variation but not all of it...
# Many of the data points are spread awy from the mean-prediction *and* the 
# 89% posterior prediction interval.

# %% What about a parabolic fit? 
b1, b2 = -3, -1
blossom_parab = 150 + b1*t + b2*t**2
plt.figure()
plt.scatter(t,blossom_parab)

# %%
cherry_nontempna['t2'] = cherry_nontempna.temp
m_blossom_parabola = pm.Model()
with m_blossom_parabola:
    intcpt = pm.Normal('intcpt',mu=150,sd=5)
    b1 = pm.Uniform('b1',lower=-20, upper=-1)
    b2 = pm.Uniform('b2',lower=-2, upper=-0.01)
    mean = pm.Deterministic('mean',intcpt + b1*cherry_nontempna.temp + b2*cherry_nontempna.t2)
    sigma = pm.Uniform('sigma',lower=1,upper=10)
    doy = pm.Normal('doy', mu=mean,sd=sigma, observed=cherry_nontempna.doy)
    trace_blossom_parab = pm.sample(10000)
    map_fit2 = pm.find_MAP()

# %% 

map_line_p = map_fit2['intcpt']+map_fit2['b1']*cherry_nontempna.temp+ map_fit2['b2']*cherry_nontempna.t2
post_predint_p = pm.sample_posterior_predictive(trace_blossom_parab,
                                              200,
                                              m_blossom_parabola,
                                              var_names=['doy','mean'])

# %% 
plt.figure()
plt.scatter(cherry_nontempna.temp, cherry_nontempna.doy)
plt.plot(cherry_nontempna.temp,map_line_p)
az.plot_hdi(cherry_nontempna.temp, post_predint_p['doy'], hdi_prob=0.89)
az.plot_hdi(cherry_nontempna.temp, post_predint_p['mean'], hdi_prob=0.89)

# %% 
plt.figure()
plt.subplot(311)
plt.scatter(cherry_nontempna.temp, cherry_nontempna.doy)
plt.plot(cherry_nontempna.temp,map_line)
az.plot_hdi(cherry_nontempna.temp, post_predint['doy'], hdi_prob=0.89)
az.plot_hdi(cherry_nontempna.temp, post_predint['mean'], hdi_prob=0.89)
plt.xticks([])
plt.title('linear')
plt.subplot(312)
plt.title('parabolic')
plt.scatter(cherry_nontempna.temp, cherry_nontempna.doy)
plt.plot(cherry_nontempna.temp,map_line_p)
az.plot_hdi(cherry_nontempna.temp, post_predint_p['doy'], hdi_prob=0.89)
az.plot_hdi(cherry_nontempna.temp, post_predint_p['mean'], hdi_prob=0.89)


# %% Chapter 5 :
waffle_div = pd.read_csv('datasets/WaffleDivorce.csv', sep=';')
def standardise(values):
    return (values-np.mean(values))/np.std(values)

waffle_div['std_divorce'] = standardise(waffle_div['Divorce'])
waffle_div['std_marriage'] = standardise(waffle_div['Marriage'])
waffle_div['std_age'] = standardise(waffle_div['MedianAgeMarriage'])


# %% plotting the variables involved

plt.figure()
plt.subplot(121)
plt.scatter(waffle_div['std_age'], waffle_div['std_divorce']);
plt.ylabel('Std. divorce rate');plt.xlabel('Std. Median age of marriage')
plt.subplot(122)
plt.scatter(waffle_div['std_marriage'], waffle_div['std_divorce'])
plt.xlim(-2,2);plt.xlabel('Std. marriage rate')
# %% Code. 5.3
m_53 = pm.Model()
with m_53:
    sigma = pm.Exponential('sigma',1)
    bA = pm.Normal('bA', mu=0, sd=0.5)
    a = pm.Normal('a', mu=0, sd=0.2)
    mu = pm.Deterministic('mu', a+bA*waffle_div['std_age'])
    divrate = pm.Normal('divrate', mu=mu, sd=sigma,
                        observed=waffle_div['std_divorce'])
    prior_samples = pm.sample_prior_predictive()
    trac_m53 = pm.sample(2000)
    map_m53 = pm.find_MAP()

# %%  plot the predictions from the priors:
plt.figure()
for a, ba in zip(prior_samples['a'][::10], prior_samples['bA'][::10]):
    y = a + ba*waffle_div['std_age']
    plt.plot(waffle_div['std_age'], y)

plt.figure()
az.plot_hdi(waffle_div['std_age'],prior_samples['divrate'],hdi_prob=0.89)
az.plot_hdi(waffle_div['std_age'],prior_samples['mu'],hdi_prob=0.89)
plt.scatter(waffle_div['std_age'], waffle_div['std_divorce'])

# %% plot the predictions from the posterior:
m53_posterior_pred = pm.sample_posterior_predictive(trac_m53, 200,
                                                    m_53,
                                                    var_names=['mu','divrate'])
plt.figure()
az.plot_hdi(waffle_div['std_age'],m53_posterior_pred['divrate'],hdi_prob=0.89)
az.plot_hdi(waffle_div['std_age'],m53_posterior_pred['mu'],hdi_prob=0.89)
plt.scatter(waffle_div['std_age'], waffle_div['std_divorce'])
plt.xlabel('Median marriage age, standardised');
plt.ylabel('Median divorce rate, standardised')

# %% m52 which explains the divorce rate only ysing the marriage rate. 
m_56 = pm.Model()
with m_56:
    sigma = pm.Exponential('sigma',1)
    bM = pm.Normal('bM', mu=0, sd=2)
    a = pm.Normal('a', mu=0, sd=0.2)
    mu = pm.Deterministic('mu', a+bM*waffle_div['std_marriage'])
    divrate = pm.Normal('divrate', mu=mu, sd=sigma,
                        observed=waffle_div['std_divorce'])
    m_56_priorsamples = pm.sample_prior_predictive()
    trac_m56 = pm.sample(2000)
    map_m56 = pm.find_MAP()


# %% 
m_56_postsamples = pm.sample_posterior_predictive(trac_m56, 200, m_56,
                                                  var_names=['mu','divrate'])
plt.figure()
az.plot_hdi(waffle_div['std_marriage'],m_56_postsamples['divrate'],hdi_prob=0.89)
az.plot_hdi(waffle_div['std_marriage'],m_56_postsamples['mu'],hdi_prob=0.89)
plt.scatter(waffle_div['std_marriage'], waffle_div['std_divorce'])
plt.xlabel('Median marriage rate, standardised');
plt.ylabel('Median divorce rate, standardised')

# %% 

from causalgraphicalmodels import CausalGraphicalModel
import daft
from theano import shared


# Making the DAG -- a bit tricky and somewhat complicated I feeell....
dag5_1 = CausalGraphicalModel(nodes=["A", "D", "M"], edges=[("A", "D"), ("A", "M"), ("M", "D")])
pgm = daft.PGM()
coordinates = {"A": (0, 0), "D": (1, 1), "M": (2, 0)}
for node in dag5_1.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag5_1.dag.edges:
    pgm.add_edge(*edge)
pgm.render()
# plt.gca().invert_yaxis()

# the second DAG 
dag5_2 = CausalGraphicalModel(nodes=["A", "D", "M"],
                              edges=[("A", "D"), ("A", "M")])
pgm2 = daft.PGM()
coordinates = {"A": (0, 0), "D": (1, 1), "M": (2, 0)}
for node in dag5_2.dag.nodes:
    pgm2.add_node(node, node, *coordinates[node])
for edge in dag5_2.dag.edges:
    pgm2.add_edge(*edge)
pgm2.render()


# %% Checking out the posterior predictions 

m_both = pm.Model()
with m_both:
    
    a = pm.Normal('a', mu=0, sd=0.2)
    bA = pm.Normal('bA', mu=0, sd=0.5)
    bM = pm.Normal('bM',mu=0, sd=0.5)
    sigma = pm.Exponential('sigma',1)
    meandiv_std = pm.Deterministic('meandiv_std',
                                a+bA*waffle_div['std_age'])+bM*waffle_div['std_marriage']
    divrate_std = pm.Normal('divrate_std', mu=meandiv_std,
                         sd=sigma, observed=waffle_div['std_divorce'])
    trace_mboth = pm.sample(2000)
# %% 
postpred_mboth = pm.sample_posterior_predictive(trace_mboth, 2000,
                                                m_both,
                                                var_names=['meandiv_std','divrate_std'])
# %% 
mean_preds = postpred_mboth['meandiv_std'].mean(axis=0)
mean_hpd = az.hdi(postpred_mboth['meandiv_std'], 0.89)
plt.figure()
plt.plot(waffle_div['std_divorce'],waffle_div['std_divorce'],
         '--',label='perfect prediction')
#plt.plot(waffle_div['std_divorce'], mean_preds,'*',label='Mean predicted')
plt.errorbar(waffle_div['std_divorce'], 
             mean_preds, 
             yerr=np.abs(mean_preds-mean_hpd.T),
             fmt="C0o",label='Mean pred. & 89%ile comp. interval')
#plt.plot(waffle_div['std_divorce'], mean_preds,'*')
#az.plot_hdi(waffle_div['std_divorce'], postpred_mboth['div_rate'])
plt.xlabel('Observed');plt.ylabel('Predicted');
plt.legend()

# %% Spurious correlation eg. 
nsamples = 100
x_real = np.random.normal(size=nsamples)
x_spur = np.random.normal(loc=x_real, size=nsamples)
y = np.random.normal(loc=x_real, size=nsamples)
simdf = pd.DataFrame(data={'y':y,'x_real':x_real, 'x_spur':x_spur})

# let's run the plots 
plt.figure()
#plt.plot(simdf['x_real'], simdf['x_spur'],'*')
plt.plot(simdf['y'], simdf['x_spur'],'*')
plt.plot(simdf['y'], simdf['x_real'],'*')

# %% 
m_realspur = pm.Model()
with m_realspur:
    b1 = pm.Normal('b1', mu=0, sd=1)
    b2 = pm.Normal('b2', mu=0, sd=1)
    a = pm.Normal('a', mu=0, sd=0.25)
    sigma = pm.Uniform('sigma', lower=0.01, upper=10)
    y_mean = pm.Deterministic('y_mean', a  + b1*simdf['x_real']+b2*simdf['x_spur'])
    y_pred = pm.Normal('y_pred', mu=y_mean, sd=sigma, observed=simdf['y'])
    trace_mrealsp = pm.sample(2000)
    mrealsp_map = pm.find_MAP()
    
# %% Counter-factual analysis
marriagestd_shared = shared(waffle_div['std_marriage'].values)
agestd_shared = shared(waffle_div['std_age'].values)


m53a = pm.Model()
with m53a:
    sigma = pm.Exponential("sigma", 1)
    bA = pm.Normal("bA", 0, 0.5)
    bM = pm.Normal("bM", 0, 0.5)

    a = pm.Normal("a", 0, 0.2)
    mu = pm.Deterministic("mu", a + bA * agestd_shared + bM * marriagestd_shared)
    divorce = pm.Normal("divorce", 
                            mu, sigma,
                            observed=waffle_div['std_divorce'])

    sigma_M = pm.Exponential("sigma_m", 1)
    bAM = pm.Normal("bAM", 0, 0.5)
    aM = pm.Normal("aM", 0, 0.2)
    mu_M = pm.Deterministic("mu_m", aM + bAM * agestd_shared)
    marriage = pm.Normal("marriage", mu_M, sigma_M, 
                         observed=marriagestd_shared)

    m53a_trace = pm.sample()

# %% Display summary of the posterior coefficient estimates 
print(az.summary(az.convert_to_dataset(m53a_trace), kind='stats'))

# %% now simulate a range of standardised median age of marriage
A_seq = np.linspace(-2, 2, 50)
A_seq.shape

agestd_shared.set_value(A_seq)

with m53a:
    m53a_post = pm.sample_posterior_predictive(m53a_trace)
# %% And now plot the predictions. 

plt.figure()
plt.subplot(121)
az.plot_hdi(A_seq, m53a_post['divorce'], 0.89)
plt.plot(A_seq, m53a_post['divorce'].mean(0))
plt.ylabel('Counterfactual Divorce')
#az.plot_hdi(A_seq, m53a_post['mu'], 0.89)
plt.subplot(122)
az.plot_hdi(A_seq, m53a_post['marriage'], 0.89)
plt.plot(A_seq, m53a_post['marriage'].mean(0))
plt.ylabel('Counterfactual Marriage')
#az.plot_hdi(A_seq, m53a_post['mu_m'], 0.89)
plt.xlabel('Manipulated Median marriage age')

# %% 
# Loading the milk dataset
milk = pd.read_csv('datasets/milk.csv', sep=';')
for new_colname, old_colname in zip(['K','N','M'], ['kcal.per.g',
                                                     'neocortex.perc','mass']):
    if old_colname != 'mass':
        milk[new_colname] = standardise(milk[old_colname])
    else:
        milk[new_colname] = standardise(np.log(milk[old_colname]))

# %% 
# Simple regression -- kilocals ~ neocortex size
all_completerows = ~pd.isna(milk['N'])
milk_nona = milk[all_completerows]

m55_draft = pm.Model()
with m55_draft:
    sigma = pm.Exponential('sigma',1)
    a = pm.Normal('a', mu=0, sd=0.2)
    b = pm.Normal('b', mu=0, sd=0.5)
    mu = pm.Deterministic('mu', a +b*milk_nona['N'])
    kcal = pm.Normal('kcal', mu=mu, sd=sigma, observed=milk_nona['K'])
    priortrace_m55drft = pm.sample_prior_predictive(var_names=['kcal'])

# %% 
plt.figure()
plt.scatter(milk_nona['N'],milk_nona['K'])
az.plot_hdi(milk_nona['N'],priortrace_m55drft['kcal'])

# %% Another  simple model: kcal ~ neocortex size

neocort_std = shared(milk_nona['N'].values)
m55 = pm.Model()
with m55:
    sigma = pm.Exponential('sigma',1)
    a = pm.Normal('a', mu=0, sd=0.2)
    b = pm.Normal('b', mu=0, sd=0.5)
    mu = pm.Deterministic('mu', a +b*neocort_std)
    kcal = pm.Normal('kcal', mu=mu, sd=sigma, observed=milk_nona['K'])
    trace_m55 = pm.sample()

# %% 
print(az.summary(trace_m55, var_names=['a','b','sigma']))

# %% Plot posterior prediction
x =  np.linspace(-2,2,50)
neocort_std.set_value(x)
with m55:
    post_m55 = pm.sample_posterior_predictive(trace_m55, var_names=['mu','kcal'])

# %% 
mean_mu =  post_m55['mu'].mean(axis=0)
plt.figure()
az.plot_hdi(x, post_m55['mu'])
az.plot_hdi(x, post_m55['kcal'])
plt.plot(x, mean_mu)
plt.scatter(milk_nona['N'], milk_nona['K'])
# %% 
# The observed data
plt.figure()
plt.scatter(milk_nona['N'], milk_nona['K'])

# %% kcal ~ body mass

m56 = pm.Model()
with m56:
    sigma = pm.Exponential('sigma',1)
    a = pm.Normal('a', mu=0, sd=0.2)
    bM = pm.Normal('bM', mu=0, sd=0.5)
    mu = pm.Deterministic('mu', a +bM*milk_nona['M'])
    kcal = pm.Normal('kcal', mu=mu, sd=sigma, observed=milk_nona['K'])
    trace_m56 = pm.sample()

# %% 
plt.figure()
plt.scatter(milk_nona['M'],milk_nona['K'])
az.plot_hdi(milk_nona['M'],trace_m56['mu'])

# %% kcal ~ neocortex + body-mass
m57 = pm.Model()
with m57:
    sigma = pm.Exponential('sigma',1)
    a = pm.Normal('a', mu=0, sd=0.2)
    bN = pm.Normal('bN', mu=0, sd=0.5)
    bM = pm.Normal('bM', mu=0, sd=0.5)
    mu = pm.Deterministic('mu', a +bM*milk_nona['M']+bN*milk_nona['N'])
    kcal = pm.Normal('kcal', mu=mu, sd=sigma, observed=milk_nona['K'])
    trace_m57 = pm.sample(2000)

# %% 
plt.figure()
plt.subplot(211)
plt.scatter(milk_nona['M'],milk_nona['K'])
az.plot_hdi(milk_nona['M'],trace_m57['mu'])
plt.subplot(212)
plt.scatter(milk_nona['N'],milk_nona['K'])
az.plot_hdi(milk_nona['N'],trace_m57['mu'])

# %% 
print(az.summary(trace_m57,var_names=['bM','bN']))

# %% Categorical variables 
howell = pd.read_csv("datasets/Howell1.csv", delimiter=";")
print(howell.head())

sex = howell["male"].values

with pm.Model() as m5_8:
    sigma = pm.Uniform("sigma", 0, 50)
    mu = pm.Normal("mu", 178, 20, shape=2)
    height = pm.Normal("height", mu[sex], sigma,
                               observed=howell["height"])
    m5_8_trace = pm.sample()

az.summary(m5_8_trace)


# %% Now to estimate the mean  for each clade :
milk_nona["clade_id"] = pd.Categorical(milk_nona["clade"]).codes


with pm.Model() as m5_9:
    sigma = pm.Exponential("sigma", 0.5)
    mu = pm.Normal("mu", 0, 0.5, shape=milk_nona['clade_id'].max()+1)
    K = pm.Normal('K', mu[milk_nona['clade_id']], sigma,
                  observed=milk_nona['K'])
    m5_9_trace = pm.sample()

print(az.summary(m5_9_trace))

# %% plot the estimate
az.plot_forest(m5_9_trace, combined=True, var_names=["mu"]);


# %% Chapter 6 : 

# R 6.2
N = 100  # number of individuals
height = np.random.normal(10, 2, N)  # sim total height of each
leg_prop = np.random.uniform(0.4, 0.5, N)  # leg as proportion of height
leg_left = leg_prop * height + np.random.normal(0, 0.02, N)  # sim left leg as proportion + error
leg_right = leg_prop * height + np.random.normal(0, 0.02, N)  # sim right leg as proportion + error

d = pd.DataFrame(
    np.vstack([height, leg_left, leg_right]).T,
    columns=["height", "leg_left", "leg_right"],
)  # combine into data frame

d.head()


# %% the leg example - regression 
m61 = pm.Model()
with m61:
    a = pm.Normal('a',mu=10,sigma=100)
    b1 = pm.Normal('b1',2,10)
    b2 = pm.Normal('b2',2,10)
    sd = pm.Exponential('sd',1)
    height_mean = a + b1*leg_left + b2*leg_right
    height = pm.Normal('height',mu=height_mean, sigma=sd, 
                                               observed = d['height'])
    m61_trace = pm.sample()

# %% summarise m61
print(az.summary(m61_trace, var_names=['a','b1','b2']))


# %% Plant example
# number of plants
N = 100
# simulate initial heights
h0 = np.random.normal(10, 2, N)
# assign treatments and simulate fungus and growth
treatment = np.repeat([0, 1], N / 2)
fungus = np.random.binomial(n=1, p=0.5 - treatment * 0.4, size=N)
h1 = h0 + np.random.normal(5 - 3 * fungus, size=N)
# compose a clean data frame
d = pd.DataFrame.from_dict({"h0": h0, "h1": h1, "treatment": treatment, "fungus": fungus})

az.summary(d.to_dict(orient="list"), kind="stats", round_to=2)

# %% 
m66 = pm.Model()
with m66:
    sd = pm.Exponential('sd',1)
    p = pm.Lognormal('p',0,0.25)
    mu = p*d.h0
    h1 = pm.Normal('h1', mu=mu, sigma=sd, observed=d['h1'])
    # m66_prior = pm.sample_prior_predictive(var_names=['p','h1'])
    m66_trace = pm.sample(var_names=['h1','p'])

# %%
plt.figure()
az.plot_hdi(d.h0,m66_trace['h1'])

# %% the full modell now 

m67 = pm.Model()
with m67:
    alpha = pm.Lognormal('alpha',0,0.25)
    bt = pm.Normal('bt',0,0.5)
    bf = pm.Normal('bf',0,0.5)
    sd = pm.Exponential('sd',1)
    p = alpha + bt*d.treatment + bf*d.fungus
    mu = p*d.h0
    h1 = pm.Normal('h1',mu=mu, sigma=sd, observed=d.h1)
    m67_trace = pm.sample()
    
# %% show the summary -- fungus is apparently *reducing* the growth
print(az.summary(m67_trace))

# %% The plant model without the outcome variable fungus

m68 = pm.Model()
with m68:
    alpha = pm.Lognormal('alpha',0,0.25)
    bt = pm.Normal('bt',0,0.5)
    sd = pm.Exponential('sd',1)
    p = alpha + bt*d.treatment 
    mu = p*d.h0
    h1 = pm.Normal('h1',mu=mu, sigma=sd, observed=d.h1)
    m68_trace = pm.sample()

# %% 
print(az.summary(m68_trace))

# %% happiness simulation
def inv_logit(x):
    return np.exp(x) / (1 + np.exp(x))


def sim_happiness(N_years=1000, seed=1234):
    np.random.seed(seed)

    popn = pd.DataFrame(np.zeros((20 * 65, 3)), columns=["age", "happiness", "married"])
    popn.loc[:, "age"] = np.repeat(np.arange(65), 20)
    popn.loc[:, "happiness"] = np.repeat(np.linspace(-2, 2, 20), 65)
    popn.loc[:, "married"] = np.array(popn.loc[:, "married"].values, dtype="bool")

    for i in range(N_years):
        # age population
        popn.loc[:, "age"] += 1
        # replace old folk with new folk
        ind = popn.age == 65
        popn.loc[ind, "age"] = 0
        popn.loc[ind, "married"] = False
        popn.loc[ind, "happiness"] = np.linspace(-2, 2, 20)

        # do the work
        elligible = (popn.married == 0) & (popn.age >= 18)
        marry = np.random.binomial(1, inv_logit(popn.loc[elligible, "happiness"] - 4)) == 1
        popn.loc[elligible, "married"] = marry

    popn.sort_values("age", inplace=True, ignore_index=True)

    return popn

# %% 
popn = sim_happiness()

popn_summ = popn.copy()
popn_summ["married"] = popn_summ["married"].astype(int)  # this is necessary before using az.summary, which doesn't work with boolean columns.
print(az.summary(popn_summ.to_dict(orient="list"), kind="stats", round_to=2))

# %% Now, let's build the model
d2 = popn_summ.loc[popn_summ['age']>=18,:]
d2['stdage'] = (d2.age - d2.age.mean())/np.std(d2.age)

# %% my own naive attempts

m000 = pm.Model()
with m000:
    a = pm.Normal('a',0,0.5)
    b = pm.Normal('b',0,0.5)
    sd = pm.Uniform('sd',lower=0.01,upper=1)
    mu_happiness =  a*d2.stdage + b*d2.married
    happiness = pm.Normal('happiness',mu_happiness,sigma=sd, 
                          observed=d2.happiness)
    m00_prior = pm.sample_prior_predictive()
    m000_trace = pm.sample()
# %% 
plt.figure()
plt.subplot(211)
plt.scatter(d2.age,d2.happiness)
az.plot_hdi(d2.age, m00_prior['happiness'])
plt.subplot(212)
plt.scatter(d2.married,d2.happiness)
az.plot_hdi(d2.married, m00_prior['happiness'])

# %% 
plt.figure()
plt.subplot(211)
plt.scatter(d2.age,d2.happiness)
az.plot_hdi(d2.age, m000_trace['happiness'])
plt.subplot(212)
plt.scatter(d2.married,d2.happiness)
az.plot_hdi(d2.married, m000_trace['happiness'])



    