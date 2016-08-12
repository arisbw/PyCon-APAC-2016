# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 01:21:08 2016

@author: Aris Budi Wibowo (http://github.com/arisbw)

Some of the codes are from several resources. Details in slides.
"""

#import all the libraries
%pylab inline #or import matplotlib
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.datasets import load_boston
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score

#1. DESCRIPTIVE STATISTICS

#generate array of random numbers
#first, set seed
np.random.seed(0) #the number of seed is anything you want. in this case I use 0.

s = sp.randn(10)
#OR: s = np.random.randn(100); those are exactly the same thing
print(s)

#now extract the information
print("Mean: {0:8.6f}".format(s.mean()))
print("Variance: {0:8.6f}".format(s.var()))
print("Standard Deviation: {0:8.6f}".format(s.std()))
print("Min: {0:8.6f}".format(s.min()))
print("Max: {0:8.6f}".format(s.max()))
print("Range: {0:8.6f}".format(np.ptp(s)))

#other ways
print("Mean : {0:8.6f}".format(sp.mean(s)))
print("Variance : {0:8.6f}".format(sp.var(s)))
print("Standard deviation : {0:8.6f}".format(sp.std(s)))

sp.stats.describe(s)

#using pandas
data = pd.DataFrame(data={'s': s})
data.describe()

#by default, when calculate std, pandas use ddof=1 (sample variance)
#why use sample variance? because to avoid underestimation of population standard deviation

#to extract the value that you want to use:
data.describe().iloc[0]['s']

#with .iloc[row location]['name of column']

#2. PROBABILITY DISTRIBUTIONS

#2.1. DISCRETE DISTRIBUTIONS

#Binomial Distribution
#PMF
bd1 = sp.stats.binom(40, 0.3)
bd2 = sp.stats.binom(40, 0.5)
bd3 = sp.stats.binom(40, 0.7)
k = np.arange(40)
plot(k, bd1.pmf(k), 'o-b')
hold(True)
plot(k, bd2.pmf(k), 'd-r')
plot(k, bd3.pmf(k), 's-g')
title('Binomial distribition')
legend(['p=0.3 and n=40', 'p=0.5 and n=40', 'p=0.7 and n=40'])
xlabel('X')
ylabel('P(X)')

#CDF
def binom_cdf(n, p):
    # 50 numbers between -3σ and 3σ
    x = np.arange(sp.stats.binom.ppf(0.01, n, p),
                  sp.stats.binom.ppf(0.99, n, p))
    # CDF at these values
    y = sp.stats.binom.cdf(x, n, p)

    plt.plot(x,y, color="black")
    plt.xlabel("Variate")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF for Binomial of number of observations = {0} & success probability = {1}".format(
               n, p))
    plt.draw()

binom_cdf(40,0.3)
#2.2. CONTINUOUS DISTRIBUTIONS
#Norm Distribution
#PDF
x = np.arange(-10,10,0.1)
n1 = sp.stats.norm(0,1)   # random variate
plot(x,n1.pdf(x))
xlim([-10,10])
title('Normal Distribution - PDF')

#CDF
def norm_cdf(mean=0, std=1):
    # 50 numbers between -3σ and 3σ
    x = sp.linspace(-3*std, 3*std, 50)
    # CDF at these values
    y = sp.stats.norm.cdf(x, loc=mean, scale=std)

    plt.plot(x,y, color="black")
    plt.xlabel("Variate")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF for Normal Distribution of mean = {0} & std. deviation = {1}".format(
               mean, std))
    plt.draw()
    
norm_cdf()

#3. GETTING TO THE NEXT LEVEL

#3.1. DATA IMPUTATION
dataset = load_boston()
rng = np.random.RandomState(0)

X_full, y_full = dataset.data, dataset.target
n_samples = X_full.shape[0]
n_features = X_full.shape[1]

# Estimate the score on the entire dataset, with no missing values
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_full, y_full).mean()
print("Score with the entire dataset = %.2f" % score)

# Add missing values in 75% of the lines
missing_rate = 0.75
n_missing_samples = np.floor(n_samples * missing_rate)
missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                      dtype=np.bool),
                             np.ones(n_missing_samples,
                                     dtype=np.bool)))
rng.shuffle(missing_samples)
missing_features = rng.randint(0, n_features, n_missing_samples)

# Estimate the score without the lines containing missing values
X_filtered = X_full[~missing_samples, :]
y_filtered = y_full[~missing_samples]
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_filtered, y_filtered).mean()
print("Score without the samples containing missing values = %.2f" % score)

# Estimate the score after imputation of the missing values
X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = 0
y_missing = y_full.copy()
estimator = Pipeline([("imputer", Imputer(missing_values=0,
                                          strategy="mean",
                                          axis=0)),
                      ("forest", RandomForestRegressor(random_state=0,
                                                       n_estimators=100))])
score = cross_val_score(estimator, X_missing, y_missing).mean()
print("Score after imputation of the missing values = %.2f" % score)
