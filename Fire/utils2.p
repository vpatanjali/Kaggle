# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 09:46:00 2014

@author: patanjali
"""

import numpy, pandas
from matplotlib.pyplot import *

def weighted_gini(predicted,actual,weights,_plot=False,binary_dv=None):
    df = pandas.DataFrame({"actual":actual,"predicted":predicted,"weights":weights,
                           "binary":binary_dv})
    df.sort(columns = 'predicted',inplace=True,ascending=False)
    df["random"] = (df.weights / df.weights.sum()).cumsum()
    df["random_binary"] = (numpy.arange(df.shape[0])/df.shape[0])
    df["lorentz"] = (df.actual * df.weights).cumsum()/(df.actual * df.weights).sum()
    df["lorentz_binary"] = (df.binary/df.binary.sum()).cumsum()
    gini = sum(df.lorentz[1:].values * (df.random[:-1])) - \
            sum(df.lorentz[:-1].values * (df.random[1:]))
    if _plot == True:
        #plot(df["lorentz"])
        #plot(df["random"])
        figure()
        #plot(df["random"], label = 'random')
        #plot(df["random_binary"], label = 'random_binary')
        plot(df["random"],df["lorentz"])#, label = 'lorentz')
        #plot(df["lorentz_binary"], label = 'lorentz_binary')
        #legend(loc=4)
    return gini
    
def normalized_weighted_gini(predicted,actual,weights,binary_dv=None):
    return weighted_gini(predicted,actual,weights,False,binary_dv) / weighted_gini(actual,actual,weights)
    
def weighted_gini2(predicted,weighted_actual,weights):
    df = pandas.DataFrame({"weighted_actual":weighted_actual,"predicted":predicted,
                           "weights":weights})
    df.sort(columns = 'predicted',inplace=True,ascending=False)
    df["random"] = (df.weights / df.weights.sum()).cumsum()
    df["lorentz"] = (df.weighted_actual).cumsum()/(df.weighted_actual).sum()
    gini = sum(df.lorentz[1:].values * (df.random[:-1])) - \
            sum(df.lorentz[:-1].values * (df.random[1:]))
    return gini

def normalized_weighted_gini2(predicted,weighted_actual,weights):
    return weighted_gini2(predicted,weighted_actual,weights)/\
            weighted_gini2(weighted_actual,weighted_actual,weights)

def plot_ks(scores,bin_dvs,true_dvs,weights):
    df = pandas.DataFrame({"scores":scores,"bin_dvs":bin_dvs,
                           "true_dvs":true_dvs,"weights":weights})
    df.sort('scores',inplace=True,ascending=False)
    df['cumpctgoods'] = (1-df.bin_dvs).cumsum()/(1-df.bin_dvs).sum()
    df['cumpctbads'] = (df.bin_dvs).cumsum()/df.bin_dvs.sum()
    df['cumpctwbads'] = (df.true_dvs).cumsum()/df.true_dvs.sum()
    df['cumpctweights'] = (df.weights).cumsum()/df.weights.sum()
    df['pcts'] = numpy.arange(df.shape[0])/(df.shape[0]*1.0)
    plot(df.pcts,df.cumpctgoods,label='goods')
    plot(df.pcts,df.cumpctbads,label='bads')
    plot(df.pcts,df.cumpctwbads,label='wbads')
    plot(df.pcts,df.cumpctweights,label='weights')
    legend()
    grid()
    print "KS of the binary model is ", max(abs(df.cumpctgoods-df.cumpctbads))*100
    print "KS of the weighted model is ", max(abs(df.cumpctwbads-df.cumpctweights))*100
    
def wg(scorer,X,actuals,weights=None):
    predictions = scorer.predict(X)
    weights = X[:,-1]
    print weights.mean()
    return normalized_weighted_gini(predictions,actuals,weights,binary_dv=None)
#%%