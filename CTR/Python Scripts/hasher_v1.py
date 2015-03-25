# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 17:22:53 2014

@author: Patanjali
"""

'''
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
'''


from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
import random
import os, gc

import config

"""
import cProfile, pstats, StringIO
pr = cProfile.Profile()
pr.enable()
"""
# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there


##############################################################################
# parameters #################################################################
##############################################################################

# A, paths

if os.environ['OS']  == 'Windows_NT':
    path_prefix = 'E:/Users/Patanjali/Documents/Python Scripts/'
else:
    path_prefix = '/home/patanjali/Kaggle/Data/CTR/'
    
train = path_prefix + 'train.csv'               # path to training file
test = path_prefix + 'test.csv'                 # path to testing file

# B, model
alpha = .1  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 1.     # L1 regularization, larger value means more regularized
L2 = 1.     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 26             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions
K = 100.1

# D, training/validation
subsample = 1
score_test = False
epoch = 1               # learn training data for N passes
profiletill = 23
holdafter = 29          # data after date N (exclusive) are used as validation
holdout = None          # use every N training instance for holdout validation
indicators = True
profiles = False
enquiries = False
normalize_profiles = True
numeric = False

metrics_file = path_prefix + 'alpha_%s_e_%s_D_%s_K_%s_profile_%s_holdafter_%s_indicators_%s_profiles_%s_normalize_%s_enquiries_%s_subsample_%s_nodevipprofiles.csv' \
                                %(alpha,epoch,D,K,profiletill,holdafter,indicators,profiles,normalize_profiles,enquiries,subsample)

##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, size, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        print size
        self.n = [0.] * (size)
        self.z = [0.] * (size)
        self.w = [0.] * (size)

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x, indices):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = self.w

        # wTx is the inner product of w and x
        wTx = 0.
        for i, x_i in zip(indices,x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]*x_i

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, indices,p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # update z and n
        for i, x_i in zip(indices,x):
            # gradient under logloss
            g = (p - y)*x_i
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


class data:
    def __init__(self,path,D,subsample=1):
        self.file = DictReader(open(path))
        self.D = D
        self.subsample = subsample
        if indicators:
            self.indicator_keys = config.indicator_keys
        else:
            self.indicator_keys = []
        if profiles:
		self.profile_keys = config.profile_keys
        else:
		self.profile_keys = []
        if enquiries:
	    self.enquiry_keys = config.enquiry_keys
        else:
            self.enquiry_keys = []
        if numeric:
            self.numeric_keys = config.numeric_keys
        else:
            self.numeric_keys = []
        
        self.profiles = [1e-10]*(D*profiles)
        self.profiles_all = [1e-10]*(2)
        self.enquiries = [1e-10]*(D*enquiries)
        self.enquiries_all = [1e-10]
        
        self.x = [1.]*(len(self.indicator_keys)+len(self.profile_keys)+len(self.enquiry_keys)+len(self.numeric_keys)+1)
        self.indices = [-1]*(len(self.indicator_keys)+len(self.profile_keys)+len(self.enquiry_keys)+len(self.numeric_keys)+1)
        self.indices_click = [0]*len(self.profile_keys)
        self.indices_nonclick = [0]*len(self.profile_keys)
        self.indices_today_click = [0]*len(self.profile_keys)
        self.indices_today_nonclick = [0]*len(self.profile_keys)
        #[x for x in self.file.fieldnames]
        #self.profile_keys.remove('id')
        #self.profile_keys.remove('click')
    def openFile(self,path):    
        self.file = DictReader(open(path))
    def items(self):
        ''' GENERATOR: Apply hash-trick to the original csv row
                       and for simplicity, we one-hot-encode everything
    
            INPUT:
                path: path to training or testing file
                D: the max index that we can hash to
    
            YIELDS:
                ID: id of the instance, mainly useless
                x: a list of hashed and one-hot-encoded 'indices'
                   we only need the index since all values are either 0 or 1
                y: y = 1 if we have a click, else we have y = 0
        '''
        D = self.D
        for t, row in enumerate(self.file):
            #if self.subsample < random.random():
            #    continue
            # process id
            ID = row['id']
            del row['id']
    
            # process clicks
            y = 0.
            if 'click' in row:
                if row['click'] == '1':
                    y = 1.
                del row['click']
    
            # extract date
            date = int(row['hour'][4:6])
    
            # turn hour really into hour, it was originally YYMMDDHH
            row['hour'] = row['hour'][6:]
    
            # build indices, x
            for i,key in enumerate(self.indicator_keys):
                value = row[key]
    
                # one-hot encode everything with hash trick
                index = abs(hash(key + '_' + value)) % D
                self.indices[i] = index
            i = len(self.indicator_keys)
            clicks_all = self.profiles_all[0]
            nonclicks_all = self.profiles_all[1]
            click_pct = clicks_all/(clicks_all+nonclicks_all)
            
            for j,key in enumerate(self.profile_keys):                

                index = D*indicators + j
                index_click = abs(hash(key + '_' + row[key] + '_clicks')) % D
                index_nonclick = abs(hash(key + '_' + row[key] + '_nonclicks')) % D
                index_today_click = abs(hash(key + '_' + row[key] + '_clicks_' + str(date))) % D
                index_today_nonclick = abs(hash(key + '_' + row[key] + '_nonclicks_' + str(date))) % D
                
                #x.append(log(self.profiles[index_click]+K*click_pct) - 
                #    log(self.profiles[index_nonclick]+K*(1-click_pct)))
                num = (self.profiles[index_click] - self.profiles[index_today_click] + K*click_pct)
                den = (self.profiles[index_nonclick]+self.profiles[index_click]-\
                            self.profiles[index_today_nonclick] - self.profiles[index_today_click]+K)
                """
                if den == 0 or self.profiles[index_click] < self.profiles[index_today_click] or\
                    self.profiles[index_nonclick] < self.profiles[index_today_nonclick]:
                    print ID,num,den, key,row[key],clicks_all, nonclicks_all,\
                        self.profiles[index_click], self.profiles[index_nonclick],\
                        self.profiles[index_today_click], self.profiles[index_today_nonclick]
                    raise
                """
                if normalize_profiles:
                    self.x[i+j] = num/den/click_pct
                else:
                    self.x[i+j] = num/den
                #print key, row[key], x[-1], y
                self.indices[i+j] = index
                self.indices_click[j] = index_click
                self.indices_nonclick[j] = index_nonclick
                self.indices_today_click[j] = index_today_click
                self.indices_today_nonclick[j] = index_today_nonclick
            j = len(self.profile_keys)
            for l,key in enumerate(self.enquiry_keys):
                #TODO : This throws errors, needs multiple fixes
                index = D*indicators + j + l
                index_inq = abs(hash(key+'_'+row[key]+'_views')) % D
                self.x[i+j+l] = self.enquiries[index_inq]
                self.indices[i+j+l] = index
                self.enquiries[index_inq] += 1
            l = len(self.enquiry_keys)
            for m, key in enumerate(self.numeric_keys):
                index = D*indicators + j + l + m
                self.x[i+j+l+m] = float(row[key])
                self.indices[i+j+l+m] = index
            
            yield t, date, ID, self.x, y, self.indices, self.indices_click,\
                self.indices_nonclick, self.indices_today_click, self.indices_today_nonclick
    
    def update_profiles(self,y,indices_click,indices_nonclick,indices_today_click,indices_today_nonclick ):
        if y == 1:
            self.profiles_all[0] += 1
            for index in indices_click:
                self.profiles[index] += 1
            for index in indices_today_click:
                self.profiles[index] += 1
        else:
            self.profiles_all[1] += 1
            for index in indices_nonclick:
                self.profiles[index] += 1
            for index in indices_today_nonclick:
                self.profiles[index] += 1


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

# initialize ourselves a learner
_data = data(train, D, subsample)
learner = ftrl_proximal(alpha, beta, L1, L2, D, D*indicators+len(_data.profile_keys)+len(_data.enquiry_keys)+len(_data.numeric_keys)+1, interaction)
del _data
gc.collect()
f = open(metrics_file,'w')
# start training
for e in xrange(epoch):
    p = 0.5
    loss = {}
    count = {}
    prev_date = 0
    loss[0] = 1
    count[0] = 1
    gc.collect()
    _data = data(train, D, subsample)
    for t, date, ID, x, y, indices, indices_click, indices_nonclick, indices_today_click, indices_today_nonclick in _data.items():
        #    t: just a instance counter
        # date: you know what this is
        #   ID: id provided in original data
        #    x: features
        #    y: label (click)
        if date != prev_date:
            f.write(str(prev_date) + ',' + str(loss[prev_date]/count[prev_date]) + '\n')
            f.flush()
            print prev_date, loss[prev_date]/count[prev_date], str(datetime.now() - start)
        # step 1, get prediction from learner

        if (holdafter and date > holdafter) or (holdout and t % holdout == 0):
            # step 2-1, calculate validation loss
            #           we do not train with the validation data so that our
            #           validation loss is an accurate estimation
            #
            # holdafter: train instances from day 1 to day N
            #            validate with instances from day N + 1 and after
            #
            # holdout: validate with every N instance, train with others
            p = learner.predict(x, indices)
            _data.update_profiles(y,indices_click,indices_nonclick,indices_today_click, indices_today_nonclick)
        elif (profiletill and date < profiletill):
            _data.update_profiles(y,indices_click,indices_nonclick,indices_today_click, indices_today_nonclick)
        else:
            # step 2-2, update learner with label (click) information
            p = learner.predict(x, indices)
            learner.update(x, indices, p, y)
            _data.update_profiles(y,indices_click,indices_nonclick,indices_today_click, indices_today_nonclick)
        if date in loss:
            loss[date] += logloss(p, y)
            count[date] += 1
        else:
            loss[date] = logloss(p,y)
            count[date] = 1                
        prev_date = date
    f.write(str(prev_date) + ',' + str(loss[prev_date]/count[prev_date]) + '\n')
    f.flush()
    print prev_date, loss[prev_date]/count[prev_date], str(datetime.now() - start)
    #print('Epoch %d finished, validation logloss: %f, elapsed time: %s' % (
    #    e, loss/count, str(datetime.now() - start)))
    

    # Scoring the test file and writing output to file
    submission = path_prefix + 'output_alpha_%s_e_%s_D_%s_K_%s_profile_%s_holdafter_%s_indicators_%s_profiles_%s_normalize_%s_enquiries_%s_subsample_%s_nodevipprofiles.csv' \
                                    %(alpha,e,D,K,profiletill,holdafter,indicators,profiles,normalize_profiles,enquiries,subsample)
    if score_test:
        with open(submission, 'w') as outfile:
            outfile.write('id,click\n')
            _data.subsample = 1
            _data.openFile(test)
            for t, date, ID, x, y, indices, indices_click, indices_nonclick, indices_today_click, indices_today_nonclick  in _data.items():
                p = learner.predict(x,indices)
                outfile.write('%s,%s\n' % (ID, str(p)))
    del _data    
print prev_date, str(datetime.now() - start)
f.close()

# Printing profiling stats
"""
pr.disable()
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()
"""

##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################

