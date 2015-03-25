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
import os, gc

import config

from algorithms import ftrl_proximal, data, logloss, dict_to_str

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

if os.name == 'posix':
    path_prefix = '/home/patanjali/Kaggle/Data/CTR/'
else:
    path_prefix = 'E:/Users/Patanjali/Documents/Python Scripts/'
    
train = path_prefix + 'train.csv'               # path to training file
test = path_prefix + 'test.csv'                 # path to testing file

# B, model
alphas = [0.05]  # learning rate
betas = [0.]   # smoothing parameter for adaptive learning rate
L1s = [1.]     # L1 regularization, larger value means more regularized
L2s = [0.]     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 24             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

# D, training/validation
subsample = 1
score_test = False
epoch = 1               # learn training data for N passes
profiletill = None
holdafter = 29          # data after date N (exclusive) are used as validation
holdout = None          # use every N training instance for holdout validation

data_params = {
                'indicators' : True,
                'profiles' : False,
                'enquiries' : False,
                'normalize_profiles' : True,
                'normalize_enquiries' : True,
                'numeric' : False,
                'interactions': True,
                'K' : 1000.5
                }

metrics_file = path_prefix + 'performance_' + dict_to_str(data_params) + '.csv'
idv_output_file = path_prefix + 'profiles_hash.csv'
##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

# initialize ourselves a learner
_data = data(train, D, config, **data_params)
learners = [ftrl_proximal(alpha, beta, L1, L2, D, _data.size, interaction) \
                for alpha in alphas for beta in betas for L1 in L1s for L2 in L2s]

gc.collect()
f = open(metrics_file,'w')
outf = open(idv_output_file,'w')

outf.write('ID,date,'+','.join(_data.profile_keys)+',bias,click\n')

del _data

def printf(*strings):
    string = ' '.join([str(t) for t in strings])
    print string
    f.write(string + '\n')
    f.flush()

def writeidvs(*strings):
    outf.write(','.join([str(t) for t in strings]) + '\n')
ctr = 0    
# start training
for e in xrange(epoch):
    p = [0.5 for learner in learners]
    loss = {}
    count = {}
    prev_date = 0
    loss[0] = [1 for learner in learners]
    count[0] = 1
    gc.collect()
    _data = data(train, D, config, **data_params)
    for t, date, ID, x, y, indices, indices_click, indices_nonclick, indices_today_click, \
        indices_today_nonclick, indices_enquiries in _data.items():
        #    t: just a instance counter
        # date: you know what this is
        #   ID: id provided in original data
        #    x: features
        #    y: label (click)
        ctr += 1
        if ctr %1000 == 0:
            print ctr, datetime.now() - start
        if date != prev_date:
            ws = [len([zz for zz in learner.w if zz != 0]) for learner in learners]
            for i, learner in enumerate(learners):
                printf(str(learner), loss[prev_date][i]/count[prev_date])
            i = loss[prev_date].index(min(loss[prev_date]))
            printf(prev_date, str(datetime.now() - start))
            printf("Winner of day %s is" %(prev_date),str(learners[i]))
            gc.collect()
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
            p = [learner.predict(x, indices) for learner in learners]
            _data.update_profiles(y,indices_click,indices_nonclick,indices_today_click, \
                                    indices_today_nonclick, indices_enquiries)
            #writeidvs(ID,date,','.join([str(temp) for temp in x]),y)
        elif (profiletill and date < profiletill):
            _data.update_profiles(y,indices_click,indices_nonclick,indices_today_click, \
                                    indices_today_nonclick, indices_enquiries)
        else:
            # step 2-2, update learner with label (click) information
            p = [learner.predict(x, indices) for learner in learners]
            for i, learner in enumerate(learners):
                learner.update(x, indices, p[i], y)
            _data.update_profiles(y,indices_click,indices_nonclick,indices_today_click, \
                                    indices_today_nonclick, indices_enquiries)
            #writeidvs(ID,date,','.join([str(i) for i in x]),y)
        if date in loss:
            for i in xrange(len(loss[date])):
                loss[date][i] += logloss(p[i], y)
            count[date] += 1
        else:
            loss[date] = [logloss(p[i],y) for i in xrange(len(p))]
            count[date] = 1                
        prev_date = date
    ws = [len([zz for zz in learner.w if zz != 0]) for learner in learners]
    for i, learner in enumerate(learners):
        printf(str(learner), loss[prev_date][i]/count[prev_date])
    printf(prev_date, str(datetime.now() - start))    
    i = loss[prev_date].index(min(loss[prev_date]))
    printf("Winner of day %s is" %(prev_date),str(learners[i]))
    
    # Scoring the test file and writing output to file
    submission = path_prefix + 'submission_' + dict_to_str(data_params) + '_epoch_%s.csv' %(e)
    if score_test:
        with open(submission, 'w') as outfile:
            outfile.write('id,click\n')
            _data.subsample = 1
            _data.openFile(test)
            for t, date, ID, x, y, indices, indices_click, indices_nonclick, \
                indices_today_click, indices_today_nonclick, indices_enquiries  in _data.items():
                p = learner.predict(x,indices)
                outfile.write('%s,%s\n' % (ID, str(p)))
    del _data
print prev_date, str(datetime.now() - start)
f.close()
outf.close()
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

