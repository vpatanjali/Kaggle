# -*- coding: utf-8 -*-
"""
Created on Sat Jan 03 19:36:05 2015

@author: Patanjali
"""

limit = 100000

f = open('train.csv')
ff = open('test/train_' + str(limit) + '.csv','w')

i = 0
for line in f:
    ff.write(line)
    i +=1
    if i > limit:
        break
    
f.close()
ff.close()