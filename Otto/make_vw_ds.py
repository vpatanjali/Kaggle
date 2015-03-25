# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:00:52 2015

@author: patanjali
"""

import os
import random

os.chdir('/home/patanjali/K/Otto/')

DEV_PCT = 0.8

infile = open('train.csv')
dev_outfile = open('dev.vwds', 'w')
val_outfile = open('val.vwds', 'w')

header = infile.readline().strip().split(',')

dataids = range(len(header))

dataids.remove(header.index('id'))
dataids.remove(header.index('target'))
counter = 0
dv = header.index('target')

for line in infile:
    if random.random() < DEV_PCT:
        outf = dev_outfile
    else:
        outf = val_outfile
    
    counter += 1
    line = line.strip().split(',')
    outf.write(line[dv][-1] + ' | ')
    
    outf.write(' '.join([header[i]+'_'+line[i] for i in dataids])+'\n')
    if counter % 10000 == 0:
        print counter
        pass
        
infile.close()
dev_outfile.close()
val_outfile.close()