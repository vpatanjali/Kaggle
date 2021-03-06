# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:17:44 2015

@author: patanjali
"""

import os
import random

os.chdir('/home/patanjali/Kaggle/Data/Otto/')

DEV_PCT = 0.8
LIBFFM = True

infile = open('train.csv')
test_infile = open('test.csv')

if LIBFFM is True:
    suffix = '.libffm'
else:
    suffix = '.libsvm'

dev_outfile = open('dev' + suffix, 'w')
val_outfile = open('val' + suffix, 'w')
test_outfile = open('test' + suffix, 'w')

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
    _dv = int(line[dv][-1])-1
    if _dv > 8:
        print dv
    outf.write(str(_dv) + ' ')

    outf.write(' '.join(
                [header[i].split('_')[-1] + ':' + header[i].split('_')[-1] +
                ':' + line[i] for i in dataids]) + '\n')
    if counter % 10000 == 0:
        print counter

# %% Test dataset

header = test_infile.readline().strip().split(',')
dataids = range(len(header))
dataids.remove(header.index('id'))

for line in test_infile:
    line = line.strip().split(',')
    test_outfile.write(str(_dv) + ' ')
    test_outfile.write(' '.join(
                [header[i].split('_')[-1] + ':' + header[i].split('_')[-1] +
                ':' + line[i] for i in dataids]) + '\n')

infile.close()
dev_outfile.close()
val_outfile.close()
test_infile.close()
test_outfile.close()
