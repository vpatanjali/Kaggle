# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 13:17:50 2015

@author: patanjali
"""

string = open('mlp.template').read()

params = {'dim' : 32,
          'dim1' : 100,
          'dim2' : 100,
          'dim_sq' : 32*32,
          'batch_size' : 100,
          'learning_rate' : 0.1,
          'epochs' : 100,}
          
import os
filelist = [x for x in os.walk('./').next()[2] if x.endswith('.yaml') and x.startswith('mlp')]

if filelist:
    max_so_far = max([int(x.split('.')[0][3:]) for x in filelist])
else:
    max_so_far = -1

outfile = open('mlp'+str(max_so_far+1)+'.yaml','w')

outfile.write(string %(params))
outfile.close()

print max_so_far+1