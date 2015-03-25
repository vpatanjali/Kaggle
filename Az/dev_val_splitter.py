# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 16:05:35 2015

@author: patanjali
"""

DEV_LIMIT=36210029

infile = open('train.vwds')

outfile1 = open('dev.vwds','w')
outfile2 = open('val.vwds','w')

counter = 0

for line in infile:
    if counter < DEV_LIMIT:
        outfile1.write(line)
    else:
        outfile2.write(line)
    counter+= 1
            
infile.close()
outfile1.close()
outfile2.close()