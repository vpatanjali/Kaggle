# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 20:07:15 2015

@author: patanjali
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 17:51:45 2015

@author: patanjali
"""

# log loss calculator for the Avazu dataset

# To generate train_dv.csv run the below

#cut -d, -f2 train train_dv.csv

import sys

part1 = open('test')
part2 = open(sys.argv[1])
outfile = open(sys.argv[1] + '.out','w')
#DEV_LIMIT = 36210029

header = part1.readline()

outfile.write('id,click'+'\n')

for line1 in part1:
    line2 = part2.readline()
    line1 = line1.strip().split(',')[0]
    outfile.write(line1 + ',' + line2)
    
part1.close()
part2.close()
outfile.close()