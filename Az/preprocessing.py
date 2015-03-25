# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 13:24:09 2015

@author: patanjali
"""

import cPickle
import csv

infile = csv.DictReader(open('train'))

categories = {}

for key in infile.fieldnames:
    if key not in ['id','click','hour']:
        categories[key] = {}

for row in infile:
    for key in categories:
        categories[key][row[key]] = 0

for key in categories:
    categories[key] = sorted(categories[key].keys())

cPickle.dump(categories,open('pickledDict.pkl','w'))