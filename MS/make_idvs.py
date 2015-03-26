# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 00:38:29 2015

@author: patanjali
"""

import os
import py7zlib
import pandas
from sklearn.feature_extraction.text import HashingVectorizer

os.chdir('/home/patanjali/Kaggle/Data/MS/train/')

train_files = os.listdir('/home/patanjali/Kaggle/Data/MS/train/')

dvs = pandas.read_csv("../trainLabels.csv")

train_libsvm = open('train.libsvm','w')

asm_files = [x for x in train_files if x.endswith('.asm.7z')][:1]

asm_hasher = HashingVectorizer(encoding = 'Latin-1',norm=None,non_negative=True,analyzer='char_wb',ngram_range=(1,3))

for compressed_file in asm_files:
    dv = dvs[dvs['Id'] == compressed_file.split('.')[0]].Class.values[0]
    train_libsvm.write(str(dv) + ' ')
    archive = py7zlib.Archive7z(open(compressed_file))
    data = archive.getmembers()[0].read()
    data = data.replace("\n", " ")
    data = data.replace("\t", " ")
    data = data.replace("  ", " ")
    features = asm_hasher.transform([data])
    train_libsvm.write(' '.join([str(feat_id)+':'+str(features[0,feat_id]) for feat_id in features.nonzero()[1]]) + '\n')

train_libsvm.close()

