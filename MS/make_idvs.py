# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 00:38:29 2015

@author: patanjali
"""

import os, time, gc
import pandas
import py7zlib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals.joblib import Parallel, delayed

start_time = time.time()

os.chdir('/home/patanjali/Kaggle/Data/MS/')

asm_datadir = '/home/patanjali/Kaggle/Data/MS/train/'

def process_bytes(bytefile, dvs, byte_vectorizer, counter):
    print counter, time.time() - start_time
    dv = dvs[dvs['Id'] == bytefile.split('.')[0]].Class.values[0]
    data = open(bytefile).readlines()
    features = byte_vectorizer.transform(data).sum(axis=0).tolist()[0]
    gc.collect()
    ret = str(dv) + ' ' + ' '.join([str(feat_id)+':'+str(features[feat_id]) for feat_id in range(len(features)) if features[feat_id] != 0]) + '\n'
    return ret

def process_asm(asmfile, dvs, asm_vectorizer, counter):
    print counter, time.time() - start_time
    dv = dvs[dvs['Id'] == asmfile.split('.')[0]].Class.values[0]
    archive = py7zlib.Archive7z(open(asm_datadir + asmfile))
    data = archive.getmembers()[0].read().split('\r\n')
    features = asm_vectorizer.transform(data).sum(axis=0).tolist()[0]
    gc.collect()
    ret = str(dv) + ' ' + ' '.join([str(feat_id)+':'+str(features[feat_id]) for feat_id in range(len(features)) if features[feat_id] != 0]) + '\n'
    return ret

if __name__ == '__main__':
    
    dvs = pandas.read_csv("trainLabels.csv")
    
    words = [('0x%02x' %(x))[2:] for x in range(2**8)]    
    train_set = words + [x + ' ' + y for x in words for y in words]    
    
    train_asm_files = os.listdir(asm_datadir)
    train_asm_libsvm = open('train_asm_1000.libsvm','w')    
    asm_files = sorted([x for x in train_asm_files if x.endswith('.asm.7z')])[:1000]
    
    asm_vectorizer = CountVectorizer(encoding = 'Latin-1', ngram_range=(1,2))
    asm_vectorizer.fit(train_set)
    
    idvs = Parallel(n_jobs=4)(delayed(process_asm)(asmfile, dvs, asm_vectorizer, counter) for counter, asmfile in enumerate(asm_files))

    for line in idvs:
        train_asm_libsvm.write(line)
    
    train_asm_libsvm.close()

    #%% Byte file analysis section
    """
    train_byte_files = os.listdir('/home/patanjali/Kaggle/Data/MS/bytefiles/')
    byte_files = sorted([x for x in train_byte_files if x.endswith('.bytes')])
    train_byte_libsvm = open('train_byte.libsvm','w')    
    
    byte_vectorizer = CountVectorizer(encoding = 'Latin-1', ngram_range=(1,2))    
    byte_vectorizer.fit(train_set)
    
    idvs = Parallel(n_jobs=1)(delayed(process_bytes)(bytefile, dvs, byte_vectorizer, counter) for counter, bytefile in enumerate(byte_files))
    
    for line in idvs:
        train_byte_libsvm.write(line)
    
    train_byte_libsvm.close()
    """