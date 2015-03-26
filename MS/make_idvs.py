# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 00:38:29 2015

@author: patanjali
"""

import os, time, gc
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals.joblib import Parallel, delayed

start_time = time.time()

os.chdir('/home/patanjali/Kaggle/Data/MS/bytefiles/')

def process_bytes(bytefile, dvs, byte_vectorizer, counter):
    print counter, time.time() - start_time
    dv = dvs[dvs['Id'] == bytefile.split('.')[0]].Class.values[0]
    data = open(bytefile).readlines()
    features = byte_vectorizer.transform(data).sum(axis=0).tolist()[0]
    gc.collect()
    ret = str(dv) + ' ' + ' '.join([str(feat_id)+':'+str(features[feat_id]) for feat_id in range(len(features)) if features[feat_id] != 0]) + '\n'
    return ret

if __name__ == '__main__':
    
    train_files = os.listdir('/home/patanjali/Kaggle/Data/MS/bytefiles/')
    
    dvs = pandas.read_csv("../trainLabels.csv")
    
    train_libsvm = open('train_bytes.libsvm','w')
    
    """
    
    
    asm_files = sorted([x for x in train_files if x.endswith('.asm.7z')])
    
    asm_hasher = HashingVectorizer(encoding = 'Latin-1',norm=None,non_negative=True,\
                        analyzer='char_wb',ngram_range=(1,2))
    
    for counter, compressed_file in enumerate(asm_files):
        print "Processing ", counter, " of ", len(asm_files), ". Time elapsed ", time.time() - start_time
        dv = dvs[dvs['Id'] == compressed_file.split('.')[0]].Class.values[0]
        train_libsvm.write(str(dv) + ' ')
        archive = py7zlib.Archive7z(open(compressed_file))
        data = archive.getmembers()[0].read()
        features = asm_hasher.transform([data])
        train_libsvm.write(' '.join([str(feat_id)+':'+str(features[0,feat_id]) for feat_id in features.nonzero()[1]]) + '\n')
        gc.collect()
    
    """
    
    byte_files = sorted([x for x in train_files if x.endswith('.bytes')])
    
    #dictionary = {('0x%02x' %(x))[2:] : x for x in range(2**8)}
    
    words = [('0x%02x' %(x))[2:] for x in range(2**8)]
    
    train_set = words + [x + ' ' + y for x in words for y in words]
    
    byte_vectorizer = CountVectorizer(encoding = 'Latin-1', ngram_range=(1,2))
    
    byte_vectorizer.fit(train_set)
    
    idvs = Parallel(n_jobs=4)(delayed(process_bytes)(bytefile, dvs, byte_vectorizer, counter) for counter, bytefile in enumerate(byte_files))
    
    for line in idvs:
        train_libsvm.write(line)
    
    train_libsvm.close()