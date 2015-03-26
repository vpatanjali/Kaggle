# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 00:45:56 2015

@author: patanjali
"""

import os, subprocess

import logging

#%%

os.chdir('/home/patanjali/Kaggle/Data/MS/')


logger = logging.getLogger('rezip')
hdlr = logging.FileHandler('rezip.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)
logger.debug("Initializing....")

filelist = open('train.filelist','w')

subprocess.call(['7z','l','train.7z'],stdout=filelist)

filelist.close()

files = open('train.filelist').readlines()

files = [x.strip().split(' ')[-1] for x in files]

asmfiles = [x for x in files if x.endswith('.asm')]
bytefiles = [x for x in files if x.endswith('.bytes')]

def extract_and_compress(filenames, extractednames, archivename):
    logger.debug("Processing ", len(filenames), " files")
    lf = open('listfile','w')
    for filename in filenames:
        lf.write(filename+'\n')
    lf.close()
    subprocess.call(['7z', '-i@listfile', 'e', archivename])
    logger.debug(len(filenames), " files extracted")
    for filename, extractedname in zip(filenames, extractednames):
        logger.debug("Compressing ", filename)
        subprocess.call(['7z', 'a', '-t7z', filename+'.7z', extractedname])
        logger.debug("Removing ", filename)
        subprocess.call(['rm', extractedname])

batch_size = 1000
for i in xrange(len(asmfiles+bytefiles)/batch_size):
    filenames = (asmfiles+bytefiles)[i*batch_size:(i+1)*batch_size]
    extractednames = [filename.split('/')[-1] for filename in filenames]
    extract_and_compress(filenames, extractednames, 'train.7z')

#'-m0=lzma', '-mx=9', '-mfb=64', '-md=32m', '-ms=on'