# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 15:24:48 2015

@author: patanjali
"""

import sys, os

sys.path.append('/home/patanjali/K/libraries/')
import profiles


input_dir = '/home/patanjali/K/Az/train_data/'
output_dir = '/home/patanjali/K/Az/train_prof/'

prof = profiles.csvprofiles('click',['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', \
                    'app_id', 'app_domain', 'app_category', 'device_model', \
                    'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], \
                    prof_key = 'hora', smoothing_constant = 100)

files = os.walk('/home/patanjali/K/Az/train_data/').next()

for f in sorted(files[2]):
    print f
    prof.score(input_dir + f,output_dir + f, 'a')
    prof.update(input_dir + f)
