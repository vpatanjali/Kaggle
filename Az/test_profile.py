# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 15:24:48 2015

@author: patanjali
"""

import sys

sys.path.append('/home/patanjali/K/libraries/')

import profiles

prof = profiles.csvprofiles('click',['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', \
                    'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', \
                    'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'])

prof.update('train_100')
prof.score('train_100','train_100.prof')
