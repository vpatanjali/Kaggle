# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 18:22:37 2014

@author: Patanjali
"""

#%% Configuration variables

working_directory = './'

#%%

import pandas

class dfv:
    def __init__(self,data,key,time,idvs,half_lifes,history=None):
        if history is None:
            history = pandas.DataFrame