# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 15:52:24 2015

@author: Patanjali
"""

indicator_keys = ['hour',# - YYMMDDHH
                    'C1',# - 1000 + small integer
                    'banner_pos',# - 0 to 7
                    'site_id',# - 2825 unique entries
                    'site_domain',# 3366 unique entries
                    'site_category',# 22 unique entries
                    'app_id',# 3952 unique entries
                    'app_domain',# 201 unique entries
                    'app_category',# 28 unique entries
                    'device_id',# unique entries 6.37% of total
                    'device_ip',# unique entries 23.5% of total
                    'device_model',# 5438 unique entries
                    'device_type',# 0 to 5 int
                    'device_conn_type',# 0 to 5 int
                    'C14',# 1257 unique integers
                    'C15',# Looks like resolution's first dimension
                    'C16',# Looks like resolution's other dimension
                    'C17',# 240 unique integers
                    'C18',# 0 to 3 int
                    'C19',# 47 unique integers
                    'C20',# 100000 + 3 digit integer, 162 unique values
                    'C21',# 39 unique integers
                    ]
                    
profile_keys = ['hour',# - YYMMDDHH
                    'C1',# - 1000 + small integer
                    'banner_pos',# - 0 to 7
                    'site_id',# - 2825 unique entries
                    'site_domain',# 3366 unique entries
                    'site_category',# 22 unique entries
                    'app_id',# 3952 unique entries
                    'app_domain',# 201 unique entries
                    'app_category',# 28 unique entries
                    #'device_id',# unique entries 6.37% of total
                    #'device_ip',# unique entries 23.5% of total
                    'device_model',# 5438 unique entries
                    'device_type',# 0 to 5 int
                    'device_conn_type',# 0 to 5 int
                    'C14',# 1257 unique integers
                    'C15',# Looks like resolution's first dimension
                    'C16',# Looks like resolution's other dimension
                    'C17',# 240 unique integers
                    'C18',# 0 to 3 int
                    'C19',# 47 unique integers
                    'C20',# 100000 + 3 digit integer, 162 unique values
                    'C21',# 39 unique integers
                    ]
                    
enquiry_keys = ['hour',# - YYMMDDHH
                    'C1',# - 1000 + small integer
                    'banner_pos',# - 0 to 7
                    'site_id',# - 2825 unique entries
                    'site_domain',# 3366 unique entries
                    'site_category',# 22 unique entries
                    'app_id',# 3952 unique entries
                    'app_domain',# 201 unique entries
                    'app_category',# 28 unique entries
                    #'device_id',# unique entries 6.37% of total
                    #'device_ip',# unique entries 23.5% of total
                    'device_model',# 5438 unique entries
                    'device_type',# 0 to 5 int
                    'device_conn_type',# 0 to 5 int
                    'C14',# 1257 unique integers
                    'C15',# Looks like resolution's first dimension
                    'C16',# Looks like resolution's other dimension
                    'C17',# 240 unique integers
                    'C18',# 0 to 3 int
                    'C19',# 47 unique integers
                    'C20',# 100000 + 3 digit integer, 162 unique values
                    'C21',# 39 unique integers
                    ]
                    
numeric_keys = ['C1',# - 1000 + small integer
                    'banner_pos',# - 0 to 7
                    'device_type',# 0 to 5 int
                    'device_conn_type',# 0 to 5 int
                    'C14',# 1257 unique integers
                    'C15',# Looks like resolution's first dimension
                    'C16',# Looks like resolution's other dimension
                    'C17',# 240 unique integers
                    'C18',# 0 to 3 int
                    'C19',# 47 unique integers
                    'C20',# 100000 + 3 digit integer, 162 unique values
                    'C21',# 39 unique integers
                    ]