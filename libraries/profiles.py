# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 14:44:40 2015

@author: patanjali
"""

from csv import DictReader, DictWriter

class csvprofiles:
    
    def __init__(self, dv, keys = None, prof_key = None, smoothing_constant = 0, 
                 profiling_type = 'dict', hashsize = 2**24, state = None):
        
        self.profiling_type = profiling_type
        self.hashsize = hashsize
        self.prof_key = prof_key
        self.prof_prefix = ''
        if self.prof_key:
            self.prof_prefix = self.prof_key + '_'
        
        self.keys = keys
        self.dv = dv
        self.smoothing_constant = smoothing_constant
        
        if profiling_type == 'dict':
            self.score = self.score_dict
            self.update = self.update_dict
        
        self.__initialize_state(state)        
        
    def __initialize_state(self, state):
        
        if state == None:
            if self.profiling_type == 'dict':
                self.profile_dict = {}
                for key in self.keys:
                    self.profile_dict[key] = {}
                    self.profile_dict['overall'] = {}
            elif self.profiling_type == 'hash':
                self.profile_hash = [0. for i in xrange(self.hashsize)]*2
        else:
            if self.profiling_type == 'dict':
                self.profile_dict = state['dict']
            elif self.profiling_type == 'hash':
                self.profile_hash = state['hash']
    
    def get_state(self):
        
        if self.profiling_type == 'dict':
            return {'dict':self.profile_dict}
        elif self.profiling_type == 'hash':
            return {'hash':self.profile_hash}    

    def score_dict(self, inputfile, outputfile, mode = 'w'):
        
        input_data = DictReader(open(inputfile))
        outfile = open(outputfile, mode)
        output_data = DictWriter(outfile, [self.prof_prefix + temp for temp in self.keys] + [self.dv])
        if mode in ['w', 'wb']:
            output_data.writeheader()
        output_row = {}
        
        for input_row in input_data:
            if self.prof_key:
                profile_key = input_row[self.prof_key] + '_'
            else:
                profile_key = ''
            if input_row[self.prof_key] in self.profile_dict['overall']:
                all_buf = self.profile_dict['overall'][input_row[self.prof_key]]
            else:
                all_buf = [0, 0]
            
            for key in self.keys:
                lookup_key = profile_key + input_row[key]
                if input_row[key] in self.profile_dict[key]:
                    buf = self.profile_dict[key][lookup_key]
                else:
                    buf = [0,0]
                
                if all_buf[1] > 0:
                    num = buf[0] + (self.smoothing_constant*1.0*all_buf[0])/all_buf[1]
                else:
                    num = buf[0]*1.0
                    
                den = buf[1] + self.smoothing_constant
                
                output_row[self.prof_prefix + key] = num/den

            output_row[self.dv] = input_row[self.dv]
            output_data.writerow(output_row)
        outfile.close()

    def update_dict(self, inputfile):
        
        input_data = DictReader(open(inputfile))
        
        for input_row in input_data:
            if self.prof_key:
                profile_key = input_row[self.prof_key] + '_'
            else:
                profile_key = ''
            dv = int(input_row[self.dv])
            
            if input_row[self.prof_key] in self.profile_dict['overall']:
                self.profile_dict['overall'][input_row[self.prof_key]][0] += dv
                self.profile_dict['overall'][input_row[self.prof_key]][1] += 1
            else:
                self.profile_dict['overall'][input_row[self.prof_key]] = [dv, 1]
                
            for key in self.keys:
                lookup_key = profile_key + input_row[key]
                if lookup_key in self.profile_dict[key]:
                    self.profile_dict[key][lookup_key][0] += dv
                    self.profile_dict[key][lookup_key][1] += 1
                else:
                    self.profile_dict[key][lookup_key] = [dv, 1]