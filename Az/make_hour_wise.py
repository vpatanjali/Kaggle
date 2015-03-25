# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 13:10:11 2015

@author: patanjali
"""

infile = open('train')

header = infile.readline()
header = header

outfiles = {}

for line in infile:
    row = line.strip().split(',')
    date = row[2]
    hour = date[6:]
    dv = row[1]
    
    if date[4:6] == '30':
        sample = 'val'
    else:
        sample = 'dev'
        
    line = line
    if hour+sample not in outfiles:
        outfiles[hour+sample] = open('train_data_hourly/' + sample + '_' + hour + '.csv', 'w')
        outfiles[hour+sample].write(header)
        print sample, hour
    outfiles[hour+sample].write(line)
    if hour not in outfiles:
        outfiles[hour] = open('train_data_hourly/dv_' + hour + '.csv', 'w')
        outfiles[hour].write('click\n')
    outfiles[hour].write(dv+'\n')

for outfile in outfiles:
    outfiles[outfile].close()