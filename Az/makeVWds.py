# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 18:24:19 2015

@author: patanjali
"""

for hour in ['_%.2d' %(i) for i in xrange(24)]:
    for ds in ['dev','val']:
        infile = ds + hour + '.csv'
        outfile = ds + hour + '.vwds'
        
        inf = open(infile)
        outf = open(outfile,'w')
        
        header = inf.readline().strip().split(',')
        
        dataids = range(len(header))
        
        dataids.remove(header.index('id'))
        if ds != 'test':
            dataids.remove(header.index('click'))
        dataids.remove(header.index('hour'))
        dataids.remove(header.index('device_id'))
        dataids.remove(header.index('device_ip'))
        
        counter = 0
        
        for line in inf:
            counter += 1
            line = line.strip().split(',')
            if ds == 'test':
                line.insert(1,'1')
            if int(line[1]) == 0:
                line[1] = '-1'
            outf.write(line[1] + ' | ')
            outf.write(' '.join([header[i]+'_'+line[i] for i in dataids])+'\n')
            if counter % 1000000 == 0:
                #print counter
                pass
                
        print infile, counter
        inf.close()
        outf.close()