# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 13:18:26 2015

@author: patanjali
"""

#%%

data_dir = '/home/patanjali/Kaggle/Data/AXA/'

import os, time
import pandas
import numpy
from sklearn.externals.joblib import Parallel, delayed
global output

dirs = os.walk(data_dir)

RADIUS_THRESHOLD = 5
SMOOTHING_WINDOW_SIZE = 1

#%%

def gen_idvs(ds):
    x_range = ds.x.max() - ds.x.min()
    y_range = ds.y.max() - ds.y.min()
    x_distance = numpy.abs(ds.x.values[1:] - ds.x.values[:-1]).sum()
    y_distance = numpy.abs(ds.y.values[1:] - ds.y.values[:-1]).sum()
    total_block_distance = numpy.abs(ds.values[1:]-ds.values[:-1]).sum()
    distances = numpy.sqrt(numpy.square(ds.values[1:]-ds.values[:-1]).sum(1))
    total_distance = distances.sum()
    total_displacement = numpy.sqrt(numpy.square(ds.values[-1,:]-ds.values[0,:]).sum())
    disp_dist_ratio = total_displacement/total_distance
    trip_duration = ds.shape[0]
    avg_x_speed = x_distance/trip_duration
    avg_y_speed = y_distance/trip_duration
    avg_speed = total_distance/trip_duration
    
    smooth_ds = pandas.rolling_mean(ds, SMOOTHING_WINDOW_SIZE).ix[SMOOTHING_WINDOW_SIZE-1:]
    speeds = numpy.sqrt(numpy.square(smooth_ds.values[1:]-smooth_ds.values[:-1]).sum(1))
    accelerations = speeds[1:]-speeds[:-1]
    jerks = accelerations[1:]-accelerations[:-1]
    speed_quantiles = numpy.percentile(speeds,range(1,100))
    acc_quantiles = numpy.percentile(accelerations,range(1,100))
    jerk_quantiles = numpy.percentile(jerks,range(1,100))
    
    # Lalala
    
    velocities = smooth_ds.values[1:] - smooth_ds.values[:-1]
    acc = velocities[1:] - velocities[:-1]
    disp_acc_unit = smooth_ds.values[2:] - smooth_ds.values[:-2]
    disp_acc_len = numpy.sqrt(numpy.square(disp_acc_unit).sum(1))    
    speeds = numpy.square(velocities).sum(1)
    speeds = (speeds[1:]+speeds[:-1])/2
    
    _turns = numpy.where(disp_acc_len!=0)
    disp_acc_unit = disp_acc_unit[_turns]
    disp_acc_len = disp_acc_len[_turns]
    acc = acc[_turns]
    speeds = speeds[_turns]
    
    disp_acc_unit[:,0] = disp_acc_unit[:,0]/disp_acc_len
    disp_acc_unit[:,1] = disp_acc_unit[:,1]/disp_acc_len
    tang_acc = numpy.sqrt(numpy.square((acc*disp_acc_unit)).sum(1))
    radial_acc = numpy.sqrt(numpy.square(acc[:,0]*disp_acc_unit[:,1])+\
                        numpy.square(acc[:,1]*-1*disp_acc_unit[:,0]))
    
    turning_radii = speeds/radial_acc
    _turns = numpy.where(turning_radii<RADIUS_THRESHOLD)
    is_turn = turning_radii<RADIUS_THRESHOLD
    num_turns = numpy.abs(is_turn[1:]-is_turn[:-1]).sum()
    turning_radii = turning_radii[_turns]
    tang_acc = tang_acc[_turns]
    radial_acc = radial_acc[_turns]
    if len(tang_acc)>0:
        turning_radii_percentiles = numpy.percentile(turning_radii,range(1,100))
        tang_acc_quantiles = numpy.percentile(tang_acc,range(1,100))
        radial_acc_quantiles = numpy.percentile(radial_acc,range(1,100))
    else:
        turning_radii_percentiles = numpy.zeros((99,))
        tang_acc_quantiles = numpy.zeros((99,))
        radial_acc_quantiles = numpy.zeros((99,))
    
    return [x_range,y_range,x_distance,y_distance,total_block_distance,\
            total_distance,total_displacement,disp_dist_ratio,
            trip_duration,avg_x_speed,avg_y_speed,avg_speed,num_turns] +\
            list(speed_quantiles) + list(acc_quantiles) + list(jerk_quantiles) +\
            list(tang_acc_quantiles) + list(radial_acc_quantiles) +\
            list(turning_radii_percentiles)

def gen_output(filelist):
    global output
    driver = filelist[0].split('/')[-1]
    idvs = []
    for i in xrange(1,201):
        ds = pandas.read_csv(filelist[0] + '/' + str(i) + '.csv')
        idvs.append([driver + '_' + str(i), driver] + [str(x) for x in gen_idvs(ds)])
    return idvs

output = open('idvs_smoothing_2_tr_5.csv','w')

output.write('id,dv,'+\
                'x_range,y_range,x_distance,y_distance,total_block_distance,'+\
                'total_distance,total_displacement,disp_dist_ratio,'+\
                'trip_duration,avg_x_speed,avg_y_speed,avg_speed,num_turns,'+\
                ','.join(['speed_' + str(x) for x in range(1,100)])+','+\
                ','.join(['acc_' + str(x) for x in range(1,100)])+','+\
                ','.join(['jerk_' + str(x) for x in range(1,100)])+','+\
                ','.join(['tang_acc_' + str(x) for x in range(1,100)])+','+\
                ','.join(['rad_acc_' + str(x) for x in range(1,100)])+','+\
                ','.join(['turning_rad_' + str(x) for x in range(1,100)])+'\n')

#%%

dirlist = [filelist for filelist in dirs if not filelist[1]]

#%%

start_time = time.time()

for n in xrange(int(len(dirlist)/16)+1):
    ret = Parallel(n_jobs = 4)(delayed(gen_output)(filelist) for filelist in dirlist[16*n:16*(n+1)])
    print n, time.time() - start_time
    for driver in ret:
        for line in driver:
            output.write(','.join(line) + '\n')

output.close()

#%%

# Route matching - translation, rotation, mirror image
