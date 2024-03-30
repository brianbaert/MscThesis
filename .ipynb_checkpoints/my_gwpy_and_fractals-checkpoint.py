import numpy as np
import os
import sys
import pandas as pd
import argparse
import scipy.optimize as op
import time
from time import process_time
from datetime import datetime
#import nds2
import numba
from numba import jit
from gwpy.detector import ChannelList, Channel
from gwpy.time import tconvert
from gwpy.timeseries import TimeSeries #if this does not work, remove h5py and reinstall h5py
from gwpy.segments import Segment
from gwpy.table import GravitySpyTable

#The following code is partially based on research of Marco Cavaglia and found at https://git.ligo.org/marco-cavaglia/ligo-fractals/-/blob/main/LIGO-Fractals-v1.6

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f'{method.__name__}: {(te - ts) * 1000} ms')
        return result
    return timed

@timeit
def fetch_and_whiten_data(start, end, server, sampling_rate, channel):
    try:
        print("Getting the TimeSeries data")
        data = pd.DataFrame(columns=['time', 'value'])
        data_end = sampling_rate * (end-start)
        data_download = TimeSeries.fetch(channel, start-1, end+1,server)
        data_conditioned = data_download.whiten(4,2)
        data_conditioned = data_conditioned[sampling_rate:-sampling_rate].value
        return data_conditioned
    except ValueError:
        data_conditioned = None
        print('Data is not available')

@timeit
def make_scan(event_time, ifo):
    try:
        print("Getting the Timeseries data")
        data = TimeSeries.fetch_open_data(ifo, event_time - 10, event_time + 10)
        print("Obtaining the Q transform")
        q_data = data.q_transform(outseg=Segment(event_time - 1, event_time + 1))
    except ValueError:
        q_data = None
        print('Data surrounding this time is not available. Please try another gpstime.')

    if q_data is not None:
        print("Plotting the Q transform")
        plot = q_data.plot(figsize=[8,4])
        ax = plot.gca()
        ax.set_xscale('seconds')
        ax.set_yscale('log')
        ax.set_ylim(20,500)
        ax.set_ylabel('Frequency [Hz]')
        ax.grid(True, axis='y', which='both')
        ax.colorbar(cmap='viridis', label='Normalized Energy')
        plot.show()
    else:
        pass

def glitch_time(run='O3b',ifo='L1', n=1, address='https://zenodo.org/record/5649212/files/'):

  ifo_run = ifo + '_' + run + '.csv'
  filename = address+ ifo_run

  df = pd.read_csv(filename)
  gpstimes = df['event_time'].sample(n).tolist()

  return gpstimes

#Computes the ANAM Estimator
@timeit
def anam_function(data):
    n = len(data['time'])
    data_value = data['value']    
    matrix_differences = np.asarray([np.power(np.abs(data_value - data_value[el]),alpha) for el in np.arange(n)]).reshape(n,n)
    anam = [(k+1)**(-2/alpha)/(n-2*k)*np.sum([np.power(np.sum([matrix_differences[el+j,el-l] for j in np.arange(k+1) for l in np.arange(k+1)]),(1/alpha)) for el in np.arange(k,n-k)]) for k in np.arange(1,n//(2*decimate))]
    return anam


@jit(nopython=True,fastmath=True)
def jitted_var_function(data,n,decimate):
    osc_k = [[np.abs(np.max(data[x-k:x+k+1]) - np.min(data[x-k:x+k+1])) for x in np.arange(k,n-k)] for k in np.arange(1,n//(2*decimate))]
    var = [np.mean(np.asarray(osc_k[k])) for k in np.arange(len(osc_k))]
    return var

# Computes the VAR Estimator
@timeit
def var_function(data, decimate):
    n = len(data['time'])
    data_value = data['value']
    var = jitted_var_function(data_value,n,decimate) 
    return var

# Definition of a linear function for fit
def linear_func(x, a, b):
    return a+b*x

# Computes the linear fit of the estimator
@timeit
def fit_est(data, sampling_rate):
	y = np.log(data)
	x = np.log(np.arange(1,len(data)+1)/sampling_rate)
	fit_param,fit_cov = op.curve_fit(linear_func, x,y)
	return (x,y,fit_param[0],fit_param[1])	
