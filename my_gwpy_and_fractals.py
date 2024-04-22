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
        data_download = pd.DataFrame(columns=['time', 'value'])
        data_download = TimeSeries.fetch(channel, start, end, server)
        data_conditioned = data_download.whiten() #was originally (4,2)
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
    # Ensure data does not contain zero or negative values
    data = np.array(data, dtype=float)
    
    avg_x = np.average(data)
    data[np.isnan(data)] = avg_x
    
    y = np.log(data)
    x = np.log(np.arange(1,len(data)+1)/sampling_rate)    
    fit_param,fit_cov = op.curve_fit(linear_func, x, y)
    return (x, y, fit_param[0], fit_param[1])


@timeit
def calculate_fd_files(start, stop, server, sampling_rate, channel, step, decimate, alpha):
    data_end = sampling_rate * (stop - start)
    data = pd.DataFrame(columns=['time', 'value'])
    data_length = int((stop - start) / step)
    data_conditioned = fetch_and_whiten_data(start, stop, server, sampling_rate, channel)
    time_stamps = np.arange(0, int(data_end/2))
    time_stamps_array = np.split(time_stamps, data_length)
    for element in time_stamps_array:
        try:
            data = pd.concat([data, pd.DataFrame(pd.Series([element, data_conditioned[element]], index=['time', 'value'])).T], ignore_index=True)
        except:
            print('No data or exception')
            pass
    data_fd = pd.DataFrame(columns=['time', 'fd'])

    for chunk in range(0, data_length):
        start_chunktime = chunk * step + start
        check_valid = data.iloc[chunk].value
        if np.isnan(check_valid.all()):
            print(f"Warning, undefined data, FD is set to zero at time {start_chunktime}.")
            continue
        print(f"Computing the var estimator for data starting at {start_chunktime} time.")
        est_eval = var_function(data.iloc[chunk], decimate)
        est_fit = fit_est(est_eval, sampling_rate)
        fractal_dimension = 2.-est_fit[3]
        print(f"fd= {fractal_dimension}")
        data_fd = pd.concat([data_fd, pd.DataFrame([[start_chunktime, fractal_dimension]], columns=['time', 'fd'])], ignore_index=True) 
    return data_fd
