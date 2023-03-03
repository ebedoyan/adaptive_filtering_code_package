from libfilter import *

#### Python Library Imports ####

# import SpikeInterface Library Modules
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw

# import numpy and pandas
import numpy as np
import pandas as pd

# import system libraries
import os
import copy
import sys
import gc

# import matplotlib
import matplotlib
import matplotlib.pylab as plt
import matplotlib.image as mpimg
from matplotlib import cm

# import scipy 
import scipy.io
import scipy.stats
from scipy import signal
from scipy.fft import fft

# import math and stat libraries
from multiprocessing import Pool
from functools import partial
import math
from math import pi
import statistics
import unicodedata
########

def runAdaptiveFiltering(folderPath, fftBandDataset, Fs, unfilteredDataSet, spikeProminenceScale, overwrite = True):
    '''
    Runs the SpikeInterface spike sorter and Adaptive Filtering on the input filteredDataSet
    - input should already be passed through traditionalFilt() found in libfilter.py
    
    Saves extracted spikes for each detected unit waveform, waveforms graphs, and other
    metrics of computation

        Parameters:
            folderPath (string) : directory path to save extracted outputs 
            fftBandDataset (numpy array): FFT of 2D numpy array 'unfilteredDataSet'
            Fs (float) : sampling frequency of raw data collection (in Hz)
            unfilteredDataSet (numpy array) : 2D numpy array of 300Hz - 7kHz bandpass filtered and 60 Hz notch filtered data
            spikeProminenceScale (float) : value to scale mean FFT noise floor by for noise spike detection cutoff
            overwrite (bool) : option to overwrite existing exported data

        Returns:
            None
    '''
    print("Filtering Beginning for " + str(spikeProminenceScale) + " SP...")
    if os.path.exists(folderPath) == False:
        os.mkdir(folderPath)

    # get filtered data
    dataFileName = "bandpassedNotchedAdaptiveFilteredData.npy"

    ### Apply adaptive filtering to input dataset
    if os.path.exists(folderPath + dataFileName) and (overwrite == False):
        print("Loading filtered data...")
        fileName = folderPath + "spikes"
        if os.path.exists(fileName + ".xlsx") == False:
            filteredDataSet = np.load(folderPath + dataFileName, allow_pickle = True)
    else:
        print("Running data filtering...")
        filteredDataSet = []
        datasetPeakFrequencies = []
        filteredDatasetFFT = []
        meanFFTList = []
        for ii, channel_data in enumerate(unfilteredDataSet):
            # provide adaptiveFiltering() with empty list to populate with all identified noise frequencies
            filterList = []
            # pass dataset channel through adaptiveFiltering 
            filtered_channel_data, peakList, filteredDataFFT, meanFFT = \
                adaptiveFiltering(channel_data, fftBandDataset[ii], ii, Fs, filterList, folderPath, spikeProminenceScale)
            
            # add filtered data channel to 2D dataset list
            filteredDataSet.append(filtered_channel_data)
            # add peak frequencies found through adaptiveFiltering()
            datasetPeakFrequencies.append(peakList)
            # add filtered FFT
            filteredDatasetFFT.append(filteredDataFFT)
            # add meanFFT, which is the mean of the FFT intensity between 300 Hz - 7 kHz
            meanFFTList.append(meanFFT)
        
        # save filtered data
        filteredDataSet = np.array(filteredDataSet)
        print("filteredDataSet size: ", np.size(filteredDataSet))
        np.save(folderPath + dataFileName, filteredDataSet, allow_pickle = True)
        # save peak frequencies caught
        datasetPeakFrequencies = np.array(datasetPeakFrequencies)
        np.save(folderPath + "noiseFreqDetected.npy", datasetPeakFrequencies, allow_pickle = True)
        # save dataset FFT
        filteredDatasetFFT = np.array(filteredDatasetFFT)
        np.save(folderPath + "bandpassedNotchedAdaptiveFilteredDataFFT.npy", filteredDatasetFFT, allow_pickle = True)
        # save meanFFT
        meanFFTList = np.array(meanFFTList)
        np.save(folderPath + "meanFFTList.npy", meanFFTList, allow_pickle = True)
    
    ### Spike sort bandpassed, 60 Hz notch-filtered, and adaptive filtered data
    fileName = folderPath + "spikes"
    if os.path.exists(fileName + ".xlsx") == False:
        print("Applying CMR...")
        # applies CMR (last step of Adaptively Filtering)
        recording = se.NumpyRecordingExtractor(timeseries = filteredDataSet, sampling_frequency = Fs)
        recording_cmr = st.preprocessing.common_reference(recording, reference='median')
        # save adaptively filtered dataset in numpy file format
        np.save(folderPath + "adaptivelyFilteredCMRData.npy", recording_cmr.get_traces(), allow_pickle = True)

        # sortingThreshold is 5 standard deviations of noise floor
        sortingThreshold = 5.0
        # detecting negative spikes only
        detect_sign = -1
        ms4_params = ss.get_default_params('mountainsort4')
        ms4_params['detect_threshold'] = sortingThreshold
        ms4_params['detect_sign'] = detect_sign

        ms4FilePath = folderPath + 'mountainsort4_output/'
        ms4JSONfileName = "MountainSortOutput"+ '.json'
        
        print("Running MountainSort...")
        # running spike sorter and waveform generator on adaptively filtered data using SpikeInterface module
        sorting_MS4 = ss.run_mountainsort4(recording=recording_cmr, **ms4_params, output_folder = ms4FilePath)
        waveforms = st.postprocessing.get_unit_waveforms(recording_cmr, sorting_MS4, ms_before = 1, ms_after = 2)

        print("Saving MountainSort Results as Json...")
        sorting_MS4.dump_to_json(file_path= (ms4FilePath + ms4JSONfileName))
            
        # report unit number
        units = sorting_MS4.get_unit_ids() 
        print("Units", units)

        figTitle = folderPath + "unitWaveforms.png"
        fig = plt.figure(figsize = (20,20))
        sw.plot_unit_waveforms(recording_cmr, sorting_MS4, max_spikes_per_unit=None, figure = fig, ms_before = 3, ms_after = 3)
        print("Saving unit waveforms...")
        plt.savefig(figTitle)
        
        # extract xpike timings
        times = sorting_MS4._times
        labels = sorting_MS4._labels
        exported_spikes = np.array([sorting_MS4.get_unit_spike_train(unit) for unit in sorting_MS4.get_unit_ids()])
        # export extracted spike timings to input folder directory
        exportSpikes2(exported_spikes, units, fileName)

        # compute snrs
        snrs = st.validation.compute_snrs(sorting_MS4, recording_cmr)
        print("SNRS", snrs)
        np.save(folderPath + "snrs.npy", snrs, allow_pickle = True)

        print("Filtering completed for " + str(spikeProminenceScale) + " SP...")
        print()
    else: 
        print("Skipping filtering for " + str(spikeProminenceScale) + " SP..." )
        print()