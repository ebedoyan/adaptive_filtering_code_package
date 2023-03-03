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

def runTraditionalFiltering(folderPath, Fs, filteredDataSet, overwrite = True):
    '''
    Runs the SpikeInterface spike sorter on the input filteredDataSet, to produce 
    the spike sorted version of recorded data using Traditional Filtering
    - input should already be passed through traditionalFilt() found in libfilter.py
    - in total, the spike sorted data would have had 300 Hz - 7 kHz bandpass filter, 60 Hz, and CMR filtering
    Saves extracted spikes for each detected unit waveform, waveforms graphs, and other
    metrics of computation

        Parameters:
            folderPath (string) : directory path to save extracted outputs 
            Fs (float) : sampling frequency of raw data collection (in Hz)
            filteredDataSet (numpy array) : 2D numpy array of 300Hz - 7kHz bandpass filtered and 60 Hz notch filtered data
            overwrite (bool) : option to overwrite existing exported data

        Returns:
            None
    '''
    print("Traditional Filtering Beginning for " + folderPath)
    if os.path.exists(folderPath) == False:
        os.mkdir(folderPath)
    
    ### PROCESS BANDPASS AND FILTERED
    # save filtered version of traditionally filtered data
    spikesFileName = folderPath + "spikes"
    print(spikesFileName)
    if os.path.exists(spikesFileName + ".xlsx") == True:
        if overwrite == False:
            return
    if (True):
        print("Applying CMR...")
        # use SpikeInterface module to extract data from numpy format
        # adds CMR to bandpass filtered and 60 Hz notch filtered data
        recording = se.NumpyRecordingExtractor(timeseries = filteredDataSet, sampling_frequency = Fs)
        recording_cmr = st.preprocessing.common_reference(recording, reference='median')
        np.save(folderPath + "traditionallyFilteredCMRData.npy", recording_cmr.get_traces(), allow_pickle = True)
        
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
        # running spike sorter and waveform generator on traditionally data using SpikeInterface module
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
        extracted_spikes = np.array([sorting_MS4.get_unit_spike_train(unit) for unit in sorting_MS4.get_unit_ids()])
        exportSpikes2(extracted_spikes, units, spikesFileName)

        #compute snrs
        snrs = st.validation.compute_snrs(sorting_MS4, recording_cmr)
        print("SNRS", snrs)
        np.save(folderPath + "snrs.npy", snrs, allow_pickle = True)
    return
