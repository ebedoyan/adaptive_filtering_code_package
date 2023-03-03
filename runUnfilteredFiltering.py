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

def runUnfilteredFiltering(folderPath, Fs, rawDataset, overwrite = True):
    '''
    Runs the SpikeInterface spike sorter on the input rawDataset
    
    Saves extracted spikes for each detected unit waveform, waveforms graphs, and other
    metrics of computation

        Parameters:
            folderPath (string) : directory path to save extracted outputs 
            Fs (float) : sampling frequency of raw data collection (in Hz)
            rawDataset (numpy array) : 2D numpy array of raw data recording
            overwrite (bool) : option to overwrite existing exported data

        Returns:
            None
    '''
    print("Unfiltered Spike Sorting for " + folderPath)
    
    # creates existing path to store data outputs
    if os.path.exists(folderPath) == False:
        os.mkdir(folderPath)

    ### PROCESS UNFILTERED DATA ###
    fileName = folderPath + "spikes"
    print(fileName)
    if os.path.exists(fileName + ".xlsx") == True:
        if overwrite == False:
            return
    else:
        # use SpikeInterface module to extract raw data from numpy format
        extracted_recording = se.NumpyRecordingExtractor(timeseries = rawDataset, sampling_frequency = Fs)
        print("No CMR...")
        np.save(folderPath + "noCMRData.npy", extracted_recording.get_traces(), allow_pickle = True)

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
        # running spike sorter and waveform generator on unfiltered data using SpikeInterface module
        sorting_MS4 = ss.run_mountainsort4(recording = extracted_recording, **ms4_params, output_folder = ms4FilePath)
        waveforms = st.postprocessing.get_unit_waveforms(extracted_recording, sorting_MS4, ms_before = 1, ms_after = 2)

        print("Saving MountainSort Results as Json...")
        sorting_MS4.dump_to_json(file_path = (ms4FilePath + ms4JSONfileName))
        
        # report unit number
        units = sorting_MS4.get_unit_ids() 
        print("Units", units)

        figTitle = folderPath + "unitWaveforms.png"
        fig = plt.figure(figsize = (20,20))
        sw.plot_unit_waveforms(extracted_recording, sorting_MS4, max_spikes_per_unit=None, figure = fig, ms_before = 3, ms_after = 3)
        print("Saving unit waveforms...")
        plt.savefig(figTitle)

        # extract xpike timings
        times = sorting_MS4._times
        labels = sorting_MS4._labels
        extracted_spikes = np.array([sorting_MS4.get_unit_spike_train(unit) for unit in sorting_MS4.get_unit_ids()])

        # export extracted spike timings to input folder directory
        exportSpikes2(extracted_spikes, units, fileName)

        #compute snrs
        snrs = st.validation.compute_snrs(sorting_MS4, extracted_recording)
        print("SNRS", snrs)
        np.save(folderPath + "snrs.npy", snrs, allow_pickle = True)
    return 

