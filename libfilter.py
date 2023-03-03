### additional pip installations required ###
'''
!pip install spiketoolkit==0.6.3
!pip install spikewidgets==0.4.3
!pip install spikeextractors==0.8.4
!pip install spikesorters==0.3.2
!pip install spikecomparison==0.2.6
!pip install spikemetrics==0.2.0
!pip install MEAutility==1.4.6
!pip install spikeinterface==0.9.9
!pip install ml_ms4alg
'''

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

#### Filtering Helpers #### 
def traditionalFilt(dataset, Fs):
    '''
    Applies bandpass fitlering between 300 Hz - 7k Hz, and 60 Hz notch filter to input dataset 

        Parameters:
            dataset (2D array) : 2D array containing set of recording channel data to be bandpass filtered
            FS (float): sampling rate of recorded data in Hz
        
        Returns:
            newfilteredDataset (2D array) : 2D array of the same size as 'dataset' input that has been bandpass filtered
    '''
    newfilteredDataset = []
    order = 5
    # bandpass butterworth filtering of order 5 from 300 Hz to 7kHz
    sos = signal.butter(order, [300, 7000], fs = Fs, analog = False, btype = "bandpass", output = "sos")
    # notch filter at 60 Hz
    bandstopF = 60
    w0 = bandstopF/(Fs/2)
    Q = bandstopF/5
    b, a = signal.iirnotch(bandstopF, Q, Fs)

    # apply filters to all channels in input dataset
    for ii, chan in enumerate(dataset):
        filteredBP = signal.sosfiltfilt(sos, chan)
        filteredBPnotch = signal.filtfilt(b,a, filteredBP)
        newfilteredDataset.append(filteredBPnotch) 
       
    newfilteredDataset = np.array(newfilteredDataset)
    return newfilteredDataset

#### Adaptive Filtering
def adaptiveFiltering(channel_data, fftBand, chanInd, Fs, filterList, SPfolderPath, spikeProminenceScale = 0.02): 
    '''
    Applies adaptive filtering to input channel data
    - data should already have been bandpass filtered between 300 hz - 7 kHz, and 60 Hz notch filtered

        Parameters:
            channel_data (1D list) : timeseries of pre-filtered (bandpass and 60 Hz notch filtered) data
            fftBand (1D list) : FFT of channel_data timeseries data
            Fs (float) : sampling rate of data recording
            filterList (empty list) : empty list that will returned populated with identified peak frequencies
            SPfolderPath (string) : folder file directory where extracted data will be saved
            spikeProminenceScale (float) : value to scale mean FFT noise floor by for noise spike detection cutoff
    
        Returns:
            filtered_data (1D list) : adaptively filtered channel data timeseries
            peakList (1D list) : list of frequencies that were detected as noise and filtered out
            filteredDataFFT (1D list) : FFT of filtered_data
            meanNoisyFFT (float) : value of mean of input fftBand between 300 Hz - 7 kHz
    '''
    print("Adaptive Filtering for Chan " + str(chanInd) + "...")
    recordingLength = len(channel_data)/Fs
    channel_data_notch = channel_data

    # calculate mean of FFT between 300 Hz to 7 kHz range
    ratio = (7/int(Fs/2000)) 
    meanNoisyFFT = statistics.mean(fftBand[0:int(len(fftBand)*(ratio))])
    print("meanNoisyFFT", meanNoisyFFT)
    print("spikeProminenceScale", spikeProminenceScale)
    # calculate cutoff threshold for noise spike detection in FFT
    spikeProminenceVal = meanNoisyFFT*spikeProminenceScale
    print("spikeProminenceVal: ", spikeProminenceVal)
    
    N = len(channel_data)
    # sample spacing
    T = 1.0/Fs
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2) # sets up x-axis to be between 0 and Nyquist frequency

    fig = plt.figure()
    plt.plot(xf, fftBand)
    plt.xlabel("Hz")
    plt.ylabel("Intensity")
    plt.title("FFT of Chan " + str(chanInd))
    spikePromLine = np.full(len(fftBand), spikeProminenceVal)
    plt.plot(xf, spikePromLine, "--", color = "red")
    plt.savefig(SPfolderPath + "FFT Cutoff Chan " + str(chanInd) + ".png")
    plt.close()

    # Filter Out Isolated Peaks
    find_peakList = signal.find_peaks(fftBand, height=None, threshold=None, distance=None, prominence = spikeProminenceVal, width=None, wlen=None, rel_height=None, plateau_size=None)
    peakList = []
    for peak in find_peakList[0]:
            freq = indToFreq(peak, recordingLength)
            freqToAdd = round(freq)
            try:
                # index will exist if frequency is already in filterList
                index = filterList.index(freqToAdd)
            except ValueError:
                # ValueError => frequency not in filterList, add to filterList
                filterList.append(freqToAdd)
            peakList.append(freqToAdd)
    peakList = np.unique(peakList)

    # filterList and peakList should be equivilent as long as filterList is initially empty
    assert(len(peakList) == len(filterList))
    assert(np.array_equal(peakList, filterList))
    print("Number of Spikes Detected: ", len(peakList))
    print("Peaks Detected: ", peakList)
    
    # peakList is list of unique noise frequencies detected
    filtered_data = channel_data_notch
    for freq in peakList:
        if (freq != 0):
            bandstopF = freq
            w0 = bandstopF/(Fs/2)
            Q = bandstopF/5
            # apply notch filter to remove detected noise frequencies
            b,a = signal.iirnotch(bandstopF, Q, Fs)
            filtered_data = signal.filtfilt(b,a, filtered_data)

    # save FFT of filtered channel
    FFTdataFolderPath = SPfolderPath + "FFT Data/"
    figureTitle = "FFT of Adaptively Filtered Data Chan " + str(chanInd)
    filteredDataFFT = FFT(filtered_data, Fs, figureTitle, filePath = FFTdataFolderPath, printFigs = False, saveFigs = False)

    # return adaptively filtered channel data
    return filtered_data, peakList, filteredDataFFT, meanNoisyFFT

#### FFT Helper Functions
def FFT(data, Fs, figureTitle, filePath = '', printFigs = True, saveFigs = False):
    '''
    Returns FFT of the input data array 
        Parameters:
            data (1D array) : array of timeseries data to be converted into FFT
            FS (float) : sampling rate of recorded data array
            figureTitle (string) : name to save .png file of FFT plot
            filePath (string) : file directory path in which to save the .png file
            printFigs (bool) : option to display FFT plot
            saveFilgs (bool) : option to save FFT plot
        
        Returns:
            FFTarray (1D array) : array of FFT version of input 'data' 1D array
    '''
    # N = number of sample points
    N = len(data)
    # data sample spacing
    T = 1.0/Fs
    yf = fft(data)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2) # sets up x axis to be between 0 and Nyquist frequency
    yfRefined = 2.0/N * np.abs(yf[0:N//2]) # isolates positive frequency and takes abs of FFT values
    if (printFigs == True) or (saveFigs == True):
        fig = plt.figure(figsize = (15,10))
        plt.plot(xf, yfRefined)
        plt.ylabel("Intensity", fontsize = 15)
        plt.xlabel("Frequency [Hz]", fontsize = 15)
        plt.title(figureTitle, fontsize = 20)
        if saveFigs == True:
            plt.savefig(filePath + figureTitle + ".png")
        plt.grid()
        if printFigs == True:
            plt.show()
        plt.clf()
        plt.close()
    
    # return FFT array
    FFTarray = yfRefined
    return FFTarray

def returnFullFFT(dataset, Fs, figTitle = "FFT", folderPath = ''):
    '''
    Returns the 2D array of the FFT of the input 2D dataset
        Parameters:
            dataset (2D array) : array of timeseries dataset recording
            Fs (float) : sampling rate in Hz of timeseries dataset
            figtitle (string) : figure title used to save FFT .png file
            folderPath (string) : file directory to save plotted FFT .png files
        Returns:
            fftDataset (2D array) : FFT of input 'dataset' array
    ''' 
    print("returnFullFFT called...")
    fftDataset = []
    for ii, chan in enumerate(dataset):
        fftChan = FFT(chan, Fs, figTitle + " Chan. " + str(ii), filePath = folderPath, printFigs = True, saveFigs = True)
        fftDataset.append(fftChan)
    fftDataset = np.array(fftDataset)
    return fftDataset


#### Data Structure Helpers ####
def exportSpikes2(dataArray, unit_ids, fileName):
    '''
    Specifically saves spikes2 data as excel for computing correlation matrices
    
        Parameters:
            dataArray (array) : relevant data array to save
            unit_ids (list of ints) : list of spikes units
            fileName (string) : full filename to save excel data
        
        Returns:
            None
    '''
    fileName = unicodedata.normalize("NFKD", fileName)
    empty_data = {}
    # This assumes dataArray is in spike order
    for ii, row in enumerate(dataArray):
        empty_data[str(unit_ids[ii])] = row

    max_len = max(map(len, empty_data.values()))

    for item in empty_data.items():
        key = item[0]
        value = item[1]
        newValue = value.copy()
        newValue.resize(max_len,)
        empty_data[key] = newValue

    empty_df = pd.DataFrame.from_dict(empty_data)
    empty_df_replaced = empty_df.replace(0, np.nan)

    if os.path.exists(fileName + ".xlsx") == True:
        print("FILE EXISTS, OVERIDING: " + fileName + ".xlsx")
    else:
        print("FILE DOES NOT EXIST, CREATING: " + fileName + ".xlsx")

    empty_df_replaced.to_excel(fileName + ".xlsx")
    print("CREATED FILE: " + fileName + ".xlsx")
    return None

def indToFreq(ind, recordingLength):
    '''
    Converts index of FFT x-axis into corresponding frequency
    '''
    return (ind/recordingLength)