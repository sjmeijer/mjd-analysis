#!/usr/local/bin/python
import sys, os
import numpy as np

from scipy import ndimage


#funnyEntries = [4110, 66905]

class Waveform:
  def __init__(self, waveform_data, channel_number, run_number, entry_number, baseline_mean, baseline_rms,
                energy=None, timeSinceLast=None, energyLast=None, current=None):
    self.waveformData = waveform_data
    self.channel = channel_number
    self.runNumber = run_number
    self.entry_number = entry_number
    self.baselineMean = baseline_mean
    self.baselineRMS = baseline_rms
    self.timeSinceLast = timeSinceLast
    self.energyLast = energyLast
    self.energy = energy
    self.current = current


  def WindowWaveform(self, numSamples, earlySamples=20, t0riseTime = 0.005, rmsMult=1):
    '''Windows to a given number of samples'''
    self.wfMax = np.amax(self.waveformData)

    startGuess = self.EstimateT0(rmsMult)
    firstFitSampleIdx = startGuess-earlySamples
    lastFitSampleIdx = firstFitSampleIdx + numSamples

    np_data_early = self.waveformData[firstFitSampleIdx:lastFitSampleIdx]

    self.t0Guess = earlySamples
    self.windowedWf = np_data_early
    self.wfLength = len(np_data_early)

  def WindowWaveformTimepoint(self, earlySamples=20, fallPercentage=None, rmsMult=1):
    '''Does "smart" windowing by guessing t0 and wf max'''
    self.wfMax = np.amax(self.waveformData)

    startGuess = self.EstimateT0(rmsMult)
    firstFitSampleIdx = startGuess-earlySamples

    lastFitSampleIdx = self.EstimateFromMax(fallPercentage)

    np_data_early = self.waveformData[firstFitSampleIdx:lastFitSampleIdx]

    self.t0Guess = earlySamples
    self.windowedWf = np_data_early
    self.wfLength = len(np_data_early)

  def WindowWaveformAroundTimepoint(self, earlySamples=200,  timePoint=0.5, fallPercentage=None, rmsMult=2, numSamples=None):
    '''Does "smart" windowing by guessing t0 and wf max'''

    self.wfMax =  np.amax(self.waveformData)
    self.tp_idx = np.argmax(self.waveformData>= timePoint * self.wfMax)
    t0Guess = self.EstimateT0(rmsMult)

    firstFitSampleIdx = self.tp_idx-earlySamples

    if fallPercentage is not None:
        lastFitSampleIdx = self.EstimateFromMax(fallPercentage)
    elif numSamples is not None:
        lastFitSampleIdx = firstFitSampleIdx + numSamples

    np_data_early = self.waveformData[firstFitSampleIdx:lastFitSampleIdx]

    self.t0Guess = t0Guess - firstFitSampleIdx
    self.windowedWf = np_data_early
    self.wfLength = len(np_data_early)

  def WindowWaveformAroundMax(self, earlySamples=200, fallPercentage=None, rmsMult=2):
    '''Does "smart" windowing by guessing t0 and wf max'''
    self.wfMaxIdx = np.argmax(self.waveformData)
    self.wfMax =    np.amax(self.waveformData)
    t0Guess = self.EstimateT0(rmsMult)
    firstFitSampleIdx = self.wfMaxIdx-earlySamples
    lastFitSampleIdx = self.EstimateFromMax(fallPercentage)

    np_data_early = self.waveformData[firstFitSampleIdx:lastFitSampleIdx]

    self.t0Guess = t0Guess - firstFitSampleIdx
    self.windowedWf = np_data_early
    self.wfLength = len(np_data_early)

  def EstimateT0(self, rmsMult=1):
    smoothed_wf = ndimage.filters.gaussian_filter1d(self.waveformData, 2, )

    return np.where(np.less(smoothed_wf, rmsMult*self.baselineRMS))[0][-1]

  def EstimateFromMax(self, fallPercentage=None):

    if fallPercentage is None:
      searchValue =  self.wfMax - self.baselineRMS
    else:
      searchValue = fallPercentage * self.wfMax
    return np.where(np.greater(self.waveformData, searchValue))[0][-1]


########################################################################

def findTimePoint(data, percent, timePointIdx=0):

  #don't screw up the data, bro
  int_data = np.copy(data)
  int_data /= np.amax(int_data)

#  print "finding percent %0.4f" % percent
#  print np.where(np.greater(int_data, percent))[0]
#  print np.where(np.greater(int_data, percent))[0][timePointIdx]

  if timePointIdx == 0:
    #this will only work assuming we don't hit
#    maxidx = np.argmax(int_data)
    return np.where(np.less(int_data, percent))[0][-1]

  if timePointIdx == -1:
    return np.where(np.greater(int_data, percent))[0][timePointIdx]

  else:
    print( "timepointidx %d is not supported" % timePointIdx)
    exit(0)
