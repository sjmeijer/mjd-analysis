#!/usr/local/bin/python
from ROOT import *

import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np

#import pymc
#import signal_model as sm
from pymc3 import *
import signal_model_pymc3 as sm3
from detector_model import *
from scipy import signal

#plt.style.use('presentation')

gatDataName = "mjd_run"
gatTreeName = "mjdTree"
builtDataName = "OR_run"
builtTreeName = "MGTree"
dataSetName = "surfmjd"


detectorName = "P3KJR"
channelNumber = 626

flatTimeSamples = 800

gradList = np.arange(0.01, 0.09, 0.01)
pcRadList = np.arange(1.65, 2.95, 0.1)


grad = 0.05
pcRad = 2.55#this is the actual starret value ish

fitSamples = 150
detName = "P42574A_grad%0.2f_pcrad%0.2f.conf" % (grad,pcRad)
det =  Detector(detName, 39.3, 33.8, zeroPadding=10, temperature=80., timeStep=1., numSteps=fitSamples*10)


####################################################################################################################################################################


def main(argv):
  runRange = (13420,13420)

  plt.ion()
  fig = plt.figure(1, figsize=(16, 12))
  fig2=None
#  fig2 = plt.figure(2)

  #Instantiate and prepare the baseline remover transform:
  baseline = MGWFBaselineRemover()
  baseline.SetBaselineSamples(flatTimeSamples)


  #i do this one run at a time, instead of in a chain, because it makes it easier when we want to run on large data sets and create skim files for each run
  for iRun in range( runRange[0],  runRange[1]+1):
    print 'processing run', iRun
    gatFilePath =  os.path.expandvars("$MJDDATADIR/%s/data/gatified/%s/%s%d.root" % (dataSetName, detectorName, gatDataName, iRun  ) )
    builtFilePath =  os.path.expandvars("$MJDDATADIR/%s/data/built/%s/%s%d.root" % (dataSetName, detectorName, builtDataName, iRun  ) )
    
    if not os.path.isfile(gatFilePath):
      print ">>>Skipping file " + gatFilePath
      continue
  
    gat_file = TFile.Open(gatFilePath)
    gatTree = gat_file.Get(gatTreeName)
    built_file = TFile.Open(builtFilePath)
    builtTree = built_file.Get(builtTreeName)
    
    builtTree.AddFriend(gatTree)
    
    regAECut = 0.01425
    highAeCut = 0.0177
    
    chanCut =  "channel == %d" % channelNumber
    energyCut = "trapECal>%f && trapECal<%f" % (1588,1594)
    #energyCut = "trapECal>%f" % 1500
    aeCut = "TSCurrent100nsMax/trapECal > %f" % regAECut

    cut = energyCut + " && " + chanCut + " && " + aeCut
    
    
    #print "The cuts will be: " + cut

    gatTree.SetEntryList(0)
    gatTree.Draw(">>elist", cut, "entrylist")
    elist = gDirectory.Get("elist")
    #print "Number of entries in the entryList is " + str(elist.GetN())
    gatTree.SetEntryList(elist);
    builtTree.SetEntryList(elist);
    
    for ientry in xrange( elist.GetN() ):
      entryNumber = gatTree.GetEntryNumber(ientry);
      waveform = getWaveform(gatTree, builtTree, entryNumber, channelNumber)

      #there are goofy things that happen at the end of the waveform for mod 1 data because of the presumming.  Just kill the last 5 samples
      waveform.SetLength(waveform.GetLength()-5)
      
      #for now, baseline subtract it here (should eventually be incorporated into the model.  won't be hard.  just lazy.)
      baseline.TransformInPlace(waveform)

#      det.preampRiseTime = 54
#      det.preampFallTime = 71.8 * 1000
      #MCMC fit and plot the results

      print "Waveform number %d in run %d" % (entryNumber, iRun)
      fitWaveform(waveform, fig, fig2, iRun, entryNumber, channelNumber)


####################################################################################################################################################################

def getWaveform(gatTree, builtTree, entryNumber, channelNumber):
    
    builtTree.GetEntry( entryNumber )
    gatTree.GetEntry( entryNumber )
    
    event = builtTree.event
    channelVec   = gatTree.channel
    numWaveforms = event.GetNWaveforms()

    for i_wfm in xrange( numWaveforms ):
        channel = channelVec[i_wfm]
        if (channel != channelNumber): continue
        return event.GetWaveform(i_wfm)

####################################################################################################################################################################



def plotWaveform(wfFig, np_data_early, wfScale, offset, r=19.46, phi=0.12, z=25.57, temp=79.3, num=None, den=None):

  if num is None:
    num = [3.64e+09, 1.88e+17, 6.05e+15]
  if den is None:
    den = [1, 4.03e+07, 5.14e+14, 7.15e+18]

  siggen_wf= findSiggenWaveform(r,phi,z,temp, wfScale, np_data_early, offset, num, den)

  plt.figure(wfFig.number)
  plt.clf()
  
  gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
  ax0 = plt.subplot(gs[0])
  ax1 = plt.subplot(gs[1], sharex=ax0)
  ax1.set_xlabel("Digitizer Time [ns]")
  ax0.set_ylabel("Voltage [Arb.]")
  ax1.set_ylabel("Residual")
  
  diff = siggen_wf - np_data_early
  
  t_data = np.arange(0, len(np_data_early)) * 10
  
  ax0.plot(t_data, np_data_early  ,color="red", label="Data (unfiltered)" )
  
  ax0.plot(t_data, siggen_wf  ,color="blue", label="Fit waveform"  )
  ax1.plot(np.arange(0, 10*len(diff), 10), diff  ,color="#7BAFD4" )
  first_legend = ax0.legend(loc=2)
  
  ax0.set_xlim(0, len(np_data_early)*10)

  #TODO: smarter margins?


def fitWaveform(wf, wfFig, zoomFig, runNumber, entryNumber, channelNumber):

  np_data = wf.GetVectorData()
  wfMax = np.amax(np_data)

  startGuess = findTimePoint(np_data, 0.005)
  firstFitSampleIdx = startGuess-20
  lastFitSampleIdx = firstFitSampleIdx + fitSamples
  t0_guess = startGuess - firstFitSampleIdx
  
  np_data_early = np_data[firstFitSampleIdx:lastFitSampleIdx]

  plt.ion()

  doFit = plotWaveform(wfFig, np_data_early, wfMax, t0_guess)
  
  value = raw_input('  --> Press s to skip,  q to quit, any other key to continue with fit\n')
  if value == 'q':
    exit(0)
  if value == 's':
    return 0
  
  siggen_model = sm3.CreateCheapDetectorModelFindTransferFunction(det, np_data_early, t0_guess, wfMax)
  with siggen_model:


    prior_num = [3.64e+09, 1.88e+17, 6.05e+15]
    prior_den = [1, 4.03e+07, 5.14e+14, 7.15e+18]
    
    step = Metropolis()
    
    this_sample = 1000#1*one_hour
    
    trace = sample(this_sample,  step = step)
    burnin = np.int(.75 * this_sample)


    t0 = np.around( np.median(  trace['switchpoint'][burnin:]), 1)
    r =             np.median(  trace['radEst'][burnin:])
    z =             np.median(  trace['zEst'][burnin:])
    phi =           np.median(  trace['phiEst'][burnin:])
    scale =         np.median(  trace['wfScale'][burnin:])
    temp =          np.median(  trace['temp'][burnin:])


    startVal = t0 + firstFitSampleIdx
    
    print "<<<t0 is %0.1f (startVal %0.1f, guess was %0.1f)" % (t0, startVal, firstFitSampleIdx+t0_guess)
    print "<<<r is %0.2f" % r
    print "<<<z is %0.2f" % z
    print "<<<phi is %0.2f" % phi
    print "<<<scale is %0.2f" % scale
    print "<<<temp is %0.2f" % (temp)
    
    num_1 =     np.median(  trace['num_1'][burnin:])
    num_2 =     np.median(  trace['num_2'][burnin:])
    num_3 =     np.median(  trace['num_3'][burnin:])
    den_1 =     np.median(  trace['den_1'][burnin:])
    den_2 =     np.median(  trace['den_2'][burnin:])
    den_3 =     np.median(  trace['den_3'][burnin:])

    print "<<<tf NUM is [%0.2e, %0.2e, %0.2e]" % (num_1, num_2, num_3)
    print "<<<tf DEN is [1, %0.2e, %0.2e, %0.2e]" % (den_1, den_2, den_3)
    
    num = [num_1, num_2, num_3]
    den = [1, den_1, den_2, den_3]
    
    plt.ioff()
    traceplot(trace)
    plt.savefig("chan%d_run%d_entry%d_chain_switchpoint.png" % (channelNumber, runNumber, entryNumber))
    plt.ion()

  plotWaveform(wfFig, np_data_early, scale, t0, r=r, phi=phi, z=z, temp=temp, num=num, den=den)
  plt.savefig("chan%d_run%d_entry%d_wf_switchpoint.png" % (channelNumber, runNumber, entryNumber))
  
  value = raw_input('  --> Press s to skip,  q to quit, any other key to continue with fit\n')
  if value == 'q':
    exit(0)


def findTimePoint(data, percent):
  #don't screw up the data, bro
  int_data = np.copy(data)
  int_data /= np.amax(int_data)
  return np.where(np.greater(int_data, percent))[0][0]

def findSiggenWaveform(r,phi,z,temp, scale, data, switchpoint, num, den):
  detector = det

  system = signal.lti(num, den)

  siggen_len = detector.num_steps + detector.zeroPadding
  siggen_step_size = detector.time_step_size
  data_to_siggen_size_ratio = np.int(10 / siggen_step_size)
  siggen_step_size_ns = siggen_step_size * 1E-9
  
  t = np.arange(0, siggen_len) * siggen_step_size_ns

  detector.SetTemperature(temp)
  siggen_wf= detector.GetSiggenWaveform(r, phi, z, energy=2600)
  siggen_wf = np.pad(siggen_wf, (detector.zeroPadding,0), 'constant', constant_values=(0, 0))


  tout, siggen_wf, x = signal.lsim(system, siggen_wf, t)
  siggen_wf /= np.amax(siggen_wf)
  
  siggen_data = siggen_wf[detector.zeroPadding::]
  
  siggen_data = siggen_data*scale
  
  siggen_start_idx = np.int(np.around(switchpoint, decimals=1) * 10 % 10)
  
  switchpoint_ceil= np.int( np.ceil(switchpoint) )
  
  #need to decide whether we rounded up or down?
  
  out = np.zeros_like(data)
  
#  print "switchpoint: %f" % switchpoint
#  print "siggen start idx: %d" % siggen_start_idx
#  print "switchpoint ceil: %d" % switchpoint_ceil
#  print "final idx: %d" % ((len(data) - switchpoint_ceil)*10)

  samples_to_fill = (len(data) - switchpoint_ceil)
  
#  print "samples to fill: %d" % samples_to_fill

  sampled_idxs = np.arange(samples_to_fill, dtype=np.int)*10 + siggen_start_idx
  
#  print sampled_idxs
#
#  print siggen_data[sampled_idxs]

  out[switchpoint_ceil:] = siggen_data[sampled_idxs]

  return out




if __name__=="__main__":
    main(sys.argv[1:])


