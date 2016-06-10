#!/usr/local/bin/python
from ROOT import *
#TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
#TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt

import numpy as np

#import pymc
#import signal_model as sm
from pymc3 import *
import signal_model_hierarchical as sm3
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


fitSamples = 120


#use the measured rad and grad
pcRad = 2.45
grad = 0.05


detName = "P42574A_grad%0.2f_pcrad%0.2f.conf" % (grad,pcRad)
det =  Detector(detName, 39.3, 33.8, zeroPadding=10, temperature=73., timeStep=1., numSteps=fitSamples*10)

####################################################################################################################################################################


def main(argv):
  #runRange = (11970,12009)
  #aeCutVal = 0.015

  runRange = (13420,13429)
  aeCutVal = 0.01425
  numWaveforms = 10
  
  plt.ion()
  fig = plt.figure(1)
  fig2=None
#  fig2 = plt.figure(2)

  #Instantiate and prepare the baseline remover transform:
  baseline = MGWFBaselineRemover()
  baseline.SetBaselineSamples(flatTimeSamples)
  
  waveformArray = []


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
    
    
    chanCut =  "channel == %d" % channelNumber
    energyCut = "trapECal>%f && trapECal<%f" % (1588,1594)
    #energyCut = "trapECal>%f" % 1500
    aeCut = "TSCurrent100nsMax/trapECal > %f" % aeCutVal

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
      waveform.SetLength(waveform.GetLength()-10)
      
      #for now, baseline subtract it here (should eventually be incorporated into the model.  won't be hard.  just lazy.)
      baseline.TransformInPlace(waveform)

#      det.preampRiseTime = 54
#      det.preampFallTime = 71.8 * 1000
      #MCMC fit and plot the results

      print "Waveform number %d in run %d" % (entryNumber, iRun)
      #fitWaveform(waveform, fig, fig2, iRun, entryNumber, channelNumber)
      
      np_data = waveform.GetVectorData()
      np_data = np.multiply(np_data, 1.)
      waveformArray.append(np_data)
      if len(waveformArray) >= numWaveforms: break
    if len(waveformArray) >= numWaveforms: break

  fitWaveforms(waveformArray, fig, fig2, runRange[0], 0, 0)

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




def fitWaveforms(wfs, wfFig, zoomFig, runNumber, entryNumber, channelNumber):
  print ">>> Going to fit %d waveforms" % len(wfs)

  #start by plotting all the wfs we're gonna use

  prior_num = [3.64e+09, 1.88e+17, 6.05e+15]
  prior_den = [1, 4.03e+07, 5.14e+14, 7.15e+18]

  plt.ion()
  
  plt.figure(wfFig.number)
  plt.clf()
  plt.title("Charge waveform")
  plt.xlabel("Sample number [10s of ns]")
  plt.ylabel("Raw ADC Value [Arb]")
  
  wf_to_fit_arr = []


  for wf in wfs:
#    np_data = wf.GetVectorData()
    np_data = wf

  #  lastFitSampleIdx = findTimePoint(np_data, .95)
    startGuess = findTimePoint(np_data, 0.005)
    firstFitSampleIdx = startGuess-20
    lastFitSampleIdx = firstFitSampleIdx + fitSamples
#    fitSamples = lastFitSampleIdx-firstFitSampleIdx # 1 microsecond
    t0_guess = startGuess - firstFitSampleIdx
    
    np_data_early = np_data[firstFitSampleIdx:lastFitSampleIdx]
    wfMax = np.amax(np_data_early)

    wf_to_fit_arr.append(np_data_early)

    plt.plot( np_data_early  ,color="red" )

  #make a simulated wf to plot for reference
  siggen_wf= findSiggenWaveform(10.,0,10.,80., wfMax, np_data_early, t0_guess, prior_num, prior_den)
  plt.plot( siggen_wf  ,color="blue" )
  
  
  plt.xlim(0, 150)

  doFit = 1
  value = raw_input('  --> Press s to skip,  q to quit, any other key to continue with fit\n')
  if value == 'q':
    exit(0)
  if value == 's':
    doFit = 0

  siggen_model = sm3.CreateFullDetectorModelGivenTransferFunctionOneField(det, wf_to_fit_arr, t0_guess, wfMax)
  with siggen_model:
    
    step = Metropolis()

    # for slice
    one_minute = 100#np.around(16380 / 114.4)
    one_hour = 60 * one_minute
    
    this_sample = 3600#50000
    
#    trace = sample(this_sample, step=[step1, step2], start=start)
    trace = sample(this_sample,  step = step)
    burnin = np.int(.75 * this_sample)
  
    temp =  np.median(  trace['temp'][burnin:])
    print "<<<detector temperature is %f" % temp
    
    num = [3.64e+09, 1.88e+17, 6.05e+15]
    den = [1, 4.03e+07, 5.14e+14, 7.15e+18]

#    print trace['switchpoint']

    plt.figure(wfFig.number)
    plt.clf()
    plt.title("Charge waveform")
    plt.xlabel("Sample number [10s of ns]")
    plt.ylabel("Raw ADC Value [Arb]")
    
    for ( wf_idx, wf) in enumerate(wf_to_fit_arr):
      t0 = np.around( np.median(  trace['switchpoint'][burnin:,wf_idx]), 1)
      r =             np.median(  trace['radEst'][burnin:,wf_idx])
      z =             np.median(  trace['zEst'][burnin:,wf_idx])
      phi =           np.median(  trace['phiEst'][burnin:,wf_idx])
      scale =         np.median(  trace['wfScale'][burnin:,wf_idx])
      
      print "wf number %d" % wf_idx
      print "  >> r:   %0.2f" % r
      print "  >> z:   %0.2f" % z
      print "  >> phi: %0.2f" % phi
      print "  >> e:   %0.2f" % scale
      print "  >> t0:  %0.2f" % t0
      
      fit_wf = findSiggenWaveform(r, phi, z, temp, scale, wf, t0, num, den)
      plt.plot(wf, color="r")
      plt.plot(fit_wf, color="b")

      
    plt.savefig("hierarchical_%dwaveforms.png" % len(wfs))
    plt.ioff()
    traceplot(trace)
    plt.savefig("hierarchical_%dchain.png" % len(wfs))
    plt.ion()






  value = raw_input('  --> Press q to quit, any other key to continue\n')

  if value == 'q':
    exit(1)


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
  
  #round here to fix floating point accuracy problem
  data_to_siggen_size_ratio = np.around(10. / siggen_step_size,3)
  
  if not data_to_siggen_size_ratio.is_integer():
    print "Error: siggen step size must evenly divide into 10 ns digitization period (ratio is %f)" % data_to_siggen_size_ratio
    exit(0)
  elif data_to_siggen_size_ratio < 10:
    round_places = 0
  elif data_to_siggen_size_ratio < 100:
    round_places = 1
  elif data_to_siggen_size_ratio < 1000:
    round_places = 2
  else:
    print "Error: Ben was too lazy to code in support for resolution this high"
    exit(0)

  data_to_siggen_size_ratio = np.int(data_to_siggen_size_ratio)


  siggen_step_size_ns = siggen_step_size * 1E-9
  
  t = np.arange(0, siggen_len) * siggen_step_size_ns
  
  detector.SetTemperature(temp)
  siggen_wf= detector.GetSiggenWaveform(r, phi, z, energy=2600)
  siggen_wf = np.pad(siggen_wf, (detector.zeroPadding,0), 'constant', constant_values=(0, 0))
  
  
  tout, siggen_wf, x = signal.lsim(system, siggen_wf, t)
  siggen_wf /= np.amax(siggen_wf)
  
  siggen_data = siggen_wf[detector.zeroPadding::]
  
  siggen_data = siggen_data*scale
  
  siggen_start_idx = np.int(np.around(switchpoint, decimals=round_places) * data_to_siggen_size_ratio % data_to_siggen_size_ratio)
  
  switchpoint_ceil= np.int( np.ceil(switchpoint) )
  
  out = np.zeros_like(data)
  samples_to_fill = (len(data) - switchpoint_ceil)
  
  
  sampled_idxs = np.arange(samples_to_fill, dtype=np.int)*data_to_siggen_size_ratio + siggen_start_idx
  
  verbose = 0
  if verbose:
    print "siggen step size: %f" % siggen_step_size
    print "data to siggen ratio: %f" % data_to_siggen_size_ratio
    print "switchpoint: %f" % switchpoint
    print "siggen start idx: %d" % siggen_start_idx
    print "switchpoint ceil: %d" % switchpoint_ceil
    print "final idx: %d" % ((len(data) - switchpoint_ceil)*10)
    print "samples to fill: %d" % samples_to_fill
    print sampled_idxs
    print siggen_data[sampled_idxs]

  out[switchpoint_ceil:] = siggen_data[sampled_idxs]
  
  return out



if __name__=="__main__":
    main(sys.argv[1:])


