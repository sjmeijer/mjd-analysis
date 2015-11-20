#!/usr/local/bin/python
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")

import sys, os
#import pylab
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import pymc
import slowpulse_model_mc2 as sm

plt.style.use('presentation')

gatDataName = "mjd_run"
gatTreeName = "mjdTree"
builtDataName = "OR_run"
builtTreeName = "MGTree"
dataSetName = "surfmjd"
detectorName = "P3JDY"
flatTimeSamples = 600
channelNumber = 688

####################################################################################################################################################################


def main(argv):
  runRange = (6000,6360)

  plt.ion()
  fig = plt.figure(1)
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
    
    chanCut =  "channel == %d" % channelNumber
    energyCut = "trapECal>%f && trapECal<%f" % (1,2)
    
    cut = energyCut + " && " + chanCut
    #print "The cuts will be: " + cut

    gatTree.SetEntryList(0)
    gatTree.Draw(">>elist", cut, "entrylist")
    elist = gDirectory.Get("elist")
    #print "Number of entries in the entryList is " + str(elist.GetN())
    gatTree.SetEntryList(elist);
    builtTree.SetEntryList(elist);
    
    for ientry in xrange( elist.GetN() ):
      entryNumber = gatTree.GetEntryNumber(ientry);
      
      if entryNumber != 36458: continue
      
      waveform = getWaveform(gatTree, builtTree, entryNumber, channelNumber)

      #there are goofy things that happen at the end of the waveform for mod 1 data because of the presumming.  Just kill the last 5 samples
      waveform.SetLength(waveform.GetLength()-5)
      
      #for now, baseline subtract it here (should eventually be incorporated into the model.  won't be hard.  just lazy.)
      baseline.TransformInPlace(waveform)

      #MCMC fit and plot the results
      fitWaveform(waveform, fig, fig2)


####################################################################################################################################################################

def getWaveform(gatTree, builtTree, entryNumber, channelNumber):
    
    builtTree.GetEntry( entryNumber )
    gatTree.GetEntry( entryNumber )
    
    event = builtTree.event
    channelVec   = gatTree.channel
    energyVec   = gatTree.trapECal
    numWaveforms = event.GetNWaveforms()

    for i_wfm in xrange( numWaveforms ):
        channel = channelVec[i_wfm]
        if (channel != channelNumber): continue
        print "Entry Number %d, Energy %f" % (entryNumber, energyVec[i_wfm])
        return event.GetWaveform(i_wfm)

####################################################################################################################################################################
    
        
def fitWaveform(wf, wfFig, zoomFig):

  np_data = wf.GetVectorData()
  
  wfMax = np.amax(np_data)
  
  #perform the fit up to this index.  Currently set by 99% timepoint (no real point in fitting on the falling edge)
#  lastFitSampleIdx = 1100#findTimePoint(np_data, 0.99)
#  fitSamples = 300 # 1 microsecond

  lastFitSampleIdx = 1200#findTimePoint(np_data, 0.99)
  fitSamples = 800 # 1 microsecond


  firstFitSampleIdx = lastFitSampleIdx - fitSamples
  
  np_data_early = np_data[firstFitSampleIdx:lastFitSampleIdx]
  
  
  #startGuess = 800
  startGuess = 630
  t0_guess = startGuess - firstFitSampleIdx
  startVal = startGuess
  baseline_guess = 0
  sigma = 0
  
  siggen_model = pymc.Model( sm.createSignalModelSiggen(np_data_early, t0_guess, 4, baseline_guess) )
  M = pymc.MCMC(siggen_model)
  M.use_step_method(pymc.AdaptiveMetropolis, [M.slowness_sigma, M.wfScale, M.switchpoint], delay=1000)
#  M.use_step_method(pymc.AdaptiveMetropolis, [M.radEst, M.zEst, M.phiEst, M.wfScale], delay=1000)
#  M.use_step_method(pymc.DiscreteMetropolis, M.switchpoint, proposal_distribution='Normal', proposal_sd=4)
  M.sample(iter=10000)

 
  burnin = 5000

  t0 = np.around( np.median(M.trace('switchpoint')[burnin:]))
  scale =  np.median(M.trace('wfScale')[burnin:])
  sigma =  np.median(M.trace('slowness_sigma')[burnin:])
#  baselineB =  np.median(M.trace('baselineB')[burnin:])
#  baselineM =  0#np.median(M.trace('baselineM')[burnin:])
  startVal = t0 + firstFitSampleIdx
  
  
  print ">>> startVal: %d" % startVal
  print ">>> scale:    %f" % scale
  print ">>> sigma:    %f" % sigma

  # Two subplots, the axes array is 1-d
  f, axarr = plt.subplots(4, sharex=True)
  
  axarr[0].plot(M.trace('switchpoint')[:])
  axarr[1].plot(M.trace('slowness_sigma')[:])
  axarr[2].plot(M.trace('noise_sigma')[:])
  axarr[3].plot(M.trace('wfScale')[:])
  
  axarr[0].set_xlabel('MCMC Step Number')
  axarr[0].set_title('Raw MCMC Sampling')


  detZ = np.floor(41.5)/2
  detRad = np.floor(35.41)
  phiAvg = np.pi/8
  siggen_fit = sm.findSiggenWaveform(detRad, phiAvg, detZ)
  siggen_fit *= scale
  

#  out = np.arange(0, len(np_data_early), 1)
#  out = np.multiply(baselineM, out)
#  out += baselineB
  out = np.zeros(len(np_data_early))
  out[t0:] += siggen_fit[0:(len(siggen_fit) - t0)]
  out = ndimage.filters.gaussian_filter1d(out, sigma)

  plt.figure(wfFig.number)
  plt.clf()
  #plt.title("Charge waveform")
  plt.xlabel("Digitizer time [ns]")
  plt.ylabel("Raw ADC Value [Arb]")
  plt.plot(np.arange(0, len(np_data)*10, 10), np_data  ,color="red" )
  plt.xlim( firstFitSampleIdx*10, (lastFitSampleIdx+25)*10)

  plt.plot(np.arange(firstFitSampleIdx*10, lastFitSampleIdx*10, 10), out  ,color="blue" )
#  plt.plot(np.arange(0, startVal), np.zeros(startVal)  ,color="blue" )
#  plt.xlim( startVal-10, startVal+10)
#  plt.ylim(-10, 25)
  plt.axvline(x=lastFitSampleIdx*10, linewidth=1, color='r',linestyle=":")
  plt.axvline(x=startVal*10, linewidth=1, color='g',linestyle=":")



  value = raw_input('  --> Press q to quit, any other key to continue\n')

  if value == 'q':
    exit(1)



def findTimePoint(data, percent):
  #don't screw up the data, bro
  int_data = np.copy(data)
  int_data /= np.amax(int_data)
  return np.where(np.greater(int_data, percent))[0][0]





if __name__=="__main__":
    main(sys.argv[1:])


