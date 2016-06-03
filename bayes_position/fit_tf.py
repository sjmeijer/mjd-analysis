#!/usr/local/bin/python
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt

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


det =  Detector("P42574A_grad0.05_pcrad2.25.conf", 39.3, 33.8, 66.5* CLHEP.ns, 71.8*CLHEP.us, zeroPadding=500, temperature=80.)
flatTimeSamples = 800


####################################################################################################################################################################


def main(argv):
  runRange = (13420,13420)

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



def plotWaveform(wfFig, np_data_early, wfScale, offset):

  plt.figure(wfFig.number)
  plt.clf()
  plt.title("Charge waveform")
  plt.xlabel("Sample number [10s of ns]")
  plt.ylabel("Raw ADC Value [Arb]")
  
  plt.plot( np_data_early  ,color="red" )

  
  siggen_wf= det.GetSiggenWaveform(10, 0, 10, energy=1)
  #siggen_fit = det.ProcessWaveform(siggen_fit)
  siggen_wf = np.pad(siggen_wf, (det.zeroPadding,0), 'constant', constant_values=(0, 0))
  
  num = [-1.089e10,  5.863e17,  6.087e15]
  den = [1,  3.009e07, 3.743e14,5.21e18]
  system = signal.lti(num, den)
  t = np.arange(0, len(siggen_wf)*10E-9, 10E-9)
  tout, siggen_wf, x = signal.lsim(system, siggen_wf, t)

#  siggen_wf /= np.amax(siggen_wf)


  siggen_wf /= np.amax(siggen_wf)
  
  siggen_wf *= wfScale
  siggen_wf = siggen_wf[500::]
  plt.plot(np.arange(offset, len(siggen_wf)+offset), siggen_wf  ,color="blue" )
#  plt.axvline(x=lastFitSampleIdx, linewidth=1, color='r',linestyle=":")
#  plt.axvline(x=startGuess, linewidth=1, color='g',linestyle=":")

  plt.xlim(0, len(np_data_early))

  value = raw_input('  --> Press s to skip,  q to quit, any other key to continue with fit\n')
  if value == 'q':
    exit(0)
  if value == 's':
    return 0
  return 1



def fitWaveform(wf, wfFig, zoomFig, runNumber, entryNumber, channelNumber):

  np_data = wf.GetVectorData()
  wfMax = np.amax(np_data)

  #perform the fit up to this index.  Currently set by 99% timepoint (no real point in fitting on the falling edge)
  lastFitSampleIdx = np.argmax(np_data)+100

#  lastFitSampleIdx = findTimePoint(np_data, .95)
  startGuess = findTimePoint(np_data, 0.005)
  firstFitSampleIdx = startGuess-20
  fitSamples = lastFitSampleIdx-firstFitSampleIdx # 1 microsecond
  t0_guess = startGuess - firstFitSampleIdx
  
  np_data_early = np_data[firstFitSampleIdx:lastFitSampleIdx]

  plt.ion()

  doFit = plotWaveform(wfFig, np_data_early, wfMax, t0_guess)

  if not doFit:
    return 0
  
  siggen_model = sm3.CreateTransferFunctionModel(det, np_data_early, t0_guess, wfMax)
  with siggen_model:

    step = Metropolis()
    

    # for slice
    one_minute = 125#np.around(16380 / 114.4)
    one_hour = 60 * one_minute
    
    this_sample = 6*one_hour
    
    trace = sample(this_sample, step)
    burnin = .75 * this_sample


    t0 = np.around( np.median(  trace['switchpoint'][burnin:]))
    r =             np.median(  trace['radEst'][burnin:])
    z =             np.median(  trace['zEst'][burnin:])
    phi =           np.median(  trace['phiEst'][burnin:])
    scale =         np.median(  trace['wfScale'][burnin:])
    temp =          np.median(  trace['temp'][burnin:])
    


    startVal = t0 + firstFitSampleIdx
    
    print "<<<startVal is %d" % startVal
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

    plt.ioff()
    traceplot(trace)
    plt.savefig("chan%d_run%d_entry%d_chain.png" % (channelNumber, runNumber, entryNumber))
    plt.ion()
  

  
  print ">>> t0guess was %d" % (firstFitSampleIdx+t0_guess)
  print ">>> fit t0 was %d" % (firstFitSampleIdx + t0)

  det.SetTemperature(temp)

  siggen_wf= det.GetSiggenWaveform(r, phi, z, energy=2600)
  siggen_wf = np.pad(siggen_wf, (det.zeroPadding,0), 'constant', constant_values=(0, 0))

  num = [num_1, num_2, num_3]
  den = [1,   den_1, den_2, den_3]
#  num = [-1.089e10,  5.863e17,  6.087e15]
#  den = [1,  3.009e07, 3.743e14,5.21e18]
  system = signal.lti(num, den)
  t = np.arange(0, len(siggen_wf)*10E-9, 10E-9)
  tout, siggen_wf, x = signal.lsim(system, siggen_wf, t)

  siggen_wf /= np.amax(siggen_wf)

  siggen_fit = siggen_wf[500::]
  
  siggen_fit *= scale
  
  

  
  #plotting

  plt.figure(wfFig.number)
  plt.clf()
  plt.title("Charge waveform")
  plt.xlabel("Sample number [10s of ns]")
  plt.ylabel("Raw ADC Value [Arb]")
  plt.plot( np_data  ,color="red" )
  plt.xlim( lastFitSampleIdx - fitSamples, lastFitSampleIdx+20)

  plt.plot(np.arange(startVal, len(siggen_fit)+startVal), siggen_fit  ,color="blue" )
#  plt.plot(np.arange(0, startVal), np.zeros(startVal)  ,color="blue" )
  plt.axvline(x=lastFitSampleIdx, linewidth=1, color='r',linestyle=":")
  plt.axvline(x=startVal, linewidth=1, color='g',linestyle=":")

  #save the important stuff
  np.save("chan%d_run%d_entry%d_fitwaveform.npy" % (channelNumber, runNumber, entryNumber), siggen_fit)
  plt.figure(wfFig.number)
  plt.savefig("chan%d_run%d_entry%d_fitwaveform.png" % (channelNumber, runNumber, entryNumber))



  plt.show()
  value = raw_input('  --> Press q to quit, any other key to continue\n')

  if value == 'q':
    exit(1)

def getParameterMedian(trace, paramName, burnin):
  return np.median(  trace[paramName][burnin:])


def findTimePoint(data, percent):
  #don't screw up the data, bro
  int_data = np.copy(data)
  int_data /= np.amax(int_data)
  return np.where(np.greater(int_data, percent))[0][0]

#def findSiggenWaveform(x,y,z,normMax):
#
#  weightedClusterPosition = TVector3(x, y, z);
#  clusterWaveform = MGTWaveform();
#  calcFlag = siggenInst.CalculateWaveform(weightedClusterPosition, clusterWaveform, 1);
#  
#  rcint = MGWFRCIntegration()
#  rcint.SetTimeConstant(rcIntTimeConstant)
#  rcint.TransformInPlace(clusterWaveform)
#  
#  siggen_data = clusterWaveform.GetVectorData()
#  siggen_data *= normMax / np.amax(siggen_data)
#
#  return siggen_data





if __name__=="__main__":
    main(sys.argv[1:])


