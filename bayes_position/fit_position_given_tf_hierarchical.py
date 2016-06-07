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

gradList = np.arange(0.01, 0.09, 0.01)
pcRadList = np.arange(1.65, 2.95, 0.1)

#shorter just to make it load faster
#gradList = np.arange(0.01, 0.03, 0.01)
#pcRadList = np.arange(1.65, 1.95, 0.1)


detArray  = np.empty( (len(gradList),len(pcRadList)), dtype=object)

for (gradIdx,grad) in enumerate(gradList):
#  grad /= 10000.
  for (radIdx, pcRad) in enumerate(pcRadList):
#    detName = "P42574A_grad%0.3f_pcrad%0.4f.conf" % (grad,pcRad)
    detName = "P42574A_grad%0.2f_pcrad%0.2f.conf" % (grad,pcRad)
    det =  Detector(detName, 39.3, 33.8, 66.5* CLHEP.ns, 71.8*CLHEP.us, zeroPadding=500, temperature=99.)
    detArray[gradIdx, radIdx] = det

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
  
  waveformArray = []
  numWaveforms = 25

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
      #fitWaveform(waveform, fig, fig2, iRun, entryNumber, channelNumber)
      
      np_data = waveform.GetVectorData()
      np_data = np.multiply(np_data, 1.)
      waveformArray.append(np_data)
      if len(waveformArray) >= numWaveforms: break

  fitWaveforms(waveformArray, fig, fig2, iRun, entryNumber, channelNumber)

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

  fitSamples = 150

  for wf in wfs:
#    np_data = wf.GetVectorData()
    np_data = wf
    wfMax = np.amax(np_data)

  #  lastFitSampleIdx = findTimePoint(np_data, .95)
    startGuess = findTimePoint(np_data, 0.005)
    firstFitSampleIdx = startGuess-20
    lastFitSampleIdx = firstFitSampleIdx + fitSamples
#    fitSamples = lastFitSampleIdx-firstFitSampleIdx # 1 microsecond
    t0_guess = startGuess - firstFitSampleIdx
    
    np_data_early = np_data[firstFitSampleIdx:lastFitSampleIdx]
    
    wf_to_fit_arr.append(np_data_early)

    plt.plot( np_data_early  ,color="red" )

  #make a simulated wf to plot for reference
  det = detArray[len(gradList)/2, len(pcRadList)/2]
  siggen_wf= det.GetSiggenWaveform(10, 0, 10, energy=1)
  siggen_wf = np.pad(siggen_wf, (det.zeroPadding,0), 'constant', constant_values=(0, 0))
  system = signal.lti(prior_num, prior_den)
  t = np.arange(0, len(siggen_wf)*10E-9, 10E-9)
  tout, siggen_wf, x = signal.lsim(system, siggen_wf, t)
  siggen_wf /= np.amax(siggen_wf)
  siggen_wf *= wfMax
  siggen_wf = siggen_wf[500::]
  plt.plot(np.arange(t0_guess, len(siggen_wf)+t0_guess), siggen_wf  ,color="blue" )
  
  
  plt.xlim(0, 150)

  doFit = 1
  value = raw_input('  --> Press s to skip,  q to quit, any other key to continue with fit\n')
  if value == 'q':
    exit(0)
  if value == 's':
    doFit = 0

  siggen_model = sm3.CreateFullDetectorModelGivenTransferFunction(detArray, wf_to_fit_arr, t0_guess, wfMax)
  with siggen_model:
    
    step = Metropolis()

    # for slice
    one_minute = 100#np.around(16380 / 114.4)
    one_hour = 60 * one_minute
    
    this_sample = 500
    
#    trace = sample(this_sample, step=[step1, step2], start=start)
    trace = sample(this_sample,  step = step)
    
    
#    burnin = np.int(.75 * this_sample)
#
#
#    t0 = np.around( np.median(  trace['switchpoint'][burnin:]))
#    r =             np.median(  trace['radEst'][burnin:])
#    z =             np.median(  trace['zEst'][burnin:])
#    phi =           np.median(  trace['phiEst'][burnin:])
#    scale =         np.median(  trace['wfScale'][burnin:])
#    temp =          np.median(  trace['temp'][burnin:])
#    gradIdx =        np.int(np.median( trace['gradIdx'][burnin:]))
#    pcRadIdx =       np.int(np.median( trace['pcRadIdx'][burnin:]))
#    
#
#
#    startVal = t0 + firstFitSampleIdx
#    
#    print "<<<startVal is %d" % startVal
#    print "<<<r is %0.2f" % r
#    print "<<<z is %0.2f" % z
#    print "<<<phi is %0.2f" % phi
#    print "<<<scale is %0.2f" % scale
#    print "<<<temp is %0.2f" % (temp)
#    print "<<<detector gradient is %0.2f (idx is %d)" % (gradList[gradIdx], gradIdx)
#    print "<<<PC Radius is %0.2f (idx is %d)" % (pcRadList[pcRadIdx], pcRadIdx)


#    num_1 =     np.median(  trace['num_1'][burnin:])
#    num_2 =     np.median(  trace['num_2'][burnin:])
#    num_3 =     np.median(  trace['num_3'][burnin:])
#    den_1 =     np.median(  trace['den_1'][burnin:])
#    den_2 =     np.median(  trace['den_2'][burnin:])
#    den_3 =     np.median(  trace['den_3'][burnin:])
#
#    print "<<<tf NUM is [%0.2e, %0.2e, %0.2e]" % (num_1, num_2, num_3)
#    print "<<<tf DEN is [1, %0.2e, %0.2e, %0.2e]" % (den_1, den_2, den_3)

    plt.ioff()
    traceplot(trace)
#    plt.savefig("chan%d_run%d_entry%d_chain_notffit.png" % (channelNumber, runNumber, entryNumber))
    plt.ion()
  

  
#  print ">>> t0guess was %d" % (firstFitSampleIdx+t0_guess)
#  print ">>> fit t0 was %d" % (firstFitSampleIdx + t0)
#
#  det = detArray[np.int(gradIdx), np.int(pcRadIdx)]
#  det.SetTemperature(temp)
#
#  siggen_wf= det.GetSiggenWaveform(r, phi, z, energy=2600)
#  siggen_wf = np.pad(siggen_wf, (det.zeroPadding,0), 'constant', constant_values=(0, 0))
#
##  num = [num_1, num_2, num_3]
##  den = [1,   den_1, den_2, den_3]
##  num = [-1.089e10,  5.863e17,  6.087e15]
##  den = [1,  3.009e07, 3.743e14,5.21e18]
#  system = signal.lti(prior_num, prior_den)
#  t = np.arange(0, len(siggen_wf)*10E-9, 10E-9)
#  tout, siggen_wf, x = signal.lsim(system, siggen_wf, t)
#
#  siggen_wf /= np.amax(siggen_wf)
#
#  siggen_fit = siggen_wf[500::]
#  
#  siggen_fit *= scale
#  
#  
#
#  
#  #plotting
#
#  plt.figure(wfFig.number)
#  plt.clf()
#  plt.title("Charge waveform")
#  plt.xlabel("Sample number [10s of ns]")
#  plt.ylabel("Raw ADC Value [Arb]")
#  plt.plot( np_data  ,color="red" )
#  plt.xlim( lastFitSampleIdx - fitSamples, lastFitSampleIdx+20)
#
#  plt.plot(np.arange(startVal, len(siggen_fit)+startVal), siggen_fit  ,color="blue" )
##  plt.plot(np.arange(0, startVal), np.zeros(startVal)  ,color="blue" )
#  plt.axvline(x=lastFitSampleIdx, linewidth=1, color='r',linestyle=":")
#  plt.axvline(x=startVal, linewidth=1, color='g',linestyle=":")
#
#  #save the important stuff
#  np.save("chan%d_run%d_entry%d_fitwaveform_notffit.npy" % (channelNumber, runNumber, entryNumber), siggen_fit)
#  plt.figure(wfFig.number)
#  plt.savefig("chan%d_run%d_entry%d_fitwaveform_notffit.png" % (channelNumber, runNumber, entryNumber))
#


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


