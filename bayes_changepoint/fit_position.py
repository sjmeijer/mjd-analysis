#!/usr/local/bin/python
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")

import sys, os
#import pylab
import matplotlib.pyplot as plt
import numpy as np

import pymc
import signal_model_mc2 as sm


gatDataName = "mjd_run"
gatTreeName = "mjdTree"
builtDataName = "OR_run"
builtTreeName = "MGTree"
dataSetName = "surfmjd"
detectorName = "P3JDY"
flatTimeSamples = 800
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
    energyCut = "trapECal>%f && trapECal<%f" % (1588,1594)
    
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
    numWaveforms = event.GetNWaveforms()

    for i_wfm in xrange( numWaveforms ):
        channel = channelVec[i_wfm]
        if (channel != channelNumber): continue
        return event.GetWaveform(i_wfm)

####################################################################################################################################################################
    
        
def fitWaveform(wf, wfFig, zoomFig):

  np_data = wf.GetVectorData()
  
  wfMax = np.amax(np_data)
  
  #perform the fit up to this index.  Currently set by 99% timepoint (no real point in fitting on the falling edge)
  lastFitSampleIdx = 1060#findTimePoint(np_data, 0.99)
  
  fitSamples = 110 # 1 microsecond
  
  firstFitSampleIdx = lastFitSampleIdx - fitSamples
  
  np_data_early = np_data[firstFitSampleIdx:lastFitSampleIdx]
  
  startGuess = 986
  t0_guess = startGuess - firstFitSampleIdx
  siggen_model = pymc.Model( sm.createSignalModelSiggen(np_data_early, t0_guess, wfMax) )
 
  M = pymc.MCMC(siggen_model)
  M.use_step_method(pymc.AdaptiveMetropolis, [M.radEst, M.zEst, M.phiEst, M.wfScale, M.switchpoint], delay=1000)
#  M.use_step_method(pymc.DiscreteMetropolis, M.switchpoint, proposal_distribution='Normal', proposal_sd=4)
  M.sample(iter=500000)
 
  t0 = np.around(M.trace('switchpoint')[-1])
  r = M.trace('radEst')[-1]
  z = M.trace('zEst')[-1]
  phi = M.trace('phiEst')[-1]
  scale = M.trace('wfScale')[-1]
  startVal = t0 + firstFitSampleIdx
  
  # Two subplots, the axes array is 1-d
  f, axarr = plt.subplots(5, sharex=True)
  
  axarr[0].plot(M.trace('radEst')[:])
  axarr[1].plot(M.trace('zEst')[:])
  axarr[2].plot(M.trace('phiEst')[:])
  axarr[3].plot(M.trace('wfScale')[:])
  axarr[4].plot(M.trace('switchpoint')[:])

  
  print ">>> t0guess was %d" % (firstFitSampleIdx+t0_guess)
  print ">>> fit t0 was %d" % (firstFitSampleIdx + t0)
  
  siggen_fit = sm.findSiggenWaveform(r, phi, z)
  
  siggen_fit *= scale
  
  #plotting

  plt.figure(wfFig.number)
  plt.clf()
  plt.title("Charge waveform")
  plt.plot( np_data  ,color="red" )
  plt.xlim( lastFitSampleIdx - fitSamples, lastFitSampleIdx+25)
  #plt.ylim(-0.1, 2.5)
  plt.plot(np.arange(startVal, 800+startVal), siggen_fit  ,color="blue" )
#  plt.plot(np.arange(0, startVal), np.zeros(startVal)  ,color="blue" )
  plt.axvline(x=lastFitSampleIdx, linewidth=1, color='r',linestyle=":")
  plt.axvline(x=startVal, linewidth=1, color='g',linestyle=":")
  
  rad_arr = M.trace('radEst')[burnin:]
  z_arr = M.trace('zEst')[burnin:]
  phi_arr = M.trace('phiEst')[burnin:]
  n_bins=25
  
  
  fig0 = plt.figure(6)
  weights = np.ones_like(rad_arr)/float(len(rad_arr))
  n, bins, patches = plt.hist(rad_arr, n_bins, histtype='step', linewidth=5, weights=weights)
  plt.xlabel("r value [mm]")
  plt.ylabel("probability")
  plt.savefig("pdf_r.pdf")
  
  fig0 = plt.figure(7)
  weights = np.ones_like(z_arr)/float(len(z_arr))
  n, bins, patches = plt.hist(z_arr, n_bins, histtype='step', linewidth=5, weights=weights)
  plt.xlabel("z value [mm]")
  plt.ylabel("probability")
  plt.savefig("pdf_z.pdf")
  
  fig0 = plt.figure(8)
  weights = np.ones_like(phi_arr)/float(len(phi_arr))
  n, bins, patches = plt.hist(phi_arr, n_bins, histtype='step', linewidth=5, weights=weights)
  plt.xlabel("phi value [rad]")
  plt.ylabel("probability")
  plt.savefig("pdf_phi.pdf")

  
#  plt.figure(zoomFig.number)
#  plt.clf()
#  plt.title("Zoom in near the start time")
#  plt.plot( np_data  ,color="red" )
#  plt.xlim( startVal-40, startVal+40)
#  plt.ylim(-10, 50)
#  plt.plot(np.arange(startVal, 800+startVal), sigFit  ,color="blue" )
#  plt.plot(np.arange(0, startVal), np.zeros(startVal)  ,color="blue" )
#  plt.axvline(x=startVal, linewidth=1, color='g',linestyle=":")


  value = raw_input('  --> Press q to quit, any other key to continue\n')

  if value == 'q':
    exit(1)



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


