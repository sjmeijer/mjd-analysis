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
from matplotlib import gridspec
plt.style.use('presentation')

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
      plotWaveform(waveform, fig, entryNumber)


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
    
        
def plotWaveform(wf, wfFig, wfNum):

  np_data = wf.GetVectorData()
  
  
  fit_data = np.load("fit_waveform.npy")
  
  fit_end = 1060
  fit_start = fit_end - 110
  
  startVal = 971
  ppc_val = 1016
  
  startVal = 994
  ppc_val = 1043
  
  
  full_fit_signal = np.zeros_like(np_data)
  full_fit_signal[startVal:startVal+800] += fit_data
  
  diff = full_fit_signal - np_data
  
  #plt.figure(wfFig.number)
  fig = plt.figure(figsize=(8, 6))
  gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
  ax0 = plt.subplot(gs[0])
  ax1 = plt.subplot(gs[1], sharex=ax0)
#  f, axarr = plt.subplots(2, sharex=True, figsize =(15,15) )
#
#  plt.title("Charge waveform")
  ax1.set_xlabel("Digitizer Time [ns]")
  ax0.set_ylabel("Voltage [Arb.]")
  ax1.set_ylabel("Residual")
  ax0.plot(np.arange(0, 10*len(np_data), 10), np_data  ,color="red", label="Data (unfiltered)" )
  ax0.plot(np.arange(startVal*10, (800+startVal)*10, 10), fit_data  ,color="blue", label="Fit waveform" )
  
  ax1.plot(np.arange(0, 10*len(diff), 10), diff  ,color="#7BAFD4" )
  first_legend = ax0.legend(loc=2)
  
  ax1.set_xlim( (fit_start)*10, (fit_end)*10)
  ax1.set_ylim( -100, 100)
  ax0.set_ylim( -100, 4000)
#
##  plt.axvline(x=ppc_val*10, linewidth=5, color='g',linestyle=":")
#  plt.axvline(x=startVal*10, linewidth=5, color='g',linestyle=":")
  plt.savefig("fit_waveform.pdf")

  value = raw_input('  --> Press q to quit, any other key to continue\n')

  if value == 'q':
    exit(1)
  if value == 's':
    plt.savefig("wf_example_%d.pdf" % wfNum)



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


