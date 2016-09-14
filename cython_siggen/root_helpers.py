#!/usr/local/bin/python
from ROOT import *
from helpers import Waveform

import sys, os
import numpy as np

gatDataName = "mjd_run"
gatTreeName = "mjdTree"
builtDataName = "OR_run"
builtTreeName = "MGTree"
dataSetName = "surfmjd"

detectorName = "P3KJR"

baselineSamples = 800


def GetWaveformByEntry(runNumber, entryNumber, channelNumber, doBaselineSub=True):
    gatFilePath =  os.path.expandvars("$MJDDATADIR/%s/data/gatified/%s/%s%d.root" % (dataSetName, detectorName, gatDataName, runNumber  ) )
    builtFilePath =  os.path.expandvars("$MJDDATADIR/%s/data/built/%s/%s%d.root" % (dataSetName, detectorName, builtDataName, runNumber  ) )

    gat_file = TFile.Open(gatFilePath)
    gatTree = gat_file.Get(gatTreeName)
    built_file = TFile.Open(builtFilePath)
    builtTree = built_file.Get(builtTreeName)
    builtTree.AddFriend(gatTree)

    waveform = getWaveform(gatTree, builtTree, entryNumber, channelNumber)
    waveform.SetLength(waveform.GetLength()-10)
    
    rms=0
    if doBaselineSub:
      baseline = MGWFBaselineRemover()
      baseline.SetBaselineSamples(baselineSamples)
      baseline.TransformInPlace(waveform)
      rms = baseline.GetBaselineRMS()

    np_data = np.array(waveform.GetVectorData())

    return Waveform(np_data, channelNumber, runNumber, entryNumber ,rms)


########################################################################

def GetWaveforms(runRanges, channelNumber, numWaveforms, cutStr):
  '''cut string should include all cuts except for channel cut'''

  cutChan = "channel == %d" % channelNumber
  cutStr = cutChan + " && " + cutStr

  baseline = MGWFBaselineRemover()
  baseline.SetBaselineSamples(baselineSamples)
  
  waveformArray = []
  
  runList = []
  for runs in runRanges:
    for run in range(runs[0], runs[1]+1):
      runList.append(run)

  for iRun in runList:
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
    
    gatTree.SetEntryList(0)
    gatTree.Draw(">>elist", cutStr, "entrylist")
    elist = gDirectory.Get("elist")
    #print "Number of entries in the entryList is " + str(elist.GetN())

    
    gatTree.Draw(">>elistChan", cutChan, "entrylist")
    elistChan = gDirectory.Get("elistChan")

    gatTree.SetEntryList(elist);
    builtTree.SetEntryList(elist);
    
    for ientry in xrange( elist.GetN() ):
      entryNumber = gatTree.GetEntryNumber(ientry);
      
#      if entryNumber in funnyEntries: continue

      waveform = getWaveform(gatTree, builtTree, entryNumber, channelNumber)
      waveform.SetLength(waveform.GetLength()-10)
      baseline.TransformInPlace(waveform)
      
      print "Waveform number %d in run %d" % (entryNumber, iRun)
      #fitWaveform(waveform, fig, fig2, iRun, entryNumber, channelNumber)
      
      np_data = waveform.GetVectorData()
      np_data = np.array(np_data)
      
#      event = builtTree.event
#      current_time = event.GetTime()
#      
#      #find the last hit event for this channel
#      for i in range(elistChan.GetN()):
#        e_num_tmp = elistChan.GetEntry(i)
#        if e_num_tmp >= entryNumber: break
#        last_entry_number = e_num_tmp
#      
#      builtTree.GetEntry(last_entry_number)
#      lastEvent = builtTree.event
#      
#      timeSinceLast =  current_time - lastEvent.GetTime()
#
#      energyLast = getWaveformEnergy(gatTree, builtTree, last_entry_number, channelNumber)

      energy = getWaveformEnergy(gatTree, builtTree, entryNumber, channelNumber)

      waveformArray.append( Waveform(np_data, channelNumber, iRun, entryNumber ,baseline.GetBaselineRMS(),  energy=energy) )
      if len(waveformArray) >= numWaveforms: break
      
    gat_file.Close()
    built_file.Close()
    if len(waveformArray) >= numWaveforms: break

  waveformArrayNumpy = np.array(waveformArray)

  return waveformArrayNumpy


########################################################################

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

def getWaveformEnergy(gatTree, builtTree, entryNumber, channelNumber):
    
    builtTree.GetEntry( entryNumber )
    gatTree.GetEntry( entryNumber )
    
    event = builtTree.event
    channelVec   = gatTree.channel
    numWaveforms = event.GetNWaveforms()

    for i_wfm in xrange( numWaveforms ):
        channel = channelVec[i_wfm]
        if (channel != channelNumber): continue
        return gatTree.trapENFCal[i_wfm]

