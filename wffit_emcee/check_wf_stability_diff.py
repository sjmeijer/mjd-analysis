#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy as np
from scipy import signal
from scipy import ndimage

from ROOT import *
import helpers
from detector_model import *
from probability_model_hier import *
from timeit import default_timer as timer

gatDataName = "mjd_run"
gatTreeName = "mjdTree"
builtDataName = "OR_run"
builtTreeName = "MGTree"
dataSetName = "surfmjd"
detectorName = "P3KJR"

def main(argv):
#  runRanges = [(11520, 11520), (11800, 11800), (12472, 12472), (12660, 12660),(13060, 13060)]
  runRanges = [ (11520, 11520), (12733,12733), (12800,12800), (13556,13556),]
  numThreads = 4
  scale_mult = 100.

  #calibration on
  
  channelList = helpers.getHighGainChannels()
  
  for channel in [626]:

    dep_energy_dict = {11520:1597, 12733:1597, 12800:1597, 13556:1592}

    side_padding = 15
    sep_energy = 2109
    
    ae_cut = "( %s > 1)" % helpers.getChannelAECut(channel)
    figure_name = "wf_shape_shift_ch%d_diff.png" % channel

  

    if False: #if you want to check out the plots of the spectra
     #check the sptrcum from root, just to make sure it looks OK
      runList = []
      for runs in runRanges:
        for run in range(runs[0], runs[1]+1):
          runList.append(run)
    
      chainGat = TChain(gatTreeName)
      chainGat.SetDirectory(0)

      gatName =  os.path.expandvars("$MJDDATADIR/%s/data/gatified/%s/%s" % (dataSetName, detectorName, gatDataName  ) )
      for i in runList:
        fileNameGAT = gatName + "%d.root" % i
        if not os.path.isfile(fileNameGAT):
          print "Skipping file " + fileNameGAT
          continue
        chainGat.Add( fileNameGAT )

      chainGat.SetBranchStatus("*",0)
      chainGat.SetBranchStatus("trapENFCal" ,1)
      chainGat.SetBranchStatus("trapECal" ,1)
      chainGat.SetBranchStatus("channel" ,1)

      binsPerKev = 2
    
      canvas = TCanvas("canvas")
      pad2 = TPad("pad2","The Superpulse", 0.05,0.05,0.95,0.95)
      gStyle.SetOptStat(0)
      pad2.Draw()
      pad2.cd()
    
      depHist = TH1F("hDEP","Double Escape Peak",2*side_padding*binsPerKev,dep_energy-side_padding,dep_energy+side_padding);
      chainGat.Project("hDEP", "trapENFCal", "channel == %d && %s" % (channel, energy_cut_dep));
      depHist.Draw()
      canvas.Update()
      value = raw_input('  --> Make sure the hist is as you expect ')
      if value == 'q': exit(0)


    plt.figure(0, figsize=(20,10))
    plt.clf()
#    plt.figure(1, figsize=(20,10))
#    plt.clf()

    colorDict = { 11520:"green",12733:"purple", 12800: "red", 12870: "b", 13556:"blue"}

    for runRange in runRanges:
    
      dep_energy = dep_energy_dict[runRange[0]]
      energy_cut_dep = "(trapENFCal > %f && trapENFCal < %f)" % (dep_energy-5,dep_energy+5)
      total_cut = "( %s ) && channel == %d && %s " % ( energy_cut_dep, channel, ae_cut)

    
      isoRunRange = [runRange]
      iso_wfs = helpers.GetWaveforms( isoRunRange,  channel, 10, total_cut)
      print "Run %d: Found %d wfs" % (runRange[0], iso_wfs.size)
      for (idx, wf) in enumerate(iso_wfs):
        
        waveform = np.diff(wf.waveformData)
        waveform = ndimage.filters.gaussian_filter1d(waveform, 5)
        alignPoint = np.argmax(waveform)
        
        plt.figure(0)
        plt.plot(waveform[alignPoint-200:alignPoint+200], color=colorDict[runRange[0]], )
      
#        plt.figure(1)
#      
#        lo_gain_Wf = helpers.GetWaveformByEntry(wf.runNumber, wf.entry_number, channel+1)
#        waveform_lo = lo_gain_Wf.waveformData
#        waveform_lo = ndimage.filters.gaussian_filter1d(waveform_lo, 5)
#        alignPoint = np.argmax(waveform_lo)
#        plt.plot(waveform_lo[alignPoint-200:alignPoint+200], color=colorDict[runRange[0]], )

      
      
      plt.figure(0)
      plt.plot(np.nan, color=colorDict[runRange[0]], label="run %d"%runRange[0])
      plt.figure(1)
      plt.plot(np.nan, color=colorDict[runRange[0]], label="run %d"%runRange[0])

    plt.figure(0)
    plt.legend(loc=4)
    plt.xlabel("Samples [10s of ns]")
    plt.ylabel("Amplitude [a.u.]")
    
    figure_name_zoomout = "wf_shape_shift_ch%d_wholewf_diff.png" % channel
    plt.savefig(figure_name_zoomout)
    
    plt.xlim(150,250)
    plt.ylim(np.amax(waveform)-500,np.amax(waveform)+50)
    

    
    plt.savefig(figure_name)
    
#    plt.figure(1)
#    plt.legend(loc=4)
#    plt.xlim(150,250)
#    plt.ylim(np.amax(waveform_lo)-500,np.amax(waveform_lo)+50)
#    plt.savefig( "wf_shape_shift_ch%d_lo.png" % channel)


#  plt.show()


def findTimePoint(data, percent):
  #don't screw up the data, bro
  int_data = np.copy(data)
  int_data /= np.amax(int_data)
  return np.where(np.greater(int_data, percent))[0][0]


if __name__=="__main__":
    main(sys.argv[1:])


