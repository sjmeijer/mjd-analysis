#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
import numpy as np

import root_helpers as rh
from ROOT import *


gatDataName = "mjd_run"
gatTreeName = "mjdTree"
builtDataName = "OR_run"
builtTreeName = "MGTree"
dataSetName = "surfmjd"
detectorName = "P3KJR"

def main(argv):

  runRanges = [(11510, 11539)]#11540)]

  #
  #channel = 690
  channel = 626

  side_padding = 15

  fep_energy = 2621
  sep_energy = 2109
  dep_energy = 1597

  ae_cut = rh.getChannelAECut(channel)

  energy_cut_fep = "(trapENFCal > %f && trapENFCal < %f)" % (fep_energy-side_padding,fep_energy+side_padding)
  energy_cut_sep = "(trapENFCal > %f && trapENFCal < %f)" % (sep_energy-side_padding,sep_energy+side_padding)
  energy_cut_dep = "(trapENFCal > %f && trapENFCal < %f)" % (dep_energy-side_padding,dep_energy+side_padding)


#  total_cut = "( %s || %s) && channel == %d  " % ( energy_cut_dep, energy_cut_sep, channel, )
#  wfFileName = "ms_event_set_runs%d-%d.npz" % (runRanges[0][0], runRanges[-1][-1])

  total_cut = "( %s ) && channel == %d && %s > 1 " % ( energy_cut_fep, channel, ae_cut)
  wfFileName = "fep_event_set_runs%d-%d_channel%d.npz" % (runRanges[0][0], runRanges[-1][-1], channel)


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

#    chainGat.SetBranchStatus("*",0)
#    chainGat.SetBranchStatus("trapENFCal" ,1)
#    chainGat.SetBranchStatus("trapECal" ,1)
#    chainGat.SetBranchStatus("channel" ,1)

    binsPerKev = 2

    canvas = TCanvas("canvas")
    pad2 = TPad("pad2","The Superpulse", 0.05,0.05,0.95,0.95)
    gStyle.SetOptStat(0)
    pad2.Draw()
    pad2.cd()

    sepHist = TH1F("hSEP","Single Escape Peak",2*side_padding*binsPerKev,sep_energy-side_padding,sep_energy+side_padding);
    chainGat.Project("hSEP", "trapENFCal", "channel == %d && %s" % (channel, energy_cut_sep));
    sepHist.Draw()
    canvas.Update()
    print "found %d sep events" %  sepHist.GetEntries()
    value = raw_input('  --> Make sure the sep hist is as you expect ')
    if value == 'q': exit(0)


    depHist = TH1F("hDEP","Double Escape Peak",2*side_padding*binsPerKev,dep_energy-side_padding,dep_energy+side_padding);
    chainGat.Project("hDEP", "trapENFCal", "channel == %d && %s" % (channel, energy_cut_dep));
    depHist.Draw()
    canvas.Update()
    print "found %d dep events" %  depHist.GetEntries()
    value = raw_input('  --> Make sure the hist is as you expect ')
    if value == 'q': exit(0)

    fepHist = TH1F("hFEP","Full Energy Peak",2*side_padding*binsPerKev,fep_energy-side_padding,fep_energy+side_padding);
    chainGat.Project("hFEP", "trapENFCal", "channel == %d && %s" % (channel, energy_cut_fep));
    fepHist.Draw()
    canvas.Update()
    print "found %d fep events" %  fepHist.GetEntries()
    value = raw_input('  --> Make sure the hist is as you expect ')
    if value == 'q': exit(0)



  if not os.path.isfile(wfFileName):
    wfs = rh.GetWaveforms( runRanges,  channel, np.inf, total_cut,)
  else:
    print "file by name %s already exists!  exiting..." % wfFileName
    exit(0)

  np.savez(wfFileName, wfs = wfs)

  print "Found %d waveforms" % wfs.size

  exit(0)




if __name__=="__main__":
    main(sys.argv[1:])
