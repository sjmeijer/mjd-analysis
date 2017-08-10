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

  runRanges = [(11510, 11630)]#11540)]

  #
  #channel = 690
  # channel = 626
  channel = 672
  # channel = 578


  side_padding = 10

  fep_energy = 2614
  sep_energy = 2103
  dep_energy = 1592
  compton_energy = 2380

  ae_cut = rh.getChannelAECut(channel)

  energy_cut_fep = "(trapENFCal > %f && trapENFCal < %f)" % (fep_energy-side_padding,fep_energy+side_padding)
  energy_cut_sep = "(trapENFCal > %f && trapENFCal < %f)" % (sep_energy-side_padding,sep_energy+side_padding)
  energy_cut_dep = "(trapENFCal > %f && trapENFCal < %f)" % (dep_energy-side_padding,dep_energy+side_padding)
  energy_cut_cs = "(trapENFCal > %f && trapENFCal < %f)" % (compton_energy-side_padding,compton_energy+side_padding)


  # total_cut = "( %s) && channel == %d " % ( energy_cut_fep, channel)
  # wfFileName = "full_fep_event_set_runs%d-%d.npz" % (runRanges[0][0], runRanges[-1][-1])


  # total_cut = "( %s || %s) && channel == %d  " % ( energy_cut_dep, energy_cut_sep, channel, )
  # wfFileName = "ms_event_set_runs%d-%d_channel%d.npz" % (runRanges[0][0], runRanges[-1][-1], channel)

  total_cut = "( %s) && channel == %d  && %s > 1" % ( energy_cut_fep, channel, ae_cut)
  wfFileName = "fep_event_set_runs%d-%d_channel%d.npz" % (runRanges[0][0], runRanges[-1][-1], channel)

  # total_cut = "( %s ) && channel == %d && %s > 1 " % ( energy_cut_cs, channel, ae_cut)

  # wfFileName = "cs_event_set_runs%d-%d_channel%d.npz" % (runRanges[0][0], runRanges[-1][-1], channel)

  if True: #if you want to check out the plots of the spectra
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
        print( "Skipping file " + fileNameGAT)
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

    # depHist = TH1F("hDEP","Single Escape Peak",2*side_padding*binsPerKev,dep_energy-side_padding,dep_energy+side_padding);
    # chainGat.Project("hDEP", "trapENFCal", "channel == %d && %s" % (channel, energy_cut_dep));
    # depHistCut = TH1F("hDEPCut","Double Escape Peak",2*side_padding*binsPerKev,dep_energy-side_padding,dep_energy+side_padding);
    # chainGat.Project("hDEPCut", "trapENFCal", "channel == %d && %s && %s > 1  " % (channel, energy_cut_dep, ae_cut));
    # depHist.Draw()
    # depHistCut.SetLineColor(kRed)
    # depHistCut.Draw("SAME")
    # canvas.Update()
    # print( "found %d dep events" %  depHist.GetEntries())
    # value = input('  --> Make sure the hist is as you expect ')
    # if value == 'q': exit(0)



    # sepHist = TH1F("hSEP","Single Escape Peak",2*side_padding*binsPerKev,sep_energy-side_padding,sep_energy+side_padding);
    # chainGat.Project("hSEP", "trapENFCal", "channel == %d && %s" % (channel, energy_cut_sep));
    # sepHistCut = TH1F("hSEPCut","S Escape Peak",2*side_padding*binsPerKev,sep_energy-side_padding,sep_energy+side_padding);
    # chainGat.Project("hSEPCut", "trapENFCal", "channel == %d && %s && %s > 1  " % (channel, energy_cut_sep, ae_cut));
    # sepHist.Draw()
    # sepHistCut.Draw("SAME")
    # canvas.Update()
    # print( "found %d sep events" %  sepHist.GetEntries())
    # value = input('  --> Make sure the hist is as you expect ')
    # if value == 'q': exit(0)

    # my_binsPerKev=1
    # # csHist = TH1F("hCS","",200*my_binsPerKev,2300,2500);
    # # chainGat.Project("hCS", "trapENFCal", "channel == %d" % (channel));
    # # csHist.SetLineColor(kBlue)
    # # csHist.Draw()
    # cs_cut = TH1F("hCS_cut","Full Energy Peak",2*side_padding*binsPerKev,compton_energy-side_padding,compton_energy+side_padding);
    # chainGat.Project("hCS_cut", "trapENFCal", total_cut);
    # cs_cut.SetLineColor(kRed)
    # cs_cut.Draw()
    # canvas.Update()
    # print( "found %d compton events" %  cs_cut.GetEntries())

    # fepHist = TH1F("hFEP","Full Energy Peak",1800,1000,2800);
    # chainGat.Project("hFEP", "trapENFCal", "channel == %d " % (channel, ));
    # fepHist.SetLineColor(kBlue)
    # fepHist.GetXaxis().SetTitle("Energy (keV)")
    # fepHist.GetYaxis().SetTitle("Counts")
    # fepHist.Draw()
    # pad2.SetLogy(1)
    # canvas.Update()
    # canvas.Print("calibration_spectrum.pdf")
    # # print( "found %d fep events" %  fepHist.GetEntries())
    # value = input('  --> Make sure the hist is as you expect ')
    # exit()


    fepHist = TH1F("hFEP","Full Energy Peak",2*side_padding*binsPerKev,fep_energy-side_padding,fep_energy+side_padding);
    chainGat.Project("hFEP", "trapENFCal", "channel == %d && %s" % (channel, energy_cut_fep));
    fepHist.SetLineColor(kBlue)
    fepHist.Draw()
    # canvas.Update()
    # print( "found %d fep events" %  fepHist.GetEntries())
    # value = input('  --> Make sure the hist is as you expect ')

    testHist = TH1F("htest","Full Energy Peak",2*side_padding*binsPerKev,fep_energy-side_padding,fep_energy+side_padding);
    chainGat.Project("htest", "trapENFCal", total_cut);
    testHist.SetLineColor(kRed)
    fepHist.GetXaxis().SetTitle("Energy (keV)")
    fepHist.GetYaxis().SetTitle("Counts")
    testHist.Draw("SAME")


    leg = TLegend(0.15,0.7,0.45,0.85)
    leg.SetBorderSize(0);
    leg.SetFillColor(0);
    # leg.SetTextSize(12);
    # leg.SetTextFont(textFont);
    leg.AddEntry(fepHist,"All events","l")
    leg.AddEntry(testHist,"Events after A/E cut","l")
    leg.Draw();

    pad2.SetLogy(1)
    canvas.Update()
    canvas.Print("2614_aecut.pdf")

    print( "saving %d events (out of %d)" %  (testHist.GetEntries(), fepHist.GetEntries()))
    value = input('  --> Make sure the hist is as you expect ')
    if value == 'q': exit(0)



  if not os.path.isfile(wfFileName):
    wfs = rh.GetWaveforms( runRanges,  channel, np.inf, total_cut,)
  else:
    print( "file by name %s already exists!  exiting..." % wfFileName)
    exit(0)

  np.savez(wfFileName, wfs = wfs)

  print( "Found %d waveforms" % wfs.size)

  exit(0)




if __name__=="__main__":
    main(sys.argv[1:])
