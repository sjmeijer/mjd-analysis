#!/usr/bin/python
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
TROOT.gApplication.ExecuteFile("~/MJDTalkPlotStyle.C")
import sys
import array
from ctypes import c_ulonglong
import os

"""
    Make nice PSA plots
"""

#if len(sys.argv) < 2:
#    print 'Usage: viewMjorSIS3302Waveforms.py [ROOT output from MJOR]'
#    sys.exit()



# Define your cuts


isInteractive = 0

plotFits = 0
doFits = 0

#nEbinskeV = 15000;
nEbinskeV=3000
EminkeV = 0;
emaxkeV=3000;

binsPerKev = nEbinskeV/emaxkeV


energyName = "trapECal"

currentName = "rcdwfMax"
ae_max = 0.06 #for the a/e hist plot
ae_min = 0.02
aeBins = 200
aeScale = 0
aeCut = 0
#aeScale = 0

plotSuffix = "highGrad"


dataSet = "surfmjd"
detector = "P3JDY"
gatName = "mjd_run"
gatTreeName = "mjdTree"

startRun = 6578
endRun = 6774

chanCutHot = [656,626,688]
chanCutCold = [646,644,642,630,628,624,632,584,674,576,680,692,690,640,676,616,614,610,664,662,696,608,598,600,594,692]


###########################################

chanCutStrHot = ""
chanCutStrCold = ""

for chan in chanCutHot:
  chanCutStrHot += "channel == %d" % chan
  if chan == chanCutHot[-1]: break
  chanCutStrHot += " || "

for chan in chanCutCold:
  chanCutStrCold += "channel == %d" % chan
  if chan == chanCutCold[-1]: break
  chanCutStrCold += " || "

sepEnergyCut = "energyCal>%f && energyCal<%f" % (2100,2107);
depEnergyCut = "energyCal>%f && energyCal<%f" % (1590,1596);

#sepCut = sepEnergyCut + " && channel == %d" % chanCut
#depCut = depEnergyCut + " && channel == %d" % chanCut
#chanCutStr = "channel == %d" % chanCut
#
#aePassStr = "abs(%s/%s) -(%.15f*(%s)) > %.7f" % (currentName, energyName, aeScale, energyName, aeCut)
#aeCutStr = "abs(%s/%s) -(%.15f*(%s)) < %.7f" % (currentName, energyName, aeScale, energyName, aeCut)
#
#chanCutStr = "channel == %f" % chanCut
#
#totalCutStr = aeCutStr + " && " + chanCutStr
#totalPassStr = aePassStr + " && " + chanCutStr

###########################################

chainGat = TChain(gatTreeName)
chainGat.SetDirectory(0)
gatName =  os.path.expandvars("$MJDDATADIR/%s/data/gatified/%s/%s" % (dataSet, detector, gatName  ) )
for i in xrange(startRun, endRun+1):
  fileNameGAT = gatName + "%d.root" % i
  
  if not os.path.isfile(fileNameGAT):
    print "Skipping file " + fileNameGAT
    continue
  chainGat.Add( fileNameGAT )

chainGat.SetBranchStatus("*",0)
chainGat.SetBranchStatus(energyName ,1)
chainGat.SetBranchStatus(currentName ,1)
chainGat.SetBranchStatus("channel" ,1)

###########################################


#set up a canvas :
canvas = TCanvas("canvas")

pad2 = TPad("pad2","The Superpulse", 0.05,0.05,0.95,0.95)
gStyle.SetOptStat(0)
pad2.Draw()
pad2.cd()

draw = "abs(%s/%s) - (%.15f)*(%s) " % (currentName, energyName, aeScale, energyName)
print "The hist drawing will be: " + draw

highECut =  "energyCal > 1000"

hotHighECut = highECut + " && " + chanCutStrHot
coldHighECut = highECut + " && " + chanCutStrHot

aeHistHot = TH1F("hAEHot","",aeBins,ae_min,ae_max);
aeHistCold = TH1F("hAECold","",aeBins,ae_min,ae_max);

chainGat.Project("hAEHot", draw, hotHighECut);
chainGat.Project("hAECold", draw, coldHighECut);

aeHistColdMax = depHist.GetMaximum()

aeHistCold.SetLineColor(kBlue)
aeHistCold.SetYTitle("Counts")
aeHistCold.SetXTitle("A/E (A.U.)")
aeHistCold.Draw()
canvas.Update()

aeHistHot.SetLineColor(kRed)
aeHistHot.Scale(1/aeHistColdMax);
aeHistHot.Draw("SAME")

leg = TLegend(0.15,0.7,0.47,0.85)
leg.SetBorderSize(0);
leg.SetFillColor(0);
textFont = 43;
textSize = 25;
leg.SetTextSize(textSize);
leg.SetTextFont(textFont);
leg.AddEntry(aeHistHot,"Hot boule","l")
leg.AddEntry(aeHistCold,"Other detectors","l")
leg.Draw();

canvas.Update()

if isInteractive:
    value = raw_input('  --> Make sure the hist is as you expect ')
canvas.Print("ae_hist_highE_%s.pdf" % plotSuffix)
