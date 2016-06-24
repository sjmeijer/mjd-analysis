from ROOT import *
#TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
#TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
#TROOT.gApplication.ExecuteFile("$HOME/Dev/Analysis/MJDTalkPlotStyle.C")

import numpy as np
import matplotlib.pyplot as plt
import os, sys


burstCutStr = "!(time_s > 2192e3 && time_s < 2195e3) && !(time_s > 7370e3 && time_s <7371e3) && !(time_s > 7840e3 && time_s < 7860e3) && !(time_s > 8384e3 && time_s < 8387e3) && !(time_s > 8984e3 && time_s < 8985e3) && !(time_s > 9002e3 && time_s < 9005e3) && run != 13075 && run != 13093 && run != 13116"

def main(argv):


  #Load the Skim files
  skimFileDir = "/Users/bshanks/Data/skim/P3KJR/"
  ch = TChain("skimTree", "skimTree")
#  ch.Add(skimFileDir + "skimDS1_*.root")
  ch.Add("/Users/bshanks/Data/skim/DS0/" + "skimDS0_*.root")
  
  #rebin to 4 keV
  binWidth = 40

  canvas = TCanvas("canvas")
  gStyle.SetOptStat(0)
  
  timeHist = TH1F("hTime","",105,15,120);
  ch.Project("hTime","time_s/86400", "trapENFCal > 100 && gain==0 && channel != 594 && isGood && isEnr && aenorm > 1 && mH==1 && dcrSlope90>0 && !isLNFill && (dtmu_s < -0.2e-3 || dtmu_s > 1) && " + burstCutStr)
  cumTimeHist = timeHist.GetCumulative()
  cumTimeHist.SetYTitle("Cumulative Counts")
  cumTimeHist.SetXTitle("Days")
  cumTimeHist.Draw()
  canvas.Update()
  
  timeHistAll = TH1F("hTimeAll","",105,15,120);
  ch.Project("hTimeAll","time_s/86400", "trapENFCal > 100 && gain==0 && channel != 594 && isGood && isEnr && aenorm > 1 && mH==1 && !isLNFill && (dtmu_s < -0.2e-3 || dtmu_s > 1) && " + burstCutStr)
  cumTimeHistAll = timeHistAll.GetCumulative()


  rightmax = 1.05*cumTimeHistAll.GetMaximum();
  scale = gPad.GetUymax()/rightmax;
  cumTimeHistAll.SetLineColor(kRed);
  cumTimeHistAll.Scale(scale);
  cumTimeHistAll.Draw("same");
  
  axis =  TGaxis(gPad.GetUxmax(),gPad.GetUymin(),
  gPad.GetUxmax(), gPad.GetUymax(),0,rightmax,510,"+L");
  axis.SetLineColor(kRed);
  axis.SetLabelColor(kRed);
  axis.Draw();
  
  leg = TLegend(0.15,0.6,0.4,0.8)
  leg.AddEntry(cumTimeHist,"DCR Cut Events","l")
  leg.AddEntry(cumTimeHistAll,"All background events","l")
  leg.Draw();
  
   
  canvas.Update()
  value = raw_input('  --> Make sure the hist is as you expect ')
  canvas.Print("dcr_cumulative_plot.pdf" )
  canvas.Print("dcr_cumulative_plot.root" )


if __name__=="__main__":
    main(sys.argv[1:])

