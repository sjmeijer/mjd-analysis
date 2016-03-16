from ROOT import *
#TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
#TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
#TROOT.gApplication.ExecuteFile("$HOME/Dev/Analysis/MJDTalkPlotStyle.C")

import numpy as np
import matplotlib.pyplot as plt
import os, sys

import ds0_model as model
import pymc

#plt.style.use('presentation')

fitRange = (590, 630)

def main(argv):




  ##############################################################################################################
  #Load the Wenqin Hist
  ##############################################################################################################
  fData = TFile("analysis_dataset_AoverE_6369_6538_offline_no_cuts.root");
  hData = fData.Get("hall_E_highgain");
  fSim = TFile("processed_MJDemonstrator_bulk_A222_Z86_from_A222_Z86_to_A214_Z82_in_N2_1700000_1keVbin_8fewer_dets.root");
  hSim = fSim.Get("h");

  simArray = rootToArray(hSim, fitRange[0], fitRange[1])
  dataArray = rootToArray(hData, fitRange[0], fitRange[1])

  hist_model = pymc.Model( model.createHistogramFitModel(dataArray, simArray, 24.) )
  M = pymc.MCMC(hist_model, verbose=0)#, db="txt", dbname="Event_%d" % entryNumber)
  M.use_step_method(pymc.Metropolis, M.scale, proposal_sd=.5, proposal_distribution='Normal')
  M.sample(iter=20000, verbose=0)
#  #Create canvas
#  canvas = TCanvas("canvas")
#  pad2 = TPad("pad2","The Superpulse", 0.05,0.05,0.95,0.95)
#  gStyle.SetOptStat(0)
#  pad2.Draw()
#  pad2.cd()
#  hData.SetLineColor(kBlue);
#  hData.Scale(1/20.);
#  hData.Draw("HIST")
#
#  hSim.SetLineColor(kRed);
#  hSim.Draw("SAME")
#
#  canvas.Update()

  burnin = 10000
  scaleFit = np.median(M.trace('scale')[burnin:])
  scalerFitStd = np.std(M.trace('scale')[burnin:])

  print "Best fit scale: %f pm %f" % (scaleFit, scalerFitStd)
  print ">>this translates to an activity of %f p, %f pCi/L" % (scaleFit * 1.7E6 * 27.  /(6.86 *24*3600) / 223.2, scalerFitStd * 1.7E6 * 27.  /(6.86 *24*3600) / 223.2 )

  plt.plot(np.arange(fitRange[0], fitRange[1]), simArray*scaleFit, ls='steps-post', color="red", label="Simulation")
  plt.plot(np.arange(fitRange[0], fitRange[1]), dataArray, ls='steps-post', color="blue", label="Data")
  
  plt.xlabel("Energy [kev]")
  plt.ylabel("Counts")
  plt.legend()
  
  plt.savefig("bestfit_peak.png")
  
  f = plt.figure(2)
  plt.plot(M.trace('scale')[:])
  plt.xlabel("MCMC Step Number")
  plt.ylabel("Scale Parameter")
  plt.savefig("mcmc_steps.png")
  
  f = plt.figure(3)
  plt.hist(M.trace('scale')[burnin:], bins=25, normed=True, histtype="step")
  plt.axvline(x=scaleFit, color="red")
  plt.axvline(x=scaleFit-scalerFitStd, color="red", linestyle=":")
  plt.axvline(x=scaleFit+scalerFitStd, color="red", linestyle=":")
  plt.ylabel("Probability")
  plt.xlabel("Scale Parameter")
  plt.savefig("posterior.png")
  
  f = plt.figure(4)
  simArrayFull = rootToArray(hSim, 0, 3000)
  dataArrayFull = rootToArray(hData, 0, 3000)
  
  plt.semilogy(np.arange(0, 3000), simArrayFull*scaleFit, ls='steps-post', color="red", label="Simulation")
  plt.semilogy(np.arange(0, 3000), dataArrayFull, ls='steps-post', color="blue", label="Data")
  plt.xlabel("Energy [kev]")
  plt.ylabel("Counts")
  plt.legend()
  
  plt.savefig("bestfit_full.png")
  
  plt.ylim(1E0, 1E6)
  plt.show()


#  value = raw_input('  --> Make sure the hist is as you expect ')

def rootToArray(hist, energyLow, energyHigh):
  binLow = hist.FindBin(energyLow)
  binHigh = hist.FindBin(energyHigh)

  loopArray = range(binLow, binHigh )

  npArray = np.empty_like(loopArray)
  for (iArray, iBin) in enumerate(loopArray):
    npArray[iArray] = hist.GetBinContent(iBin)

  return npArray



if __name__=="__main__":
    main(sys.argv[1:])




#
