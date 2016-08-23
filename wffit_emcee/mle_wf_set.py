#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy as np
from scipy import signal

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
  runRanges = [(11511, 11530)]#11540)]
  
  numThreads = 4
  scale_mult = 100.

  #calibration on
  channel = 626
  
  side_padding = 15
  sep_energy = 2109
  dep_energy = 1597
  
  energy_cut_sep = "(trapENFCal > %f && trapENFCal < %f)" % (sep_energy-side_padding,sep_energy+side_padding)
  energy_cut_dep = "(trapENFCal > %f && trapENFCal < %f)" % (dep_energy-side_padding,dep_energy+side_padding)

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
  
    sepHist = TH1F("hSEP","Single Escape Peak",2*side_padding*binsPerKev,sep_energy-side_padding,sep_energy+side_padding);
    chainGat.Project("hSEP", "trapENFCal", "channel == %d && %s" % (channel, energy_cut_sep));
    sepHist.Draw()
    canvas.Update()
    value = raw_input('  --> Make sure the hist is as you expect ')
    if value == 'q': exit(0)
    depHist = TH1F("hDEP","Double Escape Peak",2*side_padding*binsPerKev,dep_energy-side_padding,dep_energy+side_padding);
    chainGat.Project("hDEP", "trapENFCal", "channel == %d && %s" % (channel, energy_cut_dep));
    depHist.Draw()
    canvas.Update()
    value = raw_input('  --> Make sure the hist is as you expect ')
    if value == 'q': exit(0)
    print "expecting %d entries..." % (depHist.GetEntries() + sepHist.GetEntries())


  total_cut = "( %s || %s) && channel == %d " % (energy_cut_sep, energy_cut_dep, channel)


  wfFileName = "multisite_event_set_runs%d-%d.npz" % (runRanges[0][0], runRanges[-1][-1])
  if not os.path.isfile(wfFileName):
    wfs = helpers.GetWaveforms( runRanges,  channel, np.inf, total_cut)
  else:
    data = np.load(wfFileName)
    wfs = data['wfs']

  np.savez(wfFileName, wfs = wfs)

  print "Found %d waveforms" % wfs.size

  args = []
  plt.figure()
  for (idx, wf) in enumerate(wfs):
    wf.WindowWaveformTimepoint(fallPercentage=.99)
    args.append( [15., np.pi/8., 15., wf.wfMax/scale_mult, wf.t0Guess, 10.,  wfs[idx] ]  )
  
#    if wf.energy > 1800:
#      plt.plot(wf.windowedWf, color="r")
#    else:
#      plt.plot(wf.windowedWf, color="g")
#  plt.show()
#  value = raw_input('  --> Make sure the hist is as you expect ')
#  if value == 'q': exit(0)



  fitSamples = 200
  num =  [3478247474.8078203, 1.9351287044375424e+17, 6066014749714584.0]
  den = [1, 40525756.715025946, 508584795912802.25, 7.0511687850000589e+18]
  system = signal.lti(num, den)
  
  tempGuess = 77.89
  gradGuess = 0.0483
  pcRadGuess = 2.591182
  pcLenGuess = 1.613357

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.04,2.5, 1.6)
  det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)

  p = Pool(numThreads, initializer=initializeDetector, initargs=[det])
  print "performing parallelized initial fit..."
  start = timer()
  results = p.map(minimize_waveform_only_star, args)
  end = timer()
  print "Initial fit time: %f" % (end-start)


  np.savez(wfFileName, wfs = wfs, results=results)




if __name__=="__main__":
    main(sys.argv[1:])


