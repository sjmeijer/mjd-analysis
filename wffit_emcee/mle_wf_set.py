#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy as np
from scipy import signal

import helpers
from detector_model import *
from probability_model_hier import *

from pymc3 import *
import signal_model_hierarchical as sm3


def main(argv):

  plt.ion()

  
  fitSamples = 200
  
  #Prepare detector
  num =  [3478247474.8078203, 1.9351287044375424e+17, 6066014749714584.0]
  den = [1, 40525756.715025946, 508584795912802.25, 7.0511687850000589e+18]
  system = signal.lti(num, den)
  
  tempGuess = 77.89
  gradGuess = 0.0483
  pcRadGuess = 2.591182
  pcLenGuess = 1.613357
  
  numThreads=4
  runRanges = [(13420,13420)]
#  runRanges = [(13420,13429)]
  channel = 626
  wfFileName = "dep_fits_%d-%d.npz" % (runRanges[0][0], runRanges[-1][-1])
  cut = "trapECal>%f && trapECal<%f" %  (1580,1604)
  wfs = helpers.GetWaveforms(runRanges, channel, np.inf, cut)
  numWaveforms = wfs.size
  print "Going to fit %d waveforms" % numWaveforms
  

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)

  #prep holders for each wf-specific param
  r_arr = np.empty(numWaveforms)
  phi_arr = np.empty(numWaveforms)
  z_arr = np.empty(numWaveforms)
  scale_arr = np.empty(numWaveforms)
  t0_arr = np.empty(numWaveforms)
  smooth_arr = np.ones(numWaveforms)*7.

  simWfArr = np.empty((1,numWaveforms, fitSamples))

  args = []
  for (idx, wf) in enumerate(wfs):
    wf.WindowWaveformTimepoint(fallPercentage=.99)
    args.append( [15, np.pi/8, 15, wf.wfMax/100., wf.t0Guess,  5.,  wfs[idx] ]  )

  if True:
    p = Pool(numThreads, initializer=initializeDetector, initargs=[det])
    print "performing parallelized initial fit..."
    results = p.map(minimize_waveform_only_star, args)
    np.savez(wfFileName, wfs = wfs, results=results )

    fig = plt.figure()
    for (idx,wf) in enumerate(wfs):
      wf.WindowWaveform(200)
      r_arr[idx], phi_arr[idx], z_arr[:], scale_arr[idx], t0_arr[idx], smooth_arr[idx]  = results[idx]['x']
      
      print "WF number %d:" % idx
      print "  >>r: %f\n  >>phi %f\n  >>z %f\n  >>e %f\n  >>t0 %f\n >>smooth %f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx])
      ml_wf = det.GetSimWaveform(r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx]*100, t0_arr[idx], fitSamples, smoothing = smooth_arr[idx])
      plt.plot(ml_wf, color="b")
      plt.plot(wf.windowedWf, color="r")
    value = raw_input('  --> Press q to quit, any other key to continue\n')
    if value == 'q': exit(0)


if __name__=="__main__":
    main(sys.argv[1:])


