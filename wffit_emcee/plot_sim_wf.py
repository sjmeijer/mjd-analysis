#!/usr/local/bin/python
from ROOT import *

import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
import scipy.optimize as op

import numpy as np

from scipy import signal, interpolate

from detector_model import *

#from plot_wp_and_efld import plotWP, plotEF

def main(argv):

  fitSamples = 200

  #Prepare detector
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
  det.LoadFields("P42574A_fields_len.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  
  for r in np.linspace(2, det.detector_radius, 20):
    wf1 = det.GetSimWaveform(r, .18, 10, 1, 2., fitSamples)
    plt.plot(np.arange(wf1.size)*10, wf1, color="b")
  
  plt.xlim(0,1000)
  plt.ylim(-0.05,1.1)
  
  plt.xlabel("Time [ns]")
  plt.ylabel("Energy [arb]")
  
#  wf1 = det.GetSimWaveform(10, .18, 10, 3917., 2., fitSamples)
#  plt.plot(wf1, color = "r")
#  
#  wf2 = det.GetSimWaveform(10, .18, 10, 3917., 2.75, fitSamples,)
#  plt.plot(wf2, color = "b")
#  
#  wf3 = det.GetSimWaveform(10, .18, 10, 3917., 2.24, fitSamples,)
#  plt.plot(wf3,color = "g")
#  
#  wf4 = det.GetSimWaveform(10, .18, 10, 3917., 3, fitSamples)
#  plt.plot(wf4,color = "m")
#
#  plt.xlim(1,5)
#  plt.ylim(0,10)



  plt.show()


if __name__=="__main__":
    main(sys.argv[1:])


