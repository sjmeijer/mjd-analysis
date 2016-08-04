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
  num =  [8685207069.0676746, 1.7618952141698222e+18, 17521485536930826.0]
  den = [1, 50310572.447231829, 701441983664560.88, 1.4012406413698292e+19]
  system = signal.lti(num, den)
  
  tempGuess = 82.48
  gradGuess = 0.0482
  pcRadGuess = 2.563885
  pcLenGuess = 1.440751

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.04,2.5, 1.6)
  det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
  det.LoadFields("P42574A_fields_len.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  
  wf1 = det.GetSimWaveform(10, .18, 10, 3917., 2.5, fitSamples)
  plt.plot(wf1, color = "r")
  
  wf2 = det.GetSimWaveform(10, .18, 10, 3917., 2.75, fitSamples,)
  plt.plot(wf2, color = "b")
  
  wf3 = det.GetSimWaveform(10, .18, 10, 3917., 2.1, fitSamples,)
  plt.plot(wf3,color = "g")
  
  wf4 = det.GetSimWaveform(10, .18, 10, 3917., 3, fitSamples)
  plt.plot(wf4,color = "m")

  wf5 = det.GetSimWaveform(10, .18, 10, 3917., 2.4, fitSamples)
  plt.plot(wf5,color = "black")



  plt.show()


if __name__=="__main__":
    main(sys.argv[1:])


