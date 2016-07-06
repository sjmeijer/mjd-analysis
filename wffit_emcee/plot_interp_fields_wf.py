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

  gradGuess = 0.05
  pcRadGuess = 2.75
  fitSamples = 150
  
  #Create a detector model
  num = [3.64e+09, 1.88e+17, 6.05e+15]
  den = [1, 4.03e+07, 5.14e+14, 7.15e+18]
  system = signal.lti(num, den)
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f.conf" % (gradGuess,pcRadGuess)
  det =  Detector(detName, temperature=83., timeStep=1., numSteps=fitSamples*10, tfSystem=system)
  det.LoadFields("P42574A_fields.npz")

  wf1 = det.GetSimWaveform(17, .4, 19.5, 3917., 1.5, fitSamples)
  plt.plot(wf1)
  
  det.plotFields()

  
  det.SetFields(2.93, 0.03 )

  wf2 = det.GetSimWaveform(17, .18, 19.5, 3917., 1.5, fitSamples)
  plt.plot(wf2)



  plt.show()


if __name__=="__main__":
    main(sys.argv[1:])


