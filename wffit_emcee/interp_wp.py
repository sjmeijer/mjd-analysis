#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy as np
from scipy import signal

from detector_model import *


def main(argv):

  plt.ion()
  
  fitSamples = 250 #has to be longer than the longest wf you're gonna fit
  tempGuess = 81.5
  
  #Set up detectors
  num = [3.64e+09, 1.88e+17, 6.05e+15]
  den = [1, 4.03e+07, 5.14e+14, 7.15e+18]
  system = signal.lti(num, den)

  grad = 0.01, 0.09
  pcRad = 1.65

  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f.conf" % (grad,pcRad)
  det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)




  value = raw_input('  --> Press q to quit, any other key to continue\n')


if __name__=="__main__":
    main(sys.argv[1:])


