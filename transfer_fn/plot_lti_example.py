#!/usr/local/bin/python
#from ROOT import *
#TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
#TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt

import numpy as np

#import pymc
#import signal_model as sm
from pymc3 import *
import signal_model_pymc3 as sm3
from detector_model import *
from scipy import signal

####################################################################################################################################################################


def main(argv):


  step = np.zeros(800)
  step[400:] = 1


  wf1= getFitWaveformOpAmp(step)

  wf2 = getFitWaveformRLC(step)
  
  
  wf3 = getFitWaveformFoldedCascode(wf2)
  wf3 /= np.amax(wf3)
  
  wf4 = getFitWaveformFoldedCascode(wf1)
  wf4 /= np.amax(wf4)


  
  plt.figure()
  plt.plot(step, color="r")
  plt.plot(wf1, color="b")
  plt.plot(wf4, color="black")
  
  plt.plot(wf2, color="g")
  plt.plot(wf3, color="magenta")
  
  
  plt.ylim(-0.1,1.1)
  plt.xlim(375,500)

  plt.show()

def getFitWaveformRLC(step):

  lc = 10.
  rc = 10.
 
  num = [1]
  den = [lc, rc, 1]
  system = signal.TransferFunction(num, den)
  t = np.arange(0, len(step))
  
  tout, y, x = signal.lsim(system, step, t)

  return (y)

#  plt.figure()
#  plt.plot(step, color="r")
#  plt.plot(t, y, color="b")
#  plt.ylim(-0.1,1.1)
#
#  plt.show()

def getFitWaveformOpAmp(step):

  r1 = 5
  r2 = 5
  c = 2.

  num = [r2]
  den = [r1*r2*c, r1]
  system = signal.TransferFunction(num, den)
  t = np.arange(0, len(step))
  
  tout, y, x = signal.lsim(system, step, t)
  
  return (y)

def getFitWaveformFoldedCascode(step):

  r = 1.
  c1 = 1.
  c2 = 2000.
 
  num = [c1*r,0]
  den = [c2*r, 1]
  system = signal.TransferFunction(num, den)
  t = np.arange(0, len(step))
  
  tout, y, x = signal.lsim(system, step, t)
  
  return (y)



if __name__=="__main__":
    main(sys.argv[1:])


