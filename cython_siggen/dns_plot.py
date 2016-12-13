#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of DNest4 in Python using the "StraightLine" example
"""

import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib import gridspec


import scipy.optimize as op
import numpy as np
from scipy import signal

import helpers
from detector_model import *
from probability_model_hdxwf import *
from probability_model_waveform import *

from dns_wf_model import *

fitSamples = 130
timeStepSize = 1

wfFileName = "P42574A_12_fastandslow_oldwfs.npz"
if os.path.isfile(wfFileName):
  data = np.load(wfFileName)
  wfs = data['wfs']
  results = data['results']
  numWaveforms = wfs.size

else:
  print "No saved waveforms available.  Loading from Data"
  exit(0)


tempGuess = 79.071172
gradGuess = 0.04
pcRadGuess = 2.5
pcLenGuess = 1.6
#Create a detector model
detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10)
det.LoadFieldsGrad("fields_impgrad.npz", pcLen=pcLenGuess, pcRad=pcRadGuess)
det.SetFieldsGradInterp(gradGuess)

b_over_a = 0.107213
c = -0.815152
d = 0.822696
rc1 = 74.4
rc2 = 1.79
rcfrac = 0.992
trapping_rc = 120#us
det.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)
#  det.trapping_rc = trapping_rc #us
det.trapping_rc = trapping_rc

#do my own hole mobility model based on bruy
det.siggenInst.set_velocity_type(1)
h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0 = 66333., 0.744, 181., 107270., 0.580, 100.


def main(argv):
  wf = wfs[3]
  wf.WindowWaveformTimepoint(fallPercentage=.995, rmsMult=2)
  
  fig1 = plt.figure(1)
  plt.clf()
  gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
  ax0 = plt.subplot(gs[0])
  ax1 = plt.subplot(gs[1], sharex=ax0)
  ax1.set_xlabel("Digitizer Time [ns]")
  ax0.set_ylabel("Voltage [Arb.]")
  ax1.set_ylabel("Residual")

  dataLen = wf.wfLength
  t_data = np.arange(dataLen) * 10
  ax0.plot(t_data, wf.windowedWf, color="r")
  
  data = np.loadtxt("posterior_sample.txt", dtype={'names': ('r', 'phi', 'z', 'scale', 't0', 'smooth'), 'formats': ('f4', 'f4', 'f4', 'f4', 'f4', 'f4')})
  
  for (r, phi, z, scale, t0, smooth) in data:
      ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0,  fitSamples, h_smoothing = smooth)
      ax0.plot(t_data, ml_wf[:dataLen], color="g", alpha=0.1)
      ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color="g",alpha=0.1)

  plt.show()


if __name__=="__main__":
    main(sys.argv[1:])

