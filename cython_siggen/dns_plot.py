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


#from dns_wf_model import *

fitSamples = 400
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
det.LoadFieldsGrad("fields_impgrad_0-0.06.npz", pcLen=pcLenGuess, pcRad=pcRadGuess)
det.SetFieldsGradInterp(gradGuess)

b_over_a = 0.107213
c = -0.815152
d = 0.822696
rc1 = 80.013468
rc2 = 2.078342
rcfrac = 0.992
trapping_rc = 120#us
det.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)
#  det.trapping_rc = trapping_rc #us
det.trapping_rc = trapping_rc

#do my own hole mobility model based on bruy
det.siggenInst.set_velocity_type(1)
h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0 = 66333., 0.744, 181., 107270., 0.580, 100.

tf_first_idx = 8
velo_first_idx = 14
trap_idx = 20
grad_idx = 21


def main(argv):
  wf = wfs[5]
  wf.WindowWaveformTimepoint(fallPercentage=.97, rmsMult=2, earlySamples=100)

  fig1 = plt.figure(1, figsize=(20,10))
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

#  data = np.loadtxt("posterior_sample.txt")
  data = np.loadtxt("sample.txt")

  print "found %d samples" % len(data)

  for params in data[-100:]:
#  for params in data:
#      print params
      rad, phi, theta, scale, t0, smooth = params[:6]
      m, b = params[6:8]
      b_over_a, c, d, rc1, rc2, rcfrac = params[tf_first_idx:tf_first_idx+6]

      r = rad * np.cos(theta)
      z = rad * np.sin(theta)

#      print rc1, rc2, rcfrac

      det.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)

      h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0 = params[velo_first_idx:velo_first_idx+6]
      det.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)

      print "new waveform:"
      print "  wf params: ",
      print  r, phi, z, scale, t0, smooth, m, b
      print "  tf params: ",
      print b_over_a, c, d, rc1, rc2, rcfrac
      print "  velo params: ",
      print h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0

      det.trapping_rc = params[trap_idx]
      print "  charge trapping: ",
      print params[trap_idx]

      grad = np.int(params[grad_idx])
      det.SetFieldsGradIdx(grad)
      print "  grad idx (grad): ",
      print params[grad_idx],
      print " (%0.3f)" % det.gradList[grad]

      ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0,  fitSamples, h_smoothing = smooth)

      baseline_trend = np.linspace(b, m*fitSamples+b, fitSamples)
      ml_wf += baseline_trend

      if ml_wf is None:
        continue

      ax0.plot(t_data, ml_wf[:dataLen], color="g", alpha=0.1)
      ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color="g",alpha=0.1)

#  ax0.set_ylim(0, wf.wfMax*1.1)
  ax1.set_ylim(-20, 20)
  plt.show()


if __name__=="__main__":
    main(sys.argv[1:])
