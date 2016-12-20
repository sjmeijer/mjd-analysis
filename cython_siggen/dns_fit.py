#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of DNest4 in Python using the "StraightLine" example
"""

import dnest4

import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import scipy.optimize as op
import numpy as np
from scipy import signal

import helpers
from detector_model import *
import probability_model_waveform as pmw

from dns_wf_model import *

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
det.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)


nll_wf = lambda *args: -pmw.lnlike_waveform(*args)

def main(argv):
  wf = wfs[5]
  wf.WindowWaveformTimepoint(fallPercentage=.97, rmsMult=2, earlySamples=100)
  initializeDetector(det, )
  pmw.initializeDetector(det, )
  pmw.initializeWaveform(wf)

  minresult = None
  minlike = np.inf

  for r in np.linspace(4, np.floor(det.detector_radius)-3, 3):
    for z in np.linspace(4, np.floor(det.detector_length)-3, 5):
  #        for t0_guess in np.linspace(wf.t0Guess-10, wf.t0Guess+5, 3):
      if not det.IsInDetector(r,0,z): continue
      startGuess = [r, np.pi/8, z, wf.wfMax, wf.t0Guess-5, 10]
      result = op.minimize(nll_wf, startGuess,   method="Nelder-Mead")
      r, phi, z, scale, t0, smooth, = result["x"]
      ml_wf = np.copy(det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing=smooth, ))
      if ml_wf is None:
        print r, z
        continue
      if result['fun'] < minlike:
        minlike = result['fun']
        minresult = result

  initializeWaveform(wf, minresult['x'])

  print "Best fit w:aveform",
  print "  " + str(minresult['x'])

  # Create a model object and a sampler
  model = Model()
  sampler = dnest4.DNest4Sampler(model,
                                 backend=dnest4.backends.CSVBackend(".",
                                                                    sep=" "))

  # Set up the sampler. The first argument is max_num_levels
  gen = sampler.sample(max_num_levels=5000, num_steps=100000, new_level_interval=1000,
                        num_per_step=1000, thread_steps=100,
                        num_particles=5, lam=10, beta=100, seed=1234)

  # Do the sampling (one iteration here = one particle save)
  for i, sample in enumerate(gen):
      print("# Saved {k} particles.".format(k=(i+1)))

  # Run the postprocessing
  dnest4.postprocess()


if __name__=="__main__":
    main(sys.argv[1:])
