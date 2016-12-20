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

from dns_det_model import *

fitSamples = 425
timeStepSize = 1

wfFileName = "P42574A_12_fastandslow_oldwfs.npz"
if os.path.isfile(wfFileName):
  data = np.load(wfFileName)
  wfs = data['wfs']
  results = data['results']
  wfs = wfs[:10]
  results = results[:10]

  #i think wfs 1 and 3 might be MSE
  wfs = np.delete(wfs, [1,3])
  results = np.delete(results, [1,3])

  numWaveforms = wfs.size
  print "Fitting %d waveforms" % numWaveforms

else:
  print "No saved waveforms available.  Loading from Data"
  exit(0)

#Create a detector model
detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
det =  Detector(detName, timeStep=timeStepSize, numSteps=fitSamples*10)
det.LoadFieldsGrad("fields_impgrad_0-0.06.npz", pcLen=1.6, pcRad=2.5)

def main(argv):
  for wf in wfs:
      wf.WindowWaveformTimepoint(fallPercentage=.97, rmsMult=2, earlySamples=100)

  initializeDetectorAndWaveforms(det, wfs, results, reinit=False)
  initMultiThreading(8)

  # Create a model object and a sampler
  model = Model()
  sampler = dnest4.DNest4Sampler(model,
                                 backend=dnest4.backends.CSVBackend(".",
                                                                    sep=" "))

  # Set up the sampler. The first argument is max_num_levels
  gen = sampler.sample(max_num_levels=5000, num_steps=100000, new_level_interval=10000,
                        num_per_step=10000, thread_steps=100,
                        num_particles=5, lam=10, beta=100, seed=1234)

  # Do the sampling (one iteration here = one particle save)
  for i, sample in enumerate(gen):
      print("# Saved {k} particles.".format(k=(i+1)))

  # Run the postprocessing
  dnest4.postprocess()


if __name__=="__main__":
    main(sys.argv[1:])
