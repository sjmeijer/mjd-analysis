#!/usr/bin/env python

import sys, os
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import dnest4

import helpers
from pysiggen import Detector

from dns_wf_model import *

fitSamples = 200
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
gradGuess = 0.01
pcRadGuess = 2.5
pcLenGuess = 1.6
#Create a detector model
detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10)
det.LoadFieldsGrad("fields_impgrad_0-0.02.npz", pcLen=pcLenGuess, pcRad=pcRadGuess)
det.SetFieldsGradInterp(gradGuess)

tf_first_idx = 8
velo_first_idx = 14
trap_idx = 20
grad_idx = 21

wf_idx = 4

wf = wfs[wf_idx]
wf.WindowWaveformTimepoint(fallPercentage=.99, rmsMult=2, earlySamples=50)
print "wf is %d samples long" %wf.wfLength

def fit(argv):

  initializeDetector(det, )
  initializeWaveform(wf, results[wf_idx]['x'])

  # Create a model object and a sampler
  model = Model()
  sampler = dnest4.DNest4Sampler(model,
                                 backend=dnest4.backends.CSVBackend(".",
                                                                    sep=" "))

  seed = 1234
  np.random.seed(seed)
  # Set up the sampler. The first argument is max_num_levels
  gen = sampler.sample(max_num_levels=100, num_steps=1000000, new_level_interval=100000,
                        num_per_step=10000, thread_steps=100,
                        num_particles=5, lam=10, beta=100, seed=seed)

  # Do the sampling (one iteration here = one particle save)
  for i, sample in enumerate(gen):
      print("# Saved {k} particles.".format(k=(i+1)))

  # Run the postprocessing
  dnest4.postprocess()

def plot():
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
        print " (%0.4f)" % det.gradList[grad]

        ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0,  fitSamples, h_smoothing = smooth)

        if ml_wf is None:
          continue

        baseline_trend = np.linspace(b, m*fitSamples+b, fitSamples)
        ml_wf += baseline_trend

        ax0.plot(t_data, ml_wf[:dataLen], color="g", alpha=0.1)
        ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color="g",alpha=0.1)

    ax1.set_ylim(-20, 20)
    plt.show()

if __name__=="__main__":
    if len(sys.argv) < 2:
        fit(sys.argv[1:])
    elif sys.argv[1] == "plot":
        plot()
