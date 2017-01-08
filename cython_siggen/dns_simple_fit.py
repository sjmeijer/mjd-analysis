#!/usr/bin/env python

import sys, os
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import dnest4

import helpers
from pysiggen import Detector

from dns_simple_model import *

fitSamples = 120
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
det.LoadFieldsGrad("fields_impgrad_0-0.06.npz", pcLen=pcLenGuess, pcRad=pcRadGuess)
det.SetFieldsGradInterp(gradGuess)
det.SetTransferFunction(5.31003292334, -0.808557803157, 0.815966976844, 81.8681451166, 3.6629565274, 0.995895193187)

wf_idx = 5

wf = wfs[wf_idx]
wf.WindowWaveformTimepoint(fallPercentage=.997, rmsMult=2, earlySamples=10)
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
  gen = sampler.sample(max_num_levels=50, num_steps=10000, new_level_interval=10000,
                        num_per_step=1000, thread_steps=100,
                        num_particles=5, lam=10, beta=100, seed=seed)

  # Do the sampling (one iteration here = one particle save)
  for i, sample in enumerate(gen):
      print("# Saved {k} particles.".format(k=(i+1)))

  # Run the postprocessing
  # dnest4.postprocess()

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
    num_samples = len(data)
    print "found %d samples" % num_samples
    num_samples = 100
    r_arr = np.empty(num_samples)
    z_arr = np.empty(num_samples)

    for (idx,params) in enumerate(data[-num_samples:]):
  #  for params in data:
  #      print params
        rad, phi, theta, scale, t0, smooth = params[:6]
        m, b = params[6:8]

        r = rad * np.cos(theta)
        z = rad * np.sin(theta)
        r_arr[idx], z_arr[idx] = r,z

        print "new waveform:"
        print "  wf params: ",
        print  r, phi, z, scale, t0, smooth, m, b

        ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0,  fitSamples, h_smoothing = smooth)

        if ml_wf is None:
          continue

        baseline_trend = np.linspace(b, m*fitSamples+b, fitSamples)
        ml_wf += baseline_trend

        ax0.plot(t_data, ml_wf[:dataLen], color="g", alpha=0.1)
        ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color="g",alpha=0.1)

    ax1.set_ylim(-20, 20)

    positionFig = plt.figure(2)
    plt.clf()
    xedges = np.linspace(0, np.around(det.detector_radius,1), np.around(det.detector_radius,1)*10+1)
    yedges = np.linspace(0, np.around(det.detector_length,1), np.around(det.detector_length,1)*10+1)
    plt.hist2d(r_arr, z_arr,  bins=[ xedges,yedges  ])
    plt.colorbar()
    plt.xlabel("r from Point Contact (mm)")
    plt.ylabel("z from Point Contact (mm)")

    plt.show()

if __name__=="__main__":
    if len(sys.argv) < 2:
        fit(sys.argv[1:])
    elif sys.argv[1] == "plot":
        plot()
