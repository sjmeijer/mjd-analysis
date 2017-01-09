#!/usr/bin/env python

import sys, os
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import dnest4

import helpers
from pysiggen import Detector

from dns_tf_model import *

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

tf_first_idx = 8

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

rc1_prior = 74.
rc2_prior = 2.08
rc_frac_prior = 0.992

def plot():
    fig1 = plt.figure(0, figsize=(20,10))
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
    num_samples = 200

    r_arr = np.empty(num_samples)
    z_arr = np.empty(num_samples)
    tf = np.empty((3, num_samples))

    for (idx,params) in enumerate(data[-num_samples:]):
  #  for params in data:
        rad, phi, theta, scale, t0, smooth = params[:6]
        m, b = params[6:8]

        r = rad * np.cos(theta)
        z = rad * np.sin(theta)
        r_arr[idx], z_arr[idx] = r,z

        b_over_a, c, dc,  = params[tf_first_idx:tf_first_idx+3]
        d = dc*c
        det.SetTransferFunction(b_over_a, c, d, rc1_prior, rc2_prior, rc_frac_prior)
        tf[:,idx] = params[tf_first_idx:tf_first_idx+3]

        print "new waveform:"
        print "  wf params: ",
        print  r, phi, z, scale, t0, smooth, m, b
        print "  tf params: ",
        print  b_over_a, c, d

        ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0,  fitSamples, h_smoothing = smooth)

        if ml_wf is None:
          continue

        baseline_trend = np.linspace(b, m*fitSamples+b, fitSamples)
        ml_wf += baseline_trend

        ax0.plot(t_data, ml_wf[:dataLen], color="g", alpha=0.1)
        ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color="g",alpha=0.1)

    ax1.set_ylim(-20, 20)

    positionFig = plt.figure(1)
    plt.clf()
    xedges = np.linspace(0, np.around(det.detector_radius,1), np.around(det.detector_radius,1)*10+1)
    yedges = np.linspace(0, np.around(det.detector_length,1), np.around(det.detector_length,1)*10+1)
    plt.hist2d(r_arr, z_arr,  bins=[ xedges,yedges  ])
    plt.colorbar()
    plt.xlabel("r from Point Contact (mm)")
    plt.ylabel("z from Point Contact (mm)")

    plotnum = 300
    tfFig = plt.figure(2)
    tf0 = tfFig.add_subplot(plotnum+11)
    tf1 = tfFig.add_subplot(plotnum+12, )
    tf2 = tfFig.add_subplot(plotnum+13, )
    # tf3 = tfFig.add_subplot(plotnum+14, )
    # tf4 = tfFig.add_subplot(plotnum+15, )
    # tf5 = tfFig.add_subplot(plotnum+16, )

    tf0.set_ylabel('b_ov_a')
    tf1.set_ylabel('c')
    tf2.set_ylabel('dc')
    # tf3.set_ylabel('rc1')
    # tf4.set_ylabel('rc2')
    # tf5.set_ylabel('rcfrac')

    num_bins = 100
    [n, b, p] = tf0.hist(tf[0,:], bins=num_bins)
    [n, b, p] = tf1.hist(tf[1,:], bins=num_bins)
    [n, b, p] = tf2.hist(tf[2,:], bins=num_bins)
    # [n, b, p] = tf3.hist(tf[3,:], bins=num_bins)
    # [n, b, p] = tf4.hist(tf[4,:], bins=num_bins)
    # [n, b, p] = tf5.hist(tf[5,:], bins=num_bins)

    plt.show()

    plt.show()

if __name__=="__main__":
    if len(sys.argv) < 2:
        fit(sys.argv[1:])
    elif sys.argv[1] == "plot":
        plot()
