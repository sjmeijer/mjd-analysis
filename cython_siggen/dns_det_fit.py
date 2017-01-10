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
from matplotlib import gridspec
from matplotlib.colors import LogNorm


import scipy.optimize as op
import numpy as np
from scipy import signal
import multiprocessing

import helpers
from pysiggen import Detector

from dns_det_model import *

numThreads = multiprocessing.cpu_count()
timeStepSize = 1
tf_first_idx = 0
velo_first_idx = 6
# trap_idx = 12
grad_idx = 12

wfFileName = "P42574A_12_fastandslow_oldwfs.npz"
if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    wfs = data['wfs']
    results = data['results']
    # wfs = wfs[:3]
    # results = results[:3]
    #
    # #i think wfs 1 and 3 might be MSE
    # wfs = np.delete(wfs, [0,1])
    # results = np.delete(results, [0,1])

    wfs = wfs[:8]
    results = results[:8]

    #i think wfs 1 and 3 might be MSE
    wfs = np.delete(wfs, [0,1,2,3])
    results = np.delete(results, [0,1,2,3])

    numWaveforms = wfs.size
    print "Fitting %d waveforms" % numWaveforms,
    if numWaveforms < numThreads:
      numThreads = numWaveforms
    print "using %d threads" % numThreads

else:
  print "Saved waveform file %s not available" % wfFileName
  exit(0)

colors = ["red" ,"blue", "green", "purple", "orange", "cyan", "magenta", "goldenrod", "brown", "deeppink", "lightsteelblue", "maroon", "violet", "lawngreen", "grey" ]
fitSamples = 0

doInitPlot = False#True
if doInitPlot: plt.figure(500)
for (wf_idx,wf) in enumerate(wfs):
  wf.WindowWaveformTimepoint(fallPercentage=.99, rmsMult=2, earlySamples=10)
  print "wf %d length %d" % (wf_idx, wf.wfLength)
  if wf.wfLength >= fitSamples:
      fitSamples = wf.wfLength + 1
  if doInitPlot:  plt.plot(wf.windowedWf, color=colors[wf_idx])

if doInitPlot: plt.show()

#Create a detector model
detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
det =  Detector(detName, timeStep=timeStepSize, numSteps=fitSamples*10)
det.LoadFieldsGrad("fields_impgrad_0-0.02.npz", pcLen=1.6, pcRad=2.5)

def fit(argv):

  initializeDetectorAndWaveforms(det, wfs, results, reinit=False)
  initMultiThreading(numThreads)

  # Create a model object and a sampler
  model = Model()
  sampler = dnest4.DNest4Sampler(model,
                                 backend=dnest4.backends.CSVBackend(".",
                                                                    sep=" "))

  # Set up the sampler. The first argument is max_num_levels
  gen = sampler.sample(max_num_levels=500, num_steps=100000, new_level_interval=10000,
                        num_per_step=1000, thread_steps=100,
                        num_particles=5, lam=10, beta=100, seed=1234)

  # Do the sampling (one iteration here = one particle save)
  for i, sample in enumerate(gen):
      print("# Saved {k} particles.".format(k=(i+1)))

  # Run the postprocessing
  # dnest4.postprocess()



def plot(sample_file_name):
    fig1 = plt.figure(0, figsize=(20,10))
    plt.clf()
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.set_xlabel("Digitizer Time [ns]")
    ax0.set_ylabel("Voltage [Arb.]")
    ax1.set_ylabel("Residual")

    for wf in wfs:
      dataLen = wf.wfLength
      t_data = np.arange(dataLen) * 10
      ax0.plot(t_data, wf.windowedWf, color="black")

#  data = np.loadtxt("posterior_sample.txt")
    data = np.loadtxt(sample_file_name)
    num_samples = len(data)
    print "found %d samples" % num_samples

    if sample_file_name=="sample.txt":
        num_samples = 500

    r_arr = np.empty((numWaveforms, num_samples))
    z_arr = np.empty((numWaveforms, num_samples))
    tf = np.empty((6, num_samples))
    velo = np.empty((6, num_samples))
    velo_priors = [ 66333., 0.744, 181., 107270., 0.580, 100.]
    velo_lims = 0.2

    for (idx,params) in enumerate(data[-num_samples:]):

        b_over_a, c, dc, rc1, rc2, rcfrac = params[tf_first_idx:tf_first_idx+6]
        h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0 = params[velo_first_idx:velo_first_idx+6]
        # charge_trapping = params[trap_idx]
        grad = np.int(params[grad_idx])

        tf[:,idx] = params[tf_first_idx:tf_first_idx+6]
        velo[:,idx] = params[velo_first_idx:velo_first_idx+6]

        d = c*dc
        det.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)
        det.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
        # det.trapping_rc = charge_trapping
        det.SetFieldsGradIdx(grad)

        rad_arr, phi_arr, theta_arr, scale_arr, t0_arr, smooth_arr, m_arr, b_arr = params[grad_idx+1:].reshape((8, numWaveforms))
        print "sample %d:" % idx
        print "  tf params: ",
        print b_over_a, c, d, rc1, rc2, rcfrac
        print "  velo params: ",
        print h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0
        # print "  charge trapping: ",
        # print params[trap_idx]
        print "  grad idx (grad): ",
        print params[grad_idx],
        print " (%0.3f)" % det.gradList[grad]

        for (wf_idx,wf) in enumerate(wfs):
          rad, phi, theta = rad_arr[wf_idx], phi_arr[wf_idx], theta_arr[wf_idx]
          scale, t0, smooth =  scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx]
          m, b = m_arr[wf_idx], b_arr[wf_idx]
          r = rad * np.cos(theta)
          z = rad * np.sin(theta)
          print "  wf number %d:" % wf_idx
          print "    r: %0.2f , phi: %0.4f, z:%0.2f" % (r, phi/np.pi, z)
          print "    rad: %0.2f, theta: %0.4f" % (rad, theta/np.pi)
          print "    t0: %0.2f" % t0
          print "    m: %0.3f, b: %0.3f" % (m,b)
          r_arr[wf_idx, idx], z_arr[wf_idx, idx] = r,z

          ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0,  fitSamples, h_smoothing = smooth)
          if ml_wf is None:
            continue

          baseline_trend = np.linspace(b, m*fitSamples+b, fitSamples)
          ml_wf += baseline_trend

          dataLen = wf.wfLength
          t_data = np.arange(dataLen) * 10
          ax0.plot(t_data, ml_wf[:dataLen], color=colors[wf_idx], alpha=0.1)
          ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color=colors[wf_idx],alpha=0.1)

    ax0.set_ylim(-20, wf.wfMax*1.1)
    ax1.set_ylim(-20, 20)

    plotnum = 600
    tfFig = plt.figure(1)
    tf0 = tfFig.add_subplot(plotnum+11)
    tf1 = tfFig.add_subplot(plotnum+12, )
    tf2 = tfFig.add_subplot(plotnum+13, )
    tf3 = tfFig.add_subplot(plotnum+14, )
    tf4 = tfFig.add_subplot(plotnum+15, )
    tf5 = tfFig.add_subplot(plotnum+16, )
    # tf6 = tfFig.add_subplot(plotnum+17, )

    tf0.set_ylabel('b_ov_a')
    tf1.set_ylabel('c')
    tf2.set_ylabel('dc')
    tf3.set_ylabel('rc1')
    tf4.set_ylabel('rc2')
    tf5.set_ylabel('rcfrac')
    # tf6.set_ylabel('grad_idx')

    num_bins = 100
    [n, b, p] = tf0.hist(tf[0,:], bins=num_bins)
    [n, b, p] = tf1.hist(tf[1,:], bins=num_bins)
    [n, b, p] = tf2.hist(tf[2,:], bins=num_bins)
    [n, b, p] = tf3.hist(tf[3,:], bins=num_bins)
    [n, b, p] = tf4.hist(tf[4,:], bins=num_bins)
    [n, b, p] = tf5.hist(tf[5,:], bins=num_bins)
    # [n, b, p] = tf6.hist(tf[6,:], bins=num_bins)

    plotnum = 600
    vFig = plt.figure(2)
    vLabels = ['h_100_mu0', 'h_100_beta', 'h_100_e0','h_111_mu0','h_111_beta', 'h_111_e0']
    vAxes = []
    num_bins = 100
    for i in range(plotnum/100):
        axis = vFig.add_subplot(plotnum+10 + i+1)
        axis.set_ylabel('h_100_mu0')
        [n, b, p] = axis.hist(velo[i,:], bins=num_bins)
        axis.axvline(x=(1-velo_lims)*velo_priors[i], color="r")
        axis.axvline(x=(1+velo_lims)*velo_priors[i], color="r")
        axis.axvline(x=velo_priors[i], color="g")


    positionFig = plt.figure(3)
    plt.clf()
    colorbars = ["Reds","Blues", "Greens", "Purples", "Oranges"]

    for wf_idx in range(numWaveforms):
        xedges = np.linspace(0, np.around(det.detector_radius,1), np.around(det.detector_radius,1)*10+1)
        yedges = np.linspace(0, np.around(det.detector_length,1), np.around(det.detector_length,1)*10+1)
        plt.hist2d(r_arr[wf_idx,:], z_arr[wf_idx,:],  bins=[ xedges,yedges  ], norm=LogNorm(), cmap=plt.get_cmap(colorbars[wf_idx]))
        # plt.colorbar()
    plt.xlabel("r from Point Contact (mm)")
    plt.ylabel("z from Point Contact (mm)")

    plt.show()

if __name__=="__main__":
    if len(sys.argv) < 2:
        fit(sys.argv[1:])
    elif sys.argv[1] == "plot":
        plot("sample.txt")
    elif sys.argv[1] == "plot_post":
        plot("posterior_sample.txt")
