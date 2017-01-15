#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of DNest4 in Python using the "StraightLine" example
"""

import dnest4

import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os, shutil
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm

# import pandas as pd
import numpy as np
from scipy import signal
import multiprocessing

import helpers
from pysiggen import Detector

from dns_detzgrad_model import *

doInitPlot = False
doWaveformPlot = True
doHists = False
plotNum = 50 #for plotting during the Run

numThreads = multiprocessing.cpu_count()

wfFileName = "P42574A_24_spread.npz"
if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    #i think wfs 1 and 3 might be MSE
    #wf 2 is super weird

    wfs = data['wfs']
    results = data['results']

    #one slow waveform
    #fitwfnum = 11
    fitwfnum = 7
    wfs = wfs[:fitwfnum+1]
    results = results[:fitwfnum+1]
    wfs = np.delete(wfs, range(0,fitwfnum))
    results = np.delete(results, range(0,fitwfnum))

    # 4 medium waveforms
    # wfs = wfs[:8]
    # results = results[:8]
    # wfs = np.delete(wfs, [0,1,2,3])
    # results = np.delete(results, [0,1,2,3])

    # #8 wfs questionable provenance
    # wfs = wfs[:11]
    # results = results[:11]
    # wfs = np.delete(wfs, [1,2,3])
    # results = np.delete(results, [1,2,3])

    numWaveforms = wfs.size
    print "Fitting %d waveforms" % numWaveforms,
    if numWaveforms < numThreads:
      numThreads = numWaveforms
    print "using %d threads" % numThreads

else:
  print "Saved waveform file %s not available" % wfFileName
  exit(0)

baselineLengths = np.empty(numWaveforms)
for (wf_idx, wf) in enumerate(wfs):
    baselineLengths[wf_idx] = wf.EstimateT0(rmsMult=2)
min_length = np.amin(baselineLengths)
t0_padding = min_length - 10
baseline_origin_idx = t0_padding - 30

if baseline_origin_idx < 0:
    print "You need to make the pre-wf baseline longer"
    exit(0)

colors = ["red" ,"blue", "green", "purple", "orange", "cyan", "magenta", "goldenrod", "brown", "deeppink", "lightsteelblue", "maroon", "violet", "lawngreen", "grey" ]


# t0_padding = 500
wfLengths = np.empty(numWaveforms)
wfMaxes = np.empty(numWaveforms)

if doInitPlot: plt.figure(500)
for (wf_idx,wf) in enumerate(wfs):
  wf.WindowWaveformTimepoint(fallPercentage=.92, rmsMult=2, earlySamples=t0_padding)
  # wf.WindowWaveformTimepoint(fallPercentage=.99, rmsMult=2, earlySamples=10)

  print "wf %d length %d (entry %d from run %d)" % (wf_idx, wf.wfLength, wf.entry_number, wf.runNumber)
  wfLengths[wf_idx] = wf.wfLength
  wfMaxes[wf_idx] = np.argmax(wf.windowedWf)

  if doInitPlot:  plt.plot(wf.windowedWf, color=colors[wf_idx])
if doInitPlot: plt.show()

siggen_wf_length = (np.amax(wfMaxes) - t0_padding + 10)*10
output_wf_length = np.amax(wfLengths)

#Create a detector model
timeStepSize = 1 #ns
detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
det =  Detector(detName, timeStep=timeStepSize, numSteps=siggen_wf_length, maxWfOutputLength =output_wf_length )
det.LoadFieldsGrad("fields_impgrad_0-0.02.npz", pcLen=1.6, pcRad=2.5)

def fit(argv):

  initializeDetectorAndWaveforms(det, wfs, results, reinit=False)
  initMultiThreading(numThreads)
  initT0Padding(t0_padding, baseline_origin_idx)

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



def plot(sample_file_name, directory):
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

      sample_file_name = directory + sample_file_name
      if sample_file_name == directory + "sample.txt":
          shutil.copy(directory+ "sample.txt", directory+"sample_plot.txt")
          sample_file_name = directory + "sample_plot.txt"

    data = np.loadtxt( sample_file_name)
    num_samples = len(data)

    # data = pd.read_csv("sample_plot.txt", delim_whitespace=True, header=None)
    # num_samples = len(data.index)
    print "found %d samples" % num_samples

    print sample_file_name

    if sample_file_name== (directory+"sample_plot.txt"):
        if num_samples > plotNum: num_samples = plotNum
    print "plotting %d samples" % num_samples
    # exit(0)

    r_arr = np.empty((numWaveforms, num_samples))
    z_arr = np.empty((numWaveforms, num_samples))
    tf = np.empty((6, num_samples))
    velo = np.empty((6, num_samples))
    wf_params = np.empty((8, num_samples))

    velo_priors, velo_lims = get_velo_params()
    tf_first_idx, velo_first_idx, grad_idx, trap_idx = get_param_idxs()

    for (idx,params) in enumerate(data[-num_samples:]):
        # params = data.iloc[-(idx+1)]
        # print params

        b_over_a, c, dc, rc1, rc2, rcfrac = params[tf_first_idx:tf_first_idx+6]
        h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0 = params[velo_first_idx:velo_first_idx+6]
        charge_trapping = params[trap_idx]
        grad = np.int(params[grad_idx])

        tf[:,idx] = params[tf_first_idx:tf_first_idx+6]
        velo[:,idx] = params[velo_first_idx:velo_first_idx+6]

        d = c*dc
        det.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)
        det.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
        det.trapping_rc = charge_trapping
        det.SetFieldsGradIdx(grad)

        rad_arr, phi_arr, theta_arr, scale_arr, t0_arr, smooth_arr, m_arr, b_arr = params[trap_idx+1:].reshape((8, numWaveforms))
        print "sample %d:" % idx
        print "  tf params: ",
        print b_over_a, c, d, rc1, rc2, rcfrac
        print "  velo params: ",
        print h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0
        print "  charge trapping: ",
        print params[trap_idx]
        print "  grad idx (grad): ",
        print params[grad_idx],
        print " (%0.3f)" % det.gradList[grad]

        for (wf_idx,wf) in enumerate(wfs):
          rad, phi, theta = rad_arr[wf_idx], phi_arr[wf_idx], theta_arr[wf_idx]
          scale, t0, smooth =  scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx]
          m, b = m_arr[wf_idx], b_arr[wf_idx]
          wf_params[:, idx] = rad, phi, theta, scale, t0, smooth, m, b

          r = rad * np.cos(theta)
          z = rad * np.sin(theta)
          print "  wf number %d:" % wf_idx
          print "    r: %0.2f , phi: %0.4f, z:%0.2f" % (r, phi/np.pi, z)
          print "    rad: %0.2f, theta: %0.4f" % (rad, theta/np.pi)
          print "    t0: %0.2f" % t0
          print "    m: %0.3f, b: %0.3f" % (m,b)
          r_arr[wf_idx, idx], z_arr[wf_idx, idx] = r,z

          if doWaveformPlot:
            ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0,  np.int(output_wf_length), h_smoothing = smooth)
            if ml_wf is None:
                continue

            start_idx = -baseline_origin_idx
            end_idx = dataLen - baseline_origin_idx - 1
            baseline_trend = np.linspace(m*start_idx+b, m*end_idx+b, dataLen)
            ml_wf += baseline_trend

            dataLen = wf.wfLength
            t_data = np.arange(dataLen) * 10
            ax0.plot(t_data, ml_wf[:dataLen], color=colors[wf_idx], alpha=0.1)
            ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color=colors[wf_idx],alpha=0.1)

    ax0.set_ylim(-20, wf.wfMax*1.1)
    ax1.set_ylim(-20, 20)

    if not doHists:
        plt.show()
        exit()

    vFig = plt.figure(2, figsize=(20,10))
    tfLabels = ['b_ov_a', 'c', 'd', 'rc1', 'rc2', 'rcfrac']
    vLabels = ['h_100_mu0', 'h_100_beta', 'h_100_e0','h_111_mu0','h_111_beta', 'h_111_e0']
    vmodes, tfmodes = np.empty(6), np.empty(6)
    num_bins = 100
    for i in range(6):
        idx = (i+1)*2
        axis = vFig.add_subplot(6,2,idx)
        axis.set_ylabel(vLabels[i])
        [n, b, p] = axis.hist(velo[i,:], bins=num_bins)
        axis.axvline(x=(1-velo_lims)*velo_priors[i], color="r")
        axis.axvline(x=(1+velo_lims)*velo_priors[i], color="r")
        axis.axvline(x=velo_priors[i], color="g")
        max_idx = np.argmax(n)
        print "%s mode: %f" % (vLabels[i], b[max_idx])

        axis = vFig.add_subplot(6,2,idx-1)
        axis.set_ylabel(tfLabels[i])
        [n, b, p] = axis.hist(tf[i,:], bins=num_bins)
        max_idx = np.argmax(n)
        print "%s mode: %f" % (tfLabels[i], b[max_idx])

    positionFig = plt.figure(3, figsize=(15,15))
    plt.clf()
    colorbars = ["Reds","Blues", "Greens", "Purples", "Oranges", "Greys", "YlOrBr", "PuRd"]

    for wf_idx in range(numWaveforms):
        xedges = np.linspace(0, np.around(det.detector_radius,1), np.around(det.detector_radius,1)*10+1)
        yedges = np.linspace(0, np.around(det.detector_length,1), np.around(det.detector_length,1)*10+1)
        plt.hist2d(r_arr[wf_idx,:], z_arr[wf_idx,:],  bins=[ xedges,yedges  ], norm=LogNorm(), cmap=plt.get_cmap(colorbars[wf_idx]))
        # plt.colorbar()
    plt.xlabel("r from Point Contact (mm)")
    plt.ylabel("z from Point Contact (mm)")
    plt.axis('equal')

    if numWaveforms == 1:
        #TODO: make this plot work for a bunch of wfs
        vFig = plt.figure(4, figsize=(20,10))
        wfLabels = ['rad', 'phi', 'theta', 'scale', 't0', 'smooth', 'm', 'b']
        num_bins = 100
        for i in range(8):
            axis = vFig.add_subplot(4,2,i+1)
            axis.set_ylabel(wfLabels[i])
            [n, b, p] = axis.hist(wf_params[i,:], bins=num_bins)

    plt.show()

if __name__=="__main__":
    if len(sys.argv) < 2:
        fit(sys.argv[1:])

    if len(sys.argv) >= 3:
        directory = sys.argv[2]
    else:
        directory = ""

    if sys.argv[1] == "plot":
        plot("sample.txt", directory)
    elif sys.argv[1] == "plot_post":
        plot("posterior_sample.txt", directory)
