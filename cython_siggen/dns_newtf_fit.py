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
# plt.style.use('presentation')
from matplotlib import gridspec
from matplotlib.colors import LogNorm

# import pandas as pd
import numpy as np
from scipy import signal, interpolate
import multiprocessing

import helpers
from pysiggen import Detector

from dns_newtf_model import *

doInitPlot =0
doContourHist = 0


# doWaveformPlot =0
# doHists = 1
# plotNum = 1000 #for plotting during the Run
doWaveformPlot =1
doHists = 0
plotNum = 100 #for plotting during the Run
numThreads = multiprocessing.cpu_count()

max_sample_idx = 200
fallPercentage = 0.97
fieldFileName = "P42574A_fields_impgrad_0.00000-0.00100.npz"


wfFileName = "P42574A_24_spread.npz"
# wfFileName = "P42574A_12_fastandslow_oldwfs.npz"
if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    #i think wfs 1 and 3 might be MSE
    #wf 2 is super weird

    wfs = data['wfs']

    #one slow waveform
    # fitwfnum = 5
    # wfs = wfs[:fitwfnum+1]
    # wfs = np.delete(wfs, range(0,fitwfnum))
    # numLevels = 150

    # wfs = wfs[0:16:2]
    # wfidxs = [4, 11, 19, 22]
    wfidxs = [0, 5, 8, 14]
    wfs = wfs[wfidxs]
    numLevels = 450

    # 4 medium waveforms
    # wfs = wfs[:8]
    # wfs = np.delete(wfs, [0,1,2,3])

    # #8 wfs questionable provenance
    # wfs = wfs[:11]
    # wfs = np.delete(wfs, [1,2,3])

    numWaveforms = wfs.size
    print "Fitting %d waveforms" % numWaveforms,
    if numWaveforms < numThreads:
      numThreads = numWaveforms
    print "using %d threads" % numThreads

else:
  print "Saved waveform file %s not available" % wfFileName
  exit(0)

colors = ["red" ,"blue", "green", "purple", "orange", "cyan", "magenta", "goldenrod", "brown", "deeppink", "lightsteelblue", "maroon", "violet", "lawngreen", "grey", "chocolate" ]

wfLengths = np.empty(numWaveforms)
wfMaxes = np.empty(numWaveforms)


if doInitPlot: plt.figure(500)
baselineLengths = np.empty(numWaveforms)
for (wf_idx,wf) in enumerate(wfs):
  wf.WindowWaveformAroundMax(fallPercentage=fallPercentage, rmsMult=2, earlySamples=max_sample_idx)
  baselineLengths[wf_idx] = wf.t0Guess

  print "wf %d length %d (entry %d from run %d)" % (wf_idx, wf.wfLength, wf.entry_number, wf.runNumber)
  wfLengths[wf_idx] = wf.wfLength
  wfMaxes[wf_idx] = np.argmax(wf.windowedWf)

  if doInitPlot:
      if len(colors) < numWaveforms:
          color = "red"
      else: color = colors[wf_idx]
      plt.plot(wf.windowedWf, color=color)

baseline_origin_idx = np.amin(baselineLengths) - 30
if baseline_origin_idx < 0:
    print "not enough baseline!!"
    exit(0)

initT0Padding(max_sample_idx, baseline_origin_idx)

if doInitPlot:
    plt.show()
    exit()

siggen_wf_length = (max_sample_idx - np.amin(baselineLengths) + 10)*10
output_wf_length = np.amax(wfLengths) + 1

#Create a detector model
timeStepSize = 1 #ns
detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
det =  Detector(detName, timeStep=timeStepSize, numSteps=siggen_wf_length, maxWfOutputLength =output_wf_length, t0_padding=100 )
det.LoadFieldsGrad(fieldFileName)

def fit(directory):

  initializeDetectorAndWaveforms(det, wfs, reinit=False)
  initMultiThreading(numThreads)

  # Create a model object and a sampler
  model = Model()
  sampler = dnest4.DNest4Sampler(model,
                                 backend=dnest4.backends.CSVBackend(basedir ="./" + directory,
                                                                    sep=" "))

  # Set up the sampler. The first argument is max_num_levels
  gen = sampler.sample(max_num_levels=numLevels, num_steps=100000, new_level_interval=10000,
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

    if doWaveformPlot:
        if num_samples > plotNum: num_samples = plotNum
    print "plotting %d samples" % num_samples

    r_arr = np.empty((numWaveforms, num_samples))
    z_arr = np.empty((numWaveforms, num_samples))
    tf = np.empty((6, num_samples))
    velo = np.empty((6, num_samples))
    wf_params = np.empty((numWaveforms, 8, num_samples))
    det_params = np.empty((2, num_samples))

    velo_priors, velo_lims = get_velo_params()
    # t0_guess, t0_min, t0_max = get_t0_params()
    tf_first_idx, velo_first_idx, grad_idx, trap_idx = get_param_idxs()

    for (idx,params) in enumerate(data[-num_samples:]):
        # params = data.iloc[-(idx+1)]
        # print params

        tf_phi, tf_omega, d, rc1, rc2, rcfrac = params[tf_first_idx:tf_first_idx+6]
        #tf_d = tf_c * tf_dc
        c = -d * np.cos(tf_omega)
        b_ov_a = c - np.tan(tf_phi) * np.sqrt(d**2-c**2)
        a = 1./(1+b_ov_a)
        tf_b = a * b_ov_a
        tf_c = c
        tf_d = d

        h_100_mu0, h_100_lnbeta, h_100_emu, h_111_mu0, h_111_lnbeta, h_111_emu = params[velo_first_idx:velo_first_idx+6]
        charge_trapping = params[trap_idx]
        grad = np.int(params[grad_idx])

        # rc1 = -1./np.log(e_rc1)
        # rc2 = -1./np.log(e_rc2)
        # charge_trapping = -1./np.log(e_charge_trapping)

        h_100_beta = 1./np.exp(h_100_lnbeta)
        h_111_beta = 1./np.exp(h_111_lnbeta)
        h_100_e0 = h_100_emu / h_100_mu0
        h_111_e0 = h_111_emu / h_111_mu0

        tf[:,idx] = tf_phi, tf_omega, tf_d, rc1, rc2, rcfrac
        velo[:,idx] = h_100_mu0, h_100_beta, h_100_emu, h_111_mu0, h_111_beta, h_111_emu
        det_params[:,idx] = np.int(params[grad_idx]), charge_trapping


        det.SetTransferFunction(tf_b, tf_c, tf_d, rc1, rc2, rcfrac)
        det.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
        det.trapping_rc = charge_trapping
        det.SetFieldsGradIdx(grad)

        rad_arr, phi_arr, theta_arr, scale_arr, t0_arr, smooth_arr, m_arr, b_arr = params[trap_idx+1:].reshape((8, numWaveforms))
        print "sample %d:" % idx
        print "  tf params: ",
        print tf_phi, tf_omega, d, rc1, rc2, rcfrac
        print "  velo params: ",
        print h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0
        print "  charge trapping: ",
        print params[trap_idx]
        print "  grad idx (grad): ",
        print params[grad_idx],
        print " (%0.3f)" % det.gradList[grad]

        for (wf_idx,wf) in enumerate(wfs):
          r, phi, z = rad_arr[wf_idx], phi_arr[wf_idx], theta_arr[wf_idx]
          scale, t0, smooth =  scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx]
          m, b = m_arr[wf_idx], b_arr[wf_idx]
          wf_params[wf_idx, :, idx] = r, phi, z, scale, t0, smooth, m, b

        #   r = rad * np.cos(theta)
        #   z = rad * np.sin(theta)
          print "  wf number %d:" % wf_idx
          print "    r: %0.2f , phi: %0.4f, z:%0.2f" % (r, phi/np.pi, z)
        #   print "    rad: %0.2f, theta: %0.4f" % (rad, theta/np.pi)
          print "    t0: %0.2f" % t0
          print "    m: %0.3e, b: %0.3e" % (m,b)
          r_arr[wf_idx, idx], z_arr[wf_idx, idx] = r,z

          if doWaveformPlot:
            ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0,  np.int(output_wf_length), h_smoothing = smooth, alignPoint="max")
            if ml_wf is None:
                continue

            start_idx = -baseline_origin_idx
            end_idx = output_wf_length - baseline_origin_idx - 1
            baseline_trend = np.linspace(m*start_idx+b, m*end_idx+b, output_wf_length)
            ml_wf += baseline_trend

            dataLen = wf.wfLength
            t_data = np.arange(dataLen) * 10
            ax0.plot(t_data, ml_wf[:dataLen], color=colors[wf_idx], alpha=0.1)
            ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color=colors[wf_idx],alpha=0.1)

    ax0.set_ylim(-20, wf.wfMax*1.1)
    ax1.set_ylim(-20, 20)

    if not doHists:
        plt.tight_layout()
        plt.savefig("waveforms.png")
        plt.show()
        exit()

    vFig = plt.figure(2, figsize=(20,10))
    tfLabels = ['b_ov_a', 'c', 'd', 'rc1', 'rc2', 'rcfrac']
    vLabels = ['h_100_mu0', 'h_100_beta', 'h_100_e0','h_111_mu0','h_111_beta', 'h_111_e0']
    vmodes, tfmodes = np.empty(6), np.empty(6)
    num_bins = 100
    for i in range(6):
        idx = (i+1)*3
        axis = vFig.add_subplot(6,3,idx-1)
        axis.set_ylabel(vLabels[i])
        [n, b, p] = axis.hist(velo[i,:], bins=num_bins)
        # axis.axvline(x=(1-velo_lims)*velo_priors[i], color="r")
        # axis.axvline(x=(1+velo_lims)*velo_priors[i], color="r")
        # axis.axvline(x=velo_priors[i], color="g")
        max_idx = np.argmax(n)
        print "%s mode: %f" % (vLabels[i], b[max_idx])

        axis = vFig.add_subplot(6,3,idx-2)
        axis.set_ylabel(tfLabels[i])
        [n, b, p] = axis.hist(tf[i,:], bins=num_bins)
        max_idx = np.argmax(n)
        print "%s mode: %f" % (tfLabels[i], b[max_idx])

        if i==0:
            axis = vFig.add_subplot(6,3,idx)
            axis.set_ylabel("imp grad")
            [n, b, p] = axis.hist(det_params[i,:], bins=num_bins)
            max_idx = np.argmax(n)
            print "%s mode: %f" % ("imp grad", b[max_idx])
        if i==1:
            axis = vFig.add_subplot(6,3,idx)
            axis.set_ylabel("trapping_rc")
            [n, b, p] = axis.hist(det_params[i,:], bins=num_bins)
            max_idx = np.argmax(n)
            print "%s mode: %f" % ("trapping_rc grad", b[max_idx])



    positionFig = plt.figure(3, figsize=(10,10))
    plt.clf()
    colorbars = ["Reds","Blues", "Greens", "Purples", "Oranges", "Greys", "YlOrBr", "PuRd"]

    if not doContourHist:
        for wf_idx in range(numWaveforms):
            xedges = np.linspace(0, np.around(det.detector_radius,1), np.around(det.detector_radius,1)*10+1)
            yedges = np.linspace(0, np.around(det.detector_length,1), np.around(det.detector_length,1)*10+1)
            plt.hist2d(r_arr[wf_idx,:], z_arr[wf_idx,:],  bins=[ xedges,yedges  ],  cmap=plt.get_cmap(colorbars[wf_idx]), cmin=0.1)
            rad_mean = np.mean(wf_params[wf_idx, 0,:])
            print "wf %d rad: %f + %f - %f" % (wf_idx, rad_mean, np.percentile(wf_params[wf_idx, 0,:], 84.1)-rad_mean, rad_mean- np.percentile(wf_params[wf_idx, 0,:], 15.9) )
            # print "--> guess was at %f" %  (np.sqrt(results[wf_idx]['x'][0]**2 + results[wf_idx]['x'][2]**2))
            # plt.colorbar()
        plt.xlabel("r from Point Contact (mm)")
        plt.ylabel("z from Point Contact (mm)")
        plt.xlim(0, det.detector_radius)
        plt.ylim(0, det.detector_length)
        plt.gca().set_aspect('equal', adjustable='box')
    else:
        for wf_idx in range(numWaveforms):
            xedges = np.linspace(0, np.around(det.detector_radius,1), np.around(det.detector_radius,1)*10+1)
            yedges = np.linspace(0, np.around(det.detector_length,1), np.around(det.detector_length,1)*10+1)
            z, xe, ye = np.histogram2d(r_arr[wf_idx,:], z_arr[wf_idx,:],  bins=[ xedges,yedges  ])
            z /= z.sum()
            n=100
            t = np.linspace(0, z.max(), n)
            integral = ((z >= t[:, None, None]) * z).sum(axis=(1,2))
            from scipy import interpolate
            f = interpolate.interp1d(integral, t)
            t_contours = f(np.array([0.9, 0.8]))

            cs = plt.contourf(z.T, t_contours, extent=[0,det.detector_radius,0,det.detector_length],  alpha=1, colors = (colors[wf_idx]), extend='max')
            # cs.cmap.set_over(colors[wf_idx])

        plt.xlabel("r from Point Contact (mm)")
        plt.ylabel("z from Point Contact (mm)")
        # plt.axis('equal')
        plt.xlim(0, det.detector_radius)
        plt.ylim(0, det.detector_length)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()

        plt.savefig("credible_intervals.pdf")

    if numWaveforms == 1:
        #TODO: make this plot work for a bunch of wfs
        vFig = plt.figure(4, figsize=(20,10))
        wfLabels = ['rad', 'phi', 'theta', 'scale', 't0', 'smooth', 'm', 'b']
        num_bins = 100
        for i in range(8):
            axis = vFig.add_subplot(4,2,i+1)
            axis.set_ylabel(wfLabels[i])
            [n, b, p] = axis.hist(wf_params[0, i,:], bins=num_bins)
            # if i == 4:
            #     axis.axvline(x=t0_min, color="r")
            #     axis.axvline(x=t0_max, color="r")
            #     axis.axvline(x=t0_guess, color="g")


    plt.show()

if __name__=="__main__":
    if len(sys.argv) < 2:
        fit("")
    if len(sys.argv) >= 3:
        directory = sys.argv[2]
    else:
        directory = ""

    if sys.argv[1] == "plot":
        plot("sample.txt", directory)
    elif sys.argv[1] == "plot_post":
        plot("posterior_sample.txt", directory)
    else:
        fit(sys.argv[1])
