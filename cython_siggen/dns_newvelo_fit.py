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

from dns_newvelo_model import *


doMaxInterp = 1
reinitializeDetector = 0

doInitPlot =0

doContourHist = 0
doWaveformPlot =0
doHists = 1
plotNum = 1000 #for plotting during the Run
doVeloPlot = 1

# doWaveformPlot =1
# doHists = 0
# doVeloPlot = 0
# plotNum = 100 #for plotting during the Run
#

numThreads = multiprocessing.cpu_count()

max_sample_idx = 200
fallPercentage = 0.97
fieldFileName = "P42574A_mar25_21by21.npz"
# fieldFileName = "P42574A_bull_fields_impAndAvg_11by11.npz"
#fieldFileName = "P42574A_bull_fields_lowimp_impAndAvg_11by11.npz"
#
detName = "conf/P42574A_bull.conf"
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

    #wfs = wfs[0:24:3]
    # wfidxs = [0, 5, 8, 11, 14, 17, 20, 23]
    # wfidxs = [0, 5, 8, 14]
    # wfs = wfs[wfidxs]

    #it looks like wf10 is bad.  Let's try something else
    wfidxs = [8, 11, 12, 14, 16, 18, 20, 22]
    # wfidxs = [7, 8, 9, 11, 12, 13, 14, 15]
    wfs = wfs[wfidxs]

    numLevels = 600

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

  print "wf %d length %d (entry %d from run %d, color is %s)" % (wf_idx, wf.wfLength, wf.entry_number, wf.runNumber, colors[wf_idx])
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
det =  Detector(detName, timeStep=timeStepSize, numSteps=siggen_wf_length, maxWfOutputLength =output_wf_length, t0_padding=100 )
det.LoadFieldsGrad(fieldFileName)

def fit(directory):
  if reinitializeDetector:
      initializeDetectorAndWaveforms(det.__getstate__(), wfs, reinit=reinitializeDetector, doInterp=doMaxInterp)
  else:
      initializeDetectorAndWaveforms(det, wfs, reinit=reinitializeDetector, doInterp=doMaxInterp)
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

    if doWaveformPlot:
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

    if sample_file_name== (directory+"sample_plot.txt"):
        if num_samples > plotNum: num_samples = plotNum

    if doWaveformPlot:
        if num_samples > plotNum: num_samples = plotNum
    print "plotting %d samples" % num_samples

    r_arr = np.empty((numWaveforms, num_samples))
    z_arr = np.empty((numWaveforms, num_samples))
    phi_hist_arr = np.empty((numWaveforms, num_samples))

    tf = np.empty((6, num_samples))
    velo = np.empty((6, num_samples))
    wf_params = np.empty((numWaveforms, 8, num_samples))
    det_params = np.empty((3, num_samples))


    tf_first_idx, velo_first_idx,  grad_idx, trap_idx = get_param_idxs()

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

        h_100_vlo, h_111_vlo, h_100_vhi, h_111_vhi, h_100_lnbeta, h_111_lnbeta, = params[velo_first_idx:velo_first_idx+6]
        # k0_0, k0_1, k0_2, k0_3 = params[k0_first_idx:k0_first_idx+4]
        charge_trapping = params[trap_idx]
        grad, avg_imp = params[grad_idx], params[grad_idx+1]

        # rc1 = -1./np.log(e_rc1)
        # rc2 = -1./np.log(e_rc2)
        # charge_trapping = -1./np.log(e_charge_trapping)

        tf[:,idx] = tf_phi, tf_omega, tf_d, rc1, rc2, rcfrac
        velo[:,idx] = params[velo_first_idx:velo_first_idx+6]
        det_params[:,idx] = grad, avg_imp, charge_trapping

        h_100_mu0, h_100_beta, h_100_e0 = get_velo_params(h_100_vlo, h_100_vhi, h_100_lnbeta)
        h_111_mu0, h_111_beta, h_111_e0 = get_velo_params(h_111_vlo, h_111_vhi, h_111_lnbeta)

        det.SetTransferFunction(tf_b, tf_c, tf_d, rc1, rc2, rcfrac)
        det.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
        # det.siggenInst.set_k0_params(k0_0, k0_1, k0_2, k0_3)
        det.trapping_rc = charge_trapping
        det.SetGrads(grad, avg_imp)

        rad_arr, phi_arr, theta_arr, scale_arr, t0_arr, smooth_arr, m_arr, b_arr = params[trap_idx+1:].reshape((8, numWaveforms))


        for (wf_idx,wf) in enumerate(wfs):
          rad, phi, theta = rad_arr[wf_idx], phi_arr[wf_idx], theta_arr[wf_idx]
          scale, t0, smooth =  scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx]
          m, b = m_arr[wf_idx], b_arr[wf_idx]
          wf_params[wf_idx, :, idx] = rad, phi, theta, scale, t0, smooth, m, b

          r = rad * np.cos(theta)
          z = rad * np.sin(theta)
          r_arr[wf_idx, idx], z_arr[wf_idx, idx] = r,z
          phi_hist_arr[wf_idx,idx] = phi

          if doWaveformPlot:
            plt.figure(fig1.number)
            ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0,  np.int(output_wf_length), h_smoothing = smooth, alignPoint="max", doMaxInterp=doMaxInterp)
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

    if doWaveformPlot:
        ax0.set_ylim(-20, wf.wfMax*1.1)
        ax1.set_ylim(-20, 20)

    if doWaveformPlot and not doHists:
        plt.tight_layout()
        plt.savefig("waveforms.png")
        plt.show()
        exit()

    vFig = plt.figure(2, figsize=(20,10))
    tfLabels = ['tf_phi', 'tf_omega', 'd', 'rc1', 'rc2', 'rcfrac']
    vLabels = ['h_100_vlo', 'h_111_vlo', 'h_100_vhi', 'h_111_vhi', 'h_100_beta', 'h_111_beta']
    vmodes, tfmodes = np.empty(6), np.empty(6)
    num_bins = 100
    for i in range(6):
        idx = (i+1)*3

        # if i < 5:
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
            print "%s mode: %f" % ("imp_grad", b[max_idx])
        if i==1:
            axis = vFig.add_subplot(6,3,idx)
            axis.set_ylabel("imp average")
            [n, b, p] = axis.hist(det_params[i,:], bins=num_bins)
            max_idx = np.argmax(n)
            print "%s mode: %f" % ("avg_imp", b[max_idx])
        if i==2:
            axis = vFig.add_subplot(6,3,idx)
            axis.set_ylabel("trapping_rc")
            [n, b, p] = axis.hist(det_params[i,:], bins=num_bins)
            max_idx = np.argmax(n)
            print "%s mode: %f" % ("trapping_rc", b[max_idx])



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

    if doVeloPlot:
        velo_plot = plt.figure(5)
        fields_log = np.linspace(np.log(100), np.log(10000), 100)
        fields = np.exp(fields_log)

        for  idx in range(num_samples):
            h_100_vlo, h_111_vlo, h_100_vhi, h_111_vhi, h_100_lnbeta, h_111_lnbeta, = velo[:,idx]
            # print "v params: ",
            # print mu_0, h_100_lnbeta, h_111_lnbeta, h_111_emu, h_100_mult

            h100 = np.empty_like(fields)
            h111 = np.empty_like(fields)

            h_100_mu0, h_100_beta, h_100_e0 = get_velo_params(h_100_vlo, h_100_vhi, h_100_lnbeta)
            h_111_mu0, h_111_beta, h_111_e0 = get_velo_params(h_111_vlo, h_111_vhi, h_111_lnbeta)


            for (idx,field) in enumerate(fields):
                h100[idx] = find_drift_velocity_bruyneel(field, h_100_mu0, h_100_beta,h_100_e0)
                h111[idx] = find_drift_velocity_bruyneel(field, h_111_mu0, h_111_beta,h_111_e0)



            # if h100[-1] > 2E7:
            #     print  h_100_vlo, h_100_vhi, h_100_lnbeta
            # if h111[idx] > 2E7:
            #     print  h_111_vlo, h_111_vhi, h_111_lnbeta

            # test_fields = [10, 500,2000]
            # for (idx,field) in enumerate(test_fields):
            #     v100 = find_drift_velocity_bruyneel(field, h_100_mu0, h_100_beta,h_100_e0)
            #     v111 = find_drift_velocity_bruyneel(field, h_111_mu0, h_111_beta,h_111_e0)
            #
            #     if v100 < v111: print "bad! %d V" % field


            plt.plot(fields, h100, color="r", alpha=1./100)
            plt.plot(fields, h111, color="b", alpha=1./100)

        h100_reg = np.zeros_like(h100)
        h111_reg = np.zeros_like(h100)
        h100_bruy = np.zeros_like(h100)
        h111_bruy = np.zeros_like(h100)
        for (idx,field) in enumerate(fields):
            h100_reg[idx] = find_drift_velocity_bruyneel(field, 66333., 0.744, 181.)
            h111_reg[idx] = find_drift_velocity_bruyneel(field, 107270., 0.580, 100.)
            h100_bruy[idx] = find_drift_velocity_bruyneel(field, 61824., 0.942, 185.)
            h111_bruy[idx] = find_drift_velocity_bruyneel(field, 61215., 0.662, 182.)

        plt.plot(fields, h100_reg, color="g")
        plt.plot(fields, h111_reg, color="g", ls="--")
        plt.plot(fields, h100_bruy, color="purple")
        plt.plot(fields, h111_bruy, color="purple", ls="--")

        # plt.axvline(x=250, color="black", ls=":")
        plt.axvline(x=500, color="black", ls=":")
        plt.xscale('log')
        # plt.yscale('log')
        # plt.xlim(.45, 1E5)
        # plt.ylim(1E4, 1E8)


    plt.figure(6)
    for wf_idx in range(numWaveforms):
        plt.hist(phi_hist_arr[wf_idx,:], color=colors[wf_idx])
    plt.axvline(x=0, color="r", ls=":")
    plt.axvline(x=np.pi/4, color="r", ls=":")
    plt.xlabel("Azimuthal Angle (0 to pi/4)")
    plt.show()

def find_drift_velocity_bruyneel(E, mu_0, beta, E_0, mu_n = 0):
    # mu_0 = 61824
    # beta = 0.942
    # E_0 = 185.
    v = (mu_0 * E) / np.power(1+(E/E_0)**beta, 1./beta) - mu_n*E

    return v #* 10 * 1E-9

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
