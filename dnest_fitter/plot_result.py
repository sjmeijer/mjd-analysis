#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of DNest4 in Python using the "StraightLine" example
"""

# import dnest4

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
from dns_model import global_get_indices

import helpers
from pysiggen import Detector

doInitPlot =0
doContourHist = 0
doPositionHist = 0

doWaveformPlot =0
doHists = 1
plotNum = 5000 #for plotting during the Run
# doWaveformPlot =1
# doHists = 0
# plotNum = 50 #for plotting during the Run

colors = ["red" ,"blue", "green", "purple", "orange", "cyan", "magenta", "goldenrod", "brown", "deeppink", "lightsteelblue", "maroon", "violet", "lawngreen", "grey", "chocolate" ]

def plot(sample_file_name, directory):
    max_sample_idx = 200
    fallPercentage = 0.97

    if os.path.isfile(directory + "fit_params.npz"):
        data = np.load(directory + "fit_params.npz", encoding="latin1")

        max_sample_idx = data['max_sample_idx']
        fallPercentage=data['fallPercentage']
        wf_idxs=data['wf_idxs']
        wf_file_name=data['wf_file_name']
        field_file_name=data['field_file_name']
        doMaxInterp=data['doMaxInterp']
        # det_name = data['det_name']

        wf_file_name = "%s" % wf_file_name
        field_file_name = "%s" % field_file_name

    else:
      print( "Saved fit param %s not available" % (directory + "fit_params.npz"))
      exit(0)

    if os.path.isfile(wf_file_name):
        data = np.load(wf_file_name)
        wfs = data['wfs']
        wfs = wfs[wf_idxs]
        numWaveforms = wfs.size
    else:
      print( "Saved waveform file %s not available" % wfFileName)
      exit(0)


    wfLengths = np.empty(numWaveforms)
    wfMaxes = np.empty(numWaveforms)

    if doInitPlot: plt.figure(500)
    baselineLengths = np.empty(numWaveforms)
    for (wf_idx,wf) in enumerate(wfs):
      wf.WindowWaveformAroundMax(fallPercentage=fallPercentage, rmsMult=2, earlySamples=max_sample_idx)
      baselineLengths[wf_idx] = wf.t0Guess

      print( "wf %d length %d (entry %d from run %d)" % (wf_idx, wf.wfLength, wf.entry_number, wf.runNumber))
      wfLengths[wf_idx] = wf.wfLength
      wfMaxes[wf_idx] = np.argmax(wf.windowedWf)

      if doInitPlot:
          if len(colors) < numWaveforms:
              color = "red"
          else: color = colors[wf_idx]
          plt.plot(wf.windowedWf, color=color)

    if doInitPlot:
        plt.show()
        exit()

    siggen_wf_length = (max_sample_idx - np.amin(baselineLengths) + 10)*10
    output_wf_length = np.amax(wfLengths) + 1

    #Create a detector model
    timeStepSize = 1 #ns
    detName = "conf/P42574A_bull.conf"
    det =  Detector(detName, timeStep=timeStepSize, numSteps=siggen_wf_length, maxWfOutputLength =output_wf_length, t0_padding=100 )
    det.LoadFieldsGrad(field_file_name)


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
    data_len = num_samples

    # data = pd.read_csv("sample_plot.txt", delim_whitespace=True, header=None)
    # num_samples = len(data.index)
    print( "found %d samples" % num_samples)

    if sample_file_name== (directory+"sample_plot.txt"):
        if num_samples > plotNum: num_samples = plotNum

    if doWaveformPlot:
        if num_samples > plotNum: num_samples = plotNum
    print( "plotting %d samples" % num_samples)

    r_arr = np.empty((numWaveforms, num_samples))
    z_arr = np.empty((numWaveforms, num_samples))
    tf = np.empty((6, num_samples))
    velo = np.empty((6, num_samples))
    wf_params = np.empty((numWaveforms, 6, num_samples))
    det_params = np.empty((4, num_samples))

    tf_first_idx, velo_first_idx, grad_idx, trap_idx = global_get_indices()


    for (idx,params) in enumerate(data[-num_samples:]):
        # params = data.iloc[-(idx+1)]
        # print params

        tf_phi, tf_omega, d, rc1, rc2, rcfrac, aliasrc = params[tf_first_idx:tf_first_idx+7]
        #tf_d = tf_c * tf_dc
        c = -d * np.cos(tf_omega)
        b_ov_a = c - np.tan(tf_phi) * np.sqrt(d**2-c**2)
        a = 1./(1+b_ov_a)
        tf_b = a * b_ov_a
        tf_c = c
        tf_d = d

        h_111_va, h_111_vmax, h_100_multa, h_100_multmax, h_100_beta, h_111_beta, = params[velo_first_idx:velo_first_idx+6]
        charge_trapping = params[trap_idx]
        grad = (params[grad_idx])
        imp_avg = (params[grad_idx+1])

        # rc1 = -1./np.log(e_rc1)
        # rc2 = -1./np.log(e_rc2)
        # charge_trapping = -1./np.log(e_charge_trapping)

        h_100_va = h_100_multa * h_111_va
        h_100_vmax = h_100_multmax * h_111_vmax

        h_100_mu0, h_100_beta, h_100_e0 = get_velo_params(h_100_va, h_100_vmax, h_100_beta)
        h_111_mu0, h_111_beta, h_111_e0 = get_velo_params(h_111_va, h_111_vmax, h_111_beta)

        tf[:,idx] = tf_phi, tf_omega, tf_d, rc1, rc2, rcfrac
        velo[:,idx] = params[velo_first_idx:velo_first_idx+6]
        det_params[:,idx] = grad, imp_avg, charge_trapping, aliasrc

        det.SetTransferFunction(tf_b, tf_c, tf_d, rc1, rc2, rcfrac)
        det.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
        det.trapping_rc = charge_trapping
        det.SetGrads(grad, imp_avg)

        det.SetAntialiasingRC(aliasrc)

        rad_arr, phi_arr, theta_arr, scale_arr, t0_arr, smooth_arr,= params[trap_idx+1:].reshape((6, numWaveforms))


        for (wf_idx,wf) in enumerate(wfs):
          rad, phi, theta = rad_arr[wf_idx], phi_arr[wf_idx], theta_arr[wf_idx]
          scale, t0, smooth =  scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx]
        #   m, b = m_arr[wf_idx], b_arr[wf_idx]
          wf_params[wf_idx, :, idx] = rad, phi, theta, scale, t0, smooth,

          r = rad * np.cos(theta)
          z = rad * np.sin(theta)
          r_arr[wf_idx, idx], z_arr[wf_idx, idx] = r,z

          if doWaveformPlot:
            color_idx = wf_idx % len(colors)
            ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0,  np.int(output_wf_length), h_smoothing = smooth, alignPoint="max", doMaxInterp=True)
            if ml_wf is None:
                continue

            dataLen = wf.wfLength
            t_data = np.arange(dataLen) * 10
            ax0.plot(t_data, ml_wf[:dataLen], color=colors[color_idx], alpha=0.1)
            ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color=colors[color_idx],alpha=0.1)

    ax0.set_ylim(-20, wf.wfMax*1.1)
    ax1.set_ylim(-20, 20)

    if not doHists:
        plt.tight_layout()
        plt.savefig("waveforms.png")
        plt.show()
        exit()

    vFig = plt.figure(2, figsize=(20,10))
    tfLabels = ['tf_phi', 'tf_omega', 'tf_d', 'rc1', 'rc2', 'rcfrac']
    vLabels = ['h_111_va', 'h_111_vmax', 'h_100_multa', 'h_100_multmax', 'h_100_beta', 'h_111_beta']
    vmodes, tfmodes = np.empty(6), np.empty(6)
    num_bins = 100
    for i in range(6):
        idx = (i+1)*3
        axis = vFig.add_subplot(6,3,idx-1)
        axis.set_ylabel(vLabels[i])
        [n, b, p] = axis.hist(velo[i,:], bins=num_bins)
        max_idx = np.argmax(n)
        print ("%s mode: %f" % (vLabels[i], b[max_idx]))

        axis = vFig.add_subplot(6,3,idx-2)
        axis.set_ylabel(tfLabels[i])
        [n, b, p] = axis.hist(tf[i,:], bins=num_bins)
        max_idx = np.argmax(n)
        print( "%s mode: %f" % (tfLabels[i], b[max_idx]))

        if i==0:
            axis = vFig.add_subplot(6,3,idx)
            axis.set_ylabel("imp grad")
            [n, b, p] = axis.hist(det_params[i,:], bins=num_bins)
            max_idx = np.argmax(n)
            print ("%s mode: %f" % ("imp_grad", b[max_idx]))
        if i==1:
            axis = vFig.add_subplot(6,3,idx)
            axis.set_ylabel("imp average")
            [n, b, p] = axis.hist(det_params[i,:], bins=num_bins)
            max_idx = np.argmax(n)
            print ("%s mode: %f" % ("imp_avg", b[max_idx]))
        if i==2:
            axis = vFig.add_subplot(6,3,idx)
            axis.set_ylabel("trapping_rc")
            [n, b, p] = axis.hist(det_params[i,:], bins=num_bins)
            max_idx = np.argmax(n)
            print ("%s mode: %f" % ("trapping_rc", b[max_idx]))
        if i==3:
            axis = vFig.add_subplot(6,3,idx)
            axis.set_ylabel("alias_rc")
            [n, b, p] = axis.hist(det_params[i,:], bins=num_bins)
            max_idx = np.argmax(n)
            print ("%s mode: %f" % ("alias_rc", b[max_idx]))





    if doPositionHist:
        positionFig = plt.figure(3, figsize=(10,10))
        plt.clf()
        colorbars = ["Reds","Blues", "Greens", "Purples", "Oranges", "Greys", "YlOrBr", "PuRd"]

        if not doContourHist:
            for wf_idx in range(numWaveforms):
                color_idx = wf_idx % len(colorbars)

                xedges = np.linspace(0, np.around(det.detector_radius,1), np.around(det.detector_radius,1)*10+1)
                yedges = np.linspace(0, np.around(det.detector_length,1), np.around(det.detector_length,1)*10+1)
                plt.hist2d(r_arr[wf_idx,:], z_arr[wf_idx,:],  bins=[ xedges,yedges  ],  cmap=plt.get_cmap(colorbars[color_idx])) #cmin=0.1
                rad_mean = np.mean(wf_params[wf_idx, 0,:])
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



    if False:
        velo_plot = plt.figure(5)
        fields_log = np.linspace(np.log(100), np.log(10000), 100)
        fields = np.exp(fields_log)

        for  idx in range(num_samples):
            h_111_va, h_111_vmax, h_100_multa, h_100_multmax, h_100_beta, h_111_beta, = velo[:,idx]
            # print "v params: ",
            # print mu_0, h_100_lnbeta, h_111_lnbeta, h_111_emu, h_100_mult

            h_100_va = h_100_multa * h_111_va
            h_100_vmax = h_100_multmax * h_111_vmax

            h100 = np.empty_like(fields)
            h111 = np.empty_like(fields)

            h_100_mu0, h_100_beta, h_100_e0 = get_velo_params(h_100_va, h_100_vmax, h_100_beta)
            h_111_mu0, h_111_beta, h_111_e0 = get_velo_params(h_111_va, h_111_vmax, h_111_beta)


            for (idx,field) in enumerate(fields):
                h100[idx] = find_drift_velocity_bruyneel(field, h_100_mu0, h_100_beta,h_100_e0)
                h111[idx] = find_drift_velocity_bruyneel(field, h_111_mu0, h_111_beta,h_111_e0)

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
        plt.plot(fields, h100_bruy, color="orange")
        plt.plot(fields, h111_bruy, color="orange", ls="--")

        # plt.axvline(x=250, color="black", ls=":")
        plt.axvline(x=500, color="black", ls=":")
        plt.xscale('log')
        # plt.yscale('log')
        # plt.xlim(.45, 1E5)
        # plt.ylim(1E4, 1E8)



    plt.show()

def get_velo_params( v_a, v_max, beta):
    E_a = 500
    E_0 = np.power( (v_max*E_a/v_a)**beta - E_a**beta , 1./beta)
    mu_0 = v_max / E_0

    return (mu_0,  beta, E_0)


def find_drift_velocity_bruyneel(E, mu_0, beta, E_0, mu_n = 0):
    # mu_0 = 61824
    # beta = 0.942
    # E_0 = 185.
    v = (mu_0 * E) / np.power(1+(E/E_0)**beta, 1./beta) - mu_n*E

    return v #* 10 * 1E-9

if __name__=="__main__":
    if len(sys.argv) >= 2:
        directory = sys.argv[1]
    else:
        directory = ""
    plot("sample.txt", directory)
