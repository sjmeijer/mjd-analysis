#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil

import matplotlib.pyplot as plt
# plt.style.use('presentation')
from matplotlib import gridspec
from matplotlib.colors import LogNorm

import pandas as pd
import numpy as np

from pysiggen import Detector

from FitConfiguration import FitConfiguration
from Model import Model

doContourHist = 0

doPositionHist = 0
doRSquaredPlot = 0

doVeloPlot =0

doWaveformPlot =1
doHists = 0
plotNum = 100 #for plotting during the Run

# doWaveformPlot =1
# doHists = 0
# plotNum = 10 #for plotting during the Run

colors = ["red" ,"blue", "green", "purple", "orange", "cyan", "magenta", "goldenrod", "brown", "deeppink", "lightsteelblue", "maroon", "violet", "lawngreen", "grey", "chocolate" ]

def plot(sample_file_name, directory):

    conf2 = FitConfiguration(directory=directory, loadSavedConfig=True)

    # params = FitConfiguration(directory=directory, loadSavedConfig=True)
    model = Model(conf2)

    #Load the data from csv (using pandas so its fast)
    sample_file_name = os.path.join(directory, sample_file_name)
    data = pd.read_csv(sample_file_name, delim_whitespace=True, header=None)
    num_samples = len(data.index)
    print( "found %d samples... " % num_samples,)

    if num_samples > plotNum: num_samples = plotNum
    print( "plotting %d samples" % num_samples)

    plot_data = data.iloc[-(num_samples+1):]

    #chunk data out into sub arrays
    tf_first_idx, velo_first_idx, grad_idx, trap_idx = model.get_indices()
    velo = plot_data.as_matrix(range(velo_first_idx,velo_first_idx+6))
    tf = plot_data.as_matrix(range(tf_first_idx,tf_first_idx+7))
    det_params =  plot_data.as_matrix(range(grad_idx,grad_idx+3))

    rad_idx, phi_idx, th_idx, scale_idx, maxt_idx, smoothidx = range(model.num_det_params, 6*model.num_waveforms+model.num_det_params, model.num_waveforms)

    rad_arr =  plot_data.as_matrix(range(rad_idx,rad_idx+ model.num_waveforms))
    theta_arr =  plot_data.as_matrix(range(th_idx,th_idx+ model.num_waveforms))

    phi_arr =  plot_data.as_matrix(range(phi_idx,phi_idx+ model.num_waveforms))
    scale_arr =  plot_data.as_matrix(range(scale_idx,scale_idx+ model.num_waveforms))
    maxt_arr =  plot_data.as_matrix(range(maxt_idx,maxt_idx+ model.num_waveforms))
    smooth_arr =  plot_data.as_matrix(range(smoothidx,smoothidx+ model.num_waveforms))

    if doWaveformPlot:
        plot_waveforms(plot_data, model)

    if doHists:
        plot_det_hist(velo, tf, det_params)
        plot_wf_hist(phi_arr, scale_arr, maxt_arr, smooth_arr)

    if doPositionHist:
        plot_position_hist(rad_arr, theta_arr, model.detector.detector_radius, model.detector.detector_length)

    if doVeloPlot:
        plot_velo_curves(velo, model)

    plt.show()
    exit()


def plot_waveforms(data, model):
    plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.set_xlabel("Digitizer Time [ns]")
    ax0.set_ylabel("Voltage [Arb.]")
    ax1.set_ylabel("Residual")

    num_det_params = model.num_det_params
    wf_params = np.empty(num_det_params+6)
    residWarnings = np.zeros(model.num_waveforms)

    for wf in model.wfs:
      dataLen = wf.wfLength
      t_data = np.arange(dataLen) * 10
      ax0.plot(t_data, wf.windowedWf, color="black")

    for (idx) in range(len(data.index)):
        params = data.iloc[idx].as_matrix()

        wfs_param_arr = params[num_det_params:].reshape((6, model.num_waveforms))
        wf_params[:num_det_params] = params[:num_det_params]

        for (wf_idx,wf) in enumerate(model.wfs):
            wf_params[num_det_params:] = wfs_param_arr[:,wf_idx]

            fit_wf = model.make_waveform(wf.wfLength,wf_params)
            if fit_wf is None:
                continue

            t_data = np.arange(wf.wfLength) * 10
            color_idx = wf_idx % len(colors)
            ax0.plot(t_data,fit_wf, color=colors[color_idx], alpha=0.1)

            resid = fit_wf -  wf.windowedWf
            ax1.plot(t_data, resid, color=colors[color_idx],alpha=0.1)

            if np.amax(np.abs(resid)) > 20 and residWarnings[wf_idx] == 0:
                print ("wf %d has a big residual!" % wf_idx)
                residWarnings[wf_idx] = 1

    ax0.set_ylim(-20, wf.wfMax*1.1)
    ax1.set_ylim(-20, 20)

def plot_wf_hist(phi, scale, maxt, smooth):
    fig = plt.figure(figsize=(15,10))

    num_bins = 100
    data = [phi, scale, maxt, smooth]
    labels = ["phi", "scale", "maxt", "smooth"]
    for i in range(4):
        axis = fig.add_subplot(4,1,i+1)
        axis.set_ylabel(labels[i])
        for wf_idx in range(phi.shape[1]):
            color_idx = wf_idx % len(colors)
            [n, b, p] = axis.hist(data[i][:,wf_idx], bins=num_bins, color=colors[color_idx])



    # for wf_idx in range(numWaveforms):
    #     color_idx = wf_idx % len(colors)
    #     plt.hist(phi_hist_arr[wf_idx,:], color=colors[color_idx])
    # plt.axvline(x=0, color="r", ls=":")
    # plt.axvline(x=np.pi/4, color="r", ls=":")
    # plt.xlabel("Azimuthal Angle (0 to pi/4)")

def plot_det_hist(velo, tf, det_params):

    vFig = plt.figure(figsize=(20,10))

    num_bins = 100

    #velo params
    vLabels = ["h_100_va", "h_111_va", "h_100_vmax", "h_111_vmax", 'h_100_beta', 'h_111_beta']
    for i in range(6):
        idx = (i+1)*3
        # if i < 5:
        axis = vFig.add_subplot(6,3,idx-1)
        axis.set_ylabel(vLabels[i])
        [n, b, p] = axis.hist(velo[:,i], bins=num_bins)
        max_idx = np.argmax(n)
        print ("%s mode: %f" % (vLabels[i], b[max_idx]))

    #tf params
    tfLabels = ['tf_phi', 'tf_omega', 'd', 'rc1', 'rc2', 'rcfrac']
    for i in range(6):
        idx = (i+1)*3
        axis = vFig.add_subplot(6,3,idx-2)
        axis.set_ylabel(tfLabels[i])
        [n, b, p] = axis.hist(tf[:,i], bins=num_bins)
        max_idx = np.argmax(n)
        print ("%s mode: %f" % (tfLabels[i], b[max_idx]))

    #"other"
    for i in range(4):
        idx = (i+1)*3
        if i==0:
            axis = vFig.add_subplot(6,3,idx)
            axis.set_ylabel("imp grad")
            [n, b, p] = axis.hist(det_params[:,i], bins=num_bins)
            max_idx = np.argmax(n)
            print ("%s mode: %f" % ("imp_grad", b[max_idx]))
        if i==1:
            axis = vFig.add_subplot(6,3,idx)
            axis.set_ylabel("imp average")
            [n, b, p] = axis.hist(det_params[:,i], bins=num_bins)
            max_idx = np.argmax(n)
            print ("%s mode: %f" % ("avg_imp", b[max_idx]))
        if i==2:
            axis = vFig.add_subplot(6,3,idx)
            axis.set_ylabel("trapping_rc")
            [n, b, p] = axis.hist(det_params[:,i], bins=num_bins)
            max_idx = np.argmax(n)
            print ("%s mode: %f" % ("trapping_rc", b[max_idx]))
        if i==3:
            axis = vFig.add_subplot(6,3,idx)
            axis.set_ylabel("alias_rc")
            [n, b, p] = axis.hist(tf[:,6], bins=num_bins)
            max_idx = np.argmax(n)
            print( "%s mode: %f" % ("alias_rc", b[max_idx]))

def plot_position_hist(rad_arr, theta_arr,  det_radius, det_length, doContourHist=False,):
    positionFig = plt.figure(figsize=(10,10))
    plt.clf()
    colorbars = ["Reds","Blues", "Greens", "Purples", "Oranges", "Greys", "YlOrBr", "PuRd"]

    if not doContourHist:
        for wf_idx in range(rad_arr.shape[1]):
            colorbar_idx = wf_idx % len(colorbars)
            xedges = np.linspace(0, np.around(det_radius,1), np.around(det_radius,1)*10+1)
            yedges = np.linspace(0, np.around(det_length,1), np.around(det_length,1)*10+1)

            r_arr = rad_arr[:,wf_idx] * np.cos(theta_arr[:,wf_idx])
            z_arr = rad_arr[:,wf_idx] * np.sin(theta_arr[:,wf_idx])

            plt.hist2d(r_arr, z_arr,  bins=[ xedges,yedges  ],  cmap=plt.get_cmap(colorbars[colorbar_idx]), cmin=0.01)

        plt.xlabel("r from Point Contact (mm)")
        plt.ylabel("z from Point Contact (mm)")
        plt.xlim(0, det_radius)
        plt.ylim(0, det_length)
        plt.gca().set_aspect('equal', adjustable='box')
    # else:
    #     for wf_idx in range(numWaveforms):
    #         xedges = np.linspace(0, np.around(det.detector_radius,1), np.around(det.detector_radius,1)*10+1)
    #         yedges = np.linspace(0, np.around(det.detector_length,1), np.around(det.detector_length,1)*10+1)
    #         z, xe, ye = np.histogram2d(r_arr[wf_idx,:], z_arr[wf_idx,:],  bins=[ xedges,yedges  ])
    #         z /= z.sum()
    #         n=100
    #         t = np.linspace(0, z.max(), n)
    #         integral = ((z >= t[:, None, None]) * z).sum(axis=(1,2))
    #         from scipy import interpolate
    #         f = interpolate.interp1d(integral, t)
    #         t_contours = f(np.array([0.9, 0.8]))
    #
    #         cs = plt.contourf(z.T, t_contours, extent=[0,det.detector_radius,0,det.detector_length],  alpha=1, colors = (colors[wf_idx]), extend='max')
    #         # cs.cmap.set_over(colors[wf_idx])
    #
    #     plt.xlabel("r from Point Contact (mm)")
    #     plt.ylabel("z from Point Contact (mm)")
    #     # plt.axis('equal')
    #     plt.xlim(0, det.detector_radius)
    #     plt.ylim(0, det.detector_length)
    #     plt.gca().set_aspect('equal', adjustable='box')
    #     plt.tight_layout()
    #
    #     plt.savefig("credible_intervals.pdf")


def plot_velo_curves(velo, model):
    velo_plot = plt.figure()
    fields_log = np.linspace(np.log(100), np.log(10000), 100)
    fields = np.exp(fields_log)

    for  idx in range(velo.shape[0]):
        h_100_va, h_111_va, h_100_vmax, h_111_vmax, h_100_beta, h_111_beta, = velo[idx,:]

        h100 = np.empty_like(fields)
        h111 = np.empty_like(fields)

        h_100_mu0, h_100_beta, h_100_e0 = model.get_velo_params(h_100_va, h_100_vmax, h_100_beta)
        h_111_mu0, h_111_beta, h_111_e0 = model.get_velo_params(h_111_va, h_111_vmax, h_111_beta)

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


def find_drift_velocity_bruyneel(E, mu_0, beta, E_0, mu_n = 0):
    v = (mu_0 * E) / np.power(1+(E/E_0)**beta, 1./beta) - mu_n*E

    return v #* 10 * 1E-9


if __name__=="__main__":

    if len(sys.argv) >= 2:
        directory = sys.argv[1]
    else:
        directory = ""
    plot("sample.txt", directory)
