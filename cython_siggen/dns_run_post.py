#!/usr/bin/env python

import sys
import numpy as np
import dnest4
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pysiggen import Detector

tf_first_idx = 8
velo_first_idx = 14
trap_idx = 20
grad_idx = 21

def postprocess(argv):

  # Run the postprocessing
  dnest4.postprocess()

def plot():

    detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
    det =  Detector(detName,timeStep=1, numSteps=10)

    data = np.loadtxt("posterior_sample.txt")
    num_samples = len(data)
    print "found %d samples" % num_samples

    r_arr = np.empty(num_samples)
    z_arr = np.empty(num_samples)

    h_100_mu0 = np.empty(num_samples)
    h_100_beta = np.empty(num_samples)
    h_100_e0 = np.empty(num_samples)
    h_111_mu0 = np.empty(num_samples)
    h_111_beta = np.empty(num_samples)
    h_111_e0 = np.empty(num_samples)

    tf = np.empty((6, num_samples))

    for (idx,params) in enumerate(data):
        rad, phi, theta, scale, t0, smooth = params[:6]
        r_arr[idx] = rad * np.cos(theta)
        z_arr[idx] = rad * np.sin(theta)
        h_100_mu0[idx], h_100_beta[idx], h_100_e0[idx], h_111_mu0[idx], h_111_beta[idx], h_111_e0[idx] = params[velo_first_idx:velo_first_idx+6]
        tf[:,idx] = params[tf_first_idx:tf_first_idx+6]

    positionFig = plt.figure(0)
    plt.clf()
    xedges = np.linspace(0, np.around(det.detector_radius,1), np.around(det.detector_radius,1)*10+1)
    yedges = np.linspace(0, np.around(det.detector_length,1), np.around(det.detector_length,1)*10+1)
    plt.hist2d(r_arr, z_arr,  norm=LogNorm(), bins=[ xedges,yedges  ])
    plt.colorbar()
    plt.xlabel("r from Point Contact (mm)")
    plt.ylabel("z from Point Contact (mm)")

    plotnum = 600
    veloFig = plt.figure(1)
    tf0 = veloFig.add_subplot(plotnum+11)
    tf1 = veloFig.add_subplot(plotnum+12, )
    tf2 = veloFig.add_subplot(plotnum+13, )
    tf3 = veloFig.add_subplot(plotnum+14, )
    tf4 = veloFig.add_subplot(plotnum+15, )
    tf5 = veloFig.add_subplot(plotnum+16, )

    tf0.set_ylabel('h_100_mu0')
    tf1.set_ylabel('h_100_beta')
    tf2.set_ylabel('h_100_e0')
    tf3.set_ylabel('h_111_mu0')
    tf4.set_ylabel('h_111_beta')
    tf5.set_ylabel('h_111_e0')

    num_bins = 100
    [n, b, p] = tf0.hist(h_100_mu0, bins=num_bins)
    [n, b, p] = tf1.hist(h_100_beta, bins=num_bins)
    [n, b, p] = tf2.hist(h_100_e0, bins=num_bins)
    [n, b, p] = tf3.hist(h_111_mu0, bins=num_bins)
    [n, b, p] = tf4.hist(h_111_beta, bins=num_bins)
    [n, b, p] = tf5.hist(h_111_e0, bins=num_bins)

    plotnum = 600
    tfFig = plt.figure(2)
    tf0 = tfFig.add_subplot(plotnum+11)
    tf1 = tfFig.add_subplot(plotnum+12, )
    tf2 = tfFig.add_subplot(plotnum+13, )
    tf3 = tfFig.add_subplot(plotnum+14, )
    tf4 = tfFig.add_subplot(plotnum+15, )
    tf5 = tfFig.add_subplot(plotnum+16, )

    tf0.set_ylabel('b_ov_a')
    tf1.set_ylabel('c')
    tf2.set_ylabel('d')
    tf3.set_ylabel('rc1')
    tf4.set_ylabel('rc2')
    tf5.set_ylabel('rcfrac')

    num_bins = 100
    [n, b, p] = tf0.hist(tf[0,:], bins=num_bins)
    [n, b, p] = tf1.hist(tf[1,:], bins=num_bins)
    [n, b, p] = tf2.hist(tf[2,:], bins=num_bins)
    [n, b, p] = tf3.hist(tf[3,:], bins=num_bins)
    [n, b, p] = tf4.hist(tf[4,:], bins=num_bins)
    [n, b, p] = tf5.hist(tf[5,:], bins=num_bins)

    plt.show()


if __name__=="__main__":
    if len(sys.argv) < 2:
        postprocess(sys.argv[1:])
    elif sys.argv[1] == "plot":
        plot()
