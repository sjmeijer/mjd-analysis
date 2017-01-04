#!/usr/bin/env python

import sys
import numpy as np
import dnest4
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pysiggen import Detector

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

    for (idx,params) in enumerate(data):
        rad, phi, theta, scale, t0, smooth = params[:6]
        r_arr[idx] = rad * np.cos(theta)
        z_arr[idx] = rad * np.sin(theta)

    positionFig = plt.figure(5)
    plt.clf()
    xedges = np.linspace(0, np.around(det.detector_radius,1), np.around(det.detector_radius,1)*10+1)
    yedges = np.linspace(0, np.around(det.detector_length,1), np.around(det.detector_length,1)*10+1)
    plt.hist2d(r_arr, z_arr,  norm=LogNorm(), bins=[ xedges,yedges  ])
    plt.colorbar()
    plt.xlabel("r from Point Contact (mm)")
    plt.ylabel("z from Point Contact (mm)")

    plt.show()


if __name__=="__main__":
    if len(sys.argv) < 2:
        postprocess(sys.argv[1:])
    elif sys.argv[1] == "plot":
        plot()
