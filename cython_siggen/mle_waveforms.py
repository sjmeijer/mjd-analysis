#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os, time
import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import helpers
from pysiggen import Detector
from probability_model_waveform import *

from progressbar import ProgressBar, Percentage, Bar, ETA
from multiprocessing import Pool
from timeit import default_timer as timer



#Prepare detector

timeStepSize = 1
fitSamples = 200
detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
detector =  Detector(detName, timeStep=timeStepSize, numSteps=fitSamples*10./timeStepSize, maxWfOutputLength=500)
detector.LoadFieldsGrad("fields_impgrad_0-0.06.npz",pcLen=1.6, pcRad=2.5)
#sets the impurity gradient.  Don't bother changing this
detector.SetFieldsGradIdx(0)

b_over_a = 0.085487
d = 0.814511
c = -0.807451
rc1 = 74.
rc2 = 2.08
rcfrac = 0.992

charge_trapping  = 212.540970

h_100_mu0 = 73048.920673
h_100_beta = 0.910594
h_100_e0 = 125.410298
h_111_mu0 = 76317.444771
h_111_beta = 0.617693
h_111_e0 = 168.208161

max_sample_idx = 100
fallPercentage = 0.99

detector.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)
detector.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
detector.trapping_rc = charge_trapping

rad_mult = 10.
phi_mult = 0.1
z_mult = 10.
scale_mult = 1000.
maxt_mult = 100.
smooth_mult = 10.

def main(argv):

    # numThreads = 8

    # wfFileName = "ms_event_set_runs11530-11560.npz"
    wfFileName = "P42574A_24_spread.npz"
    if os.path.isfile(wfFileName):
        data = np.load(wfFileName)
        wfs = data['wfs']
        numWaveforms = wfs.size
    else:
        print "No saved waveforms available."
        exit(0)

    # initializeDetector(det, )

    plt.ion()
    fig1 = plt.figure(0, figsize=(20,10))
    plt.clf()
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.set_xlabel("Digitizer Time [ns]")
    ax0.set_ylabel("Voltage [Arb.]")
    ax1.set_ylabel("Residual")



    # bar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(wfs)).start()
    global waveform

    start = timer()

    for (idx,wf) in enumerate(wfs):
        ax0.cla()
        ax1.cla()

        wf.WindowWaveformAroundMax(fallPercentage=fallPercentage, rmsMult=2, earlySamples=max_sample_idx)
        waveform = wf

        dataLen = wf.wfLength
        t_data = np.arange(dataLen) * 10
        ax0.plot(t_data, wf.windowedWf, color="black")

        # rad = np.sqrt(15**2+15**2)
        # theta = np.pi/4
        r,phi, z, scale, maxt, smooth  = 25,np.pi/8, 25, wf.wfMax, max_sample_idx, 10
        r /= rad_mult
        phi /= phi_mult
        z /= z_mult
        scale /= scale_mult
        maxt /= maxt_mult
        smooth /= smooth_mult

        minresult = None
        minlike = np.inf
        for r in np.linspace(10, np.floor(detector.detector_radius)-5, 5):
            for z in np.linspace(10, np.floor(detector.detector_length)-5, 5):
                r /= rad_mult
                z /= z_mult
                startGuess = [r, phi, z, scale, maxt, smooth ]
                result = op.minimize(nll, startGuess,   method="Powell")
                if result['fun'] < minlike:
                  minlike = result['fun']
                  minresult = result
        # bounds =[(0, detector.detector_radius/rad_mult), (0, np.pi/4), (0, detector.detector_length/rad_mult),
        #          (scale - 0.02*scale, scale+0.02*scale), (maxt-2./maxt_mult, maxt+2./maxt_mult), (0,25./scale_mult)]
        # result = op.differential_evolution(nll, bounds, polish=False)
        #

        # result = op.minimize(nll, startGuess,   method="Nelder-Mead", options={"maxfev": 10E4})

        r, phi, z, scale, maxt, smooth, = minresult["x"]
        # r = rad * np.cos(theta)
        # z = rad * np.sin(theta)

        r *= rad_mult
        phi *= phi_mult
        z *= z_mult
        scale *= scale_mult
        maxt *= maxt_mult
        smooth *= smooth_mult

        print r, phi, z, scale, maxt, smooth
        mle_wf = detector.MakeSimWaveform(r, phi, z, scale, maxt, dataLen, h_smoothing=smooth, alignPoint="max")

        ax0.plot(t_data, mle_wf, color="g")
        ax1.plot(t_data, mle_wf - wf.windowedWf, color="g")

        # value = raw_input('  --> Press q to quit, any other key to continue\n')
        # if value == 'q': exit(0)

    end = timer()
    "print total time: %f" % str(end-start)
        # bar.finish()
        # wfFileName += "_mlefit.npz"
        # np.savez(wfFileName, wfs = wfs )

def nll(*args):
  return -WaveformLogLike(*args)

def WaveformLogLike(theta):
    rad, phi, theta, scale, maxt, smooth = theta

    r,z = rad,theta
    # r = rad * np.cos(theta)
    # z = rad * np.sin(theta)

    r *= rad_mult
    phi *= phi_mult
    z *= z_mult
    scale *= scale_mult
    maxt *= maxt_mult
    smooth *= smooth_mult

    if scale < 0:
      return -np.inf
    if smooth < 0:
       return -np.inf
    if not detector.IsInDetector(r, phi, z):
      return -np.inf

    data = waveform.windowedWf
    model_err = waveform.baselineRMS
    data_len = len(data)

    model = detector.MakeSimWaveform(r, phi, z, scale, maxt, data_len, h_smoothing=smooth, alignPoint="max")
    if model is None:
        return -np.inf

    inv_sigma2 = 1.0/(model_err**2)
    ln_like = -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))
    return ln_like

if __name__=="__main__":
    main(sys.argv[1:])
