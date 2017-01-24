#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os, time
import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import helpers

from progressbar import ProgressBar, Percentage, Bar, ETA
from multiprocessing import Pool
from timeit import default_timer as timer
from scipy import signal

def main(argv):

    # numThreads = 8

    wfFileName = "fep_event_set_runs11510-11539_channel626.npz"
    #wfFileName = "P42574A_24_spread.npz"
    if os.path.isfile(wfFileName):
        data = np.load(wfFileName)
        wfs = data['wfs']
        numWaveforms = wfs.size
    else:
        print "No saved waveforms available."
        exit(0)

    # initializeDetector(det, )

    plt.ion()
    # fig1 = plt.figure(0, figsize=(20,10))
    # plt.clf()
    # gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    # ax0 = plt.subplot(gs[0])
    # ax1 = plt.subplot(gs[1], sharex=ax0)
    # ax1.set_xlabel("Digitizer Time [ns]")
    # ax0.set_ylabel("Voltage [Arb.]")
    # ax1.set_ylabel("Residual")

    global waveform

    rc1_arr = np.empty(numWaveforms)
    rc2_arr = np.empty(numWaveforms)
    rcfrac_arr = np.empty(numWaveforms)
    baseline_arr = np.empty(numWaveforms)

    for (idx,wf) in enumerate(wfs):
        # ax0.cla()
        # ax1.cla()
        waveform = wf
        # ax0.plot(wf.waveformData + wf.baselineMean)

        wf_max_idx = np.argmax(wf.waveformData)
        wf_data = wf.waveformData[wf_max_idx+20:]

        startGuess = [72, 2, 0.992]

        result = op.minimize(nll, startGuess,   method="Nelder-Mead")
        rc1, rc2, rcfrac = result['x']
        rc1_arr[idx] = rc1
        rc2_arr[idx] = rc2
        rcfrac_arr[idx] = rcfrac
        baseline_arr[idx] = wf.baselineMean

        square_data = make_rc_decay(rc1, rc2, rcfrac, wf_data)

        if rc1 > 74.5:
            print "waveform %d" % idx
            print "  rc1: %f" % rc1
            print "  rc2: %f" % rc2
            print "  rcfrac: %f" % rcfrac
            print "  baseline mean: %f" % wf.baselineMean
        #     ax0.plot(wf.waveformData, color="r")
        # else:
        #     ax0.plot(wf.waveformData, color="g")

    plt.xlim(950,1250)

    print "rc1: %f pm %f" % (np.mean(rc1_arr), np.std(rc1_arr))
    print "rc2: %f pm %f" % (np.mean(rc2_arr), np.std(rc2_arr))
    print "rcfrac: %f pm %f" % (np.mean(rcfrac_arr), np.std(rcfrac_arr))

    fig2 = plt.figure(2)
    ax0 = fig2.add_subplot(2,2,1)
    ax0.hist(rc1_arr, bins=50)
    ax0.set_xlabel("long rc constant (us)")

    ax1 = fig2.add_subplot(2,2,2)
    ax1.hist(rc2_arr, bins=50)
    ax1.set_xlabel("short rc constant (us)")

    ax2 = fig2.add_subplot(2,2,3)
    ax2.hist(rcfrac_arr, bins=50)
    ax2.set_xlabel("rc1 fraction")

    ax3 = fig2.add_subplot(2,2,4)
    ax3.hist(baseline_arr, bins=np.linspace(90,110, 21 ))
    ax3.set_xlabel("baseline mean (adc)")

    value = raw_input('  --> Press q to quit, any other key to continue\n')
    if value == 'q': exit(0)


def rc_to_tf(rc_in_us):
    rc= 1E-6 * (rc_in_us)
    return np.exp(-1./1E8/rc)

def nll(*args):
  return -WaveformLogLike(*args)

def WaveformLogLike(theta):
    rc1, rc2, rcfrac = theta

    model_err = waveform.baselineRMS
    wf_max_idx = np.argmax(waveform.waveformData)
    data = waveform.waveformData[wf_max_idx+20:]

    square_data = make_rc_decay(rc1, rc2, rcfrac, data)

    inv_sigma2 = 1.0/(model_err**2)
    ln_like = -0.5*(np.sum((data-square_data)**2*inv_sigma2 - np.log(inv_sigma2)))
    return ln_like

def make_rc_decay(RC1_in_us, RC2_in_us, rc1_frac, wf_data):
    rc1_for_tf = rc_to_tf(RC1_in_us)
    rc2_for_tf = rc_to_tf(RC2_in_us)

    square_data = np.ones_like(wf_data)
    rc2_num_term = rc1_for_tf*rc1_frac - rc1_for_tf - rc2_for_tf*rc1_frac
    square_data= signal.lfilter([1., -1], [1., -rc1_for_tf], square_data)
    square_data= signal.lfilter([1., rc2_num_term], [1., -rc2_for_tf], square_data)

    square_data *= np.amax(wf_data)

    return square_data

if __name__=="__main__":
    main(sys.argv[1:])
