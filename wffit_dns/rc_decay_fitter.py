#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os, time
import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt

import helpers

from scipy import signal

def fit_decay(wf):
    startGuess = [72, 2, 0.992]
    result = op.minimize(nll, startGuess, args=(wf,),  method="Nelder-Mead")
    return result

def rc_to_tf(rc_in_us):
    rc= 1E-6 * (rc_in_us)
    return np.exp(-1./1E8/rc)

def nll(*args):
  return -WaveformLogLike(*args)

def WaveformLogLike(theta, waveform):
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
