#!/usr/local/bin/python

import numpy as np
from pymc3 import *#DiscreteUniform, Exponential, Poisson, Uniform, Normal, find_MAP, HalfNormal, switch, Model, Deterministic, exp, Metropolis, traceplot, sample, summary
import theano.tensor as T
from theano.compile.ops import as_op
from scipy import signal
"""
    Models for ppc response
    """

import matplotlib.pyplot as plt


def CreateFullDetectorModelWithTransferFunction(detectorList, data, t0_guess, energy_guess):
  detector = detectorList[0,0]
  with Model() as signal_model:
  
    radEst = Uniform('radEst', lower=0,   upper=np.floor(detector.radius))
    zEst = Uniform('zEst', lower=5,   upper=np.floor(detector.length))
    phiEst = Uniform('phiEst', lower=0,   upper=np.pi/4)
    
    tempEst = Uniform('temp', lower=40,   upper=120, testval=80)

    t0 = DiscreteUniform('switchpoint', lower = t0_guess-10, upper=t0_guess+10, testval=t0_guess)
    wfScale = Normal('wfScale', mu=energy_guess, sd=.01*energy_guess)

    prior_num = [3.64e+09, 1.88e+17, 6.05e+15]
    prior_den = [1, 4.03e+07, 5.14e+14, 7.15e+18]

    num_1 = Normal('num_1', mu=prior_num[0], sd=.3*prior_num[0])
    num_2 = Normal('num_2', mu=prior_num[1], sd=.3*prior_num[1])
    num_3 = Normal('num_3', mu=prior_num[2], sd=.3*prior_num[2])
    
    den_1 = Normal('den_1', mu=prior_den[1], sd=.3*prior_den[1])
    den_2 = Normal('den_2', mu=prior_den[2], sd=.3*prior_den[2])
    den_3 = Normal('den_3', mu=prior_den[3], sd=.3*prior_den[3])
    
    gradIdx = DiscreteUniform('gradIdx', lower = 0, upper= detectorList.shape[0]-1, testval=detectorList.shape[0]/2)
    pcRadIdx = DiscreteUniform('pcRadIdx', lower = 0, upper= detectorList.shape[1]-1, testval=detectorList.shape[1]/2)

 
    @as_op(itypes=[T.lscalar, T.dscalar, T.dscalar, T.dscalar,  T.dscalar, T.dscalar, T.dscalar,  T.dscalar, T.dscalar, T.dscalar,  T.dscalar, T.dscalar, T.lscalar, T.lscalar], otypes=[T.dvector])
#    @as_op(itypes=[T.lscalar, T.dscalar, T.dscalar, T.dscalar,  T.dscalar, T.dscalar], otypes=[T.dvector])
    def siggen_model(s, rad, phi, z, e, temp, num_1, num_2, num_3, den_1, den_2, den_3, gradIdx, pcRadIdx):
      out = np.zeros_like(data)
      if (gradIdx > detectorList.shape[0]-1) or (pcRadIdx > detectorList.shape[1]-1) :
        return np.ones_like(data)*-1.
      detector = detectorList[gradIdx, pcRadIdx]
      
      detector.SetTemperature(temp)
      siggen_wf= detector.GetSiggenWaveform(rad, phi, z, energy=2600)

      if siggen_wf is None:
        return np.ones_like(data)*-1.
      if np.amax(siggen_wf) == 0:
        print "wtf is even happening here?"
        return np.ones_like(data)*-1.
      siggen_wf = np.pad(siggen_wf, (detector.zeroPadding,0), 'constant', constant_values=(0, 0))

      num = [num_1, num_2, num_3]
      den = [1,   den_1, den_2, den_3]
#      num = [-1.089e10,  5.863e17,  6.087e15]
#      den = [1,  3.009e07, 3.743e14,5.21e18]
      system = signal.lti(num, den)
      t = np.arange(0, len(siggen_wf)*10E-9, 10E-9)
      tout, siggen_wf, x = signal.lsim(system, siggen_wf, t)
      siggen_wf /= np.amax(siggen_wf)
      
      siggen_data = siggen_wf[detector.zeroPadding::]
      
      siggen_data = siggen_data*e
      
      out[s:] = siggen_data[0:(len(data) - s)]

      return out

#    baseline_model = siggen_model(t0, radEst, phiEst, zEst, wfScale, tempEst, rc_int, rc_diff, rc_diff_short, fall_time_short_frac, detectorListIdx)
    baseline_model = siggen_model(t0, radEst, phiEst, zEst, wfScale, tempEst, num_1, num_2, num_3, den_1, den_2, den_3, gradIdx, pcRadIdx)
    baseline_observed = Normal("baseline_observed", mu=baseline_model, sd=10., observed= data )
  return signal_model

def CreateFullDetectorModelGivenTransferFunction(detectorList, data, t0_guess, energy_guess):
  detector = detectorList[0,0]
  
  n_waveforms = len(data)
  sample_length = len(data[0])
  
  with Model() as signal_model:
  
    #waveform-specific parameters
    radEst = Uniform('radEst', lower=0,   upper=np.floor(detector.radius), shape=n_waveforms)
    zEst = Uniform('zEst', lower=5,   upper=np.floor(detector.length), shape=n_waveforms)
    phiEst = Uniform('phiEst', lower=0,   upper=np.pi/4, shape=n_waveforms)
    t0 = DiscreteUniform('switchpoint', lower = t0_guess-10, upper=t0_guess+10, testval=t0_guess,  shape=n_waveforms)
    wfScale = Normal('wfScale', mu=energy_guess, sd=.01*energy_guess, shape=n_waveforms)
    
    #detector-specific parameters
    tempEst = Uniform('temp', lower=40,   upper=120, testval=80)
    gradIdx = DiscreteUniform('gradIdx', lower = 0, upper= detectorList.shape[0]-1, testval=detectorList.shape[0]/2)
    pcRadIdx = DiscreteUniform('pcRadIdx', lower = 0, upper= detectorList.shape[1]-1, testval=detectorList.shape[1]/2)

    #take the transfer function as given
    prior_num = [3.64e+09, 1.88e+17, 6.05e+15]
    prior_den = [1, 4.03e+07, 5.14e+14, 7.15e+18]
    system = signal.lti(prior_num, prior_den)
    
    siggen_len = 800 + detector.zeroPadding
    t = np.arange(0, (siggen_len)*10E-9, 10E-9)

    @as_op(itypes=[T.lscalar, T.dscalar, T.dscalar, T.dscalar,  T.dscalar, T.dscalar,  T.lscalar, T.lscalar], otypes=[T.dvector])
    def siggen_model(s, rad, phi, z, e, temp, gradIdx, pcRadIdx):
    
      out = np.zeros(sample_length)
      if (gradIdx > detectorList.shape[0]-1) or (pcRadIdx > detectorList.shape[1]-1) :
        return np.ones(sample_length)*-1.
      
      detector = detectorList[gradIdx, pcRadIdx]
      detector.SetTemperature(temp)
      
      siggen_wf= detector.GetSiggenWaveform(rad, phi, z, energy=2600)

      if siggen_wf is None:
        return np.ones(sample_length)*-1.
      if np.amax(siggen_wf) == 0:
        print "wtf is even happening here?"
        return np.ones(sample_length)*-1.
      siggen_wf = np.pad(siggen_wf, (detector.zeroPadding,0), 'constant', constant_values=(0, 0))

      tout, siggen_wf, x = signal.lsim(system, siggen_wf, t)
      siggen_wf /= np.amax(siggen_wf)
      
      siggen_data = siggen_wf[detector.zeroPadding::]
      siggen_data = siggen_data*e
      
      out[s:] = siggen_data[0:(sample_length - s)]

      return out

#    baseline_model = siggen_model(t0, radEst, phiEst, zEst, wfScale, tempEst, rc_int, rc_diff, rc_diff_short, fall_time_short_frac, detectorListIdx)
    baseline_observed = []
    for i in range(n_waveforms):
      baseline_observed.append( Normal("baseline_observed_%d" % i, mu=siggen_model(t0[i], radEst[i], phiEst[i], zEst[i], wfScale[i], tempEst, gradIdx, pcRadIdx), sd=10., observed= data[i] ) )
  return signal_model
