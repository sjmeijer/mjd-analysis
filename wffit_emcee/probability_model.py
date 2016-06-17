#!/usr/local/bin/python


import numpy as np
import scipy.stats as stats



def lnprob(theta, data, detector, baseline_rms, wfMax, t0Guess):
  '''Bayes theorem'''
  lp = lnprior(theta, detector, wfMax, t0Guess)
  if not np.isfinite(lp):
    return -np.inf
  return lp + lnlike(theta, data, detector, baseline_rms)

#################################################################
'''Likelihood functions'''
#################################################################
def lnlike(theta, data, detector, baseline_rms):
  '''assumes the data comes in w/ 10ns sampling period'''
  model_err = baseline_rms

  r, phi, z, scale, t0 = theta

  if not detector.IsInDetector(r, phi, z):
    return -np.inf

  model = detector.GetSimWaveform(r, phi, z, scale, t0, len(data))
  
  #if siggen couldn't calculate it, its outside the crystal (hopefully, anyway!)
  if model is None:
    return -np.inf
  
  inv_sigma2 = 1.0/(model_err**2)
  return -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnlike_simple(theta, data, detector, baseline_rms, t0):
  '''assumes the data comes in w/ 10ns sampling period
     Does not estimate t0 (MLE is very bad / useless at discrete parameters)
  '''
  
  model_err = baseline_rms

  r, phi, z, scale = theta

  if not detector.IsInDetector(r, phi, z):
    return -np.inf

  model = detector.GetSimWaveform(r, phi, z, scale, t0, len(data))
  
  #if siggen couldn't calculate it, its outside the crystal (hopefully, anyway!)
  if model is None:
    return -np.inf
  
  inv_sigma2 = 1.0/(model_err**2)
  return -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))

#################################################################
'''Priors'''
#################################################################
def lnprior(theta, detector, wfMax, t0Guess):
  '''Uniform prior on position
     Normal prior on t0 with sigma=5
     Normal prior on energy scale with sigma = 0.1 * wfMax
  '''
  r, phi, z, scale, t0 = theta
  
  if not detector.IsInDetector(r, phi, z):
    return -np.inf
  else:
    location_prior = 1.

  #TODO: rename this so it isn't so confusing
  scale_prior = stats.norm.pdf(scale, loc=wfMax, scale=0.1*wfMax )

  #TODO: rename this so it isn't so confusing
  t0_prior = stats.norm.pdf(t0, loc=t0Guess, scale=5. )

  return np.log(location_prior * location_prior * t0_prior )


