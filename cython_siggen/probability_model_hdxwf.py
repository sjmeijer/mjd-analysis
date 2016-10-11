#!/usr/local/bin/python
import numpy as np
import scipy.stats as stats
import scipy.optimize as op
'''Fit detector properties given an array of waveforms'''

def initializeDetector(det):
  global detector
  detector = det

def initializeWaveform( wf_init):
  global wf
  wf = wf_init

def initializeDetectorAndWaveform(det, wf_init):
  initializeWaveform(wf_init)
  initializeDetector(det)


def lnprob_waveform(theta):
  '''Bayes theorem'''
  r, phi, z, scale, t0, smooth,temp, b_over_a, c, d, rc  = theta
  lp = lnprior_waveform(r, phi, z, scale, t0, smooth , )
  if not np.isfinite(lp):
    return -np.inf
  return lp + lnlike_waveform(theta, )

def lnlike_waveform(theta):
  r_det, phi, z_det, scale, t0, smooth, temp,  b_over_a, c, d, rc   = theta
#  r, phi, z, scale, t0,  = theta

  if temp < 40 or temp > 120: return -np.inf

#  if collection_rc < 0: return -np.inf
#  detector.collection_rc = collection_rc

#  rc=72

  detector.SetTransferFunction(b_over_a, c, d, rc)
  detector.SetTemperature(temp)
  
#  r_det = np.cos(theta) * r
#  z_det = np.sin(theta) * r

  if scale < 0 or t0 < 0:
    return -np.inf
  if smooth < 0:
     return -np.inf
  if not detector.IsInDetector(r_det, phi, z_det):
    return -np.inf

  data = wf.windowedWf
  model_err = wf.baselineRMS

  model = detector.MakeSimWaveform(r_det, phi, z_det, scale, t0, len(data), h_smoothing=smooth)
#  model = detector.GetSimWaveform(r, phi, z, scale, t0, len(data))

  if model is None:
    return -np.inf

  inv_sigma2 = 1.0/(model_err**2)

  return -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior_waveform(r, phi, z, scale, t0,  smooth, ):
  if not detector.IsInDetector(r, phi, z):
#    print "bad prior: position (%f, %f, %f)" % (r, phi, z)
    return -np.inf
  else:
    location_prior = 1.
  if smooth < 0:
    return -np.inf

  #TODO: rename this so it isn't so confusing
  scale_prior = stats.norm.pdf(scale, loc=wf.wfMax, scale=0.1*wf.wfMax )
  t0_prior = stats.norm.pdf(t0, loc=wf.t0Guess, scale=5. )

  return np.log(location_prior  * t0_prior * scale_prior )


