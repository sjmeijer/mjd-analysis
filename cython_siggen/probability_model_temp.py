#!/usr/local/bin/python
import numpy as np
import scipy.stats as stats
import scipy.optimize as op
from scipy import signal
'''Fit detector properties given an array of waveforms'''

#keeping the field parameters (grad, rad, length) constant!!!

def initializeDetector(det):
  global detector
  detector = det
  detector.ReinitializeDetector()

def initializeDetectorAndWaveforms(det, wf_Arr):
  global detector
  global wf_arr
  detector = det
  wf_arr = wf_Arr

def lnprob(theta):
  '''Bayes theorem'''
  lp = lnprior(theta)
  if not np.isfinite(lp):
    return -np.inf
  
#  like = lnlike_detector(theta)
#  if not np.isfinite(like):
#    print "bad like..."

  return lp + lnlike_detector(theta)

#################################################################
'''Likelihood functions'''
#################################################################
def lnlike_detector(theta):
  '''assumes the data comes in w/ 10ns sampling period'''

  temp, zero_1, pole_1, pole_real, pole_imag = theta[-5:]
#  r_arr, phi_arr, z_arr, scale_arr, t0_arr = theta[:-10].reshape((5, len(wf_arr)))
  r_arr, phi_arr, z_arr, scale_arr, t0_arr, smooth_arr = theta[:-5].reshape((6, len(wf_arr)))

  if temp <40 or temp > 120:
    return -np.inf
  
  zeros = [zero_1,0 ]
  poles = [ pole_real+pole_imag*1j, pole_real-pole_imag*1j, pole_1]
  system = signal.lti(zeros, poles, 1E7 )
  detector.SetTransferFunction(system)
  
  if temp != detector.temperature:
    detector.SetTemperature(temp)

  totalLike = 0
  for (wf_idx) in np.arange(r_arr.size):
    wf_like = lnlike_waveform( [r_arr[wf_idx], phi_arr[wf_idx], z_arr[wf_idx], scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx]], wf_arr[wf_idx], )
#    wf_like = lnlike_waveform( [r_arr[wf_idx], phi_arr[wf_idx], z_arr[wf_idx], scale_arr[wf_idx], t0_arr[wf_idx]], wf_arr[wf_idx], )

    if not np.isfinite(wf_like):
      return -np.inf

    #NORMALIZE FOR WF LENGTH
    totalLike += wf_like / wf_arr[wf_idx].wfLength

  return totalLike

def lnlike_waveform(theta, wf):
  r, phi, z, scale, t0, smooth = theta
#  r, phi, z, scale, t0,  = theta


  if scale < 0 or t0 < 0:
    return -np.inf
  if smooth < 0:
     return -np.inf
  if not detector.IsInDetector(r, phi, z):
    return -np.inf

  data = wf.windowedWf
  model_err = wf.baselineRMS

  model = detector.GetSimWaveform(r, phi, z, scale, t0, len(data), smoothing=smooth)
#  model = detector.GetSimWaveform(r, phi, z, scale, t0, len(data))

  if model is None:
    return -np.inf

  inv_sigma2 = 1.0/(model_err**2)

  return -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))

#################################################################
'''Priors'''
#################################################################
def lnprior(theta):
  '''Uniform prior on position
     Normal prior on t0 with sigma=5
     Normal prior on energy scale with sigma = 0.1 * wfMax
  '''
  
  temp,  zero_1, pole_1, pole_real, pole_imag = theta[-5:]
  r_arr, phi_arr, z_arr, scale_arr, t0_arr, smooth_arr = theta[:-5].reshape((6, len(wf_arr)))
#  r_arr, phi_arr, z_arr, scale_arr, t0_arr = theta[:-10].reshape((5, len(wf_arr)))

  if temp <40 or temp > 120:
#    print "bad prior temp %f" % temp
    return -np.inf
  else:
    temp_prior = stats.norm.pdf(temp, loc=78., scale=5. )

  totalPrior = 0
  for (wf_idx) in np.arange(r_arr.size):
    wf_like = lnprior_waveform(r_arr[wf_idx], phi_arr[wf_idx], z_arr[wf_idx], scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx], wf_arr[wf_idx], )
#    wf_like = lnprior_waveform(r_arr[wf_idx], phi_arr[wf_idx], z_arr[wf_idx], scale_arr[wf_idx], t0_arr[wf_idx], wf_arr[wf_idx], )
#    print "      r: %0.2f, phi: %0.3f, z: %0.2f, e: %0.2f, t0: %0.2f" % (r_arr[wf_idx], phi_arr[wf_idx], z_arr[wf_idx], scale_arr[wf_idx], t0_arr[wf_idx])
    if not np.isfinite(wf_like):
#      print "bad wf prior"
      return -np.inf

    totalPrior += wf_like

  return totalPrior + np.log(temp_prior)

def lnprior_waveform(r, phi, z, scale, t0,  smooth, wf, ):
  if not detector.IsInDetector(r, phi, z):
#    print "bad prior: position (%f, %f, %f)" % (r, phi, z)
    return -np.inf
  else:
    location_prior = 1.
  if smooth < 0:
    return -np.inf

  #TODO: rename this so it isn't so confusing
  scale_prior = stats.norm.pdf(scale, loc=wf.wfMax, scale=0.1*wf.wfMax )
  t0_prior = 1.#stats.norm.pdf(t0, loc=wf.t0Guess, scale=5. )

  return np.log(location_prior * location_prior * t0_prior )

def neg_lnlike_wf(theta, wf):
  return -1*lnlike_waveform(theta, wf)

def minimize_waveform_only(r, phi, z, scale, t0, smooth, wf,):
  result = op.minimize(neg_lnlike_wf, [r, phi, z, scale, t0, smooth], args=(wf) ,method="Powell")
  return result

def minimize_waveform_only_star(a_b):
  return minimize_waveform_only(*a_b)


