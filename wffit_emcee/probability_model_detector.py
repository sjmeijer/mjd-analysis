#!/usr/local/bin/python
import numpy as np
import scipy.stats as stats

'''Fit detector properties given an array of waveforms'''


def initializeDetector(det):
  global detector
  detector = det
  detector.ReinitializeDetector()

def init_wfs(wf_Arr):
  global wf_arr
  wf_arr = wf_Arr

def lnprob(theta):

  '''Bayes theorem'''
  lp = lnprior(theta)
  if not np.isfinite(lp):
    return -np.inf
  return lp + lnlike_detector(theta)

#################################################################
'''Likelihood functions'''
#################################################################
def lnlike_detector(theta):
  '''assumes the data comes in w/ 10ns sampling period'''

  temp, impGrad, pcRad, pcLen, num1, num2, num3, den1, den2, den3 = theta[-10:]
  r_arr, phi_arr, z_arr, scale_arr, t0_arr, smooth_arr = theta[:-10].reshape((6, len(wf_arr)))
  
  gradList  = detector.gradList
  pcRadList =  detector.pcRadList
  pcLenList =  detector.pcLenList
  
  #Take care of detector business
  if pcRad < pcRadList[0] or pcRad > pcRadList[-1]:
    return -np.inf
  if pcLen < pcLenList[0] or pcRad > pcLenList[-1]:
    return -np.inf
  if impGrad < gradList[0] or impGrad > gradList[-1]:
    return -np.inf
  if temp < 40 or temp > 120:
    return -np.inf
  
  num = [num1, num2, num3]
  den = [1, den1, den2, den3]
  detector.SetTransferFunction(num, den)
  detector.SetTemperature(temp)
  detector.SetFields(pcRad, impGrad)

  totalLike = 0
  for (wf_idx) in np.arange(r_arr.size):
    wf_like = lnlike_waveform( [r_arr[wf_idx], phi_arr[wf_idx], z_arr[wf_idx], scale_arr[wf_idx], t0_arr[wf_idx]], smooth_arr[wf_idx], wf_arr[wf_idx], )

    if not np.isfinite(wf_like):
      return -np.inf

    #NORMALIZE FOR WF LENGTH
    totalLike += wf_like / wf_arr[wf_idx].wfLength
  
#  print "likelihood: %0.3f" % totalLike
#  print ">>> temp: %0.2f, pcrad %0.6f, impgrad = %0.4f" % (temp, pcRad, impGrad)
#  print ">>> num: " + str(num) + ", den: " + str(den)
  return totalLike

def lnlike_waveform(theta, wf):
  r, phi, z, scale, t0, smooth = theta
  
  if scale < 0 or t0 < 0 or smooth<0:
    return -np.inf
  if not detector.IsInDetector(r, phi, z):
    return -np.inf

  data = wf.windowedWf
  model_err = wf.baselineRMS

  model = detector.GetSimWaveform(r, phi, z, scale, t0, len(data), smoothing=smooth)
  
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
  
  temp, impGrad, pcRad, pcLen, num1, num2, num3, den1, den2, den3 = theta[-10:]
  r_arr, phi_arr, z_arr, scale_arr, t0_arr, smooth_arr = theta[:-10].reshape((6, len(wf_arr)))
  
  gradList  = detector.gradList
  pcRadList =  detector.pcRadList
  pcLenList =  detector.pcLenList
  
  #Flat priors on most the detector params...
  if pcRad < pcRadList[0] or pcRad > pcRadList[-1]:
    return -np.inf
  if pcLen < pcLenList[0] or pcRad > pcLenList[-1]:
    return -np.inf
  if impGrad < gradList[0] or impGrad > gradList[-1]:
    return -np.inf
  #...except for temp.
  if temp <40 or temp > 120:
#    print "bad prior temp %f" % temp
    return -np.inf
  else:
    temp_prior = stats.norm.pdf(temp, loc=81., scale=5. )

  totalPrior = 0
  for (wf_idx) in np.arange(r_arr.size):
    wf_like = lnprior_waveform(r_arr[wf_idx], phi_arr[wf_idx], z_arr[wf_idx], scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx], wf_arr[wf_idx], )

    if not np.isfinite(wf_like):
#      print "bad prior wf"
      return -np.inf

    totalPrior += wf_like

  return totalPrior + np.log(temp_prior)

def lnprior_waveform(r, phi, z, scale, t0, smoothing, wf, ):
  if not detector.IsInDetector(r, phi, z):
#    print "bad prior: position (%f, %f, %f)" % (r, phi, z)
    return -np.inf
  else:
    location_prior = 1.

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

#def lnlike_detector_holdwf(theta,  wfParams):
#  '''assumes the data comes in w/ 10ns sampling period'''
#
#  temp, impGrad, pcRad, num1, num2, num3, den1, den2, den3 = theta
#  r_arr, phi_arr, z_arr, scale_arr, t0_arr = wfParams.reshape((5, len(wf_arr)))
#  
#  gradList  = detector.gradList
#  pcRadList =  detector.pcRadList
#  
#  #Take care of detector business
#  if pcRad < pcRadList[0] or pcRad > pcRadList[-1]:
#    return -np.inf
#  if impGrad < gradList[0] or impGrad > gradList[-1]:
#    return -np.inf
#  if temp < 40 or temp > 120:
#    return -np.inf
#  
#  num = [num1, num2, num3]
#  den = [1, den1, den2, den3]
#  detector.SetTransferFunction(num, den)
#  detector.SetTemperature(temp)
#  detector.SetFields(pcRad, impGrad)
#
#  totalLike = 0
#  for (wf_idx) in np.arange(r_arr.size):
#    wf_like = lnlike_waveform( [r_arr[wf_idx], phi_arr[wf_idx], z_arr[wf_idx], scale_arr[wf_idx], t0_arr[wf_idx]], wf_arr[wf_idx], detector)
#
#    if not np.isfinite(wf_like):
#      return -np.inf
#    #NORMALIZE FOR WF LENGTH
#    totalLike += wf_like / wf_arr[wf_idx].wfLength
#  
#  print "likelihood: %0.3f" % totalLike
#  print ">>> temp: %0.2f, pcrad %0.6f, impgrad = %0.4f" % (temp, pcRad, impGrad)
#  print ">>> num: " + str(num) + ", den: " + str(den)
#  return totalLike

