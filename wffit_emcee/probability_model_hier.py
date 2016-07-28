#!/usr/local/bin/python
import numpy as np
import scipy.stats as stats
import scipy.optimize as op
import itertools
from multiprocessing import Pool
'''Fit detector properties given an array of waveforms'''


def init_detector( det):
  global detector
  detector = det
  
def init_wfs(wf_Arr):
  global wf_arr
  wf_arr = wf_Arr

def lnlike_detector(theta, *wfParams):
  '''assumes the data comes in w/ 10ns sampling period'''

  temp, impGrad, pcRad = np.copy(theta[:3])
  
  pool = wfParams[0]
  
  wfParams = np.array(wfParams[1:])
  r_arr, phi_arr, z_arr, scale_arr, t0_arr = wfParams[:].reshape((5, len(wf_arr)))
  num = [np.copy(theta[3]) *1E9 , np.copy(theta[4]) *1E17, np.copy(theta[5])*1E15 ]
  den = [1, np.copy(theta[6]) *1E7 , np.copy(theta[7]) *1E14, np.copy(theta[8])*1E18 ]
  
  temp *= 10.
  impGrad /= 100.
  
  print ">>>> temp: %0.2f, pcrad %0.6f, impgrad = %0.4f" % (temp, pcRad, impGrad)
  print ">>>>              num: " + str(num)
  print ">>>>              den: " + str(den)

  gradList  = detector.gradList
  pcRadList =  detector.pcRadList
  
  #Take care of detector business
  if pcRad < pcRadList[0] or pcRad > pcRadList[-1]:
    return -np.inf
  if impGrad < gradList[0] or impGrad > gradList[-1]:
    return -np.inf
  if temp < 40 or temp > 120:
    return -np.inf
  
  totalLike = 0

  a_args = np.arange(r_arr.size)

  args = []

  for idx in a_args:
    args.append( [r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], temp, pcRad, impGrad, num, den, wf_arr[idx] ]  )

  results = pool.map(minimize_wf_star, args)

  for (idx, result) in enumerate(results):
  
    r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx] = result["x"]
    wf_like = -1*result['fun']
    
    print "  >> wf %d (normalized likelihood %0.2f):" % (idx, wf_like/wf_arr[idx].wfLength)
    print "      r: %0.2f, phi: %0.3f, z: %0.2f, e: %0.2f, t0: %0.2f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx])
    
    if not np.isfinite(wf_like):
      return -np.inf
    #NORMALIZE FOR WF LENGTH
    totalLike += wf_like / wf_arr[idx].wfLength
  
  print "  >>total likelihood: %0.3f" % totalLike

  return totalLike


def lnlike_waveform(theta, wf):
  r, phi, z, scale, t0 = np.copy(theta)
  
  r *= 10.
  z *= 10.
  scale *= 1000.
  
  if scale < 0 or t0 < 0:
    return -np.inf

  if not detector.IsInDetector(r, phi, z):
    return -np.inf

  data = wf.windowedWf
  model_err = wf.baselineRMS

  model = detector.GetSimWaveform(r, phi, z, scale, t0, len(data))
  
  if model is None:
    return -np.inf

  inv_sigma2 = 1.0/(model_err**2)


  return -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def neg_lnlike_wf(theta, wf):
  return -1*lnlike_waveform(theta, wf)

def minimize_wf(r, phi, z, scale, t0, temp, pcRad, impGrad, num, den, wf):

  detector.SetTemperature(temp)
  detector.SetFields(pcRad, impGrad)
  detector.SetTransferFunction(num, den)
  
  result = op.minimize(neg_lnlike_wf, [r, phi, z, scale, t0], args=wf ,method="Nelder-Mead", tol=0.5)

  return result

def minimize_wf_star(a_b):
  return minimize_wf(*a_b)



