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
  r, phi, z, scale, t0, smooth,  b_over_a, c, d, h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0,    = theta
  lp = lnprior_waveform(r, phi, z, scale, t0, smooth , )
#  if rcfrac > 1: return -np.inf
#  if rc1 < 0 or rc2 <0: return -np.inf

#  if trapping_rc < 0: return -np.inf
#  if not np.isfinite(lp):
#    return -np.inf

#  gradList  = detector.gradList
#    if grad < gradList[0] or grad > gradList[-1]:
#    return -np.inf

#  pcRadList =  detector.pcRadList
#  pcLenList =  detector.pcLenList
#  
#  if pcRad < pcRadList[0] or pcRad > pcRadList[-1]:
#    return -np.inf
#  if pcLen < pcLenList[0] or pcLen > pcLenList[-1]:
#    return -np.inf

#  if temp <40 or temp > 120:
#    return -np.inf

  return lp + lnlike_waveform(theta, )

def lnlike_waveform(theta):
  r, phi, z, scale, t0, smooth,  b_over_a, c, d, h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0, = theta
#  c = -0.815152
#  d = 0.822696
  rc1 = 74.4
  rc2 = 1.79
  rcfrac = 0.992
  
  
#  r, phi, z, scale, t0,  = theta

#  if collection_rc < 0: return -np.inf
#  detector.trapping_rc = trapping_rc

#  rc=72

  if rcfrac > 1: return -np.inf

  detector.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)
  detector.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
#  detector.SetTemperature(temp)

#  if detector.impurityGrad != grad:
##      print "   actually setting the grad!!"
#      detector.SetFieldsGradInterp(grad)
#  if detector.pcRad != pcRad or detector.pcLen != pcLen or detector.impurityGrad != impGrad:
#    detector.SetFields(pcRad, pcLen, impGrad)

#  r_det = np.cos(theta) * r
#  z_det = np.sin(theta) * r

  if scale < 0 or t0 < 0:
    return -np.inf
  if smooth < 0:
     return -np.inf
  if not detector.IsInDetector(r, phi, z):
    return -np.inf

  data = wf.windowedWf
  model_err = wf.baselineRMS

  model = detector.MakeSimWaveform(r, phi, z, scale, t0, len(data), h_smoothing=smooth)
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


