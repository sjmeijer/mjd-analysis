import numpy as np
import scipy.stats as stats
import scipy.optimize as op
import dnest4

import numpy.random as rng

def initializeDetector(det):
  global detector
  detector = det
def initializeWaveform( wf_init):
  global wf
  wf = wf_init

def initializeDetectorAndWaveform(det, wf_init):
  initializeWaveform(wf_init)
  initializeDetector(det)

class Model(object):
    """
    Specify the model in Python.
    """
    def __init__(self):
        """
        Parameter values *are not* stored inside the class
        """
        pass

    def from_prior(self):
        """
        Unlike in C++, this must *return* a numpy array of parameters.
        """
        
        r = rng.rand() * detector.detector_radius
        phi = rng.rand() * np.pi/4
        z = rng.rand() * detector.detector_length
        t0 = rng.rand()*25
        scale = 0.1*wf.wfMax*rng.randn() + wf.wfMax
        smooth = np.clip(rng.randn() + 10, 0, 20)

        return np.array([r, phi, z, scale, t0, smooth])

    def perturb(self, params):
        """
        Unlike in C++, this takes a numpy array of parameters as input,
        and modifies it in-place. The return value is still logH.
        """
        logH = 0.0
        which = rng.randint(5)

        if which == 0:
          max_val = detector.detector_radius
        elif which == 1:
          max_val = np.pi/4
        elif which == 2:
          max_val = detector.detector_length
        elif which == 3:
          logH -= -0.5*(params[which]/wf.wfMax)**2
          params[which] += wf.wfMax*dnest4.randh()
          logH += -0.5*(params[which]/wf.wfMax)**2
        elif which == 4:
          max_val = 25.
        else:
          max_val = 20.

        if which != 3:
          log_sigma = np.log(params[which])
          log_sigma += dnest4.randh()
          log_sigma = dnest4.wrap(log_sigma, 0, max_val)
          params[which] = np.exp(log_sigma)

        return logH

    def log_likelihood(self, params):
        """
        Gaussian sampling distribution.
        """
        r, phi, z, scale, t0, smooth = params
        
        if scale < 0 or t0 < 0:
          return -np.inf
        if smooth < 0:
           return -np.inf
        if not detector.IsInDetector(r, phi, z):
          return -np.inf
          
        data = wf.windowedWf
        model_err = wf.baselineRMS

        model = detector.MakeSimWaveform(r, phi, z, scale, t0, len(data), h_smoothing=smooth)
        if model is None:
          return -np.inf
  
        inv_sigma2 = 1.0/(model_err**2)
        return -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))
