#!/usr/local/bin/python

import numpy as np
from pymc3 import *#DiscreteUniform, Exponential, Poisson, Uniform, Normal, find_MAP, HalfNormal, switch, Model, Deterministic, exp, Metropolis, traceplot, sample, summary
import theano.tensor as T
from theano.compile.ops import as_op
from scipy import signal

import matplotlib.pyplot as plt
"""
    Models for ppc response
    """

def CreateFullDetectorModel(detector, waveforms, startGuess, prior_zero, prior_pole1, prior_pole_real, prior_pole_imag):
  
  n_waveforms = len(waveforms)
  sample_length = len(waveforms[0].windowedWf)
  
  print "Initting with prior_pole_imag = " + str(prior_pole_imag)
  
  with Model() as signal_model:
    
    #waveform-specific parameters
    radEst = Uniform('radEst', lower=0,   upper=np.floor(detector.detector_radius*10.)/10., shape=n_waveforms, testval=(startGuess['radEst'])  )
    zEst = Uniform('zEst', lower=0.1,   upper=np.floor(detector.detector_length*10.)/10., shape=n_waveforms, testval=(startGuess['zEst']))
    phiEst = Uniform('phiEst', lower=0,   upper=np.pi/4, shape=n_waveforms, testval=(startGuess['phiEst']))
    
    wfScale = Normal('wfScale', mu=startGuess['wfScale'], sd=0.05*startGuess['wfScale'], shape=n_waveforms  )
    
    BoundAtZero = Bound(Normal, lower=0)
    t0 =  BoundAtZero('switchpoint', mu=startGuess['switchpoint'], sd=5, shape=n_waveforms)
    
    SigmaBound = Bound(Normal, lower=0, upper=5)
    sigma =   SigmaBound('sigma', mu=startGuess['smooth'], sd=0.3, shape=n_waveforms)
    
    #detector-wide params
#    tempEst = Bound( Normal('temp', mu=startGuess['temp'], sd=3.), lower=40, upper=120)
#    grad = Bound( Normal('temp', mu=startGuess['grad'], sd=3.), lower=detector.gradList[0], upper=detector.gradList[-1])
#    pcRad = Bound( Normal('pcRad', mu=startGuess['pcRad'], sd=0.2), lower=detector.pcRadList[0], upper=detector.pcRadList[-1])
#    pcLen = Bound( Normal('pcLen', mu=startGuess['pcLen'], sd=0.2), lower=detector.pcLenList[0], upper=detector.pcLenList[-1])


    BoundTemp = Bound(Normal, lower=40, upper=120)
    tempEst = BoundTemp('temp', mu=startGuess['temp'], sd=3.)
    
    BoundGrad = Bound(Normal, lower=detector.gradList[0], upper=detector.gradList[-1])
    BoundRad = Bound(Normal, lower=detector.pcRadList[0], upper=detector.pcRadList[-1])
    BoundLen = Bound(Normal, lower=detector.pcLenList[0], upper=detector.pcLenList[-1])
    
    grad =  BoundGrad('grad', mu=startGuess['grad'], sd=3., testval=startGuess['grad'])
    pcRad =  BoundRad('pcRad', mu=startGuess['pcRad'], sd=0.2, testval=startGuess['pcRad'])
    pcLen = BoundLen('pcLen', mu=startGuess['pcLen'], sd=0.2, testval=startGuess['pcLen'])

    zero_1 = BoundAtZero('zero_1', mu=prior_zero, sd=.01*prior_zero, testval=prior_zero)
    pole_1 = BoundAtZero('pole_1', mu=prior_pole1, sd=.001*prior_pole1, testval=prior_pole1)
    pole_real = BoundAtZero('pole_real', mu=prior_pole_real, sd=.01*prior_pole_real, testval=prior_pole_real)
    pole_imag = BoundAtZero('pole_imag', mu=prior_pole_imag, sd=.01*prior_pole_imag, testval=prior_pole_imag)
    
    @as_op(itypes=[T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar,  T.dscalar,  T.dscalar, T.dscalar,T.dscalar,  T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.wscalar], otypes=[T.fvector])
    def siggen_model(s, rad, phi, z, e, smooth, temp, zero_1, pole_1, pole_real, pole_imag, grad, pc_rad, pc_len, fit_length):
    
      print "pole_imag is " + str(pole_imag)
      
      if pole_imag <0 or pole_real < 0 or pole_1 < 0 or zero_1 < 0:
        return np.ones(fit_length,dtype=np.dtype('f4'))*-np.inf
    
      if s<0 or s>= fit_length:
        return np.ones(fit_length,dtype=np.dtype('f4'))*-np.inf
      if smooth<0:
        return np.ones(fit_length,dtype=np.dtype('f4'))*-np.inf
    
      if (grad > detector.gradList[-1]) or (grad < detector.gradList[0]) :
        return np.ones(fit_length,dtype=np.dtype('f4'))*-np.inf
      if (pc_rad > detector.pcRadList[-1]) or (pc_rad < detector.pcRadList[0]) :
        return np.ones(fit_length,dtype=np.dtype('f4'))*-np.inf
      if (pc_len > detector.pcLenList[-1]) or (pc_len < detector.pcLenList[0]) :
        return np.ones(fit_length,dtype=np.dtype('f4'))*-np.inf
      
      if not detector.IsInDetector(rad, phi, z):
        return -np.inf * np.ones(fit_length,dtype=np.dtype('f4'))
      
      zeros = [zero_1, -1., 1. ]
      poles = [pole_1, pole_real+pole_imag*1j, pole_real-pole_imag*1j, ]
      detector.SetTransferFunction(zeros, poles)
      
      if detector.pcRad != pc_rad or detector.pcLen != pc_len or detector.impurityGrad != grad:
        detector.SetFields(pc_rad, pc_len, grad)
      
      siggen_wf = detector.GetSimWaveform(rad, phi, z, e, s, fit_length, smoothing=smooth)
      if siggen_wf is None:
#        print "siggen wf is none at (%0.2f, %0.2f, %0.2f)" % (rad, phi, z)
        return np.ones(fit_length, dtype=np.dtype('f4'))*-np.inf


#      plt.ion()
#      plt.figure(14)
#      plt.clf()
#      plt.plot(siggen_wf)
#      for (i, wf) in enumerate(waveforms):
#        plt.plot(wf.windowedWf, color="r")
#      value = raw_input('  --> Press q to quit, any other key to continue\n')
#      plt.ioff()

      return siggen_wf
        
    baseline_observed = []
    for (i, wf) in enumerate(waveforms):
      wflength = T.as_tensor_variable(np.int16(wf.wfLength))
      baseline_observed.append( Normal("baseline_observed_%d" % i, mu=siggen_model(t0[i], radEst[i], phiEst[i], zEst[i], wfScale[i], sigma[i], tempEst, zero_1, pole_1, pole_real, pole_imag, grad, pcRad, pcLen, wflength), sd=wf.baselineRMS, observed= wf.windowedWf ) )
    return signal_model


################################################################################################################################

class DetectorSignal(Continuous):
    def __init__(self, detector, *args, **kwargs):
        super(Beta, self).__init__(*args, **kwargs)
        self.detector = detector

    def logp(self, value):
        mu = self.mu
        return beta_logp(value - mu)

def signal_model(s, rad, phi, z, e, temp, grad, pcRad, fit_length):
  if s<0 or s>= fit_length:
    print "cutting off from s at %f" % s
    return np.ones(fit_length)*-np.inf
  if (grad > gradList[-1]) or (grad <gradList[0]):
    print "cutting off from grad at %f" % grad
    return np.ones(fit_length)*-np.inf
  if (pcRad > pcRadList[-1]) or (pcRad <pcRadList[0]):
    print "cutting off from pc rad at %f" % pcRad
    return np.ones(fit_length)*-np.inf
  if temp < 40 or temp >120:
    print "cutting off from temp at %f" % temp
    return np.ones(fit_length)*-np.inf
      
  detector.SetTemperature(temp)
  detector.SetFields(pcRad, grad)
  siggen_wf = detector.GetSimWaveform(rad, phi, z, e, s, fit_length)

  if siggen_wf is None:
    print "siggen wf was none (r:%f, phi:%f, z:%f)" % (rad, phi, z)
    return np.ones(fit_length)*-np.inf
  return siggen_wf


