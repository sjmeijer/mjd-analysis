#!/usr/local/bin/python

import numpy as np
from pymc3 import *#DiscreteUniform, Exponential, Poisson, Uniform, Normal, find_MAP, HalfNormal, switch, Model, Deterministic, exp, Metropolis, traceplot, sample, summary
import theano.tensor as T
from theano.compile.ops import as_op
from scipy import signal

#import matplotlib.pyplot as plt
"""
    Models for ppc response
    """



#def CreateFullDetectorModelGivenTransferFunctionOneField(detector, waveforms, tempGuess):
#  
#  n_waveforms = len(waveforms)
#  sample_length = len(waveforms[0].windowedWf)
#  
#  with Model() as signal_model:
#    
#    #waveform-specific parameters
#    radEst = Uniform('radEst', lower=0,   upper=np.floor(detector.detector_radius), shape=n_waveforms)
#    zEst = Uniform('zEst', lower=0,   upper=np.floor(detector.detector_length), shape=n_waveforms)
#    phiEst = Uniform('phiEst', lower=0,   upper=np.pi/4, shape=n_waveforms)
#    
#    wfScale = Normal('wfScale', mu=waveforms[0].wfMax, sd=.01*waveforms[0].wfMax, shape=n_waveforms)
#    t0 = Normal('switchpoint', mu=waveforms[0].t0Guess, sd=3, shape=n_waveforms)
#    
#    tempEst = Normal('temp', mu=tempGuess, sd=3.)
#    
#    @as_op(itypes=[T.dscalar, T.dscalar, T.dscalar, T.dscalar,  T.dscalar, T.dscalar, T.wscalar], otypes=[T.dvector])
#    def siggen_model(s, rad, phi, z, e, temp, fit_length):
#      if s<0 or s>= fit_length:
#        return np.ones(fit_length)*-np.inf
#     
#      detector.SetTemperature(temp)
#      siggen_wf = detector.GetSimWaveform(rad, phi, z, e, s, fit_length)
#  
#      if siggen_wf is None:
#        return np.ones(fit_length)*-np.inf
#      return siggen_wf
#        
#    baseline_observed = []
#    for (i, wf) in enumerate(waveforms):
#      wflength = T.as_tensor_variable(np.int16(wf.wfLength))
#      baseline_observed.append( Normal("baseline_observed_%d" % i, mu=siggen_model(t0[i], radEst[i], phiEst[i], zEst[i], wfScale[i], tempEst, wflength), sd=wf.baselineRMS, observed= wf.windowedWf ) )
#    return signal_model
#
#def CreateFullDetectorModelGivenTransferFunction(detector, waveforms, startGuess):
#  
#  n_waveforms = len(waveforms)
#  gradList = detector.gradList
#  pcRadList = detector.pcRadList
#  
#  print startGuess['radEst']
#  
#  
#  with Model() as signal_model:
#    
#    #waveform-specific parameters
#    radEst = Uniform('radEst', lower=0,   upper=np.floor(detector.detector_radius), shape=n_waveforms, testval = 15.)
#    zEst = Uniform('zEst', lower=0,   upper=np.floor(detector.detector_length), shape=n_waveforms,  testval = startGuess['zEst'])
#    phiEst = Uniform('phiEst', lower=0,   upper=np.pi/4, shape=n_waveforms,  testval = startGuess['phiEst'])
#    
#    #keep an eye on these, theyre sketchy
#    wfScale = Normal('wfScale', mu= startGuess['wfScale'], sd=.01*startGuess['wfScale'], shape=n_waveforms, testval=startGuess['wfScale'])
#    t0 = Normal('switchpoint', mu=startGuess['switchpoint'], sd=3, shape=n_waveforms, testval=startGuess['switchpoint'])
#    
#    #detector-wide params
#    tempEst = Normal('temp', mu= startGuess['temp'], sd=3., testval = startGuess['temp'])
#    grad = Uniform('grad', lower = gradList[0], upper= gradList[-1], testval=startGuess['grad'])
#    pcRad = Uniform('pcRad', lower = pcRadList[0], upper= pcRadList[-1], testval=startGuess['pcRad'])
#    
#    @as_op(itypes=[T.dscalar, T.dscalar, T.dscalar, T.dscalar,  T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.wscalar], otypes=[T.dvector])
#    def siggen_model(s, rad, phi, z, e, temp, grad, pcRad, fit_length):
##      print "grad is %0.4f, pcrad is %0.4f" % (grad, pcRad)
#
#      if s<0 or s>= fit_length:
#        print "cutting off from s at %f" % s
#        return np.ones(fit_length)*-np.inf
#      if (grad > gradList[-1]) or (grad <gradList[0]):
#        print "cutting off from grad at %f" % grad
#        return np.ones(fit_length)*-np.inf
#      if (pcRad > pcRadList[-1]) or (pcRad <pcRadList[0]):
#        print "cutting off from pc rad at %f" % pcRad
#        return np.ones(fit_length)*-np.inf
#      if temp < 40 or temp >120:
#        print "cutting off from temp at %f" % temp
#        return np.ones(fit_length)*-np.inf
#          
#      detector.SetTemperature(temp)
#      detector.SetFields(pcRad, grad)
#      siggen_wf = detector.GetSimWaveform(rad, phi, z, e, s, fit_length)
#  
#      if siggen_wf is None:
#        print "siggen wf was none (r:%f, phi:%f, z:%f)" % (rad, phi, z)
#        return np.ones(fit_length)*-np.inf
#      return siggen_wf
#        
#    baseline_observed = []
#    for (i, wf) in enumerate(waveforms):
#      wflength = T.as_tensor_variable(np.int16(wf.wfLength))
#      baseline_observed.append( Normal("baseline_observed_%d" % i, mu=siggen_model(t0[i], radEst[i], phiEst[i], zEst[i], wfScale[i], tempEst, grad, pcRad, wflength), sd=wf.baselineRMS, observed= wf.windowedWf ) )
#    return signal_model

def CreateFullDetectorModel(detector, waveforms, startGuess, prior_zero, prior_pole1, prior_pole_real, prior_pole_imag):
  
  n_waveforms = len(waveforms)
  sample_length = len(waveforms[0].windowedWf)
  
  with Model() as signal_model:
    
    #waveform-specific parameters
    radEst = Uniform('radEst', lower=0,   upper=np.floor(detector.detector_radius*10.)/10., shape=n_waveforms, testval=(startGuess['radEst'])  )
    zEst = Uniform('zEst', lower=0,   upper=np.floor(detector.detector_length*10.)/10., shape=n_waveforms, testval=(startGuess['zEst']))
    phiEst = Uniform('phiEst', lower=0,   upper=np.pi/4, shape=n_waveforms, testval=(startGuess['phiEst']))
    
    wfScale = Normal('wfScale', mu=startGuess['wfScale'], sd=0.05*startGuess['wfScale'], shape=n_waveforms  )
    
    BoundAtZero = Bound(Normal, lower=0)
    t0 =  BoundAtZero('switchpoint', mu=startGuess['switchpoint'], sd=5, shape=n_waveforms)
    
    SigmaBound = Bound(Normal, lower=0, upper=20)
    sigma =   SigmaBound('sigma', mu=startGuess['smooth'], sd=3, shape=n_waveforms)
    
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
    
    grad =  BoundGrad('grad', mu=startGuess['grad'], sd=3.)
    pcRad =  BoundRad('pcRad', mu=startGuess['pcRad'], sd=0.2)
    pcLen = BoundLen('pcLen', mu=startGuess['pcLen'], sd=0.2)


    zero_1 = Normal('zero_1', mu=prior_zero, sd=.3*prior_zero)
    pole_1 = Normal('pole_1', mu=prior_pole1, sd=.3*prior_pole1)
    pole_real = Normal('pole_real', mu=prior_pole_real, sd=.3*prior_pole_real)
    pole_imag = Normal('pole_imag', mu=prior_pole_imag, sd=.3*prior_pole_imag)
    
    
#    fig1 = plt.figure(100)

    
    @as_op(itypes=[T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar,  T.dscalar,  T.dscalar, T.dscalar,T.dscalar,  T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.wscalar], otypes=[T.dvector])
    def siggen_model(s, rad, phi, z, e, smooth, temp, zero_1, pole_1, pole_real, pole_imag, grad, pc_rad, pc_len, fit_length):
    
      if s<0 or s>= fit_length:
        return np.ones(fit_length)*-np.inf
      if smooth<0:
        return np.ones(fit_length)*-np.inf
    
      if (grad > detector.gradList[-1]) or (grad < detector.gradList[0]) :
        return np.ones(fit_length)*-np.inf
      if (pc_rad > detector.pcRadList[-1]) or (pc_rad < detector.pcRadList[0]) :
        return np.ones(fit_length)*-np.inf
      if (pc_len > detector.pcLenList[-1]) or (pc_len < detector.pcLenList[0]) :
        return np.ones(fit_length)*-np.inf
      
      if not detector.IsInDetector(rad, phi, z):
        return -np.inf * np.ones(fit_length)
      
      zeros = [zero_1, 0]
      poles = [pole_real + pole_imag*1j, pole_real - pole_imag*1j, pole_1]
      detector.SetTransferFunction(zeros, poles, 1E7)
      
      if detector.pcRad != pc_rad or detector.pcLen != pc_len or detector.impurityGrad != grad:
        detector.SetFields(pc_rad, pc_len, grad)
      
      siggen_wf = detector.GetSimWaveform(rad, phi, z, e, s, fit_length, smoothing=smooth)
      if siggen_wf is None:
#        print "siggen wf is none at (%0.2f, %0.2f, %0.2f)" % (rad, phi, z)
        return np.ones(fit_length)*-np.inf
      
            
#      plt.figure(100)
#      plt.clf()
#      for wf in waveforms:
#        plt.plot(wf.windowedWf, color="r")
#      plt.plot(siggen_wf)
#      value = raw_input('  --> Press q to quit, any other key to continue\n')

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


