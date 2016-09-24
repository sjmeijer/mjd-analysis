#!/usr/local/bin/python

import numpy as np
from pymc import DiscreteUniform, Uniform, Normal, deterministic, Deterministic, TruncatedNormal
from scipy import signal

import matplotlib.pyplot as plt

"""
    Models for ppc response
    """

def CreateFullDetectorModel(detector, waveforms, startGuess, prior_zero, prior_pole1, prior_pole_real, prior_pole_imag):
  
  n_waveforms = len(waveforms)
  sample_length = len(waveforms[0].windowedWf)
  
  #detector-wide params
  tempEst = TruncatedNormal('temp', mu=startGuess['temp'], tau=sigToTau(3.), a=40, b=120)
  grad =  Uniform('grad', lower=detector.gradList[0], upper=detector.gradList[-1],    value=startGuess['grad'] )
  pcRad =  Uniform('pcRad', lower=detector.pcRadList[0], upper=detector.pcRadList[-1],value=startGuess['pcRad'] )
  pcLen = Uniform('pcLen', lower=detector.pcLenList[0], upper=detector.pcLenList[-1], value=startGuess['pcLen'] )
  
#  grad =  TruncatedNormal('grad', a=detector.gradList[0], b=detector.gradList[-1],    value=startGuess['grad'], mu=startGuess['grad'],tau=sigToTau(0.03) )
#  pcRad =  TruncatedNormal('pcRad', a=detector.pcRadList[0], b=detector.pcRadList[-1],value=startGuess['pcRad'], mu=startGuess['pcRad'],tau=sigToTau(0.2) )
#  pcLen = TruncatedNormal('pcLen', a=detector.pcLenList[0], b=detector.pcLenList[-1], value=startGuess['pcLen'], mu=startGuess['pcLen'],tau=sigToTau(0.2) )
  
  zero_1 =    TruncatedNormal('zero_1', mu=prior_zero,         tau=sigToTau(.3*prior_zero), value=prior_zero, a=0, b=2)
  pole_1 =    TruncatedNormal('pole_1', mu=prior_pole1,        tau=sigToTau(.3*prior_pole1), value=prior_pole1, a=0, b=2)
  pole_real = TruncatedNormal('pole_real', mu=prior_pole_real, tau=sigToTau(.3*prior_pole_real), value=prior_pole_real, a=0, b=2)
  pole_imag = TruncatedNormal('pole_imag', mu=prior_pole_imag, tau=sigToTau(.3*prior_pole_imag), value=prior_pole_imag, a=0, b=2)
  
  #Make an array of priors for each waveform-specific parameter
  radiusArray = np.empty(n_waveforms, dtype=object)
  zArray = np.empty(n_waveforms, dtype=object)
  phiArray = np.empty(n_waveforms, dtype=object)
  scaleArray = np.empty(n_waveforms, dtype=object)
  t0Array = np.empty(n_waveforms, dtype=object)
  sigArray = np.empty(n_waveforms, dtype=object)
  
  for idx in range(n_waveforms):
    radiusArray[idx] =( Uniform('radEst_%d'%idx, lower=0,   upper=detector.detector_radius, value=startGuess['radEst'][idx]       )  )
    zArray[idx] =(      Uniform('zEst_%d'%idx,   lower=0,   upper=detector.detector_length, value=startGuess['zEst'][idx]         )  )
    phiArray[idx] =(    Uniform('phiEst_%d'%idx, lower=0,   upper=np.pi/4 ,                 value=startGuess['phiEst'][idx]       )  )
    scaleArray[idx] =(  Normal('wfScale_%d'%idx,     mu=startGuess['wfScale'][idx],     tau=sigToTau(0.01*startGuess['wfScale'][idx]), value=startGuess['wfScale'][idx]) )
    t0Array[idx] =(     Normal('switchpoint_%d'%idx, mu=startGuess['switchpoint'][idx], tau=sigToTau(5.),                              value=startGuess['switchpoint'][idx]))
    sigArray[idx] =(    Normal('sigma_%d'%idx,       mu=startGuess['smooth'][idx],      tau=sigToTau(0.3), value=startGuess['smooth'][idx] ))
  
  #This is a deterministic (implicitly?  is this a problem?)
  def siggen_model(s, rad, phi, z, e, smooth, temp, zero_1, pole_1, pole_real, pole_imag, grad, pc_rad, pc_len, fit_length):
  
    if s<0 or s>= fit_length:
      return np.ones(fit_length)*-np.inf
    if smooth<0:
      return np.ones(fit_length)*-np.inf
    if not detector.IsInDetector(rad, phi, z):
      return -np.inf * np.ones(fit_length)

    if temp < 40 or temp > 120:
      return np.ones(fit_length)*-np.inf
    if (grad > detector.gradList[-1]) or (grad < detector.gradList[0]) :
      return np.ones(fit_length)*-np.inf
    if (pc_rad > detector.pcRadList[-1]) or (pc_rad < detector.pcRadList[0]) :
      return np.ones(fit_length)*-np.inf
    if (pc_len > detector.pcLenList[-1]) or (pc_len < detector.pcLenList[0]) :
      return np.ones(fit_length)*-np.inf
    
    zeros = [zero_1, -1., 1. ]
    poles = [pole_1, pole_real+pole_imag*1j, pole_real-pole_imag*1j, ]
    detector.SetTransferFunction(zeros, poles, 1E7)
    detector.SetTemperature(temp)

    if detector.pcRad != pc_rad or detector.pcLen != pc_len or detector.impurityGrad != grad:
      detector.SetFields(pc_rad, pc_len, grad)
    
    siggen_wf = detector.GetSimWaveform(rad, phi, z, e, s, fit_length, smoothing=smooth)
    if siggen_wf is None:
      return np.ones(fit_length)*-np.inf

#    plt.ion()
#    plt.figure(14)
#    plt.clf()
#    plt.plot(siggen_wf)
#    for (i, wf) in enumerate(waveforms):
#      plt.plot(wf.windowedWf, color="r")
#    print "Detector parameters: "
#    print "  temp = %0.3f" % temp
#    print "  zero_1 = %f" % zero_1
#    print "  pole_1 = %f" % pole_1
#    print "  pole_real = %f" % pole_real
#    print "  pole_imag = %f" % pole_imag
#    print "  grad = %0.3f" % grad
#    print "  pc_rad = %0.3f" % pc_rad
#    print "  pc_len = %0.3f" % pc_len
#
#    print "Waveform parameters: "
#    print "  (r,phi,z) = (%0.2f,%0.3f,%0.2f)" % (rad,phi,z)
#    print "  e = %0.3f" % e
#    print "  smooth = %0.3f" % smooth
#    print "  t0 = %0.3f" % s
#    value = raw_input('  --> Press q to quit, any other key to continue\n')
#    plt.ioff()

    return siggen_wf
      
  baseline_observed = np.empty(n_waveforms, dtype=object)
  baseline_sim = np.empty(n_waveforms, dtype=object)
  
  for (i, wf) in enumerate(waveforms):
    baseline_sim[i] = Deterministic(eval = siggen_model,
                  doc='siggen wf %d' % i,
                  name = 'siggen_model_%d'%i,
                  parents = {'s': t0Array[i],
                          'rad': radiusArray[i],'phi': phiArray[i],'z': zArray[i],
                          'e':scaleArray[i],
                          'smooth':sigArray[idx],
                          'temp': tempEst,
                          'zero_1':zero_1, 'pole_1':pole_1, 'pole_real':pole_real, 'pole_imag':pole_imag,
                          'grad':grad,'pc_rad':pcRad,'pc_len':pcLen,
                          'fit_length':wf.wfLength                          },
                  trace = False,
                  plot=False)
    baseline_observed[i] = Normal("baseline_observed_%d" % i, mu=baseline_sim[i], tau=sigToTau(wf.baselineRMS), observed= True, value= wf.windowedWf )
  
  return locals()

def sigToTau(sig):
  tau = np.power(np.float(sig), -2)
#  print "tau is %f" % tau
  return tau

