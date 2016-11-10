#!/usr/local/bin/python

import numpy as np
from pymc import DiscreteUniform, Uniform, Normal, deterministic, Deterministic, TruncatedNormal
from scipy import signal

import matplotlib.pyplot as plt

"""
    Models for ppc response
    """

def CreateTFModel(detector, waveform, startGuess):
  furthest_point = np.sqrt( detector.detector_radius**2 + detector.detector_length**2  )
  fit_length = waveform.wfLength

  radEst =     TruncatedNormal('radEst',    mu=startGuess['radEst'],   a=0,   b=furthest_point, tau=sigToTau(2), value=startGuess['radEst']      )
  thetaEst =   TruncatedNormal('thetaEst',  mu=startGuess['thetaEst'], a=0,   b=np.pi/2,        tau=sigToTau(0.2), value=startGuess['thetaEst']         )
  
  phiEst =     Uniform('phiEst', lower=0,   upper=np.pi/4 ,                 value=startGuess['phiEst']      )
  scaleEst =   Normal('wfScale',     mu=startGuess['wfScale'],     tau=sigToTau(0.01*startGuess['wfScale']), value=startGuess['wfScale'])
  t0Est =      Normal('switchpoint', mu=startGuess['switchpoint'], tau=sigToTau(5.),                         value=startGuess['switchpoint'])
  sigEst=      Normal('sigma',       mu=startGuess['smooth'],      tau=sigToTau(3),                           value=startGuess['smooth'] )
  
  tempEst = TruncatedNormal('temp', mu=startGuess['temp'],    tau=sigToTau(2.),  value=startGuess['temp'], a=40, b=120)
  b_over_a =    Normal('b_over_a',  mu=startGuess['b_over_a'],tau=sigToTau(5),  value=startGuess['b_over_a'])
  c =           Normal('c',         mu=startGuess['c'],       tau=sigToTau(0.2), value=startGuess['c'])
  d =           Normal('d',         mu=startGuess['d'],       tau=sigToTau(0.2), value=startGuess['d'])
  rc1 =         Normal('rc1',       mu=startGuess['rc1'],     tau=sigToTau(5),   value=startGuess['rc1'])
  rc2 =         Normal('rc2',       mu=startGuess['rc2'],     tau=sigToTau(0.5),   value=startGuess['rc2'])
  rcfrac =      Uniform('rcfrac',   lower = 0, upper = 1, value=startGuess['rcfrac'])
  
  def siggen_model(s, r, theta, phi, e, smooth, temp, b_over_a, c, d, rc1, rc2, rcfrac):
  
    if s<0 or s>= fit_length:
      return np.ones(fit_length)*-np.inf
    if smooth<0:
      return np.ones(fit_length)*-np.inf
    if rc1<0 or rc2<0 or rcfrac <0 or rcfrac > 1:
      return np.ones(fit_length)*-np.inf
    
    detector.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)
    detector.SetTemperature(temp)
    
    rad = np.cos(theta)* r
    z = np.sin(theta) * r
    
    if not detector.IsInDetector(rad, phi, z):
      return -np.inf * np.ones(fit_length)
    
    siggen_wf = detector.MakeSimWaveform(rad, phi, z, e, s, fit_length, h_smoothing=smooth )
    if siggen_wf is None:
      return np.ones(fit_length)*-np.inf
      
#    plt.ion()
#    plt.figure(14)
#    plt.clf()
#    plt.plot(siggen_wf)
#    plt.plot(waveform.windowedWf, color="r")
#
#    print "Waveform parameters: "
#    print "  (r,phi,z) = (%0.2f,%0.3f,%0.2f)" % (rad,phi,z)
#    print "  e = %0.3f" % e
#    print "  smooth = %0.3f" % smooth
#    print "  t0 = %0.3f" % s
#    value = raw_input('  --> Press q to quit, any other key to continue\n')
#    plt.ioff()


    return siggen_wf
    
  baseline_sim = Deterministic(eval = siggen_model,
              doc='siggen wf' ,
              name = 'siggen_model',
              parents = {'s': t0Est,
                      'r': radEst,'phi': phiEst,'theta': thetaEst,
                      'e':scaleEst,
                      'smooth':sigEst,
                      'b_over_a': b_over_a, 'c':c, 'd':d,
                      'rc1': rc1, 'rc2': rc2, 'rcfrac': rcfrac, 'temp': tempEst
                      },
              trace = False,
              plot=False)
  baseline_observed = Normal('baseline_observed', mu=baseline_sim, tau=sigToTau(waveform.baselineRMS*0.5773), observed= True, value= waveform.windowedWf )
  
  return locals()



def createWaveformModel(detector, waveform, startGuess):

  furthest_point = np.sqrt( detector.detector_radius**2 + detector.detector_length**2  )

  radEst =     TruncatedNormal('radEst',    mu=startGuess['radEst'],   a=0,   b=furthest_point, tau=sigToTau(2), value=startGuess['radEst']      )
  thetaEst =   TruncatedNormal('thetaEst',  mu=startGuess['thetaEst'], a=0,   b=np.pi/2,        tau=sigToTau(0.2), value=startGuess['thetaEst']         )
  
  phiEst =     Uniform('phiEst', lower=0,   upper=np.pi/4 ,                 value=startGuess['phiEst']      )
  scaleEst =   Normal('wfScale',     mu=startGuess['wfScale'],     tau=sigToTau(0.01*startGuess['wfScale']), value=startGuess['wfScale'])
  t0Est =      Normal('switchpoint', mu=startGuess['switchpoint'], tau=sigToTau(5.),                         value=startGuess['switchpoint'])
  sigEst=      Normal('sigma',       mu=startGuess['smooth'],      tau=sigToTau(3),                           value=startGuess['smooth'] )
  
  fit_length = waveform.wfLength
  
  def siggen_model(s, r, theta, phi, e, smooth):
  
    if s<0 or s>= fit_length:
      return np.ones(fit_length)*-np.inf
    if smooth<0:
      return np.ones(fit_length)*-np.inf
    
    rad = np.cos(theta)* r
    z = np.sin(theta) * r
    
    if not detector.IsInDetector(rad, phi, z):
      return -np.inf * np.ones(fit_length)
    
    siggen_wf = detector.MakeSimWaveform(rad, phi, z, e, s, fit_length, h_smoothing=smooth )
    if siggen_wf is None:
      return np.ones(fit_length)*-np.inf
      
#    plt.ion()
#    plt.figure(14)
#    plt.clf()
#    plt.plot(siggen_wf)
#    plt.plot(waveform.windowedWf, color="r")
#
#    print "Waveform parameters: "
#    print "  (r,phi,z) = (%0.2f,%0.3f,%0.2f)" % (rad,phi,z)
#    print "  e = %0.3f" % e
#    print "  smooth = %0.3f" % smooth
#    print "  t0 = %0.3f" % s
#    value = raw_input('  --> Press q to quit, any other key to continue\n')
#    plt.ioff()


    return siggen_wf
    
  baseline_sim = Deterministic(eval = siggen_model,
              doc='siggen wf' ,
              name = 'siggen_model',
              parents = {'s': t0Est,
                      'r': radEst,'phi': phiEst,'theta': thetaEst,
                      'e':scaleEst,
                      'smooth':sigEst
                      },
              trace = False,
              plot=False)
  baseline_observed = Normal('baseline_observed', mu=baseline_sim, tau=sigToTau(waveform.baselineRMS*0.5773), observed= True, value= waveform.windowedWf )
  
  return locals()

def CreateFullDetectorModel(detector, waveforms, startGuess, b_over_a0, c0, d0, rc0):
  
  n_waveforms = len(waveforms)
  sample_length = len(waveforms[0].windowedWf)
  
  #detector-wide params
  tempEst = TruncatedNormal('temp', mu=startGuess['temp'], tau=sigToTau(2.), value=startGuess['temp'], a=40, b=120)
  grad =  Uniform('grad', lower=detector.gradList[0], upper=detector.gradList[-1],    value=startGuess['grad'] )
  pcRad =  Uniform('pcRad', lower=detector.pcRadList[0], upper=detector.pcRadList[-1],value=startGuess['pcRad'] )
  pcLen = Uniform('pcLen', lower=detector.pcLenList[0], upper=detector.pcLenList[-1], value=startGuess['pcLen'] )
  
#  grad =  TruncatedNormal('grad', a=detector.gradList[0], b=detector.gradList[-1],    value=startGuess['grad'], mu=startGuess['grad'],tau=sigToTau(0.03) )
#  pcRad =  TruncatedNormal('pcRad', a=detector.pcRadList[0], b=detector.pcRadList[-1],value=startGuess['pcRad'], mu=startGuess['pcRad'],tau=sigToTau(0.2) )
#  pcLen = TruncatedNormal('pcLen', a=detector.pcLenList[0], b=detector.pcLenList[-1], value=startGuess['pcLen'], mu=startGuess['pcLen'],tau=sigToTau(0.2) )
  
  b_over_a =    Normal('b_over_a', mu=b_over_a0,         tau=sigToTau(.5), value=b_over_a0)
  c =    Normal('c', mu=c0,        tau=sigToTau(0.2), value=c0)
  d = Normal('d', mu=d0, tau=sigToTau(0.2), value=d0)
  rc = Normal('rc', mu=rc0, tau=sigToTau(5), value=rc0)
  
  #Make an array of priors for each waveform-specific parameter
  radiusArray = np.empty(n_waveforms, dtype=object)
  zArray = np.empty(n_waveforms, dtype=object)
  phiArray = np.empty(n_waveforms, dtype=object)
  scaleArray = np.empty(n_waveforms, dtype=object)
  t0Array = np.empty(n_waveforms, dtype=object)
  sigArray = np.empty(n_waveforms, dtype=object)
  
  for idx in range(n_waveforms):
    radiusArray[idx] =( TruncatedNormal('radEst_%d'%idx, mu=3, a=0,   b=detector.detector_radius, value=startGuess['radEst'][idx]       )  )
    zArray[idx] =(      TruncatedNormal('zEst_%d'%idx,   mu=3, a=0,   b=detector.detector_length, value=startGuess['zEst'][idx]         )  )
    phiArray[idx] =(    Uniform('phiEst_%d'%idx, lower=0,   upper=np.pi/4 ,                 value=startGuess['phiEst'][idx]       )  )
    scaleArray[idx] =(  Normal('wfScale_%d'%idx,     mu=startGuess['wfScale'][idx],     tau=sigToTau(0.01*startGuess['wfScale'][idx]), value=startGuess['wfScale'][idx]) )
    t0Array[idx] =(     Normal('switchpoint_%d'%idx, mu=startGuess['switchpoint'][idx], tau=sigToTau(5.),                              value=startGuess['switchpoint'][idx]))
    sigArray[idx] =(    Normal('sigma_%d'%idx,       mu=startGuess['smooth'][idx],      tau=sigToTau(3), value=startGuess['smooth'][idx] ))
  
  #This is a deterministic (implicitly?  is this a problem?)
  def siggen_model(s, rad, phi, z, e, smooth, temp, b_over_a, c, d, rc, grad, pc_rad, pc_len, fit_length):
  
    if s<0 or s>= fit_length:
      return np.ones(fit_length)*-np.inf
#    if smooth<0:
#      return np.ones(fit_length)*-np.inf
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
    
    detector.SetTransferFunction(b_over_a, c, d, rc)
    detector.SetTemperature(temp)

    if detector.pcRad != pc_rad or detector.pcLen != pc_len or detector.impurityGrad != grad:
      detector.SetFields(pc_rad, pc_len, grad)
    
    siggen_wf = detector.MakeSimWaveform(rad, phi, z, e, s, fit_length, h_smoothing=None )
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
                          'smooth':sigArray[i],
                          'temp': tempEst,
                          'b_over_a':b_over_a, 'c':c, 'd':d, 'rc':rc,
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

