#!/usr/local/bin/python

import numpy as np
from pymc3 import *#DiscreteUniform, Exponential, Poisson, Uniform, Normal, find_MAP, HalfNormal, switch, Model, Deterministic, exp, Metropolis, traceplot, sample, summary
import theano.tensor as T
from theano.compile.ops import as_op

"""
    Models for ppc response
    """

import matplotlib.pyplot as plt

doPlots = 0
if doPlots:
  plt.ion()
  fig = plt.figure(11)


def CreateCheapDetectorModel(detector, data, t0_guess, energy_guess):
  with Model() as signal_model:

    switchpoint = DiscreteUniform('switchpoint', lower = t0_guess-3, upper=t0_guess+3, testval=t0_guess)
    wfScale = Normal('wfScale', mu=energy_guess, sd=.01*energy_guess)

    timestamp = np.arange(0, len(data))

    siggen_wf= detector.GetSiggenWaveform(10, 0, 10, energy=2600)
    
    print switchpoint.eval()
    print "the value is %d" % switchpoint.eval()
    
    baseline_model = switch(switchpoint>=timestamp, 0, siggen_wf[timestamp-switchpoint])


#    @theano.Op(itypes=[T.dscalar, T.dscalar], otypes=[T.dvector])
#    def siggen_model(s,e):
#      siggen_wf= detector.GetSiggenWaveform(10, 0, 10, energy=2600)
#    
#      out = np.zeros_like(data)
#      s = np.around(s)
#      out[s:] = siggen_wf[0:(len(data) - s)]
#      return pm.Deterministicout
#
#    baseline_model = siggen_model(switchpoint, wfScale)

    baseline_observed = Normal("baseline_observed", mu=baseline_model, sd=10., observed= data )
  return signal_model

def CreateFullDetectorModel(detectorList, data, t0_guess, energy_guess, rcint_guess, rcdiff_guess):
  
  detector = detectorList[0,0]
  with Model() as signal_model:
  
    radEst = Uniform('radEst', lower=0,   upper=np.floor(detector.radius))
    zEst = Uniform('zEst', lower=0,   upper=np.floor(detector.length))
    phiEst = Uniform('phiEst', lower=0,   upper=np.pi/4)
    
    tempEst = Uniform('temp', lower=40,   upper=120, testval=80)
    
    gradIdx = DiscreteUniform('gradIdx', lower = 0, upper= detectorList.shape[0]-1, testval=detectorList.shape[0]/2)
    pcRadIdx = DiscreteUniform('pcRadIdx', lower = 0, upper= detectorList.shape[1]-1, testval=detectorList.shape[1]/2)

    

#    if zEst < 5:
#      "setting initial z value guess safely above 5 mm"
#      zEst = 5

    #t0 = Normal('switchpoint', mu=t0_guess, sd=1.)
    t0 = DiscreteUniform('switchpoint', lower = t0_guess-10, upper=t0_guess+10, testval=t0_guess)
    wfScale = Normal('wfScale', mu=energy_guess, sd=.01*energy_guess)

    rc_int = Normal('rc_int', mu=rcint_guess, sd=1.) #should be in ns
    rc_diff = Normal('rc_diff', mu=rcdiff_guess, sd=100.) #also in ns
    
    gaussSmooth = HalfNormal('gaussSmooth', sd=3.)
    
    
#    rc_diff_short = Normal('rc_diff_short', mu=detector.preampFalltimeShort, sd=1.) #also in ns
#    fall_time_short_frac = Uniform('rc_diff_short_frac', lower=0,   upper=1, testval=detector.preampFalltimeShortFraction)

#    @as_op(itypes=[T.lscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.lscalar], otypes=[T.dvector])
#    def siggen_model(s, rad, phi, z, e, temp, rise_time, fall_time, fall_time_short, fall_time_short_frac, detectorListIdx):
    @as_op(itypes=[T.lscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.lscalar, T.lscalar], otypes=[T.dvector])
    def siggen_model(s, rad, phi, z, e, temp, rise_time, fall_time, gaussSmooth, gradIdx, pcRadIdx):

      #    print "rc diff is %0.3f" % fall_time
      
#      print "grad idx is %d" % detectorListIdx

      if (gradIdx > detectorList.shape[0]-1) or (pcRadIdx > detectorList.shape[1]-1) :
        return np.ones_like(data)*-1.

      detector = detectorList[gradIdx, pcRadIdx]
      
      out = np.zeros_like(data)
      
      detector.preampRiseTime = rise_time
      detector.preampFallTime = fall_time
      detector.gaussian_smoothing = gaussSmooth
      detector.SetTemperature(temp)
      
      siggen_wf= detector.GetSiggenWaveform(rad, phi, z, energy=2600)
  #    print siggen_wf
      if siggen_wf is None:
        return np.ones_like(data)*-1.
      if np.amax(siggen_wf) == 0:
        print "wtf is even happening here?"
        return np.ones_like(data)*-1.
      
      siggen_data = detector.ProcessWaveform(siggen_wf)
      siggen_data = siggen_data[500::]
      
      siggen_data = siggen_data*e
      
#      s = np.around(s)
#      if s <0:
#        #print "S is zero dude."
#        s = 0

      out[s:] = siggen_data[0:(len(data) - s)]

      return out

#    baseline_model = siggen_model(t0, radEst, phiEst, zEst, wfScale, tempEst, rc_int, rc_diff, rc_diff_short, fall_time_short_frac, detectorListIdx)
    baseline_model = siggen_model(t0, radEst, phiEst, zEst, wfScale, tempEst, rc_int, rc_diff,  gaussSmooth, gradIdx, pcRadIdx)
    baseline_observed = Normal("baseline_observed", mu=baseline_model, sd=10., observed= data )
  return signal_model

def CreateFullDetectorModelLookup(detectorList, data, t0_guess, energy_guess, rcint_guess, rcdiff_guess):
  
  #cant do termperature right now
  
  detector = detectorList[0]
  with Model() as signal_model:
  
    shape = detector.lookupTable.shape
    
    subNumber = 2 # wtf man
    
#    print "lookup table shape is " + str(shape)

    radIdx = DiscreteUniform('radIdx', lower=0,   upper = shape[0]-subNumber )
    phiIdx = DiscreteUniform('phiIdx', lower=0,   upper= shape[1]-subNumber)
    zIdx = DiscreteUniform('zIdx', lower=0,   upper= shape[2]-subNumber)
    
#    tempEst = Uniform('temp', lower=40,   upper=120, testval=80)

    detectorListIdx = DiscreteUniform('gradIdx', lower = 0, upper= len(detectorList)-subNumber, testval=len(detectorList)/2)
    

    t0 = DiscreteUniform('switchpoint', lower = t0_guess-10, upper=t0_guess+10, testval=t0_guess)
    wfScale = Normal('wfScale', mu=energy_guess, sd=.01*energy_guess)

    rc_int = Normal('rc_int', mu=rcint_guess, sd=1.) #should be in ns
    rc_diff = Normal('rc_diff', mu=rcdiff_guess, sd=100.) #also in ns
#    rc_diff_short = Normal('rc_diff_short', mu=detector.preampFalltimeShort, sd=1.) #also in ns
#    fall_time_short_frac = Uniform('rc_diff_short_frac', lower=0,   upper=1, testval=detector.preampFalltimeShortFraction)

#    @as_op(itypes=[T.lscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.lscalar], otypes=[T.dvector])
#    def siggen_model(s, rad, phi, z, e, temp, rise_time, fall_time, fall_time_short, fall_time_short_frac, detectorListIdx):
    @as_op(itypes=[T.lscalar, T.lscalar, T.lscalar, T.lscalar, T.dscalar, T.dscalar, T.dscalar, T.lscalar], otypes=[T.dvector])
    def siggen_model(s, rad, phi, z, e, rise_time, fall_time, detectorListIdx):

      #    print "rc diff is %0.3f" % fall_time
      
      if (phi >  shape[1]-1) or (detectorListIdx > len(detectorList)-1) or (rad > shape[0]-1) or (z > shape[2]-1) :
        return np.ones_like(data)*-1.

      
#      print "grad idx is %d" % detectorListIdx
      detector = detectorList[detectorListIdx]
      
      out = np.zeros_like(data)
      
      detector.preampRiseTime = rise_time
      detector.preampFallTime = fall_time
#      detector.SetTemperature(temp)

      siggen_wf= detector.GetWaveformByIndex(rad, phi, z)
  #    print siggen_wf
      if siggen_wf is None:
        return np.ones_like(data)*-1.
      if np.amax(siggen_wf) == 0:
        print "wtf is even happening here?"
        return np.ones_like(data)*-1.
      
      siggen_data = detector.ProcessWaveform(siggen_wf)
      siggen_data = siggen_data[500::]
      
      siggen_data = siggen_data*e
      
#      s = np.around(s)
#      if s <0:
#        #print "S is zero dude."
#        s = 0

      out[s:] = siggen_data[0:(len(data) - s)]

      return out

#    baseline_model = siggen_model(t0, radEst, phiEst, zEst, wfScale, tempEst, rc_int, rc_diff, rc_diff_short, fall_time_short_frac, detectorListIdx)
    baseline_model = siggen_model(t0, radIdx, phiIdx, zIdx, wfScale, rc_int, rc_diff,  detectorListIdx)
    baseline_observed = Normal("baseline_observed", mu=baseline_model, sd=10., observed= data )
  return signal_model