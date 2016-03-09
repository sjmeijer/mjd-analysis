#!/usr/local/bin/python

import numpy as np
from pymc import DiscreteUniform, Uniform, Normal, HalfNormal, deterministic, exp

"""
    Models for ppc response
    """


def CreateFullDetectorModel(detector, data, t0_guess, energy_guess):

  z_min = 5 #temporary hack to keep you off the taper

#  switchpoint = t0_gues

  #This is friggin ridiculous
  noise_sigma = HalfNormal('baseline_sigma', tau=sigToTau(.01))
  exp_sigma = HalfNormal('siggen_sigma', tau=sigToTau(.05))
  
  radEst = Uniform('radEst', lower=0,   upper=detector.radius)
  zEst = Uniform('zEst', lower=z_min,   upper=detector.length)
  phiEst = Uniform('phiEst', lower=0,   upper=np.pi/4)
  
  switchpoint = Normal('switchpoint', mu=t0_guess, tau=sigToTau(1))
  wfScale = Normal('wfScale', mu=energy_guess, tau=sigToTau(.01*energy_guess))
  
  print "switchpoint is %d" % switchpoint
  print "wfScale is %f" % wfScale

  ############################
  @deterministic(plot=False, name="test")
  def uncertainty_model(s=switchpoint, n=noise_sigma, e=exp_sigma):
    ''' Concatenate Uncertainty sigmas (or taus or whatever) '''
    
    s = np.around(s)
    out = np.empty(len(data))
    out[:s] = n
    out[s:] = e
    return out
  
  ############################
  @deterministic
  def tau(eps=uncertainty_model):
    return np.power(eps, -2)
  
  ############################
  @deterministic(plot=False, name="siggenmodel")
  def siggen_model(s=switchpoint, rad = radEst, phi=phiEst, z = zEst, e=wfScale):
    out = np.zeros(len(data))
    
    #Let the rounding happen organically in the detector model...
    siggen_data = detector.GetWaveformByPosition(rad,phi,z)
    
    siggen_data *= e
    s = np.around(s)
    out[s:] = siggen_data[0:(len(data) - s)]

    return out
  
  ############################

  baseline_observed = Normal("baseline_observed", mu=siggen_model, tau=tau, value=data, observed= True )
  return locals()

################################################################################################################################

def sigToTau(sig):
  tau = np.power(sig, -2)
#  print "tau is %f" % tau
  return tau
################################################################################################################################
