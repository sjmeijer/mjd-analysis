#!/usr/local/bin/python

import numpy as np
from pymc import DiscreteUniform, Uniform, Normal, HalfNormal, deterministic, exp

"""
    Models for ppc response
    """

import matplotlib.pyplot as plt

doPlots = 0
if doPlots:
  plt.ion()
  fig = plt.figure(11)


def CreateFullDetectorModel(detector, data, t0_guess, energy_guess, rcint_guess, rcdiff_guess):


  z_min = 0 #temporary hack to keep you off the taper

#  switchpoint = t0_gues

  #This is friggin ridiculous
  #noise_sigma = HalfNormal('baseline_sigma', tau=sigToTau(.01))
  siggen_sigma = HalfNormal('siggen_sigma', tau=sigToTau(.01))
  siggen_tau = np.power(siggen_sigma, -2)
  
  radEst = Uniform('radEst', lower=0,   upper=np.floor(detector.radius))
  zEst = Uniform('zEst', lower=z_min,   upper=np.floor(detector.length))
  phiEst = Uniform('phiEst', lower=0,   upper=np.pi/4)
  
#  print "rc int guess is %f" % rcint_guess
#  print "rc diff guess is %f" % rcdiff_guess

  rc_int = Normal('rc_int', mu=rcint_guess, tau=sigToTau(1.)) #should be in ns
  rc_diff = Normal('rc_diff', mu=rcdiff_guess, tau=sigToTau(100.)) #should
  
  print "z value is %f" % zEst.value
  
  
  if zEst.value < 5:
    "setting initial z value guess safely above 5 mm"
    zEst.value = 5

  switchpoint = Normal('switchpoint', mu=t0_guess, tau=sigToTau(1))
  wfScale = Normal('wfScale', mu=energy_guess, tau=sigToTau(.01*energy_guess))
  
  print "switchpoint is %d" % switchpoint
  print "wfScale is %f" % wfScale
  
  ############################
  @deterministic(plot=False, name="siggenmodel")
  def siggen_model(s=switchpoint, rad = radEst, phi=phiEst, z = zEst, e=wfScale, rise_time = rc_int, fall_time = rc_diff ):
  
#    print "rc diff is %0.3f" % fall_time

    out = np.zeros(len(data))
    
    #Let the rounding happen organically in the detector model...
#    siggen_data = detector.GetWaveformByPosition(rad,phi,z)
    detector.preampRiseTime = rise_time
    detector.preampFallTime = fall_time
    
    siggen_wf= detector.GetSiggenWaveform(rad, phi, z, energy=2600)
#    print siggen_wf
    if siggen_wf is None:
      return -np.inf
    if np.amax(siggen_wf) == 0:
      print "wtf is even happening here?"
      return -np.inf
    siggen_data = detector.ProcessWaveform(siggen_wf)
    siggen_data = siggen_data[500::]
    
    siggen_data *= e
    s = np.around(s)
    
    if s <0:
      #print "S is zero dude."
      s = 0

    
    out[s:] = siggen_data[0:(len(data) - s)]

    if doPlots:
      if np.isnan(siggen_data[0]):
        return out
      plt.figure(11)
      plt.clf()
      plt.plot(data, color="red")
      plt.plot(out, color="blue")
      print "r: %0.1f, z: %0.1f" % (rad, z)
      value = raw_input('  --> Press q to quit, any other key to continue\n')
      if value == 'q':
        exit(1)

    return out
  
  ############################

  baseline_observed = Normal("baseline_observed", mu=siggen_model, tau=siggen_tau, value=data, observed= True )
  return locals()

################################################################################################################################

def sigToTau(sig):
  tau = np.power(sig, -2)
#  print "tau is %f" % tau
  return tau
################################################################################################################################
