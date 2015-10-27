#!/usr/local/bin/python
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
import numpy as np
from scipy import ndimage
from pymc import DiscreteUniform, Uniform, Normal, HalfNormal, deterministic, exp
import matplotlib.pyplot as plt


#siggen_conf = "P42661C_autogen_final.conf"
siggen_conf = "P42661C_lowgrad.conf" #works better
rcIntTimeConstant = 48.7 * CLHEP.ns
pzCorrTimeConstant = 72*CLHEP.us
gaussianSmoothing = 1.6
detZ = np.floor(41.5)
detRad = np.floor(35.41)


siggenInst = GATSiggenInstance(siggen_conf)
rcint = MGWFRCIntegration()
rcint.SetTimeConstant(rcIntTimeConstant)
rcdiff = MGWFRCDifferentiation()
rcdiff.SetTimeConstant(pzCorrTimeConstant)


"""
    Models for ppc response
    """

doPlots = 0

if doPlots:
  plt.ion()
  fig = plt.figure(10)

def createSignalModelSiggen(data, t0_guess, energy_guess, baseline_guess):

  rAvg = detRad
  zAvg = detZ/2
  phiAvg = np.pi/8
  
  print "t0 guess is %d" % t0_guess
  
  noise_sigma = HalfNormal('noise_sigma', tau=.01)
  slowness_sigma = HalfNormal('slowness_sigma', tau=.01)
  
  switchpoint = Normal('switchpoint', mu=t0_guess, tau=.005)
  wfScale = Normal('wfScale', mu=energy_guess, tau=sigToTau(.05*energy_guess))
  
#  baselineB = Normal('baselineB', mu=baseline_guess, tau=10)
#  baselineM = Normal('baselineM', mu=0, tau=1000000)

  print "switchpoint is %d" % switchpoint
  print "wfScale is %f" % wfScale


  ############################
  @deterministic
  def tau(eps=noise_sigma):
    return np.power(eps, -2)
  
  ############################
  @deterministic(plot=False, name="siggenmodel")
  def siggen_model(s=switchpoint, rad = rAvg, phi=phiAvg, z = zAvg, e=wfScale, sig=slowness_sigma ):
    out = np.zeros(len(data))
#    out = np.multiply(baselineM, out)
#    out = np.add(baselineB, out)

    siggen_data = findSiggenWaveform(rad,phi,z)
    
    if e<0:
      e=0
    
    siggen_data *= e
    if s<0:
      s =0
    if s>len(data):
      s=len(data)
    s = np.around(s)
    
    out[s:] += siggen_data[0:(len(data) - s)]
    
    out = ndimage.filters.gaussian_filter1d(out, sig)
    
    
    if doPlots:
      plt.figure(fig.number)
      plt.clf()
      plt.plot( data  ,color="red" )
      plt.plot( out  ,color="blue" )
      value = raw_input('  --> Press q to quit, any other key to continue\n')
      if value == 'q':
        exit(1)

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

def findSiggenWaveform(r,phi,z, gaussianSmoothing=0):

  x = r * np.sin(phi)
  y = r * np.cos(phi)

  hitPosition = TVector3(x, y, z);
  sigWf = MGTWaveform();
  calcFlag = siggenInst.CalculateWaveform(hitPosition, sigWf, 1);
  
  if calcFlag == -1:
    print "Point out of crystal alert! (%0.3f,%0.3f,%0.3f)" % (r,phi,z)
    return np.ones(8000)
  
  rcint.TransformInPlace(sigWf)
  rcdiff.TransformInPlace(sigWf)
  siggen_data = sigWf.GetVectorData()

  if gaussianSmoothing>0:
    siggen_data = ndimage.filters.gaussian_filter1d(siggen_data, gaussianSmoothing)
  
  siggen_data /= np.amax(siggen_data)

#  print siggen_data

  return siggen_data