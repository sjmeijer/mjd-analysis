#!/usr/local/bin/python
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
import numpy as np
from scipy import ndimage
from pymc import DiscreteUniform, Uniform, Normal, HalfNormal, deterministic, exp
import matplotlib.pyplot as plt


#siggen_conf = "P42661C_autogen_final.conf"
#siggen_conf = "P42661C_lowgrad.conf" #works better

siggen_conf = "malbek.conf"
rcIntTimeConstant = 50 * CLHEP.ns
pzCorrTimeConstant = 69.88*CLHEP.us
gaussianSmoothing = 1.6
detZ = np.floor(30.)
detRad = np.floor(30.3)
signalLength = 2000


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

def createSignalModelSiggen(data, t0_guess, energy_guess, noise_sigma_guess, baseline_guess):


  verbose = 0

  slowness_sigma = HalfNormal('slowness_sigma', tau=.001)
  
  switchpoint = Normal('switchpoint', mu=t0_guess, tau=.001)
  wfScale = Normal('wfScale', mu=energy_guess, tau=sigToTau(.25*energy_guess))
  
#  baselineB = Normal('baselineB', mu=baseline_guess, tau=10)
#  baselineM = Normal('baselineM', mu=0, tau=1000000)

  #### Baseline noise Model
  #  noise_sigma = HalfNormal('noise_sigma', tau=.01)
#  @deterministic
#  def tau(eps=noise_sigma):
#    return np.power(eps, -2)

  noise_tau = np.power(noise_sigma_guess, -2)
  
  if verbose:
    print "t0 guess is %d" % t0_guess
    print "noise sigma guess is %0.2f" % noise_sigma_guess
    print "t0 init is %d" % switchpoint
    print "wfScale init is %f" % wfScale
    print "noise tau is %0.2e" % noise_tau


  ############################
  @deterministic(plot=False, name="siggenmodel")
  def siggen_model(s=switchpoint, e=wfScale, sig=slowness_sigma ):
    out = np.zeros(len(data))
#    out = np.multiply(baselineM, out)
#    out = np.add(baselineB, out)

    siggen_data = np.copy(siggen_wf)#findSiggenWaveform(rad,phi,z)
    
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

  baseline_observed = Normal("baseline_observed", mu=siggen_model, tau=noise_tau, value=data, observed= True )
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

  #print "x, y, z is (%0.1f, %0.1f, %0.1f)" % (x,y,z)

  hitPosition = TVector3(x, y, z);
  sigWf = MGTWaveform();
  calcFlag = siggenInst.CalculateWaveform(hitPosition, sigWf, 1);
  
  if calcFlag == 0:
    siggen_data = sigWf.GetVectorData()
    siggen_data = np.multiply(siggen_data, 1)
  
    print siggen_data
    print "Point out of crystal alert! (%0.3f,%0.3f,%0.3f)" % (r,phi,z)
    exit(0)
  
  rcint.TransformInPlace(sigWf)
  rcdiff.TransformInPlace(sigWf)
  siggen_data = sigWf.GetVectorData()
  
  siggen_data = np.lib.pad(siggen_data, (0,signalLength-len(siggen_data)), 'edge')
  
  if gaussianSmoothing>0:
    siggen_data = ndimage.filters.gaussian_filter1d(siggen_data, gaussianSmoothing)
  
  siggen_data /= np.amax(siggen_data)

#  print siggen_data

  return siggen_data

#calculate a "characteristic" detector wf we'll use for fits
siggen_wf =   findSiggenWaveform(detRad,np.pi/8,detZ/2.)