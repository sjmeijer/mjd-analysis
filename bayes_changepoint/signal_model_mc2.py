#!/usr/local/bin/python
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
import numpy as np
from scipy import ndimage
from pymc import DiscreteUniform, Uniform, Normal, HalfNormal, deterministic, exp
import matplotlib.pyplot as plt



#for lookup table
#dt_array = np.load("siggen_lookup.npy")


siggen_conf = "P42661C_autogen_final.conf"
rcIntTimeConstant = 48.7 * CLHEP.ns
pzCorrTimeConstant = 74.1*CLHEP.us
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

def createSignalModelSiggen(data, t0_guess, energy_guess):


#  switchpoint = t0_gues

  noise_sigma = HalfNormal('noise_sigma', tau=sigToTau(.01))
  exp_sigma = HalfNormal('exp_sigma', tau=sigToTau(.05))
  
  radEst = Uniform('radEst', lower=0, upper=detRad)
  zEst = Uniform('zEst', lower=5, upper=detZ)
  phiEst = Uniform('phiEst', lower=0, upper=np.pi/4)
  
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
    
    siggen_data = findSiggenWaveform(rad,phi,z)
    
    siggen_data *= e
    s = np.around(s)
    out[s:] = siggen_data[0:(len(data) - s)]
    
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


def createSignalModelExponential(data):
  """
    Toy model that treats the first ~10% of the waveform as an exponential.  Does a good job of finding the start time (t_0)
    Since I made this as a toy, its super brittle.  Waveform must be normalized
  """
  print "Creating model"
  switchpoint = DiscreteUniform('switchpoint', lower=0, upper=len(data))
  
  noise_sigma = HalfNormal('noise_sigma', tau=sigToTau(.01))
  exp_sigma = HalfNormal('exp_sigma', tau=sigToTau(.05))
  
  #Modeling these parameters this way is why wf needs to be normalized
  exp_rate = Uniform('exp_rate', lower=0, upper=.1)
  exp_scale = Uniform('exp_scale', lower=0, upper=.1)
  
  timestamp = np.arange(0, len(data), dtype=np.float)
  
  @deterministic(plot=False, name="test")
  def uncertainty_model(s=switchpoint, n=noise_sigma, e=exp_sigma):
    ''' Concatenate Poisson means '''
    out = np.empty(len(data))
    out[:s] = n
    out[s:] = e
    return out
  
  @deterministic
  def tau(eps=uncertainty_model):
    return np.power(eps, -2)
  
##  @deterministic(plot=False, name="test2")
##  def adjusted_scale(s=switchpoint, s1=exp_scale):
##    out = np.empty(len(data))
##    out[:s] = s1
##    out[s:] = s1
##    return out
#
#  scale_param = adjusted_scale(switchpoint, exp_scale)

  @deterministic(plot=False)
  def baseline_model(s=switchpoint, r=exp_rate, scale=exp_scale):
    out = np.zeros(len(data))
    out[s:] = scale * ( np.exp(r * (timestamp[s:] - s)) - 1.)
    
#    plt.figure(fig.number)
#    plt.clf()
#    plt.plot(out ,color="blue" )
#    plt.plot(data ,color="red" )
#    value = raw_input('  --> Press q to quit, any other key to continue\n')

    return out

  baseline_observed = Normal("baseline_observed", mu=baseline_model, tau=tau, value=data, observed= True )
  return locals()

################################################################################################################################



def createSignalModelLinear(data):
  """
    Toy model that treats the first ~10% of the waveform as an exponential.  Does a good job of finding the start time (t_0)
    Since I made this as a toy, its super brittle.  Waveform must be normalized
  """
  print "Creating model"
  switchpoint = DiscreteUniform('switchpoint', lower=0, upper=len(data))
  
  noise_sigma = HalfNormal('noise_sigma', tau=sigToTau(.01))
  exp_sigma = HalfNormal('exp_sigma', tau=sigToTau(.05))
  
  lin_scale = Uniform('lin_scale', lower=0, upper=.01)
  
  timestamp = np.arange(0, len(data), dtype=np.float)
  
  @deterministic(plot=False, name="test")
  def uncertainty_model(s=switchpoint, n=noise_sigma, e=exp_sigma):
    ''' Concatenate Poisson means '''
    out = np.empty(len(data))
    out[:s] = n
    out[s:] = e
    return out
  
  @deterministic
  def tau(eps=uncertainty_model):
    return np.power(eps, -2)
  

  @deterministic(plot=False)
  def baseline_model(s=switchpoint, scale=lin_scale):
    out = np.zeros(len(data))
    out[s:] = scale * (timestamp[s:] - s)
    
#    plt.figure(fig.number)
#    plt.clf()
#    plt.plot(out ,color="blue" )
#    plt.plot(data ,color="red" )
#    value = raw_input('  --> Press q to quit, any other key to continue\n')

    return out


  baseline_observed = Normal("baseline_observed", mu=baseline_model, tau=tau, value=data, observed= True )
  return locals()
################################################################################################################################

def sigToTau(sig):
  tau = np.power(sig, -2)
#  print "tau is %f" % tau
  return tau
################################################################################################################################

def findSiggenWaveform(r,phi,z):

  x = r * np.sin(phi)
  y = r * np.cos(phi)

  hitPosition = TVector3(x, y, z);
  sigWf = MGTWaveform();
  calcFlag = siggenInst.CalculateWaveform(hitPosition, sigWf, 1);
  
  if calcFlag == -1:
    return np.ones(8000)
  
  rcint.TransformInPlace(sigWf)
  rcdiff.TransformInPlace(sigWf)
  siggen_data = sigWf.GetVectorData()
  siggen_data = ndimage.filters.gaussian_filter1d(siggen_data, gaussianSmoothing)
  
  siggen_data /= np.amax(siggen_data)

  return siggen_data