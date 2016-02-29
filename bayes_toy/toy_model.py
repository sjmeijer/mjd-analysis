import numpy as np
from pymc import DiscreteUniform, Uniform, Normal, HalfNormal, deterministic, exp


def createSignalModel(data):
  #set up your model parameters
  switchpoint = DiscreteUniform('switchpoint', lower=0, upper=len(data))
  early_sigma = HalfNormal('early_sigma', tau=sigToTau(1))
  late_sigma = HalfNormal('late_sigma', tau=sigToTau(1))
  early_mu = Normal('early_mu', mu=.5, tau=sigToTau(1))
  late_mu = Normal('late_mu', mu=.5, tau=sigToTau(1))
  

  #set up the model for uncertainty (ie, the noise) and the signal (ie, the step function)

  ############################
  @deterministic(plot=False, name="test")
  def uncertainty_model(s=switchpoint, n=early_sigma, e=late_sigma):
    #Concatenate Uncertainty sigmas (or taus or whatever) around t0
    s = np.around(s)
    out = np.empty(len(data))
    out[:s] = n
    out[s:] = e
    return out
  
  ############################
  @deterministic
  def tau(eps=uncertainty_model):
    #pymc uses this tau parameter instead of sigma to model a gaussian.  its annoying.
    return np.power(eps, -2)
  
  ############################
  @deterministic(plot=False, name="siggenmodel")
  def signal_model(s=switchpoint, e=early_mu, l=late_mu):
    #makes the step function using the means
    out = np.zeros(len(data))
    out[:s] = e
    out[s:] = l
    return out
  
  ############################

  #Full model: normally distributed noise around a step function
  baseline_observed = Normal("baseline_observed", mu=signal_model, tau=tau, value=data, observed= True )
  return locals()

def sigToTau(sig):
  tau = np.power(sig, -2)
  return tau