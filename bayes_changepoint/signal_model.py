#!/usr/local/bin/python

import numpy as np

from pymc3 import DiscreteUniform, Exponential, Poisson, Uniform, Normal, find_MAP, HalfNormal, switch, Model, Deterministic, exp, Metropolis, traceplot, sample, summary
import theano.tensor as T 
from theano.compile.ops import as_op


dt_array = np.load("siggen_lookup.npy")


"""
    Models for ppc response
    """

def createSignalModelExponential(data):
  """
    Toy model that treats the first ~10% of the waveform as an exponential.  Does a good job of finding the start time (t_0)
    Since I made this as a toy, its super brittle.  Waveform must be normalized
  """
  with Model() as signal_model:
    switchpoint = Uniform('switchpoint', lower=0, upper=len(data), testval=len(data)/2)
    
    noise_sigma = HalfNormal('noise_sigma', sd=1.)
    
    #Modeling these parameters this way is why wf needs to be normalized
    exp_rate = Uniform('exp_rate', lower=0, upper=.5, testval = 0.05)
    exp_scale = Uniform('exp_scale', lower=0, upper=.5, testval = 0.1)
    
    timestamp = np.arange(0, len(data), dtype=np.float)
    
    rate = switch(switchpoint >= timestamp, 0, exp_rate)
    
    baseline_model = Deterministic('baseline_model', exp_scale * (exp( (timestamp-switchpoint)*rate)-1.) )
    
    baseline_observed = Normal("baseline_observed", mu=baseline_model, sd=noise_sigma, observed= data )
  return signal_model





def createSignalModelWithLookup(data, wfMax):
  """
    Uses a lookup table to avoid having to call siggen.  Lookup locations are along a one-dimensional line from PC to the detector corner.  See generate_siggen_lookup.py
    
    wfMax: maximum of the input signal.  Used as a prior for the for scaling of the simulated pulse
    
  """

  with Model() as signal_model:
    
    switchpoint = DiscreteUniform('switchpoint', lower=0, upper=len(data))
    noise_sigma = HalfNormal('noise_sigma', sd=1.)
    siggen_sigma = HalfNormal('siggen_sigma', sd=10.)
    
    
    
    timestamp = np.arange(0, len(data), dtype=np.int)

    uncertainty_model = switch(switchpoint >= timestamp, noise_sigma, siggen_sigma)
    
    wf_scale = Normal('wf_scale', sd=10., mu=wfMax)
    
    detRad = np.floor(35.41)
    detZ = np.floor(41.5)
    
    dtEstimate = DiscreteUniform('dtEstimate', lower=0, upper=99  )

    
  #          radiusEstimate = DiscreteUniform('radiusEstimate', lower=0, upper=35  )
  #          zEstimate =      DiscreteUniform('zEstimate', lower=0, upper=41)

    
    
    @as_op(itypes=[T.lscalar, T.lscalar, T.dscalar], otypes=[T.dvector])
    def siggen_model_dt(switchpoint, dtEstimate, wf_scale):
      siggen_out = dt_array[dtEstimate, :]
      siggen_out *= wf_scale

      T.clip(dtEstimate, 0, 99) #THIS IS A DISASTER. NEED to find a better way to handle this

      out = np.zeros(len(data))
      out[switchpoint:] = siggen_out[0:(len(data) - switchpoint)]
      
  #            print "length of out is %d" % len(out)
      return out
    
    @as_op(itypes=[T.lscalar, T.lscalar, T.lscalar], otypes=[T.dvector])
    def siggen_model(switchpoint, r, z):
      siggen_out = findSiggenWaveform(0,r,z,np.amax(np_data))
      out = np.zeros(len(data))
      out[switchpoint:] = siggen_out[0:(len(data) - switchpoint)]
      
      return out
    
    
  #          print "length of data is %d" % len(data)

  #          @as_op(itypes=[T.lscalar, T.dscalar, T.dscalar], otypes=[T.dvector])
  #          
  #          def crazy_modulo3(switchpoint, exp_scale, exp_rate):
  #            out = np.zeros(len(data))
  #            out[switchpoint:] = exp_scale * (np.exp( exp_rate * (timestamp[switchpoint:] - switchpoint))-1.)
  #            return out

    
    #baseline_model = Deterministic('baseline_model', exp_scale * (exp( (timestamp-switchpoint)*rate)-1.) )
    
  #          baseline_model = siggen_model(switchpoint, radiusEstimate, zEstimate)
    baseline_model_dt = siggen_model_dt(switchpoint, dtEstimate, wf_scale)
    
    
    baseline_observed = Normal("baseline_observed", mu=baseline_model_dt, sd=uncertainty_model, observed= data )

  return signal_model

#def createSignalModelDynamic(data, wfMax):
#  """
#    Calls siggen in real time
#    
#  """
#
#  with Model() as signal_model:
#    
#    switchpoint = DiscreteUniform('switchpoint', lower=0, upper=len(data))
#    noise_sigma = HalfNormal('noise_sigma', sd=1.)
#    siggen_sigma = HalfNormal('siggen_sigma', sd=10.)
#    
#    timestamp = np.arange(0, len(data), dtype=np.int)
#
#    uncertainty_model = switch(switchpoint >= timestamp, noise_sigma, siggen_sigma)
#    
#    detRad = np.floor(35.41)
#    detZ = np.floor(41.5)
#    
#    dtEstimate = DiscreteUniform('dtEstimate', lower=0, upper=99  )
#
#    
#  #          radiusEstimate = DiscreteUniform('radiusEstimate', lower=0, upper=35  )
#  #          zEstimate =      DiscreteUniform('zEstimate', lower=0, upper=41)
#
#    
#    
#    @as_op(itypes=[T.lscalar, T.lscalar], otypes=[T.dvector])
#    def siggen_model_dt(switchpoint, dtEstimate):
#      siggen_out = dt_array[dtEstimate, :]
#      siggen_out *= wfMax
#
#      T.clip(dtEstimate, 0, 99) #THIS IS A DISASTER. NEED to find a better way to handle this
#
#      out = np.zeros(len(data))
#      out[switchpoint:] = siggen_out[0:(len(data) - switchpoint)]
#      
#  #            print "length of out is %d" % len(out)
#      return out
#    
#    @as_op(itypes=[T.lscalar, T.lscalar, T.lscalar], otypes=[T.dvector])
#    def siggen_model(switchpoint, r, z):
#      siggen_out = findSiggenWaveform(0,r,z,np.amax(np_data))
#      out = np.zeros(len(data))
#      out[switchpoint:] = siggen_out[0:(len(data) - switchpoint)]
#      
#      return out
#    
#    
#  #          print "length of data is %d" % len(data)
#
#  #          @as_op(itypes=[T.lscalar, T.dscalar, T.dscalar], otypes=[T.dvector])
#  #          
#  #          def crazy_modulo3(switchpoint, exp_scale, exp_rate):
#  #            out = np.zeros(len(data))
#  #            out[switchpoint:] = exp_scale * (np.exp( exp_rate * (timestamp[switchpoint:] - switchpoint))-1.)
#  #            return out
#
#    
#    #baseline_model = Deterministic('baseline_model', exp_scale * (exp( (timestamp-switchpoint)*rate)-1.) )
#    
#  #          baseline_model = siggen_model(switchpoint, radiusEstimate, zEstimate)
#    baseline_model_dt = siggen_model_dt(switchpoint, dtEstimate)
#    
#    
#    baseline_observed = Normal("baseline_observed", mu=baseline_model_dt, sd=uncertainty_model, observed= data )
#
#  return signal_model