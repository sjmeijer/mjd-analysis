#!/usr/bin/python
import ROOT
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
TROOT.gApplication.ExecuteFile("$GATDIR/LoadGATClasses.C")

import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy import ndimage
#from pymc3 import DiscreteUniform, Exponential, Poisson, Uniform, Normal, find_MAP, HalfNormal, switch, Model, Deterministic, exp, Metropolis, traceplot, sample, summary
#import theano.tensor as T
#from theano.compile.ops import as_op


def main(argv):

  samplingPeriod = 10 * CLHEP.ns

  rcIntTimeConstant = 50 * CLHEP.ns / samplingPeriod
  rdDiffTimeConstant = 50*CLHEP.us / samplingPeriod
  gaussian_smoothing = 1.7

#  rcint = MGWFRCIntegration()
#  rcint.SetTimeConstant(rcIntTimeConstant)
#  rcdiff = MGWFRCDifferentiation()
#  rcdiff.SetTimeConstant(pzCorrTimeConstant)
#
#  pzcorr = MGWFPoleZeroCorrection()
#  pzcorr.SetDecayConstant(pzCorrTimeConstant)


  #pulserOutputFileName = "P1D3_Pulser_out_superpulse_20151001.root"
  #pulserOutputFileName = "P5D2_Pulser_out_superpulse_20151006.root"
  fitStart = 900
  fitEnd = 1075
  
  pulserOutputFileName = "P1D3_Pulser_out_superpulse_20151001.root"
  

  
  pulserOutputFile = TFile(pulserOutputFileName)
  pulserOutputTree = pulserOutputFile.Get("pulser_superpulse")
  pulserOutputBranch  = pulserOutputTree.GetBranch("PulserWaveform")
  pulserOutputWf = MGTWaveform();
  pulserOutputBranch.SetAddress(AddressOf(pulserOutputWf))
  pulserOutputBranch.GetEntry(0)
  print "pulser output length is %d" % pulserOutputWf.GetLength()


  wf = pulserOutputWf
  pulser_data = wf.GetVectorData()

#  basic_model = createPulserModelNoSmooth(pulser_data[fitStart:fitEnd])
#  
#  with basic_model:
#    #start =find_MAP()
#    step = Metropolis()
#    #2000 MCMC steps is pulled out of thin air for now.  Its on you to make sure its converged.
#    tr = sample(2000, step)
#
#  summary(tr)
#
#  burnNumber=500
#  startVal = np.median(tr["switchpoint"][burnNumber:]) + fitStart
#  rc_int =  np.median(tr["rc_int"][burnNumber:])
#  rc_diff = np.median(tr["rc_diff"][burnNumber:])
##  smooth = np.median(tr["smooth"][burnNumber:])

  startVal = 987
  rc_int = 51.5 * CLHEP.ns /  (10 * CLHEP.ns)
  rc_diff = 72 * CLHEP.us /  (10 * CLHEP.ns)
  gaussian_smoothing = 1.75
  
  print "rc int:  %f ns" % ( rc_int * 10 * CLHEP.ns / CLHEP.ns)
  print "rc diff: %f us" % ( rc_diff * 10 * CLHEP.ns / CLHEP.us)
#  print "gauss smooth: %f samples" % ( smooth)


  tf_data = np.zeros(len(pulser_data))
  tf_data[startVal:] = 1
  
  tf_data = rc_integrate(tf_data, rc_int)
  tf_data = rc_differentiate(tf_data, rc_diff)
  
  tf_data_smooth = ndimage.filters.gaussian_filter1d(tf_data, gaussian_smoothing)
  
  tf_data/= np.amax(tf_data)
  tf_data_smooth/= np.amax(tf_data_smooth)

  fig1 = plt.figure(1)

  plt.plot( tf_data_smooth  ,color="red" )
  plt.plot( pulser_data  ,color="blue" )

  plt.xlim( 950,1200)
#  plt.ylim( .9,1)

  plt.show()



  #
  #outfile = TFile(outFileName, "RECREATE");
  #outTree = TTree("transfer_function", "Transfer Function from Pulser Data")
  #outTree.Branch("TransferFunctionWaveform", "MGTWaveform", transferFunctionWf);
  #
  #outTree.Fill()
  #outfile.Write()
  #outfile.Close()
  #


def createPulserModel2(pulser_data, t0_guess, scale_guess):
  samplingPeriod = 10 * CLHEP.ns

  switchpoint = Normal('switchpoint', mu=t0_guess, tau=sigToTau(1))
  wfScale = Normal('wfScale', mu=energy_guess, tau=sigToTau(.1*scale_guess))

  rcInt = Normal('rcInt', mu=rc_int, tau=sigToTau(.1*rc_int))
  rcDiff = Normal('rcDiff', mu=rc_diff, tau=sigToTau(.1*rc_diff))
  gaussSmooth = Normal('gaussSmooth', mu=gaussian_smoothing, tau=sigToTau(.1*gaussian_smoothing))


def createPulserModel(pulser_data):
  samplingPeriod = 10 * CLHEP.ns

  with Model() as pulser_model:
  
  
    switchpoint = DiscreteUniform('switchpoint', lower=0, upper=len(pulser_data))
    
    noise_sigma = HalfNormal('noise_sigma', sd=1.)
    
    #Modeling these parameters this way is why wf needs to be normalized
    rc_int = Uniform('rc_int', lower=0, upper=100* CLHEP.ns / samplingPeriod)
    rc_diff = Uniform('rc_diff', lower=0, upper=100* CLHEP.us / samplingPeriod)
    gaussian_smoothing = Uniform('smooth', lower=0, upper=10)
    
    timestamp = np.arange(0, len(pulser_data), dtype=np.float)
    
    @as_op(itypes=[T.lscalar, T.dscalar, T.dscalar, T.dscalar], otypes=[T.dvector])
    def pulser_baseline(switchpoint, rc_int, rc_diff, gaussian_smoothing):
      baseline = np.zeros(len(pulser_data))
      baseline[switchpoint:] = 1
      
      tf_data = rc_integrate(baseline, rc_int)
      tf_data/= np.amax(tf_data)
      tf_data = rc_differentiate(tf_data, rc_diff)
      tf_data_smooth = ndimage.filters.gaussian_filter1d(tf_data, gaussian_smoothing)
      
      return tf_data_smooth
    
    baseline_model = pulser_baseline(switchpoint, rc_int, rc_diff, gaussian_smoothing)
    
    baseline_observed = Normal("baseline_observed", mu=baseline_model, sd=noise_sigma, observed= pulser_data )
  return pulser_model


def rc_integrate(anInput, timeConstant):
  output = np.copy(anInput)
  expTimeConstant = np.exp(-1./timeConstant);
  for i in range(1,len(output)):
    output[i] = output[i] + expTimeConstant*output[i-1];
  
  return output

def rc_differentiate(input, timeConstant):
  output = np.copy(input)
  dummy = output[0]
  output[0] = 0.0
  
  for i in range(1,len(input)):
    dummy2  = output[i-1] + output[i] - dummy - output[i-1] / timeConstant;
    dummy = output[i];
    output[i] = dummy2;
  
  return output



if __name__=="__main__":
    main(sys.argv[1:])

