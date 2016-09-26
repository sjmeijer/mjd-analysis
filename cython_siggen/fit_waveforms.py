#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os, time
import scipy.optimize as op
import numpy as np
import emcee
from scipy import signal

import helpers
from detector_model import *
from probability_model_waveform import *

from progressbar import ProgressBar, Percentage, Bar, ETA
from multiprocessing import Pool
from timeit import default_timer as timer

def main(argv):

  numThreads = 8

  runRange = (13420,13429)
  channel = 626
  aeCutVal = 0.01425
  
  fitSamples = 250
  timeStepSize = 10
  
  wfFileName = "multisite_event_set_runs13385-13557"
#  wfFileName = "P42574A_512waveforms_30risetimeculled"
  if os.path.isfile(wfFileName + ".npz"):
    data = np.load(wfFileName+ ".npz")
    wfs = data['wfs']
    numWaveforms = wfs.size
  else:
    print "No saved waveforms available.  Loading from Data"
    exit(0)
  
  zero_1 = 0.470677
  pole_1 = 0.999857
  pole_real = 0.807248
  pole_imag = 0.085347
  tempGuess = 78.474793
  gradGuess = 0.045049
  pcRadGuess = 2.574859
  pcLenGuess = 1.524812
  
  zeros = [zero_1, -1., 1. ]
  poles = [pole_1, pole_real+pole_imag*1j, pole_real-pole_imag*1j, ]

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10./timeStepSize, poles=poles, zeros=zeros)
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  initializeDetector(det, )


#  bar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(wfs)).start()
#
#  for (idx,wf) in enumerate(wfs):
#    bar.update(idx+1)
#    
#    wf.WindowWaveformTimepoint(fallPercentage=.99)
#    if wf.wfLength > fitSamples:
#      wf.lnlike = np.nan
#      continue
#    
#    initializeWaveform(wf)
#
#    startGuess = [15., np.pi/8, 15., wf.wfMax, wf.t0Guess, 1]
#    result = op.minimize(nll, startGuess,   method="Nelder-Mead")
#
#    r, phi, z, scale, t0, smooth, = result["x"]
#    r_new = np.amin( [z, np.floor(det.detector_radius)] )
#    z_new = np.amin( [r, np.floor(det.detector_length)] )
#    
#    result2 = op.minimize(nll, [r_new, phi, z_new, scale, t0,smooth],  method="Nelder-Mead")
#    r, phi, z, scale, t0, smooth, = result2["x"]
#
#    wf.lnlike = np.amin([result['fun'], result2['fun']])
#
#  bar.finish()
#  wfFileName += "_mlefit.npz"
#  np.savez(wfFileName, wfs = wfs )



  p = Pool(numThreads, initializer=initializeDetector, initargs=[det])
  wf_arr = []
  
  for (idx,wf) in enumerate(wfs):
    wf_arr.append([wf])
#    if idx == 16: break

  wf_save_arr = np.empty(len(wf_arr), dtype=np.object)
  bar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(wf_arr)).start()
  start = timer()
  for i, wf in enumerate(p.imap_unordered(mcmc_waveform_star, wf_arr, )):
    wf_save_arr[i] = wf
#    print wf.lnprob
    bar.update(i+1)
  end = timer()
  bar.finish()
  p.close()
  print "Elapsed time: " + str(end-start)

  wfFileName += "_mcmcfit.npz"
  np.savez(wfFileName, wfs = wf_save_arr )


def mcmc_waveform_star(a_b):
  return mcmc_waveform(*a_b)

def nll(*args):
  return -lnlike_waveform(*args)

def mcmc_waveform(wf):
  fitSamples = 250
  wf.WindowWaveformTimepoint(fallPercentage=.99)
  
  if wf.wfLength > fitSamples:
    #skip really long waveforms
    wf.lnprob = np.nan
    return wf
  
  initializeWaveform(wf)
  
  mcmc_startguess = [15., np.pi/8, 15., wf.wfMax, wf.t0Guess, 1.]
  
  result = op.minimize(nll, mcmc_startguess,  method="Nelder-Mead")
  r, phi, z, scale, t0, smooth, = result["x"]
  result2 = op.minimize(nll, [z, phi, r, scale, wf.t0Guess,1],  method="Nelder-Mead")
  
  if result['fun']/wf.wfLength and result2['fun']/wf.wfLength  > 100:
    wf.lnprob = np.nan
    return wf

  #Do the MCMC
  ndim, nwalkers = 6, 100
  
  pos0 = [mcmc_startguess + 1e-1*np.random.randn(ndim)*mcmc_startguess for i in range(nwalkers)]
  sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_waveform, )

  iter, burnIn = 1000, 900
  
  for (idx,result) in enumerate(sampler.sample(pos0, iterations=iter, storechain=True)):
    continue

  lnprobs = sampler.lnprobability[:, burnIn:].reshape((-1))
  median_prob = np.median(lnprobs)
  wf.lnprob = median_prob
  return wf

if __name__=="__main__":
    main(sys.argv[1:])


