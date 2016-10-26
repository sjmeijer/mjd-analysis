#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy as np
import emcee
from emcee.utils import MPIPool
from scipy import signal

import helpers
from detector_model import *
from probability_model_temp import *

from progressbar import ProgressBar, Percentage, Bar, ETA
from timeit import default_timer as timer
from multiprocessing import Pool

def main():
##################
#These change a lot
  numWaveforms = 12
  numThreads = 8
  
  ndim = 6*numWaveforms + 7
  nwalkers = 25*ndim
  
  iter=100
  burnIn = 90
  wfPlotNumber = 10
  
######################

  doPlots = 1

#  plt.ion()

  fitSamples = 350
  timeStepSize = 1. #ns
  
  #Prepare detector
  tempGuess = 79.310080
  gradGuess = 0.05
  pcRadGuess = 2.5
  pcLenGuess = 1.6

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10 )
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  
  b_over_a = 0.107213
  c = -0.821158
  d = 0.828957
  rc1 = 74.4
  rc2 = 1.79
  rcfrac = 0.992
  det.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)
  
  tempIdx = -7
  #and the remaining 4 are for the transfer function
  fig_size = (20,10)
  
  #Create a decent start guess by fitting waveform-by-waveform
  wfFileName = "P42574A_12_fastandslow_oldwfs.npz"
#  wfFileName =  "P42574A_5_fast.npz"
  
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    results = data['results']
    wfs = data['wfs']
    
#    wfs = wfs[::3]
#    results = results[::3]

    numWaveforms = wfs.size
  else:
    print "No saved waveforms available.  Exiting."
    exit(0)

  #prep holders for each wf-specific param
  r_arr = np.empty(numWaveforms)
  phi_arr = np.empty(numWaveforms)
  z_arr = np.empty(numWaveforms)
  scale_arr = np.empty(numWaveforms)
  t0_arr = np.empty(numWaveforms)
  smooth_arr = np.ones(numWaveforms)*7.
  simWfArr = np.empty((1,numWaveforms, fitSamples))

  #Prepare the initial value arrays
  for (idx, wf) in enumerate(wfs):
    wf.WindowWaveformTimepoint(fallPercentage=.97, rmsMult=2,)
    r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx]  = results[idx]['x']
#    t0_arr[idx] -= 15

  #Initialize the multithreading
#  p = Pool(numThreads, initializer=initializeDetectorAndWaveforms, initargs=[det, wfs])

  initializeDetectorAndWaveforms(det, wfs)
  p = MPIPool()
  if not pool.is_master():
    pool.wait()
    sys.exit(0)

  #Do the MCMC
  mcmc_startguess = np.hstack((r_arr[:], phi_arr[:], z_arr[:], scale_arr[:], t0_arr[:], smooth_arr[:],       # waveform-specific params
                              tempGuess, b_over_a, c, d, rc1, rc2, rcfrac)) # detector-specific

  #number of walkers _must_ be even
  if nwalkers % 2:
    nwalkers +=1

  pos0 = [mcmc_startguess + 1e-2*np.random.randn(ndim)*mcmc_startguess for i in range(nwalkers)]
  rc1idx = -3
  rc2idx = -2
  rcfracidx = -1

  #Make sure everything in the initial guess is within bounds
  for pos in pos0:
    pos[:numWaveforms] = np.clip( pos[:numWaveforms], 0, np.floor(det.detector_radius*10.)/10.)
    pos[numWaveforms:2*numWaveforms] = np.clip(pos[numWaveforms:2*numWaveforms], 0, np.pi/4)
    pos[2*numWaveforms:3*numWaveforms] = np.clip(pos[2*numWaveforms:3*numWaveforms], 0, np.floor(det.detector_length*10.)/10.)
    pos[4*numWaveforms:5*numWaveforms] = np.clip(pos[4*numWaveforms:5*numWaveforms], 0, fitSamples)
    pos[5*numWaveforms:6*numWaveforms] = np.clip(pos[5*numWaveforms:6*numWaveforms], 0, 20.)

    pos[tempIdx] = np.clip(pos[tempIdx], 40, 120)
    pos[rcfracidx] = np.clip(pos[rcfracidx], 0, 1)
    pos[rc2idx] = np.clip(pos[rc2idx], 0, np.inf)
    pos[rc1idx] = np.clip(pos[rc1idx], 0, np.inf)


    prior = lnprior(pos,)
    if not np.isfinite(prior) :
      print "BAD PRIOR WITH START GUESS YOURE KILLING ME SMALLS"
      print pos
      exit(0)

  #Initialize, run the MCMC
  sampler = emcee.EnsembleSampler( nwalkers, ndim,  lnprob,  pool=p)

  #w/ progress bar, & time the thing
  bar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=iter).start()
  for (idx,result) in enumerate(sampler.sample(pos0, iterations=iter, storechain=True)):
    bar.update(idx+1)
  bar.finish()

  p.close()

  if not doPlots:
    exit(0)


  print "Dumping chain to file..."
  np.save("sampler_%dwfs.npy" % numWaveforms, sampler.chain)


  print "Making MCMC steps figure..."



  #########  Plots for Waveform params
  stepsFig = plt.figure(2, figsize=fig_size)
  plt.clf()
  ax0 = stepsFig.add_subplot(611)
  ax1 = stepsFig.add_subplot(612, sharex=ax0)
  ax2 = stepsFig.add_subplot(613, sharex=ax0)
  ax3 = stepsFig.add_subplot(614, sharex=ax0)
  ax4 = stepsFig.add_subplot(615, sharex=ax0)
  ax5 = stepsFig.add_subplot(616, sharex=ax0)

  ax0.set_ylabel('r')
  ax1.set_ylabel('phi')
  ax2.set_ylabel('z')
  ax3.set_ylabel('scale')
  ax4.set_ylabel('t0')
  ax5.set_ylabel('smoothing')

  for i in range(nwalkers):
    for j in range(wfs.size):
      ax0.plot(sampler.chain[i,:,0+j], alpha=0.3)                 # r
      ax1.plot(sampler.chain[i,:,numWaveforms + j], alpha=0.3)    # phi
      ax2.plot(sampler.chain[i,:,2*numWaveforms + j], alpha=0.3)  #z
      ax3.plot(sampler.chain[i,:,3*numWaveforms + j],  alpha=0.3) #energy
      ax4.plot(sampler.chain[i,:,4*numWaveforms + j],  alpha=0.3) #t0
      ax5.plot(sampler.chain[i,:,5*numWaveforms + j],  alpha=0.3) #smoothing

  plt.savefig("emcee_wfchain_%dwfs.png" % numWaveforms)


  #and for the transfer function
  stepsFigTF = plt.figure(7, figsize=fig_size)
  plt.clf()
  tf0 = stepsFigTF.add_subplot(711)
  tf1 = stepsFigTF.add_subplot(712, sharex=ax0)
  tf2 = stepsFigTF.add_subplot(713, sharex=ax0)
  tf3 = stepsFigTF.add_subplot(714, sharex=ax0)
  tf4 = stepsFigTF.add_subplot(715, sharex=ax0)
  tf5 = stepsFigTF.add_subplot(716, sharex=ax0)
  tf6 = stepsFigTF.add_subplot(717, sharex=ax0)

  tf0.set_ylabel('b_over_a')
  tf1.set_ylabel('c')
  tf2.set_ylabel('d')
  tf3.set_ylabel('rc1')
  tf4.set_ylabel('rc2')
  tf5.set_ylabel('rcfrac')
  tf6.set_ylabel('temp')

  for i in range(nwalkers):
    tf0.plot(sampler.chain[i,:,-6], "b", alpha=0.3) #2
    tf1.plot(sampler.chain[i,:,-5], "b", alpha=0.3) #den1
    tf2.plot(sampler.chain[i,:,-4], "b", alpha=0.3) #2
    tf3.plot(sampler.chain[i,:,-3], "b", alpha=0.3) #3
    tf4.plot(sampler.chain[i,:,-2], "b", alpha=0.3) #3
    tf5.plot(sampler.chain[i,:,-1], "b", alpha=0.3) #3
    tf6.plot(sampler.chain[i,:,tempIdx], "b", alpha=0.3) #3

  plt.savefig("emcee_tfchain_%dwfs.png" % numWaveforms)


  samples = sampler.chain[:, burnIn:, :].reshape((-1, ndim))

  print "temp is %f" % np.median(samples[:,tempIdx])
  print "b_over_a is %f" % np.median(samples[:,-6])
  print "c is %f" % np.median(samples[:,-5])
  print "d is %f" % np.median(samples[:,-4])
  print "rc_decay1 is %f" % np.median(samples[:,-3])
  print "rc_decay2 is %f" % np.median(samples[:,-2])
  print "rc_frac   is %f" % np.median(samples[:,-1])

  #TODO: Aaaaaaand plot some waveforms..
  simWfs = np.empty((wfPlotNumber,numWaveforms, fitSamples))

  for idx, (theta) in enumerate(samples[np.random.randint(len(samples), size=wfPlotNumber)]):
    temp  = theta[tempIdx]
    b_over_a, c, d, rc1, rc2, rcfrac = theta[-6:]
    r_arr, phi_arr, z_arr, scale_arr, t0_arr, smooth_arr  = theta[:-7].reshape((6, numWaveforms))
    det.SetTemperature(temp)
    
    det.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)

    for wf_idx in range(numWaveforms):
      wf_i = det.MakeSimWaveform(r_arr[wf_idx], phi_arr[wf_idx], z_arr[wf_idx], scale_arr[wf_idx], t0_arr[wf_idx], fitSamples, h_smoothing=smooth_arr[wf_idx] )
      simWfs[idx, wf_idx, :] = wf_i
      if wf_i is None:
        print "Waveform %d, %d is None" % (idx, wf_idx)

  residFig = plt.figure(4, figsize=(20, 15))
  helpers.plotManyResidual(simWfs, wfs, figure=residFig)
  plt.savefig("emcee_waveforms_%dwfs.png" % numWaveforms)


  value = raw_input('  --> Press q to quit, any other key to continue\n')

if __name__=="__main__":
    main()


