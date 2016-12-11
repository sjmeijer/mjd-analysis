#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.optimize as op
import numpy as np
import emcee
from scipy import signal

import helpers
from detector_model import *
from probability_model_temp import *

from progressbar import ProgressBar, Percentage, Bar, ETA
from timeit import default_timer as timer
from multiprocessing import Pool

import probability_model_waveform as mlw
nll = lambda *args: -mlw.lnlike_waveform(*args)

def main():
##################
#These change a lot
  numWaveforms = 2
  numThreads = 8
  
  ndim = 6*numWaveforms + 13
  nwalkers = 100*ndim
  
  iter=50000
  burnIn = 49000
  wfPlotNumber = 100
  
######################

  doPlots = 1

#  plt.ion()

  fitSamples = 200
  timeStepSize = 1. #ns
  
  #Prepare detector
  tempGuess = 79.310080
  gradGuess = 0.04
  pcRadGuess = 2.5
  pcLenGuess = 1.6

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10 )
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  
  b_over_a = 0.107213
  c = -0.815152
  d = 0.822696
  rc1 = 74.4
  rc2 = 1.79
  rcfrac = 0.992
  trapping_rc = 120#us
  det.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)
  det.trapping_rc = trapping_rc
  
  det.siggenInst.set_velocity_type(1)
  h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0 = 66333., 0.744, 181., 107270., 0.580, 100.
  
  mlw.initializeDetector(det, )
  

  #and the remaining 6 are for the transfer function
  fig_size = (20,10)
  
  #Create a decent start guess by fitting waveform-by-waveform
  wfFileName = "P42574A_12_fastandslow_oldwfs.npz"
#  wfFileName =  "P42574A_24_spread.npz"
#  wfFileName =  "P42574A_5_fast.npz"
  
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    results = data['results']
    wfs = data['wfs']
    
#    wfs = np.delete(wfs, [2])
#    results = np.delete(results, [2])
#    results = results[::3]

    idxs = [4,5]
    
    wfs = wfs[idxs]
    results = results[idxs]

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
#  plt.ion()
#  fig = plt.figure()
  for (idx, wf) in enumerate(wfs):
    wf.WindowWaveformTimepoint(fallPercentage=.99, rmsMult=2,)
    r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx]  = results[idx]['x']
#    plt.plot(wf.windowedWf)
#    value = raw_input('  --> Press q to quit, any other key to continue\n')

#    t0_arr[idx] -= 15
  #Plot the waveforms to take a look at the initial guesses
  if True:
    plt.ion()


    fig1 = plt.figure(1)
    plt.clf()
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.set_xlabel("Digitizer Time [ns]")
    ax0.set_ylabel("Voltage [Arb.]")
    ax1.set_ylabel("Residual")
    
    
    for (idx,wf) in enumerate(wfs):
      
      print "WF number %d:" % idx
      mlw.initializeWaveform(wf)
      dataLen = wf.wfLength
      t_data = np.arange(dataLen) * 10
      ax0.plot(t_data, wf.windowedWf, color="r")
      
#      minresult = None
#      minlike = np.inf
#  
#      for r in np.linspace(4, np.floor(det.detector_radius)-3, 6):
#        for z in np.linspace(4, np.floor(det.detector_length)-3, 6):
#  #        for t0_guess in np.linspace(wf.t0Guess-10, wf.t0Guess+5, 3):
#            if not det.IsInDetector(r,0,z): continue
#            startGuess = [r, np.pi/8, z, wf.wfMax, wf.t0Guess-5, 10]
#            result = op.minimize(nll, startGuess,   method="Nelder-Mead")
#            r, phi, z, scale, t0, smooth, = result["x"]
#            ml_wf = np.copy(det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing=smooth, ))
#            if ml_wf is None:
#              print r, z
#              continue
#            if result['fun'] < minlike:
#              minlike = result['fun']
#              minresult = result
#      r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx]  = minresult['x']
      r, phi, z, scale, t0, smooth  = results[idx]['x']
      startGuess = [r, phi, z, scale, t0, smooth]
      result = op.minimize(nll, startGuess,   method="Powell")
      r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx] = result['x']
      
      print "  >>r: %f\n  >>phi %f\n  >>z %f\n  >>e %f\n  >>t0 %f\n >>smooth %f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx])
      ml_wf = det.MakeSimWaveform(r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], fitSamples, h_smoothing = smooth_arr[idx])
      ax0.plot(t_data, ml_wf[:dataLen], color="g")
      ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color="g",)
    value = raw_input('  --> Press q to quit, any other key to continue\n')
    plt.ioff()
    if value == 'q': exit(0)

  #Initialize the multithreading
  p = Pool(numThreads, initializer=initializeDetectorAndWaveforms, initargs=[det, wfs])
  initializeDetectorAndWaveforms(det, wfs)

  #Do the MCMC
  mcmc_startguess = np.hstack((r_arr[:], phi_arr[:], z_arr[:], scale_arr[:], t0_arr[:], smooth_arr[:],       # waveform-specific params
                              trapping_rc, b_over_a, c, d, rc1, rc2, rcfrac,
                              h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)) # detector-specific

  #number of walkers _must_ be even
  if nwalkers % 2:
    nwalkers +=1

  pos0 = [mcmc_startguess + 1e-2*np.random.randn(ndim)*mcmc_startguess for i in range(nwalkers)]

  trapIdx = -13
  rc1idx = -9
  rc2idx = -8
  rcfracidx = -7

  #Make sure everything in the initial guess is within bounds
  for pos in pos0:
    pos[:numWaveforms] = np.clip( pos[:numWaveforms], 0, np.floor(det.detector_radius*10.)/10.)
    pos[numWaveforms:2*numWaveforms] = np.clip(pos[numWaveforms:2*numWaveforms], 0, np.pi/4)
    pos[2*numWaveforms:3*numWaveforms] = np.clip(pos[2*numWaveforms:3*numWaveforms], 0, np.floor(det.detector_length*10.)/10.)
    pos[4*numWaveforms:5*numWaveforms] = np.clip(pos[4*numWaveforms:5*numWaveforms], 0, fitSamples)
    pos[5*numWaveforms:6*numWaveforms] = np.clip(pos[5*numWaveforms:6*numWaveforms], 0, 20.)

#    pos[tempIdx] = np.clip(pos[tempIdx], 40, 120)
    pos[trapIdx] = np.clip(pos[trapIdx], 0, np.inf)
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

  if not doPlots:
    exit(0)


  print "Dumping chain to file..."
  np.save("sampler_%dwfs.npy" % numWaveforms, sampler.chain)


  print "Making MCMC steps figure..."



#  #########  Plots for Waveform params
#  stepsFig = plt.figure(2, figsize=fig_size)
#  plt.clf()
#  ax0 = stepsFig.add_subplot(611)
#  ax1 = stepsFig.add_subplot(612, sharex=ax0)
#  ax2 = stepsFig.add_subplot(613, sharex=ax0)
#  ax3 = stepsFig.add_subplot(614, sharex=ax0)
#  ax4 = stepsFig.add_subplot(615, sharex=ax0)
#  ax5 = stepsFig.add_subplot(616, sharex=ax0)
#
#  ax0.set_ylabel('r')
#  ax1.set_ylabel('phi')
#  ax2.set_ylabel('z')
#  ax3.set_ylabel('scale')
#  ax4.set_ylabel('t0')
#  ax5.set_ylabel('smoothing')
#
#  for i in range(nwalkers):
#    for j in range(wfs.size):
#      ax0.plot(sampler.chain[i,:,0+j], alpha=0.3)                 # r
#      ax1.plot(sampler.chain[i,:,numWaveforms + j], alpha=0.3)    # phi
#      ax2.plot(sampler.chain[i,:,2*numWaveforms + j], alpha=0.3)  #z
#      ax3.plot(sampler.chain[i,:,3*numWaveforms + j],  alpha=0.3) #energy
#      ax4.plot(sampler.chain[i,:,4*numWaveforms + j],  alpha=0.3) #t0
#      ax5.plot(sampler.chain[i,:,5*numWaveforms + j],  alpha=0.3) #smoothing
#
#  plt.savefig("emcee_wfchain_%dwfs.png" % numWaveforms)
#
#
#  #and for the transfer function
  stepsFigTF = plt.figure(7, figsize=fig_size)
  plt.clf()
  tf0 = stepsFigTF.add_subplot(711)
  tf1 = stepsFigTF.add_subplot(712, sharex=tf0)
  tf2 = stepsFigTF.add_subplot(713, sharex=tf0)
  tf3 = stepsFigTF.add_subplot(714, sharex=tf0)
  tf4 = stepsFigTF.add_subplot(715, sharex=tf0)
  tf5 = stepsFigTF.add_subplot(716, sharex=tf0)
  tf6 = stepsFigTF.add_subplot(717, sharex=tf0)
#  tf7 = stepsFigTF.add_subplot(818, sharex=tf0)

  tf0.set_ylabel('b_over_a')
  tf1.set_ylabel('c')
  tf2.set_ylabel('d')
  tf3.set_ylabel('rc1')
  tf4.set_ylabel('rc2')
  tf5.set_ylabel('rcfrac')
  tf6.set_ylabel('temp')
#  tf7.set_ylabel('trapping')

  for i in range(nwalkers):
    tf0.plot(sampler.chain[i,:,trapIdx+1], "b", alpha=0.3) #2
    tf1.plot(sampler.chain[i,:,trapIdx+2], "b", alpha=0.3) #den1
    tf2.plot(sampler.chain[i,:,trapIdx+3], "b", alpha=0.3) #2
    tf3.plot(sampler.chain[i,:,trapIdx+4], "b", alpha=0.3) #3
    tf4.plot(sampler.chain[i,:,trapIdx+5], "b", alpha=0.3) #3
    tf5.plot(sampler.chain[i,:,trapIdx+6], "b", alpha=0.3) #3
    tf6.plot(sampler.chain[i,:,trapIdx], "b", alpha=0.3) #3
#    tf6.plot(sampler.chain[i,:,trapIdx], "b", alpha=0.3) #3

  plt.savefig("emcee_tfchain_%dwfs.png" % numWaveforms)

  stepsFigTF = plt.figure(6, figsize=fig_size)
  plt.clf()
  tf0 = stepsFigTF.add_subplot(611)
  tf1 = stepsFigTF.add_subplot(612, sharex=tf0)
  tf2 = stepsFigTF.add_subplot(613, sharex=tf0)
  tf3 = stepsFigTF.add_subplot(614, sharex=tf0)
  tf4 = stepsFigTF.add_subplot(615, sharex=tf0)
  tf5 = stepsFigTF.add_subplot(616, sharex=tf0)
#  tf6 = stepsFigTF.add_subplot(717, sharex=tf0)
#  tf7 = stepsFigTF.add_subplot(818, sharex=tf0)

  tf0.set_ylabel('1')
  tf1.set_ylabel('2')
  tf2.set_ylabel('3')
  tf3.set_ylabel('4')
  tf4.set_ylabel('5')
  tf5.set_ylabel('6')
#  tf6.set_ylabel('temp')
#  tf7.set_ylabel('trapping')

  for i in range(nwalkers):
    tf0.plot(sampler.chain[i,:,trapIdx+7], "b", alpha=0.3) #2
    tf1.plot(sampler.chain[i,:,trapIdx+8], "b", alpha=0.3) #den1
    tf2.plot(sampler.chain[i,:,trapIdx+9], "b", alpha=0.3) #2
    tf3.plot(sampler.chain[i,:,trapIdx+10], "b", alpha=0.3) #3
    tf4.plot(sampler.chain[i,:,trapIdx+11], "b", alpha=0.3) #3
    tf5.plot(sampler.chain[i,:,trapIdx+12], "b", alpha=0.3) #3
#    tf6.plot(sampler.chain[i,:,trapIdx], "b", alpha=0.3) #3
#    tf6.plot(sampler.chain[i,:,trapIdx], "b", alpha=0.3) #3

  plt.savefig("emcee_velochain_%dwfs.png" % numWaveforms)


  samples = sampler.chain[:, burnIn:, :].reshape((-1, ndim))

  stepsFigTF = plt.figure(5, figsize = (20,10))
  plotnum = 600
  tf0 = stepsFigTF.add_subplot(plotnum+11)
  tf1 = stepsFigTF.add_subplot(plotnum+12, )
  tf2 = stepsFigTF.add_subplot(plotnum+13, )
  tf3 = stepsFigTF.add_subplot(plotnum+14, )
  tf4 = stepsFigTF.add_subplot(plotnum+15, )
  tf5 = stepsFigTF.add_subplot(plotnum+16, )

  tf0.set_ylabel('h_100_mu0')
  tf1.set_ylabel('h_100_beta')
  tf2.set_ylabel('h_100_e0')
  tf3.set_ylabel('h_111_mu0')
  tf4.set_ylabel('h_111_beta')
  tf5.set_ylabel('h_111_e0')

  num_bins = 300
  [n, b, p] = tf0.hist(samples[:,-6], bins=num_bins)
  print "h_100_mu0 mode is %f" % b[np.argmax(n)]

  [n, b, p] = tf1.hist(samples[:,-5],bins=num_bins)
  print "h_100_beta mode is %f" % b[np.argmax(n)]

  [n, b, p] = tf2.hist(samples[:,-4],bins=num_bins)
  print "h_100_e0 mode is %f" % b[np.argmax(n)]

  [n, b, p] = tf3.hist(samples[:,-3],bins=num_bins)
  print "h_111_mu0 mode is %f" % b[np.argmax(n)]

  [n, b, p] = tf4.hist(samples[:,-2],bins=num_bins)
  print "h_111_beta mode is %f" % b[np.argmax(n)]

  [n, b, p] = tf5.hist(samples[:,-1],bins=num_bins)
  print "h_111_e0 mode is %f" % b[np.argmax(n)]

#  print "temp is %f" % np.median(samples[:,tempIdx])
  print "trapping is %f" % np.median(samples[:,trapIdx])
  print "b_over_a is %f" % np.median(samples[:,-6])
  print "c is %f" % np.median(samples[:,-5])
  print "d is %f" % np.median(samples[:,-4])
  print "rc_decay1 is %f" % np.median(samples[:,-3])
  print "rc_decay2 is %f" % np.median(samples[:,-2])
  print "rc_frac   is %f" % np.median(samples[:,-1])

  #TODO: Aaaaaaand plot some waveforms..
  simWfs = np.empty((wfPlotNumber,numWaveforms, fitSamples))

  for idx, (theta) in enumerate(samples[np.random.randint(len(samples), size=wfPlotNumber)]):
#    temp  = theta[tempIdx]
#    trapping_rc  = theta[trapIdx]
    trapping_rc, b_over_a, c, d, rc1, rc2, rcfrac, h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0, = theta[-13:]
    r_arr, phi_arr, z_arr, scale_arr, t0_arr, smooth_arr  = theta[:-13].reshape((6, numWaveforms))
    
    det.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
#    det.SetTemperature(temp)
    det.trapping_rc = trapping_rc
    
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


