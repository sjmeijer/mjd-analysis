#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy as np
import emcee
from scipy import signal

import helpers
from detector_model import *
from probability_model_detector import *

from progressbar import ProgressBar, Percentage, Bar
from timeit import default_timer as timer
from multiprocessing import Pool


def main(argv):
##################
#These change a lot
  numWaveforms = 8
  numThreads = 4
  
  ndim = 5*numWaveforms + 10
  nwalkers = 3*ndim
  
  iter=10
  burnIn = 8
  
######################


  plt.ion()
  
  fitSamples = 200

  #Prepare detector
  num =  [8685207069.0676746, 1.7618952141698222e+18, 17521485536930826.0]
  den = [1, 50310572.447231829, 701441983664560.88, 1.4012406413698292e+19]
  system = signal.lti(num, den)
  
  tempGuess = 78
  gradGuess = 0.0482
  pcRadGuess = 2.563885
  pcLenGuess = 1.440751

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.04,2.5, 1.6)
  det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
  det.LoadFields("P42574A_fields_len.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  initializeDetector(det)
  
  tempIdx = -10
  gradIdx = -9
  pcRadIdx = -8
  pcLenIdx = -7
  
  fig_size = (20,10)
  
  
  #Create a decent start guess by fitting waveform-by-waveform
  
  wfFileName = "P42574A_256waveforms_%drisetimeculled.npz"  % numWaveforms
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    results = data['results']
    wfs = data['wfs']
  
  else:
    print "No saved waveforms available."
    exit(0)

  r_arr = np.empty(numWaveforms)
  phi_arr = np.empty(numWaveforms)
  z_arr = np.empty(numWaveforms)
  scale_arr = np.empty(numWaveforms)
  t0_arr = np.empty(numWaveforms)
  simWfArr = np.empty((1,numWaveforms, fitSamples))

  for (idx, result) in enumerate(results):
    r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx] = result['x']
    print "  >> wf %d (normalized likelihood %0.2f):" % (idx, result["fun"]/wfs[idx].wfLength)
    print "      r: %0.2f, phi: %0.3f, z: %0.2f, e: %0.2f, t0: %0.2f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx])

  p = Pool(numThreads, initializer=initializeDetectorAndWaveforms, initargs=[det, wfs])
  initializeDetectorAndWaveforms(det, wfs)
  #Do the MCMC

  mcmc_startguess = np.hstack((r_arr[:], phi_arr[:], z_arr[:], scale_arr[:]*100., t0_arr[:], tempGuess, gradGuess,pcRadGuess, pcLenGuess, num[:], den[1:]))


  if nwalkers % 2:
    nwalkers +=1

  #Do it without the parallel tempering

  pos0 = [mcmc_startguess + 1e-2*np.random.randn(ndim)*mcmc_startguess for i in range(nwalkers)]

  for pos in pos0:
#    print "radius is %f" % det.detector_radius
#    print "length is %f" % det.detector_length
#  
    pos[:numWaveforms] = np.clip( pos[:numWaveforms], 0, np.floor(det.detector_radius*10.)/10.)
    pos[numWaveforms:2*numWaveforms] = np.clip(pos[numWaveforms:2*numWaveforms], 0, np.pi/4)
    pos[2*numWaveforms:3*numWaveforms] = np.clip(pos[2*numWaveforms:3*numWaveforms], 0, np.floor(det.detector_length*10.)/10.)
    pos[4*numWaveforms:5*numWaveforms] = np.clip(pos[4*numWaveforms:5*numWaveforms], 0, fitSamples)
#    pos[5*numWaveforms:6*numWaveforms] = np.clip(pos[5*numWaveforms:6*numWaveforms], 0, 100.)

    pos[tempIdx] = np.clip(pos[tempIdx], 40, 120)
    pos[gradIdx] = np.clip(pos[gradIdx], det.gradList[0], det.gradList[-1])
    pos[pcRadIdx] = np.clip(pos[pcRadIdx], det.pcRadList[0], det.pcRadList[-1])
    pos[pcLenIdx] = np.clip(pos[pcLenIdx], det.pcLenList[0], det.pcLenList[-1])

    prior = lnprior(pos,)
    if not np.isfinite(prior) :
      print "BAD PRIOR WITH START GUESS YOURE KILLING ME SMALLS"
      print pos
      exit(0)

  sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,  pool=p)

  start = timer()
  
  #w/ progress bar
  bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=iter).start()
  for (idx,result) in enumerate(sampler.sample(pos0, iterations=iter, storechain=True)):
    bar.update(idx+1)

  end = timer()
  bar.finish()
  
  print "Elapsed time: " + str(end-start)

  print "Dumping chain to file..."
  np.save("sampler.npy", sampler.chain)


  print "Making MCMC steps figure..."

  #########  Plots for Waveform params
  stepsFig = plt.figure(2, figsize=fig_size)
  plt.clf()
  ax0 = stepsFig.add_subplot(511)
  ax1 = stepsFig.add_subplot(512, sharex=ax0)
  ax2 = stepsFig.add_subplot(513, sharex=ax0)
  ax3 = stepsFig.add_subplot(514, sharex=ax0)
  ax4 = stepsFig.add_subplot(515, sharex=ax0)
#  ax5 = stepsFig.add_subplot(616, sharex=ax0)

  ax0.set_ylabel('r')
  ax1.set_ylabel('phi')
  ax2.set_ylabel('z')
  ax3.set_ylabel('scale')
  ax4.set_ylabel('t0')
#  ax5.set_ylabel('smoothing')

  for i in range(nwalkers):
    for j in range(wfs.size):
      ax0.plot(sampler.chain[i,:,0+j], alpha=0.3)                 # r
      ax1.plot(sampler.chain[i,:,numWaveforms + j], alpha=0.3)    # phi
      ax2.plot(sampler.chain[i,:,2*numWaveforms + j], alpha=0.3)  #z
      ax3.plot(sampler.chain[i,:,3*numWaveforms + j],  alpha=0.3) #energy
      ax4.plot(sampler.chain[i,:,4*numWaveforms + j],  alpha=0.3) #t0
#      ax5.plot(sampler.chain[i,:,5*numWaveforms + j],  alpha=0.3) #smoothing

  plt.savefig("emcee_wfchain_%dwfs.png" % numWaveforms)


  #########  Plots for Detector params
  stepsFigDet = plt.figure(3, figsize=fig_size)
  plt.clf()
  ax0 = stepsFigDet.add_subplot(411)
  ax1 = stepsFigDet.add_subplot(412, sharex=ax0)
  ax2 = stepsFigDet.add_subplot(413, sharex=ax0)
  ax3 = stepsFigDet.add_subplot(414, sharex=ax0)
  
  ax0.set_ylabel('temp')
  ax1.set_ylabel('grad')
  ax2.set_ylabel('pcRad')
  ax3.set_ylabel('pcLen')

  for i in range(nwalkers):
    ax0.plot(sampler.chain[i,:,tempIdx], "b", alpha=0.3) #temp
    ax1.plot(sampler.chain[i,:,gradIdx], "b", alpha=0.3) #grad
    ax2.plot(sampler.chain[i,:,pcRadIdx], "b", alpha=0.3) #pcrad
    ax3.plot(sampler.chain[i,:,pcLenIdx], "b", alpha=0.3) #pclen
    
  plt.savefig("emcee_detchain_%dwfs.png" % numWaveforms)


  #and for the transfer function
  stepsFigTF = plt.figure(4, figsize=fig_size)
  plt.clf()
  tf0 = stepsFigTF.add_subplot(611)
  tf1 = stepsFigTF.add_subplot(612, sharex=ax0)
  tf2 = stepsFigTF.add_subplot(613, sharex=ax0)
  tf3 = stepsFigTF.add_subplot(614, sharex=ax0)
  tf4 = stepsFigTF.add_subplot(615, sharex=ax0)
  tf5 = stepsFigTF.add_subplot(616, sharex=ax0)
  tf0.set_ylabel('num0')
  tf1.set_ylabel('num1')
  tf2.set_ylabel('num2')
  tf3.set_ylabel('den1')
  tf4.set_ylabel('den2')
  tf5.set_ylabel('den3')

  for i in range(nwalkers):
    tf0.plot(sampler.chain[i,:,-6], "b", alpha=0.3) #num0
    tf1.plot(sampler.chain[i,:,-5], "b", alpha=0.3) #1
    tf2.plot(sampler.chain[i,:,-4], "b", alpha=0.3) #2
    tf3.plot(sampler.chain[i,:,-3], "b", alpha=0.3) #den1
    tf4.plot(sampler.chain[i,:,-2], "b", alpha=0.3) #2
    tf5.plot(sampler.chain[i,:,-1], "b", alpha=0.3) #3

  plt.savefig("emcee_tfchain_%dwfs.png" % numWaveforms)


  samples = sampler.chain[:, burnIn:, :].reshape((-1, ndim))

  print "temp is %f" % np.median(samples[:,tempIdx])
  print "grad is %f" % np.median(samples[:,gradIdx])
  print "pcrad is %f" % np.median(samples[:,pcRadIdx])
  print "pclen is %f" % np.median(samples[:,pcLenIdx])
  print "num1 is %f" % np.median(samples[:,-6])
  print "num2 is %f" % np.median(samples[:,-5])
  print "num3 is %f" % np.median(samples[:,-4])
  print "den1 is %f" % np.median(samples[:,-3])
  print "den2 is %f" % np.median(samples[:,-2])
  print "den3 is %f" % np.median(samples[:,-1])

  plt.show()
  
  value = raw_input('  --> Press q to quit, any other key to continue\n')
  if value == 'q':
    exit(0)


#below is stored code for PT sampler

#  pos0 = np.random.uniform(low=0.9, high=1.1, size=(ntemps, nwalkers, ndim))
#
#  for tidx in range(ntemps):
#    for widx in range(nwalkers):
#      pos0[tidx, widx, :] = pos0[tidx, widx, :] * mcmc_startguess
#
#      pos = pos0[tidx, widx, :]
#
#      pos[:numWaveforms] = np.clip( pos[:numWaveforms], 0, np.floor(det.detector_radius))
#      pos[numWaveforms:2*numWaveforms] = np.clip(pos[numWaveforms:2*numWaveforms], 0, np.pi/4)
#      pos[2*numWaveforms:3*numWaveforms] = np.clip(pos[2*numWaveforms:3*numWaveforms], 0, det.detector_length)
#      pos[4*numWaveforms:5*numWaveforms] = np.clip(pos[4*numWaveforms:5*numWaveforms], 0, fitSamples)
#    
#      pos[tempIdx] = np.clip(pos[tempIdx], 40, 120)
#      pos[gradIdx] = np.clip(pos[gradIdx], det.gradList[0], det.gradList[-1])
#      pos[pcRadIdx] = np.clip(pos[pcRadIdx], det.pcRadList[0], det.pcRadList[-1])
#  
#      prior = lnprior(pos,)
#      if not np.isfinite(prior) :
#        print "BAD PRIOR WITH START GUESS YOURE KILLING ME SMALLS"
#        exit(0)
#
#
#  #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,  threads=numThreads)
#  
#  sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlike_detector, lnprob,  threads=numThreads)
#  
#  start = timer()
#  
#  #w/ progress bar
#  bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=iter).start()
#  for (idx,result) in enumerate(sampler.sample(pos0, iterations=iter, storechain=True)):
#    bar.update(idx+1)
#
#  end = timer()
#  bar.finish()
#  
#  print "Elapsed time: " + str(end-start)
#
#  print "Dumpimng chain to file..."
#  np.save("sampler.npy", sampler.chain)
#  
##
#  print "Making MCMC steps figure..."
#  #########  Plots for MC Steps
#  stepsFig = plt.figure(2, figsize=(20, 15))
#  plt.clf()
#  ax0 = stepsFig.add_subplot(511)
#  ax1 = stepsFig.add_subplot(512, sharex=ax0)
#  ax2 = stepsFig.add_subplot(513, sharex=ax0)
#  ax3 = stepsFig.add_subplot(514, sharex=ax0)
#  ax4 = stepsFig.add_subplot(515, sharex=ax0)
#  
#  ax0.set_ylabel('r')
#  ax1.set_ylabel('phi')
#  ax2.set_ylabel('z')
#  ax3.set_ylabel('scale')
#  ax4.set_ylabel('t0')
#
#  for k in range(ntemps):
#    for i in range(nwalkers):
#      for j in range(wfs.size):
#        ax0.plot(sampler.chain[k, i,:,0+j], alpha=0.3)                 # r
#        ax1.plot(sampler.chain[k, i,:,numWaveforms + j], alpha=0.3)    # phi
#        ax2.plot(sampler.chain[k, i,:,2*numWaveforms + j], alpha=0.3)  #z
#        ax3.plot(sampler.chain[k, i,:,3*numWaveforms + j],  alpha=0.3) #energy
#        ax4.plot(sampler.chain[k, i,:,4*numWaveforms + j],  alpha=0.3) #t0
#
#  plt.savefig("emcee_wfchain_%dwfs.png" % numWaveforms)
#
#  stepsFigDet = plt.figure(3, figsize=(20, 15))
#  plt.clf()
#  ax0 = stepsFigDet.add_subplot(911)
#  ax1 = stepsFigDet.add_subplot(912, sharex=ax0)
#  ax2 = stepsFigDet.add_subplot(913, sharex=ax0)
#  ax3 = stepsFigDet.add_subplot(914, sharex=ax0)
#  ax4 = stepsFigDet.add_subplot(915, sharex=ax0)
#  ax5 = stepsFigDet.add_subplot(916, sharex=ax0)
#  ax6 = stepsFigDet.add_subplot(917, sharex=ax0)
#  ax7 = stepsFigDet.add_subplot(918, sharex=ax0)
#  ax8 = stepsFigDet.add_subplot(919, sharex=ax0)
#  
#  ax0.set_ylabel('temp')
#  ax1.set_ylabel('grad')
#  ax2.set_ylabel('pcRad')
#  ax3.set_ylabel('num1')
#  ax4.set_ylabel('num2')
#  ax5.set_ylabel('num3')
#  ax6.set_ylabel('den1')
#  ax7.set_ylabel('den2')
#  ax8.set_ylabel('den3')
#
#  for j in range(ntemps):
#    for i in range(nwalkers):
#      ax0.plot(sampler.chain[j, i,:,tempIdx], "b", alpha=0.3) #temp
#      ax1.plot(sampler.chain[j, i,:,gradIdx], "b", alpha=0.3) #grad
#      ax2.plot(sampler.chain[j, i,:,pcRadIdx], "b", alpha=0.3) #pcrad
#      ax3.plot(sampler.chain[j, i,:,-6], "b", alpha=0.3) #temp
#      ax4.plot(sampler.chain[j, i,:,-5], "b", alpha=0.3) #grad
#      ax5.plot(sampler.chain[j, i,:,-4], "b", alpha=0.3) #pcrad
#      ax6.plot(sampler.chain[j, i,:,-3], "b", alpha=0.3) #temp
#      ax7.plot(sampler.chain[j, i,:,-2], "b", alpha=0.3) #grad
#      ax8.plot(sampler.chain[j, i,:,-1], "b", alpha=0.3) #pcrad
#
#
#  plt.savefig("emcee_detchain_%dwfs.png" % numWaveforms)
#
#
#  print "making waveforms figure..."
#  det2 =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
#  det2.LoadFields("P42574A_fields.npz")
#
#  #pull the samples after burn-in
#
#  samples = sampler.chain[:, :, burnIn:, :].reshape((-1, ndim))
#  simWfs = np.empty((wfPlotNumber, numWaveforms, fitSamples))
#
#  print "temp is %f" % np.median(samples[:,tempIdx])
#  print "grad is %f" % np.median(samples[:,gradIdx])
#  print "pcrad is %f" % np.median(samples[:,pcRadIdx])
#  print "num1 is %f" % np.median(samples[:,-6])
#  print "num2 is %f" % np.median(samples[:,-5])
#  print "num3 is %f" % np.median(samples[:,-4])
#  print "den1 is %f" % np.median(samples[:,-3])
#  print "den2 is %f" % np.median(samples[:,-2])
#  print "den3 is %f" % np.median(samples[:,-1])
#
#  for idx, (theta) in enumerate(samples[np.random.randint(len(samples), size=wfPlotNumber)]):
#    temp, impGrad, pcRad = theta[tempIdx], theta[gradIdx], theta[pcRadIdx]
#    r_arr, phi_arr, z_arr, scale_arr, t0_arr = theta[:-9].reshape((5, numWaveforms))
#    det2.SetTemperature(temp)
#    det2.SetFields(pcRad, impGrad)
#    num = [theta[-6], theta[-5], theta[-4]]
#    den = [1, theta[-3], theta[-2], theta[-1]]
#    det2.SetTransferFunction(num, den)
#    
#    for wf_idx in range(wfs.size):
#      wf_i = det2.GetSimWaveform(r_arr[wf_idx], phi_arr[wf_idx], z_arr[wf_idx], scale_arr[wf_idx], t0_arr[wf_idx], fitSamples)
#      simWfs[idx, wf_idx, :] = wf_i
#      if wf_i is None:
#        print "Waveform %d, %d is None" % (idx, wf_idx)
#
#
#  residFig = plt.figure(4, figsize=(20, 15))
#  helpers.plotManyResidual(simWfs, wfs, figure=residFig)
#
#  plt.savefig("emcee_waveforms_%dwfs.png" % numWaveforms)
#
#  plt.show()
#  value = raw_input('  --> Press q to quit, any other key to continue\n')
#  if value == 'q':
#    exit(0)


if __name__=="__main__":
    main(sys.argv[1:])


