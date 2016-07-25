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



def main(argv):

  plt.ion()

  runRange = (13420,13429)
  channel = 626
  aeCutVal = 0.01425
  
  numThreads=8
  tempGuess = 118
  fitSamples = 200
  numWaveforms = 10
  
  #Prepare detector
  num = [3.64e+09, 1.88e+17, 6.05e+15]
  den = [1, 4.03e+07, 5.14e+14, 7.15e+18]
  system = signal.lti(num, den)
  
  gradGuess = 0.04
  pcRadGuess = 2.4
#10 wfs
#MLE temp is 118.688837
#MLE grad is 0.042946
#MLE pc rad is 2.380447


#  gradGuess = 0.05
#  pcRadGuess = 2.6
#  MLE temp is 119.349995
#MLE grad is 0.058667
#MLE pc rad is 2.700000

  #Create a detector model
  detName = "conf/P42574A_grad%0.3f_pcrad%0.4f.conf" % (gradGuess,pcRadGuess)
  det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
  det.LoadFields("P42574A_fields.npz")
  
  tempIdx = -9
  gradIdx = -8
  pcRadIdx = -7
  
  wfFileName = "P42574A_%dwaveforms.npz" % numWaveforms
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    r_arr  = data['r_arr']
    phi_arr = data['phi_arr']
    z_arr = data['z_arr']
    scale_arr = data['scale_arr']
    t0_arr = data['t0_arr']
    wfs = data['wfs']
  
  else:
    print "No saved waveforms available.  Loading from Data"
    #get waveforms
    cut = "trapECal>%f && trapECal<%f && TSCurrent100nsMax/trapECal > %f" %  (1588,1594, aeCutVal)
    wfs = helpers.GetWaveforms(runRange, channel, numWaveforms, cut)

    #prep holders for each wf-specific param
    r_arr = np.empty(numWaveforms)
    phi_arr = np.empty(numWaveforms)
    z_arr = np.empty(numWaveforms)
    scale_arr = np.empty(numWaveforms)
    t0_arr = np.empty(numWaveforms)

    #ML as a start, using each individual wf
    nll_wf = lambda *args: -lnlike_waveform(*args)
    
    for (idx,wf) in enumerate(wfs):
      print "Doing MLE for waveform %d" % idx
      wf.WindowWaveformTimepoint(fallPercentage=.99)
      startGuess = [15., np.pi/8, 15., wf.wfMax, wf.t0Guess]
      
      result = op.minimize(nll_wf, startGuess, args=(wf, det),  method="Powell")
      r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx] = result["x"]

    np.savez(wfFileName, wfs = wfs, r_arr=r_arr, phi_arr = phi_arr, z_arr = z_arr, scale_arr = scale_arr,  t0_arr=t0_arr,  )



  if True:
    fig = plt.figure()
    simWfArr = np.empty((1,numWaveforms, fitSamples))
    for (idx,wf) in enumerate(wfs):
      print "r: %f\nphi %f\nz %f\n e %f\nt0 %f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx])
      simWfArr[0,idx,:] = det.GetSimWaveform(r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], fitSamples)
    helpers.plotManyResidual(simWfArr, wfs, fig, residAlpha=1)
    value = raw_input('  --> Press q to quit, any other key to continue\n')

  if False:
    print "Starting detector MLE..."
    detmleFileName = "P42574A_%dwaveforms_detectormle.npz" % numWaveforms
    nll_det = lambda *args: -lnlike_detector_holdwf(*args)
    detector_startguess = np.hstack((tempGuess, gradGuess,pcRadGuess, num[:], den[1:]))
    wfParams = np.hstack((r_arr[:], phi_arr[:], z_arr[:], scale_arr[:], t0_arr[:],) )
    
    result = op.basinhopping(nll_det, detector_startguess, minimizer_kwargs={"args":(wfs, det, wfParams),  "method":"Powell"})

    temp, impGrad, pcRad, num1, num2, num3, den1, den2, den3 = result["x"]
#    temp, impGrad, pcRad = result["x"][1], result["x"][gradIdx], result["x"][pcRadIdx]
##    r_arr, phi_arr, z_arr, scale_arr, t0_arr = result["x"][:-9].reshape((5, numWaveforms))
#    num = [result["x"][-6], result["x"][-5], result["x"][-4]]
#    den = [1, result["x"][-3], result["x"][-2], result["x"][-1]]

    num = [num1, num2, num3]
    den = [1, den1, den2, den3]

    print "MLE temp is %f" % temp
    print "MLE grad is %f" % impGrad
    print "MLE pc rad is %f" % pcRad
    
    fig2 = plt.figure()
    det.SetTemperature(temp)
    det.SetFields(pcRad, impGrad)
    det.SetTransferFunction(num, den)
    simWfArr = np.empty((1,numWaveforms, fitSamples))
    for (idx,wf) in enumerate(wfs):
      simWfArr[0,idx,:]   = det.GetSimWaveform(r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], fitSamples)
    helpers.plotManyResidual(simWfArr, wfs, fig2, residAlpha=1)

    np.savez(detmleFileName, wfs = wfs, r_arr=r_arr, phi_arr = phi_arr, z_arr = z_arr, scale_arr = scale_arr,  t0_arr=t0_arr,  temp=temp, impGrad=impGrad, pcRad=pcRad)
    value = raw_input('  --> Press q to quit, any other key to continue\n')
    exit(0)

  #Do the MCMC
  ndim = 5*numWaveforms + 3 + 6
  nwalkers = ndim * 3
  mcmc_startguess = np.hstack((r_arr[:], phi_arr[:], z_arr[:], scale_arr[:], t0_arr[:], tempGuess, gradGuess,pcRadGuess, num[:], den[1:]))

  pos0 = [mcmc_startguess + 1e-2*np.random.randn(ndim)*mcmc_startguess for i in range(nwalkers)]

  for pos in pos0:
#    print "radius is %f" % det.detector_radius
#    print "length is %f" % det.detector_length
#  
    pos[:numWaveforms] = np.clip( pos[:numWaveforms], 0, np.floor(det.detector_radius))
    pos[numWaveforms:2*numWaveforms] = np.clip(pos[numWaveforms:2*numWaveforms], 0, np.pi/4)
    pos[2*numWaveforms:3*numWaveforms] = np.clip(pos[2*numWaveforms:3*numWaveforms], 0, det.detector_length)
    pos[4*numWaveforms:5*numWaveforms] = np.clip(pos[4*numWaveforms:5*numWaveforms], 0, fitSamples)
    
    pos[tempIdx] = np.clip(pos[tempIdx], 40, 120)
    pos[gradIdx] = np.clip(pos[gradIdx], det.gradList[0], det.gradList[-1])
    pos[pcRadIdx] = np.clip(pos[pcRadIdx], det.pcRadList[0], det.pcRadList[-1])
  
#    print pos[0:30]

    prior = lnprior(pos, wfs, det)
    if not np.isfinite(prior) :
      print "BAD PRIOR WITH START GUESS YOURE KILLING ME SMALLS"
      exit(0)

  sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(wfs, det), threads=numThreads)
#    f = open("chain.dat", "w")
#    f.close()

  iter, burnIn = 1000, 800
  wfPlotNumber = 10
  
  start = timer()
  
  #w/ progress bar
  bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=iter).start()
  for (idx,result) in enumerate(sampler.sample(pos0, iterations=iter, storechain=True)):
    bar.update(idx+1)

  end = timer()
  bar.finish()
  
  print "Elapsed time: " + str(end-start)

  print "Dumpimng chain to file..."
  np.save("sampler.npy", sampler.chain)
  
#
  print "Making MCMC steps figure..."
  #########  Plots for MC Steps
  stepsFig = plt.figure(2, figsize=(20, 15))
  plt.clf()
  ax0 = stepsFig.add_subplot(511)
  ax1 = stepsFig.add_subplot(512, sharex=ax0)
  ax2 = stepsFig.add_subplot(513, sharex=ax0)
  ax3 = stepsFig.add_subplot(514, sharex=ax0)
  ax4 = stepsFig.add_subplot(515, sharex=ax0)
  
  ax0.set_ylabel('r')
  ax1.set_ylabel('phi')
  ax2.set_ylabel('z')
  ax3.set_ylabel('scale')
  ax4.set_ylabel('t0')

  for i in range(nwalkers):
    for j in range(wfs.size):
      ax0.plot(sampler.chain[i,:,0+j], alpha=0.3)                 # r
      ax1.plot(sampler.chain[i,:,numWaveforms + j], alpha=0.3)    # phi
      ax2.plot(sampler.chain[i,:,2*numWaveforms + j], alpha=0.3)  #z
      ax3.plot(sampler.chain[i,:,3*numWaveforms + j],  alpha=0.3) #energy
      ax4.plot(sampler.chain[i,:,4*numWaveforms + j],  alpha=0.3) #t0

  plt.savefig("emcee_wfchain_%dwfs.png" % numWaveforms)

  stepsFigDet = plt.figure(3, figsize=(20, 15))
  plt.clf()
  ax0 = stepsFigDet.add_subplot(911)
  ax1 = stepsFigDet.add_subplot(912, sharex=ax0)
  ax2 = stepsFigDet.add_subplot(913, sharex=ax0)
  ax3 = stepsFigDet.add_subplot(914, sharex=ax0)
  ax4 = stepsFigDet.add_subplot(915, sharex=ax0)
  ax5 = stepsFigDet.add_subplot(916, sharex=ax0)
  ax6 = stepsFigDet.add_subplot(917, sharex=ax0)
  ax7 = stepsFigDet.add_subplot(918, sharex=ax0)
  ax8 = stepsFigDet.add_subplot(919, sharex=ax0)
  
  ax0.set_ylabel('temp')
  ax1.set_ylabel('grad')
  ax2.set_ylabel('pcRad')
  ax3.set_ylabel('num1')
  ax4.set_ylabel('num2')
  ax5.set_ylabel('num3')
  ax6.set_ylabel('den1')
  ax7.set_ylabel('den2')
  ax8.set_ylabel('den3')

  for i in range(nwalkers):
    ax0.plot(sampler.chain[i,:,tempIdx], "b", alpha=0.3) #temp
    ax1.plot(sampler.chain[i,:,gradIdx], "b", alpha=0.3) #grad
    ax2.plot(sampler.chain[i,:,pcRadIdx], "b", alpha=0.3) #pcrad
    ax3.plot(sampler.chain[i,:,-6], "b", alpha=0.3) #temp
    ax4.plot(sampler.chain[i,:,-5], "b", alpha=0.3) #grad
    ax5.plot(sampler.chain[i,:,-4], "b", alpha=0.3) #pcrad
    ax6.plot(sampler.chain[i,:,-3], "b", alpha=0.3) #temp
    ax7.plot(sampler.chain[i,:,-2], "b", alpha=0.3) #grad
    ax8.plot(sampler.chain[i,:,-1], "b", alpha=0.3) #pcrad


  plt.savefig("emcee_detchain_%dwfs.png" % numWaveforms)


  print "making waveforms figure..."
  det2 =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
  det2.LoadFields("P42574A_fields.npz")

  #pull the samples after burn-in

  samples = sampler.chain[:, burnIn:, :].reshape((-1, ndim))
  simWfs = np.empty((wfPlotNumber, numWaveforms, fitSamples))

  print "temp is %f" % np.median(samples[:,tempIdx])
  print "grad is %f" % np.median(samples[:,gradIdx])
  print "pcrad is %f" % np.median(samples[:,pcRadIdx])
  print "num1 is %f" % np.median(samples[:,-6])
  print "num2 is %f" % np.median(samples[:,-5])
  print "num3 is %f" % np.median(samples[:,-4])
  print "den1 is %f" % np.median(samples[:,-3])
  print "den2 is %f" % np.median(samples[:,-2])
  print "den3 is %f" % np.median(samples[:,-1])

  for idx, (theta) in enumerate(samples[np.random.randint(len(samples), size=wfPlotNumber)]):
    temp, impGrad, pcRad = theta[tempIdx], theta[gradIdx], theta[pcRadIdx]
    r_arr, phi_arr, z_arr, scale_arr, t0_arr = theta[:-9].reshape((5, numWaveforms))
    det2.SetTemperature(temp)
    det2.SetFields(pcRad, impGrad)
    num = [theta[-6], theta[-5], theta[-4]]
    den = [1, theta[-3], theta[-2], theta[-1]]
    det2.SetTransferFunction(num, den)
    
    for wf_idx in range(wfs.size):
      wf_i = det2.GetSimWaveform(r_arr[wf_idx], phi_arr[wf_idx], z_arr[wf_idx], scale_arr[wf_idx], t0_arr[wf_idx], fitSamples)
      simWfs[idx, wf_idx, :] = wf_i
      if wf_i is None:
        print "Waveform %d, %d is None" % (idx, wf_idx)


  residFig = plt.figure(4, figsize=(20, 15))
  helpers.plotManyResidual(simWfs, wfs, figure=residFig)

  plt.savefig("emcee_waveforms_%dwfs.png" % numWaveforms)

  plt.show()
  value = raw_input('  --> Press q to quit, any other key to continue\n')
  if value == 'q':
    exit(0)


if __name__=="__main__":
    main(sys.argv[1:])


