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
  
  tempGuess = 81
  fitSamples = 150
  numWaveforms = 20
  
  #Prepare detector
  num = [3.64e+09, 1.88e+17, 6.05e+15]
  den = [1, 4.03e+07, 5.14e+14, 7.15e+18]
  system = signal.lti(num, den)
  
  gradGuess = 0.05
  pcRadGuess = 2.55
  
  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f.conf" % (gradGuess,pcRadGuess)
  det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
  det.LoadFields("P42574A_fields.npz")
  
  
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
      wf.WindowWaveformTimepoint(fallPercentage=.995)
      startGuess = [15., np.pi/8, 15., wf.wfMax, wf.t0Guess]
      
      result = op.minimize(nll_wf, startGuess, args=(wf, det),  method="Powell")
      r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx] = result["x"]

    np.savez(wfFileName, wfs = wfs, r_arr=r_arr, phi_arr = phi_arr, z_arr = z_arr, scale_arr = scale_arr,  t0_arr=t0_arr,  )


  if False:
    fig = plt.figure()
    for (idx,wf) in enumerate(wfs):
      print "r: %f\nphi %f\nz %f\n e %f\nt0 %f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx])
      ml_wf = det.GetSimWaveform(r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], fitSamples)
      plt.plot(ml_wf, color="b")
      plt.plot(wf.windowedWf, color="r")
    value = raw_input('  --> Press q to quit, any other key to continue\n')

  if False:
    print "Starting detector MLE..."
    detmleFileName = "P42574A_%dwaveforms_detectormle.npz" % numWaveforms
    nll_det = lambda *args: -lnlike_detector(*args)
    detector_startguess = np.hstack((r_arr[:], phi_arr[:], z_arr[:], scale_arr[:], t0_arr[:], tempGuess, gradGuess,pcRadGuess))
    result = op.minimize(nll_det, detector_startguess, args=(wfs, det),  method="Powell")
    temp, impGrad, pcRad = result["x"][-3:]
    r_arr, phi_arr, z_arr, scale_arr, t0_arr = result["x"][:-3].reshape((5, numWaveforms))
    
    print "MLE temp is %f" % temp
    print "MLE grad is %f" % impGrad
    print "MLE pc rad is %f" % pcRad
    
    fig = plt.figure()
    det.SetTemperature(temp)
    det.SetFields(pcRad, impGrad)
    for (idx,wf) in enumerate(wfs):
      ml_wf = det.GetSimWaveform(r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], fitSamples)
    
      plt.plot(ml_wf, color="b")
      plt.plot(wf.windowedWf, color="r")
    
    np.savez(detmleFileName, wfs = wfs, r_arr=r_arr, phi_arr = phi_arr, z_arr = z_arr, scale_arr = scale_arr,  t0_arr=t0_arr,  temp=temp, impGrad=impGrad, pcRad=pcRad)
    plt.show()
    value = raw_input('  --> Press q to quit, any other key to continue\n')
    exit(0)

  #Do the MCMC
  ndim = 5*numWaveforms + 3
  nwalkers = ndim * 3+1
  mcmc_startguess = np.hstack((r_arr[:], phi_arr[:], z_arr[:], scale_arr[:], t0_arr[:], tempGuess, gradGuess,pcRadGuess))

  pos0 = [mcmc_startguess + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]

  for pos in pos0:
#    print "radius is %f" % det.detector_radius
#    print "length is %f" % det.detector_length
#  
    pos[:numWaveforms] = np.clip( pos[:numWaveforms], 0, np.floor(det.detector_radius))
    pos[numWaveforms:2*numWaveforms] = np.clip(pos[numWaveforms:2*numWaveforms], 0, np.pi/4)
    pos[2*numWaveforms:3*numWaveforms] = np.clip(pos[2*numWaveforms:3*numWaveforms], 0, det.detector_length)
    pos[4*numWaveforms:5*numWaveforms] = np.clip(pos[4*numWaveforms:5*numWaveforms], 0, fitSamples)
    
    pos[-3] = np.clip(pos[-3], 40, 120)
    pos[-2] = np.clip(pos[-2], det.gradList[0], det.gradList[-1])
    pos[-1] = np.clip(pos[-1], det.pcRadList[0], det.pcRadList[-1])
  
#    print pos[0:30]

    prior = lnprior(pos, wfs, det)
    if not np.isfinite(prior) :
      print "BAD PRIOR WITH START GUESS YOURE KILLING ME SMALLS"
      exit(0)

  sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(wfs, det), threads=8)
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
  ax0 = stepsFig.add_subplot(811)
  ax1 = stepsFig.add_subplot(812, sharex=ax0)
  ax2 = stepsFig.add_subplot(813, sharex=ax0)
  ax3 = stepsFig.add_subplot(814, sharex=ax0)
  ax4 = stepsFig.add_subplot(815, sharex=ax0)
  ax5 = stepsFig.add_subplot(816, sharex=ax0)
  ax6 = stepsFig.add_subplot(817, sharex=ax0)
  ax7 = stepsFig.add_subplot(818, sharex=ax0)
  
  ax0.set_ylabel('r')
  ax1.set_ylabel('phi')
  ax2.set_ylabel('z')
  ax3.set_ylabel('scale')
  ax4.set_ylabel('t0')
  ax5.set_ylabel('temp')
  ax6.set_ylabel('grad')
  ax7.set_ylabel('pcRad')

  for i in range(nwalkers):
    for j in range(wfs.size):
      ax0.plot(sampler.chain[i,:,0+j], alpha=0.3)                 # r
      ax1.plot(sampler.chain[i,:,numWaveforms + j], alpha=0.3)    # phi
      ax2.plot(sampler.chain[i,:,2*numWaveforms + j], alpha=0.3)  #z
      ax3.plot(sampler.chain[i,:,3*numWaveforms + j],  alpha=0.3) #energy
      ax4.plot(sampler.chain[i,:,4*numWaveforms + j],  alpha=0.3) #t0
    ax5.plot(sampler.chain[i,:,-3], "b", alpha=0.3) #temp
    ax6.plot(sampler.chain[i,:,-2], "b", alpha=0.3) #grad
    ax7.plot(sampler.chain[i,:,-1], "b", alpha=0.3) #pcrad


  plt.savefig("emcee_chain_%dwfs.png" % numWaveforms)


  print "making waveforms figure..."
  det2 =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
  det2.LoadFields("P42574A_fields.npz")

  #pull the samples after burn-in

  samples = sampler.chain[:, burnIn:, :].reshape((-1, ndim))
  simWfs = np.empty((wfPlotNumber, numWaveforms, fitSamples))

  print "temp is %f" % np.median(samples[:,-3])
  print "grad is %f" % np.median(samples[:,-2])
  print "pcrad is %f" % np.median(samples[:,-1])

  for idx, (theta) in enumerate(samples[np.random.randint(len(samples), size=wfPlotNumber)]):
    temp, impGrad, pcRad = theta[-3:]
    r_arr, phi_arr, z_arr, scale_arr, t0_arr = theta[:-3].reshape((5, numWaveforms))
#    print "temp is %f, impGrad is %f, pcRad is %f" % (temp, impGrad, pcRad)
    det2.SetTemperature(temp)
    det2.SetFields(pcRad, impGrad)
    for wf_idx in range(wfs.size):
      wf_i = det2.GetSimWaveform(r_arr[wf_idx], phi_arr[wf_idx], z_arr[wf_idx], scale_arr[wf_idx], t0_arr[wf_idx], fitSamples)
      simWfs[idx, wf_idx, :] = wf_i
      if wf_i is None:
        print "Waveform %d, %d is None" % (idx, wf_idx)


  residFig = plt.figure(3)
  helpers.plotManyResidual(simWfs, wfs, figure=residFig)

  plt.savefig("emcee_waveforms_%dwfs.png" % numWaveforms)

  plt.show()
  value = raw_input('  --> Press q to quit, any other key to continue\n')
  if value == 'q':
    exit(0)


if __name__=="__main__":
    main(sys.argv[1:])


