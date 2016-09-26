#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.optimize as op
import numpy as np
import emcee
from scipy import signal

import helpers
from detector_model import *
from probability_model_waveform import *

from progressbar import ProgressBar, Percentage, Bar, ETA
from matplotlib import gridspec
from multiprocessing import Pool

def main(argv):

  plt.ion()

  runRange = (13420,13429)
  channel = 626
  aeCutVal = 0.01425
  numThreads = 8
  
  fitSamples = 200
  timeStepSize = 10
  
  wfFileName = "multisite_event_set_runs13385-13557.npz"
#  numWaveforms = 16
#  wfFileName = "P42574A_512waveforms_%drisetimeculled.npz" % numWaveforms
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    wfs = data['wfs']
    numWaveforms = wfs.size
  else:
    print "No saved waveforms available.  Loading from Data"
    exit(0)
  
  #Prepare detector
  tempGuess = 78.421865
  gradGuess = 0.047338
  pcRadGuess = 2.550635
  pcLenGuess = 1.546489
  zero_1 = 0.453902
  pole_1 = 0.999862
  pole_real = 0.797921
  pole_imag = 0.080834
  
  zeros = [zero_1, -1., 1. ]
  poles = [pole_1, pole_real+pole_imag*1j, pole_real-pole_imag*1j, ]
  
#  tempGuess = 78.103233
#  gradGuess = 0.047049
#  pcRadGuess = 2.541187
#  pcLenGuess = 1.568158

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10./timeStepSize, poles=poles, zeros=zeros)
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  initializeDetector(det, )

  #ML as a start
  nll = lambda *args: -lnlike_waveform(*args)
  
  for (idx,wf) in enumerate(wfs):
#    if idx == 0: continue

    if wf.energy > 1700: continue

    fig1 = plt.figure(1)
    plt.clf()
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.set_xlabel("Digitizer Time [ns]")
    ax0.set_ylabel("Voltage [Arb.]")
    ax1.set_ylabel("Residual")
    
    
    wf.WindowWaveformTimepoint(fallPercentage=.99)
    initializeWaveform(wf)
    dataLen = wf.wfLength
    t_data = np.arange(dataLen) * 10
    
    ax0.plot(t_data, wf.windowedWf, color="r")
  
    startGuess = [15., np.pi/8, 15., wf.wfMax, wf.t0Guess, 1]
    init_wf = det.GetSimWaveform(15, np.pi/8, 15, wf.wfMax, wf.t0Guess, fitSamples)

    result = op.minimize(nll, startGuess,   method="Powell")
    #result = op.basinhopping(nll, startGuess,   T=1000,stepsize=15, minimizer_kwargs= {"method": "Nelder-Mead", "args":(wf)})
    r, phi, z, scale, t0, smooth, = result["x"]
    r_new = np.amin( [z, np.floor(det.detector_radius)] )
    z_new = np.amin( [r, np.floor(det.detector_length)] )
    print "Initial LN like is %f" %  (result['fun'])
    
#    print r, phi, z, scale, t0, smooth
    
    ml_wf = np.copy(det.GetSimWaveform(r, phi, z, scale, t0, fitSamples, smoothing=smooth, ))
    ax0.plot(t_data, ml_wf[:dataLen], color="b")
    ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color="b")
    
    result2 = op.minimize(nll, [z, phi, r, scale, wf.t0Guess,1],  method="Powell")
    r, phi, z, scale, t0, smooth, = result2["x"]
    inv_ml_wf = det.GetSimWaveform(r, phi, z, scale, t0, fitSamples, smoothing=smooth, )
    ax0.plot(t_data,inv_ml_wf[:dataLen], color="g")
#    print r, phi, z, scale, t0, smooth
    ax1.plot(t_data,inv_ml_wf[:dataLen] -  wf.windowedWf, color="g")
    

    ax1.set_ylim(-20,20)


#    if result2['fun'] < result['fun']:
#      mcmc_startguess = result2["x"]
#    else:
#      mcmc_startguess = result["x"]


    print "Inverted LN like is %f" %  (result2['fun'])

    value = raw_input('  --> Press q to quit, s to skip, any other key to continue\n')
    if value == 'q':
      exit(0)
    if value == 's':
      continue

    #Do the MCMC
    ndim, nwalkers = 6, 200
    mcmc_startguess = startGuess#np.array([r, phi, z, scale, t0, smooth])
    
    pos0 = [mcmc_startguess + 1e-1*np.random.randn(ndim)*mcmc_startguess for i in range(nwalkers)]
    p = Pool(numThreads, initializer=initializeWaveform, initargs=[ wf])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_waveform, )


    iter, burnIn = 1000, 900
    
    bar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=iter).start()
    for (idx,result) in enumerate(sampler.sample(pos0, iterations=iter, storechain=True)):
      bar.update(idx+1)
    bar.finish()
    p.close()
        
    #samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

    #########  Plots for MC Steps
    stepsFig = plt.figure(2)
    plt.clf()
    ax0 = stepsFig.add_subplot(611)
    ax1 = stepsFig.add_subplot(612, sharex=ax0)
    ax2 = stepsFig.add_subplot(613, sharex=ax0)
    ax3 = stepsFig.add_subplot(614, sharex=ax0)
    ax4 = stepsFig.add_subplot(615, sharex=ax0)
    ax5 = stepsFig.add_subplot(616, sharex=ax0)
#    ax6 = stepsFig.add_subplot(717, sharex=ax0)

    ax0.set_ylabel('r')
    ax1.set_ylabel('phi')
    ax2.set_ylabel('z')
    ax3.set_ylabel('scale')
    ax4.set_ylabel('t0')
    ax5.set_ylabel('smooth')
#    ax6.set_ylabel('esmooth')

    for i in range(nwalkers):
      ax0.plot(sampler.chain[i,:,0], "b", alpha=0.3)
      ax1.plot(sampler.chain[i,:,1], "b", alpha=0.3)
      ax2.plot(sampler.chain[i,:,2], "b", alpha=0.3)
      ax3.plot(sampler.chain[i,:,3], "b", alpha=0.3)
      ax4.plot(sampler.chain[i,:,4], "b", alpha=0.3)
      ax5.plot(sampler.chain[i,:,5], "b", alpha=0.3)
#      ax6.plot(sampler.chain[i,:,6], "b", alpha=0.3)

    #pull the samples after burn-in

    samples = sampler.chain[:, burnIn:, :].reshape((-1, ndim))
    wfPlotNumber = 100
    simWfs = np.empty((wfPlotNumber, fitSamples) )

    for idx, (r, phi, z, scale, t0, smooth, ) in enumerate(samples[np.random.randint(len(samples), size=wfPlotNumber)]):
      simWfs[idx,:] = det.GetSimWaveform(r, phi, z, scale, t0, fitSamples, smoothing = smooth,)

    residFig = plt.figure(3)
    helpers.plotResidual(simWfs, wf.windowedWf, figure=residFig)

    lnprobFig = plt.figure(4)
    plt.clf()
    lnprobs = sampler.lnprobability[:, burnIn:].reshape((-1))
    median_prob = np.median(lnprobs)
    print "median lnprob is %f (length normalized: %f)" % (median_prob, median_prob/wf.wfLength)
    plt.hist(lnprobs)
    
    positionFig = plt.figure(5)
    plt.clf()
    xedges = np.linspace(0, np.around(det.detector_radius,1), np.around(det.detector_radius,1)*10+1)
    yedges = np.linspace(0, np.around(det.detector_radius,1), np.around(det.detector_radius,1)*10+1)
    plt.hist2d(samples[:,0], samples[:,2],  norm=LogNorm(), bins=[ xedges,yedges  ])
    plt.colorbar()
    plt.xlabel("r from Point Contact (mm)")
    plt.ylabel("z from Point Contact (mm)")

    phiFig = plt.figure(6)
    plt.clf()
    plt.hist(samples[:,1], bins=np.linspace(0, np.pi/4, 100),  )
    plt.xlabel("Phi (0 to pi/4)")
    plt.ylabel("Marginalized Posterior Probability")
#    plt.xlim(0, det.detector_radius)
#    plt.ylim(0, det.detector_length)

    phiFig = plt.figure(7)
    plt.clf()
    plt.hist(samples[:,4]*10,   bins=np.linspace(100, 299, 2000),)
    plt.xlim(np.amin(samples[:,4]*10), np.amax(samples[:,4]*10))
    plt.xlabel("t0 [ns]" )
    plt.ylabel("Marginalized Posterior Probability")

    print "1 sigma percentiles: "
    print " r = %f (%f,%f)" % (np.median(samples[:,0]), np.percentile(samples[:,0], 50-34), np.percentile(samples[:,0], 50+34))
    print " phi = %f (%f,%f)" % (np.median(samples[:,1]), np.percentile(samples[:,1], 50-34), np.percentile(samples[:,1], 50+34))
    print " z = %f (%f,%f)" % (np.median(samples[:,2]), np.percentile(samples[:,2], 50-34), np.percentile(samples[:,2], 50+34))
    print " t0 = %f (%f,%f)" % (np.median(samples[:,4]), np.percentile(samples[:,4], 50-34), np.percentile(samples[:,4], 50+34))


#    plt.show()
    value = raw_input('  --> Press q to quit, any other key to continue\n')
    if value == 'q':
      exit(0)


if __name__=="__main__":
    main(sys.argv[1:])


