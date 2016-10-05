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
  timeStepSize = 1
  
  wfFileName = "fep_old_postpsa.npz"
  numWaveforms = 30
  #wfFileName = "P42574A_512waveforms_%drisetimeculled.npz" % numWaveforms
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    wfs = data['wfs']
    numWaveforms = wfs.size
  else:
    print "No saved waveforms available.  Loading from Data"
    exit(0)
  
  #Prepare detector
  tempGuess = 79.310080
  gradGuess = 0.051005
  pcRadGuess = 2.499387
  pcLenGuess = 1.553464

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10./timeStepSize,)
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  
  b_over_a = 0.107213
  c = -0.821158
  d = 0.828957
  rc = 76.710043
  det.SetTransferFunction(b_over_a, c, d, rc)
  
  initializeDetector(det, )

  #ML as a start
  nll = lambda *args: -lnlike_waveform(*args)
  
  for (idx,wf) in enumerate(wfs):
    print "Waveform number %d" % idx
    if idx < 22: continue
  
    fig1 = plt.figure(1)
    plt.clf()
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.set_xlabel("Digitizer Time [ns]")
    ax0.set_ylabel("Voltage [Arb.]")
    ax1.set_ylabel("Residual")
    
    
    wf.WindowWaveformTimepoint(fallPercentage=.999, rmsMult=2)
    initializeWaveform(wf)
    dataLen = wf.wfLength
    t_data = np.arange(dataLen) * 10
    
    ax0.plot(t_data, wf.windowedWf, color="r")
  
    startGuess = [15., np.pi/8, 15., wf.wfMax, wf.t0Guess-5, 10]
    init_wf = det.MakeSimWaveform(15, np.pi/8, 15, wf.wfMax, wf.t0Guess, fitSamples, h_smoothing=10)

    result = op.minimize(nll, startGuess,   method="Powell")
    #result = op.basinhopping(nll, startGuess,   T=1000,stepsize=15, minimizer_kwargs= {"method": "Nelder-Mead", "args":(wf)})
    r, phi, z, scale, t0, smooth, = result["x"]
    
    r_new, z_new = det.ReflectPoint(r,z)

    print "Initial LN like is %f" %  (result['fun'])
    
#    print r, phi, z, scale, t0, smooth
    
    ml_wf = np.copy(det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing=smooth, ))
    ax0.plot(t_data, ml_wf[:dataLen], color="b")
    ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color="b")
    
    result2 = op.minimize(nll, [r_new, phi, z_new, scale, wf.t0Guess-5,10],  method="Powell")
    r, phi, z, scale, t0, smooth, = result2["x"]
    inv_ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing=smooth, )
    ax0.plot(t_data,inv_ml_wf[:dataLen], color="g")
#    print r, phi, z, scale, t0, smooth
    ax1.plot(t_data,inv_ml_wf[:dataLen] -  wf.windowedWf, color="g")

    ax1.set_ylim(-20,20)

    if result2['fun'] < result['fun']:
      mcmc_startguess = result2["x"]
    else:
      mcmc_startguess = result["x"]

    wf.prior = mcmc_startguess

    print "Inverted LN like is %f" %  (result2['fun'])

    value = raw_input('  --> Press q to quit, s to skip, any other key to continue\n')
    if value == 'q':
      exit(0)
    if value == 's':
      continue

    #Do the MCMC
    ndim, nwalkers = 6, 12
    mcmc_startguess = startGuess#np.array([r, phi, z, scale, t0, smooth])
    
    ntemps = 512
    pos0 = np.random.uniform(low=0.99, high=1.01, size=(ntemps, nwalkers, ndim))
    for tidx in range(ntemps):
      for widx in range(nwalkers):
        pos0[tidx, widx, :] = pos0[tidx, widx, :] * mcmc_startguess

    
    
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_waveform, args=([wf]))
    p = Pool(numThreads, initializer=initializeWaveform, initargs=[ wf])
    sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlike_waveform, lnprob_waveform, pool=p, )

    iter, burnIn = 1000, 900
    
    bar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=iter).start()
    for (idx,result) in enumerate(sampler.sample(pos0, iterations=iter, storechain=True)):
      bar.update(idx+1)
    bar.finish()
    p.close()
        
    #samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

#    #########  Plots for MC Steps
#    stepsFig = plt.figure(2)
#    plt.clf()
#    ax0 = stepsFig.add_subplot(611)
#    ax1 = stepsFig.add_subplot(612, sharex=ax0)
#    ax2 = stepsFig.add_subplot(613, sharex=ax0)
#    ax3 = stepsFig.add_subplot(614, sharex=ax0)
#    ax4 = stepsFig.add_subplot(615, sharex=ax0)
#    ax5 = stepsFig.add_subplot(616, sharex=ax0)
##    ax6 = stepsFig.add_subplot(717, sharex=ax0)
#
#    ax0.set_ylabel('r')
#    ax1.set_ylabel('phi')
#    ax2.set_ylabel('z')
#    ax3.set_ylabel('scale')
#    ax4.set_ylabel('t0')
#    ax5.set_ylabel('smooth')
##    ax6.set_ylabel('esmooth')
#
#    for k in range(ntemps):
#      for i in range(nwalkers):
#        ax0.plot(sampler.chain[k,i,:,0], "b", alpha=0.3)
#        ax1.plot(sampler.chain[k,i,:,1], "b", alpha=0.3)
#        ax2.plot(sampler.chain[k,i,:,2], "b", alpha=0.3)
#        ax3.plot(sampler.chain[k,i,:,3], "b", alpha=0.3)
#        ax4.plot(sampler.chain[k,i,:,4], "b", alpha=0.3)
#        ax5.plot(sampler.chain[k,i,:,5], "b", alpha=0.3)
##      ax6.plot(sampler.chain[i,:,6], "b", alpha=0.3)
#
#    #pull the samples after burn-in


    samples = sampler.chain[:,:, burnIn:, :].reshape((-1, ndim))
    wfPlotNumber = 100
    simWfs = np.empty((wfPlotNumber, fitSamples) )


    for idx, (r, phi, z, scale, t0, smooth, ) in enumerate(samples[np.random.randint(len(samples), size=wfPlotNumber)]):
      simWfs[idx,:] = det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing = smooth,)
    
#    plt.ioff()
    plt.ion()
    residFig = plt.figure(3)
    helpers.plotResidual(simWfs, wf.windowedWf, figure=residFig)
    plt.savefig("waveform_fig.png")
    
    
    plt.figure(4)
    plt.hist(samples[:,0], bins=40)
    plt.xlabel("radius [mm]")
    
    plt.figure(5)
    plt.hist(samples[:,2], bins=40)
    plt.ylabel("height [mm]")
   
   

#    lnprobFig = plt.figure(4)
#    plt.clf()
#    lnprobs = sampler.lnprobability[:, burnIn:].reshape((-1))
#    median_prob = np.median(lnprobs)
#    print "median lnprob is %f (length normalized: %f)" % (median_prob, median_prob/wf.wfLength)
#    plt.hist(lnprobs)
#    
    positionFig = plt.figure(6)
    plt.hist2d(samples[:,0], samples[:,2],  norm=LogNorm(), bins=[ np.around(det.detector_radius,1)*10,np.around(det.detector_length,1)*10  ])
    plt.xlim(0, det.detector_radius)
    plt.ylim(0, det.detector_length)
    plt.colorbar()
    
    lnprobFig = plt.figure(7)
    plt.clf()
    lnprobs = sampler.lnprobability[:, :, burnIn:].reshape((-1))
    median_prob = np.median(lnprobs)
    print "median lnprob is %f (length normalized: %f)" % (median_prob, median_prob/wf.wfLength)
    hist, bins = np.histogram(lnprobs/wf.wfLength, bins=np.linspace(-10,0,1000))
    plt.plot(bins[:-1], hist, ls="steps-post")

#    plt.show()
    value = raw_input('  --> Press q to quit, any other key to continue\n')
    if value == 'q':
      exit(0)


if __name__=="__main__":
    main(sys.argv[1:])


