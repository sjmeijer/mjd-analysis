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
from probability_model import *

from progressbar import ProgressBar, Percentage, Bar
from timeit import default_timer as timer

def main(argv):

  plt.ion()

  runRange = (13420,13429)
  channel = 626
  aeCutVal = 0.01425
  
  fitSamples = 150
  numWaveforms = 2
  
  #get waveforms
  wfFileName = "P42574A_32waveforms_risetimeculled.npz"
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    wfs = data['wfs']
  else:
    exit(0)
  
  
#  cut = "trapECal>%f && trapECal<%f && TSCurrent100nsMax/trapECal > %f" %  (1588,1594, aeCutVal)
#  wfs = helpers.GetWaveforms(runRange, channel, numWaveforms, cut)

  num = [8772059139.7583485, 1.7795141663115218e+18, 17696700392300130.0]
  den = [1, 50813678.171704151, 708456403501206.5, 1.4152530477835272e+19]
  system = signal.lti(num, den)
  
  tempGuess = 77
  gradGuess = 0.0487
  pcRadGuess = 2.535691
  pcLenGuess = 1.655159

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.04,2.5, 1.6)
  det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
  det.LoadFields("P42574A_fields_len.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)


#  #plot to make sure you like what you see
#  plt.figure()
#  for wf in wfs:
#    wf.WindowWaveform(fitSamples)
#    plt.plot(wf.windowedWf, color="r")
#  
#  sim_wf = det.GetSimWaveform(10, 0, 10, wfs[0].wfMax, wfs[0].t0Guess, fitSamples)
#  plt.plot(sim_wf, color="b")
#
#  plt.show()

  #ML as a start
  nll = lambda *args: -lnlike(*args)
  
  for (idx,wf) in enumerate(wfs):
#    if idx == 0: continue
    plt.figure(1)
  
    wf.WindowWaveformTimepoint()
    startGuess = [15., np.pi/8, 15., wf.wfMax, wf.t0Guess, 3., 1.]
    init_wf = det.GetSimWaveform(15, np.pi/8, 15, wf.wfMax, wf.t0Guess, fitSamples)
    plt.plot(init_wf, color="g")
    
    result = op.minimize(nll, startGuess, args=(wf.windowedWf, det,  wf.baselineRMS),  method="Powell")
    r, phi, z, scale, t0, smooth, esmooth= result["x"]
    
    plt.plot(wf.windowedWf, color="r")
    
    ml_wf = det.GetSimWaveform(r, phi, z, scale, t0, fitSamples, smoothing=smooth, electron_smoothing = esmooth)
    ml_wf_inv = det.GetSimWaveform(z, phi, r, scale, t0, fitSamples, smoothing=smooth, electron_smoothing = esmooth)
    
    plt.plot(ml_wf, color="b")
    plt.plot(ml_wf_inv, "b:")

#    print result["x"]
#    plt.show()

#    plt.figure(2)
#    plt.scatter(r, z)


    #Do the MCMC
    ndim, nwalkers = 7, 100
    mcmc_startguess = np.array([r, phi, z, scale, t0, smooth, esmooth])
    
    pos0 = [mcmc_startguess + 1e-1*np.random.randn(ndim)*mcmc_startguess for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(wf.windowedWf, det,  wf.baselineRMS, wf.wfMax, wf.t0Guess))
#    f = open("chain.dat", "w")
#    f.close()


    iter, burnIn = 2000, 1500
    
    bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=iter).start()
    start = timer()
    for (idx,result) in enumerate(sampler.sample(pos0, iterations=iter, storechain=True)):
      bar.update(idx+1)
#      position = result[0]
#      f = open("chain.dat", "a")
#      for k in range(position.shape[0]):
#        print " ".join(str(position[k]))
##        f.write( "{0:4d} {1:s}\n".format(k, " ".join(position[k])))
#        f.close()

#    sampler.run_mcmc(pos, 500)
    end = timer()
    bar.finish()
    
    print "Elapsed time: " + str(end-start)
    
    #samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

    #########  Plots for MC Steps
    stepsFig = plt.figure(2)
    plt.clf()
    ax0 = stepsFig.add_subplot(711)
    ax1 = stepsFig.add_subplot(712, sharex=ax0)
    ax2 = stepsFig.add_subplot(713, sharex=ax0)
    ax3 = stepsFig.add_subplot(714, sharex=ax0)
    ax4 = stepsFig.add_subplot(715, sharex=ax0)
    ax5 = stepsFig.add_subplot(716, sharex=ax0)
    ax6 = stepsFig.add_subplot(717, sharex=ax0)
    
    ax0.set_ylabel('r')
    ax1.set_ylabel('phi')
    ax2.set_ylabel('z')
    ax3.set_ylabel('scale')
    ax4.set_ylabel('t0')
    ax5.set_ylabel('smooth')
    ax6.set_ylabel('esmooth')

    for i in range(nwalkers):
      ax0.plot(sampler.chain[i,:,0], "b", alpha=0.3)
      ax1.plot(sampler.chain[i,:,1], "b", alpha=0.3)
      ax2.plot(sampler.chain[i,:,2], "b", alpha=0.3)
      ax3.plot(sampler.chain[i,:,3], "b", alpha=0.3)
      ax4.plot(sampler.chain[i,:,4], "b", alpha=0.3)
      ax5.plot(sampler.chain[i,:,5], "b", alpha=0.3)
      ax6.plot(sampler.chain[i,:,6], "b", alpha=0.3)

    #pull the samples after burn-in

    samples = sampler.chain[:, burnIn:, :].reshape((-1, ndim))
    wfPlotNumber = 100
    simWfs = np.empty(wfPlotNumber, dtype=object)
    for idx, (r, phi, z, scale, t0, smooth, e_smooth) in enumerate(samples[np.random.randint(len(samples), size=wfPlotNumber)]):
      simWfs[idx] = det.GetSimWaveform(r, phi, z, scale, t0, fitSamples, smoothing = smooth, electron_smoothing = e_smooth)


    residFig = plt.figure(3)
    helpers.plotResidual(simWfs, wf.windowedWf, figure=residFig)

#    plt.show()
    value = raw_input('  --> Press q to quit, any other key to continue\n')
    if value == 'q':
      exit(0)


if __name__=="__main__":
    main(sys.argv[1:])


