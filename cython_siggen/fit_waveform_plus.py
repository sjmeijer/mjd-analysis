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
from probability_model_hdxwf import *

from progressbar import ProgressBar, Percentage, Bar, ETA
from matplotlib import gridspec
from multiprocessing import Pool

def main(argv):

  plt.ion()

  numThreads = 4
  
  fitSamples = 200
  timeStepSize = 10
  
  wfFileName = "P42574A_16_fastandslow.npz"
#  wfFileName = "fep_event_set_runs11531-11539.npz"
#  numWaveforms = 16
#  wfFileName = "P42574A_512waveforms_%drisetimeculled.npz" % numWaveforms
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    wfs = data['wfs']
    numWaveforms = wfs.size
  else:
    print "No saved waveforms available.  Loading from Data"
    exit(0)
  
  zero_1 = 0.48
  pole_1 = 0.999848
  pole_real = 0.798211
  pole_imag = 0.079924

  zeros = [zero_1, -1., 1. ]
  poles = [pole_1, pole_real+pole_imag*1j, pole_real-pole_imag*1j, ]
  
  tempGuess = 78.655244
  gradGuess = 0.045049
  pcRadGuess = 2.574859
  pcLenGuess = 1.524812


  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10./timeStepSize, poles=poles, zeros=zeros)
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess, method="Full")
  initializeDetector(det, )

  #ML as a start
  nll = lambda *args: -lnlike_waveform(*args)
  
  for (idx,wf) in enumerate(wfs):
#    if idx <= 3: continue

    if wf.energy < 1700: continue

    fig1 = plt.figure(1)
    plt.clf()
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.set_xlabel("Digitizer Time [ns]")
    ax0.set_ylabel("Voltage [Arb.]")
    ax1.set_ylabel("Residual")
    
    
    wf.WindowWaveformTimepoint(fallPercentage=.995, rmsMult=2)
    initializeWaveform(wf)
    dataLen = wf.wfLength
    t_data = np.arange(dataLen) * 10
    
    ax0.plot(t_data, wf.windowedWf, color="r")
  
    startGuess = [15., np.pi/8, 15., wf.wfMax, wf.t0Guess, 10, tempGuess, zero_1]
    init_wf = det.MakeSimWaveform(15, np.pi/8, 15, wf.wfMax, wf.t0Guess, fitSamples, h_smoothing=10)

    result = op.minimize(nll, startGuess,   method="Powell")
    #result = op.basinhopping(nll, startGuess,   T=1000,stepsize=15, minimizer_kwargs= {"method": "Nelder-Mead", "args":(wf)})
    r, phi, z, scale, t0, smoove, temp, zero = result["x"]
    zeros = [zero, -1., 1. ]
    det.SetTransferFunction(zeros, poles)
    det.SetTemperature(temp)
    
    print "Waveform number %d" % idx
    print "   " , r, phi, z, scale, t0, smoove
    print "   Initial LN like is %f" %  (result['fun'])

    ml_wf = np.copy(det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing=smoove ))
    ax0.plot(t_data, ml_wf[:dataLen], color="b")
    ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color="b")
    
    
    r_new = np.amin( [z, np.floor(det.detector_radius)-2] )
    z_new = np.amin( [r, np.floor(det.detector_length)-2] )

    result2 = op.minimize(nll, [r_new, np.pi/4, z_new, scale, wf.t0Guess, 1,tempGuess, zero_1],  method="Powell")
    r, phi, z, scale, t0, smoove,temp, zero =  result2["x"]
    zeros = [zero, -1., 1. ]
    det.SetTransferFunction(zeros, poles)
    det.SetTemperature(temp)
    
    inv_ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing=smoove)
    ax0.plot(t_data,inv_ml_wf[:dataLen], color="g")
    print "   ", r, phi, z, scale, t0, smoove
    ax1.plot(t_data,inv_ml_wf[:dataLen] -  wf.windowedWf, color="g")
    

    ax1.set_ylim(-20,20)


#    if result2['fun'] < result['fun']:
#      mcmc_startguess = result2["x"]
#    else:
#      mcmc_startguess = result["x"]


    print "   Inverted LN like is %f" %  (result2['fun'])

    value = raw_input('  --> Press q to quit, s to skip, any other key to continue\n')
    if value == 'q':
      exit(0)
    if value == 's':
      continue
#    else:
#      continue

    #Do the MCMC
    ndim, nwalkers = 9, 200
    mcmc_startguess = startGuess#np.array([r, phi, z, scale, t0, smooth])
    
    pos0 = [mcmc_startguess + 1e-2*np.random.randn(ndim)*mcmc_startguess for i in range(nwalkers)]
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
    plotnum = 800
    ax0 = stepsFig.add_subplot(plotnum+11)
    ax1 = stepsFig.add_subplot(plotnum+12, sharex=ax0)
    ax2 = stepsFig.add_subplot(plotnum+13, sharex=ax0)
    ax3 = stepsFig.add_subplot(plotnum+14, sharex=ax0)
    ax4 = stepsFig.add_subplot(plotnum+15, sharex=ax0)
    ax5 = stepsFig.add_subplot(plotnum+16, sharex=ax0)
    ax6 = stepsFig.add_subplot(plotnum+17, sharex=ax0)
    ax7 = stepsFig.add_subplot(plotnum+18, sharex=ax0)
#    ax8 = stepsFig.add_subplot(plotnum+19, sharex=ax0)

    ax0.set_ylabel('r')
    ax1.set_ylabel('phi')
    ax2.set_ylabel('z')
    ax3.set_ylabel('scale')
    ax4.set_ylabel('t0')
    ax5.set_ylabel('smooth')
    ax6.set_ylabel('temp')
    ax7.set_ylabel('zero1')
#    ax6.set_ylabel('temp')

    for i in range(nwalkers):
      ax0.plot(sampler.chain[i,:,0], "b", alpha=0.3)
      ax1.plot(sampler.chain[i,:,1], "b", alpha=0.3)
      ax2.plot(sampler.chain[i,:,2], "b", alpha=0.3)
      ax3.plot(sampler.chain[i,:,3], "b", alpha=0.3)
      ax4.plot(sampler.chain[i,:,4], "b", alpha=0.3)
      ax5.plot(sampler.chain[i,:,5], "b", alpha=0.3)
      ax6.plot(sampler.chain[i,:,6], "b", alpha=0.3)
      ax7.plot(sampler.chain[i,:,7], "b", alpha=0.3)

    #pull the samples after burn-in

    samples = sampler.chain[:, burnIn:, :].reshape((-1, ndim))
    wfPlotNumber = 100
    simWfs = np.empty((wfPlotNumber, fitSamples) )

    for idx, (r, phi, z, scale, t0, smooth, temp, zero) in enumerate(samples[np.random.randint(len(samples), size=wfPlotNumber)]):
      zeros = [zero, -1., 1. ]
      det.SetTransferFunction(zeros, poles)
      det.SetTemperature(temp)
      simWfs[idx,:] = det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing = smooth,)

    residFig = plt.figure(3)
    helpers.plotResidual(simWfs, wf.windowedWf, figure=residFig)


#    plt.show()
    value = raw_input('  --> Press q to quit, any other key to continue\n')
    if value == 'q':
      exit(0)


if __name__=="__main__":
    main(sys.argv[1:])


