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

  numThreads = 8
  
  fitSamples = 350
  timeStepSize = 1
  
  wfFileName = "fep_old_postpsa.npz"
#  wfFileName = "P42574A_pc_highe.npz"
  #wfFileName = "fep_event_set_runs11531-11539.npz"
#  numWaveforms = 16
#  wfFileName = "P42574A_512waveforms_%drisetimeculled.npz" % numWaveforms
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    wfs = data['wfs']
    numWaveforms = wfs.size
  else:
    print "No saved waveforms available.  Loading from Data"
    exit(0)
  
  
  tempGuess = 79.204603
  gradGuess = 0.05
  pcRadGuess = 2.5
  pcLenGuess = 1.6


  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10)
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess, method="Full")
  
  b_over_a = 0.037429
  c = -0.810930
  d = 0.818203
  rc = 66.5
  det.SetTransferFunction(b_over_a, c, d, rc)
  
#  collection_rc = 4
#  det.collection_rc = collection_rc

  initializeDetector(det, )

  #ML as a start
  nll = lambda *args: -lnlike_waveform(*args)
  
  
  plt.ioff()
  
  for (wf_idx,wf) in enumerate(wfs):
    if wf_idx <= 3: continue

    if wf.energy < 1700: continue
    
    wf.WindowWaveformTimepoint(fallPercentage=.97, rmsMult=2)
    initializeWaveform(wf)
    dataLen = wf.wfLength
    t_data = np.arange(dataLen) * 10
    startGuess = [15., np.pi/8, 15., wf.wfMax, wf.t0Guess, 10, tempGuess,  b_over_a, c, d, rc ]


    plt.ion()

    fig1 = plt.figure(1)
    plt.clf()
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.set_xlabel("Digitizer Time [ns]")
    ax0.set_ylabel("Voltage [Arb.]")
    ax1.set_ylabel("Residual")

    ax0.plot(t_data, wf.windowedWf, color="r")

    init_wf = det.MakeSimWaveform(15, np.pi/8, 15, wf.wfMax, wf.t0Guess, fitSamples, h_smoothing=10)
    ax0.plot(t_data, init_wf[:dataLen])
    
    ax1.plot(t_data, wf.windowedWf - init_wf[:dataLen])
    
#    plt.ioff()
    value = raw_input('  --> Press q to quit, s to skip, any other key to continue\n')
    if value == 'q':
      exit(0)
    if value == 's': continue

#
#    result = op.minimize(nll, startGuess,   method="Powell")
#    #result = op.basinhopping(nll, startGuess,   T=1000,stepsize=15, minimizer_kwargs= {"method": "Nelder-Mead", "args":(wf)})
#    r, phi, z, scale, t0, smoove, temp, zero, pole_real, pole_imag  = result["x"]
#    zeros = [zero, -1., 1. ]
#    poles = [pole_1, pole_real+pole_imag*1j, pole_real-pole_imag*1j, ]
#    det.SetTransferFunction(zeros, poles)
#    det.SetTemperature(temp)
#    
#    print "Waveform number %d" % wf_idx
#    print "   " , r, phi, z, scale, t0, smoove, temp, zero,  pole_real, pole_imag
#    print "   Initial LN like is %f" %  (result['fun'])
#
#    ml_wf = np.copy(det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing=smoove ))
#    ax0.plot(t_data, ml_wf[:dataLen], color="b")
#    ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color="b")
#    
#    
#    r_new = np.amin( [z, np.floor(det.detector_radius)-2] )
#    z_new = np.amin( [r, np.floor(det.detector_length)-2] )
#
#    result2 = op.minimize(nll, [r_new, np.pi/4, z_new, scale, wf.t0Guess, 1,tempGuess, zero_1,  pole_real, pole_imag],  method="Powell")
#    r, phi, z, scale, t0, smoove,temp, zero,  pole_real, pole_imag  =  result2["x"]
#    zeros = [zero, -1., 1. ]
#    poles = [pole_1, pole_real+pole_imag*1j, pole_real-pole_imag*1j, ]
#    det.SetTransferFunction(zeros, poles)
#    det.SetTemperature(temp)
#    
#    inv_ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing=smoove)
#    ax0.plot(t_data,inv_ml_wf[:dataLen], color="g")
#    print "   ", r, phi, z, scale, t0, smoove, temp, zero,  pole_real, pole_imag
#    ax1.plot(t_data,inv_ml_wf[:dataLen] -  wf.windowedWf, color="g")
#    
#
#    ax1.set_ylim(-20,20)
#
#
##    if result2['fun'] < result['fun']:
##      mcmc_startguess = result2["x"]
##    else:
##      mcmc_startguess = result["x"]
#
#
#    print "   Inverted LN like is %f" %  (result2['fun'])
#
#    value = raw_input('  --> Press q to quit, s to skip, any other key to continue\n')
#    if value == 'q':
#      exit(0)
#    if value == 's':
#      continue
#    else:
#      continue

    #Do the MCMC
    ndim, nwalkers = 11, 100
    mcmc_startguess = startGuess#np.array([r, phi, z, scale, t0, smooth])
    
    pos0 = [mcmc_startguess + 1e-2*np.random.randn(ndim)*mcmc_startguess for i in range(nwalkers)]
    p = Pool(numThreads, initializer=initializeWaveform, initargs=[ wf])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_waveform, )


    iter, burnIn = 1000, 900
    
    bar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=iter).start()
    for (i,result) in enumerate(sampler.sample(pos0, iterations=iter, storechain=True)):
      bar.update(i+1)
    bar.finish()
    p.close()
        
    #samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

    #########  Plots for MC Steps
    stepsFig = plt.figure(2)
    plt.clf()
    plotnum = 600
    ax0 = stepsFig.add_subplot(plotnum+11)
    ax1 = stepsFig.add_subplot(plotnum+12, sharex=ax0)
    ax2 = stepsFig.add_subplot(plotnum+13, sharex=ax0)
    ax3 = stepsFig.add_subplot(plotnum+14, sharex=ax0)
    ax4 = stepsFig.add_subplot(plotnum+15, sharex=ax0)
    ax5 = stepsFig.add_subplot(plotnum+16, sharex=ax0)


    ax0.set_ylabel('r')
    ax1.set_ylabel('phi')
    ax2.set_ylabel('z')
    ax3.set_ylabel('scale')
    ax4.set_ylabel('t0')
    ax5.set_ylabel('smooth')

#    ax6.set_ylabel('temp')

    for i in range(nwalkers):
      ax0.plot(sampler.chain[i,:,0], "b", alpha=0.3)
      ax1.plot(sampler.chain[i,:,1], "b", alpha=0.3)
      ax2.plot(sampler.chain[i,:,2], "b", alpha=0.3)
      ax3.plot(sampler.chain[i,:,3], "b", alpha=0.3)
      ax4.plot(sampler.chain[i,:,4], "b", alpha=0.3)
      ax5.plot(sampler.chain[i,:,5], "b", alpha=0.3)

    plt.savefig("hdxwf%d_wf_params.png"%wf_idx)

    stepsFig2 = plt.figure(11)
    plotnum = 500
    ax6 = stepsFig2.add_subplot(plotnum+11)
    ax7 = stepsFig2.add_subplot(plotnum+12, sharex=ax6)
    ax8 = stepsFig2.add_subplot(plotnum+13, sharex=ax6)
    ax9 = stepsFig2.add_subplot(plotnum+14, sharex=ax6)
    ax10 = stepsFig2.add_subplot(plotnum+15, sharex=ax6)

    ax6.set_ylabel('temp')
    ax7.set_ylabel('b_over_a')
    ax8.set_ylabel('c')
    ax9.set_ylabel('d')
    ax10.set_ylabel('rc')
    for i in range(nwalkers):
      ax6.plot(sampler.chain[i,:,6], "b", alpha=0.3)
      ax7.plot(sampler.chain[i,:,7], "b", alpha=0.3)
      ax8.plot(sampler.chain[i,:,8], "b", alpha=0.3)
      ax9.plot(sampler.chain[i,:,9], "b", alpha=0.3)
      ax10.plot(sampler.chain[i,:,10], "b", alpha=0.3)

    plt.savefig("hdxwf%d_det_params.png"%wf_idx)

    #pull the samples after burn-in
    samples = sampler.chain[:, burnIn:, :].reshape((-1, ndim))
    wfPlotNumber = 10
    simWfs = np.empty((wfPlotNumber, fitSamples) )

    for idx, (r, phi, z, scale, t0, smooth, temp,  b_over_a, c, d, rc) in enumerate(samples[np.random.randint(len(samples), size=wfPlotNumber)]):
      det.SetTransferFunction(b_over_a, c, d, rc)
      det.SetTemperature(temp)
#      det.collection_rc = collection_rc
      simWfs[idx,:] = det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing=smooth)

    residFig = plt.figure(3)
    helpers.plotResidual(simWfs, wf.windowedWf, figure=residFig)

    print "Median values wf %d: " % wf_idx
    for idx in range(ndim):
      print "  param %d: %f" % (idx, np.median(samples[:,idx]))

    plt.savefig("hdxwf%d_resid.png"%wf_idx)

#    plt.show()
#    value = raw_input('  --> Press q to quit, any other key to continue\n')
#    if value == 'q':
#      exit(0)


if __name__=="__main__":
    main(sys.argv[1:])


