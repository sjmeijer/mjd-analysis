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

import probability_model_waveform as pmw

from progressbar import ProgressBar, Percentage, Bar, ETA
from matplotlib import gridspec
from multiprocessing import Pool

def main(argv):

  plt.ion()

  numThreads = 8
  
  fitSamples = 130
  timeStepSize = 1
  
  wfFileName = "P42574A_12_fastandslow_oldwfs.npz"
#  wfFileName = "P42574A_pc_highe.npz"
  #wfFileName = "fep_event_set_runs11531-11539.npz"
#  numWaveforms = 16
#  wfFileName = "P42574A_512waveforms_%drisetimeculled.npz" % numWaveforms
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    wfs = data['wfs']
    results = data['results']
    numWaveforms = wfs.size
  
  else:
    print "No saved waveforms available.  Loading from Data"
    exit(0)
  
  
  tempGuess = 79.071172
  gradGuess = 0.04
  pcRadGuess = 2.5
  pcLenGuess = 1.6


  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10)
  det.LoadFieldsGrad("fields_impgrad.npz", pcLen=pcLenGuess, pcRad=pcRadGuess)
  det.SetFieldsGradInterp(gradGuess)
  
  b_over_a = 0.107213
  c = -0.815152
  d = 0.822696
  rc1 = 74.4
  rc2 = 1.79
  rcfrac = 0.992
  trapping_rc = 120#us
  det.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)
#  det.trapping_rc = trapping_rc #us
  det.trapping_rc = trapping_rc
  
  #do my own hole mobility model based on bruy
  det.siggenInst.set_velocity_type(1)
  h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0 = 66333., 0.744, 181., 107270., 0.580, 100.
  

#  trapping_rc = 4
#  det.trapping_rc = trapping_rc

  initializeDetector(det, )
  pmw.initializeDetector(det, )

  #ML as a start
  nll = lambda *args: -lnlike_waveform(*args)
  
  nll_wf = lambda *args: -pmw.lnlike_waveform(*args)
  
  
#  plt.ioff()

  for (wf_idx,wf) in enumerate(wfs):
    if wf_idx <= 4: continue

    if wf.energy < 1700: continue
    
    wf.WindowWaveformTimepoint(fallPercentage=.995, rmsMult=2)
    initializeWaveform(wf)
    pmw.initializeWaveform(wf)
    
    dataLen = wf.wfLength
    t_data = np.arange(dataLen) * 10

    fig1 = plt.figure(1)
    plt.clf()
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.set_xlabel("Digitizer Time [ns]")
    ax0.set_ylabel("Voltage [Arb.]")
    ax1.set_ylabel("Residual")
    
    ax0.plot(t_data, wf.windowedWf, color="r")
#    minresult = None
#    minlike = np.inf
##
#    for r in np.linspace(4, np.floor(det.detector_radius)-3, 3):
#      for z in np.linspace(4, np.floor(det.detector_length)-3, 3):
##        for t0_guess in np.linspace(wf.t0Guess-10, wf.t0Guess+5, 3):
#          if not det.IsInDetector(r,0,z): continue
#          startGuess = [r, np.pi/8, z, wf.wfMax, wf.t0Guess-5, 10]
#          result = op.minimize(nll_wf, startGuess,   method="Nelder-Mead")
#          r, phi, z, scale, t0, smooth, = result["x"]
#          ml_wf = np.copy(det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing=smooth, ))
#          if ml_wf is None:
#            print r, z
#            continue
#          if result['fun'] < minlike:
#            minlike = result['fun']
#            minresult = result
#
#    ax1.set_ylim(-20,20)
#    r, phi, z, scale, t0, smooth, = minresult["x"]
#    print r, phi, z, scale, t0, smooth

    r, phi, z, scale, t0, smooth  = results[wf_idx]['x']
    startGuess = [r, phi, z, scale, t0, smooth]
    result = op.minimize(nll_wf, startGuess,   method="Powell")
    r, phi, z, scale, t0, smooth = result['x']

    ml_wf = np.copy(det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing=smooth, ))
    ax0.plot(t_data, ml_wf[:dataLen], color="g")
    ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color="g",)
#
    value = raw_input('  --> Press q to quit, s to skip, any other key to continue\n')
    if value == 'q':
      exit(0)
    if value == 's': continue

    mcmc_startguess = [r, phi, z, scale, t0, smooth,  b_over_a, c, d, h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0,]# trapping_rc, ]#gradGuess, pcRadGuess, pcLenGuess   ]
#    mcmc_startguess = [r, phi, z, scale, t0, smooth, tempGuess, omega, decay, rc1, rc2, rcfrac, ]
    #Do the MCMC
    ndim = 15
    nwalkers = ndim*10
#    mcmc_startguess = startGuess#np.array([r, phi, z, scale, t0, smooth])

    pos0 = [mcmc_startguess + 1e-2*np.random.randn(ndim)*mcmc_startguess for i in range(nwalkers)]
    gradIdx = -1
    mobIdx = 9
    
#    pcRadIdx = -2
#    pcLenIdx = -1

#    rc1idx = -4
#    rc2idx = -3
#    rcfracidx = -2

#    for pos in pos0:
##      pos[rcfracidx] = np.clip(pos[rcfracidx], 0, 1)
##      pos[rc2idx] = np.clip(pos[rc2idx], 0, np.inf)
##      pos[rc1idx] = np.clip(pos[rc1idx], 0, np.inf)
#
#      pos[gradIdx] = np.clip(pos[gradIdx], det.gradList[0], det.gradList[-1])
##      pos[pcRadIdx] = np.clip(pos[pcRadIdx], det.pcRadList[0], det.pcRadList[-1])
##      pos[pcLenIdx] = np.clip(pos[pcLenIdx], det.pcLenList[0], det.pcLenList[-1])
#
#      prior = lnprob_waveform(pos,)
#      if not np.isfinite(prior) :
#        print "BAD PRIOR WITH START GUESS YOURE KILLING ME SMALLS"
#        print pos
#        exit(0)

    p = Pool(numThreads, initializer=initializeWaveform, initargs=[ wf])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_waveform, )


    iter, burnIn = 2000, 1800
    wfPlotNumber = 100
    
    bar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=iter).start()
    for (i,result) in enumerate(sampler.sample(pos0, iterations=iter, storechain=True)):
      bar.update(i+1)
    bar.finish()
    p.close()
        
    #samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

    #########  Plots for MC Steps
    stepsFig = plt.figure(2, figsize= (20,10))
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

    stepsFig = plt.figure(3, figsize= (20,10))
    plt.clf()
    plotnum = 600
    ax0 = stepsFig.add_subplot(plotnum+11)
    ax1 = stepsFig.add_subplot(plotnum+12, sharex=ax0)
    ax2 = stepsFig.add_subplot(plotnum+13, sharex=ax0)
    ax3 = stepsFig.add_subplot(plotnum+14, sharex=ax0)
    ax4 = stepsFig.add_subplot(plotnum+15, sharex=ax0)
    ax5 = stepsFig.add_subplot(plotnum+16, sharex=ax0)

    ax0.set_ylabel('h_100_mu0')
    ax1.set_ylabel('h_100_beta')
    ax2.set_ylabel('h_100_e0')
    ax3.set_ylabel('h_111_mu0')
    ax4.set_ylabel('h_111_beta')
    ax5.set_ylabel('h_111_e0')

#    ax6.set_ylabel('temp')

    for i in range(nwalkers):
      ax0.plot(sampler.chain[i,:,mobIdx+0], "b", alpha=0.3)
      ax1.plot(sampler.chain[i,:,mobIdx+1], "b", alpha=0.3)
      ax2.plot(sampler.chain[i,:,mobIdx+2], "b", alpha=0.3)
      ax3.plot(sampler.chain[i,:,mobIdx+3], "b", alpha=0.3)
      ax4.plot(sampler.chain[i,:,mobIdx+4], "b", alpha=0.3)
      ax5.plot(sampler.chain[i,:,mobIdx+5], "b", alpha=0.3)

    plt.savefig("hdxwf%d_mobility_params.png"%wf_idx)

    stepsFig2 = plt.figure(4, figsize = (20,10))
    plotnum = 400
    ax6 = stepsFig2.add_subplot(plotnum+11)
    ax7 = stepsFig2.add_subplot(plotnum+12, sharex=ax6)
    ax8 = stepsFig2.add_subplot(plotnum+13, sharex=ax6)
    ax9 = stepsFig2.add_subplot(plotnum+14, sharex=ax6)
#    ax10 = stepsFig2.add_subplot(plotnum+15, sharex=ax6)
#    ax11 = stepsFig2.add_subplot(plotnum+16, sharex=ax6)
#    ax12 = stepsFig2.add_subplot(plotnum+17, sharex=ax6)
#    ax13 = stepsFig2.add_subplot(plotnum+18, sharex=ax6)

    ax6.set_ylabel('b_ov_a')
    ax7.set_ylabel('c')
    ax8.set_ylabel('d')
    ax9.set_ylabel('trapping')
#    ax10.set_ylabel('rc1')
#    ax11.set_ylabel('rc2')
#    ax12.set_ylabel('rcfrac')
#    ax8.set_ylabel('trapping')
#    ax9.set_ylabel('grad')
    for i in range(nwalkers):
      ax6.plot(sampler.chain[i,:,6], "b", alpha=0.3)
      ax7.plot(sampler.chain[i,:,7], "b", alpha=0.3)
      ax8.plot(sampler.chain[i,:,8], "b", alpha=0.3)
#      ax9.plot(sampler.chain[i,:,9], "b", alpha=0.3)
#      ax10.plot(sampler.chain[i,:,10], "b", alpha=0.3)
#      ax11.plot(sampler.chain[i,:,11], "b", alpha=0.3)
#      ax12.plot(sampler.chain[i,:,12], "b", alpha=0.3)
#      ax13.plot(sampler.chain[i,:,-1], "b", alpha=0.3)

    plt.savefig("hdxwf%d_tf_params.png"%wf_idx)
    
    
    samples = sampler.chain[:, burnIn:, :].reshape((-1, ndim))
    
#    stepsFigTF = plt.figure(5, figsize = (20,10))
#
#    tf0 = stepsFigTF.add_subplot(plotnum+11)
#    tf1 = stepsFigTF.add_subplot(plotnum+12, )
#    tf2 = stepsFigTF.add_subplot(plotnum+13, )
##    tf3 = stepsFigTF.add_subplot(plotnum+14, )
##    tf4 = stepsFigTF.add_subplot(815, )
##    tf5 = stepsFigTF.add_subplot(816, )
##    tf6 = stepsFigTF.add_subplot(817,)
##    tf7 = stepsFigTF.add_subplot(818,)
#
#    tf0.set_ylabel('temp')
##    tf1.set_ylabel('c')
##    tf2.set_ylabel('d')
##    tf3.set_ylabel('rc1')
##    tf4.set_ylabel('rc2')
##    tf5.set_ylabel('rcfrac')
#    tf1.set_ylabel('b_ov_a')
#    tf2.set_ylabel('trapping')
##    tf3.set_ylabel('grad')
#
#    num_bins = 300
#    [n, b, p] = tf0.hist(samples[:,6], bins=num_bins)
#    print "b_over_a mode is %f" % b[np.argmax(n)]
#
#    [n, b, p] = tf1.hist(samples[:,7],bins=num_bins)
#    print "c mode is %f" % b[np.argmax(n)]
#
#    [n, b, p] = tf2.hist(samples[:,8],bins=num_bins)
#    print "d mode is %f" % b[np.argmax(n)]

#    [n, b, p] = tf3.hist(samples[:,-1],bins=num_bins)
#    print "rc_decay1 mode is %f" % b[np.argmax(n)]
#
#    [n, b, p] = tf4.hist(samples[:,-3],bins=num_bins)
#    print "rc_decay2 mode is %f" % b[np.argmax(n)]
#
#    [n, b, p] = tf5.hist(samples[:,-2],bins=num_bins)
#    print "rc_frac mode is %f" % b[np.argmax(n)]
#
#    [n, b, p] = tf6.hist(samples[:,-8],bins=num_bins)
#    print "temp mode is %f" % b[np.argmax(n)]
#    
#    [n, b, p] = tf6.hist(samples[:,-1],bins=num_bins)
#    print "grad mode is %f" % b[np.argmax(n)]

    plt.savefig("hdxwf%d_tf_hist.png"%wf_idx)
    
    
#    stepsFig3 = plt.figure(13)
#    plotnum = 300
#    ax13 = stepsFig3.add_subplot(plotnum+11)
#    ax14 = stepsFig3.add_subplot(plotnum+12, sharex=ax13)
#    ax15 = stepsFig3.add_subplot(plotnum+13, sharex=ax13)
#    ax13.set_ylabel('grad')
#    ax14.set_ylabel('pcrad')
#    ax15.set_ylabel('pclem')
#    for i in range(nwalkers):
#      ax13.plot(sampler.chain[i,:,13], "b", alpha=0.3)
#      ax14.plot(sampler.chain[i,:,14], "b", alpha=0.3)
#      ax15.plot(sampler.chain[i,:,15], "b", alpha=0.3)
#
#    plt.savefig("hdxwf%d_det_params.png"%wf_idx)

    #pull the samples after burn-in
    
    simWfs = np.empty((wfPlotNumber, fitSamples) )

    for idx, (r, phi, z, scale, t0, smooth,  b_over_a, c, d, h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0, ) in enumerate(samples[np.random.randint(len(samples), size=wfPlotNumber)]):
#    for idx, (r, phi, z, scale, t0, smooth, temp,  omega,decay, rc1, rc2, rcfrac, ) in enumerate(samples[np.random.randint(len(samples), size=wfPlotNumber)]):

      det.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)
      det.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
#      det.SetTemperature(temp)
#      det.trapping_rc = trapping_rc
#      det.SetFieldsGradInterp(grad)
#      det.SetFields(pcRad, pcLen, impGrad)
#      det.trapping_rc = trapping_rc
      simWfs[idx,:] = det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing=smooth)

    residFig = plt.figure(6)
    helpers.plotResidual(simWfs, wf.windowedWf, figure=residFig)

    print "Median values wf %d: " % wf_idx
    for idx in range(ndim):
      print "  param %d: %f" % (idx, np.median(samples[:,idx]))

    plt.savefig("hdxwf%d_resid.png"%wf_idx)

#    plt.show()
    value = raw_input('  --> Press q to quit, any other key to continue\n')
    if value == 'q':
      exit(0)


if __name__=="__main__":
    main(sys.argv[1:])


