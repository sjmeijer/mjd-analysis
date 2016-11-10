#!/usr/local/bin/python
#matplotlib.use('CocoaAgg')
import sys, os
import scipy.optimize as op
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import helpers
from detector_model import *
from probability_model_waveform import *

import pymc
import signal_model_hierarchical_pymc2 as sm

def main(argv):

  plt.ion()
  
  numSamples = 4000000
  burnin = 0.9*numSamples
  
  doInitPlot = False

  
  fitSamples = 175
  timeStepSize = 1. #ns
  
  #Prepare detector
  tempGuess = 79.071172
  gradGuess = 0.05
  pcRadGuess = 2.5
  pcLenGuess = 1.6

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10 )
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  
  b_over_a = 3
  c = -0.809211
  d = 0.816583
  rc1 = 74.4
  rc2 = 1.79
  rcfrac = 0.992
  det.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)
  
  initializeDetector(det, )
  
#  wfFileName = "P42574A_512waveforms_30risetimeculled.npz"
  wfFileName = "P42574A_12_fastandslow_oldwfs.npz"
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    results = data['results']
    wfs = data['wfs']
    numWaveforms = wfs.size
  else:
    print "No saved waveforms available.  Go hard or go home."
    exit(0)

  #prep holders for each wf-specific param
  r_arr = np.empty(numWaveforms)
  phi_arr = np.empty(numWaveforms)
  z_arr = np.empty(numWaveforms)
  scale_arr = np.empty(numWaveforms)
  t0_arr = np.empty(numWaveforms)
  smooth_arr = np.ones(numWaveforms)
  simWfArr = np.empty((1,numWaveforms, fitSamples))

  nll = lambda *args: -lnlike_waveform(*args)

#  for (idx, wf) in enumerate(wfs):
#    wf.WindowWaveformTimepoint(fallPercentage=0.999, rmsMult=2)
#    
#    initializeWaveform(wf)
#    
#    r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx],  smooth_arr[idx]= results[idx]['x']
#    startGuess = [r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], 10]
#    result = op.minimize(nll, startGuess,   method="Powell")
#    r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx],  smooth_arr[idx]= result['x']


#    t0_arr[idx] += 10 #because i had a different windowing offset back in the day
    #smooth_arr[idx] /= 10.
#    scale_arr[idx]*=100


  #Plot the waveforms to take a look at the initial guesses
#  if doInitPlot:
#    plt.ion()
#    fig = plt.figure(11)
#    for (idx,wf) in enumerate(wfs):
#      
#      print "WF number %d:" % idx
#      print "  >>r: %f\n  >>phi %f\n  >>z %f\n  >>e %f\n  >>t0 %f, >>smooth:%f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx])
#      ml_wf = det.MakeSimWaveform(r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], fitSamples, h_smoothing=smooth_arr[idx])
#      plt.plot(ml_wf, color="b")
#      plt.plot(wf.windowedWf, color="r")
#    value = raw_input('  --> Press q to quit, any other key to continue\n')
##    plt.ioff()
#    if value == 'q': exit(0)


  for (wf_idx, wf) in enumerate(wfs):
    if wf_idx <= 0: continue
    
    wf.WindowWaveformTimepoint(fallPercentage=0.98, rmsMult=2)
    initializeWaveform(wf)
    
    minresult = None
    minlike = np.inf
  
    for r in np.linspace(4, np.floor(det.detector_radius)-3, 6):
      for z in np.linspace(4, np.floor(det.detector_length)-3, 6):
        if not det.IsInDetector(r,0,z): continue
        startGuess = [r, np.pi/8, z, wf.wfMax, wf.t0Guess-5, 10]
        result = op.minimize(nll, startGuess,   method="Nelder-Mead")
        r, phi, z, scale, t0, smooth, = result["x"]
        ml_wf = np.copy(det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing=smooth, ))
        if ml_wf is None:
          print r, z
          continue
        if result['fun'] < minlike:
          minlike = result['fun']
          minresult = result
    
    r_arr[wf_idx], phi_arr[wf_idx], z_arr[wf_idx], scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx], = minresult["x"]
    
    
    fig = plt.figure(11)
    plt.clf()
    ml_wf = det.MakeSimWaveform(r_arr[wf_idx], phi_arr[wf_idx], z_arr[wf_idx], scale_arr[wf_idx], t0_arr[wf_idx], fitSamples, h_smoothing=smooth_arr[wf_idx])
    plt.plot(ml_wf, color="b")
    plt.plot(wf.windowedWf, color="r")
    value = raw_input('  --> Press q to quit, any other key to continue\n')
    if value == 'q': exit(0)
    
    
    new_r = np.sqrt(r_arr[wf_idx]**2 + z_arr[wf_idx]**2)
    new_theta = np.arctan(z_arr[wf_idx] / r_arr[wf_idx])
    
    print new_theta

    startGuess = {'radEst': new_r,
                  'thetaEst': new_theta,
                  'phiEst': phi_arr[wf_idx],
                  'wfScale': scale_arr[wf_idx],
                  'switchpoint': t0_arr[wf_idx],
                  'smooth': smooth_arr[wf_idx],
                  'b_over_a': b_over_a,
                  'c': c,
                  'd': d,
                  'rc1': rc1,
                  'rc2': rc2,
                  'rcfrac': rcfrac,
                  'temp': tempGuess
                  }
#    for k, v in startGuess.iteritems():
#      print k, v

    model_locals = sm.CreateTFModel(det, wf,  startGuess)

    M = pymc.MCMC(model_locals, )


    M.use_step_method(pymc.AdaptiveMetropolis, [M.radEst,  M.phiEst, M.t0Est, M.thetaEst, M.b_over_a, M.tempEst],
                       scales={M.radEst:  0.1,
                               M.t0Est: 0.1,
                               M.phiEst: np.pi/10,
                               M.thetaEst: 0.01,
                               M.b_over_a: 0.5,
                               M.tempEst: 0.5
                               }, delay=10000, interval=10000,shrink_if_necessary=True, )

    M.use_step_method(pymc.AdaptiveMetropolis, [M.c,  M.d, ],
                       scales={M.c:  0.1,
                               M.d: 0.1,
                               }, delay=10000, interval=10000,shrink_if_necessary=True, )

#    M.use_step_method(pymc.Metropolis, M.radEst, proposal_sd=0.5, proposal_distribution='Normal')
#    M.use_step_method(pymc.Metropolis, M.thetaEst, proposal_sd=np.pi/5, proposal_distribution='Normal', )
#    M.use_step_method(pymc.Metropolis, M.phiEst, proposal_sd=np.pi/10, proposal_distribution='Normal')
    M.use_step_method(pymc.Metropolis, M.scaleEst, proposal_sd=0.01*startGuess['wfScale'], proposal_distribution='Normal')
#    M.use_step_method(pymc.Metropolis, M.t0Est, proposal_sd=0.05, proposal_distribution='Normal')
    M.use_step_method(pymc.Metropolis, M.sigEst, proposal_sd=01, proposal_distribution='Normal')
    
#    M.use_step_method(pymc.Metropolis, M.tempEst,   proposal_sd=1, proposal_distribution='Normal')
#    M.use_step_method(pymc.Metropolis, M.b_over_a,  proposal_sd=0.5, proposal_distribution='Normal')
#    M.use_step_method(pymc.Metropolis, M.c,         proposal_sd=0.1, proposal_distribution='Normal')
#    M.use_step_method(pymc.Metropolis, M.d,         proposal_sd=0.1, proposal_distribution='Normal')

    M.use_step_method(pymc.Metropolis, M.rc1,       proposal_sd=0.5, proposal_distribution='Normal')
    M.use_step_method(pymc.Metropolis, M.rc2,       proposal_sd=0.1, proposal_distribution='Normal')
    M.use_step_method(pymc.Metropolis, M.rcfrac,    proposal_sd=0.01, proposal_distribution='Normal')
#
#    M.use_step_method(pymc.AdaptiveMetropolis, [M.c,  M.d,],
#                       scales={M.c:  0.1,
#                               M.d: 0.1,
#                               }, delay=1000, interval=1000,shrink_if_necessary=True, )
#
#    M.use_step_method(pymc.AdaptiveMetropolis, [M.radEst,  M.t0Est,],
#                       scales={M.radEst:  0.1,
#                               M.t0Est: 0.1,
#                               }, delay=1000, interval=1000,shrink_if_necessary=True, )



#    M.use_step_method(pymc.Slicer, M.radEst)
#    M.use_step_method(pymc.Slicer, M.thetaEst,  )
#    M.use_step_method(pymc.Slicer, M.phiEst, )
#    M.use_step_method(pymc.Slicer, M.scaleEst, )
#    M.use_step_method(pymc.Slicer, M.t0Est,)# proposal_sd=0.05, proposal_distribution='Normal')
#    M.use_step_method(pymc.Slicer, M.sigEst,)# proposal_sd=01, proposal_distribution='Normal')
#    
#    M.use_step_method(pymc.Slicer, M.tempEst, )#  proposal_sd=1, proposal_distribution='Normal')
#    M.use_step_method(pymc.Slicer, M.b_over_a,)#  proposal_sd=0.5, proposal_distribution='Normal')
#    M.use_step_method(pymc.Slicer, M.c,        )# proposal_sd=0.1, proposal_distribution='Normal')
#    M.use_step_method(pymc.Slicer, M.d,        )# proposal_sd=0.1, proposal_distribution='Normal')
#
#    M.use_step_method(pymc.Slicer, M.rc1,     )#  proposal_sd=0.5, proposal_distribution='Normal')
#    M.use_step_method(pymc.Slicer, M.rc2,     )# proposal_sd=0.1, proposal_distribution='Normal')
#    M.use_step_method(pymc.Slicer, M.rcfrac,   )# proposal_sd=0.01, proposal_distribution='Normal')

    print M.step_method_dict[M.radEst][0].C
    print M.step_method_dict[M.c][0].C

    print M.step_method_dict[M.radEst][0].stochastics
    print M.step_method_dict[M.c][0].stochastics

    M.sample(iter=numSamples, burn=0, tune_interval=1000)
    M.db.close()
    
    print M.step_method_dict[M.radEst][0].C
#    print M.step_method_dict[M.c][0].C

  #  totalIter = 0
  #  while totalIter < this_sample:
  #    M.sample(iter=10, verbose=0)
  #    totalIter += 10

    
  #  #pymc.Matplot.plot(M)
  #  
  #  #########  Plots for MC Steps
    stepsFig = plt.figure(1,figsize=(20,10))
    plt.clf()
    ax0 = stepsFig.add_subplot(611)
    ax1 = stepsFig.add_subplot(612, sharex=ax0)
    ax2 = stepsFig.add_subplot(613, sharex=ax0)
    ax3 = stepsFig.add_subplot(614, sharex=ax0)
    ax4 = stepsFig.add_subplot(615, sharex=ax0)
    ax5 = stepsFig.add_subplot(616, sharex=ax0)

    ax0.set_ylabel('r')
    ax1.set_ylabel('theta')
    ax2.set_ylabel('phi')
    ax3.set_ylabel('e')
    ax4.set_ylabel('t0')
    ax5.set_ylabel('sig')

    ax0.plot(M.trace('radEst')[:])
    ax1.plot(M.trace('thetaEst')[:])
    ax2.plot(M.trace('phiEst')[:])
    ax3.plot(M.trace('wfScale')[:])
    ax4.plot(M.trace('switchpoint')[:])
    ax5.plot(M.trace('sigma')[:])

    plt.savefig("pymc_wf_params.png")
    
    stepsFig = plt.figure(2,figsize=(20,10))
    plt.clf()
    ax0 = stepsFig.add_subplot(711)
    ax1 = stepsFig.add_subplot(712, sharex=ax0)
    ax2 = stepsFig.add_subplot(713, sharex=ax0)
    ax3 = stepsFig.add_subplot(714, sharex=ax0)
    ax4 = stepsFig.add_subplot(715, sharex=ax0)
    ax5 = stepsFig.add_subplot(716, sharex=ax0)
    ax6 = stepsFig.add_subplot(717, sharex=ax0)


    ax0.set_ylabel('temp')
    ax1.set_ylabel('b_over_a')
    ax2.set_ylabel('c')
    ax3.set_ylabel('d')
    ax4.set_ylabel('rc1')
    ax5.set_ylabel('rc2')
    ax5.set_ylabel('rcfrac')


    ax0.plot(M.trace('temp')[:])
    ax1.plot(M.trace('b_over_a')[:])
    ax2.plot(M.trace('c')[:])
    ax3.plot(M.trace('d')[:])
    ax4.plot(M.trace('rc1')[:])
    ax5.plot(M.trace('rc2')[:])
    ax6.plot(M.trace('rcfrac')[:])

    plt.savefig("pymc_tf_params.png")
    
    
    stepsFig = plt.figure(3,figsize=(20,10))
    plt.clf()
    ax0 = stepsFig.add_subplot(321)
    ax1 = stepsFig.add_subplot(322, )
    ax2 = stepsFig.add_subplot(323, )
    ax3 = stepsFig.add_subplot(324, )
    ax4 = stepsFig.add_subplot(325, )
    ax5 = stepsFig.add_subplot(326, )

    ax0.set_ylabel('r')
    ax1.set_ylabel('theta')
    ax2.set_ylabel('phi')
    ax3.set_ylabel('e')
    ax4.set_ylabel('t0')
    ax5.set_ylabel('sig')


    hist, bins = np.histogram(M.trace('radEst')[burnin:])
    ax0.plot(bins[:-1], hist, ls="steps-post")
    hist, bins = np.histogram(M.trace('thetaEst')[burnin:])
    ax1.plot(bins[:-1], hist, ls="steps-post")
    hist, bins = np.histogram(M.trace('phiEst')[burnin:])
    ax2.plot(bins[:-1], hist, ls="steps-post")
    hist, bins = np.histogram(M.trace('wfScale')[burnin:])
    ax3.plot(bins[:-1], hist, ls="steps-post")
    hist, bins = np.histogram(M.trace('switchpoint')[burnin:])
    ax4.plot(bins[:-1], hist, ls="steps-post")
    hist, bins = np.histogram(M.trace('sigma')[burnin:])
    ax5.plot(bins[:-1], hist, ls="steps-post")
    
    plt.savefig("pymc_wf_params_hist.png")


    wfPlotNumber = 5000
    simWfs = np.empty((wfPlotNumber, fitSamples) )

    if burnin > len(M.trace('radEst')[:]):
      burnin = len(M.trace('radEst')[:]) - wfPlotNumber
      numSamples = len(M.trace('radEst')[:])

    for (sim_idx, chain_idx) in enumerate(np.random.randint(low=burnin, high=numSamples, size=wfPlotNumber)):
      t0 =    M.trace('switchpoint')[chain_idx]
      rad =     M.trace('radEst')[chain_idx]
      theta =     M.trace('thetaEst')[chain_idx]
      phi =   M.trace('phiEst')[chain_idx]
      scale = M.trace('wfScale')[chain_idx]
      sigma = M.trace('sigma')[chain_idx]
      
      r = np.cos(theta)*rad
      z = np.sin(theta)*rad

      simWfs[sim_idx,:] = det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing = sigma,)

    residFig = plt.figure(4,figsize=(20,10))
    helpers.plotResidual(simWfs, wf.windowedWf, figure=residFig)

    plt.savefig("pymc_waveforms.png")

    value = raw_input('  --> Press q to quit, any other key to continue\n')
    if value == 'q': exit(0)


if __name__=="__main__":
    main(sys.argv[1:])


