#!/usr/local/bin/python
#matplotlib.use('CocoaAgg')
import sys, os
import scipy.optimize as op
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import helpers
from detector_model import *

import pymc
import signal_model_hierarchical_pymc2 as sm

def main(argv):

  plt.ion()

  fitSamples = 210
  timeStepSize = 10. #ns
  
  numSamples = 20000
  burnin = 0.9*numSamples
  
  doInitPlot = False
  
  #Prepare detector
  zero_1 = 0.470677
  pole_1 = 0.999857
  pole_real = 0.807248
  pole_imag = 0.085347

  zeros = [zero_1, -1., 1. ]
  poles = [pole_1, pole_real+pole_imag*1j, pole_real-pole_imag*1j, ]
  
  tempGuess = 78.474793
  gradGuess = 0.045049
  pcRadGuess = 2.574859
  pcLenGuess = 1.524812

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10./timeStepSize, poles=poles, zeros=zeros)
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  
  wfFileName = "P42574A_512waveforms_30risetimeculled.npz"
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
  smooth_arr = np.empty(numWaveforms)
  simWfArr = np.empty((1,numWaveforms, fitSamples))

  for (idx, wf) in enumerate(wfs):
    wf.WindowWaveform(200)
    r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx]  = results[idx]['x']
    t0_arr[idx] += 10 #because i had a different windowing offset back in the day
    smooth_arr[idx] /= 10.
    scale_arr[idx]*=100


  #Plot the waveforms to take a look at the initial guesses
  if doInitPlot:
    plt.ion()
    fig = plt.figure()
    for (idx,wf) in enumerate(wfs):
      
      print "WF number %d:" % idx
      print "  >>r: %f\n  >>phi %f\n  >>z %f\n  >>e %f\n  >>t0 %f\n >>smooth %f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx])
      ml_wf = det.GetSimWaveform(r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], fitSamples, smoothing = smooth_arr[idx])
      plt.plot(ml_wf, color="b")
      plt.plot(wf.windowedWf, color="r")
    value = raw_input('  --> Press q to quit, any other key to continue\n')
    plt.ioff()
    if value == 'q': exit(0)

  startGuess = {'radEst': r_arr,
                'zEst': z_arr,
                'phiEst': phi_arr,
                'wfScale': scale_arr,
                'switchpoint': t0_arr,
                'smooth': smooth_arr,
                'temp': tempGuess,
                'grad': gradGuess,
                'pcRad': pcRadGuess,
                'pcLen':pcLenGuess}

  model_locals = sm.CreateFullDetectorModel(det, wfs,  startGuess, zero_1, pole_1, pole_real, pole_imag)

  M = pymc.MCMC(model_locals, db='pickle', dbname='Detector.pickle')
#  M.sample(iter=100, burn=90)

  #12:30 pm 9/24

  M.use_step_method(pymc.Slicer, M.grad, w = 0.03)
  M.use_step_method(pymc.Slicer, M.pcRad, w = 0.2)
  M.use_step_method(pymc.Slicer, M.pcLen, w=0.2)

  M.use_step_method(pymc.Slicer, M.tempEst,  w=8)
  M.use_step_method(pymc.Slicer, M.zero_1, w=0.2)
  M.use_step_method(pymc.Slicer, M.pole_1, w=0.1)
  M.use_step_method(pymc.Slicer, M.pole_real, w=0.1)
  M.use_step_method(pymc.Slicer, M.pole_imag, w=0.01)

#  M.use_step_method(pymc.Metropolis, M.grad, proposal_sd=0.01, proposal_distribution='Normal')
#  M.use_step_method(pymc.Metropolis, M.pcRad, proposal_sd=0.05, proposal_distribution='Normal')
#  M.use_step_method(pymc.Metropolis, M.pcLen, proposal_sd=0.05, proposal_distribution='Normal')
#
#  M.use_step_method(pymc.Metropolis, M.tempEst,  proposal_sd=3., proposal_distribution='Normal')
#  M.use_step_method(pymc.Metropolis, M.zero_1, proposal_sd=0.5, proposal_distribution='Normal')
#  M.use_step_method(pymc.Metropolis, M.pole_1, proposal_sd=0.1, proposal_distribution='Normal')
#  M.use_step_method(pymc.Metropolis, M.pole_real, proposal_sd=0.5, proposal_distribution='Normal')
#  M.use_step_method(pymc.Metropolis, M.pole_imag, proposal_sd=0.1, proposal_distribution='Normal')

  for idx in range(numWaveforms):
    M.use_step_method(pymc.AdaptiveMetropolis, [M.radiusArray[idx], M.zArray[idx]],
                     scales={M.radiusArray[idx]:  10,
                             M.zArray[idx]:       10}, delay=100, interval=100,shrink_if_necessary=True)
                      

#    M.use_step_method(pymc.Metropolis, M.radiusArray[idx], proposal_sd=10., proposal_distribution='Normal')
#    M.use_step_method(pymc.Metropolis, M.zArray[idx], proposal_sd=10., proposal_distribution='Normal')

    M.use_step_method(pymc.Metropolis, M.phiArray[idx], proposal_sd=0.3, proposal_distribution='Normal')
    M.use_step_method(pymc.Metropolis, M.scaleArray[idx], proposal_sd=0.01*startGuess['wfScale'][idx], proposal_distribution='Normal')
    M.use_step_method(pymc.Metropolis, M.t0Array[idx], proposal_sd=5, proposal_distribution='Normal')
    M.use_step_method(pymc.Metropolis, M.sigArray[idx], proposal_sd=0.5, proposal_distribution='Normal')


  # morning 9/24


#  M.use_step_method(pymc.Metropolis, M.tempEst, proposal_sd=3., proposal_distribution='Normal')
#  M.use_step_method(pymc.Metropolis, M.grad, proposal_sd=0.005, proposal_distribution='Normal')
#  M.use_step_method(pymc.Metropolis, M.pcRad, proposal_sd=0.05, proposal_distribution='Normal')
#  M.use_step_method(pymc.Metropolis, M.pcLen, proposal_sd=0.05, proposal_distribution='Normal')
#  M.use_step_method(pymc.Metropolis, M.zero_1, proposal_sd=0.01, proposal_distribution='Normal')
#  M.use_step_method(pymc.Metropolis, M.pole_1, proposal_sd=0.001, proposal_distribution='Normal')
#  M.use_step_method(pymc.Metropolis, M.pole_real, proposal_sd=0.1, proposal_distribution='Normal')
#  M.use_step_method(pymc.Metropolis, M.pole_imag, proposal_sd=0.01, proposal_distribution='Normal')
#  
#  for idx in range(numWaveforms):
#    M.use_step_method(pymc.AdaptiveMetropolis, [M.radiusArray[idx], M.zArray[idx]],
#                     scales={M.radiusArray[idx]:  10,
#                             M.zArray[idx]:       10}, delay=100, interval=100,shrink_if_necessary=True)
#                      
#
##    M.use_step_method(pymc.Metropolis, M.radiusArray[idx], proposal_sd=10., proposal_distribution='Normal')
##    M.use_step_method(pymc.Metropolis, M.zArray[idx], proposal_sd=10., proposal_distribution='Normal')
#
#    M.use_step_method(pymc.Metropolis, M.phiArray[idx], proposal_sd=0.3, proposal_distribution='Normal')
#    M.use_step_method(pymc.Metropolis, M.scaleArray[idx], proposal_sd=0.01*startGuess['wfScale'][idx], proposal_distribution='Normal')
#    M.use_step_method(pymc.Metropolis, M.t0Array[idx], proposal_sd=5, proposal_distribution='Normal')
#    M.use_step_method(pymc.Metropolis, M.sigArray[idx], proposal_sd=0.5, proposal_distribution='Normal')


#
#  M.use_step_method(pymc.AdaptiveMetropolis, [M.tempEst, M.grad, M.pcRad, M.pcLen,M.zero_1,M.pole_1,M.pole_real,M.pole_imag],
#                 scales={M.tempEst:  .3,
#                         M.grad:  0.001,
#                         M.pcRad: 0.01,
#                         M.pcLen: 0.01,
#                         M.zero_1:0.01,
#                         M.pole_1:0.001,
#                         M.pole_real:0.01,
#                         M.pole_imag:0.001
#                         }, delay=100, interval=100,shrink_if_necessary=True)
#
##  zero_1 = 0.474472
##  pole_1 = 0.999845
##  pole_real = 0.807801
##  pole_imag = 0.081791
##
#  for idx in range(numWaveforms):
#    M.use_step_method(pymc.AdaptiveMetropolis, [M.radiusArray[idx],M.phiArray[idx],M.zArray[idx],
#                                                M.scaleArray[idx], M.t0Array[idx], M.sigArray[idx],],
#                     scales={M.radiusArray[idx]:  10,
#                             M.phiArray[idx]:     np.pi/4/4,
#                             M.zArray[idx]:       10,
#                             M.scaleArray[idx]:   0.01*startGuess['wfScale'][idx],
#                             M.t0Array[idx]:      5,
#                             M.sigArray[idx]:     0.5
#                             }, delay=100, interval=100,shrink_if_necessary=True)


  M.sample(iter=numSamples, burn=0, tune_interval=100)
  M.db.close()

#  totalIter = 0
#  while totalIter < this_sample:
#    M.sample(iter=10, verbose=0)
#    totalIter += 10

  
#  #pymc.Matplot.plot(M)
#  
#  #########  Plots for MC Steps
  stepsFig = plt.figure(figsize=(20,10))
  plt.clf()
  ax0 = stepsFig.add_subplot(611)
  ax1 = stepsFig.add_subplot(612, sharex=ax0)
  ax2 = stepsFig.add_subplot(613, sharex=ax0)
  ax3 = stepsFig.add_subplot(614, sharex=ax0)
  ax4 = stepsFig.add_subplot(615, sharex=ax0)
  ax5 = stepsFig.add_subplot(616, sharex=ax0)

  ax0.set_ylabel('r')
  ax1.set_ylabel('z')
  ax2.set_ylabel('phi')
  ax3.set_ylabel('e')
  ax4.set_ylabel('t0')
  ax5.set_ylabel('sig')

  for i in range(len(wfs)):
    ax0.plot(M.trace('radEst_%d'%i)[:])
    ax1.plot(M.trace('zEst_%d'%i)[:])
    ax2.plot(M.trace('phiEst_%d'%i)[:])
    ax3.plot(M.trace('wfScale_%d'%i)[:])
    ax4.plot(M.trace('switchpoint_%d'%i)[:])
    ax5.plot(M.trace('sigma_%d'%i)[:])

  plt.savefig("pymc_wf_params.png")
#
  stepsFig2 = plt.figure(figsize=(20,10))
  plt.clf()
  ax0 = stepsFig2.add_subplot(811)
  ax1 = stepsFig2.add_subplot(812, sharex=ax0)
  ax2 = stepsFig2.add_subplot(813, sharex=ax0)
  ax3 = stepsFig2.add_subplot(814, sharex=ax0)
  ax4 = stepsFig2.add_subplot(815, sharex=ax0)
  ax5 = stepsFig2.add_subplot(816, sharex=ax0)
  ax6 = stepsFig2.add_subplot(817, sharex=ax0)
  ax7 = stepsFig2.add_subplot(818, sharex=ax0)

  ax0.set_ylabel('zero_1')
  ax1.set_ylabel('pole_1')
  ax2.set_ylabel('pole_real')
  ax3.set_ylabel('pole_imag')
  ax4.set_ylabel('temp')
  ax5.set_ylabel('grad')
  ax6.set_ylabel('pcRad')
  ax7.set_ylabel('pcLen')

  ax0.plot(M.trace('zero_1')[:])
  ax1.plot(M.trace('pole_1')[:])
  ax2.plot(M.trace('pole_real')[:])
  ax3.plot(M.trace('pole_imag')[:])
  ax4.plot(M.trace('temp')[:])
  ax5.plot(M.trace('grad')[:])
  ax6.plot(M.trace('pcRad')[:])
  ax7.plot(M.trace('pcLen')[:])

  plt.savefig("pymc_detector.png")


  fig3 = plt.figure(3, figsize = (20,10))
  plt.clf()
  plt.title("Charge waveform")
  plt.xlabel("Sample number [10s of ns]")
  plt.ylabel("Raw ADC Value [Arb]")
      
  wfPlotNumber = 10
  simWfArr = np.empty((wfPlotNumber,numWaveforms, fitSamples))


  if burnin > len(M.trace('temp')[:]):
    burnin = len(M.trace('temp')[:]) - wfPlotNumber
    numSamples = len(M.trace('temp')[:])

  for (sim_idx, chain_idx) in enumerate(np.random.randint(low=burnin, high=numSamples, size=wfPlotNumber)):
    
      temp = M.trace('temp')[chain_idx]
      grad = M.trace('grad')[chain_idx]
      pcRad= M.trace('pcRad')[chain_idx]
      pcLen= M.trace('pcLen')[chain_idx]
      zero_1 = M.trace('zero_1')[chain_idx]
      pole_1 = M.trace('pole_1')[chain_idx]
      pole_real = M.trace('pole_real')[chain_idx]
      pole_imag = M.trace('pole_imag')[chain_idx]
      
      zeros = [zero_1, -1., 1. ]
      poles = [pole_1, pole_real+pole_imag*1j, pole_real-pole_imag*1j, ]
      det.SetTransferFunction(zeros, poles)
      det.SetTemperature(temp)
      det.SetFields(pcRad, pcLen, grad)
  
      for (wf_idx, wf) in enumerate(wfs):
        t0 =    M.trace('switchpoint_%d' % wf_idx)[chain_idx]
        r =     M.trace('radEst_%d' % wf_idx)[chain_idx]
        z =     M.trace('zEst_%d' % wf_idx)[chain_idx]
        phi =   M.trace('phiEst_%d' % wf_idx)[chain_idx]
        scale = M.trace('wfScale_%d' % wf_idx)[chain_idx]
        sigma = M.trace('sigma_%d' % wf_idx)[chain_idx]

        simWfArr[sim_idx,wf_idx,:]  = det.GetSimWaveform(r, phi, z, scale, t0, fitSamples, smoothing=sigma)
  helpers.plotManyResidual(simWfArr, wfs, fig3, residAlpha=1)

  plt.savefig("pymc_waveforms.png")

  value = raw_input('  --> Press q to quit, any other key to continue\n')


if __name__=="__main__":
    main(sys.argv[1:])


