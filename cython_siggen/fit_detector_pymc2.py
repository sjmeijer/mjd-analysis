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

  fitSamples = 210
  timeStepSize = 10. #ns
  numSamples = 10000
  
  doInitPlot = False
  
  #Prepare detector
  zero_1 = 0.474472
  pole_1 = 0.999845
  pole_real = 0.807801
  pole_imag = 0.081791

  zeros = [zero_1, -1., 1. ]
  poles = [pole_1, pole_real+pole_imag*1j, pole_real-pole_imag*1j, ]
  
  tempGuess = 77.747244
  gradGuess = 0.046950
  pcRadGuess = 2.547418
  pcLenGuess = 1.569172

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10./timeStepSize, poles=poles, zeros=zeros)
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  
  wfFileName = "P42574A_512waveforms_8risetimeculled.npz"
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

  M = pymc.MCMC(model_locals)
#  M.sample(iter=100, burn=90)


  M.use_step_method(pymc.AdaptiveMetropolis, [M.tempEst, M.grad, M.pcRad, M.pcLen,M.zero_1,M.pole_1,M.pole_real,M.pole_imag],
                 scales={M.tempEst:  .3,
                         M.grad:  0.001,
                         M.pcRad: 0.01,
                         M.pcLen: 0.01,
                         M.zero_1:0.01,
                         M.pole_1:0.001,
                         M.pole_real:0.01,
                         M.pole_imag:0.001
                         }, delay=100, interval=100,shrink_if_necessary=True)

#  zero_1 = 0.474472
#  pole_1 = 0.999845
#  pole_real = 0.807801
#  pole_imag = 0.081791
#
#  for idx in range(numWaveforms):
#    M.use_step_method(pymc.AdaptiveMetropolis, [M.radiusArray[idx],M.phiArray[idx],M.zArray[idx],
#                                                M.scaleArray[idx], M.t0Array[idx], M.sigArray[idx],],
#                     scales={M.radiusArray[idx]:  10,
#                             M.phiArray[idx]:     np.pi/4/4,
#                             M.zArray[idx]:       10,
#                             M.scaleArray[idx]:   0.01*startGuess['wfScale'][idx],
#                             M.t0Array[idx]:      5,
#                             M.sigArray[idx]:     0.5
#                             }, delay=100)


  M.sample(iter=1000, burn=0)

#  totalIter = 0
#  while totalIter < this_sample:
#    M.sample(iter=10, verbose=0)
#    totalIter += 10

  
#  #pymc.Matplot.plot(M)
#  
#  #########  Plots for MC Steps
#  stepsFig = plt.figure(figsize=(20,10))
#  plt.clf()
#  ax0 = stepsFig.add_subplot(511)
#  ax1 = stepsFig.add_subplot(512, sharex=ax0)
#  ax2 = stepsFig.add_subplot(513, sharex=ax0)
#  ax3 = stepsFig.add_subplot(514, sharex=ax0)
#  ax4 = stepsFig.add_subplot(515, sharex=ax0)
#  ax0.set_ylabel('r')
#  ax1.set_ylabel('z')
#  ax2.set_ylabel('phi')
#  ax3.set_ylabel('e')
#  ax4.set_ylabel('t0')
#
#  for i in range(len(wfs)):
#    ax0.plot(M.trace('radEst_%d'%i)[:])
#    ax1.plot(M.trace('zEst_%d'%i)[:])
#    ax2.plot(M.trace('phiEst_%d'%i)[:])
#    ax3.plot(M.trace('wfScale_%d'%i)[:])
#    ax4.plot(M.trace('switchpoint_%d'%i)[:])
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

  plt.show()

  value = raw_input('  --> Press q to quit, any other key to continue\n')


if __name__=="__main__":
    main(sys.argv[1:])


