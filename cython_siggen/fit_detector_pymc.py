#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy as np
from scipy import signal

import helpers
from detector_model import *

from pymc3 import *
import signal_model_hierarchical as sm3


def main(argv):

  plt.ion()

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
    print "No saved waveforms available.  Loading from Data"
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


  #Plot the waveforms to take a look at the initial guesses
  if doInitPlot:
    plt.ion()
    fig = plt.figure()
    for (idx,wf) in enumerate(wfs):
      
      print "WF number %d:" % idx
      print "  >>r: %f\n  >>phi %f\n  >>z %f\n  >>e %f\n  >>t0 %f\n >>smooth %f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx])
      ml_wf = det.GetSimWaveform(r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx]*100, t0_arr[idx], fitSamples, smoothing = smooth_arr[idx])
      plt.plot(ml_wf, color="b")
      plt.plot(wf.windowedWf, color="r")
    value = raw_input('  --> Press q to quit, any other key to continue\n')
    plt.ioff()
    if value == 'q': exit(0)

  startGuess = {'radEst': r_arr,
                'zEst': z_arr,
                'phiEst': phi_arr,
                'wfScale': scale_arr*100,
                'switchpoint': t0_arr,
                'smooth': smooth_arr,
                'temp': tempGuess,
                'grad': gradGuess,
                'pcRad': pcRadGuess,
                'pcLen':pcLenGuess}

  siggen_model = sm3.CreateFullDetectorModel(det, wfs,  startGuess, zero_1, pole_1, pole_real, pole_imag)
  with siggen_model:
  
    step = Metropolis(scaling = [np.ones(1)*1,
                                np.ones(1)*5,
                                np.ones(1)*np.pi/4/10,
                                np.ones(1)*5,
                                np.ones(1)*100,
                                np.ones(1)*0.1,
                                3, 0.01, 0.0001, 0.01, 0.001, 0.005, 0.05, 0.05])
    #step = Metropolis(scaling=0.001)
    trace = sample(numSamples,  step = step, )
    
    burnin = np.int(len(trace['temp'][:]) - 0.25*len(trace['temp'][:]))#no clue what to choose, for now, just make it 75%
  
    temp =  np.median(  trace['temp'][burnin:])
    print "<<<detector temperature is %f" % temp
    grad =       np.median(  trace['grad'][burnin:])
    pcRad=      np.median(  trace['pcRad'][burnin:])
    pcLen=      np.median(  trace['pcLen'][burnin:])
    print "<<<detector gradient is %0.4f " % (grad)
    print "<<<PC Radius is %0.4f, Length is %0.4f " % (pcRad, pcLen)
    
    zero_1 =      np.median(  trace['zero_1'][burnin:])
    pole_1 =      np.median(  trace['pole_1'][burnin:])
    pole_real =      np.median(  trace['pole_real'][burnin:])
    pole_imag =      np.median(  trace['pole_imag'][burnin:])

    print "<<< zero_1=%e"   % zero_1
    print "<<< pole_1=%e"   % pole_1
    print "<<< pole_real=%e"% pole_real
    print "<<< pole_imag=%e"% pole_imag

    plt.ioff()
    traceplot(trace)
    plt.savefig("hierarchical_full_%dchain.png" % len(wfs))
    plt.ion()
    
    fig3 = plt.figure(3, figsize = (20,10))
    plt.clf()
    plt.title("Charge waveform")
    plt.xlabel("Sample number [10s of ns]")
    plt.ylabel("Raw ADC Value [Arb]")
    
    wfPlotNumber = 10
    simWfArr = np.empty((wfPlotNumber,numWaveforms, fitSamples))
    
    for (sim_idx, chain_idx) in enumerate(np.random.randint(low=burnin, high=numSamples, size=wfPlotNumber)):
    
      temp = trace['temp'][chain_idx]
      grad = trace['grad'][chain_idx]
      pcRad= trace['pcRad'][chain_idx]
      pcLen= trace['pcLen'][chain_idx]
      zero_1 = trace['zero_1'][chain_idx]
      pole_1 = trace['pole_1'][chain_idx]
      pole_real = trace['pole_real'][chain_idx]
      pole_imag = trace['pole_imag'][chain_idx]
      
      zeros = [zero_1, -1., 1. ]
      poles = [pole_1, pole_real+pole_imag*1j, pole_real-pole_imag*1j, ]
      det.SetTransferFunction(zeros, poles)
      det.SetTemperature(temp)
      det.SetFields(pcRad, pcLen, grad)
  
      for (wf_idx, wf) in enumerate(wfs):
        t0 =    trace['switchpoint'][chain_idx,wf_idx]
        r =     trace['radEst'][chain_idx,wf_idx]
        z =     trace['zEst'][chain_idx,wf_idx]
        phi =   trace['phiEst'][chain_idx,wf_idx]
        scale = trace['wfScale'][chain_idx,wf_idx]
        sigma = trace['sigma'][chain_idx,wf_idx]

        simWfArr[sim_idx,wf_idx,:]  = det.GetSimWaveform(r, phi, z, scale, t0, fitSamples, smoothing=sigma)
    helpers.plotManyResidual(simWfArr, wfs, fig3, residAlpha=1)

#
    plt.savefig("hierarchical_full_%dwaveforms.png" % len(wfs))
    summary(trace)
    value = raw_input('  --> Press q to quit, any other key to continue\n')


if __name__=="__main__":
    main(sys.argv[1:])


