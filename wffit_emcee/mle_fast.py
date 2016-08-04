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
from probability_model_hier import *

from timeit import default_timer as timer
from multiprocessing import Pool

def main(argv):

  numThreads = 4
  numWfs = 4

  plt.ion()
  
  r_mult = 1.
  z_mult = 1.
  scale_mult = 100.
  
  fitSamples = 200

  #Prepare detector
  num =  [8685207069.0676746, 1.7618952141698222e+18, 17521485536930826.0]
  den = [1, 50310572.447231829, 701441983664560.88, 1.4012406413698292e+19]
  system = signal.lti(num, den)
  
  tempGuess = 82.48
  gradGuess = 0.0482
  pcRadGuess = 2.563885
  pcLenGuess = 1.440751

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.04,2.5, 1.6)
  det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
  det.LoadFields("P42574A_fields_len.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  simWfArr = np.empty((1,numWfs, fitSamples))  
  
  wfFileName = "P42574A_32waveforms_risetimeculled.npz"
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    r_arr  = data['r_arr']
    phi_arr = data['phi_arr']
    z_arr = data['z_arr']
    scale_arr = data['scale_arr']
    t0_arr = data['t0_arr']
    smooth_arr = data['smooth_arr']
    wfs = data['wfs']
  
  else:
    exit(0)

  initializeDetectorAndWfs(det, wfs[:numWfs])
  p = Pool(numThreads, initializer=initializeDetectorAndWfs, initargs=[det, wfs[:numWfs]])

  args = []
  for (idx, wf) in enumerate(wfs[:numWfs]):
    args.append( [15./r_mult, np.pi/8., 15./z_mult, wf.wfMax/scale_mult, wf.t0Guess, 10.,  wfs[idx] ]  )
  print "performing parallelized initial fit..."
  start = timer()
  results = p.map(minimize_waveform_only_star, args)
  end = timer()

  print "Initial fit time: %f" % (end-start)


  for (idx,result) in enumerate(results):
    r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx] = result["x"]
    print "  >> wf %d (normalized likelihood %0.2f):" % (idx, result["fun"]/wfs[idx].wfLength)
    print "      r: %0.2f, phi: %0.3f, z: %0.2f, e: %0.2f, t0: %0.2f, smooth:%0.2f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx])
    
    simWfArr[0,idx,:] = det.GetSimWaveform(r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx]*100., t0_arr[idx],  fitSamples, smoothing=smooth_arr[idx],)


  fig1 = plt.figure(figsize=(20,15))
  helpers.plotManyResidual(simWfArr, wfs[:numWfs], fig1, residAlpha=1)
  value = raw_input('  --> Press q to quit, any other key to continue\n')
  if value == 'q': exit(0)

  tempMult = 10.
  gradMult = 100.
  


  mcmc_startguess = np.hstack((tempGuess/tempMult, gradGuess*gradMult, pcRadGuess, pcLenGuess, num[:], den[1:]))
  wfParams = np.hstack((np.ones(numWfs)*3., phi_arr[:numWfs], np.ones(numWfs)*3., scale_arr[:numWfs], t0_arr[:numWfs],smooth_arr[:numWfs],) )

  nll_det = lambda *args: -lnlike_detector(*args)
  result = op.minimize(nll_det, mcmc_startguess, method="Nelder-Mead", args=(p, wfParams))

  temp, impGrad, pcRad, pcLen = result["x"][0], result["x"][1], result["x"][2],  result["x"][3]
  temp *= tempMult
  impGrad /= 100.
  
  tfStartIdx = 4
  num = [result["x"][tfStartIdx] *1E9 , result["x"][tfStartIdx+1] *1E17, result["x"][tfStartIdx+2]*1E15 ]
  den = [1, result["x"][tfStartIdx+3] *1E7 , result["x"][tfStartIdx+4] *1E14, result["x"][tfStartIdx+5]*1E18 ]


  print "MLE Values..."
  print "  >> temp is %f" % temp
  print "  >> grad is %f" % impGrad
  print "  >> pc rad is %f" % pcRad
  print "  >> pc len is %f" % pcLen

  print " >> num = [%e, %e, %e]" % (num[0], num[1], num[2])
  print " >> den = [1., %e, %e, %e]" % (den[1], den[2], den[3])


  print "Plotting best fit..."
  
  fig2 = plt.figure(figsize=(20,10))
  det.SetTemperature(temp)
  det.SetFields(pcRad, pcLen, impGrad)
  det.SetTransferFunction(num, den)
  simWfArr = np.empty((1,numWfs, fitSamples))

  #one last minimization for the plotzzz
  args = []
  for idx in np.arange(numWfs):
    args.append( [r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx],  wfs[idx] ]  )
  
  results = p.map(minimize_waveform_only_star, args)
  
  for (idx,result) in enumerate(results):
    r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx] = result["x"]
    print "  >> wf %d (normalized likelihood %0.2f):" % (idx, result["fun"]/wfs[idx].wfLength)
    print "      r: %0.2f, phi: %0.3f, z: %0.2f, e: %0.2f, t0: %0.2f, smooth:%0.2f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx])
    
    simWfArr[0,idx,:]   = det.GetSimWaveform(r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx]*100, t0_arr[idx], fitSamples, smoothing=smooth_arr[idx])
  helpers.plotManyResidual(simWfArr, wfs[:numWfs], fig2, residAlpha=1)

  plt.savefig("mle_waveforms_fast.pdf")
  value = raw_input('  --> Press q to quit, any other key to continue\n')
  exit(0)


if __name__=="__main__":
    main(sys.argv[1:])


