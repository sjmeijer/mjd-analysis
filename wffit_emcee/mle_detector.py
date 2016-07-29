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

  plt.ion()

  runRange = (13420,13429)
  channel = 626
  aeCutVal = 0.01425
  
  
  fitSamples = 200
  numWaveforms = 4
  numThreads = 4
  

  #Prepare detector
  num = [9444619140.7602386, 5.6541009375868186e+17, 6862099457187322.0]
  den = [1, 39786017.017542012, 507044769163371.25, 6.6153719754940365e+18]
  system = signal.lti(num, den)
  
  tempGuess = 81.6
  gradGuess = 0.0388
  pcRadGuess = 2.51
  pcLenGuess = 1.607490

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.04,2.5, 1.6)
  det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
  det.LoadFields("P42574A_fields_len.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  init_detector(det)
  
  wfFileName = "P42574A_%dwaveforms.npz" % numWaveforms
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
    print "No saved waveforms available.  Loading from Data"
    #get waveforms
    cut = "trapECal>%f && trapECal<%f && TSCurrent100nsMax/trapECal > %f" %  (1588,1594, aeCutVal)
    wfs = helpers.GetWaveforms(runRange, channel, numWaveforms, cut)

    #prep holders for each wf-specific param
    r_arr = np.empty(numWaveforms)
    phi_arr = np.empty(numWaveforms)
    z_arr = np.empty(numWaveforms)
    scale_arr = np.empty(numWaveforms)
    t0_arr = np.empty(numWaveforms)
    smooth_arr = np.empty(numWaveforms)

    #ML as a start, using each individual wf
    nll_wf = lambda *args: -lnlike_waveform(*args)
    
    start = timer()
    for (idx,wf) in enumerate(wfs):
      print "Doing MLE for waveform %d" % idx
      wf.WindowWaveformTimepoint(fallPercentage=.995)
      
      startGuess = [15./10, np.pi/8, 15./10, wf.wfMax/1000., wf.t0Guess, 10.]
      result = op.minimize(nll_wf, startGuess, args=(wf) ,method="Nelder-Mead")
      r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx] = result["x"]
    end = timer()
    print "Elapsed time: " + str(end-start)
    np.savez(wfFileName, wfs = wfs, r_arr=r_arr, phi_arr = phi_arr, z_arr = z_arr, scale_arr = scale_arr,  t0_arr=t0_arr, smooth_arr=smooth_arr  )

  init_wfs(wfs)


  #plot the wf-wise MLE:
  if True:
    fig = plt.figure(figsize=(20,10))
    simWfArr = np.empty((1,numWaveforms, fitSamples))
    for (idx,wf) in enumerate(wfs):
      print "  >> wf %d:" % (idx)
      print "      r: %0.2f, phi: %0.3f, z: %0.2f, e: %0.2f, t0: %0.2f, smooth:%0.2f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx])
      simWfArr[0,idx,:] = det.GetSimWaveform(r_arr[idx]*10, phi_arr[idx], z_arr[idx]*10, scale_arr[idx]*1000, t0_arr[idx], fitSamples, smoothing = smooth_arr[idx] )
    helpers.plotManyResidual(simWfArr, wfs, fig, residAlpha=1)
    value = raw_input('  --> Press q to quit, any other key to continue\n')
    if value == 'q': exit(0)

  #do the detector-wide fit
  if True:
    print "Starting detector MLE..."
    detmleFileName = "P42574A_%dwaveforms_detectormle.npz" % numWaveforms

    nll_det = lambda *args: -lnlike_detector(*args)
    
    num = [9.408524745, 5.6758925068323565, 6.777]
    den = [ 1., 4., 5.12, 6.49]

    tf_bound = 2.
    tempMult = 10.
    gradMult = 100.

    mcmc_startguess = np.hstack((tempGuess/tempMult, gradGuess*gradMult, pcRadGuess, pcLenGuess, num[:], den[1:]))
    wfParams = np.hstack((r_arr[:], phi_arr[:], z_arr[:], scale_arr[:], t0_arr[:],smooth_arr[:],) )
    
    bounds = [ (70./tempMult, 100./tempMult), (det.gradList[0]*gradMult, det.gradList[-1]*gradMult), (det.pcRadList[0], det.pcRadList[-1]),
                (num[0]/tf_bound, num[0]*tf_bound),(num[1]/tf_bound, num[1]*tf_bound),(num[2]/tf_bound, num[2]*tf_bound),
                (den[1]/tf_bound, den[1]*tf_bound),(den[2]/tf_bound, den[2]*tf_bound),(den[3]/tf_bound, den[3]*tf_bound) ]

    p = Pool(numThreads)

    #fitting only the 3 detector params
    start = timer()
    #result = op.basinhopping(nll_det, mcmc_startguess, minimizer_kwargs={  "method":"Nelder-Mead", "tol":50, "args": (p, wfParams)})
    result = op.minimize(nll_det, mcmc_startguess, method="Nelder-Mead", tol=1., args=(p, wfParams))
    #result = op.minimize(nll_det, mcmc_startguess, method="L-BFGS-B", tol=5, bounds=bounds, args=(p, wfParams))
    #result = op.differential_evolution(nll_det, bounds, args=(p, wfParams), polish=False)
    end = timer()
    print "Elapsed time: " + str(end-start)
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
    simWfArr = np.empty((1,numWaveforms, fitSamples))
    
    #one last minimization for the plotzzz
    args = []
    for idx in np.arange(numWaveforms):
      args.append( [r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx],  wfs[idx] ]  )
    results = p.map(minimize_waveform_only_star, args)
    
    for (idx,result) in enumerate(results):
      r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx] = result["x"]
      print "  >> wf %d (normalized likelihood %0.2f):" % (idx, result["fun"]/wfs[idx].wfLength)
      print "      r: %0.2f, phi: %0.3f, z: %0.2f, e: %0.2f, t0: %0.2f, smooth:%0.2f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx])
      
      simWfArr[0,idx,:]   = det.GetSimWaveform(r_arr[idx]*10, phi_arr[idx], z_arr[idx]*10, scale_arr[idx]*1000, t0_arr[idx], fitSamples)
    helpers.plotManyResidual(simWfArr, wfs, fig2, residAlpha=1)

    plt.savefig("mle_waveforms.pdf")
    np.savez(detmleFileName, wfs = wfs, r_arr=r_arr, phi_arr = phi_arr, z_arr = z_arr, scale_arr = scale_arr,  t0_arr=t0_arr,  temp=temp, impGrad=impGrad, pcRad=pcRad)
    value = raw_input('  --> Press q to quit, any other key to continue\n')
    exit(0)



if __name__=="__main__":
    main(sys.argv[1:])


