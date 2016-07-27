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

def main(argv):

  plt.ion()

  runRange = (13420,13429)
  channel = 626
  aeCutVal = 0.01425
  
  numThreads=1
  tempGuess = 80
  fitSamples = 200
  numWaveforms = 5
  

  #Prepare detector
  num = [3.64e+09, 1.88e+17, 6.05e+15]
  den = [1, 4.03e+07, 5.14e+14, 7.15e+18]
  
#  num = [1977053898.8128378, 1.88e+17, 6050000000000461.0]
#  den = [1, 39709003.905781068, 514546561393036.69, 7.15e+18]

  system = signal.lti(num, den)
  
  gradGuess = 0.04
  pcRadGuess = 2.3

  #Create a detector model
  detName = "conf/P42574A_grad%0.3f_pcrad%0.4f.conf" % (gradGuess,pcRadGuess)
  det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
  det.LoadFields("P42574A_fields.npz")
#  det.SetFields(2.28, 0.043)
  init_detector(det)
  
  
  tempIdx = -9
  gradIdx = -8
  pcRadIdx = -7
  
  wfFileName = "P42574A_%dwaveforms.npz" % numWaveforms
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    r_arr  = data['r_arr']
    phi_arr = data['phi_arr']
    z_arr = data['z_arr']
    scale_arr = data['scale_arr']
    t0_arr = data['t0_arr']
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

    #ML as a start, using each individual wf
    nll_wf = lambda *args: -lnlike_waveform(*args)
    
    start = timer()
    for (idx,wf) in enumerate(wfs):
      print "Doing MLE for waveform %d" % idx
      wf.WindowWaveformTimepoint(fallPercentage=.99)
      
      
      startGuess = [15./10, np.pi/8, 15./10, wf.wfMax/1000, wf.t0Guess]
      
      
      result = op.minimize(nll_wf, startGuess, args=(wf) ,method="Nelder-Mead")
      r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx] = result["x"]
    end = timer()
    print "Elapsed time: " + str(end-start)
    np.savez(wfFileName, wfs = wfs, r_arr=r_arr, phi_arr = phi_arr, z_arr = z_arr, scale_arr = scale_arr,  t0_arr=t0_arr,  )


  init_wfs(wfs)


  #plot the wf-wise MLE:
  if True:
    fig = plt.figure()
    simWfArr = np.empty((1,numWaveforms, fitSamples))
    for (idx,wf) in enumerate(wfs):
      print "r: %f\nphi %f\nz %f\n e %f\nt0 %f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx])
      simWfArr[0,idx,:] = det.GetSimWaveform(r_arr[idx]*10, phi_arr[idx], z_arr[idx]*10, scale_arr[idx]*1000, t0_arr[idx], fitSamples)
    helpers.plotManyResidual(simWfArr, wfs, fig, residAlpha=1)
    value = raw_input('  --> Press q to quit, any other key to continue\n')

  if True:
    print "Starting detector MLE..."
    detmleFileName = "P42574A_%dwaveforms_detectormle.npz" % numWaveforms

    nll_det = lambda *args: -lnlike_detector_holdpos(*args)
    
#    num = [3.64e+09, 1.88e+17, 6.05e+15]
#    den = [1, 4.03e+07, 5.14e+14, 7.15e+18]
    num = [3.64, 1.88, 6.05]
    den = [1, 4.03, 5.14, 7.15]

    mcmc_startguess = np.hstack((tempGuess/100, gradGuess*10,pcRadGuess, num[:], den[1:]))
    wfParams = np.hstack((r_arr[:], phi_arr[:], z_arr[:], scale_arr[:], t0_arr[:],) )

    #fitting only the 3 detector params
    start = timer()
    #result = op.basinhopping(nll_det, mcmc_startguess, minimizer_kwargs={  "method":"Nelder-Mead", "tol":0.5, "args": (wfParams)})
    #result = op.minimize(nll_det, mcmc_startguess, method="Nelder-Mead", tol=10, args=(wfParams))#options={"fatol":0.5, "xatol":5})
    
    bounds = [ (.70, 1.00), (det.gradList[0]*10, det.gradList[-1]*10), (det.pcRadList[0], det.pcRadList[-1]),
                (0.1, 10),(0.1, 10),(0.1, 10),(0.1, 10),(0.1, 10),(0.1, 10) ]
    result = op.differential_evolution(nll_det, bounds, args=(wfParams), polish=False)
    end = timer()
    print "Elapsed time: " + str(end-start)
    temp, impGrad, pcRad = result["x"][0], result["x"][1], result["x"][2]

    temp *= 100
    impGrad /= 10
    num = [result["x"][3] *1E9 , result["x"][4] *1E17, result["x"][5]*1E15 ]
    den = [1, result["x"][6] *1E7 , result["x"][7] *1E14, result["x"][8]*1E18 ]

    print "MLE temp is %f" % temp
    print "MLE grad is %f" % impGrad
    print "MLE pc rad is %f" % pcRad
    
    fig2 = plt.figure()
    det.SetTemperature(temp)
    det.SetFields(pcRad, impGrad)
    det.SetTransferFunction(num, den)
    simWfArr = np.empty((1,numWaveforms, fitSamples))
    for (idx,wf) in enumerate(wfs):
      simWfArr[0,idx,:]   = det.GetSimWaveform(r_arr[idx]*10, phi_arr[idx], z_arr[idx]*10, scale_arr[idx]*1000, t0_arr[idx], fitSamples)
    helpers.plotManyResidual(simWfArr, wfs, fig2, residAlpha=1)

    np.savez(detmleFileName, wfs = wfs, r_arr=r_arr, phi_arr = phi_arr, z_arr = z_arr, scale_arr = scale_arr,  t0_arr=t0_arr,  temp=temp, impGrad=impGrad, pcRad=pcRad)
    value = raw_input('  --> Press q to quit, any other key to continue\n')
    exit(0)



if __name__=="__main__":
    main(sys.argv[1:])


