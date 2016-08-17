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
from probability_model_hier import *

from pymc3 import *
import signal_model_hierarchical as sm3


def main(argv):

  plt.ion()

  
  fitSamples = 200
  
  #Prepare detector
  num =  [3478247474.8078203, 1.9351287044375424e+17, 6066014749714584.0]
  den = [1, 40525756.715025946, 508584795912802.25, 7.0511687850000589e+18]
  system = signal.lti(num, den)
  
  tempGuess = 77.89
  gradGuess = 0.0483
  pcRadGuess = 2.591182
  pcLenGuess = 1.613357

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.04,2.5, 1.6)
  det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  
  
  wfFileName = "P42574A_512waveforms_16risetimeculled.npz"
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
  smooth_arr = np.ones(numWaveforms)*7.

  simWfArr = np.empty((1,numWaveforms, fitSamples))

  args = []
  for (idx, wf) in enumerate(wfs):
    wf.WindowWaveformTimepoint(fallPercentage=.99)
    r_arr[idx], phi_arr[idx], z_arr[:], scale_arr[idx], t0_arr[idx], smooth_arr[idx]  = results[idx]['x']
    args.append( [r_arr[idx], phi_arr[idx], z_arr[:], scale_arr[idx], wf.t0Guess,  5.,  wfs[idx] ]  )

  if True:
#    p = Pool(8, initializer=initializeDetector, initargs=[det])
#    print "performing parallelized initial fit..."
#    results = p.map(minimize_waveform_only_star, args)
#    np.savez(wfFileName, wfs = wfs, results=results )

    fig = plt.figure()
    for (idx,wf) in enumerate(wfs):
      wf.WindowWaveform(200)
      r_arr[idx], phi_arr[idx], z_arr[:], scale_arr[idx], t0_arr[idx], smooth_arr[idx]  = results[idx]['x']
      
      print "WF number %d:" % idx
      print "  >>r: %f\n  >>phi %f\n  >>z %f\n  >>e %f\n  >>t0 %f\n >>smooth %f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx])
      ml_wf = det.GetSimWaveform(r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx]*100, t0_arr[idx], fitSamples, smoothing = smooth_arr[idx])
      plt.plot(ml_wf, color="b")
      plt.plot(wf.windowedWf, color="r")
    value = raw_input('  --> Press q to quit, any other key to continue\n')
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

  siggen_model = sm3.CreateFullDetectorModel(det, wfs,  startGuess, num, den )
  with siggen_model:
  
    step = Metropolis()
    trace = sample(500,  step = step)
    
    burnin = np.int(len(trace['temp'][:]) - 0.25*len(trace['temp'][:]))#no clue what to choose, for now, just make it 75%
  
    temp =  np.median(  trace['temp'][burnin:])
    print "<<<detector temperature is %f" % temp
    grad =       np.median(  trace['grad'][burnin:])
    pcRad=      np.median(  trace['pcRad'][burnin:])
    pcLen=      np.median(  trace['pcLen'][burnin:])
    print "<<<detector gradient is %0.4f " % (grad)
    print "<<<PC Radius is %0.4f, Length is %0.4f " % (pcRad, pcLen)
    
    num_1 =      np.median(  trace['num_1'][burnin:])
    num_2 =      np.median(  trace['num_2'][burnin:])
    num_3 =      np.median(  trace['num_3'][burnin:])
    den_1 =      np.median(  trace['den_1'][burnin:])
    den_2 =      np.median(  trace['den_2'][burnin:])
    den_3 =      np.median(  trace['den_3'][burnin:])

    print "<<<Detector num is [%0.4e, %0.4e, %0.4e]" % (num_1, num_2, num_3)
    print "<<<Detector den is [1, %0.4e, %0.4e, %0.4e]" % (den_1, den_2, den_3)

    plt.ioff()
    traceplot(trace)
    plt.savefig("hierarchical_full_%dchain.png" % len(wfs))
    plt.ion()
##
##    num = [num_1, num_2, num_3]
##    den = [1,   den_1, den_2, den_3]
##    system = signal.lti(num, den)
##
#
#    plt.figure(3)
#    plt.clf()
#    plt.title("Charge waveform")
#    plt.xlabel("Sample number [10s of ns]")
#    plt.ylabel("Raw ADC Value [Arb]")
#    
#    det.tfSystem = system
#    det.SetTemperature(temp)
#    det.SetFields(pcRad, grad)
#    
    for ( wf_idx, wf) in enumerate(wfs):
      if len(trace['switchpoint'][:,wf_idx]) < burnin:
        burnin = 0
      
#      t0 =  np.median(  trace['switchpoint'][burnin:,wf_idx])
#      r =             np.median(  trace['radEst'][burnin:,wf_idx])
#      z =             np.median(  trace['zEst'][burnin:,wf_idx])
#      phi =           np.median(  trace['phiEst'][burnin:,wf_idx])
#      scale =         np.median(  trace['wfScale'][burnin:,wf_idx])
#      sigma =         np.median(  trace['sigma'][burnin:,wf_idx])
#      
#      print "wf number %d" % wf_idx
#      print "  >> r:   %0.2f" % r
#      print "  >> z:   %0.2f" % z
#      print "  >> phi: %0.2f" % phi
#      print "  >> e:   %0.2f" % scale
#      print "  >> t0:  %0.2f" % t0
#
#      fit_wf = det.GetSimWaveform(r, phi, z, scale, t0, fitSamples)
#      plt.plot(wf.windowedWf, color="r")
#      plt.plot(fit_wf, color="b")
#
#      
#    plt.savefig("hierarchical_full_%dwaveforms.png" % len(wfs))
    summary(trace)
    value = raw_input('  --> Press q to quit, any other key to continue\n')


if __name__=="__main__":
    main(sys.argv[1:])


