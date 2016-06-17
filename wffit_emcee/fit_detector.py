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
from probability_model import *

from pymc3 import *
from pymc3.backends import SQLite
import signal_model_hierarchical as sm3

'''
I wonder if this leaks less memory as pymc2
'''

this_sample = 10000 #number of samples for the MCMC
numWaveforms = 30

def main(argv):

  plt.ion()

  runRange = (13420,13429)
  channel = 626
  aeCutVal = 0.01425
  
  fitSamples = 250 #has to be longer than the longest wf you're gonna fit
  fallPercentage = 0.985 #need to go out a little long to get the TF down
  tempGuess = 81.5
  
  
  #Set up detectors
  num = [3.64e+09, 1.88e+17, 6.05e+15]
  den = [1, 4.03e+07, 5.14e+14, 7.15e+18]
  system = signal.lti(num, den)

  gradList = np.arange(0.01, 0.09, 0.01)
  pcRadList = np.arange(1.65, 2.95, 0.1)
  detArray  = np.empty( (len(gradList),len(pcRadList)), dtype=object)

  for (gradIdx,grad) in enumerate(gradList):
    for (radIdx, pcRad) in enumerate(pcRadList):
      detName = "conf/P42574A_grad%0.2f_pcrad%0.2f.conf" % (grad,pcRad)
      det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
      detArray[gradIdx, radIdx] = det

  #choose a "default" WP for the MLE
  det = detArray[len(gradList)-2, len(pcRadList)-3]
  

  #Perform MLE
  nll = lambda *args: -lnlike(*args)
  plt.figure(1)
  
  rStart = np.empty(numWaveforms)
  zStart = np.empty(numWaveforms)
  phiStart = np.empty(numWaveforms)
  t0Start = np.empty(numWaveforms)
  scaleStart = np.empty(numWaveforms)

  if not os.path.isfile("%dwaveforms_saved.npz" % numWaveforms):
    #get waveforms
    cut = "trapECal>%f && trapECal<%f && TSCurrent100nsMax/trapECal > %f" %  (1588,1594, aeCutVal)
    wfs = helpers.GetWaveforms(runRange, channel, numWaveforms, cut)

    for (idx,wf) in enumerate(wfs):
      print "Performing MLE on waveform %d" % idx
      wf.WindowWaveformTimepoint(fallPercentage=fallPercentage)
      
      if wf.wfLength > fitSamples:
        print "skipping wf %d (length is %d)" % (idx, wf.wfLength)
        continue
      
      startGuess = [15., np.pi/8, 15., wf.wfMax, wf.t0Guess]
      
      result = op.minimize(nll, startGuess, args=(wf.windowedWf, det,  wf.baselineRMS),  method="Powell")
      r, phi, z, scale, t0= result["x"]
      
      rStart[idx] = r
      phiStart[idx] = phi
      zStart[idx] = z
      scaleStart[idx] = scale
      t0Start[idx] = t0
      
      ml_wf = det.GetSimWaveform(r, phi, z, scale, t0, fitSamples)
      ml_wf_inv = det.GetSimWaveform(z, phi, r, scale, t0, fitSamples)
      
      plt.plot(wf.windowedWf, color="r")
      plt.plot(ml_wf, color="b")
      plt.plot(ml_wf_inv, "b:")
    np.savez("%dwaveforms_saved.npz" % numWaveforms, wfs, rStart, phiStart, zStart, scaleStart, t0Start)
  else:
    npzfile = np.load("%dwaveforms_saved.npz" % numWaveforms)
    wfs = npzfile['arr_0']
    rStart = npzfile['arr_1']
    phiStart = npzfile['arr_2']
    zStart = npzfile['arr_3']
    scaleStart = npzfile['arr_4']
    t0Start = npzfile['arr_5']
#  value = raw_input('  --> Press s to skip,  q to quit, any other key to continue with fit\n')
#  if value == 'q':
#    exit(0)


  startGuess = {'radEst': rStart,
                'zEst': zStart,
                'phiEst': phiStart,
                'wfScale': scaleStart,
                'switchpoint': t0Start,
                'temp': tempGuess}

  siggen_model = sm3.CreateFullDetectorModel(detArray, wfs,tempGuess, num, den )
  with siggen_model:
  
    
    step = Metropolis()
    
    backend = SQLite('trace.sqlite')
    trace = sample(this_sample,  step = step, start=startGuess, trace=backend)
    
    summary(trace)
    
#    burnin = len(trace['temp'][:]) - 100#np.int(.75 * this_sample)
#  
#    temp =  np.median(  trace['temp'][burnin:])
#    print "<<<detector temperature is %f" % temp
#    gradIdx =       np.int(np.median(  trace['gradIdx'][burnin:]))
#    pcRadIdx =      np.int(np.median(  trace['pcRadIdx'][burnin:]))
#    print "<<<detector gradient is %0.2f (idx is %d)" % (gradList[gradIdx], gradIdx)
#    print "<<<PC Radius is %0.2f (idx is %d)" % (pcRadList[pcRadIdx], pcRadIdx)
#    
#    num_1 =      np.median(  trace['num_1'][burnin:])
#    num_2 =      np.median(  trace['num_2'][burnin:])
#    num_3 =      np.median(  trace['num_3'][burnin:])
#    den_1 =      np.median(  trace['den_1'][burnin:])
#    den_2 =      np.median(  trace['den_2'][burnin:])
#    den_3 =      np.median(  trace['den_3'][burnin:])
#
#    print "<<<Detector num is [%0.4e, %0.4e, %0.4e]" % (num_1, num_2, num_3)
#    print "<<<Detector den is [1, %0.4e, %0.4e, %0.4e]" % (den_1, den_2, den_3)
#
#    plt.ioff()
#    traceplot(trace)
#    plt.savefig("hierarchical_full_%dchain.png" % len(wfs))
#    plt.ion()
#    
#    num = [num_1, num_2, num_3]
#    den = [1,   den_1, den_2, den_3]
#    system = signal.lti(num, den)
#
#
#    plt.figure(3)
#    plt.clf()
#    plt.title("Charge waveform")
#    plt.xlabel("Sample number [10s of ns]")
#    plt.ylabel("Raw ADC Value [Arb]")
#    
#    det = detArray[gradIdx, pcRadIdx]
#    det.tfSystem = system
#    det.SetTemperature(temp)
#    
#    for ( wf_idx, wf) in enumerate(wfs):
#      if len(trace['switchpoint'][:,wf_idx]) < burnin:
#        burnin = 0
#      
#      t0 = np.around( np.median(  trace['switchpoint'][burnin:,wf_idx]), 1)
#      r =             np.median(  trace['radEst'][burnin:,wf_idx])
#      z =             np.median(  trace['zEst'][burnin:,wf_idx])
#      phi =           np.median(  trace['phiEst'][burnin:,wf_idx])
#      scale =         np.median(  trace['wfScale'][burnin:,wf_idx])
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

    value = raw_input('  --> Press q to quit, any other key to continue\n')


if __name__=="__main__":
    main(sys.argv[1:])


