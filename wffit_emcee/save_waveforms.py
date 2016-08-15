#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
import numpy as np

import helpers

from detector_model import *
from probability_model_hier import *

from timeit import default_timer as timer
from multiprocessing import Pool

def main(argv):
  numWaveforms =  512
  numThreads = 8
  figsize = (20,10)
  plt.ion()

  channel = 626
  aeCutVal = 0.01425
  runRanges = [(13385, 13392),  (13420,13429), (13551,13557)]
  
  r_mult = 1.
  z_mult = 1.
  scale_mult = 100.
  
  #Prepare detectoren = [1, 40503831.367655091, 507743886451386.06, 7.0164915381862738e+18]
  num =
  den =
  
  system = signal.lti(num, den)
  fitSamples=200
  
  tempGuess = 78
  gradGuess = 0.0487
  pcRadGuess = 2.495226
  pcLenGuess = 1.632869

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.04,2.5, 1.6)
  det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
  det.LoadFields("P42574A_fields_len.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  initializeDetector(det)
  
  p = Pool(numThreads, initializer=initializeDetector, initargs=[det])
  

  
  #Crate a decent start guess by fitting waveform-by-waveform
  
  wfFileName = "P42574A_%dwaveforms_raw.npz" % numWaveforms
  wfFileNameProcessed = "P42574A_%dwaveforms_fit_nosmooth_de.npz" % numWaveforms
  
  if os.path.isfile(wfFileName):
    print "Raw File already exists %s" % wfFileName
    
    data = np.load(wfFileName)
    wfs = data['wfs']
  
    print "We actually have %d waveforms here..." % len(wfs)
    
  else:
    print "No saved waveforms available.  Loading from Data"
    #get waveforms
    cut = "trapECal>%f && trapECal<%f && TSCurrent100nsMax/trapECal > %f" %  (1588,1594, aeCutVal)
    wfs = helpers.GetWaveforms(runRanges, channel, numWaveforms, cut)
    
  args = []
  fig = plt.figure(figsize=figsize)
  for (idx, wf) in enumerate(wfs):
    wf.WindowWaveformTimepoint(fallPercentage=.99)
    args.append( [15./r_mult, np.pi/8., 15./z_mult, wf.wfMax/scale_mult, wf.t0Guess,  wfs[idx] ]  )

#    args.append( [15./r_mult, np.pi/8., 15./z_mult, wf.wfMax/scale_mult, wf.t0Guess, 10., 5.,  wfs[idx] ]  )
    plt.plot(wf.windowedWf, color="r")

  np.savez(wfFileName, wfs = wfs )
  value = raw_input('  --> Press q to quit, any other key to continue\n')
  if value == 'q': exit(0)


  print "performing parallelized initial fit..."
  start = timer()
  results = p.map(minimize_waveform_only_nosmooth_star, args)
  end = timer()
  print "Initial fit time: %f" % (end-start)

  simWfArr = np.empty((1,len(results), fitSamples))
  for (idx,result) in enumerate(results):
#    r, phi, z, scale, t0, smooth, esmooth = result["x"]

    r, phi, z, scale, t0,  = result["x"]
    print "  >> wf %d (normalized likelihood %0.2f):" % (idx, result["fun"]/wfs[idx].wfLength)
    print "      r: %0.2f, phi: %0.3f, z: %0.2f, e: %0.2f, t0: %0.2f" % (r, phi, z, scale, t0,  )
    simWfArr[0,idx,:] = det.GetSimWaveform(r*r_mult, phi, z*z_mult, scale*scale_mult, t0,  fitSamples, )

#    print "      r: %0.2f, phi: %0.3f, z: %0.2f, e: %0.2f, t0: %0.2f, smooth:%0.2f, esmooth:%0.2f" % (r, phi, z, scale, t0, smooth, esmooth)
#    simWfArr[0,idx,:] = det.GetSimWaveform(r*r_mult, phi, z*z_mult, scale*scale_mult, t0,  fitSamples, smoothing=smooth, electron_smoothing=esmooth)


  fig1 = plt.figure(figsize=figsize)
  helpers.plotManyResidual(simWfArr, wfs, fig1, residAlpha=1)
  np.savez(wfFileNameProcessed, wfs = wfs, result_arr=results  )
  value = raw_input('  --> Press q to quit, any other key to continue\n')
  if value == 'q': exit(0)




if __name__=="__main__":
    main(sys.argv[1:])


