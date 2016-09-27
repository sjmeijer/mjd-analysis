#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os, time
import scipy.optimize as op
import numpy as np
import emcee
from scipy import signal

import helpers
from detector_model import *
from probability_model_waveform import *

from progressbar import ProgressBar, Percentage, Bar, ETA
from multiprocessing import Pool
from timeit import default_timer as timer

import matplotlib.pyplot as plt
from matplotlib import gridspec

def main(argv):

  numWfsToSave = 16 #must be a mult of 4

  runRange = (13420,13429)
  channel = 626
  aeCutVal = 0.01425
  
  fitSamples = 250
  timeStepSize = 10
  
  wfFileName = "fep_and_dep_event_set_runs13385-13557"
#  wfFileName = "P42574A_512waveforms_30risetimeculled"
  if os.path.isfile(wfFileName + ".npz"):
    data = np.load(wfFileName+ ".npz")
    wfs = data['wfs']
    numWaveforms = wfs.size
  else:
    print "No saved waveforms available.  Loading from Data"
    exit(0)
  
  zero_1 = 0.470677
  pole_1 = 0.999857
  pole_real = 0.807248
  pole_imag = 0.085347
  tempGuess = 78.474793
  gradGuess = 0.045049
  pcRadGuess = 2.574859
  pcLenGuess = 1.524812
  
  zeros = [zero_1, -1., 1. ]
  poles = [pole_1, pole_real+pole_imag*1j, pole_real-pole_imag*1j, ]

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10./timeStepSize, poles=poles, zeros=zeros)
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  initializeDetector(det, )

  risetimes = []
  amax = []
  results = []
  wfs_saved = []
  energy = []

  fig = plt.figure(1)
  
  for (idx,wf) in enumerate(wfs):
#    if wf.energy > 1800: continue

    wf.WindowWaveformTimepoint(fallPercentage=.99, rmsMult=10)
    rt = findTimePointBeforeMax(wf.windowedWf, 0.995) - wf.t0Guess
    
    rt_90 = findTimePointBeforeMax(wf.windowedWf, 0.90) - wf.t0Guess
    
    if rt - rt_90 > 20: continue
    
    if rt > 200: continue
#    
    if rt < 86 and rt > 45: continue
    
    if rt < 40: continue

#    if np.amax(wf.windowedWf) < 3820: continue

    ae = np.amax(np.diff(wf.windowedWf)) / wf.energy
    
    if ae <0.165: continue

    risetimes.append( rt)
    amax.append( ae )
    wfs_saved.append(wf)
    energy.append(wf.energy)
  
    if wf.energy > 1800:
      colorStr = "r"
    else:
      colorStr = "g"
  
    plt.plot(wf.windowedWf)
  
  wfs_saved = np.array(wfs_saved)
  risetimes = np.array(risetimes)
  print  "found %d fast wfs" % len(np.where(risetimes < 60)[0])
  print  "found %d slow wfs" % len(np.where(risetimes > 60)[0])
  
  
  
#  plt.figure()
#  plt.hist(amax)
#  plt.show()
#  exit()

  wfs_fast_hi =  wfs_saved[np.logical_and(np.less(risetimes, 50), np.greater(energy, 1800) ) ][:numWfsToSave/4]
  wfs_slow_hi =  wfs_saved[np.logical_and(np.greater(risetimes, 50), np.greater(energy, 1800))][:numWfsToSave/4]
  wfs_fast_lo =  wfs_saved[np.logical_and(np.less(risetimes, 50), np.less(energy, 1800) ) ][:numWfsToSave/4]
  wfs_slow_lo =  wfs_saved[np.logical_and(np.greater(risetimes, 50), np.less(energy, 1800))][:numWfsToSave/4]

  wfs_to_save = np.concatenate((wfs_fast_hi, wfs_slow_hi, wfs_fast_lo, wfs_slow_lo))
  results = np.empty_like(wfs_to_save)

  fig1 = plt.figure(2)
  plt.clf()
  gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2])
  ax0 = plt.subplot(gs[0])
  ax1 = plt.subplot(gs[1], sharex=ax0)
  ax1.set_xlabel("Digitizer Time [ns]")
  ax0.set_ylabel("Voltage [Arb.]")
  ax1.set_ylabel("Residual")

  for (idx,wf) in enumerate(wfs_to_save):
    initializeWaveform(wf)
    startGuess = [15., np.pi/8, 15., wf.wfMax, wf.t0Guess, 1]
    result = op.minimize(nll, startGuess,   method="Powell")

    r, phi, z, scale, t0, smoove = result["x"]
    r_new = np.amin( [z, np.floor(det.detector_radius)] )
    z_new = np.amin( [r, np.floor(det.detector_length)] )
    
    result2 = op.minimize(nll, [r_new, phi, z_new, scale, wf.t0Guess, 1],  method="Powell")
    r, phi, z, scale, t0, smoove = result2["x"]

    if result2['fun'] < result['fun']:
      best_result = result2
    else:
      best_result = result

    results[idx] = best_result
      
    r, phi, z, scale, t0, smoove= best_result["x"]
    ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples,  h_smoothing=smoove)
    
    print "wf number %d" % idx
    print "  energy: %f" % wf.energy
    print "  smoove: %f" % smoove
    print "  pos:    (%0.2f, %0.2f, %0.2f)" % (r, phi, z)
    
    
    if wf.energy > 1800:
      colorStr = "r"
    else:
      colorStr = "g"
#
    dataLen = wf.wfLength
    ax0.plot(ml_wf[:dataLen], color="black")
    ax0.plot( wf.windowedWf, color=colorStr)

    ax1.plot(ml_wf[:dataLen] -  wf.windowedWf, color=colorStr)




  wfFileName = "P42574A_%d_fastandslow.npz" % numWfsToSave

  np.savez(wfFileName, wfs = wfs_to_save, results=results  )

  plt.show()


def nll(*args):
  return -lnlike_waveform(*args)


def findTimePointBeforeMax(data, percent):

  #don't screw up the data, bro
  int_data = np.copy(data)
  max_idx = np.argmax(int_data)
  int_data /= int_data[max_idx]
  
  int_data = int_data[0:max_idx]

  return np.where(np.less(int_data, percent))[0][-1]



if __name__=="__main__":
    main(sys.argv[1:])


