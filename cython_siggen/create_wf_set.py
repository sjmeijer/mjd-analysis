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

  numWfsToSave = 24 #must be a mult of 4

  channel = 626
  aeCutVal = 0.01425

  fitSamples = 250
  timeStepSize = 1

  wfFileName = "fep_event_set_runs11531-11539"
#  wfFileName = "P42574A_512waveforms_30risetimeculled"
  if os.path.isfile(wfFileName + ".npz"):
    data = np.load(wfFileName+ ".npz")
    wfs = data['wfs']
    numWaveforms = wfs.size
  else:
    print "No saved waveforms available.  Loading from Data"
    exit(0)

  #Prepare detector
  tempGuess = 79.310080
  gradGuess = 0.05
  pcRadGuess = 2.5
  pcLenGuess = 1.6

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10 )
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)

  b_over_a = 0.107213
  c = -0.821158
  d = 0.828957
  rc1 = 74.4
  rc2 = 1.79
  rcfrac = 0.992
  det.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)

  initializeDetector(det, )

  risetimes = []
  amax = []
  results = []
  wfs_saved = []
  energy = []

  fig = plt.figure(1)

  for (idx,wf) in enumerate(wfs):
    if wf.energy < 1800: continue


    wf.WindowWaveformTimepoint(fallPercentage=.99, rmsMult=2)
    if wf.wfMax < 6380: continue


    rt = findTimePointBeforeMax(wf.windowedWf, 0.995) - wf.t0Guess

    rt_90 = findTimePointBeforeMax(wf.windowedWf, 0.90) - wf.t0Guess

    if rt - rt_90 > 20: continue

    if rt > 200: continue
#
#    if rt < 95 and rt > 45: continue

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
  print  "found %d fast wfs" % len(np.where(risetimes < 45)[0])
  print  "found %d slow wfs" % len(np.where(risetimes > 92)[0])

#  plt.figure()
#  plt.hist(amax)
#  plt.show()
#  exit()

#  np.savez("fep_old_postpsa.npz", wfs = wfs_saved )
#  exit(0)

  wfs_fast_hi =  wfs_saved[np.logical_and(np.less(risetimes, 45), np.greater(energy, 1800) ) ][:4]
  wfs_mid_hi =  wfs_saved[np.logical_and(np.greater(risetimes, 50), np.less(risetimes, 60))][:4]
  wfs_mid_hi2 =  wfs_saved[np.logical_and(np.greater(risetimes, 60), np.less(risetimes, 70))][:4]
  wfs_mid_hi3 =  wfs_saved[np.logical_and(np.greater(risetimes, 70), np.less(risetimes, 80))][:4]
  wfs_mid_hi4 =  wfs_saved[np.logical_and(np.greater(risetimes, 80), np.less(risetimes, 90))][:4]
  wfs_slow_hi =  wfs_saved[np.logical_and(np.greater(risetimes, 90), np.greater(energy, 1800))][:4]
#  wfs_fast_lo =  wfs_saved[np.logical_and(np.less(risetimes, 50), np.less(energy, 1800) ) ][:numWfsToSave/4]
#  wfs_slow_lo =  wfs_saved[np.logical_and(np.greater(risetimes, 50), np.less(energy, 1800))][:numWfsToSave/4]

#  wfs_to_save = wfs_fast_hi


  wfs_to_save = np.concatenate((wfs_fast_hi, wfs_slow_hi,  wfs_mid_hi, wfs_mid_hi2, wfs_mid_hi3, wfs_mid_hi4))#, wfs_fast_lo, wfs_slow_lo))
  results = np.empty_like(wfs_to_save)

  plt.figure()
  for wf in wfs_to_save:
      plt.plot(wf.windowedWf)
  plt.show()
  exit()

  plt.ion()
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
    minresult = None
    minlike = np.inf

    for r in np.linspace(4, np.floor(det.detector_radius)-3, 4):
      for z in np.linspace(4, np.floor(det.detector_length)-3, 4):
#        for t0 in np.linspace(wf.t0Guess-5, wf.t0Guess+5, 3):
          if not det.IsInDetector(r,0,z): continue
          startGuess = [r, np.pi/8, z, wf.wfMax, wf.t0Guess-5, 10]
          result = op.minimize(nll, startGuess,   method="Nelder-Mead")

          if result['fun'] < minlike:
            minlike = result['fun']
            minresult = result

    r, phi, z, scale, t0,smooth  = minresult["x"]


    results[idx] = minresult

#    r, phi, z, scale, t0,smooth = best_result["x"]
#    r_det = np.cos(theta) * r
#    z_det = np.sin(theta) * r
    ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples,  h_smoothing=smooth)

    print "wf number %d" % idx
    print "  energy: %f" % wf.energy
#    print "  smoove: %f" % smoove
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

#

  wfFileName = "P42574A_24_spread.npz"

  np.savez(wfFileName, wfs = wfs_to_save, results=results  )

  value = raw_input('  --> Press q to quit, any other key to continue\n')
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
