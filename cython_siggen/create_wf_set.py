#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import numpy as np

import helpers
# from pysiggen import Detector
# from probability_model_waveform import *

import matplotlib.pyplot as plt
from matplotlib import gridspec

def main(argv):

  numBins = 6
  numWfsToSavePerBin = 4

  # channel = 690
  # wfFileName = "fep_event_set_runs11510-11539_channel690"
  # save_file_name = "P42662A_%d_spread.npz" % (numBins*numWfsToSavePerBin)
  channel = 626
  wfFileName = "fep_event_set_runs11510-11539_channel%d" % channel
  save_file_name = "P42574A_%d_spread.npz" % (numBins*numWfsToSavePerBin)
  baseline_value = 107

  if os.path.isfile(wfFileName + ".npz"):
    data = np.load(wfFileName+ ".npz")
    wfs = data['wfs']
    numWaveforms = wfs.size
  else:
    print "No saved waveforms available... exiting."
    exit(0)

  risetimes = []
  amax = []
  results = []
  wfs_saved = []
  energy = []

  fig = plt.figure(1)

  for (idx,wf) in enumerate(wfs):
    if wf.energy < 1800: continue
    if wf.baselineMean < baseline_value: continue

    wf.WindowWaveformTimepoint(fallPercentage=.99, rmsMult=2)
    if wf.wfMax < 6380: continue

    rt = findTimePointBeforeMax(wf.windowedWf, 0.995) - wf.t0Guess
    rt_90 = findTimePointBeforeMax(wf.windowedWf, 0.90) - wf.t0Guess
    rt_50 = findTimePointBeforeMax(wf.windowedWf, 0.50) - wf.t0Guess
    amax.append( rt - rt_50 )

    if rt - rt_50 > 20: continue
    if rt < 40: continue
#
#     if rt > 200: continue
# #
# #    if rt < 95 and rt > 45: continue
#
#     if rt < 40: continue
#
# #    if np.amax(wf.windowedWf) < 3820: continue
#
#    ae = np.amax(np.diff(wf.windowedWf)) / wf.energy
#     if ae <0.165: continue

    risetimes.append( rt)
    wfs_saved.append(wf)
    energy.append(wf.energy)

    if wf.energy > 1800:
      colorStr = "r"
    else:
      colorStr = "g"

    plt.plot(wf.windowedWf)

  wfs_saved = np.array(wfs_saved)
  risetimes = np.array(risetimes)

  print  "found %d OK looking wfs..." % len(wfs_saved)
  print "generating a set of %d waveforms" % (numWfsToSavePerBin*numBins)

  # plt.figure()
  # plt.hist(risetimes, bins=50)
  # plt.show()
  # exit()

  (n,b) = np.histogram(risetimes, bins=numBins)
  wfs_to_save = np.empty(numWfsToSavePerBin*numBins, dtype=np.object)
  for i in range(len(n)):
      if n[i] < numWfsToSavePerBin:
          print "not enough waveforms per bin to save!"
          exit(0)
      else:
          idxs = np.logical_and(np.less(risetimes, b[i+1]), np.greater(risetimes, b[i]) )
        #   print wfs_saved[ idxs ][:numWfsToSavePerBin]
          wfs_to_save[i*numWfsToSavePerBin:i*numWfsToSavePerBin+numWfsToSavePerBin] = wfs_saved[ idxs ][:numWfsToSavePerBin]

  plt.figure()
  for wf in wfs_to_save:
      plt.plot(wf.windowedWf)
  plt.show()

  np.savez(save_file_name, wfs = wfs_to_save )

  # exit()

def findTimePointBeforeMax(data, percent):

  #don't screw up the data, bro
  int_data = np.copy(data)
  max_idx = np.argmax(int_data)
  int_data /= int_data[max_idx]

  int_data = int_data[0:max_idx]

  return np.where(np.less(int_data, percent))[0][-1]



if __name__=="__main__":
    main(sys.argv[1:])
