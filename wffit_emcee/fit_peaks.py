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
from ROOT import *



detectorName = "P3KJR"


def main(argv):

  plt.ion()

#  runRanges = [(13385, 13392), (13420, 13429), (13551, 13557)]
  runRanges = [(11510,11510), (11511, 11530)]
  

  all_wfs = np.empty(0, dtype=np.object)
  all_results = np.empty(0, dtype=np.object)
  
  for (idx,runRange) in enumerate(runRanges):
    file_name = "multisite_event_set_runs%d-%d.npz" % (runRange[0], runRange[-1])
    if os.path.isfile(file_name):
      data = np.load(file_name)
      wfs = data['wfs']
      results = data['results']
    else:
      print "no wf file"
      exit(0)
    
    
    print "Number of wfs: %d" % wfs.size
    print "Number of results: %d" % results.size

    wf_old = all_wfs.size
    res_old = all_results.size
    
    all_wfs.resize(all_wfs.size + wfs.size)
    all_results.resize(all_results.size + results.size)
    
    all_wfs[wf_old:] = wfs[:]
    all_results[res_old:] = results[:]

  print "Total number of wfs: %d" % all_wfs.size
  print "Total umber of results: %d" % all_results.size
  
  energy_arr = np.zeros(all_wfs.size)
  like_arr = np.zeros(all_results.size)

  for (idx, wf) in enumerate(all_wfs):
    energy_arr[idx] = wf.energy
    like_arr[idx] = all_results[idx]['fun'] / wf.wfLength

  #real cheap energy "cut" to differentiate peaks
  dep_idxs = np.where(energy_arr < 1800)
  sep_idxs = np.where(energy_arr > 1800)

  print "DEP events: %d" % dep_idxs[0].size
  print "SEP events: %d" % sep_idxs[0].size

  like_arr[ np.where( np.isfinite(like_arr) == 0) ] = np.nan

  like_arr_dep = like_arr[dep_idxs]
  like_arr_sep = like_arr[sep_idxs]
  energy_arr_dep = energy_arr[dep_idxs]
  energy_arr_sep = energy_arr[sep_idxs]


  plt.figure()
  plt.hist(like_arr_dep[ np.where( like_arr_dep < 200) ], 80)
  plt.figure()
  plt.hist(like_arr_sep[ np.where( like_arr_sep < 200) ], 80)
  
  
  cut_like = 90
  dep_pass = energy_arr_dep[ np.where( like_arr_dep < cut_like) ]
  sep_pass = energy_arr_sep[ np.where( like_arr_sep < cut_like) ]


  #DEP hists
  binwidth = 0.5

  dep_bins = np.arange(1592-10, 1592+10+binwidth, binwidth)
  (dep_nocut,bins) = np.histogram(energy_arr_dep, bins=dep_bins)
  (dep_cut,bins) = np.histogram(dep_pass, bins=dep_bins)
  plt.figure()
  plt.plot(bins[:-1], dep_nocut, ls='steps-post', color="black")
  plt.plot(bins[:-1], dep_cut, ls='steps-post', color="blue")

  sep_bins = np.arange(2103-10, 2103+10+binwidth, binwidth)
  (sep_nocut,bins) = np.histogram(energy_arr_sep, bins=sep_bins)
  (sep_cut,bins) = np.histogram(sep_pass, bins=sep_bins)
  plt.figure()
  plt.plot(bins[:-1], sep_nocut, ls='steps-post', color="black")
  plt.plot(bins[:-1], sep_cut, ls='steps-post', color="blue")

  value = raw_input('  --> Press q to quit, any other key to continue\n')
  if value == 'q': exit(0)


if __name__=="__main__":
    main(sys.argv[1:])


