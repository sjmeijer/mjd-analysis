#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy as np
from scipy import signal

from ROOT import *



detectorName = "P3KJR"

def main(argv):

  plt.ion()

  runRanges = [(13385, 13391), (13420, 13427), (13550, 13557)]
  #runRanges = [(11510,11510), (11511, 11530)]


  file_name = "ms_event_set_runs11510-11530_mcmcfit.npz"
  if os.path.isfile(file_name):
    data = np.load(file_name)
    all_wfs = data['wfs']
  else:
    print "no wf file"
    exit(0)

#  for (idx,wf) in enumerate(all_wfs):
#    print "wf %d" % idx
#    print "   ln prob %f" % (wf.lnprob / wf.wfLength)
##    print "   result: " + str(wf.median_fit)


  print "Total number of wfs: %d" % all_wfs.size

  energy_arr = np.zeros(all_wfs.size)
  like_arr = np.zeros(all_wfs.size)

  for (idx, wf) in enumerate(all_wfs):
    energy_arr[idx] = wf.energy
    like_arr[idx] = -1*wf.lnprob / wf.wfLength

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
  plt.hist(like_arr_dep[ np.where( like_arr_dep < 20) ], 80)
  plt.figure()
  plt.hist(like_arr_sep[ np.where( like_arr_sep < 20) ], 80)
  
  
  cut_like = 2.5
  dep_pass = energy_arr_dep[ np.where( like_arr_dep < cut_like) ]
  sep_pass = energy_arr_sep[ np.where( like_arr_sep < cut_like) ]


  #DEP hists
  binwidth = 0.5

  dep_bins = np.arange(1597-15, 1597+15+binwidth, binwidth)
  (dep_nocut,bins) = np.histogram(energy_arr_dep, bins=dep_bins)
  (dep_cut,bins) = np.histogram(dep_pass, bins=dep_bins)
  plt.figure()
  plt.plot(bins[:-1], dep_nocut, ls='steps-post', color="black")
  plt.plot(bins[:-1], dep_cut, ls='steps-post', color="blue")

  sep_bins = np.arange(2109-15, 2109+15+binwidth, binwidth)
  (sep_nocut,bins) = np.histogram(energy_arr_sep, bins=sep_bins)
  (sep_cut,bins) = np.histogram(sep_pass, bins=sep_bins)
  plt.figure()
  plt.plot(bins[:-1], sep_nocut, ls='steps-post', color="black")
  plt.plot(bins[:-1], sep_cut, ls='steps-post', color="blue")

  value = raw_input('  --> Press q to quit, any other key to continue\n')
  if value == 'q': exit(0)


if __name__=="__main__":
    main(sys.argv[1:])


