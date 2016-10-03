#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy as np
from scipy import signal


from scipy.optimize import curve_fit
from scipy.special import erfc

detectorName = "P3KJR"

def main(argv):

#  plt.ion()

  side_padding = 15
  sep_energy = 2109
  dep_energy = 1597
  binwidth = 0.5
  
  file_names = ["ms_event_set_runs11510-11530_mcmcfit.npz"]
  all_wfs = []
  for file_name in file_names:
    if os.path.isfile(file_name):
      data = np.load(file_name)
      all_wfs.append(  data['wfs'][:])
    else:
      print "no wf file named %s" % file_name
      exit(0)

  all_wfs = np.concatenate(all_wfs[:])
  energy_arr = np.zeros(all_wfs.size)
  like_arr = np.zeros(all_wfs.size)

  import warnings
  with warnings.catch_warnings():
    warnings.filterwarnings('error')
    for (idx, wf) in enumerate(all_wfs):
      energy_arr[idx] = wf.energy
      like_arr[idx] = -1*wf.lnprob / wf.wfLength

  like_arr[ np.where( np.isnan(like_arr) == 1) ] = np.inf

  good_dep_idxs = np.where(np.logical_and(np.less(energy_arr, 1800), np.less(like_arr, 2)))[0]

  r_arr = np.empty(len(good_dep_idxs))
  z_arr = np.empty(len(good_dep_idxs))
  
#  print len(r_arr)

  for (new_idx, all_wf_idx) in enumerate(good_dep_idxs):
    samples = all_wfs[all_wf_idx].samples
    r_arr[new_idx] = np.median(samples[:,0])
    z_arr[new_idx] = np.median(samples[:,2])

  plt.figure()
  plt.scatter(r_arr, z_arr)

  plt.xlim(0, 34)
  plt.ylim(0,38)

  plt.show()


if __name__=="__main__":
    main(sys.argv[1:])


