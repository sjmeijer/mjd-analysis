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
  
  file_names = ["ms_event_set_runs11510-11530_mcmcfit.npz", "ms_event_set_runs11530-11560_mcmcfit.npz", "ms_event_set_runs11560-11570_mcmcfit.npz"]
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
  
  for (idx, wf) in enumerate(all_wfs):
    energy_arr[idx] = wf.energy
    like_arr[idx] = -1*wf.lnprob / wf.wfLength

  like_arr[ np.where( np.isnan(like_arr) == 1) ] = np.inf

  dep_idxs =  np.where(np.logical_and(np.less(energy_arr, 1800), np.isfinite(like_arr)))[0]
  r_arr = np.empty(len(dep_idxs))
  z_arr = np.empty(len(dep_idxs))
  like_arr_dep = like_arr[dep_idxs]

  for (new_idx, all_wf_idx) in enumerate(dep_idxs):
    samples = all_wfs[all_wf_idx].samples
    r_hist, r_bins = np.histogram(samples[:,0], bins=np.linspace(0, 33.8, 339 ))
    z_hist, z_bins = np.histogram(samples[:,2], bins=np.linspace(0, 39.3, 394 ))
    
    r_arr[new_idx] = r_bins[np.argmax(r_hist)]
    z_arr[new_idx] = z_bins[np.argmax(z_hist)]

  best_dep_idxs = np.where( np.less(like_arr_dep, 2) )[0]
  ok_dep_idxs = np.where(np.logical_and( np.greater(like_arr_dep, 2),np.less(like_arr_dep, 3) ))[0]
  bad_dep_idxs = np.where(np.greater(like_arr_dep, 3) )[0]

  plt.figure()
  plt.scatter(r_arr[best_dep_idxs], z_arr[best_dep_idxs], color="g")
  plt.scatter(r_arr[ok_dep_idxs], z_arr[ok_dep_idxs], color="b")
  plt.scatter(r_arr[bad_dep_idxs], z_arr[bad_dep_idxs], color="r")

  plt.xlim(0, 34)
  plt.ylim(0,38)

  plt.show()


if __name__=="__main__":
    main(sys.argv[1:])


