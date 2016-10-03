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

  plt.ion()
  
  side_padding = 15
  sep_energy = 2109
  dep_energy = 1597
  binwidth = 0.5
  
  file_names = ["ms_event_set_runs11510-11530.npz", "ms_event_set_runs11530-11560.npz", "ms_event_set_runs11560-11570.npz"]
  all_wfs = []
  for file_name in file_names:
    if os.path.isfile(file_name):
      data = np.load(file_name)
      all_wfs.append(  data['wfs'][:])
    else:
      print "no wf file named %s" % file_name
      exit(0)

  all_wfs = np.concatenate(all_wfs[:])



  print "Total number of wfs: %d" % all_wfs.size

  energy_arr = np.zeros(all_wfs.size)
  like_arr = np.zeros(all_wfs.size)
  ae_arr = np.zeros(all_wfs.size)

  for (idx, wf) in enumerate(all_wfs):
    energy_arr[idx] = wf.energy
#    like_arr[idx] = -1*wf.lnprob / wf.wfLength
#    ae_arr[idx] = wf.ae

  #real cheap energy "cut" to differentiate peaks
  dep_idxs = np.where(energy_arr < 1800)
  sep_idxs = np.where(energy_arr > 1800)

  print "DEP events: %d" % dep_idxs[0].size
  print "SEP events: %d" % sep_idxs[0].size

#  like_arr[ np.where( np.isfinite(like_arr) == 0) ] = np.nan
#
#  like_arr_dep = like_arr[dep_idxs]
#  like_arr_sep = like_arr[sep_idxs]

#  ae_arr_dep = ae_arr[dep_idxs]
#  ae_arr_dep = ae_arr[sep_idxs]
  energy_arr_dep = energy_arr[dep_idxs]
  energy_arr_sep = energy_arr[sep_idxs]

#
#  sep_bins = np.arange(sep_energy-side_padding, sep_energy+side_padding+binwidth, binwidth)
#  (sep_nocut,bins) = np.histogram(energy_arr_sep, bins=sep_bins)
#  bins_centered = sep_bins[:-1] + binwidth/2
#  
#  plt.figure()
#  plt.errorbar(bins_centered, sep_nocut, yerr=np.sqrt(sep_nocut), color="black", fmt='o')
#
#  p0 = [sep_energy, 1.5,  500, 0.002, 0.03, 1.5, -288, 0.14, ]
#
#  c, pcov = curve_fit(radford_peak, bins_centered, sep_nocut,
#                p0 = p0,
#                sigma = np.sqrt(sep_nocut),
#                bounds = ( [sep_energy-side_padding, 0,      0,       0,     0, 0,      -np.inf, -np.inf],
#                           [sep_energy+side_padding, np.inf, np.inf, np.inf, 0.5, np.inf, np.inf,  np.inf])
#                )
#
#  x0, sigma, a, hstep, htail, tau, bg0, bg1 = c
#  print "Fit Coefficients:"
#  print x0, sigma, a, hstep, htail, tau, bg0, bg1
##  a_nocut = c[2]
#
#  energies = np.linspace(sep_energy - side_padding, sep_energy + side_padding, 1000)
#  plt.plot(energies, radford_peak(energies, *c), color="red")
#
#  plt.plot(energies, gaussian(energies, x0, sigma, a, htail) + bg(energies, bg0, bg1), color="blue")
#  plt.plot(energies, low_energy_tail(energies, x0, sigma, a, htail, tau) + bg(energies, bg0, bg1), color="green")
#  plt.plot(energies, step(energies, x0, sigma, a, hstep)+ bg(energies, bg0, bg1), color="purple")
#  
#  plt.xlim(sep_energy - side_padding, sep_energy + side_padding)
#
#  value = raw_input('  --> Press q to quit, any other key to continue\n')
#  if value == 'q': exit(0)
#
########
#  exit()
#  #####


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


def radford_peak(x, x0, sigma, a, hstep, htail, tau, bg0, bg1):
    bg_term = bg0 + x*bg1
    
    step = a * hstep * erfc( (x - x0)/(sigma * np.sqrt(2)) )
    
    le_tail = a * htail * erfc( (x - x0)/(sigma * np.sqrt(2))  + sigma/(tau*np.sqrt(2)) ) * np.exp( (x-x0)/tau ) / (2 * tau * np.exp( -(sigma / (np.sqrt(2)*tau))**2 ))
    
    gauss_term = a* (1-htail) * (1./np.sqrt(2*np.pi)*sigma)* np.exp(-(x-x0)**2/(2*sigma**2))

    return  gauss_term+bg_term + step + le_tail

def gaussian(x, x0, sigma, a, htail):
  return a* (1-htail) * (1./np.sqrt(2*np.pi)*sigma)* np.exp(-(x-x0)**2/(2*sigma**2))

def low_energy_tail(x, x0, sigma, a, htail, tau):
  return a * htail * erfc( (x - x0)/(sigma * np.sqrt(2))  + sigma/(tau*np.sqrt(2)) ) * np.exp( (x-x0)/tau ) / (2 * tau * np.exp( -(sigma / (np.sqrt(2)*tau))**2 ))

def step(x, x0, sigma, a, hstep):
  return a * hstep * erfc( (x - x0)/(sigma * np.sqrt(2)) )

def bg(x, bg0, bg1):
  return bg0 + x*bg1

if __name__=="__main__":
    main(sys.argv[1:])


