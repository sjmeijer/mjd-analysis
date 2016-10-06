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

side_padding = 15
sep_energy = 2109
dep_energy = 1597
binwidth = 1

def main(argv):

  plt.ion()
  
  file_names = ["ms_event_set_runs11510-11520_mcmcfit.npz"]
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
    like_arr[idx] = -1*wf.lnprob / wf.wfLength
#    ae_arr[idx] = wf.ae
#    print wf.ae

  #real cheap energy "cut" to differentiate peaks
  dep_idxs = np.where(energy_arr < 1800)
  sep_idxs = np.where(energy_arr > 1800)

  print "DEP events: %d" % dep_idxs[0].size
  print "SEP events: %d" % sep_idxs[0].size

  like_arr[ np.where( np.isfinite(like_arr) == 0) ] = 10000
  like_arr[ np.where( np.isnan(like_arr)) ] = 10000
  like_arr_dep = like_arr[dep_idxs]
  like_arr_sep = like_arr[sep_idxs]

#  ae_arr_dep = ae_arr[dep_idxs]
#  ae_arr_sep = ae_arr[sep_idxs]
  energy_arr_dep = energy_arr[dep_idxs]
  energy_arr_sep = energy_arr[sep_idxs]

  hist_max = 10
  bins = np.linspace(0, hist_max, 25)
  (hist_dep,___) = np.histogram(like_arr_dep, bins=bins)
  (hist_sep,___) = np.histogram(like_arr_sep, bins=bins)

  plt.figure()
  plt.plot(bins[:-1], hist_dep, ls="steps-post", color="blue", label="DEP")
  plt.plot(bins[:-1], hist_sep, ls="steps-post", color="red", label="SEP")
  plt.xlabel( "normalized ln posterior" )
  plt.legend(loc=1)
  plt.savefig("mse_lnprob_hist.png")

  value = raw_input('  --> Press q to quit, any other key to continue\n')
  if value == 'q': exit(0)

  cut_like = 100
#  fit_peak(sep_energy, energy_arr_sep, like_arr_sep, cut_like,)# ae_arr_sep)
  fit_peak(dep_energy, energy_arr_dep, like_arr_dep, cut_like,)# ae_arr_dep)

  value = raw_input('  --> Press q to quit, any other key to continue\n')
  if value == 'q': exit(0)

  

  dep_pass = energy_arr_dep[ np.where( like_arr_dep < cut_like) ]
  sep_pass = energy_arr_sep[ np.where( like_arr_sep < cut_like) ]


  #DEP hists

  dep_bins = np.arange(1597-15, 1597+15+binwidth, binwidth)
  (dep_nocut,bins) = np.histogram(energy_arr_dep, bins=dep_bins)
  (dep_cut,bins) = np.histogram(dep_pass, bins=dep_bins)
  plt.figure()
  plt.plot(bins[:-1], dep_nocut, ls='steps-post', color="black")
  plt.plot(bins[:-1], dep_cut, ls='steps-post', color="blue")

  plt.savefig("mse_dep_peak_%0.2fcutoff.png" % cut_like)

  sep_bins = np.arange(2109-15, 2109+15+binwidth, binwidth)
  (sep_nocut,bins) = np.histogram(energy_arr_sep, bins=sep_bins)
  (sep_cut,bins) = np.histogram(sep_pass, bins=sep_bins)
  plt.figure()
  plt.plot(bins[:-1], sep_nocut, ls='steps-post', color="black")
  plt.plot(bins[:-1], sep_cut, ls='steps-post', color="red")

  plt.savefig("mse_sep_peak_%0.2fcutoff.png" % cut_like)

  value = raw_input('  --> Press q to quit, any other key to continue\n')
  if value == 'q': exit(0)


def fit_peak(energy, energy_array, like_array, cut_like, ae_arr=None):
  bins = np.arange(energy-side_padding, energy+side_padding+binwidth, binwidth)
  (hist_nocut,___) = np.histogram(energy_array, bins=bins)
  bins_centered = bins[:-1] + binwidth/2
  
  plt.figure()
  energies = np.linspace(energy - side_padding, energy + side_padding, 1000)
  plt.errorbar(bins_centered, hist_nocut, yerr=np.sqrt(hist_nocut), color="black", fmt='o')

  p0 = [energy, 1.5,  500, 20,  ]
  
#  plt.plot(energies, gaussian_with_background(energies, *p0), color="black", label="Before Cut")
#  value = raw_input('  --> Press q to quit, any other key to continue\n')
#  if value == 'q': exit(0)

  c, pcov = curve_fit(gaussian_with_background, bins_centered, hist_nocut,
                p0 = p0,
                sigma = np.sqrt(hist_nocut),
                bounds = ( [energy-side_padding, 0,      0,        0, ],
                           [energy+side_padding, np.inf, np.inf,   np.inf,  ])
                )

  x0, sigma, a_nocut, bg0,  = c
  print "Fit Coefficients:"
  print x0, sigma, a_nocut, bg0,
#  a_nocut = c[2]


  plt.plot(energies, gaussian_with_background(energies, *c), color="black", label="Before Cut")
  
  
  energies_pass = energy_array[ np.where( like_array < cut_like) ]
  (hist_cut,___) = np.histogram(energies_pass, bins=bins)
  c, pcov = curve_fit(gaussian_with_background, bins_centered, hist_cut,
                p0 = c,
                sigma = np.sqrt(hist_cut),
                bounds = ( [c[0]-1E-6, c[1]-1E-6,      0,   0,],
                           [c[0]+1E-6, c[1]+1E-6, np.inf,   np.inf,  ])
                )
  
  a_cut = c[2]
  
  print "Total reduction: %0.2f%%" % (a_cut/a_nocut*100)
  
  plt.errorbar(bins_centered, hist_cut, yerr=np.sqrt(hist_cut), color="red", fmt='o')
  plt.plot(energies, gaussian_with_background(energies, *c), color="red", label="After Cut (%0.2f%%)" % (a_cut/a_nocut*100))
  
  if ae_arr is not None:
    energies_aepass = energy_array[ np.where( ae_arr > 2.4) ]
    (hist_ae,___) = np.histogram(energies_aepass, bins=bins)
    plt.errorbar(bins_centered, hist_ae, yerr=np.sqrt(hist_ae), color="blue", fmt='o')
    

#  plt.plot(energies, gaussian(energies, x0, sigma, a, htail) + bg(energies, bg0, bg1), color="blue")
#  plt.plot(energies, low_energy_tail(energies, x0, sigma, a, htail, tau) + bg(energies, bg0, bg1), color="green")
#  plt.plot(energies, step(energies, x0, sigma, a, hstep)+ bg(energies, bg0, bg1), color="purple")

  plt.xlim(energy - side_padding, energy + side_padding)
  plt.legend(loc=2)



def gaussian_with_background(x, x0, sigma, a,  bg0, ):
#    bg_term = bg(x, bg0, bg1)
    gauss_term = gaussian(x, x0, sigma, a, htail=0)

    return gauss_term + bg0

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


