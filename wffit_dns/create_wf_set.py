#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import numpy as np

from scipy import signal, ndimage, interpolate, stats

import helpers
import rc_decay_fitter

import matplotlib.pyplot as plt

def main(argv):
  '''Tries to pull out a nice training set from a bunch of 2614 keV events.
  The input set should already have an A/E cut on it

  uses a bunch of cuts based on:
  -baseline value
  -waveform max (it works OK b/c of high S/N @ 2614)
  -variety of risetime parameters to cut point contact events and stuff that sneaks thru A/E cut

  For the waveforms that pass, it bins them by risetime, and pulls a number
  from each bin.  The idea is to get a training set with a wide variety of risetimes,
  so you have waveforms from all over in the detector.
  '''

  numBins = 2
  numWfsToSavePerBin = 4

  bl_variance = 3 #deviation (in adc) allowed from mode baseline value
  wfmax_variance = 15 #deviation (in adc) allowed from mode wf_max value

  # channel = 690
  # wfFileName = "fep_event_set_runs11510-11539_channel690"
  # save_file_name = "P42662A_%d_spread.npz" % (numBins*numWfsToSavePerBin)

  # channel = 626
  # wfFileName = "fep_event_set_runs11510-11630_channel%d.npz" % channel
  # save_file_name = "P42574A_%d_spread.npz" % (numBins*numWfsToSavePerBin)
  # risetime_limits = (40, 110) #this will be detector specific
  # collecttime_limits = (0, 16.0) #this will be detector specific
  # ae_return_cut = 150
  # ignore_wfs = []

#   channel = 672
#   wfFileName = "fep_event_set_runs11510-11610.npz"
#   # wfFileName = "cs_event_set_runs11510-11630_channel%d.npz" % channel
#   save_file_name = "P42661A_%d_slow_secondset.npz" % (numBins*numWfsToSavePerBin)
#   risetime_limits = (50, 100) #this will be detector specific
#   collecttime_limits = (0, 17.1) #this will be detector specific
#   ae_return_cut = 151
#   ignore_wfs = []

#  channel = 598
#  wfFileName = "dat/fep_event_set_runs11510-11630_channel598.npz"
#  save_file_name = "dat/P42574B_%d_slow.npz" % (numBins*numWfsToSavePerBin)

  channel = 600
  wfFileName = "dat/fep_event_set_runs11510-11630_channel%d.npz" % (channel)
  save_file_name = "dat/B8482_%d_slow.npz" % (numBins*numWfsToSavePerBin)

  risetime_limits = (50, 100) #this will be detector specific
  collecttime_limits = (0, 18.2) #this will be detector specific
  ae_return_cut = 151
  ignore_wfs = []

#   ignore_wfs = [(11515,128965),
# (11511,96633),
# (11510,317207),
# (11511,48362),
# (11510,56040),
# (11511,202862),
# (11515,62152),
# (11512,71698),
# (11511,95195),
# (11514,208461),
# (11511,13079),
# (11516,92744),
# (11515,225892),
# (11513,1225),
# (11513,6869),
# (11542,8401)]

  # channel = 578
  # wfFileName = "fep_event_set_runs11510-11630_channel%d.npz" % channel
  # # wfFileName = "cs_event_set_runs11510-11630_channel%d.npz" % channel
  # save_file_name = "P42538A_%d_slow.npz" % (numBins*numWfsToSavePerBin)
  # risetime_limits = (45, 110) #this will be detector specific
  # collecttime_limits = (0, 20.0) #this will be detector specific
  # ae_return_cut = 161


  doPlots = True

  if os.path.isfile(wfFileName ):
    data = np.load(wfFileName)
    wfs = data['wfs']
    numWaveforms = wfs.size
  else:
    print( "No saved waveforms available... exiting.")
    exit(0)

  rt_995 = np.ones(numWaveforms)
  rt_50 = np.empty(numWaveforms)
  rt_95 = np.empty(numWaveforms)
  baseline = np.empty(numWaveforms)
  length = np.empty(numWaveforms)
  max_current = np.empty(numWaveforms)
  num_ae_peaks = np.empty(numWaveforms)
  energy = np.empty(numWaveforms)

  current_rejoin_idxs = np.empty(numWaveforms)

  if doPlots: preCutfig = plt.figure(1)

  preCutfigCurrent = plt.figure(0)

  #calculate some params.
  for (idx,wf) in enumerate(wfs):
    wf.WindowWaveformAroundTimepoint(earlySamples=125,  timePoint=0.5, fallPercentage=.99, rmsMult=2)
    energy[idx] = wf.energy
    baseline[idx] = wf.baselineMean
    length[idx] = wf.wfLength

    rt_995[idx] = findTimePointBeforeMax(wf.windowedWf, 0.995) - wf.t0Guess
    rt_95[idx] = findTimePointBeforeMax(wf.windowedWf, 0.95) - wf.t0Guess
    rt_50[idx] = findTimePointBeforeMax(wf.windowedWf, 0.50) - wf.t0Guess

    #Find the last point where the current is above a certain value
    #Gets rid of some multisite stragglers
    diff_threshold = 10
    current = ndimage.gaussian_filter1d(wf.windowedWf, sigma=5, order=1)
    current_rejoin_idx = np.argwhere(current >diff_threshold)[-1]
    current_rejoin_idxs[idx] = current_rejoin_idx

    if doPlots:
        plt.figure(0)
        plt.plot(current)

        plt.figure(1)
        plt.plot(wf.windowedWf + wf.baselineMean)
        plt.xlabel("Digitizer sampler [10s of ns]")
        plt.ylabel("ADC Value [unchanged from digitzer]")
        plt.title("Pre-cut waveforms")

    #find the current max
    current = ndimage.gaussian_filter1d(wf.windowedWf, sigma=1, order=1)
    current_max_idx = np.argmax(current)
    interp = interpolate.interp1d(range(current_max_idx-5,current_max_idx+5),
            current[current_max_idx-5:current_max_idx+5],
            kind="cubic", assume_sorted=True)
    fine_current = interp(  np.linspace(current_max_idx-5,current_max_idx+4, 1000))
    max_current[idx] = np.amax(fine_current)

    ae = ndimage.gaussian_filter1d(wf.windowedWf, sigma=3, order=1) / wf.energy
    maxes = signal.argrelmax(ae, order = 5)
    ae_peaks = np.where( ae[maxes[0]] > 0.01)[0]
    num_ae_peaks[idx] = len(ae_peaks)

    # num_maxes[idx] = len(maxes)


  #find the normal baseline value
  bl_mode = findMode(baseline, doPlots, (bl_variance,), xlabel="mean baseline [adc]")
  risetime_mode = findMode(rt_995, doPlots, risetime_limits, xlabel="99.5% risetime [10s of ns]")
  currentmax_mode = findMode(max_current, doPlots, (10,), xlabel="max current")
  length_mode = findMode(length, doPlots, (10,), xlabel="length (time stamps)")
  energy_mode = findMode(energy, doPlots, (4,), xlabel="energy (keV)")

  #cut based on wf max and baseline
  bl_cut = np.logical_and(baseline > (bl_mode - bl_variance), baseline < (bl_mode + bl_variance))
  print ("baseline cut: %d pass (of %d)" % (np.count_nonzero(bl_cut),   len(wfs)))

  #cuts REAL long waveforms (can also be used to cut real fast ones)
  rt_cut = np.logical_and(rt_995 > risetime_limits[0], rt_995 < risetime_limits[1])
  print ("rt cut: %d pass" % np.count_nonzero(rt_cut)  )

  #cuts those weird slow-on-top events that survive A/E
  current_rejoin_cut = (current_rejoin_idxs) < ae_return_cut
  print ("current_rejoin_cut cut: %d pass" % np.count_nonzero(current_rejoin_cut)  )

  length_cut = np.logical_and(length > (length_mode - 10), length < (length_mode + 10))
  energy_cut = np.logical_and(energy > (energy_mode - 1), energy < (energy_mode + 1))

  print ("length_cut cut: %d pass" % np.count_nonzero(length_cut)  )
  print ("energy_cut cut: %d pass" % np.count_nonzero(energy_cut)  )

  #cut based on t50-t995: cuts events near the PC
  t_collect = rt_995 - rt_50
  t_collect_mode = findMode(t_collect, doPlots, collecttime_limits, xlabel="t50 - t99.5% risetime [10s of ns]", binMult=5)
  pc_cut = np.logical_and(t_collect > collecttime_limits[0], t_collect < collecttime_limits[1])
  print ("pc_cut cut: %d pass" % np.count_nonzero(pc_cut)  )


  num_ae_peaks_cut = num_ae_peaks == 1
  print ("num_ae_peaks_cut cut: %d pass" % np.count_nonzero(num_ae_peaks_cut)  )

  #stack the cuts
  full_cut = np.stack(( bl_cut, rt_cut,current_rejoin_cut, length_cut,  pc_cut, energy_cut, num_ae_peaks_cut) )
  full_cut = np.all(full_cut, axis=0)

# #2d hist of max current vs rt
#   plt.figure()
#   plt.hist2d( rt_50[full_cut], max_current[full_cut], bins=50)
#   plt.show()
#   exit()

  for (idx,wf) in enumerate(wfs):
      if (wf.runNumber, wf.entry_number) in ignore_wfs:
          full_cut[idx] = 0
          print ("skipping wf %d" % idx)
  cut_wfs = wfs[full_cut]

  print("wfs surving all cuts %d of %d" % (np.count_nonzero(full_cut), numWaveforms))
  if(np.count_nonzero(full_cut) == 0):
      print("There were no waveforms left. Check your cuts and try again...\n")
      return

  # plt.ion()
  if doPlots:
      plt.figure()
      for wf in cut_wfs:
        #   plt.clf()
        #   smoothwf = ndimage.gaussian_filter1d(wf.windowedWf, sigma=5, order=1)
        #   plt.plot(smoothwf)
        #   val = input("Any key for next wf")
        #   current[current<diff_threshold] = 10
          plt.plot(np.arange(len(wf.windowedWf))*10, wf.windowedWf )
      plt.xlabel("Time (ns)")
      plt.ylabel("Voltage [adc]")

  # plt.show()
  # exit()

  #check out charge trapping properties
  plt.figure()
  cut_rt = rt_995[full_cut] * 10 #to move to ns
  wf_maxes = np.empty(len(cut_wfs))
  for (idx,wf) in enumerate(cut_wfs):
      wf_maxes[idx] = np.amax(wf.windowedWf)
      plt.scatter(cut_rt[idx], np.amax(wf.windowedWf))

  slope, intercept, r_value, p_value, std_err = stats.linregress(cut_rt,wf_maxes)

  plt.plot( np.linspace(0, np.amax(cut_rt),100), slope*np.linspace(0, np.amax(cut_rt),100) + intercept )
  plt.xlim(np.amin(cut_rt), np.amax(cut_rt))
  print (slope)
  print (np.log(-slope))

  # plt.show()
  # exit()
  #
  # # snippet of code thats nice for evaluating new cuts
  # alt_cut = np.stack((bl_cut, rt_cut, current_rejoin_cut, pc_cut, np.logical_not(length_cut)) )
  # alt_cut = np.all(alt_cut, axis=0)
  # alt_wfs = wfs[alt_cut]
  #
  # plt.figure()
  # print ("alt wfs: %d" % len(alt_wfs))
  # for wf in alt_wfs:
  #     plt.plot(wf.windowedWf )
  #
  # plt.show()
  # exit()

  #Fit some tails to get an idea of the RC decay params for priors
  numWaveforms = len(cut_wfs)
  rc_arr = np.empty((3, numWaveforms))

  print ("Fitting RC tails -- might take a few seconds")
  for (idx,wf) in enumerate(cut_wfs):
      if idx % 100 == 0: print ("on idx %d of %d" % (idx, numWaveforms))
      result = rc_decay_fitter.fit_decay(wf)
      rc1, rc2, rcfrac = result['x']
      rc_arr[:,idx] = rc1, rc2, rcfrac

  np.save("rc_fit.npy", rc_arr)

  [rc1, rc1_bins] = np.histogram(rc_arr[0,:], bins=20)
  [rc2, rc2_bins] = np.histogram(rc_arr[1,:], bins=20)
  [rcfrac, rcfrac_bins] = np.histogram(rc_arr[2,:], bins=20)

  if doPlots:
      f, axarr = plt.subplots(3, )
      axarr[0].plot(rc1_bins[:-1],rc1)
      axarr[1].plot(rc2_bins[:-1],rc2)
      axarr[2].plot(rcfrac_bins[:-1],rcfrac)

  rc1_mode = rc1_bins[np.argmax(rc1)]
  rc2_mode = rc2_bins[np.argmax(rc2)]
  rcfrac_mode = rcfrac_bins[np.argmax(rcfrac)]

  print ("RC Param Modes:\n --> RC1: {0}, RC2: {1}, RC Frac: {2}".format(rc1_mode, rc2_mode, rcfrac_mode))


  print( "generating a set of %d waveforms" % (numWfsToSavePerBin*numBins) )
  #regen risetime array w/ cut
  risetimes = rt_95[full_cut]

  #snag a random subset from the bins
  (n,b) = np.histogram(risetimes, bins=numBins)
  wfs_to_save = np.empty(numWfsToSavePerBin*numBins, dtype=np.object)
  for i in range(len(n)):
      if n[i] < numWfsToSavePerBin:
          print( "not enough waveforms per bin to save! index %d has %d wfs between: " %(i, n[i]), b[i], b[i+1] )
          exit(0)
      else:
          idxs = np.logical_and(np.less(risetimes, b[i+1]), np.greater(risetimes, b[i]) )
        #   print wfs_saved[ idxs ][:numWfsToSavePerBin]
          wfs_to_save[i*numWfsToSavePerBin:i*numWfsToSavePerBin+numWfsToSavePerBin] = cut_wfs[ idxs ][:numWfsToSavePerBin]

  plt.figure()
  for wf in wfs_to_save:
      plt.plot(wf.windowedWf)
  plt.title("Waveforms in the saved set {0}".format(save_file_name))

  # plt.show()
  np.savez(save_file_name, wfs = wfs_to_save, rc1 = rc1_mode, rc2 = rc2_mode, rcfrac=rcfrac_mode)
  print("saved file {0}".format(save_file_name))

  plt.show()
  # exit()

def findMode(array, doPlots = False, limits=None, xlabel="", binMult=1):
  bin_offet = 0.5 #half the bin width
  bl_min = np.floor(np.amin(array))
  bl_max = np.ceil(np.amax(array))
  (n,b) = np.histogram(array, bins=np.linspace( bl_min, bl_max, (bl_max-bl_min)*binMult + 1))
  bl_mode = b[np.argmax(n)] + bin_offet

  if doPlots:
      plt.figure()
      plt.step(b[:-1]+bin_offet,n)
      if not (limits is None):
          if len(limits) == 1:
              low = bl_mode - limits
              hi = bl_mode + limits
          else:
              low = limits[0]
              hi = limits[1]

          plt.axvline(x=low, color="r")
          plt.axvline(x=hi, color="r")
      plt.xlabel(xlabel)
  return bl_mode


def findTimePointBeforeMax(data, percent):

  #don't screw up the data, bro
  int_data = np.copy(data)
  max_idx = np.argmax(int_data)
  int_data /= int_data[max_idx]

  alignarr = int_data

  first_idx = np.searchsorted(int_data[0:max_idx], percent, side='left') - 1

  if first_idx == len(data)-1:
      return first_idx

  return (percent - alignarr[first_idx]) * (1) / (alignarr[first_idx+1] - alignarr[first_idx]) + first_idx


if __name__=="__main__":
    main(sys.argv[1:])
