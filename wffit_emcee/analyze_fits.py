#!/usr/local/bin/python
from ROOT import *

import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
from matplotlib import gridspec

import scipy.optimize as op
import numpy as np

from scipy import signal

import helpers
from detector_model import *
from probability_model import *

def main(argv):
  
  
  fitSamples = 200
  plt.ion()

  pzCorrTimeConstant = 72*1000.
  pz = MGWFPoleZeroCorrection()
  pz.SetDecayConstant(pzCorrTimeConstant)
  

  wfFileName = "P42574A_512waveforms_fit_smooth_de.npz"
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    results = data['result_arr']
    wfs = data['wfs']
  
  else:
    print "No saved waveforms available."
    exit(0)

  wfFileName = "P42574A_512waveforms_raw.npz"
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    wfs_raw = data['wfs']
    wfs_no_bl_sub = data['raw_wfs']
  else:
    print "No raw waveforms available."
    exit(0)

  print "There are %d waveforms fit" % wfs.size
  print "...And i pulled out %d raw ones" % wfs_raw.size


  det = prepdetector(fitSamples)
  simWfArr = np.empty((1,len(wfs), fitSamples))


  likes = np.empty(len(wfs))
  risetimes = np.empty(len(wfs))
  r_arr = np.empty(len(wfs))
  z_arr = np.empty(len(wfs))
  t0_arr = np.empty(len(wfs))

  time_since_last_arr = np.empty(len(wfs))
  last_energy_arr = np.empty(len(wfs))

  f1 = plt.figure(1, figsize=(15,8))

  good_risetimes = []

  likeCutoff = 20

  simWfArr = np.empty((1,len(wfs), fitSamples))
  
  nll = lambda *args: -lnlike_diffusion(*args)


  for (idx, wf) in enumerate(wfs):
    r, phi, z, scale, t0, smooth = results[idx]['x']
    
    simWfArr[0,idx,:] = det.GetSimWaveform(r, phi, z, scale*100, t0,  200, smoothing=smooth)
    
    r_arr[idx], z_arr[idx], t0_arr[idx] = r,z,t0
    
#    simWfArr[0,idx,:] = det.GetSimWaveform(r, phi, z, scale*100., t0,  fitSamples, smoothing=smooth, electron_smoothing=esmooth)
    likes[idx] = results[idx]['fun'] / wf.wfLength

    risetimes[idx] = findTimePointBeforeMax(wf.windowedWf, 0.99) - wf.t0Guess
    
    time_since_last_arr[idx] = wfs_raw[idx].timeSinceLast
    last_energy_arr[idx] = wfs_raw[idx].energyLast

#    #cheap PZ correction
#    mgwf = MGWaveform()
#    mgwf.SetSamplingPeriod(10*1)
#    mgwf.SetLength(len(wf.waveformData))
#    for i, wfVal in enumerate(wf.waveformData):
#      mgwf.SetValue(i, wfVal)
#    mgwfpz = MGWaveform()
#    pz.TransformOutOfPlace(mgwf, mgwfpz)
#    waveform = mgwfpz.GetVectorData()
#    waveform = np.multiply(waveform, 1)
#    wfToPlot = waveform
#    alignPoint = findTimePointBeforeMax(wfToPlot, 0.99)

    wfToPlot = wf.windowedWf

    if likes[idx] > 60:
      continue
    
      plt.plot(wfToPlot, color="orange")
      continue

    if likes[idx] > likeCutoff:
      continue
      
      plt.plot(wfToPlot, color="r")
      print "run number is wf %d" % wf.runNumber
      print ">>fit t0 is %f" % t0
      simWf = det.GetSimWaveform(r, phi, z, scale*100, t0, fitSamples, smoothing=smooth)
      
      plt.plot(simWf, color="b")
#      
#      new_result = op.minimize(nll, [r, phi, z, scale, t0, smooth], args=(det, wf.windowedWf,   wf.baselineRMS),  method="Powell")
#      r, phi, z, scale, t0, smooth = new_result["x"]
#      new_simWf = det.GetSimWaveform(r, phi, z, scale*100, t0, fitSamples, smoothing=smooth)
#      print ">>new like is %f" % (new_result['fun'] / wf.windowedWf.size)
#
#      plt.plot(new_simWf, color="g")

    else:
      if idx < 10: continue
      
      
#      print ">>old like is %f" % (results[idx]['fun'] / wf.windowedWf.size)
#    
#      new_result = op.minimize(nll, [r, phi, z, scale, t0, 2.], args=(det, wf.windowedWf,   wf.baselineRMS),  method="Powell")
#      r, phi, z, scale, t0, charge_cloud = new_result["x"]
#      new_simWf = det.GetSimWaveform(r, phi, z, scale*100, t0, fitSamples, energy=1592., charge_cloud_size=charge_cloud)
#
#      
#      print ">>new like is %f" % (new_result['fun'] / wf.windowedWf.size)
#
#
#      gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
#      ax0 = plt.subplot(gs[0])
#      ax1 = plt.subplot(gs[1], sharex=ax0)
#      ax1.set_xlabel("Digitizer Time [ns]")
#      ax0.set_ylabel("Voltage [Arb.]")
#      ax1.set_ylabel("Residual")
#      
#      ax0.plot(simWfArr[0,idx,:] ,color="blue", label="sim" )
#      ax0.plot( wf.windowedWf ,color="red", label="data" )
#      ax1.plot(wf.windowedWf-simWfArr[0,idx,:wf.windowedWf.size], color="b")
#      ax0.legend(loc=4)
#    
#    
#      break

      
      
#      plt.plot(wfToPlot , "g", alpha=0.1)
#
#    if likes[idx] > likeCutoff:
#      plt.plot(np.arange(wf.windowedWf.size)*10, wf.windowedWf, color="r", )
#    else:
#      plt.plot(np.arange(wf.windowedWf.size)*10, wf.windowedWf, color="g", alpha=0.05)
  
  
  

  plt.xlabel("time [ns]")
  plt.ylabel("energy [arb]")
  plt.plot(np.nan, color="r", label="bad fit :(")
  plt.plot(np.nan, color="g", label="good fit :)")
  plt.legend(loc=4)
  plt.savefig("512_all_waveforms.png")

  f2 = plt.figure(2, figsize=(15,8))
  helpers.plotManyResidual(simWfArr[:15], wfs[:15], f2, residAlpha=1)
  plt.savefig("512_residuals.png")
  

  value = raw_input('  --> Press q to quit, any other key to continue\n')
  if value == 'q': exit(0)
  
  '''

  plt.close(f1)
  plt.close(f2)

  good_idxs = np.where(likes <= likeCutoff)
  bad_idxs = np.where(likes > likeCutoff)
  
  f3 = plt.figure(3, figsize=(15,8))
  helpers.plotManyResidual(simWfArr[0,bad_idxs], wfs[bad_idxs], f3, residAlpha=0.5)
  plt.savefig("512_badwfs.png")


  fig_timesincelast = plt.figure()
  plt.scatter(time_since_last_arr[good_idxs], last_energy_arr[good_idxs],  color="g")
  plt.scatter(time_since_last_arr[bad_idxs], last_energy_arr[bad_idxs] , color="r")
  plt.xlabel("Time since last event")
  plt.ylabel("Energy [keV]")

#  fig_like = plt.figure()
#  plt.scatter(risetimes*10, likes)
#  plt.xlabel("Risetime [ns]")
#  plt.ylabel("NLL [normalized by wf length]")
#  plt.savefig("512_liklihoods.png")

  fig_pos = plt.figure()
  plt.scatter(r_arr[good_idxs], z_arr[good_idxs], color="g")
  plt.scatter(r_arr[bad_idxs], z_arr[bad_idxs], color="r")
  plt.xlim(0, det.detector_radius)
  plt.ylim(0, det.detector_length)
  plt.xlabel("Radial posion [mm from point contact]")
  plt.ylabel("Axial posion [mm above point contact]")
  plt.savefig("512_positions.png")

  fig = plt.figure()
  plt.hist(likes, bins="auto")
  plt.xlabel("NLL [normalized by wf length]")
  plt.savefig("512_likes.png")
  plt.xlim(0,50)
  plt.savefig("512_likes_zoom.png")

  figt0 = plt.figure()
  plt.hist(t0_arr*10, bins=20)
  plt.xlabel("Start Time [ns]")
  plt.savefig("512_times.png")
  
  value = raw_input('  --> Press q to quit, any other key to continue\n')
  if value == 'q': exit(0)
  
  '''
  
  
#

  numWfsToSave = 30

  print "minimum risetime: %f" % np.min(risetimes)
  print "maximum risetime: %f" % np.max(risetimes)

  hist, bin_edges = np.histogram(risetimes, range=(30,100), bins=numWfsToSave)

  print bin_edges

  #bin the rise-times in 8 bins, from 30-100.  Then, take enough from each to hit our waveform number goal. Try to maximize the number of low risetime events
  # So, say we want 32... equal would be 4 from each bin.  Let's try that.

#  print np.where(risetimes < bin_edges[1])

  idxlist_raw = []
  
  for bin_idx in np.arange(len(bin_edges)-1):
    indices =  np.nonzero( np.logical_and(  np.greater(risetimes, bin_edges[bin_idx]), np.less_equal(risetimes, bin_edges[bin_idx+1])   )  )[0]

    for idx in indices:
      if likes[idx] < 5:
        idxlist_raw.append(idx)
        break
    else:
      print "Couldn't find any wfs with like <5 in bin %d" % bin_idx
      for idx in indices:
        if likes[idx] < 10:
          idxlist_raw.append(idx)
          break
      else:
        print "...Couldn't find any wfs with like <10 in bin %d either" % bin_idx
        

  idxlist = idxlist_raw

  plt.figure(figsize=(20,10))

  wfsto_save = np.empty(len(bin_edges)-1, dtype=np.object)

  for idx in idxlist:
    plt.plot(wfs[idx].windowedWf, color="r")

  wfFileName = "P42574A_512waveforms_%drisetimeculled.npz" % numWfsToSave

  np.savez(wfFileName, wfs = wfs[idxlist], results=results[idxlist]  )

  value = raw_input('  --> Press q to quit, any other key to continue\n')
  if value == 'q': exit(0)

def prepdetector(fitSamples):
  #Prepare detector
  #Prepare detectoren = [1, 40503831.367655091, 507743886451386.06, 7.0164915381862738e+18]
  num = [3478247477.3386812, 1.9351287018998266e+17, 6066014750118182.0]
  den = [1, 40525756.700980239, 508584796045376.5, 7.0511687821902561e+18]
  system = signal.lti(num, den)
  fitSamples=200
  
  tempGuess = 78
  gradGuess = 0.0483
  pcRadGuess = 2.591182
  pcLenGuess = 1.613357

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system, useDiffusion=0)
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)

  return det


def findTimePointBeforeMax(data, percent):

  #don't screw up the data, bro
  int_data = np.copy(data)
  max_idx = np.argmax(int_data)
  int_data /= int_data[max_idx]
  
  int_data = int_data[0:max_idx]

  return np.where(np.less(int_data, percent))[0][-1]



if __name__=="__main__":
    main(sys.argv[1:])
